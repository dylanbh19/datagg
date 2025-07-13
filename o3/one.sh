#!/usr/bin/env bash
# ===========================================================================
# 01_bootstrap_data.sh            – Customer-Comms  (Epic-1, Stories 1.1-1.4)
# ===========================================================================
#  • Installs core libs  (only if missing)
#  • Creates   ./customer_comms   package
#  • Aggregates raw “1-row-per-call” files  ➜  00_daily_call_volume.csv
#  • Generates data-quality report           01_data_quality_report.json
#  • Builds MA-aligned augmented calls       02_augmented_calls.csv
#  • Downloads econ indicators               03_econ_features.csv
#  • Re-run safe: skips steps whose output is newer than their inputs
# ===========================================================================

set -euo pipefail
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

PKG="customer_comms"
OUT="output"
LOG="logs"
DATA="data"

# ----------------------------------------------------------------------------
# 0. Install core Python deps (silent if already present)
# ----------------------------------------------------------------------------
python - <<'PY'
import importlib, subprocess, sys, contextlib
deps = ('pandas','numpy','matplotlib','seaborn','scipy',
        'scikit-learn','statsmodels','holidays',
        'pydantic','pydantic-settings','yfinance')
for p in deps:
    with contextlib.suppress(ModuleNotFoundError):
        importlib.import_module(p.replace('-','_')); continue
    subprocess.check_call([sys.executable,'-m','pip','install','-q',p])
PY

# ----------------------------------------------------------------------------
# 1. Scaffold package / dirs   (idempotent)
# ----------------------------------------------------------------------------
mkdir -p "$PKG"/{data,processing,analytics,features} "$LOG" "$OUT"
for d in "" data processing analytics features; do
  touch "$PKG/${d:+$d/}__init__.py"
done

# ----------------------------------------------------------------------------
# 2.  customer_comms/config.py
# ----------------------------------------------------------------------------
cat > "$PKG/config.py" << 'PY'
from pathlib import Path
try:    from pydantic_settings import BaseSettings     # Pydantic v2
except ModuleNotFoundError: from pydantic import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # -------- raw CSV names (place in ./data/) ----------------------------
    call_files: list[str] = ["GenesysExtract_20250609.csv",
                             "GenesysExtract_20250703.csv"]
    mail_file: str        = "merged_call_data.csv"

    # -------- column names ------------------------------------------------
    call_date_cols:   list[str] = ["ConversationStart","Date"]
    call_intent_cols: list[str] = ["uui_Intent","intent","Intent"]
    mail_date_col:    str = "mail_date"
    mail_type_col:    str = "mail_type"
    mail_volume_col:  str = "mail_volume"

    # -------- processing params ------------------------------------------
    ma_window:    int = 7      # moving-avg window for augmentation
    max_gap:      int = 3      # max business-day gap that will be filled
    min_rows:     int = 20

    # -------- econ tickers (yfinance) ------------------------------------
    econ_tickers: dict[str,str] = {
        "sp500"  : "^GSPC",
        "vix"    : "^VIX",
        "fedfund": "FEDFUNDS",
        "unemp"  : "UNRATE",
    }

    # -------- paths -------------------------------------------------------
    data_dir: Path = Field(default=Path("data"))
    out_dir:  Path = Field(default=Path("output"))
    log_dir:  Path = Field(default=Path("logs"))
settings = Settings()
PY

# ----------------------------------------------------------------------------
# 3.  customer_comms/logging_utils.py
# ----------------------------------------------------------------------------
cat > "$PKG/logging_utils.py" << 'PY'
import logging, sys, traceback
from datetime import datetime
from .config import settings

def get_logger(name="customer_comms"):
    log = logging.getLogger(name)
    if log.handlers:
        return log
    fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"; dh="%Y-%m-%d %H:%M:%S"
    log.setLevel(logging.INFO)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(fmt, dh))
    try: sh.stream.reconfigure(encoding="utf-8")
    except AttributeError: pass
    log.addHandler(sh)

    settings.log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(settings.log_dir/
                             f"{name}_{datetime.now():%Y%m%d}.log",
                             encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt, dh))
    log.addHandler(fh)
    return log
PY

# ----------------------------------------------------------------------------
# 4.  customer_comms/data/loader.py  – multi-codec CSV reader
# ----------------------------------------------------------------------------
cat > "$PKG/data/loader.py" << 'PY'
from __future__ import annotations
import pandas as pd, io, hashlib, os
from pathlib import Path
from ..config import settings
from ..logging_utils import get_logger
log = get_logger(__name__)
_CODECS = ("utf-8","utf-8-sig","latin-1","cp1252","utf-16","iso-8859-1")

def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        log.error(f"Missing {path}"); return pd.DataFrame()
    for enc in _CODECS:
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="skip", low_memory=False)
        except UnicodeDecodeError:
            continue
    # last resort – replace undecodable bytes and parse
    text = path.read_bytes().decode("utf-8", "replace")
    return pd.read_csv(io.StringIO(text), on_bad_lines="skip", low_memory=False)

def _dt(s: pd.Series) -> pd.Series:
    return (pd.to_datetime(s, errors="coerce")
              .dt.tz_localize(None)
              .dt.normalize())

def _needs_refresh(output: Path, inputs: list[Path]) -> bool:
    """Return True if output is absent or older than any input."""
    if not output.exists():
        return True
    out_mtime = output.stat().st_mtime
    return any(p.exists() and p.stat().st_mtime > out_mtime for p in inputs)
PY

# ----------------------------------------------------------------------------
# 5.  processing/aggregate_calls.py  – Story 1.4
# ----------------------------------------------------------------------------
cat > "$PKG/processing/aggregate_calls.py" << 'PY'
"""Aggregate raw call rows ➜ 1 row per business-day."""
import pandas as pd
from pathlib import Path
from ..config import settings
from ..data.loader import _read, _dt, _needs_refresh
from ..logging_utils import get_logger
log = get_logger(__name__)

def run() -> pd.DataFrame:
    out_csv = settings.out_dir/"00_daily_call_volume.csv"
    if not _needs_refresh(out_csv,
                          [settings.data_dir/f for f in settings.call_files]):
        log.info("Daily call-volume already up-to-date"); 
        return pd.read_csv(out_csv, parse_dates=["date"])

    frames = []
    tot_rows = 0
    for fn in settings.call_files:
        path = settings.data_dir/fn
        df = _read(path)
        if df.empty:
            continue
        tot_rows += len(df)
        dcol = next((c for c in settings.call_date_cols if c in df.columns), None)
        df = df.rename(columns={dcol: "date"}) if dcol else df.assign(date=pd.NaT)
        df["date"] = _dt(df["date"])
        frames.append(df[["date"]])

    if not frames:
        log.error("No call rows loaded"); return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True).dropna(subset=["date"])
    agg = (raw.groupby("date").size()
                .reset_index(name="call_volume_raw")
                .sort_values("date"))
    agg.to_csv(out_csv, index=False)
    # validation
    lost = tot_rows - agg["call_volume_raw"].sum()
    pct  = lost * 100 / max(tot_rows,1)
    if pct > 0.1:
        log.warning(f"Lost {lost} rows ({pct:.2f} %) during aggregation (bad dates?)")
    else:
        log.info(f"Aggregated {tot_rows:,} rows ➜ {len(agg):,} business-days")
    return agg
PY

# ----------------------------------------------------------------------------
# 6.  processing/ingest.py  – Story 1.1  (now calls aggregator)
# ----------------------------------------------------------------------------
cat > "$PKG/processing/ingest.py" << 'PY'
"""Ingest CSVs, run QC report, and return raw-frames."""
import json, pandas as pd
from ..config import settings
from ..logging_utils import get_logger
from ..processing.aggregate_calls import run as aggregate_calls
from ..data.loader import _read, _dt
log=get_logger(__name__)

def run():
    # ----------- call aggregation already done -------------------------
    call_daily = aggregate_calls()

    # ----------- load intents (if any) ---------------------------------
    intent_frames = []
    for fn in settings.call_files:
        df = _read(settings.data_dir/fn)
        icol = next((c for c in settings.call_intent_cols if c in df.columns), None)
        dcol = next((c for c in settings.call_date_cols  if c in df.columns), None)
        if not (icol and dcol):
            continue
        df = df.rename(columns={dcol:"date",icol:"intent"})
        df["date"]=_dt(df["date"]); intent_frames.append(df[["date","intent"]])
    intents = pd.concat(intent_frames, ignore_index=True) if intent_frames else pd.DataFrame()

    # ----------- load mail --------------------------------------------
    mail = _read(settings.data_dir/settings.mail_file)
    if not mail.empty:
        mail = mail.rename(columns={settings.mail_date_col:"date",
                                    settings.mail_type_col:"mail_type",
                                    settings.mail_volume_col:"mail_volume"})
        mail["date"]=_dt(mail["date"])
        mail["mail_volume"]=pd.to_numeric(mail["mail_volume"],errors="coerce")
        mail.dropna(subset=["date","mail_volume"], inplace=True)

    # ----------- profile ----------------------------------------------
    def prof(df,name): return {"name":name,"rows":len(df),
        "na":df.isna().sum().to_dict() if not df.empty else {}}
    rep=[prof(call_daily,"call_daily"),prof(intents,"intents"),prof(mail,"mail")]
    settings.out_dir.mkdir(exist_ok=True)
    json.dump(rep, open(settings.out_dir/"01_data_quality_report.json","w",encoding="utf-8"), indent=2)
    log.info("Data-quality report written")
    return call_daily, intents, mail
PY

# ----------------------------------------------------------------------------
# 7.  analytics/augment.py  – Story 1.2  (unchanged but reads call_daily)
# ----------------------------------------------------------------------------
cat > "$PKG/analytics/augment.py" << 'PY'
import pandas as pd, numpy as np, numpy
from ..config import settings
from ..logging_utils import get_logger
log=get_logger(__name__)

def augment(call_df: pd.DataFrame, intents: pd.DataFrame) -> pd.DataFrame:
    """Align call volume to intent M-A and output augmented series."""
    out_csv = settings.out_dir/"02_augmented_calls.csv"
    if not call_df.empty and not intents.empty and out_csv.exists():
        # if newer than inputs skip
        if out_csv.stat().st_mtime > max(p.stat().st_mtime
                                         for p in settings.data_dir.glob("*")):
            return pd.read_csv(out_csv,parse_dates=["date"])

    if call_df.empty or intents.empty:
        log.warning("Missing calls or intents – augmentation skipped"); 
        return pd.DataFrame()

    intent_daily=(intents.groupby(["date","intent"]).size()
                          .unstack(fill_value=0)
                          .reset_index())
    total_int = intent_daily.set_index("date").sum(axis=1)
    df = call_df.set_index("date").join(total_int.rename("intent_total"), how="inner")
    w = settings.ma_window
    scale = (df["intent_total"].rolling(w,1).mean() /
             df["call_volume_raw"].rolling(w,1).mean().replace(0,np.nan)).clip(0.25,4).fillna(1)
    df["call_volume_aug"] = df["call_volume_raw"] * scale
    df = df.reset_index()
    df.to_csv(out_csv, index=False)
    log.info(f"Augmented calls saved ({len(df)} rows)")
    return df
PY

# ----------------------------------------------------------------------------
# 8.  features/econ.py  – Story 1.3
# ----------------------------------------------------------------------------
cat > "$PKG/features/econ.py" << 'PY'
import pandas as pd, yfinance as yf, datetime as dt
from ..config import settings
from ..data.loader import _needs_refresh
from ..logging_utils import get_logger
log=get_logger(__name__)

def fetch():
    out_csv = settings.out_dir/"03_econ_features.csv"
    if not _needs_refresh(out_csv, []):
        return pd.read_csv(out_csv,parse_dates=["date"])
    dfs=[]
    start="2023-01-01"
    for name,ticker in settings.econ_tickers.items():
        try:
            data=yf.download(ticker,start=start,progress=False,auto_adjust=True,threads=False)
            if data.empty: continue
            ser=(data["Close"] if "Close" in data.columns else data.squeeze())
            dfs.append(ser.rename(name))
            log.info(f"Fetched {name}")
        except Exception as e:
            log.warning(f"{ticker} failed: {e}")
    if not dfs:
        return pd.DataFrame()
    econ=pd.concat(dfs,axis=1).ffill().reset_index().rename(columns={"Date":"date"})
    econ["date"]=pd.to_datetime(econ["date"]).dt.normalize()
    econ.to_csv(out_csv,index=False)
    log.info("Econ file saved")
    return econ
PY

# ----------------------------------------------------------------------------
# 9.  Run Stage-1 pipeline (self-healing)
# ----------------------------------------------------------------------------
python - <<'PY'
from customer_comms.processing.ingest     import run as ingest
from customer_comms.analytics.augment     import augment
from customer_comms.features.econ         import fetch as econ_fetch
from customer_comms.logging_utils         as CU
log=CU.get_logger("bootstrap")

try:
    call_daily, intents, mail = ingest()
    augment(call_daily, intents)
    econ_fetch()
    log.info("🎉  Stage-1 pipeline complete")
except Exception as e:
    log.critical(f"Stage-1 failed: {e}", exc_info=True)
    raise
PY

echo "✅  Stage-1 artefacts in ./output/, logs in ./logs/"
echo "You can re-run this script anytime; it will only regenerate stale files."