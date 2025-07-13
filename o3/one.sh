#!/usr/bin/env bash
# ==============================================================================
# 01_bootstrap_data.sh      –  Customer-Comms  Stage-1  (Epics 1.1 – 1.4)
# ==============================================================================
#  - Works on Windows Git-Bash / WSL / MINGW, no venv, no make.
#  - Re-runnable & self-healing: only regenerates artefacts that are stale/missing.
#  - UTF-8 logging everywhere (avoids “charmap” errors on Windows).
# ==============================================================================

set -euo pipefail
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

# ------------------------------------------------------------------------------ 
# Folder constants
# ------------------------------------------------------------------------------
PKG="customer_comms"
DATA="data"
OUT="output"
LOG="logs"

mkdir -p "$PKG" "$DATA" "$OUT" "$LOG"

# ------------------------------------------------------------------------------
# 0 ── Ensure core Python deps (silent if present)
# ------------------------------------------------------------------------------
python - <<'PY'
import importlib, subprocess, sys, contextlib, json, pathlib
deps = ("pandas","numpy","matplotlib","seaborn","scipy",
        "scikit-learn","statsmodels","holidays",
        "pydantic","pydantic-settings","yfinance")
for d in deps:
    with contextlib.suppress(ModuleNotFoundError):
        importlib.import_module(d.replace("-","_")); continue
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", d])
PY

# ------------------------------------------------------------------------------
# 1 ── Build/refresh package skeleton   (idempotent)
# ------------------------------------------------------------------------------
for sub in "" data processing analytics features; do
  mkdir -p "$PKG/$sub"
  touch     "$PKG/${sub:+$sub/}__init__.py"
done

# ------------------------------------------------------------------------------
# 2 ── config.py  (central config object)
# ------------------------------------------------------------------------------
cat > "$PKG/config.py" << 'PY'
from pathlib import Path
try:    from pydantic_settings import BaseSettings
except ModuleNotFoundError:                                       # Pydantic 1.x
    from pydantic import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # ---------- CSV filenames (drop them into ./data/) -----------------------
    call_files: list[str] = ["GenesysExtract_20250609.csv",
                             "GenesysExtract_20250703.csv"]
    mail_file:  str        = "merged_call_data.csv"

    # ---------- column names -------------------------------------------------
    call_date_cols:   list[str] = ["ConversationStart", "Date"]
    call_intent_cols: list[str] = ["uui_Intent", "intent", "Intent"]
    mail_date_col:    str       = "mail_date"
    mail_type_col:    str       = "mail_type"
    mail_volume_col:  str       = "mail_volume"

    # ---------- processing params -------------------------------------------
    ma_window: int = 7             # moving-average window for augmentation
    max_gap:   int = 3             # max biz-day gap to fill when stitching
    min_rows:  int = 20

    # ---------- econ tickers -------------------------------------------------
    econ_tickers: dict[str, str] = {
        "sp500"  : "^GSPC",
        "vix"    : "^VIX",
        "fedfund": "FEDFUNDS",
        "unemp"  : "UNRATE",
    }

    # ---------- paths --------------------------------------------------------
    data_dir: Path = Field(default=Path("data"))
    out_dir:  Path = Field(default=Path("output"))
    log_dir:  Path = Field(default=Path("logs"))

settings = Settings()
PY

# ------------------------------------------------------------------------------
# 3 ── logging_utils.py  (UTF-8 console + file)
# ------------------------------------------------------------------------------
cat > "$PKG/logging_utils.py" << 'PY'
import logging, sys
from datetime import datetime
from .config import settings

def get_logger(name="customer_comms"):
    """Return a UTF-8 logger that logs to console + rotating daily file."""
    log = logging.getLogger(name)
    if log.handlers:
        return log

    fmt  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date = "%Y-%m-%d %H:%M:%S"
    log.setLevel(logging.INFO)

    # --- console ---
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(fmt, date))
    try: sh.stream.reconfigure(encoding="utf-8")
    except AttributeError: pass
    log.addHandler(sh)

    # --- file ---
    settings.log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(
        settings.log_dir / f"{name}_{datetime.now():%Y%m%d}.log",
        encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt, date))
    log.addHandler(fh)
    return log
PY

# ------------------------------------------------------------------------------
# 4 ── data/loader.py   (multi-codec CSV + helper utils)
# ------------------------------------------------------------------------------
cat > "$PKG/data/loader.py" << 'PY'
from __future__ import annotations
import pandas as pd, io
from pathlib import Path
from ..config import settings
from ..logging_utils import get_logger
log = get_logger(__name__)

# Multiple codecs so Windows CSVs don’t choke
_CODECS = ("utf-8","utf-8-sig","latin-1","cp1252","utf-16","iso-8859-1")

def read_csv_multi(path: Path) -> pd.DataFrame:
    if not path.exists():
        log.error(f"Missing {path}"); return pd.DataFrame()
    for enc in _CODECS:
        try: return pd.read_csv(path, encoding=enc, on_bad_lines="skip", low_memory=False)
        except UnicodeDecodeError: continue
    # Last-ditch: replace undecodable
    text = path.read_bytes().decode("utf-8","replace")
    return pd.read_csv(io.StringIO(text), on_bad_lines="skip", low_memory=False)

def to_date(series: pd.Series) -> pd.Series:
    return (pd.to_datetime(series, errors="coerce")
              .dt.tz_localize(None)
              .dt.normalize())

def stale(output: Path, inputs: list[Path]) -> bool:
    """True if output is absent or older than any input."""
    if not output.exists():
        return True
    out_m = output.stat().st_mtime
    return any(p.exists() and p.stat().st_mtime > out_m for p in inputs)
PY

# ------------------------------------------------------------------------------
# 5 ── processing/aggregate_calls.py  (row-level ➜ daily)
# ------------------------------------------------------------------------------
cat > "$PKG/processing/aggregate_calls.py" << 'PY'
import pandas as pd, numpy as np
from pathlib import Path
from ..config import settings
from ..data.loader import read_csv_multi, to_date, stale
from ..logging_utils import get_logger
log=get_logger(__name__)

OUT_FILE = settings.out_dir/"00_daily_call_volume.csv"

def run() -> pd.DataFrame:
    inputs = [settings.data_dir/f for f in settings.call_files]
    if not stale(OUT_FILE, inputs):
        return pd.read_csv(OUT_FILE, parse_dates=["date"])

    frames = []
    raw_rows = 0
    for fn in settings.call_files:
        path = settings.data_dir/fn
        df = read_csv_multi(path)
        if df.empty:
            continue
        raw_rows += len(df)
        dcol = next((c for c in settings.call_date_cols if c in df.columns), None)
        if not dcol:
            log.error(f"{fn}: no recognised date col"); continue
        df = df.rename(columns={dcol:"date"})
        df["date"] = to_date(df["date"])
        frames.append(df[["date"]])

    if not frames:
        log.error("No call data aggregated"); return pd.DataFrame()

    daily = (pd.concat(frames)
               .dropna(subset=["date"])
               .groupby("date").size()
               .reset_index(name="call_volume_raw")
               .sort_values("date"))
    settings.out_dir.mkdir(exist_ok=True)
    daily.to_csv(OUT_FILE, index=False)
    pct_loss = 100 - (daily["call_volume_raw"].sum() * 100 / max(raw_rows,1))
    log.info(f"Aggregated {raw_rows:,} rows to {len(daily):,} daily points "
             f"({pct_loss:.2f}% lost to bad dates)")
    return daily
PY

# ------------------------------------------------------------------------------
# 6 ── processing/ingest.py  (Profiles + QC report)
# ------------------------------------------------------------------------------
cat > "$PKG/processing/ingest.py" << 'PY'
import json, pandas as pd
from ..config import settings
from pathlib import Path
from ..logging_utils import get_logger
from ..processing.aggregate_calls import run as agg_calls
from ..data.loader import read_csv_multi, to_date
log=get_logger(__name__)

def run():
    call_daily = agg_calls()

    # ---------- intents ----------
    intents_frames=[]
    for fn in settings.call_files:
        df=read_csv_multi(settings.data_dir/fn)
        dcol=next((c for c in settings.call_date_cols  if c in df.columns),None)
        icol=next((c for c in settings.call_intent_cols if c in df.columns),None)
        if dcol and icol:
            df=df.rename(columns={dcol:"date",icol:"intent"})
            df["date"]=to_date(df["date"]); intents_frames.append(df[["date","intent"]])
    intents=pd.concat(intents_frames,ignore_index=True) if intents_frames else pd.DataFrame()

    # ---------- mail -------------
    mail=read_csv_multi(settings.data_dir/settings.mail_file)
    if not mail.empty:
        mail=mail.rename(columns={settings.mail_date_col:"date",
                                  settings.mail_type_col:"mail_type",
                                  settings.mail_volume_col:"mail_volume"})
        mail["date"]=to_date(mail["date"])
        mail["mail_volume"]=pd.to_numeric(mail["mail_volume"],errors="coerce")
        mail.dropna(subset=["date","mail_volume"],inplace=True)

    # ---------- QC report --------
    def profile(df,name):
        return {"name":name,"rows":len(df),
                "missing":int(df.isna().sum().sum()) if not df.empty else 0}
    rep=[profile(call_daily,"call_daily"),
         profile(intents,"intents"),
         profile(mail,"mail")]

    settings.out_dir.mkdir(exist_ok=True)
    json.dump(rep, open(settings.out_dir/"01_data_quality_report.json","w",encoding="utf-8"), indent=2)
    log.info("Data-quality report saved")
    return call_daily,intents,mail
PY

# ------------------------------------------------------------------------------
# 7 ── analytics/augment.py  (Intent → Call M-A scaling)
# ------------------------------------------------------------------------------
cat > "$PKG/analytics/augment.py" << 'PY'
import pandas as pd, numpy as np
from ..config import settings
from ..logging_utils import get_logger
log=get_logger(__name__)
OUT_FILE=settings.out_dir/"02_augmented_calls.csv"

def run(call_daily: pd.DataFrame, intents: pd.DataFrame) -> pd.DataFrame:
    if call_daily.empty or intents.empty:
        log.warning("Augmentation skipped (missing calls or intents)")
        return pd.DataFrame()
    if OUT_FILE.exists() and OUT_FILE.stat().st_mtime > max(
        call_daily["date"].max().timestamp(), intents["date"].max().timestamp()):
        return pd.read_csv(OUT_FILE,parse_dates=["date"])

    tot_int=(intents.groupby(["date","intent"])
                    .size().unstack(fill_value=0)
                    .sum(axis=1).rename("intent_total"))
    df=call_daily.set_index("date").join(tot_int,how="left").fillna(0)
    ma=df.rolling(settings.ma_window,min_periods=1).mean()
    scale=np.where(ma["call_volume_raw"]>0, 
                   ma["intent_total"]/ma["call_volume_raw"],1.0)
    scale=np.clip(scale,0.25,4)                       # tame extremes
    df["call_volume_aug"]=df["call_volume_raw"]*scale
    df.reset_index().to_csv(OUT_FILE,index=False)
    log.info("Augmented call series saved")
    return df.reset_index()
PY

# ------------------------------------------------------------------------------
# 8 ── features/econ.py  (economic indicators via yfinance)
# ------------------------------------------------------------------------------
cat > "$PKG/features/econ.py" << 'PY'
import pandas as pd, datetime as dt, contextlib, yfinance as yf
from ..config import settings
from ..data.loader import stale
from ..logging_utils import get_logger
log=get_logger(__name__)
OUT_FILE=settings.out_dir/"03_econ_features.csv"

def run():
    if not stale(OUT_FILE, []):
        return pd.read_csv(OUT_FILE,parse_dates=["date"])
    dfs=[]
    start="2023-01-01"
    for name,ticker in settings.econ_tickers.items():
        try:
            df=yf.download(ticker,start=start,progress=False,auto_adjust=True,threads=False)
            if df.empty: continue
            close=df["Close"] if "Close" in df.columns else df.squeeze()
            dfs.append(close.rename(name))
            log.info(f"Fetched {name}")
        except Exception as e:
            log.warning(f"{ticker} failed: {e}")
    if not dfs:
        log.error("No econ data pulled"); return pd.DataFrame()
    econ=pd.concat(dfs,axis=1).ffill().reset_index().rename(columns={"Date":"date"})
    econ["date"]=pd.to_datetime(econ["date"]).dt.normalize()
    econ.to_csv(OUT_FILE,index=False)
    log.info("Econ features saved")
    return econ
PY

# ------------------------------------------------------------------------------
# 9 ── Python driver to orchestrate Stage-1
# ------------------------------------------------------------------------------
python - <<'PY'
import traceback
from customer_comms.logging_utils       import get_logger
from customer_comms.processing.ingest   import run as ingest
from customer_comms.analytics.augment   import run as augment
from customer_comms.features.econ       import run as econ_run

log=get_logger("stage1")
try:
    calls,intents,mail=ingest()
    augment(calls,intents)
    econ_run()
    log.info("🎉  Stage-1 finished OK – artefacts ready")
except Exception:
    log.exception("Stage-1 FAILED")
    raise
PY

echo "-----------------------------------------------------------------"
echo "✅  Stage-1 artefacts in ./$OUT/ ; detailed logs in ./$LOG/"
echo "Re-run this script any time — it only rebuilds when inputs change."
echo "-----------------------------------------------------------------"
