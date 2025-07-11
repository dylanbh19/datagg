#!/usr/bin/env bash
# =========================================================================
#  part1_bootstrap_data.sh                 (Customer-Comms  –  Stage 1/4)
#  -----------------------------------------------------------------------
#  • Creates ./customer_comms package (fresh) plus logs/ output/
#  • Robust UTF-8 logging, multi-codec CSV reader
#  • Ingestion & profiling  (Story 1.1)
#  • Call-volume augmentation via intent MA  (Story 1.2)
#  • Economic-indicator ingest & feature-engineer  (Story 1.3)
#  • Make targets: ingest | augment | econ
# =========================================================================
set -euo pipefail
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

PKG="customer_comms"
echo "🛠️  Stage-1 bootstrap → $PKG"; rm -rf "$PKG" {logs,output} 2>/dev/null || true
mkdir -p "$PKG"/{data,processing,analytics,features} logs output
touch "$PKG"/__init__.py
for d in data processing analytics features; do  touch "$PKG/$d/__init__.py"; done

# -------------------------------------------------------------------------
# 0 . Dependencies (silent install only if missing)
# -------------------------------------------------------------------------
python - <<'PY'
import importlib, subprocess, sys, contextlib
deps = ['pandas','numpy','matplotlib','seaborn','scipy',
        'scikit-learn','statsmodels','holidays',
        'pydantic','pydantic-settings','yfinance']
for p in deps:
    with contextlib.suppress(ModuleNotFoundError):
        importlib.import_module(p.replace('-','_')); continue
    subprocess.check_call([sys.executable,'-m','pip','install','-q',p])
PY

# -------------------------------------------------------------------------
# 1 . config.py
# -------------------------------------------------------------------------
cat > "$PKG/config.py" << 'PY'
from pathlib import Path
try:    from pydantic_settings import BaseSettings          # Pydantic v2
except ModuleNotFoundError: from pydantic import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # ==== raw CSVs =======================================================
    call_files: list[str] = ["GenesysExtract_20250609.csv",
                             "GenesysExtract_20250703.csv"]
    mail_file:  str       = "merged_call_data.csv"

    # ==== columns ========================================================
    call_date_cols:   list[str] = ["ConversationStart","Date"]
    call_intent_cols: list[str] = ["uui_Intent","intent","Intent"]
    mail_date_col:    str = "mail_date"
    mail_type_col:    str = "mail_type"
    mail_volume_col:  str = "mail_volume"

    # ==== augmentation + thresholds =====================================
    augment_gap_limit:int = 3
    ma_window:int         = 7
    min_intent_rows:int   = 250
    min_rows:int          = 20

    # ==== economic feed ==================================================
    econ_tickers: dict[str,str] = {
        "sp500"      : "^GSPC",
        "vix"        : "^VIX",
        "fed_funds"  : "FEDFUNDS",      # FRED code via yfinance
        "unemploy"   : "UNRATE",
    }

    # ==== dirs ===========================================================
    data_dir: Path = Field(default=Path("data"))
    out_dir:  Path = Field(default=Path("output"))
    log_dir:  Path = Field(default=Path("logs"))
settings = Settings()
PY

# -------------------------------------------------------------------------
# 2 . logging_utils.py
# -------------------------------------------------------------------------
cat > "$PKG/logging_utils.py" << 'PY'
import logging, sys, traceback
from datetime import datetime
from .config import settings

def get_logger(name="customer_comms"):
    log = logging.getLogger(name)
    if log.handlers: return log
    log.setLevel(logging.INFO)
    fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s"; dh="%Y-%m-%d %H:%M:%S"
    sh=logging.StreamHandler(sys.stdout); sh.setFormatter(logging.Formatter(fmt,dh))
    try: sh.stream.reconfigure(encoding="utf-8")
    except AttributeError: pass
    log.addHandler(sh)
    settings.log_dir.mkdir(exist_ok=True)
    fh=logging.FileHandler(settings.log_dir/f"{name}_{datetime.now():%Y%m%d}.log",
                           encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt,dh)); log.addHandler(fh)
    return log
PY

# -------------------------------------------------------------------------
# 3 . data/loader.py  – multi-codec reader
# -------------------------------------------------------------------------
cat > "$PKG/data/loader.py" << 'PY'
from __future__ import annotations
import pandas as pd, io
from pathlib import Path
from ..config import settings
from ..logging_utils import get_logger
log=get_logger(__name__)
_CODECS=["utf-8","utf-8-sig","latin-1","cp1252","utf-16","iso-8859-1"]
def _read(path:Path)->pd.DataFrame:
    if not path.exists():
        log.error(f"Missing {path}"); return pd.DataFrame()
    for enc in _CODECS:
        try: return pd.read_csv(path,encoding=enc,on_bad_lines="skip",low_memory=False)
        except UnicodeDecodeError: continue
    text=path.read_bytes().decode("utf-8","replace")
    return pd.read_csv(io.StringIO(text),on_bad_lines="skip",low_memory=False)
def _dt(s): return pd.to_datetime(s,errors="coerce").dt.tz_localize(None).dt.normalize()
PY

# -------------------------------------------------------------------------
# 4 . processing/ingest.py  – Story 1.1
# -------------------------------------------------------------------------
cat > "$PKG/processing/ingest.py" << 'PY'
"""Load & profile raw data (Story 1.1)."""
import pandas as pd, json
from ..config import settings
from ..data.loader import _read, _dt
from ..logging_utils import get_logger
log=get_logger(__name__)
def run():
    # ---- call volumes + intents ---------------------------------------
    frames=[]; intents=False
    for fn in settings.call_files:
        df=_read(settings.data_dir/fn); df.columns=df.columns.str.strip()
        dcol=next((c for c in settings.call_date_cols if c in df.columns),None)
        icol=next((c for c in settings.call_intent_cols if c in df.columns),None)
        if not dcol: continue
        df=df.rename(columns={dcol:"date",**({icol:"intent"} if icol else {})})
        df["date"]=_dt(df["date"]); frames.append(df[["date"]+(['intent'] if icol else [])])
        intents |= bool(icol)
    raw_call=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()

    # ---- mail ----------------------------------------------------------
    mail=_read(settings.data_dir/settings.mail_file)
    if not mail.empty:
        mail=mail.rename(columns={settings.mail_date_col:"date",
                                  settings.mail_type_col:"mail_type",
                                  settings.mail_volume_col:"mail_volume"})
        mail["date"]=_dt(mail["date"])
        mail["mail_volume"]=pd.to_numeric(mail["mail_volume"],errors="coerce")
        mail=mail.dropna(subset=["date","mail_volume"])

    # ---- profile -------------------------------------------------------
    def prof(df,name): return {
        "name":name,"rows":len(df),
        "dtypes":df.dtypes.astype(str).to_dict() if not df.empty else {},
        "na":df.isna().sum().to_dict() if not df.empty else {}}
    rpt=[prof(raw_call,"call"),prof(mail,"mail")]
    settings.out_dir.mkdir(exist_ok=True)
    json.dump(rpt,open(settings.out_dir/"01_data_quality_report.json","w",encoding="utf-8"),indent=2)
    log.info("Data-quality report saved")
    return raw_call, mail
PY

# -------------------------------------------------------------------------
# 5 . analytics/augment.py  – Story 1.2
# -------------------------------------------------------------------------
cat > "$PKG/analytics/augment.py" << 'PY'
"""MA-aligned call-volume augmentation (Story 1.2)."""
import pandas as pd, numpy as np
from ..config import settings
from ..logging_utils import get_logger
log=get_logger(__name__)
def augment(call:pd.DataFrame)->pd.DataFrame:
    if call.empty or "intent" not in call.columns:
        log.warning("No intent column - augmentation skipped"); return pd.DataFrame()
    intent_daily=(call.groupby(["date","intent"]).size().unstack(fill_value=0).reset_index())
    total_int=intent_daily.set_index("date").sum(axis=1)
    call_daily=call.groupby("date").size().rename("call_volume").to_frame()
    df=call_daily.join(total_int.rename("intent_total"),how="inner")
    w=settings.ma_window
    scale=(df["intent_total"].rolling(w,1).mean() /
           df["call_volume"].rolling(w,1).mean().replace(0,np.nan)).clip(0.25,4).fillna(1)
    df["call_volume_aug"]=df["call_volume"]*scale
    df=df.reset_index()
    log.info(f"Augmented call volume (MA w={w}) rows={len(df)}")
    return df
PY

# -------------------------------------------------------------------------
# 6 . features/econ.py  – Story 1.3
# -------------------------------------------------------------------------
cat > "$PKG/features/econ.py" << 'PY'
"""Download & align economic indicators (Story 1.3)."""
import pandas as pd, yfinance as yf
from ..config import settings
from ..logging_utils import get_logger
log=get_logger(__name__)
def fetch(start:str="2023-01-01"):
    dfs=[]
    for name,ticker in settings.econ_tickers.items():
        log.info(f"Fetching {name} ({ticker})")
        data=yf.download(ticker,start=start,progress=False,auto_adjust=True,threads=False)
        if data.empty: continue
        series=(data["Close"] if "Close" in data.columns else data.squeeze())
        dfs.append(series.rename(name))
    if not dfs: return pd.DataFrame()
    econ=pd.concat(dfs,axis=1).ffill().reset_index().rename(columns={"Date":"date"})
    econ["date"]=pd.to_datetime(econ["date"]).dt.normalize()
    return econ
PY

# -------------------------------------------------------------------------
# 7 . Makefile (Stage-1 targets)
# -------------------------------------------------------------------------
cat > Makefile << 'MK'
.PHONY: ingest augment econ clean

ingest:        ## Load & profile raw data
	python - <<'PY'
import customer_comms.processing.ingest as ing; ing.run()
PY

augment:       ## MA-align call augmentation
	python - <<'PY'
import customer_comms.processing.ingest as ing, customer_comms.analytics.augment as aug, pandas as pd
call, _ = ing.run(); df=aug.augment(call)
if not df.empty: df.to_csv("output/02_augmented_calls.csv",index=False)
print("→ output/02_augmented_calls.csv")
PY

econ:          ## Download economic indicators
	python - <<'PY'
from customer_comms.features.econ import fetch; import pandas as pd
econ=fetch(); econ.to_csv("output/03_econ_features.csv",index=False)
print("→ output/03_econ_features.csv")
PY

clean:
	rm -rf logs/* output/*
MK

# -------------------------------------------------------------------------
# 8 . smoke-test (non-fatal)
# -------------------------------------------------------------------------
python - <<'PY'
from customer_comms.processing.ingest import run
from customer_comms.analytics.augment import augment
from customer_comms.features.econ import fetch
from customer_comms.logging_utils import get_logger
log=get_logger("smoke")
try:
    call, _ = run(); _ = augment(call); _ = fetch()
    log.info("Stage-1 smoke-test OK")
except Exception as e:
    log.critical(f"Smoke-test failed: {e}", exc_info=True)
PY

echo "✅  Stage 1 complete.  Next:"
echo "   make ingest   # profile data"
echo "   make augment  # call-volume augmentation"
echo "   make econ     # economics alignment"
echo "   Logs → ./logs/, artefacts → ./output/"
