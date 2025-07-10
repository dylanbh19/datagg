#!/usr/bin/env bash
# -------------------------------------------------------------
#  Customer-Comms full scaffold + load/clean + Stage-0…4 plots
#  * Flexible call-date column detection
#  * Works with Pydantic v1 or v2 (via pydantic-settings)
# -------------------------------------------------------------
#  Usage:
#       ./setup_comms_full.sh              # ./customer_comms
#       ./setup_comms_full.sh myproject    # ./myproject
# -------------------------------------------------------------
set -euo pipefail

PROJ="${1:-customer_comms}"
PKG="customer_comms"

echo "🔧 Creating / refreshing project: ${PROJ}"
mkdir -p "${PROJ}"/{data,logs,output,tests}
cd "${PROJ}"

# -------------------------------------------------------------
# 0.  Dependencies  (add pydantic-settings for v2 support)
# -------------------------------------------------------------
echo "📦  Ensuring python dependencies …"
python -m pip install -q --upgrade pip
python -m pip install -q pandas numpy matplotlib seaborn scipy \
                          pydantic pydantic-settings holidays

# -------------------------------------------------------------
# 1.  Package skeleton
# -------------------------------------------------------------
mkdir -p "${PKG}"/{data,processing,analytics,viz}
touch "${PKG}"/__init__.py
for d in data processing analytics viz; do
  touch "${PKG}/${d}/__init__.py"
done

# -------------------------------------------------------------
# 2.  config.py  – flexible date columns list
# -------------------------------------------------------------
cat > "${PKG}/config.py" << 'PY'
from pathlib import Path

# -------------------------------------------------------------
#  Pydantic v1 / v2 compatibility shim
# -------------------------------------------------------------
try:
    from pydantic_settings import BaseSettings
except ModuleNotFoundError:          # v1
    from pydantic import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # ---- CSV files (put in ./data/) ----
    call_files: list[str] = ["GenesysExtract_20250609.csv", "GenesysExtract_20250703.csv"]
    mail_file:  str       = "merged_call_data.csv"

    # ---- Column names ----
    call_date_cols: list[str] = ["ConversationStart", "Date"]  # ORDER matters
    call_intent_col: str      = "uui_Intent"

    mail_date_col:  str = "mail_date"
    mail_type_col:  str = "mail_type"
    mail_volume_col:str = "mail_volume"

    # ---- Misc ----
    data_dir:   Path = Field(default=Path("data"))
    log_dir:    Path = Field(default=Path("logs"))
    output_dir: Path = Field(default=Path("output"))
    min_rows:   int  = 20
    start_date: str  = "2024-01-01"
    max_lag:    int  = 21        # for lag scan

settings = Settings()
PY

# -------------------------------------------------------------
# 3.  logging_utils.py
# -------------------------------------------------------------
cat > "${PKG}/logging_utils.py" << 'PY'
import logging, sys
from pathlib import Path
from datetime import datetime
from .config import settings

def get_logger(name: str = "customer_comms") -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    dh  = "%Y-%m-%d %H:%M:%S"
    log.setLevel(logging.INFO)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(fmt, dh))
    log.addHandler(sh)

    settings.log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(settings.log_dir /
                             f"{name}_{datetime.now():%Y%m%d}.log")
    fh.setFormatter(logging.Formatter(fmt, dh))
    log.addHandler(fh)
    return log
PY

# -------------------------------------------------------------
# 4.  data/loader.py  – flexible date detection
# -------------------------------------------------------------
cat > "${PKG}/data/loader.py" << 'PY'
from __future__ import annotations
import pandas as pd, holidays
from pathlib import Path
from ..config import settings
from ..logging_utils import get_logger

log = get_logger(__name__)
ENCODINGS = ("utf-8","latin-1","cp1252","utf-16")
us_holidays = holidays.UnitedStates()

def _read_csv(path: Path)->pd.DataFrame:
    if not path.exists():
        log.error(f"File not found: {path}"); return pd.DataFrame()
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path,encoding=enc,on_bad_lines="skip",low_memory=False)
        except UnicodeDecodeError: continue
    log.error(f"Could not decode {path}"); return pd.DataFrame()

def _to_date(series: pd.Series)->pd.Series:
    return pd.to_datetime(series,errors="coerce").dt.tz_localize(None).dt.normalize()

# ------------------ CALL DATA -------------------------------
def load_call_data():
    dfs=[]
    for fname in settings.call_files:
        path=settings.data_dir/fname
        df=_read_csv(path)
        if df.empty: continue

        # find date column
        date_col=None
        for cand in settings.call_date_cols:
            if cand in df.columns: date_col=cand; break
        if date_col is None:
            log.error(f"No date column in {fname}; looked for {settings.call_date_cols}")
            continue

        df=df.rename(columns={date_col:"date",
                              settings.call_intent_col:"intent"})
        df["date"]=_to_date(df["date"])
        df=df.dropna(subset=["date"])
        dfs.append(df[["date","intent"]])

    if not dfs:
        return pd.DataFrame(),pd.DataFrame()

    raw=pd.concat(dfs,ignore_index=True)

    call_volume=(raw.groupby("date").size()
                      .reset_index(name="call_volume")
                      .sort_values("date"))

    intents=(raw.groupby(["date","intent"]).size()
                  .unstack(fill_value=0)
                  .reset_index()
                  .sort_values("date"))

    log.info(f"Call volume rows: {len(call_volume)}")
    return call_volume,intents

# ------------------ MAIL DATA -------------------------------
def load_mail():
    df=_read_csv(settings.data_dir/settings.mail_file)
    if df.empty: return df
    df=df.rename(columns={settings.mail_date_col:"date",
                          settings.mail_type_col:"mail_type",
                          settings.mail_volume_col:"mail_volume"})
    df["date"]=_to_date(df["date"])
    df["mail_volume"]=pd.to_numeric(df["mail_volume"],errors="coerce")
    df=df.dropna(subset=["date","mail_volume"])
    log.info(f"Mail rows: {len(df)}")
    return df
PY

# -------------------------------------------------------------
# 5.  processing/combine.py
# -------------------------------------------------------------
cat > "${PKG}/processing/combine.py" << 'PY'
import pandas as pd, numpy as np
from ..data.loader import load_call_data, load_mail
from ..config import settings
from ..logging_utils import get_logger
log=get_logger(__name__)

def _norm(s):
    return (s-s.min())/(s.max()-s.min())*100 if s.max()!=s.min() else s*0

def build_dataset():
    call_vol,intents=load_call_data()
    mail=load_mail()
    if call_vol.empty or mail.empty:
        log.error("Missing call/mail"); return pd.DataFrame()

    mail_daily=mail.groupby("date")["mail_volume"].sum().reset_index()
    df=pd.merge(call_vol,mail_daily,on="date",how="inner")
    if len(df)<settings.min_rows:
        log.error("Too few intersecting days"); return pd.DataFrame()

    df["call_norm"]=_norm(df["call_volume"])
    df["mail_norm"]=_norm(df["mail_volume"])
    log.info(f"Combined rows: {len(df)}")
    return df.sort_values("date")
PY

# -------------------------------------------------------------
# 6.  viz/overview.py
# -------------------------------------------------------------
cat > "${PKG}/viz/overview.py" << 'PY'
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from ..config import settings
from ..logging_utils import get_logger
log=get_logger(__name__)

def save_overview(df,fname="overview.png"):
    if df.empty: return
    plt.figure(figsize=(14,6))
    plt.bar(df["date"],df["mail_norm"],alpha=.6,label="Mail (norm)")
    plt.plot(df["date"],df["call_norm"],color="tab:red",lw=2,label="Calls (norm)")
    plt.title("Mail vs Call volume (0-100 normalised)")
    plt.ylabel("0-100"); plt.xlabel("Date"); plt.legend(); plt.grid(ls="--",alpha=.3)
    ax=plt.gca(); ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(),rotation=45,ha="right")
    plt.tight_layout()
    settings.output_dir.mkdir(exist_ok=True)
    out=settings.output_dir/fname
    plt.savefig(out,dpi=300); plt.close()
    log.info(f"Saved {out}")
PY

# -------------------------------------------------------------
# 7.  analytics/eda.py  (Stages 1-4)
# -------------------------------------------------------------
cat > "${PKG}/analytics/eda.py" << 'PY'
import json, seaborn as sns, matplotlib.pyplot as plt, pandas as pd
from scipy.stats import pearsonr
from matplotlib.dates import DateFormatter
from ..config import settings
from ..logging_utils import get_logger
log=get_logger(__name__)

def validate(df):
    rep={"rows":len(df),
         "date_range":[str(df["date"].min()),str(df["date"].max())],
         "na":df.isna().sum().to_dict(),
         "dup_dates":int(df["date"].duplicated().sum())}
    with open(settings.output_dir/"data_health_report.json","w") as f:
        json.dump(rep,f,indent=2)
    log.info("Health report saved")

def corr_heat(df):
    num=df.select_dtypes("number")
    corr=num.corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr,annot=True,cmap="vlag",fmt=".2f")
    plt.title("Pearson correlation")
    out=settings.output_dir/"corr_heatmap.png"
    plt.tight_layout(); plt.savefig(out,dpi=300); plt.close()
    log.info(f"Saved {out}")

def lag_scan(df,feat="mail_volume",tgt="call_volume"):
    lags=range(0,settings.max_lag+1)
    vals=[]
    for lag in lags:
        v=df[[feat,tgt]].dropna()
        r,_=pearsonr(v[feat],v[tgt].shift(-lag).dropna())
        vals.append(r)
    plt.figure(figsize=(10,4))
    plt.plot(lags,vals,marker="o"); plt.grid(ls="--",alpha=.4)
    plt.title("Mail → Call cross-correlation"); plt.xlabel("Lag (days)"); plt.ylabel("r")
    out=settings.output_dir/"lag_scan.png"
    plt.tight_layout(); plt.savefig(out,dpi=300); plt.close()
    log.info(f"Saved {out}")

def rolling(df,window=30):
    r=df.set_index("date")[["call_volume","mail_volume"]].rolling(window)
    m=r.mean(); s=r.std()
    plt.figure(figsize=(14,6))
    plt.plot(m.index,m["call_volume"],label=f"Call mean {window}",color="tab:red")
    plt.plot(m.index,m["mail_volume"],label=f"Mail mean {window}",color="tab:blue")
    plt.fill_between(m.index,
                     m["call_volume"]-s["call_volume"],
                     m["call_volume"]+s["call_volume"],
                     alpha=.2,color="tab:red")
    plt.title(f"{window}-day rolling mean ±1σ")
    ax=plt.gca(); ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(),rotation=45,ha="right")
    plt.legend(); plt.grid(ls="--",alpha=.3); plt.tight_layout()
    out=settings.output_dir/"rolling_stats.png"
    plt.savefig(out,dpi=300); plt.close()
    log.info(f"Saved {out}")
PY

# -------------------------------------------------------------
# 8.  pipeline.py  – build + all plots
# -------------------------------------------------------------
cat > "${PKG}/pipeline.py" << 'PY'
from .processing.combine import build_dataset
from .viz.overview import save_overview
from .analytics.eda import validate, corr_heat, lag_scan, rolling
from .logging_utils import get_logger
log=get_logger(__name__)

def main():
    df=build_dataset()
    if df.empty: 
        log.error("Dataset build failed"); return
    save_overview(df)
    validate(df); corr_heat(df); lag_scan(df); rolling(df)
    log.info("🎉 All stages completed.  Check ./output/")
if __name__=="__main__":
    main()
PY

# -------------------------------------------------------------
# 9.  RUN – simple load/clean/plot pipeline
# -------------------------------------------------------------
echo "🚀 Running full load-clean-EDA pipeline …"
python -m ${PKG}.pipeline || echo "⚠️  Pipeline failed"

echo ""
echo "✅  Done.  Plots & reports → ./output/, logs → ./logs/"
