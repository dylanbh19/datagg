#!/usr/bin/env bash
# -------------------------------------------------------------
#  Customer-Comms full scaffold + basic & advanced analytics
#  NO virtual-env – installs libs into system/user site-packages
# -------------------------------------------------------------
#  Usage:
#       ./setup_comms_full.sh            # creates ./customer_comms
#       ./setup_comms_full.sh  myproj    # creates ./myproj
# -------------------------------------------------------------
set -euo pipefail

PROJ="${1:-customer_comms}"
PKG="customer_comms"

echo "🛠  Creating/refreshing project: ${PROJ}"
mkdir -p "${PROJ}"/{data,logs,output,tests}
cd "${PROJ}"

# -------------------------------------------------------------
# 0. Dependencies
# -------------------------------------------------------------
echo "📦  Ensuring python dependencies …"
python3 -m pip install -q --upgrade pip
python3 -m pip install -q pandas numpy matplotlib seaborn scipy pydantic holidays

# -------------------------------------------------------------
# 1. Package skeleton
# -------------------------------------------------------------
mkdir -p "${PKG}"/{data,processing,analytics,viz}
touch "${PKG}"/__init__.py
for d in data processing analytics viz; do
  touch "${PKG}/${d}/__init__.py"
done

# -------------------------------------------------------------
# 2.  CONFIG  (column mapping & file names)
# -------------------------------------------------------------
cat > "${PKG}/config.py" << 'PY'
from pathlib import Path
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    # ---- CSV file names (place files in ./data/) ----
    call_files: list[str] = ["call_file_1.csv", "call_file_2.csv"]
    mail_file: str        = "mail_file.csv"

    # ---- Column names in those CSVs ----
    call_date_col: str    = "call_date"
    call_intent_col: str  = "uui_Intent"
    mail_date_col: str    = "mail_date"
    mail_type_col: str    = "mail_type"
    mail_volume_col: str  = "mail_volume"

    # ---- Misc processing ----
    data_dir:   Path = Field(default=Path("data"))
    log_dir:    Path = Field(default=Path("logs"))
    output_dir: Path = Field(default=Path("output"))
    min_rows:   int  = 20
    start_date: str  = "2024-01-01"
    max_lag:    int  = 21           # for lag scan

settings = Settings()
PY

# -------------------------------------------------------------
# 3.  LOGGING UTILS
# -------------------------------------------------------------
cat > "${PKG}/logging_utils.py" << 'PY'
import logging, sys
from datetime import datetime
from pathlib import Path
from .config import settings

def get_logger(name: str = "customer_comms") -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log

    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    dh  = "%Y-%m-%d %H:%M:%S"
    log.setLevel(logging.INFO)

    # Console
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(fmt, dh))
    log.addHandler(sh)

    # File
    settings.log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(settings.log_dir / f"{name}_{datetime.now():%Y%m%d}.log")
    fh.setFormatter(logging.Formatter(fmt, dh))
    log.addHandler(fh)

    return log
PY

# -------------------------------------------------------------
# 4.  DATA LOADER  (robust CSV + date cleaning)
# -------------------------------------------------------------
cat > "${PKG}/data/loader.py" << 'PY'
from __future__ import annotations
import pandas as pd, numpy as np, holidays
from pathlib import Path
from typing import List
from ..config import settings
from ..logging_utils import get_logger

log = get_logger(__name__)
us_holidays = holidays.UnitedStates()
ENCODINGS = ("utf-8","latin-1","cp1252","utf-16")

def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        log.error(f"Missing file: {path}")
        return pd.DataFrame()
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="skip", low_memory=False)
        except UnicodeDecodeError:
            continue
    log.error(f"Failed to decode {path}")
    return pd.DataFrame()

def _to_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.tz_localize(None).dt.normalize()

def load_call_data() -> tuple[pd.DataFrame,pd.DataFrame]:
    dfs=[]
    for fname in settings.call_files:
        df=_read_csv(settings.data_dir/fname)
        if df.empty: continue
        df=df.rename(columns={settings.call_date_col:"date",
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

def load_mail() -> pd.DataFrame:
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
# 5.  PROCESSING – intersection + normalisation
# -------------------------------------------------------------
cat > "${PKG}/processing/combine.py" << 'PY'
import pandas as pd, numpy as np
from ..data.loader import load_call_data, load_mail
from ..logging_utils import get_logger
from ..config import settings

log = get_logger(__name__)

def _norm(series: pd.Series)->pd.Series:
    if series.empty or series.max()==series.min(): return series*0
    return (series-series.min())/(series.max()-series.min())*100

def build_dataset() -> pd.DataFrame:
    call_vol,intents=load_call_data()
    mail=load_mail()
    if call_vol.empty or mail.empty:
        log.error("Missing mail or call data")
        return pd.DataFrame()

    mail_daily=(mail.groupby("date")["mail_volume"].sum().reset_index())
    df=pd.merge(call_vol,mail_daily,on="date",how="inner")
    if len(df)<settings.min_rows:
        log.error("Too few intersecting days")
        return pd.DataFrame()

    df["call_norm"]=_norm(df["call_volume"])
    df["mail_norm"]=_norm(df["mail_volume"])
    log.info(f"Combined rows after intersection: {len(df)}")
    return df.sort_values("date")
PY

# -------------------------------------------------------------
# 6.  BASIC OVERVIEW PNG (Matplotlib)
# -------------------------------------------------------------
cat > "${PKG}/viz/overview.py" << 'PY'
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from ..config import settings
from ..logging_utils import get_logger

log = get_logger(__name__)

def save_overview(df, fname="overview.png"):
    if df.empty:
        log.error("Empty DF – no plot")
        return
    plt.figure(figsize=(14,6))
    plt.bar(df["date"], df["mail_norm"], label="Mail (norm)", alpha=.6)
    plt.plot(df["date"], df["call_norm"], label="Calls (norm)", color="tab:red", lw=2)
    plt.title("Mail vs Call volume (normalised 0-100)")
    plt.ylabel("Normalised 0-100"); plt.xlabel("Date")
    plt.legend(); plt.grid(axis="y",ls="--",alpha=.4)
    ax=plt.gca()
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(),rotation=45,ha="right")
    plt.tight_layout()
    settings.output_dir.mkdir(exist_ok=True)
    out=settings.output_dir/fname
    plt.savefig(out,dpi=300); plt.close()
    log.info(f"Saved {out}")
PY

# -------------------------------------------------------------
# 7.  ADVANCED EDA (validation, corr, lag, rolling)
# -------------------------------------------------------------
cat > "${PKG}/analytics/eda.py" << 'PY'
import json, seaborn as sns, matplotlib.pyplot as plt, pandas as pd, numpy as np
from scipy.stats import pearsonr
from matplotlib.dates import DateFormatter
from ..config import settings
from ..logging_utils import get_logger

log=get_logger(__name__)

# ---------- Stage 1 : data validation ----------
def data_health(df: pd.DataFrame)->dict:
    rep={
        "rows":len(df),
        "date_range":[str(df["date"].min()), str(df["date"].max())],
        "na_by_col":df.isna().sum().to_dict(),
        "duplicate_dates":int(df["date"].duplicated().sum())
    }
    settings.output_dir.mkdir(exist_ok=True)
    with open(settings.output_dir/"data_health_report.json","w") as f:
        json.dump(rep,f,indent=2)
    log.info("Wrote data_health_report.json")
    return rep

# ---------- Stage 2 : correlation heatmap ----------
def corr_heatmap(df: pd.DataFrame):
    num=df.select_dtypes("number")
    corr=num.corr(method="pearson")
    plt.figure(figsize=(8,6))
    sns.heatmap(corr,annot=True,cmap="vlag",fmt=".2f")
    plt.title("Pearson correlation matrix")
    out=settings.output_dir/"corr_heatmap.png"
    plt.tight_layout(); plt.savefig(out,dpi=300); plt.close()
    log.info(f"Saved {out}")

    # significant cells
    sig=[]
    for c1 in num.columns:
        for c2 in num.columns:
            if c1>=c2: continue
            valid=df[[c1,c2]].dropna()
            if len(valid)<settings.min_rows: continue
            r,p=pearsonr(valid[c1],valid[c2])
            if p<0.05:
                sig.append({"var1":c1,"var2":c2,"r":r,"p":p})
    pd.DataFrame(sig).to_csv(settings.output_dir/"corr_significant.csv",index=False)

# ---------- Stage 3 : lag explorer ----------
def lag_scan(df: pd.DataFrame, feature="mail_volume",target="call_volume"):
    lags=range(0,settings.max_lag+1)
    vals=[]
    for lag in lags:
        x=df[feature]; y=df[target].shift(-lag)
        v=pd.concat([x,y],axis=1).dropna()
        if len(v)<settings.min_rows: continue
        r,_=pearsonr(v[feature],v[target]); vals.append(r)
    plt.figure(figsize=(10,4))
    plt.plot(lags,vals,marker="o")
    plt.title(f"Cross-corr {feature} → {target}")
    plt.xlabel("Lag days"); plt.ylabel("Pearson r"); plt.grid(ls="--",alpha=.4)
    out=settings.output_dir/"lag_scan.png"
    plt.tight_layout(); plt.savefig(out,dpi=300); plt.close()
    log.info(f"Saved {out}")

# ---------- Stage 4 : rolling stats ----------
def rolling_stats(df: pd.DataFrame, window=30):
    r=df.set_index("date")[["call_volume","mail_volume"]].rolling(window=window)
    roll=r.mean().rename(columns=lambda c:f"{c}_mean{window}")
    std =r.std().rename(columns=lambda c:f"{c}_std{window}")
    merged=pd.concat([df.set_index("date"),roll,std],axis=1).dropna()
    plt.figure(figsize=(14,6))
    plt.plot(merged.index, merged["call_volume_mean30"], label="Call 30-d mean",color="tab:red")
    plt.plot(merged.index, merged["mail_volume_mean30"], label="Mail 30-d mean",color="tab:blue")
    plt.fill_between(merged.index,
                     merged["call_volume_mean30"]-merged["call_volume_std30"],
                     merged["call_volume_mean30"]+merged["call_volume_std30"],
                     alpha=.2,color="tab:red")
    plt.title("30-day rolling mean ±1σ"); plt.legend(); plt.grid(ls="--",alpha=.4)
    ax=plt.gca(); ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(),rotation=45,ha="right")
    out=settings.output_dir/"rolling_stats.png"
    plt.tight_layout(); plt.savefig(out,dpi=300); plt.close()
    log.info(f"Saved {out}")
PY

# -------------------------------------------------------------
# 8.  BASIC CLI  (overview)
# -------------------------------------------------------------
cat > "${PKG}/cli.py" << 'PY'
import argparse
from .processing.combine import build_dataset
from .viz.overview import save_overview
from .logging_utils import get_logger

log=get_logger(__name__)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--plot",action="store_true")
    args=p.parse_args()
    df=build_dataset()
    if df.empty:
        log.error("build_dataset failed"); return
    if args.plot:
        save_overview(df)

if __name__=="__main__":
    main()
PY

# -------------------------------------------------------------
# 9.  ADVANCED PIPELINE  (stages 1-4)
# -------------------------------------------------------------
cat > "${PKG}/pipeline.py" << 'PY'
from .processing.combine import build_dataset
from .viz.overview import save_overview
from .analytics.eda import data_health, corr_heatmap, lag_scan, rolling_stats
from .logging_utils import get_logger

log=get_logger(__name__)

def main():
    log.info("🚀 Stage 0 – building base dataset")
    df=build_dataset()
    if df.empty:
        log.error("Dataset build failed – abort"); return
    save_overview(df)

    log.info("✅ Stage 1 – validation")
    data_health(df)

    log.info("✅ Stage 2 – correlation heat-map")
    corr_heatmap(df)

    log.info("✅ Stage 3 – lag scan")
    lag_scan(df)

    log.info("✅ Stage 4 – rolling stats")
    rolling_stats(df)
    log.info("🎉 Pipeline finished – check ./output/")
if __name__=="__main__":
    main()
PY

# -------------------------------------------------------------
# 10.  RUN the two command lines
# -------------------------------------------------------------
echo "🚀 Running base overview script ..."
python -m ${PKG}.cli --plot || echo "⚠️  Base CLI failed"

echo ""
echo "🚀 Running advanced pipeline ..."
python -m ${PKG}.pipeline || echo "⚠️  Advanced pipeline failed"

echo ""
echo "✅  All done.  Outputs -> output/, logs -> logs/"