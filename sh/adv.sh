#!/usr/bin/env bash
# =====================================================================
#  setup_full_pipeline.sh
#  • ONE SHOT: create / refresh customer_comms package
#  • UTF-8 logging on Windows
#  • Weekday-only analysis
#  • Generates 13 artefacts (5 stage-0…4 + 4 extras + 3 new correlations
#    + data-gap visual)
# =====================================================================
set -euo pipefail

# ---------------------------------------------------------------------
# Choose / create project directory
# ---------------------------------------------------------------------
PROJ="${1:-customer_comms}"      # ./setup_full_pipeline.sh MyProj
PKG="customer_comms"

echo "🔧  Creating / refreshing project: ${PROJ}"
mkdir -p "$PROJ"/{data,logs,output,tests}
cd "$PROJ"

# ---------------------------------------------------------------------
# 0.  Dependencies (quiet install if already there)
# ---------------------------------------------------------------------
echo "📦  Ensuring Python dependencies …"
python - <<'PY'
import importlib, subprocess, sys
pkgs = ['pandas','numpy','matplotlib','seaborn','scipy',
        'scikit-learn','statsmodels','holidays',
        'pydantic','pydantic-settings']
for p in pkgs:
    try: importlib.import_module(p.replace('-','_'))
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable,'-m','pip','install','-q',p])
PY

# ---------------------------------------------------------------------
# 1.  Skeleton and __init__.py
# ---------------------------------------------------------------------
for d in "$PKG" "$PKG/data" "$PKG/processing" "$PKG/analytics" "$PKG/viz"; do
  mkdir -p "$d"; touch "$d/__init__.py"
done

# ---------------------------------------------------------------------
# 2.  config.py
# ---------------------------------------------------------------------
cat > "$PKG/config.py" << 'PY'
from pathlib import Path
try:
    from pydantic_settings import BaseSettings           # Pydantic v2
except ModuleNotFoundError:
    from pydantic import BaseSettings                    # Pydantic v1
from pydantic import Field

class Settings(BaseSettings):
    # CSVs (put raw files in ./data)
    call_files: list[str] = ["GenesysExtract_20250609.csv",
                             "GenesysExtract_20250703.csv"]
    mail_file:  str       = "merged_call_data.csv"

    # Column names
    call_date_cols:   list[str] = ["ConversationStart", "Date"]
    call_intent_cols: list[str] = ["uui_Intent", "uuiIntent", "intent", "Intent"]

    mail_date_col:   str = "mail_date"
    mail_type_col:   str = "mail_type"
    mail_volume_col: str = "mail_volume"

    # Misc
    data_dir:   Path = Field(default=Path("data"))
    log_dir:    Path = Field(default=Path("logs"))
    output_dir: Path = Field(default=Path("output"))
    min_rows:   int  = 20
    max_lag:    int  = 21

settings = Settings()
PY

# ---------------------------------------------------------------------
# 3.  logging_utils.py (UTF-8 console & file)
# ---------------------------------------------------------------------
cat > "$PKG/logging_utils.py" << 'PY'
import logging, sys
from datetime import datetime
from .config import settings

def get_logger(name="customer_comms"):
    log = logging.getLogger(name)
    if log.handlers:
        return log

    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    dt  = "%Y-%m-%d %H:%M:%S"
    log.setLevel(logging.INFO)

    con = logging.StreamHandler(sys.stdout)
    con.setFormatter(logging.Formatter(fmt, dt))
    try: con.stream.reconfigure(encoding="utf-8")
    except AttributeError: pass
    log.addHandler(con)

    settings.log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(settings.log_dir /
                             f"{name}_{datetime.now():%Y%m%d}.log",
                             encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt, dt))
    log.addHandler(fh)
    return log
PY

# ---------------------------------------------------------------------
# 4.  data/loader.py  (flex date + optional intent)
# ---------------------------------------------------------------------
cat > "$PKG/data/loader.py" << 'PY'
from __future__ import annotations
import pandas as pd, holidays
from pathlib import Path
from ..config import settings
from ..logging_utils import get_logger
log = get_logger(__name__)

_ENCODINGS = ("utf-8","latin-1","cp1252","utf-16")
def _read(p: Path)->pd.DataFrame:
    if not p.exists(): log.error(f"Missing {p}"); return pd.DataFrame()
    for enc in _ENCODINGS:
        try: return pd.read_csv(p, encoding=enc, on_bad_lines="skip", low_memory=False)
        except UnicodeDecodeError: continue
    log.error(f"Cannot decode {p}"); return pd.DataFrame()

def _to_date(s: pd.Series)->pd.Series:
    return (pd.to_datetime(s, errors="coerce")
              .dt.tz_localize(None)
              .dt.normalize())

# ------------------- CALL DATA -------------------
def load_call_data():
    dfs=[]
    for fname in settings.call_files:
        df=_read(settings.data_dir/fname)
        if df.empty: continue
        date_col=next((c for c in settings.call_date_cols if c in df.columns),None)
        if not date_col: log.error(f"{fname}: date col missing"); continue
        intent_col=next((c for c in settings.call_intent_cols if c in df.columns),None)
        rename={date_col:"date"} | ({intent_col:"intent"} if intent_col else {})
        df=df.rename(columns=rename)
        df["date"]=_to_date(df["date"]); df=df.dropna(subset=["date"])
        cols=["date"]+(["intent"] if "intent" in df.columns else [])
        dfs.append(df[cols])
    if not dfs: return pd.DataFrame(), pd.DataFrame()
    raw=pd.concat(dfs,ignore_index=True)
    call_vol=(raw.groupby("date").size()
                    .reset_index(name="call_volume")
                    .sort_values("date"))
    if "intent" in raw.columns:
        intents=(raw.groupby(["date","intent"]).size()
                      .unstack(fill_value=0)
                      .reset_index()
                      .sort_values("date"))
    else: intents=pd.DataFrame()
    return call_vol, intents

# ------------------- MAIL DATA -------------------
def load_mail():
    df=_read(settings.data_dir/settings.mail_file)
    if df.empty: return df
    df=df.rename(columns={settings.mail_date_col:"date",
                          settings.mail_type_col:"mail_type",
                          settings.mail_volume_col:"mail_volume"})
    df["date"]=_to_date(df["date"])
    df["mail_volume"]=pd.to_numeric(df["mail_volume"],errors="coerce")
    return df.dropna(subset=["date","mail_volume"])
PY

# ---------------------------------------------------------------------
# 5.  processing/combine.py  (weekday-only)
# ---------------------------------------------------------------------
cat > "$PKG/processing/combine.py" << 'PY'
import pandas as pd, numpy as np
from ..data.loader import load_call_data, load_mail
from ..config import settings
from ..logging_utils import get_logger
log=get_logger(__name__)
_norm=lambda s:(s-s.min())/(s.max()-s.min())*100 if s.max()!=s.min() else s*0
def build_dataset():
    call_vol,_=load_call_data(); mail=load_mail()
    call_vol=call_vol[call_vol["date"].dt.weekday<5]
    mail    =mail    [mail    ["date"].dt.weekday<5]
    if call_vol.empty or mail.empty:
        log.error("No weekday call/mail"); return pd.DataFrame()
    mail_daily=mail.groupby("date")["mail_volume"].sum().reset_index()
    df=pd.merge(call_vol, mail_daily, on="date", how="inner")
    if len(df)<settings.min_rows:
        log.error("Too few intersect weekdays"); return pd.DataFrame()
    df["call_norm"]=_norm(df["call_volume"]); df["mail_norm"]=_norm(df["mail_volume"])
    return df.sort_values("date")
PY

# ---------------------------------------------------------------------
# 6.  viz/overview.py
# ---------------------------------------------------------------------
cat > "$PKG/viz/overview.py" << 'PY'
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from ..config import settings
from ..logging_utils import get_logger
log=get_logger(__name__)
def save_overview(df,fname="overview.png"):
    if df.empty: return
    plt.figure(figsize=(14,6))
    plt.bar(df["date"],df["mail_norm"],alpha=.6,label="Mail (norm)")
    plt.plot(df["date"],df["call_norm"],lw=2,color="tab:red",label="Calls (norm)")
    plt.title("Mail vs Call volume (normalised 0–100, weekdays)")
    plt.xlabel("Date"); plt.ylabel("0-100"); plt.legend(); plt.grid(ls="--",alpha=.3)
    ax=plt.gca(); ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(),rotation=45,ha="right")
    settings.output_dir.mkdir(exist_ok=True)
    plt.tight_layout(); plt.savefig(settings.output_dir/fname,dpi=300); plt.close()
    log.info(f"Saved {settings.output_dir/fname}")
PY

# ---------------------------------------------------------------------
# 7.  viz/raw_calls.py
# ---------------------------------------------------------------------
cat > "$PKG/viz/raw_calls.py" << 'PY'
import matplotlib.pyplot as plt, pandas as pd, os
from matplotlib.dates import DateFormatter
from ..config import settings; from ..logging_utils import get_logger
log=get_logger(__name__)
def plot_raw_call_files():
    paths=[settings.data_dir/f for f in settings.call_files if (settings.data_dir/f).exists()]
    if not paths: return
    series=[]
    for p in paths:
        df=pd.read_csv(p,low_memory=False)
        date_col=next((c for c in settings.call_date_cols if c in df.columns),None)
        if not date_col: continue
        df[date_col]=pd.to_datetime(df[date_col],errors="coerce").dt.normalize()
        daily=(df.groupby(date_col).size().reset_index(name="calls")
                 .rename(columns={date_col:"date"}))
        daily=daily[daily["date"].dt.weekday<5]
        daily["file"]=os.path.basename(p); series.append(daily)
    if not series: return
    full=pd.concat(series)
    plt.figure(figsize=(14,6))
    for f,sub in full.groupby("file"):
        plt.plot(sub["date"],sub["calls"],label=f)
    plt.title("Raw daily call volume (weekdays)"); plt.xlabel("Date"); plt.ylabel("Calls")
    plt.legend(); plt.grid(ls="--",alpha=.3)
    ax=plt.gca(); ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(),rotation=45,ha="right"); plt.tight_layout()
    out=settings.output_dir/"raw_call_files.png"; plt.savefig(out,dpi=300); plt.close()
    log.info(f"Saved {out}")
PY

# ---------------------------------------------------------------------
# 8.  viz/data_gaps.py
# ---------------------------------------------------------------------
cat > "$PKG/viz/data_gaps.py" << 'PY'
import pandas as pd, matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from ..data.loader import load_call_data, load_mail
from ..config import settings
from ..logging_utils import get_logger
log=get_logger(__name__)
def plot_data_gaps():
    calls,_=load_call_data(); mail=load_mail()
    calls=calls[calls["date"].dt.weekday<5]; mail=mail[mail["date"].dt.weekday<5]
    if calls.empty and mail.empty: return
    start=min(calls["date"].min() if not calls.empty else mail["date"].min(),
              mail["date"].min() if not mail.empty else calls["date"].min())
    end=max(calls["date"].max() if not calls.empty else mail["date"].max(),
            mail["date"].max() if not mail.empty else calls["date"].max())
    cal=pd.DataFrame({"date":pd.bdate_range(start,end,freq="B")})
    cal["call"]=cal["date"].isin(set(calls["date"]))
    cal["mail"]=cal["date"].isin(set(mail["date"]))
    cal["status"]=cal.apply(lambda r:"Both" if r.call and r.mail
                                        else("Call only" if r.call
                                             else("Mail only" if r.mail else "None")),axis=1)
    col={"Both":"green","Call only":"red","Mail only":"blue","None":"grey"}
    plt.figure(figsize=(14,2))
    for s,sub in cal.groupby("status"):
        plt.scatter(sub["date"],[0]*len(sub),c=col[s],s=10,label=s)
    plt.legend(ncol=4,loc="upper center"); plt.yticks([])
    plt.title("Weekday data coverage"); plt.grid(False)
    ax=plt.gca(); ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(),rotation=45,ha="right"); plt.tight_layout()
    out=settings.output_dir/"data_gaps.png"; plt.savefig(out,dpi=300); plt.close()
    log.info(f"Saved {out}")
PY

# ---------------------------------------------------------------------
# 9.  analytics/eda.py  (Stage-0…4 helpers)
# ---------------------------------------------------------------------
cat > "$PKG/analytics/eda.py" << 'PY'
import json, seaborn as sns, matplotlib.pyplot as plt, pandas as pd, numpy as np
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
    settings.output_dir.mkdir(exist_ok=True)
    json.dump(rep,open(settings.output_dir/"data_health_report.json","w",encoding="utf-8"),indent=2)
    log.info("Saved data_health_report.json")
def corr_heat(df):
    num=df.select_dtypes("number"); 
    if num.empty: return
    plt.figure(figsize=(8,6)); sns.heatmap(num.corr(),annot=True,cmap="vlag",fmt=".2f")
    plt.title("Pearson correlation (weekdays)"); plt.tight_layout()
    out=settings.output_dir/"corr_heatmap.png"; plt.savefig(out,dpi=300); plt.close()
    log.info(f"Saved {out}")
def lag_scan(df,feat="mail_volume",tgt="call_volume"):
    lags=range(0,settings.max_lag+1); vals=[]
    for lag in lags:
        tmp=df[[feat,tgt]].dropna()
        r,_=pearsonr(tmp[feat], tmp[tgt].shift(-lag).dropna()); vals.append(r)
    plt.figure(figsize=(10,4)); plt.plot(lags,vals,marker="o"); plt.grid(ls="--",alpha=.4)
    plt.title("Mail→Call cross-correlation (weekdays)"); plt.xlabel("Lag (days)"); plt.ylabel("r")
    out=settings.output_dir/"lag_scan.png"; plt.tight_layout()
    plt.savefig(out,dpi=300); plt.close(); log.info(f"Saved {out}")
def rolling(df,window=30):
    if len(df)<window+5: return
    r=df.set_index("date")[["call_volume","mail_volume"]].rolling(window)
    m=r.mean(); s=r.std()
    plt.figure(figsize=(14,6))
    plt.plot(m.index,m["call_volume"],label=f"Call mean {window}",color="tab:red")
    plt.plot(m.index,m["mail_volume"],label=f"Mail mean {window}",color="tab:blue")
    plt.fill_between(m.index,m["call_volume"]-s["call_volume"],
                                m["call_volume"]+s["call_volume"],alpha=.2,color="tab:red")
    plt.title(f"{window}-day rolling mean ±1σ (weekdays)")
    ax=plt.gca(); ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(),rotation=45,ha="right")
    plt.legend(); plt.grid(ls="--",alpha=.3); plt.tight_layout()
    out=settings.output_dir/"rolling_stats.png"; plt.savefig(out,dpi=300); plt.close()
    log.info(f"Saved {out}")
PY

# ---------------------------------------------------------------------
# 10. analytics/corr_extras.py  (lag heat-map & rolling corr)
# ---------------------------------------------------------------------
cat > "$PKG/analytics/corr_extras.py" << 'PY'
import numpy as np, matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import pearsonr
from matplotlib.dates import DateFormatter
from ..config import settings
from ..logging_utils import get_logger
log=get_logger(__name__)
def lag_corr_heatmap(df,max_lag=21):
    if df.empty: return
    vals=[]
    for lag in range(0,max_lag+1):
        x=df["mail_volume"]; y=df["call_volume"].shift(-lag)
        r,_=pearsonr(x.dropna(), y.dropna()); vals.append(r)
    plt.figure(figsize=(8,4)); sns.heatmap(np.array(vals).reshape(1,-1),annot=True,fmt=".2f",
        cmap="vlag",cbar=False,xticklabels=list(range(max_lag+1)),yticklabels=["r"])
    plt.title("Mail→Call correlation vs lag"); plt.xlabel("Lag (days)")
    out=settings.output_dir/"lag_corr_heatmap.png"; plt.tight_layout()
    plt.savefig(out,dpi=300); plt.close(); log.info(f"Saved {out}")
def rolling_corr(df,window=30):
    if df.empty or len(df)<window+5: return
    r=(df.set_index("date")["mail_volume"]
         .rolling(window).corr(df.set_index("date")["call_volume"]))
    plt.figure(figsize=(10,4)); plt.plot(r.index,r); plt.grid(ls="--",alpha=.4)
    plt.title(f"{window}-day rolling Mail↔Call correlation")
    ax=plt.gca(); ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(),rotation=45,ha="right"); plt.tight_layout()
    out=settings.output_dir/"rolling_corr.png"; plt.savefig(out,dpi=300); plt.close()
    log.info(f"Saved {out}")
PY

# ---------------------------------------------------------------------
# 11. analytics/mail_intent_corr.py  (top-10 mail-type × intent)
# ---------------------------------------------------------------------
cat > "$PKG/analytics/mail_intent_corr.py" << 'PY'
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import pearsonr
from ..data.loader import load_call_data, load_mail
from ..config import settings
from ..logging_utils import get_logger
log=get_logger(__name__)
def top10_mail_intent_corr():
    call_vol,intents=load_call_data(); mail=load_mail()
    mail=mail[mail["date"].dt.weekday<5]; intents=intents[intents["date"].dt.weekday<5]
    if intents.empty or mail.empty: return
    mail_piv=mail.pivot_table(index="date",columns="mail_type",values="mail_volume",aggfunc="sum").fillna(0)
    merged=pd.merge(mail_piv.reset_index(), intents, on="date", how="inner").set_index("date")
    if merged.empty: return
    results=[]
    mail_cols=[c for c in mail_piv.columns if c in merged.columns]
    intent_cols=[c for c in intents.columns if c!="date" and c in merged.columns]
    for m in mail_cols:
        for i in intent_cols:
            if merged[m].std()==0 or merged[i].std()==0: continue
            r,_=pearsonr(merged[m],merged[i]); results.append((m,i,abs(r),r))
    if not results: return
    top=sorted(results,key=lambda x:x[2],reverse=True)[:10]
    heat=pd.DataFrame(top,columns=["mail_type","intent","abs_r","r"]).set_index(["mail_type","intent"])
    plt.figure(figsize=(8,6)); sns.heatmap(heat[["r"]],annot=True,cmap="vlag",center=0,fmt=".2f")
    plt.title("Top-10 |r| Mail-Type × Call-Intent (weekdays)"); plt.tight_layout()
    out=settings.output_dir/"mailtype_intent_corr.png"; plt.savefig(out,dpi=300); plt.close()
    log.info(f"Saved {out}")
PY

# ---------------------------------------------------------------------
# 12. analytics/advanced.py  (coverage, ratio_anomaly, seasonality)
# ---------------------------------------------------------------------
cat > "$PKG/analytics/advanced.py" << 'PY'
import json, numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from ..config import settings
from ..logging_utils import get_logger
log=get_logger(__name__)
def coverage_report(call_df, mail_df):
    call_dates=set(call_df["date"]); mail_dates=set(mail_df["date"]); both=call_dates&mail_dates
    rep={"call_dates":len(call_dates),"mail_dates":len(mail_dates),"intersection":len(both)}
    json.dump(rep,open(settings.output_dir/"call_coverage.json","w",encoding="utf-8"),indent=2)
    log.info("Saved call_coverage.json")
def ratio_anomaly(df):
    if df.empty: return
    df=df.copy(); df["ratio"]=np.where(df["mail_volume"]>0,df["call_volume"]/df["mail_volume"],np.nan)
    feat=df[["call_volume","mail_volume","ratio"]].fillna(0)
    iso=IsolationForest(contamination=0.05,random_state=42)
    df["anom"]=iso.fit_predict(feat)==-1
    plt.figure(figsize=(10,6))
    plt.scatter(df["mail_volume"],df["call_volume"],c=np.where(df["anom"],"red","grey"),alpha=.6)
    plt.title("Call vs Mail (IsolationForest anomalies)"); plt.xlabel("Mail"); plt.ylabel("Call")
    plt.grid(ls="--",alpha=.3); plt.tight_layout()
    out=settings.output_dir/"ratio_anomaly.png"; plt.savefig(out,dpi=300); plt.close()
    log.info(f"Saved {out}")
def call_seasonality(call_df):
    if call_df.empty: return
    call_df=call_df.copy(); call_df["month"]=call_df["date"].dt.strftime("%b"); call_df["dow"]=call_df["date"].dt.day_name()
    piv=call_df.pivot_table(index="dow",columns="month",values="call_volume",aggfunc="mean").reindex(index=["Monday","Tuesday","Wednesday","Thursday","Friday"])
    plt.figure(figsize=(10,6)); sns.heatmap(piv,cmap="Reds",annot=True,fmt=".0f")
    plt.title("Average call volume – Month × Weekday"); plt.tight_layout()
    out=settings.output_dir/"call_seasonality.png"; plt.savefig(out,dpi=300); plt.close()
    log.info(f"Saved {out}")
PY

# ---------------------------------------------------------------------
# 13. pipeline3.py
# ---------------------------------------------------------------------
cat > "$PKG/pipeline3.py" << 'PY'
from .processing.combine            import build_dataset
from .data.loader                   import load_call_data, load_mail
from .viz.overview                  import save_overview
from .viz.raw_calls                 import plot_raw_call_files
from .viz.data_gaps                 import plot_data_gaps
from .analytics.eda                 import validate, corr_heat, lag_scan, rolling
from .analytics.corr_extras         import lag_corr_heatmap, rolling_corr
from .analytics.mail_intent_corr    import top10_mail_intent_corr
from .analytics.advanced            import coverage_report, ratio_anomaly, call_seasonality
from .logging_utils                 import get_logger
log=get_logger(__name__)
def main():
    call_vol,_=load_call_data(); mail_df=load_mail()
    coverage_report(call_vol, mail_df)
    df=build_dataset()
    if df.empty: log.error("Dataset build failed"); return
    save_overview(df); validate(df); corr_heat(df); lag_scan(df); rolling(df)
    plot_raw_call_files(); ratio_anomaly(df); call_seasonality(call_vol)
    lag_corr_heatmap(df); rolling_corr(df); top10_mail_intent_corr(); plot_data_gaps()
    log.info("All artefacts generated – see ./output/")
if __name__=="__main__": main()
PY

# ---------------------------------------------------------------------
# 14.  Run the pipeline once
# ---------------------------------------------------------------------
echo "🚀  Running weekday-only pipeline3 …"
python -m ${PKG}.pipeline3 || echo "⚠️  Pipeline failed"

echo ""
echo "✅  Done – 13 artefacts in ./output/, log in ./logs/"