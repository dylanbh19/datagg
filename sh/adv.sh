#!/usr/bin/env bash
# ================================================================
#  analytics_add-ons_fix.sh
#    • Idempotent patch: UTF-8 logging + weekday filter
#    • Adds/overwrites missing sub-package __init__.py
#    • Re-creates correlation & gap modules
#    • Runs pipeline3 and regenerates 12+ artefacts
# ================================================================
set -euo pipefail

PKG="customer_comms"

echo "🔧  Patching existing project …"

# ----------------------------------------------------------------
# 0.  Dependencies (quiet install if missing)
# ----------------------------------------------------------------
python - <<'PY'
import importlib, subprocess, sys
for lib in ("scikit-learn","seaborn"):
    try: importlib.import_module(lib.replace("-","_"))
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable,"-m","pip","install","-q",lib])
PY

# ----------------------------------------------------------------
# 1.  Guarantee sub-package init files
# ----------------------------------------------------------------
for sub in analytics viz; do
  mkdir -p "$PKG/$sub"
  touch   "$PKG/$sub/__init__.py"
done

# ----------------------------------------------------------------
# 2.  UTF-8 logger  (overwrite logging_utils.py)
# ----------------------------------------------------------------
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
    try:
        con.stream.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    log.addHandler(con)

    settings.log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(settings.log_dir /
                             f"{name}_{datetime.now():%Y%m%d}.log",
                             encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt, dt))
    log.addHandler(fh)
    return log
PY

# ----------------------------------------------------------------
# 3.  Weekday-filtered combine module
# ----------------------------------------------------------------
cat > "$PKG/processing/combine.py" << 'PY'
import pandas as pd, numpy as np
from ..data.loader import load_call_data, load_mail
from ..config import settings
from ..logging_utils import get_logger
log = get_logger(__name__)
_norm = lambda s:(s-s.min())/(s.max()-s.min())*100 if s.max()!=s.min() else s*0
def build_dataset():
    call_vol,_ = load_call_data()
    mail       = load_mail()
    call_vol = call_vol[call_vol["date"].dt.weekday<5]
    mail      = mail     [mail["date"].dt.weekday<5]
    if call_vol.empty or mail.empty:
        log.error("No weekday call or mail data"); return pd.DataFrame()
    mail_daily = mail.groupby("date")["mail_volume"].sum().reset_index()
    df = pd.merge(call_vol, mail_daily, on="date", how="inner")
    if len(df) < settings.min_rows:
        log.error("Too few intersecting weekdays"); return pd.DataFrame()
    df["call_norm"]=_norm(df["call_volume"]); df["mail_norm"]=_norm(df["mail_volume"])
    return df.sort_values("date")
PY

# ----------------------------------------------------------------
# 4.  Raw call lines  (viz/raw_calls.py)
# ----------------------------------------------------------------
cat > "$PKG/viz/raw_calls.py" << 'PY'
import matplotlib.pyplot as plt, pandas as pd
from matplotlib.dates import DateFormatter
from ..config import settings
from ..logging_utils import get_logger
log=get_logger(__name__)
def plot_raw_call_files():
    paths=[settings.data_dir/f for f in settings.call_files]
    series=[]
    for p in paths:
        if not p.exists(): continue
        df=pd.read_csv(p,low_memory=False)
        date_col=next((c for c in settings.call_date_cols if c in df.columns),None)
        if not date_col: continue
        df[date_col]=pd.to_datetime(df[date_col],errors="coerce").dt.normalize()
        grp=(df.groupby(date_col).size().reset_index(name="vol")
               .rename(columns={date_col:"date"}))
        grp=grp[grp["date"].dt.weekday<5]  # weekdays only
        grp["file"]=p.name; series.append(grp)
    if not series: return
    full=pd.concat(series)
    plt.figure(figsize=(14,6))
    for f, sub in full.groupby("file"):
        plt.plot(sub["date"],sub["vol"],label=f)
    plt.title("Raw daily call volume (weekdays)"); plt.xlabel("Date"); plt.ylabel("Calls")
    plt.legend(); plt.grid(ls="--",alpha=.3)
    ax=plt.gca(); ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(),rotation=45,ha="right"); plt.tight_layout()
    out=settings.output_dir/"raw_call_files.png"; settings.output_dir.mkdir(exist_ok=True)
    plt.savefig(out,dpi=300); plt.close(); log.info(f"Saved {out}")
PY

# ----------------------------------------------------------------
# 5.  Correlation extras & intent-mail correlation (unchanged)
# ----------------------------------------------------------------
cat > "$PKG/analytics/corr_extras.py" << 'PY'
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import pearsonr
from matplotlib.dates import DateFormatter
from ..config import settings
from ..logging_utils import get_logger
log=get_logger(__name__)
def lag_corr_heatmap(df,max_lag=21):
    if df.empty: return
    lags=range(0,max_lag+1); vals=[]
    for lag in lags:
        v=df[["mail_volume","call_volume"]].dropna()
        r,_=pearsonr(v["mail_volume"],v["call_volume"].shift(-lag).dropna())
        vals.append(r)
    plt.figure(figsize=(8,4)); sns.heatmap(np.array(vals).reshape(1,-1),annot=True,fmt=".2f",
               cmap="vlag",cbar=False,xticklabels=lags,yticklabels=["r"])
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
    plt.setp(ax.get_xticklabels(),rotation=45,ha="right")
    out=settings.output_dir/"rolling_corr.png"; plt.tight_layout()
    plt.savefig(out,dpi=300); plt.close(); log.info(f"Saved {out}")
PY

cat > "$PKG/analytics/mail_intent_corr.py" << 'PY'
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
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
    mail_cols=[c for c in mail_piv.columns if c in merged.columns]
    intent_cols=[c for c in intents.columns if c in merged.columns and c!="date"]
    res=[]
    for m in mail_cols:
        for i in intent_cols:
            if merged[m].std()==0 or merged[i].std()==0: continue
            r,_=pearsonr(merged[m],merged[i]); res.append((m,i,abs(r),r))
    if not res: return
    top=pd.DataFrame(sorted(res,key=lambda x:x[2],reverse=True)[:10],
                     columns=["mail_type","intent","abs_r","r"]).set_index(["mail_type","intent"])
    plt.figure(figsize=(8,6)); sns.heatmap(top[["r"]],annot=True,cmap="vlag",center=0,fmt=".2f")
    plt.title("Top-10 |r| Mail-Type × Call-Intent"); plt.tight_layout()
    out=settings.output_dir/"mailtype_intent_corr.png"; plt.savefig(out,dpi=300); plt.close()
    log.info(f"Saved {out}")
PY

# ----------------------------------------------------------------
# 6.  Data-gaps visual
# ----------------------------------------------------------------
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
    cal["call"]=cal["date"].isin(set(calls["date"])); cal["mail"]=cal["date"].isin(set(mail["date"]))
    def _status(r): return "Both" if r.call and r.mail else ("Call only" if r.call else ("Mail only" if r.mail else "None"))
    cal["status"]=cal.apply(_status,axis=1)
    color={"Both":"green","Call only":"red","Mail only":"blue","None":"grey"}
    plt.figure(figsize=(14,2))
    for s, sub in cal.groupby("status"):
        plt.scatter(sub["date"], [0]*len(sub), c=color[s], s=10, label=s)
    plt.legend(ncol=4,loc="upper center"); plt.yticks([]); plt.title("Weekday data coverage")
    ax=plt.gca(); ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(),rotation=45,ha="right"); plt.tight_layout()
    out=settings.output_dir/"data_gaps.png"; settings.output_dir.mkdir(exist_ok=True)
    plt.savefig(out,dpi=300); plt.close(); log.info(f"Saved {out}")
PY

# ----------------------------------------------------------------
# 7.  pipeline3.py  (overwrites)
# ----------------------------------------------------------------
cat > "$PKG/pipeline3.py" << 'PY'
from .processing.combine          import build_dataset
from .data.loader                 import load_call_data, load_mail
from .viz.overview                import save_overview
from .viz.raw_calls               import plot_raw_call_files
from .viz.data_gaps               import plot_data_gaps
from .analytics.eda               import validate, corr_heat, lag_scan, rolling
from .analytics.corr_extras       import lag_corr_heatmap, rolling_corr
from .analytics.mail_intent_corr  import top10_mail_intent_corr
from .analytics.advanced          import coverage_report, ratio_anomaly, call_seasonality
from .logging_utils               import get_logger
log=get_logger(__name__)
def main():
    call_vol,_=load_call_data(); mail_df=load_mail()
    coverage_report(call_vol, mail_df)
    df=build_dataset()
    if df.empty: log.error("Dataset build failed"); return
    save_overview(df); validate(df); corr_heat(df); lag_scan(df); rolling(df)
    plot_raw_call_files(); ratio_anomaly(df); call_seasonality(call_vol)
    lag_corr_heatmap(df); rolling_corr(df); top10_mail_intent_corr(); plot_data_gaps()
    log.info("All artefacts generated (weekday-only) – see ./output/")
if __name__=="__main__": main()
PY

# ----------------------------------------------------------------
# 8.  Run pipeline3
# ----------------------------------------------------------------
echo "🚀  Running pipeline3 (weekday-only)…"
python -m ${PKG}.pipeline3 || echo "⚠️  Pipeline failed"

echo "✅  Finished – outputs in ./output/, logs in ./logs/"