#!/usr/bin/env bash
# ================================================================
#  analytics_add-ons.sh  – extend existing customer_comms package
#      • UTF-8 logging patch (no UnicodeEncodeError on Windows)
#      • Weekday-only filter in build_dataset()
#      • Adds 3 correlation plots + data-gaps plot
#      • Regenerates all previous artefacts
# ================================================================
set -euo pipefail

PKG="customer_comms"

# ----------------------------------------------------------------
# 0. Extra dependencies (sklearn already? install if missing)
# ----------------------------------------------------------------
python - <<'PY'
import importlib, subprocess, sys
for lib in ("scikit-learn","seaborn"):
    try:
        importlib.import_module(lib.replace("-","_"))
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable,"-m","pip","install","-q",lib])
PY

# ----------------------------------------------------------------
# 1. Patch logging_utils.py  (UTF-8 console & file)
# ----------------------------------------------------------------
cat > "$PKG/logging_utils.py" << 'PY'
import logging, sys
from datetime import datetime
from .config import settings

def get_logger(name="customer_comms") -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log

    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    dt  = "%Y-%m-%d %H:%M:%S"
    log.setLevel(logging.INFO)

    # Console handler (UTF-8)
    con = logging.StreamHandler(sys.stdout)
    con.setFormatter(logging.Formatter(fmt, dt))
    try:
        con.stream.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    log.addHandler(con)

    # File handler (UTF-8)
    settings.log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(settings.log_dir /
                             f"{name}_{datetime.now():%Y%m%d}.log",
                             encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt, dt))
    log.addHandler(fh)
    return log
PY

# ----------------------------------------------------------------
# 2. Replace processing/combine.py  (drop weekends)
# ----------------------------------------------------------------
cat > "$PKG/processing/combine.py" << 'PY'
import pandas as pd, numpy as np
from ..data.loader import load_call_data, load_mail
from ..config import settings
from ..logging_utils import get_logger
log = get_logger(__name__)

_norm = lambda s: (s-s.min())/(s.max()-s.min())*100 if s.max()!=s.min() else s*0

def build_dataset():
    call_vol, _ = load_call_data()
    mail_df     = load_mail()

    # --- keep weekdays only ------------------------------------
    call_vol = call_vol[call_vol["date"].dt.weekday < 5]
    mail_df  = mail_df [mail_df ["date"].dt.weekday < 5]

    if call_vol.empty or mail_df.empty:
        log.error("Missing weekday call or mail data"); return pd.DataFrame()

    mail_daily = mail_df.groupby("date")["mail_volume"].sum().reset_index()
    df = pd.merge(call_vol, mail_daily, on="date", how="inner")

    if len(df) < settings.min_rows:
        log.error("Too few intersecting weekdays"); return pd.DataFrame()

    df["call_norm"] = _norm(df["call_volume"])
    df["mail_norm"] = _norm(df["mail_volume"])
    return df.sort_values("date")
PY

# ----------------------------------------------------------------
# 3. New correlation & gap modules
# ----------------------------------------------------------------
mkdir -p "$PKG/analytics" "$PKG/viz"

# 3a. analytics/corr_extras.py  (lag heat-map, rolling corr)
cat > "$PKG/analytics/corr_extras.py" << 'PY'
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import pearsonr
from ..config import settings
from matplotlib.dates import DateFormatter
from ..logging_utils import get_logger
log=get_logger(__name__)

def lag_corr_heatmap(df: pd.DataFrame, max_lag:int=21):
    if df.empty: return
    lags = range(0,max_lag+1)
    corrs=[]
    for lag in lags:
        v=df[["mail_volume","call_volume"]].dropna()
        r,_=pearsonr(v["mail_volume"], v["call_volume"].shift(-lag).dropna())
        corrs.append(r)
    plt.figure(figsize=(8,4))
    sns.heatmap(np.array(corrs).reshape(1,-1), annot=True, fmt=".2f",
                cmap="vlag", cbar=False, xticklabels=[f"{l}" for l in lags],
                yticklabels=["r"])
    plt.title("Mail → Call correlation across lags"); plt.xlabel("Lag (days)")
    out=settings.output_dir/"lag_corr_heatmap.png"; plt.tight_layout()
    plt.savefig(out,dpi=300); plt.close(); log.info(f"Saved {out}")

def rolling_corr(df: pd.DataFrame, window:int=30):
    if df.empty or len(df)<window+5: return
    df=df.set_index("date")
    r=df["mail_volume"].rolling(window).corr(df["call_volume"])
    plt.figure(figsize=(10,4)); plt.plot(r.index,r); plt.grid(ls="--",alpha=.4)
    plt.title(f"{window}-day rolling Mail↔Call correlation (lag 0)")
    ax=plt.gca(); ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    out=settings.output_dir/"rolling_corr.png"; plt.tight_layout()
    plt.savefig(out,dpi=300); plt.close(); log.info(f"Saved {out}")
PY

# 3b. analytics/mail_intent_corr.py  (top-10 mail-type × intent)
cat > "$PKG/analytics/mail_intent_corr.py" << 'PY'
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import pearsonr
from ..data.loader import load_call_data, load_mail
from ..config import settings
from ..logging_utils import get_logger
log=get_logger(__name__)

def top10_mail_intent_corr():
    call_vol, intents = load_call_data()
    mail_df           = load_mail()

    # keep weekdays
    mail_df  = mail_df [mail_df ["date"].dt.weekday < 5]
    intents  = intents[intents["date"].dt.weekday < 5]

    if intents.empty or mail_df.empty:
        log.warning("No intent or mail data"); return

    # daily mail volume by type
    mail_piv = mail_df.pivot_table(index="date",
                                   columns="mail_type",
                                   values="mail_volume",
                                   aggfunc="sum").fillna(0)

    # align dates where both present
    merged = pd.merge(mail_piv.reset_index(),
                      intents,
                      on="date",
                      how="inner").set_index("date")

    if merged.empty:
        log.warning("No overlapping dates for mail-type & intent"); return

    mail_cols   = [c for c in merged.columns if c in mail_piv.columns]
    intent_cols = [c for c in merged.columns if c in intents.columns]

    results=[]
    for m in mail_cols:
        for i in intent_cols:
            if merged[m].std()==0 or merged[i].std()==0: continue
            r,_=pearsonr(merged[m], merged[i])
            results.append((m,i,r,abs(r)))

    if not results: return
    top10=sorted(results, key=lambda x:x[3], reverse=True)[:10]
    heat=pd.DataFrame(top10, columns=["mail_type","intent","r","abs"]).set_index(["mail_type","intent"])
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(heat[["r"]], annot=True, cmap="vlag", center=0, fmt=".2f", ax=ax)
    plt.title("Top-10 |r| Mail-Type × Call-Intent correlations")
    out=settings.output_dir/"mailtype_intent_corr.png"
    plt.tight_layout(); plt.savefig(out,dpi=300); plt.close()
    log.info(f"Saved {out}")
PY

# 3c. viz/data_gaps.py  (weekday coverage)
cat > "$PKG/viz/data_gaps.py" << 'PY'
import pandas as pd, matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from ..data.loader import load_call_data, load_mail
from ..config import settings
from ..logging_utils import get_logger
log=get_logger(__name__)
def plot_data_gaps():
    calls,_ = load_call_data(); mail = load_mail()
    calls=calls[calls["date"].dt.weekday<5]; mail=mail[mail["date"].dt.weekday<5]
    if calls.empty and mail.empty: return
    start=min(calls["date"].min() if not calls.empty else mail["date"].min(),
              mail["date"].min() if not mail.empty else calls["date"].min())
    end=max(calls["date"].max(), mail["date"].max())
    calendar=pd.DataFrame({"date":pd.bdate_range(start,end,freq="B")})
    calendar["call"]=calendar["date"].isin(set(calls["date"]))
    calendar["mail"]=calendar["date"].isin(set(mail["date"]))
    calendar["status"]=calendar.apply(lambda r:
        "Both" if r.call and r.mail else ("Call only" if r.call else ("Mail only" if r.mail else "None")), axis=1)

    colors={"Both":"green","Call only":"red","Mail only":"blue","None":"grey"}
    plt.figure(figsize=(14,2))
    for status, sub in calendar.groupby("status"):
        plt.scatter(sub["date"], [0]*len(sub), label=status, c=colors[status], s=10)
    plt.yticks([]); plt.title("Weekday data coverage"); plt.legend(loc="upper center", ncol=4)
    ax=plt.gca(); ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    out=settings.output_dir/"data_gaps.png"; plt.tight_layout()
    plt.savefig(out,dpi=300); plt.close(); log.info(f"Saved {out}")
PY

# ----------------------------------------------------------------
# 4. New pipeline3.py – everything in one run
# ----------------------------------------------------------------
cat > "$PKG/pipeline3.py" << 'PY'
from .processing.combine    import build_dataset
from .data.loader           import load_call_data, load_mail
from .viz.overview          import save_overview
from .viz.raw_calls         import plot_raw_call_files
from .viz.data_gaps         import plot_data_gaps
from .analytics.eda         import validate, corr_heat, lag_scan, rolling
from .analytics.corr_extras import lag_corr_heatmap, rolling_corr
from .analytics.mail_intent_corr import top10_mail_intent_corr
from .analytics.advanced    import coverage_report, ratio_anomaly, call_seasonality
from .logging_utils         import get_logger
log=get_logger(__name__)

def main():
    call_vol,_ = load_call_data(); mail_df = load_mail()
    coverage_report(call_vol, mail_df)

    df = build_dataset()
    if df.empty:
        log.error("Dataset build failed"); return

    # Stage 0-4
    save_overview(df); validate(df); corr_heat(df); lag_scan(df); rolling(df)

    # Original extras
    plot_raw_call_files(); ratio_anomaly(df); call_seasonality(call_vol)

    # New correlations & gaps
    lag_corr_heatmap(df); rolling_corr(df); top10_mail_intent_corr(); plot_data_gaps()

    log.info("All artefacts generated – see ./output/")

if __name__=="__main__":
    main()
PY

# ----------------------------------------------------------------
# 5. Run pipeline3
# ----------------------------------------------------------------
echo "🚀  Running pipeline3 (weekday-only)…"
python -m ${PKG}.pipeline3 || echo "⚠️  Pipeline failed"

echo "✅  Done – outputs in ./output/, logs in ./logs/"