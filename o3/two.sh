#!/usr/bin/env bash
# =====================================================================
#  part2_advanced_eda.sh              (Customer-Comms  –  Stage 2 of 4)
# =====================================================================
set -euo pipefail
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

PKG="customer_comms"
[[ -d $PKG ]] || { echo "❌  Run part1 first"; exit 1; }

# ---------------------------------------------------------------------
# 0.  (No new pip installs – deps already present in Part 1)
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 1.  analytics/eda_ts.py  – Story 2.1
# ---------------------------------------------------------------------
cat > "$PKG/analytics/eda_ts.py" << 'PY'
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import pearsonr
from matplotlib.dates import DateFormatter
from ..config import settings
from ..logging_utils import get_logger
log=get_logger(__name__)

def decompose(df: pd.DataFrame):
    if df.empty or len(df) < 60:
        log.warning("Too few rows for decomposition"); return
    series = df.set_index("date")["call_volume_aug"].dropna()
    try:
        res = seasonal_decompose(series, model="additive", period=7)
        plt.figure(figsize=(12,8))
        res.plot(); plt.tight_layout()
        out = settings.out_dir/"04_ts_decomposition.png"
        plt.savefig(out,dpi=300); plt.close()
        log.info(f"Saved {out}")
    except Exception as e:
        log.error(f"Decomposition failed: {e}")

def lag_heatmap(df: pd.DataFrame, max_lag:int=14):
    vals=[]
    for lag in range(0,max_lag+1):
        v=df.dropna(subset=["mail_volume","call_volume_aug"])
        r,_=pearsonr(v["mail_volume"], v["call_volume_aug"].shift(-lag).dropna())
        vals.append(r)
    plt.figure(figsize=(8,3))
    sns.heatmap(np.array(vals).reshape(1,-1),annot=True,fmt=".2f",
                cmap="vlag",center=0,cbar=False,
                xticklabels=list(range(max_lag+1)),yticklabels=["r"])
    plt.title("Mail→Call Pearson r by Lag (days)")
    out=settings.out_dir/"05_lag_corr_heatmap.png"
    plt.tight_layout(); plt.savefig(out,dpi=300); plt.close()
    log.info(f"Saved {out}")

def weekday_month_heat(df: pd.DataFrame):
    piv=(df.assign(month=df["date"].dt.strftime("%b"),
                   dow=df["date"].dt.day_name())
            .pivot_table(index="dow",columns="month",
                         values="call_volume_aug",aggfunc="mean"))
    piv=piv.reindex(index=["Monday","Tuesday","Wednesday","Thursday","Friday"])
    plt.figure(figsize=(10,6)); sns.heatmap(piv,annot=True,fmt=".0f",cmap="Reds")
    plt.title("Avg Call Vol – Month × Weekday")
    out=settings.out_dir/"06_seasonality_heat.png"
    plt.tight_layout(); plt.savefig(out,dpi=300); plt.close()
    log.info(f"Saved {out}")
PY

# ---------------------------------------------------------------------
# 2.  analytics/mail_effectiveness.py  – Story 2.2
# ---------------------------------------------------------------------
cat > "$PKG/analytics/mail_effectiveness.py" << 'PY'
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from ..config import settings
from ..logging_utils import get_logger
log=get_logger(__name__)

def analyse(mail:pd.DataFrame, call_aug:pd.DataFrame):
    if mail.empty or call_aug.empty: return
    daily_calls=call_aug[["date","call_volume_aug"]]
    merged=mail.merge(daily_calls,on="date",how="inner")
    resp=(merged.groupby("mail_type")
                 .apply(lambda g:(g["call_volume_aug"].sum()/g["mail_volume"].sum())*1000)
                 .rename("calls_per_1k_mail")
                 .sort_values(ascending=False))
    resp.to_csv(settings.out_dir/"07_mail_response_rates.csv")
    plt.figure(figsize=(10,6)); sns.barplot(y=resp.index,x=resp.values,color="tab:blue")
    plt.xlabel("Calls per 1 000 Mail"); plt.ylabel("Mail Type")
    plt.title("Mail-Campaign Effectiveness")
    out=settings.out_dir/"07_mail_effectiveness.png"
    plt.tight_layout(); plt.savefig(out,dpi=300); plt.close()
    log.info(f"Saved {out} & CSV")
PY

# ---------------------------------------------------------------------
# 3.  analytics/econ_sensitivity.py  – Story 2.3
# ---------------------------------------------------------------------
cat > "$PKG/analytics/econ_sensitivity.py" << 'PY'
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from scipy.stats import pearsonr
from ..config import settings
from ..logging_utils import get_logger
log=get_logger(__name__)

def corr_heat(call:pd.DataFrame, econ:pd.DataFrame):
    if call.empty or econ.empty: return
    df=call.merge(econ,on="date",how="inner").dropna()
    if df.empty: return
    cols=[c for c in econ.columns if c!="date"]
    res=[pearsonr(df[c],df["call_volume_aug"])[0] for c in cols]
    sns.set(font_scale=1.0)
    plt.figure(figsize=(8,2)); sns.heatmap([res],annot=True,fmt=".2f",
            cmap="vlag",center=0,yticklabels=["r"],xticklabels=cols)
    plt.title("Call vs Econ Indicator Correlations")
    out=settings.out_dir/"08_econ_corr.png"
    plt.tight_layout(); plt.savefig(out,dpi=300); plt.close()
    log.info(f"Saved {out}")

def add_regime(econ:pd.DataFrame)->pd.DataFrame:
    if "sp500" not in econ.columns: return econ
    econ=econ.copy()
    econ["sp500_ret"]=econ["sp500"].pct_change()*100
    econ["market_regime"]=pd.cut(econ["sp500_ret"],
                                 bins=[-1e9,-0.5,0.5,1e9],
                                 labels=["bear","sideways","bull"])
    return econ
PY

# ---------------------------------------------------------------------
# 4.  analytics/patterns.py  – Story 2.4
# ---------------------------------------------------------------------
cat > "$PKG/analytics/patterns.py" << 'PY'
import numpy as np, matplotlib.pyplot as plt, seaborn as sns, pandas as pd
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.stattools import grangercausalitytests
from ..config import settings
from ..logging_utils import get_logger
log=get_logger(__name__)

def anomaly_scatter(df:pd.DataFrame):
    if df.empty: return
    X=df[["mail_volume","call_volume_aug"]].fillna(0)
    iso=IsolationForest(random_state=42,contamination=0.05)
    df["anomaly"]=iso.fit_predict(X)==-1
    plt.figure(figsize=(8,6))
    plt.scatter(df["mail_volume"],df["call_volume_aug"],
                c=df["anomaly"].map({True:"red",False:"grey"}),
                alpha=.6)
    plt.xlabel("Mail Volume"); plt.ylabel("Call Vol (aug)")
    plt.title("Isolation-Forest Anomalies")
    out=settings.out_dir/"09_anomaly_scatter.png"
    plt.tight_layout(); plt.savefig(out,dpi=300); plt.close()
    log.info(f"Saved {out}")

def granger(df:pd.DataFrame,maxlag:int=5):
    if df.empty: return
    gdf=df[["mail_volume","call_volume_aug"]].dropna()
    result=grangercausalitytests(gdf,maxlag=maxlag,verbose=False)
    pvals=[round(result[i+1][0]["ssr_ftest"][1],4) for i in range(maxlag)]
    with open(settings.out_dir/"09_granger_pvalues.txt","w") as f:
        f.write("lag,p-value\n"+ "\n".join(f"{i+1},{p}" for i,p in enumerate(pvals)))
    log.info("Granger causality p-values saved")
PY

# ---------------------------------------------------------------------
# 5.  Update Makefile (append new targets)
# ---------------------------------------------------------------------
awk '/^clean:/ {exit} {print}' Makefile > Makefile.tmp || true
cat Makefile.tmp > Makefile && rm -f Makefile.tmp

cat >> Makefile << 'MK'

eda_ts:        ## Story 2.1  – decomposition, lag, seasonality
	python - <<'PY'
import customer_comms.processing.ingest as ing
from customer_comms.analytics.augment import augment
from customer_comms.analytics.eda_ts import decompose, lag_heatmap, weekday_month_heat
call, mail = ing.run(); df=augment(call).merge(mail,on="date",how="inner")
decompose(df); lag_heatmap(df); weekday_month_heat(df)
PY

mail_eff:      ## Story 2.2  – campaign effectiveness
	python - <<'PY'
import customer_comms.processing.ingest as ing
from customer_comms.analytics.augment import augment
from customer_comms.analytics.mail_effectiveness import analyse
call, mail = ing.run(); aug=augment(call); analyse(mail, aug)
PY

econ_sens:     ## Story 2.3  – economic sensitivity
	python - <<'PY'
import customer_comms.processing.ingest as ing
from customer_comms.features.econ import fetch
from customer_comms.analytics.augment import augment
from customer_comms.analytics.econ_sensitivity import corr_heat, add_regime
call, _ = ing.run(); aug=augment(call); econ=fetch()
econ=add_regime(econ); corr_heat(aug,econ)
PY

patterns:      ## Story 2.4  – anomalies & causality
	python - <<'PY'
import customer_comms.processing.ingest as ing
from customer_comms.analytics.augment import augment
from customer_comms.analytics.patterns import anomaly_scatter, granger
call, mail = ing.run(); df=augment(call).merge(mail,on="date",how="inner")
anomaly_scatter(df); granger(df)
PY
MK

echo "✅  Stage 2 added.  Try:"
echo "   make eda_ts    # decomposition + lag & seasonality plots"
echo "   make mail_eff  # campaign effectiveness"
echo "   make econ_sens # economic correlations"
echo "   make patterns  # anomalies & Granger test"
echo "   Artefacts land in ./output/, logs in ./logs/"
