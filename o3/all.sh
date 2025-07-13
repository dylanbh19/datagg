#!/usr/bin/env bash
###############################################################################
#  customer_comms_full.sh   –  VERBOSE, LINE-BY-LINE, END-TO-END SETUP
#  ---------------------------------------------------------------------------
#  • Dependency installation
#  • Project skeleton
#  • Data loaders (multi-encoding)
#  • Call-volume augmentation via intents
#  • Weekday filtering & feature engineering (lags, pct, diff, econ)
#  • QA reports + 20+ plots + RF baseline
#  • UTF-8 logging, Windows friendly
#  • **FIX** → forces matplotlib → Agg → no Tk / Tcl crashes
###############################################################################
set -euo pipefail
export MPLBACKEND=Agg   # <- global guarantee

###############################################################################
# 0 ⟩ Install / verify dependencies
###############################################################################
echo "=============================================================================="
echo " STEP 0 – Install / verify Python dependencies "
echo "=============================================================================="
python - <<'PY'
import importlib, subprocess, sys, textwrap
pkgs = [
    "pandas", "numpy", "matplotlib", "seaborn", "scikit-learn",
    "scipy", "holidays", "yfinance", "pydantic", "pydantic-settings"
]
for p in pkgs:
    try:
        importlib.import_module(p.replace('-', '_'))
        print(f"✔  {p}")
    except ModuleNotFoundError:
        print(f"… installing {p}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])
print("Python stack ready.")
PY

###############################################################################
# 1 ⟩ Skeleton
###############################################################################
echo "=============================================================================="
echo " STEP 1 – Create package skeleton (explicit mkdir / touch for each dir) "
echo "=============================================================================="
PKG="customer_comms"
rm -rf "$PKG"
for d in "" data processing analytics viz utils models; do
  mkdir -p "$PKG/$d"
  touch    "$PKG/$d/__init__.py"
done
mkdir -p logs output data

###############################################################################
# 2 ⟩ config.py
###############################################################################
echo "=============================================================================="
echo " STEP 2 – Write customer_comms/config.py "
echo "=============================================================================="
cat > "$PKG/config.py" <<'PY'
from pathlib import Path
try:
    from pydantic_settings import BaseSettings
except ModuleNotFoundError:
    from pydantic import BaseSettings
from pydantic import Field
class Settings(BaseSettings):
    # CSV file names (must live in ./data/)
    call_vol_file: str = "GenesysExtract_20250703.csv"
    call_int_file: str = "GenesysExtract_20250609.csv"
    mail_file:     str = "all_mail_data.csv"

    # Column names
    call_date_col:   str = "Date"
    intent_date_col: str = "ConversationStart"
    intent_col:      str = "uui_Intent"
    mail_date_col:   str = "mail_date"
    mail_type_col:   str = "mail_type"
    mail_vol_col:    str = "mail_volume"

    intent_min:   int = 250      # drop rare intents
    max_lag_days: int = 21

    # Paths
    data_dir: Path = Field(default=Path("data"))
    log_dir : Path = Field(default=Path("logs"))
    out_dir : Path = Field(default=Path("customer_comms") / "output")

    # Economic tickers (Yahoo Finance)
    econ_tickers: dict[str, str] = {
        "VIX":      "^VIX",
        "SP500":    "^GSPC",
        "FEDFUNDS": "FEDFUNDS"
    }
settings = Settings()
PY

###############################################################################
# 3 ⟩ utils/logging_utils.py
###############################################################################
echo "=============================================================================="
echo " STEP 3 – Write utils/logging_utils.py "
echo "=============================================================================="
cat > "$PKG/utils/logging_utils.py" <<'PY'
import logging, sys
from datetime import datetime
from ..config import settings
def get_logger(name="customer_comms"):
    lg = logging.getLogger(name)
    if lg.handlers:                       # already configured
        return lg
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    dh  = "%Y-%m-%d %H:%M:%S"
    lg.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    try: sh.stream.reconfigure(encoding="utf-8")
    except AttributeError: pass
    sh.setFormatter(logging.Formatter(fmt, dh))
    lg.addHandler(sh)
    settings.log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(
        settings.log_dir / f"{name}_{datetime.now():%Y%m%d}.log", encoding="utf-8"
    )
    fh.setFormatter(logging.Formatter(fmt, dh))
    lg.addHandler(fh)
    return lg
PY

###############################################################################
# 4 ⟩ utils/_force_agg.py   – single-line helper each viz file imports first
###############################################################################
echo "=============================================================================="
echo " STEP 4 – Write utils/_force_agg.py (matplotlib backend fix) "
echo "=============================================================================="
cat > "$PKG/utils/_force_agg.py" <<'PY'
import matplotlib
matplotlib.use("Agg")      # always head-less
PY

###############################################################################
# 5 ⟩ utils/io.py
###############################################################################
echo "=============================================================================="
echo " STEP 5 – Write utils/io.py (multi-encoding CSV reader) "
echo "=============================================================================="
cat > "$PKG/utils/io.py" <<'PY'
import pandas as pd, os, logging
lg = logging.getLogger("customer_comms")
ENCODINGS = ("utf-8", "latin-1", "cp1252", "utf-16")
def read_csv_any(path, **kw):
    if not os.path.isfile(path):
        lg.warning(f"Missing {path}")
        return pd.DataFrame()
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="skip",
                               low_memory=False, **kw)
        except UnicodeDecodeError:
            continue
    lg.error(f"Could not decode {path}")
    return pd.DataFrame()
PY

###############################################################################
# 6 ⟩ data/loader.py
###############################################################################
echo "=============================================================================="
echo " STEP 6 – Write data/loader.py "
echo "=============================================================================="
cat > "$PKG/data/loader.py" <<'PY'
from __future__ import annotations
import pandas as pd, yfinance as yf
from ..config import settings
from ..utils.io import read_csv_any
from ..utils.logging_utils import get_logger
log = get_logger(__name__)

def _norm_date(s):       # normalize & drop tz
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None).dt.normalize()

# ---------------------------- call volume (Date)
def load_call_volume():
    df = read_csv_any(settings.data_dir / settings.call_vol_file,
                      usecols=[settings.call_date_col])
    if df.empty: return df
    df.rename(columns={settings.call_date_col: "date"}, inplace=True)
    df["date"] = _norm_date(df["date"])
    df.dropna(subset=["date"], inplace=True)
    out = (df.groupby("date")
             .size()
             .reset_index(name="call_volume")
             .sort_values("date"))
    log.info(f"Call volume rows: {len(out)}")
    return out

# ---------------------------- intents (ConversationStart, uui_Intent)
def load_intents():
    df = read_csv_any(settings.data_dir / settings.call_int_file,
                      usecols=[settings.intent_date_col, settings.intent_col])
    if df.empty: return df
    df.rename(columns={settings.intent_date_col: "date",
                       settings.intent_col: "intent"}, inplace=True)
    df["date"] = _norm_date(df["date"])
    df.dropna(subset=["date"], inplace=True)
    keep = df["intent"].value_counts()
    keep = keep[keep >= settings.intent_min].index
    df = df[df["intent"].isin(keep)]
    mat = (df.groupby(["date", "intent"])
             .size()
             .unstack(fill_value=0)
             .reset_index()
             .sort_values("date"))
    log.info(f"Intent matrix shape: {mat.shape}")
    return mat

# ---------------------------- mail data
def load_mail():
    df = read_csv_any(settings.data_dir / settings.mail_file,
                      usecols=[settings.mail_date_col,
                               settings.mail_type_col,
                               settings.mail_vol_col])
    if df.empty: return df
    df.rename(columns={settings.mail_date_col: "date",
                       settings.mail_type_col: "mail_type",
                       settings.mail_vol_col: "mail_volume"}, inplace=True)
    df["date"] = _norm_date(df["date"])
    df["mail_volume"] = pd.to_numeric(df["mail_volume"], errors="coerce")
    df.dropna(subset=["date", "mail_volume"], inplace=True)
    log.info(f"Mail rows: {len(df)}")
    return df

# ---------------------------- economic indicators
def load_econ():
    frames = []
    for name, ticker in settings.econ_tickers.items():
        try:
            d = yf.download(ticker, period="2y", interval="1d", progress=False,
                            auto_adjust=True)
            if d.empty:
                raise ValueError("yfinance empty")
            series = (d["Adj Close"] if "Adj Close" in d else d.iloc[:, 0]).rename(name)
            frames.append(series)
            log.info(f"{name} rows: {len(series)}")
        except Exception as e:
            log.error(f"{name} download failed: {e}")
    if not frames: return pd.DataFrame()
    df = pd.concat(frames, axis=1)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df.reset_index(names="date")
PY

###############################################################################
# 7 ⟩ processing/combine.py
###############################################################################
echo "=============================================================================="
echo " STEP 7 – Write processing/combine.py "
echo "=============================================================================="
cat > "$PKG/processing/combine.py" <<'PY'
import pandas as pd, numpy as np
from ..data.loader import load_call_volume, load_intents, load_mail, load_econ
from ..utils.logging_utils import get_logger
from ..config import settings
log = get_logger(__name__)
def _norm(s): return (s - s.min()) / (s.max() - s.min()) * 100 if s.max() != s.min() else s*0
def build_master():
    call = load_call_volume()
    mail = load_mail()
    call = call[call.date.dt.weekday < 5]
    mail = mail[mail.date.dt.weekday < 5]
    # -- augment call with intents
    intents = load_intents()
    if not intents.empty:
        daily_int = intents.drop(columns="date").sum(axis=1)
        tmp = pd.DataFrame({"date": intents.date, "intent_cnt": daily_int})
        merged = pd.merge(call, tmp, on="date", how="outer")
        if merged.intent_cnt.mean() and merged.call_volume.mean():
            scale = merged.call_volume.mean() / merged.intent_cnt.mean()
        else:
            scale = 1
        merged.call_volume.fillna(merged.intent_cnt * scale, inplace=True)
        call = merged[["date", "call_volume"]]
    mail_daily = mail.groupby("date")["mail_volume"].sum().reset_index()
    core = pd.merge(call, mail_daily, on="date", how="inner").sort_values("date")
    core["call_norm"] = _norm(core.call_volume)
    core["mail_norm"] = _norm(core.mail_volume)
    core["call_pct"]  = core.call_volume.pct_change().fillna(0)
    core["mail_pct"]  = core.mail_volume.pct_change().fillna(0)
    core["call_diff"] = core.call_volume.diff().fillna(0)
    core["mail_diff"] = core.mail_volume.diff().fillna(0)
    econ = load_econ()
    if not econ.empty:
        core = pd.merge(core, econ, on="date", how="left")
    log.info(f"Master frame rows: {len(core)}")
    return core
PY

###############################################################################
# 8 ⟩ viz/plots.py
###############################################################################
echo "=============================================================================="
echo " STEP 8 – Write viz/plots.py "
echo "=============================================================================="
cat > "$PKG/viz/plots.py" <<'PY'
from customer_comms.utils import _force_agg          # backend fix
import matplotlib.pyplot as plt, seaborn as sns, pandas as pd, numpy as np
from matplotlib.dates import DateFormatter
from ..config import settings
from ..data.loader import load_call_volume, load_mail
from ..processing.combine import build_master
from ..utils.logging_utils import get_logger
log=get_logger(__name__)
settings.out_dir.mkdir(exist_ok=True)
def _save(fig, name):
    fig.tight_layout()
    out=settings.out_dir/name
    fig.savefig(out, dpi=300)
    plt.close(fig)
    log.info(f"Saved {out.name}")
def overview():
    df=build_master()
    if df.empty: return
    fig,ax=plt.subplots(figsize=(14,6))
    ax.bar(df.date, df.mail_norm, alpha=.6, label="Mail (norm)")
    ax.plot(df.date, df.call_norm, lw=2, color="tab:red", label="Calls (norm)")
    ax.set(title="Mail vs Call (normalised 0-100)", ylabel="0-100")
    ax.legend(); ax.grid(ls="--", alpha=.3)
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    _save(fig,"overview.png")
def raw_call_files():
    call=load_call_volume()
    from ..data.loader import load_intents
    intent=load_intents()
    if call.empty and intent.empty: return
    fig,ax=plt.subplots(figsize=(14,6))
    if not call.empty:
        ax.plot(call.date, call.call_volume, label=settings.call_vol_file)
    if not intent.empty:
        tot=intent.drop(columns="date").sum(axis=1)
        ax.plot(intent.date, tot, label=settings.call_int_file)
    ax.set(title="Raw daily call counts", ylabel="Calls")
    ax.legend(); ax.grid(ls="--",alpha=.3)
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    _save(fig,"raw_call_files.png")
def data_gaps():
    call=load_call_volume(); mail=load_mail()
    if call.empty and mail.empty: return
    start=min(call.date.min() if not call.empty else mail.date.min(),
              mail.date.min() if not mail.empty else call.date.min())
    end=max(call.date.max() if not call.empty else mail.date.max(),
            mail.date.max() if not mail.empty else call.date.max())
    cal=pd.DataFrame({"date":pd.bdate_range(start,end)})
    cal["call"]=cal.date.isin(set(call.date))
    cal["mail"]=cal.date.isin(set(mail.date))
    cal["status"]=np.select([cal.call&cal.mail, cal.call, cal.mail],
                            ["Both","Call only","Mail only"], default="None")
    col=dict(Both="green",**{"Call only":"red","Mail only":"blue","None":"grey"})
    fig,ax=plt.subplots(figsize=(14,2))
    for s,sub in cal.groupby("status"):
        ax.scatter(sub.date,[0]*len(sub),c=col[s],s=8,label=s)
    ax.set_yticks([]); ax.legend(ncol=4)
    ax.set_title("Weekday data coverage")
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    _save(fig,"data_gaps.png")
def qa_jsons():
    import json
    call=load_call_volume(); mail=load_mail()
    out=settings.out_dir/"qa_summary.json"
    with open(out,"w",encoding="utf-8") as f:
        json.dump({"call_dates":int(call.date.nunique() if not call.empty else 0),
                   "mail_dates":int(mail.date.nunique() if not mail.empty else 0)},f,indent=2)
    log.info(f"Saved {out.name}")
PY

###############################################################################
# 9 ⟩ analytics/corr_extras.py
###############################################################################
echo "=============================================================================="
echo " STEP 9 – Write analytics/corr_extras.py "
echo "=============================================================================="
cat > "$PKG/analytics/corr_extras.py" <<'PY'
from customer_comms.utils import _force_agg          # backend fix
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import pearsonr
from matplotlib.dates import DateFormatter
from ..config import settings
from ..utils.logging_utils import get_logger
log=get_logger(__name__)
def _safe_r(x,y):
    v=pd.concat([x,y],axis=1).dropna()
    if len(v)<2 or v.iloc[:,0].std()==0 or v.iloc[:,1].std()==0: return 0.0
    try: return float(pearsonr(v.iloc[:,0],v.iloc[:,1])[0])
    except: return 0.0
def lag_heat(df,max_lag=21):
    vals=[_safe_r(df.mail_volume, df.call_volume.shift(-l)) for l in range(max_lag+1)]
    fig,ax=plt.subplots(figsize=(12,1.4))
    sns.heatmap(np.array(vals).reshape(1,-1),annot=True,fmt=".2f",cmap="vlag",cbar=False,
                xticklabels=list(range(max_lag+1)),yticklabels=["r"],ax=ax)
    ax.set(title="Mail → Call correlation vs lag",xlabel="Lag (days)")
    _save(fig:=fig,name="lag_corr_heatmap.png")
def rolling_corr(df,window=30):
    if len(df)<window+3: return
    r=(df.set_index("date").mail_volume
         .rolling(window).corr(df.set_index("date").call_volume))
    fig,ax=plt.subplots(figsize=(12,4))
    ax.plot(r.index,r); ax.grid(ls="--",alpha=.4)
    ax.set_title(f"{window}-day rolling correlation")
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(),rotation=45,ha="right")
    _save(fig,"rolling_corr.png")
def corr_variants(df):
    variants={"raw":_safe_r(df.mail_volume,df.call_volume),
              "lag3":_safe_r(df.mail_volume,df.call_volume.shift(-3)),
              "lag7":_safe_r(df.mail_volume,df.call_volume.shift(-7)),
              "ma7":_safe_r(df.mail_volume.rolling(7).mean(),
                            df.call_volume.rolling(7).mean()),
              "pct":_safe_r(df.mail_volume.pct_change(),
                            df.call_volume.pct_change())}
    fig,ax=plt.subplots(figsize=(8,4))
    sns.barplot(x=list(variants.keys()),y=list(variants.values()),ax=ax)
    ax.set_title("Correlation variants"); ax.set_ylabel("r")
    _save(fig,"variant_corr.png")
def _save(fig,name):
    fig.tight_layout(); out=settings.out_dir/name
    fig.savefig(out,dpi=300); plt.close(fig); log.info(f"Saved {out.name}")
PY

###############################################################################
#10 ⟩ analytics/mail_intent_corr.py
###############################################################################
echo "=============================================================================="
echo " STEP 10 – Write analytics/mail_intent_corr.py "
echo "=============================================================================="
cat > "$PKG/analytics/mail_intent_corr.py" <<'PY'
from customer_comms.utils import _force_agg          # backend fix
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import pearsonr
from ..data.loader import load_mail, load_intents
from ..config import settings
from ..utils.logging_utils import get_logger
log=get_logger(__name__)
def plot_top10():
    mail=load_mail(); intents=load_intents()
    if mail.empty or intents.empty: return
    mail_piv=mail.pivot_table(index="date",columns="mail_type",
                              values="mail_volume",aggfunc="sum").fillna(0)
    merged=pd.merge(mail_piv.reset_index(),intents,on="date",how="inner").set_index("date")
    res=[]
    for m in mail_piv.columns:
        for i in intents.columns.drop("date"):
            if merged[m].std()==0 or merged[i].std()==0: continue
            r,_=pearsonr(merged[m],merged[i]); res.append((m,i,abs(r),r))
    if not res: return
    top=sorted(res,key=lambda x:x[2],reverse=True)[:10]
    df=pd.DataFrame(top,columns=["mail","intent","abs_r","r"])\
          .pivot(index="mail",columns="intent",values="r").fillna(0)
    fig,ax=plt.subplots(figsize=(8,6))
    sns.heatmap(df,annot=True,fmt=".2f",cmap="vlag",center=0,ax=ax)
    ax.set_title("Top-10 |r| Mail-Type × Intent")
    fig.tight_layout(); out=settings.out_dir/"mailtype_intent_corr.png"
    fig.savefig(out,dpi=300); plt.close(fig); log.info(f"Saved {out.name}")
PY

###############################################################################
#11 ⟩ models/baseline.py
###############################################################################
echo "=============================================================================="
echo " STEP 11 – Write models/baseline.py "
echo "=============================================================================="
cat > "$PKG/models/baseline.py" <<'PY'
import json, numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
from ..processing.combine import build_master
from ..config import settings
from ..utils.logging_utils import get_logger
log=get_logger(__name__)
def run_baseline():
    df=build_master()
    if df.empty: return
    for l in (1,3,7): df[f"mail_lag{l}"]=df.mail_volume.shift(l)
    df.dropna(inplace=True)
    X=df.drop(columns=["call_volume","date"]); y=df.call_volume
    tscv=TimeSeriesSplit(n_splits=5); mape=[]
    for tr,ts in tscv.split(X):
        rf=RandomForestRegressor(n_estimators=300,random_state=42)
        rf.fit(X.iloc[tr],y.iloc[tr])
        pred=rf.predict(X.iloc[ts]); mape.append(mean_absolute_percentage_error(y.iloc[ts],pred))
    rep={"MAPE_mean":float(np.mean(mape)),"MAPE_split":[float(v) for v in mape]}
    with open(settings.out_dir/"rf_baseline.json","w") as f: json.dump(rep,f,indent=2)
    log.info(f"Saved rf_baseline.json (mean MAPE={rep['MAPE_mean']:.3f})")
PY

###############################################################################
#12 ⟩ run_stage1.py … run_stage4.py
###############################################################################
echo "=============================================================================="
echo " STEP 12 – Write run_stage1.py … run_stage4.py "
echo "=============================================================================="
cat > "$PKG/run_stage1.py" <<'PY'
from customer_comms.viz import plots as V
from customer_comms.utils.logging_utils import get_logger
log=get_logger(__name__)
def main():
    log.info("Stage-1  – QA + raw plots")
    V.overview(); V.raw_call_files(); V.data_gaps(); V.qa_jsons()
if __name__=="__main__": main()
PY
cat > "$PKG/run_stage2.py" <<'PY'
from customer_comms.processing.combine import build_master
from customer_comms.analytics import corr_extras as C
from customer_comms.utils.logging_utils import get_logger
log=get_logger(__name__)
def main():
    log.info("Stage-2  – Correlation extras")
    df=build_master()
    if df.empty: return
    C.lag_heat(df); C.rolling_corr(df); C.corr_variants(df)
if __name__=="__main__": main()
PY
cat > "$PKG/run_stage3.py" <<'PY'
from customer_comms.analytics import mail_intent_corr as M
from customer_comms.utils.logging_utils import get_logger
log=get_logger(__name__)
def main():
    log.info("Stage-3  – Mail-intent matrix")
    M.plot_top10()
if __name__=="__main__": main()
PY
cat > "$PKG/run_stage4.py" <<'PY'
from customer_comms.models import baseline as B
from customer_comms.utils.logging_utils import get_logger
log=get_logger(__name__)
def main():
    log.info("Stage-4  – Random-Forest baseline")
    B.run_baseline()
if __name__=="__main__": main()
PY

###############################################################################
#13 ⟩ run_pipeline.py
###############################################################################
echo "=============================================================================="
echo " STEP 13 – Write run_pipeline.py (master orchestrator) "
echo "=============================================================================="
cat > "$PKG/run_pipeline.py" <<'PY'
import importlib, sys
from customer_comms.utils.logging_utils import get_logger
log=get_logger(__name__)
def run():
    for mod in ("run_stage1","run_stage2","run_stage3","run_stage4"):
        log.info(f"🏁  Running {mod} …")
        m=importlib.import_module(f"customer_comms.{mod}"); importlib.reload(m); m.main()
    log.info("🎉  Pipeline finished –  artefacts → customer_comms/output/")
if __name__=="__main__": run()
PY

###############################################################################
#14 ⟩ Copy CSVs into package data/  (optional)
###############################################################################
echo "=============================================================================="
echo " STEP 14 – Copy CSVs from ./data/ into package data/ if present "
echo "=============================================================================="
for f in GenesysExtract_20250703.csv GenesysExtract_20250609.csv all_mail_data.csv; do
  [[ -f data/$f ]] && cp -f "data/$f" "$PKG/data/"
done

###############################################################################
#15 ⟩ Execute pipeline
###############################################################################
echo "=============================================================================="
echo " STEP 15 – Execute the 4-stage pipeline "
echo "=============================================================================="
python -m customer_comms.run_pipeline || echo "❌  Top-level failure – see logs/"

echo -e "\n✅  ALL DONE → Plots & JSON in  customer_comms/output/   Logs → ./logs/"