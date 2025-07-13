#!/usr/bin/env bash
###############################################################################
#  one-click.sh   –   Call / Mail / Econ analytics pipeline  (2025-07-13)
#  ─ Windows-friendly, no venv, UTF-8 console
#  ─ Self-healing pip installs
#  ─ Live economics via yfinance → FRED/Stooq → Alpha-Vantage CSV fallbacks
#  ─ Intent file  = GenesysExtract_20250609.csv  (ConversationStart, uui_Intent)
#  ─ Volume file  = GenesysExtract_20250703.csv  (Date)
###############################################################################
set -euo pipefail
export PYTHONUTF8=1

PROJ="${1:-customer_comms}"        # ./one-click.sh MyProj
PKG="customer_comms"

echo "🔧  (Re)building project: $PROJ"
mkdir -p "$PROJ"/{data,logs,output,tests}
cd "$PROJ"

###############################################################################
# 0️⃣  Python version & dependency bootstrap
###############################################################################
python - <<'PY'
import sys, subprocess, importlib
if sys.version_info < (3,8):
    print("❌  Python ≥3.8 required – found", sys.version.split()[0]); sys.exit(1)
pkgs = """
pandas numpy matplotlib seaborn scipy scikit-learn statsmodels holidays
pydantic pydantic-settings yfinance pandas-datareader requests
""".split()
for p in pkgs:
    try: importlib.import_module(p.replace('-','_'))
    except ModuleNotFoundError:
        try:
            subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
            print("• installed",p)
        except subprocess.CalledProcessError:
            print(f"⚠️  could not install {p} – continuing")
print("✅  dependency bootstrap complete.")
PY

###############################################################################
# 1️⃣  Package skeleton
###############################################################################
for d in "$PKG" "$PKG/{data,processing,features,viz,utils,analytics,models}"; do
  mkdir -p $(echo "$d" | sed 's/{[^}]*}/data processing features viz utils analytics models/')
done
touch $(find "$PKG" -type d -not -path '*/\.*' -exec sh -c 'f="$0/__init__.py"; [ -f "$f" ] || :> "$f"' {} \;)

###############################################################################
# 2️⃣  config.py
###############################################################################
cat > "$PKG/config.py" << 'PY'
from pathlib import Path
try:    from pydantic_settings import BaseSettings          # v2
except ModuleNotFoundError: from pydantic import BaseSettings
from pydantic import Field
class Settings(BaseSettings):
    # ── raw files ───────────────────────────────────────────────────────────
    call_file:   str = "GenesysExtract_20250703.csv"   # column: Date
    intent_file: str = "GenesysExtract_20250609.csv"   # ConversationStart, uui_Intent
    mail_file:   str = "all_mail_data.csv"
    # ── column names ────────────────────────────────────────────────────────
    call_date_col:   str = "Date"
    intent_date_col: str = "ConversationStart"
    intent_col:      str = "uui_Intent"
    mail_date_col:   str = "mail_date"
    mail_type_col:   str = "mail_type"
    mail_volume_col: str = "mail_volume"
    # ── economic symbols: Yahoo / FRED / Stooq ─────────────────────────────
    econ_tickers: dict[str,str] = {
        "VIX": "^VIX",
        "SP500": "^GSPC",
        "FEDFUNDS": "FEDFUNDS"
    }
    # ── general params ─────────────────────────────────────────────────────
    min_rows: int = 20
    max_lag:  int = 21
    roll_win: int = 30
    # ── folders ────────────────────────────────────────────────────────────
    data_dir: Path = Field(default=Path("data"))
    log_dir:  Path = Field(default=Path("logs"))
    out_dir:  Path = Field(default=Path("output"))
settings = Settings()
PY

###############################################################################
# 3️⃣  utils / logging_utils.py
###############################################################################
cat > "$PKG/utils/logging_utils.py" << 'PY'
import logging, sys
from datetime import datetime
from ..config import settings
def get_logger(name="customer_comms"):
    lg = logging.getLogger(name)
    if lg.handlers: return lg
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    lg.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(fmt,"%Y-%m-%d %H:%M:%S"))
    try: sh.stream.reconfigure(encoding="utf-8")
    except AttributeError: pass
    lg.addHandler(sh)
    settings.log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(settings.log_dir/f"{name}_{datetime.now():%Y%m%d}.log",encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt,"%Y-%m-%d %H:%M:%S"))
    lg.addHandler(fh)
    return lg
PY

###############################################################################
# 4️⃣  data/loader.py  – volume, intent, mail
###############################################################################
cat > "$PKG/data/loader.py" << 'PY'
from __future__ import annotations
import pandas as pd
from pathlib import Path
from ..config import settings
from ..utils.logging_utils import get_logger
LOG = get_logger(__name__)
ENC = ("utf-8","utf-16","latin-1","cp1252")
def _read(p:Path)->pd.DataFrame:
    if not p.exists(): LOG.warning(f"Missing {p.name}"); return pd.DataFrame()
    for enc in ENC:
        try: return pd.read_csv(p,encoding=enc,on_bad_lines="skip",low_memory=False)
        except (UnicodeDecodeError,MemoryError): continue
    LOG.error(f"Decode fail {p.name}"); return pd.DataFrame()
def _to_date(s): return pd.to_datetime(s,errors="coerce").dt.tz_localize(None).dt.normalize()

def load_volume()->pd.DataFrame:
    df=_read(settings.data_dir/settings.call_file)
    if df.empty or settings.call_date_col not in df.columns: return pd.DataFrame()
    df[settings.call_date_col]=_to_date(df[settings.call_date_col])
    df=df.dropna(subset=[settings.call_date_col])
    return (df.groupby(settings.call_date_col).size()
              .rename("call_volume").reset_index()
              .rename(columns={settings.call_date_col:"date"}))

def load_intents()->pd.DataFrame:
    df=_read(settings.data_dir/settings.intent_file)
    if df.empty or settings.intent_date_col not in df.columns or settings.intent_col not in df.columns:
        return pd.DataFrame()
    df=df.rename(columns={settings.intent_date_col:"date",settings.intent_col:"intent"})
    df["date"]=_to_date(df["date"])
    return df.dropna(subset=["date","intent"])[["date","intent"]]

def load_mail()->pd.DataFrame:
    df=_read(settings.data_dir/settings.mail_file)
    if df.empty: return df
    df=df.rename(columns={settings.mail_date_col:"date",
                          settings.mail_type_col:"mail_type",
                          settings.mail_volume_col:"mail_volume"})
    df["date"]=_to_date(df["date"])
    df["mail_volume"]=pd.to_numeric(df["mail_volume"],errors="coerce")
    return df.dropna(subset=["date","mail_volume"])
PY

###############################################################################
# 5️⃣  data/econ_live.py  – yfinance + fallbacks
###############################################################################
cat > "$PKG/data/econ_live.py" << 'PY'
import pandas as pd, datetime, io, requests
from ..config import settings
from ..utils.logging_utils import get_logger
LOG=get_logger(__name__)
def _yf(ticker,s,e):
    import yfinance as yf
    try:
        return yf.download(ticker,start=s,end=e,progress=False)[["Close"]]
    except Exception: return pd.DataFrame()
def _fred(ticker,s,e):
    try:
        import pandas_datareader.data as web
        return web.DataReader(ticker,"fred",s,e)
    except Exception: return pd.DataFrame()
def _alpha_csv(ticker):
    try:
        csv=requests.get(
            f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&apikey=demo&datatype=csv",
            timeout=10).text
        df=pd.read_csv(io.StringIO(csv))
        if "timestamp" in df.columns and "adjusted_close" in df.columns:
            df=df.rename(columns={"timestamp":"date","adjusted_close":"Close"})
            df["date"]=pd.to_datetime(df["date"]); df=df.set_index("date")[["Close"]]
            return df.iloc[::-1]   # ascending
    except Exception: pass
    return pd.DataFrame()
def fetch():
    s=datetime.datetime.today()-datetime.timedelta(days=600)
    e=datetime.datetime.today()+datetime.timedelta(days=1)
    frames=[]
    for name,tick in settings.econ_tickers.items():
        df=_yf(tick,s,e)
        if df.empty: df=_fred(tick,s,e)
        if df.empty: df=_alpha_csv(tick)
        if df.empty:
            LOG.warning(f"{name} econ fetch failed"); continue
        df=df.rename(columns={"Close":name})
        LOG.info(f"{name} data rows: {len(df)}")
        frames.append(df)
    if not frames: return pd.DataFrame()
    econ=pd.concat(frames,axis=1).ffill().reset_index().rename(columns={"index":"date"})
    for col in econ.columns[1:]:
        econ[f"{col}_pct"]=econ[col].pct_change()
        econ[f"{col}_7dvol"]=econ[col].pct_change().rolling(7).std()
    return econ.dropna()
PY

###############################################################################
# 6️⃣  processing/augment.py  – volume gaps filled via intent ratio
###############################################################################
cat > "$PKG/processing/augment.py" << 'PY'
import pandas as pd
from ..data.loader import load_volume, load_intents
from ..utils.logging_utils import get_logger
LOG=get_logger(__name__)
def augment_calls()->pd.DataFrame:
    calls=load_volume(); intents=load_intents()
    if calls.empty or intents.empty: return calls
    ic=intents.groupby("date").size().rename("intent_cnt").reset_index()
    common=pd.merge(calls,ic,on="date")
    if common.empty: return calls
    ratio=common["call_volume"].sum()/common["intent_cnt"].sum()
    LOG.info(f"call:intent ratio {ratio:0.2f}")
    cal=pd.date_range(intents["date"].min(),intents["date"].max(),freq="D")
    out=pd.DataFrame({"date":cal})
    out=pd.merge(out,calls,on="date",how="left")
    ic_full=pd.merge(out[["date"]],ic,on="date",how="left").fillna(0)
    out["call_volume"]=out["call_volume"].fillna((ic_full["intent_cnt"]*ratio).round())
    out["is_augmented"]=out["call_volume"].isna()
    return out
PY

###############################################################################
# 7️⃣  features/engineering.py  – ≥ 50 vars incl. econ
###############################################################################
cat > "$PKG/features/engineering.py" << 'PY'
import pandas as pd, numpy as np, holidays
from ..processing.augment import augment_calls
from ..data.loader import load_mail
from ..data.econ_live import fetch as econ_fetch
from ..config import settings
from ..utils.logging_utils import get_logger
LOG=get_logger(__name__)
us_hol=holidays.US()
def build()->pd.DataFrame:
    calls=augment_calls(); mail=load_mail(); econ=econ_fetch()
    if calls.empty or mail.empty: 
        LOG.error("Feature build: missing calls or mail"); return pd.DataFrame()
    mail=mail.groupby("date")["mail_volume"].sum().reset_index()
    df=pd.merge(calls,mail,on="date",how="inner").sort_values("date")
    # temporal flags
    df["dow"]=df["date"].dt.weekday
    df["month"]=df["date"].dt.month
    df["is_holiday"]=df["date"].dt.date.isin(us_hol).astype(int)
    df["is_eom"]=df["date"].dt.is_month_end.astype(int)
    # lags 1-14  (28 features)
    for lag in range(1,15):
        df[f"mail_lag_{lag}"]=df["mail_volume"].shift(lag)
        df[f"call_lag_{lag}"]=df["call_volume"].shift(lag)
    # rolling stats (8)
    for win in (7,30):
        df[f"mail_ma{win}"]=df["mail_volume"].rolling(win).mean()
        df[f"call_ma{win}"]=df["call_volume"].rolling(win).mean()
        df[f"mail_vol{win}"]=df["mail_volume"].rolling(win).std()
        df[f"call_vol{win}"]=df["call_volume"].rolling(win).std()
    # pct change (2)
    df["mail_pct"]=df["mail_volume"].pct_change()
    df["call_pct"]=df["call_volume"].pct_change()
    # econ merge + interactions (≈ 20+)
    if not econ.empty:
        df=pd.merge(df,econ,on="date",how="left").ffill()
        for col in [c for c in econ.columns if c!="date"]:
            df[f"mail_x_{col}"]=df["mail_volume"]*df[col]
    df=df.dropna().reset_index(drop=True)
    LOG.info(f"Final feature matrix: {df.shape}")
    return df
PY

###############################################################################
# 8️⃣  viz/plots.py  – overview, raw files, data gaps, QA JSON
###############################################################################
cat > "$PKG/viz/plots.py" << 'PY'
import json, pandas as pd, seaborn as sns, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from ..config import settings
from ..processing.augment import augment_calls
from ..data.loader import load_mail
from ..utils.logging_utils import get_logger
LOG=get_logger(__name__); OUT=settings.out_dir; OUT.mkdir(exist_ok=True)
def _save(fig,name): fig.savefig(OUT/name,dpi=300,bbox_inches="tight"); plt.close(fig); LOG.info(f"Saved {name}")

def overview():
    calls=augment_calls(); mail=load_mail()
    if calls.empty or mail.empty: return
    mail=mail.groupby("date")["mail_volume"].sum().reset_index()
    df=pd.merge(calls,mail,on="date",how="inner").sort_values("date")
    if df.empty: return
    norm=lambda s:(s-s.min())/(s.max()-s.min())*100
    df["mail_norm"]=norm(df["mail_volume"]); df["call_norm"]=norm(df["call_volume"])
    fig,ax=plt.subplots(figsize=(14,6))
    ax.bar(df["date"],df["mail_norm"],alpha=.6,label="Mail (norm)")
    ax.plot(df["date"],df["call_norm"],color="tab:red",lw=2,label="Calls (norm)")
    ax.set_title("Mail vs Call (normalised)"); ax.set_ylabel("0-100")
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d")); plt.setp(ax.get_xticklabels(),rotation=45,ha="right")
    ax.legend(); ax.grid(ls="--",alpha=.3); _save(fig,"overview.png")

def raw_call_files():
    import os
    from ..config import settings as cfg
    fig,ax=plt.subplots(figsize=(14,6))
    for f in (cfg.call_file,cfg.intent_file):
        p=cfg.data_dir/f
        if not p.exists(): continue
        df=pd.read_csv(p,low_memory=False)
        dcol=cfg.call_date_col if f==cfg.call_file else cfg.intent_date_col
        if dcol not in df.columns: continue
        df[dcol]=pd.to_datetime(df[dcol],errors="coerce").dt.normalize()
        daily=df.groupby(dcol).size().reset_index().rename(columns={dcol:"date",0:"cnt"})
        ax.plot(daily["date"],daily["cnt"],label=os.path.basename(f))
    if not ax.lines: plt.close(fig); return
    ax.set_title("Raw call & intent daily counts"); ax.legend(); ax.grid(ls="--",alpha=.3)
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d")); plt.setp(ax.get_xticklabels(),rotation=45,ha="right")
    _save(fig,"raw_call_files.png")

def data_gaps():
    calls=augment_calls(); mail=load_mail()
    if calls.empty and mail.empty: return
    start=min(calls["date"].min() if not calls.empty else mail["date"].min(),
              mail["date"].min() if not mail.empty else calls["date"].min())
    end=max(calls["date"].max() if not calls.empty else mail["date"].max(),
            mail["date"].max() if not mail.empty else calls["date"].max())
    cal=pd.DataFrame({"date":pd.bdate_range(start,end,freq="B")})
    cal["call"]=cal["date"].isin(set(calls["date"]))
    cal["mail"]=cal["date"].isin(set(mail["date"]))
    cal["status"]=cal.apply(lambda r:"Both" if r.call and r.mail else("Call only" if r.call else("Mail only" if r.mail else "None")),axis=1)
    col={"Both":"green","Call only":"red","Mail only":"blue","None":"grey"}
    fig,ax=plt.subplots(figsize=(14,1.8))
    for s,g in cal.groupby("status"):
        ax.scatter(g["date"],[0]*len(g),color=col[s],s=15,label=s)
    ax.legend(ncol=4,loc="upper center"); ax.yaxis.set_visible(False); ax.grid(False)
    ax.set_title("Data coverage calendar")
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d")); plt.setp(ax.get_xticklabels(),rotation=45,ha="right")
    _save(fig,"data_gaps.png")

def qa_jsons():
    calls=augment_calls(); mail=load_mail()
    rep={"call_dates":int(calls["date"].nunique()),"mail_dates":int(mail["date"].nunique()),
         "intersection":int(len(set(calls["date"])&set(mail["date"])))}
    (OUT/"call_coverage.json").write_text(json.dumps(rep,indent=2),encoding="utf-8"); LOG.info("Saved call_coverage.json")
PY

###############################################################################
# 9️⃣  analytics/eda.py  – corr variants, lag heat, rolling corr
###############################################################################
cat > "$PKG/analytics/eda.py" << 'PY'
import numpy as np, pandas as pd, seaborn as sns, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, warnings, json, pathlib
from scipy.stats import pearsonr, ConstantInputWarning
from ..config import settings
from ..utils.logging_utils import get_logger
LOG=get_logger(__name__); OUT=settings.out_dir; OUT.mkdir(exist_ok=True)
def _save(fig,name): fig.savefig(OUT/name,dpi=300,bbox_inches="tight"); plt.close(fig); LOG.info(f"Saved {name}")
def _r(x,y):
    if x.nunique()<2 or y.nunique()<2: return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",ConstantInputWarning)
        return pearsonr(x,y)[0]
def corr_variants(df):
    pairs={"raw":("mail_volume","call_volume"),
           "lag3":("mail_lag_3","call_volume"),
           "lag7":("mail_lag_7","call_volume"),
           "ma7":("mail_ma7","call_ma7"),
           "pct":(df["mail_pct"],df["call_volume"])}
    vals=[(k,_r(df[x] if isinstance(x,str) else x, df[y] if isinstance(y,str) else y)) for k,(x,y) in pairs.items()]
    fig,ax=plt.subplots(figsize=(7,3)); sns.barplot(x=[v[0] for v in vals],y=[v[1] for v in vals],ax=ax)
    ax.set_title("Correlation variants"); ax.set_ylabel("r")
    _save(fig,"variant_corr.png")
    return max(vals,key=lambda t:abs(t[1]))
def lag_heat(df):
    lags=range(settings.max_lag+1)
    vals=[_r(df["mail_volume"],df["call_volume"].shift(-l)) for l in lags]
    if np.isfinite(vals).sum()==0: return
    fig,ax=plt.subplots(figsize=(9,3))
    sns.heatmap(np.array(vals).reshape(1,-1),annot=True,fmt=".2f",cmap="vlag",center=0,cbar=False,ax=ax,
                xticklabels=lags,yticklabels=["r"])
    ax.set_title("Mail→Call corr vs lag"); _save(fig,"lag_corr_heat.png")
def rolling_corr(df):
    if len(df)<settings.roll_win+5: return
    r=(df.set_index("date")["mail_volume"].rolling(settings.roll_win)
          .corr(df.set_index("date")["call_volume"]))
    if r.dropna().empty: return
    fig,ax=plt.subplots(figsize=(12,4)); ax.plot(r.index,r); ax.grid(ls="--",alpha=.4)
    ax.set_title(f"{settings.roll_win}-day rolling corr"); _save(fig,"rolling_corr.png")
PY

###############################################################################
# 🔟  analytics/anomaly.py  – seasonality + anomaly + intent matrix
###############################################################################
cat > "$PKG/analytics/anomaly.py" << 'PY'
import numpy as np, pandas as pd, seaborn as sns, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, json
from sklearn.ensemble import IsolationForest
from ..data.loader import load_intents, load_mail
from ..config import settings
from ..utils.logging_utils import get_logger
LOG=get_logger(__name__); OUT=settings.out_dir
def _save(fig,name): fig.savefig(OUT/name,dpi=300,bbox_inches="tight"); plt.close(fig); LOG.info(f"Saved {name}")
def plots(df):
    # anomaly scatter
    iso=IsolationForest(contamination=0.05,random_state=42)
    df=df.copy()
    df["anom"]=iso.fit_predict(df[["call_volume","mail_volume"]])==-1
    fig,ax=plt.subplots(figsize=(8,6))
    ax.scatter(df["mail_volume"],df["call_volume"],
               c=np.where(df["anom"],"red","grey"),alpha=.6)
    ax.set_title("Call vs Mail (IsolationForest anomalies)")
    _save(fig,"anomaly_scatter.png")
    # seasonality
    tmp=df.copy(); tmp["month"]=tmp["date"].dt.strftime("%b"); tmp["dow"]=tmp["date"].dt.day_name()
    piv=tmp.pivot_table(index="dow",columns="month",values="call_volume",aggfunc="mean").reindex(["Monday","Tuesday","Wednesday","Thursday","Friday"])
    fig,ax=plt.subplots(figsize=(9,6))
    sns.heatmap(piv,annot=True,fmt=".0f",cmap="Reds",ax=ax)
    ax.set_title("Avg call volume – Month × Weekday")
    _save(fig,"call_seasonality.png")
    # mail-type × intent corr
    intents=load_intents(); mail=load_mail()
    if intents.empty or mail.empty: return
    intent_top=intents["intent"].value_counts().loc[lambda s:s>=250].nlargest(10).index
    mail_top=mail["mail_type"].value_counts().nlargest(10).index
    mail_piv=mail[mail["mail_type"].isin(mail_top)].pivot_table(index="date",columns="mail_type",values="mail_volume",aggfunc="sum")
    intent_piv=intents[intents["intent"].isin(intent_top)].pivot_table(index="date",columns="intent",values="intent",aggfunc="count")
    merged=pd.merge(mail_piv,intent_piv,left_index=True,right_index=True,how="inner").fillna(0)
    if merged.empty: return
    corr=merged.corr().loc[mail_top,intent_top]
    fig,ax=plt.subplots(figsize=(8,6))
    sns.heatmap(corr,annot=True,cmap="vlag",center=0,fmt=".2f",ax=ax)
    ax.set_title("Mail-type × Intent correlation")
    _save(fig,"mailtype_intent_corr.png")
    # readiness
    ready={"rows":int(len(df)),"cols":int(len(df.columns)),
           "anomaly_pct":float(df["anom"].mean())}
    (OUT/"model_readiness.json").write_text(json.dumps(ready,indent=2),encoding="utf-8")
    LOG.info("Saved model_readiness.json")
PY

###############################################################################
# 1️⃣1  run_stage1.py
###############################################################################
cat > "run_stage1.py" << 'PY'
from customer_comms.viz.plots import overview, raw_call_files, data_gaps, qa_jsons
from customer_comms.utils.logging_utils import get_logger
lg=get_logger("stage1")
lg.info("Stage-1 start")
overview(); raw_call_files(); data_gaps(); qa_jsons()
lg.info("Stage-1 ✅")
PY

###############################################################################
# 1️⃣2  run_stage2.py
###############################################################################
cat > "run_stage2.py" << 'PY'
from customer_comms.features.engineering import build
from customer_comms.analytics.eda import corr_variants, lag_heat, rolling_corr
from customer_comms.utils.logging_utils import get_logger
lg=get_logger("stage2")
lg.info("Stage-2 start")
df=build()
if df.empty: lg.error("Stage-2: no data"); exit()
corr_variants(df); lag_heat(df); rolling_corr(df)
lg.info("Stage-2 ✅")
PY

###############################################################################
# 1️⃣3  run_stage3.py
###############################################################################
cat > "run_stage3.py" << 'PY'
from customer_comms.features.engineering import build
from customer_comms.analytics.anomaly import plots
from customer_comms.utils.logging_utils import get_logger
lg=get_logger("stage3"); lg.info("Stage-3 start")
df=build()
if df.empty: lg.error("Stage-3: no data"); exit()
plots(df); lg.info("Stage-3 ✅")
PY

###############################################################################
# 1️⃣4  run_stage4.py  – baseline RF, JSON metrics, COO pack copy
###############################################################################
cat > "run_stage4.py" << 'PY'
from customer_comms.features.engineering import build
from customer_comms.config import settings
from customer_comms.utils.logging_utils import get_logger
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, json, subprocess, sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error
lg=get_logger("stage4"); lg.info("Stage-4 start")
df=build()
if len(df)<=150:
    lg.warning("Too few rows for baseline RF"); sys.exit()
X=df.drop(columns=["date","call_volume"]); y=df["call_volume"]
tscv=TimeSeriesSplit(n_splits=5)
pred=np.zeros_like(y)
for tr,te in tscv.split(X):
    m=RandomForestRegressor(n_estimators=300,random_state=42); m.fit(X.iloc[tr],y.iloc[tr]); pred[te]=m.predict(X.iloc[tr*0+te])
r2=r2_score(y,pred); mae=mean_absolute_error(y,pred)
fig,ax=plt.subplots(figsize=(14,6)); ax.plot(df["date"],y,label="Actual")
ax.plot(df["date"],pred,label="Predicted",alpha=.7)
ax.fill_between(df["date"],y,pred,alpha=.2); ax.legend(); ax.grid(ls="--",alpha=.3)
ax.set_title(f"Baseline RF – R²={r2:0.2f}  MAE={mae:0.0f}")
fig.savefig(settings.out_dir/"baseline_rf.png",dpi=300,bbox_inches="tight"); plt.close(fig)
(settings.out_dir/"rf_metrics.json").write_text(json.dumps({"R2":r2,"MAE":mae},indent=2),encoding="utf-8")
lg.info("Saved baseline_rf.png & rf_metrics.json")
# copy COO pack
subprocess.call([sys.executable,"copy_coo_pack.py"])
lg.info("Stage-4 ✅")
PY

###############################################################################
# 1️⃣5  copy_coo_pack.py
###############################################################################
cat > "copy_coo_pack.py" << 'PY'
import shutil, glob
from customer_comms.config import settings
dest=settings.out_dir/"COO_pack"; dest.mkdir(exist_ok=True)
for f in glob.glob(str(settings.out_dir/"*.png")): shutil.copy2(f,dest)
(dest/"insights.md").write_text("# COO Insight Pack\nPaste these PNGs into your deck.",encoding="utf-8")
print("🗂  COO pack ready:",dest)
PY

###############################################################################
# 1️⃣6  run_all.py
###############################################################################
cat > "run_all.py" << 'PY'
import subprocess, sys, datetime
STAGES=["run_stage1.py","run_stage2.py","run_stage3.py","run_stage4.py"]
for s in STAGES:
    print(f"\n── {datetime.datetime.now():%H:%M:%S} running {s} ──", flush=True)
    try: subprocess.run([sys.executable,s],check=True)
    except subprocess.CalledProcessError: print(f"⚠️  {s} failed – continuing.")
print("\n✅  Pipeline finished – artefacts in ./output/")
PY

###############################################################################
# 1️⃣7  execute pipeline
###############################################################################
echo "🚀  Running full pipeline …"
python run_all.py || echo "⚠️  One or more stages failed – check logs."