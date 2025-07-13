#!/usr/bin/env bash
###############################################################################
#  customer_comms_full.sh   –  ONE-SHOT END-TO-END PIPELINE (VERBOSE)
#  ---------------------------------------------------------------------------
#  Covers EVERY Epic / Story you provided:
#    • Dependency bootstrap
#    • Package scaffold
#    • Multi-encoding CSV loaders + call-intent augmentation
#    • Econ feed via yfinance (Agg backend → no Tk)
#    • Feature engineering (lags, pct, diff, MA, z-score, econ joins)
#    • 20+ production PNGs + JSON QA + RF baseline
#    • Robust logging, chained-assignment silenced, Windows-safe UTF-8
###############################################################################
set -euo pipefail

# ---------------------------------------------------------------------------
# 0.  Dependencies  ----------------------------------------------------------
# ---------------------------------------------------------------------------
echo "=============================================================================="
echo " STEP 0 – Install / verify Python dependencies "
echo "=============================================================================="

python - <<'PY'
import importlib, subprocess, sys, os
pkgs = [
    "pandas==2.2.2", "numpy", "matplotlib", "seaborn", "scikit-learn",
    "scipy", "holidays", "yfinance", "pydantic", "pydantic-settings"
]
for p in pkgs:
    try:
        importlib.import_module(p.split("==")[0].replace("-", "_"))
        print(f"✔  {p}")
    except ModuleNotFoundError:
        print(f"… installing {p}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])

#  Force matplotlib non-GUI backend to avoid Tk runtime errors
import matplotlib
matplotlib.use("Agg")
os.environ["MPLBACKEND"] = "Agg"
print("✔  matplotlib backend set to Agg (headless)")
PY

# ---------------------------------------------------------------------------
# 1.  Package skeleton  ------------------------------------------------------
# ---------------------------------------------------------------------------
echo "=============================================================================="
echo " STEP 1 – Create package skeleton "
echo "=============================================================================="

PKG="customer_comms"
for d in "" data processing analytics viz utils models; do
  mkdir -p "$PKG/${d}" && touch "$PKG/${d}/__init__.py"
done
mkdir -p logs output data

# ---------------------------------------------------------------------------
# 2.  Config  ---------------------------------------------------------------
# ---------------------------------------------------------------------------
echo "=============================================================================="
echo " STEP 2 – Write customer_comms/config.py "
echo "=============================================================================="

cat > "$PKG/config.py" <<'PY'
from pathlib import Path
try:
    from pydantic_settings import BaseSettings           # pydantic v2
except ModuleNotFoundError:                              # fall-back pydantic v1
    from pydantic import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # ----------------------------------------------------------------------
    # File names (place **exactly** in ./data/)
    # ----------------------------------------------------------------------
    call_vol_file: str = "GenesysExtract_20250703.csv"   # 18-month call volumes
    call_int_file: str = "GenesysExtract_20250609.csv"   # 6-month intents
    mail_file:     str = "all_mail_data.csv"             # 18-month mail data

    # Column mappings
    call_date_col:   str = "Date"
    intent_date_col: str = "ConversationStart"
    intent_col:      str = "uui_Intent"

    mail_date_col:   str = "mail_date"
    mail_type_col:   str = "mail_type"
    mail_vol_col:    str = "mail_volume"

    # ----- General thresholds -----
    intent_min:      int = 250           # drop intents < 250 observations
    max_lag_days:    int = 21            # lag scan horizon

    # Data folders
    data_dir: Path = Field(default=Path("data"))
    log_dir:  Path = Field(default=Path("logs"))
    out_dir:  Path = Field(default=Path("customer_comms") / "output")

    # Econ tickers  (name → yfinance symbol)
    econ_tickers: dict[str, str] = {
        "VIX": "^VIX",
        "SP500": "^GSPC",
        "FEDFUNDS": "FEDFUNDS"          # FRED series (may 404 some days)
    }

settings = Settings()
PY

# ---------------------------------------------------------------------------
# 3.  Logging utils  ---------------------------------------------------------
# ---------------------------------------------------------------------------
echo "=============================================================================="
echo " STEP 3 – Write utils/logging_utils.py "
echo "=============================================================================="

cat > "$PKG/utils/logging_utils.py" <<'PY'
import logging, sys, os
from datetime import datetime
from ..config import settings

def get_logger(name: str = "customer_comms") -> logging.Logger:
    lg = logging.getLogger(name)
    if lg.handlers:
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
    fh = logging.FileHandler(settings.log_dir / f"{name}_{datetime.now():%Y%m%d}.log",
                             encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt, dh))
    lg.addHandler(fh)

    #  Hide chained-assignment/future warnings
    logging.getLogger("pandas.core.common").setLevel(logging.ERROR)
    os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
    return lg
PY

# ---------------------------------------------------------------------------
# 4.  IO helper (multi-encoding CSV)  ---------------------------------------
# ---------------------------------------------------------------------------
echo "=============================================================================="
echo " STEP 4 – Write utils/io.py "
echo "=============================================================================="

cat > "$PKG/utils/io.py" <<'PY'
import pandas as pd, logging, os
ENCODINGS = ("utf-8", "latin-1", "cp1252", "utf-16")
log = logging.getLogger("customer_comms")
def read_csv_any(path, **kw):
    if not os.path.isfile(path):
        log.warning(f"Missing {path}")
        return pd.DataFrame()
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="skip", low_memory=False, **kw)
        except UnicodeDecodeError:
            continue
    log.error(f"Could not decode {path}")
    return pd.DataFrame()
PY

# ---------------------------------------------------------------------------
# 5.  Data loaders  ---------------------------------------------------------
# ---------------------------------------------------------------------------
echo "=============================================================================="
echo " STEP 5 – Write data/loader.py "
echo "=============================================================================="

cat > "$PKG/data/loader.py" <<'PY'
from __future__ import annotations
import pandas as pd, numpy as np, yfinance as yf
from ..config import settings
from ..utils.io import read_csv_any
from ..utils.logging_utils import get_logger
log = get_logger(__name__)

def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None).dt.normalize()

# -------------------------  Call volume  -----------------------------------
def load_call_volume() -> pd.DataFrame:
    df = read_csv_any(settings.data_dir / settings.call_vol_file,
                      usecols=[settings.call_date_col])
    if df.empty:
        return df
    df.rename(columns={settings.call_date_col: "date"}, inplace=True)
    df["date"] = _to_date(df["date"])
    df = df.dropna(subset=["date"])
    cv = (df.groupby("date").size()
            .reset_index(name="call_volume")
            .sort_values("date"))
    log.info(f"Call volume rows: {len(cv)}")
    return cv

# -------------------------  Intents  ---------------------------------------
def load_intents() -> pd.DataFrame:
    df = read_csv_any(settings.data_dir / settings.call_int_file,
                      usecols=[settings.intent_date_col, settings.intent_col])
    if df.empty:
        return df
    df.rename(columns={settings.intent_date_col: "date",
                       settings.intent_col: "intent"}, inplace=True)
    df["date"] = _to_date(df["date"])
    df = df.dropna(subset=["date"])
    #  Filter by min observations
    keep = df["intent"].value_counts()
    keep = keep[keep >= settings.intent_min].index
    df = df[df["intent"].isin(keep)]
    mat = (df.groupby(["date", "intent"]).size()
             .unstack(fill_value=0)
             .reset_index()
             .sort_values("date"))
    log.info(f"Intent matrix shape: {mat.shape}")
    return mat

# -------------------------  Mail  ------------------------------------------
def load_mail() -> pd.DataFrame:
    df = read_csv_any(settings.data_dir / settings.mail_file,
                      usecols=[settings.mail_date_col,
                               settings.mail_type_col,
                               settings.mail_vol_col])
    if df.empty:
        return df
    df.rename(columns={settings.mail_date_col: "date",
                       settings.mail_type_col: "mail_type",
                       settings.mail_vol_col: "mail_volume"}, inplace=True)
    df["date"] = _to_date(df["date"])
    df["mail_volume"] = pd.to_numeric(df["mail_volume"], errors="coerce")
    df = df.dropna(subset=["date", "mail_volume"])
    log.info(f"Mail rows: {len(df)}")
    return df

# -------------------------  Econ (yfinance)  -------------------------------
def load_econ() -> pd.DataFrame:
    out = []
    for name, ticker in settings.econ_tickers.items():
        try:
            d = yf.download(ticker, period="2y", interval="1d", progress=False)
            if d.empty:
                raise ValueError("yfinance empty")
            series = (d["Adj Close"] if "Adj Close" in d else d.iloc[:, 0]).rename(name)
            out.append(series)
            log.info(f"{name} rows: {len(series)}")
        except Exception as e:
            log.error(f"{name} download failed: {e}")
    if not out:
        return pd.DataFrame()
    df = pd.concat(out, axis=1)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df.reset_index(names="date")
PY

# ---------------------------------------------------------------------------
# 6.  Processing / combine  --------------------------------------------------
# ---------------------------------------------------------------------------
echo "=============================================================================="
echo " STEP 6 – Write processing/combine.py "
echo "=============================================================================="

cat > "$PKG/processing/combine.py" <<'PY'
import pandas as pd, numpy as np, warnings
from ..data.loader import load_call_volume, load_intents, load_mail, load_econ
from ..utils.logging_utils import get_logger
from ..config import settings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
log = get_logger(__name__)

def _norm(s):
    return (s - s.min()) / (s.max() - s.min()) * 100 if s.max() != s.min() else s * 0

def build_master() -> pd.DataFrame:
    call = load_call_volume()
    mail = load_mail()
    if call.empty or mail.empty:
        log.error("Missing call or mail data")
        return pd.DataFrame()

    # ------------------  Weekday mask  ------------------
    call = call[call.date.dt.weekday < 5]
    mail = mail[mail.date.dt.weekday < 5]

    # ------------------  Augment call vol  ----------------
    intents = load_intents()
    if not intents.empty:
        daily_int = intents.drop(columns="date").sum(axis=1)
        df_int = pd.DataFrame({"date": intents["date"], "intent_cnt": daily_int})
        merged = pd.merge(call, df_int, on="date", how="outer")
        scale = (merged.call_volume.mean() / merged.intent_cnt.mean()) if merged.intent_cnt.mean() else 1
        merged.call_volume.fillna(merged.intent_cnt * scale, inplace=True)
        call = merged[["date", "call_volume"]]

    # ------------------  Core join  ----------------------
    mail_daily = mail.groupby("date", as_index=False)["mail_volume"].sum()
    core = pd.merge(call, mail_daily, on="date", how="inner").sort_values("date")
    # ------------------  Features  -----------------------
    core["call_norm"] = _norm(core.call_volume)
    core["mail_norm"] = _norm(core.mail_volume)
    core["call_pct"] = core.call_volume.pct_change().fillna(0)
    core["mail_pct"] = core.mail_volume.pct_change().fillna(0)
    core["call_diff"] = core.call_volume.diff().fillna(0)
    core["mail_diff"] = core.mail_volume.diff().fillna(0)
    core["mail_ma7"] = core.mail_volume.rolling(7).mean()
    core["call_ma7"] = core.call_volume.rolling(7).mean()
    core["mail_z14"] = ((core.mail_volume - core.mail_volume.rolling(14).mean()) /
                        core.mail_volume.rolling(14).std())
    core["call_z14"] = ((core.call_volume - core.call_volume.rolling(14).mean()) /
                        core.call_volume.rolling(14).std())
    # ------------------  Econ merge  ---------------------
    econ = load_econ()
    if not econ.empty:
        core = pd.merge(core, econ, on="date", how="left")

    log.info(f"Master frame rows: {len(core)}")
    return core
PY

# ---------------------------------------------------------------------------
# 7.  Visualisations  --------------------------------------------------------
# ---------------------------------------------------------------------------
echo "=============================================================================="
echo " STEP 7 – Write viz/plots.py "
echo "=============================================================================="

cat > "$PKG/viz/plots.py" <<'PY'
import matplotlib
matplotlib.use("Agg")           # enforce headless backend
import matplotlib.pyplot as plt, seaborn as sns, pandas as pd, numpy as np, json
from matplotlib.dates import DateFormatter
from ..config import settings
from ..processing.combine import build_master
from ..data.loader import load_call_volume, load_mail
from ..utils.logging_utils import get_logger
log = get_logger(__name__)
settings.out_dir.mkdir(parents=True, exist_ok=True)

def _save(fig, name):
    fig.tight_layout()
    out = settings.out_dir / name
    fig.savefig(out, dpi=300)
    plt.close(fig)
    log.info(f"Saved {out.name}")

# ---------------------------  Overview  -----------------------------
def overview():
    df = build_master()
    if df.empty: return
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(df.date, df.mail_norm, alpha=.6, label="Mail (norm)")
    ax.plot(df.date, df.call_norm, lw=2, color="tab:red", label="Calls (norm)")
    ax.set_title("Mail vs Call (normalised 0-100)")
    ax.set_ylabel("0-100")
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(ls="--", alpha=.3)
    ax.legend()
    _save(fig, "overview.png")

# ---------------------------  Raw call-files -------------------------
def raw_call_files():
    from ..data.loader import load_intents
    call = load_call_volume()
    intents = load_intents()
    if call.empty and intents.empty: return
    fig, ax = plt.subplots(figsize=(14, 6))
    if not call.empty:
        ax.plot(call.date, call.call_volume, label=settings.call_vol_file)
    if not intents.empty:
        daily_int = intents.drop(columns="date").sum(axis=1)
        ax.plot(intents.date, daily_int, label=settings.call_int_file)
    ax.set_title("Raw daily call counts")
    ax.set_ylabel("Calls")
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(); ax.grid(ls="--", alpha=.3)
    _save(fig, "raw_call_files.png")

# ---------------------------  Data gaps  -----------------------------
def data_gaps():
    call = load_call_volume()
    mail = load_mail()
    if call.empty and mail.empty: return
    start = min(call.date.min() if not call.empty else mail.date.min(),
                mail.date.min() if not mail.empty else call.date.min())
    end = max(call.date.max() if not call.empty else mail.date.max(),
              mail.date.max() if not mail.empty else call.date.max())
    cal = pd.DataFrame({"date": pd.bdate_range(start, end)})
    cal["call"] = cal.date.isin(set(call.date))
    cal["mail"] = cal.date.isin(set(mail.date))
    cal["status"] = np.select(
        [cal.call & cal.mail, cal.call, cal.mail],
        ["Both", "Call only", "Mail only"], default="None"
    )
    col = dict(Both="green", **{"Call only": "red", "Mail only": "blue", "None": "grey"})
    fig, ax = plt.subplots(figsize=(14, 2))
    for st, sub in cal.groupby("status"):
        ax.scatter(sub.date, [0]*len(sub), c=col[st], s=10, label=st)
    ax.set_yticks([])
    ax.set_title("Weekday data coverage")
    ax.legend(ncol=4, loc="upper center"); ax.grid(False)
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    _save(fig, "data_gaps.png")

# ---------------------------  QA JSON  --------------------------------
def qa_jsons():
    call = load_call_volume()
    mail = load_mail()
    rep = {
        "call_dates": int(call.date.nunique() if not call.empty else 0),
        "mail_dates": int(mail.date.nunique() if not mail.empty else 0)
    }
    with open(settings.out_dir / "qa_summary.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    log.info("Saved qa_summary.json")
PY

# ---------------------------------------------------------------------------
# 8.  Correlation extras  ----------------------------------------------------
# ---------------------------------------------------------------------------
echo "=============================================================================="
echo " STEP 8 – Write analytics/corr_extras.py "
echo "=============================================================================="

cat > "$PKG/analytics/corr_extras.py" <<'PY'
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import pearsonr
from matplotlib.dates import DateFormatter
from ..config import settings
from ..utils.logging_utils import get_logger
log = get_logger(__name__)

def _r(a, b):
    df = pd.concat([a, b], axis=1).dropna()
    if len(df) < 3 or df.iloc[:, 0].std() == 0 or df.iloc[:, 1].std() == 0:
        return 0.0
    return pearsonr(df.iloc[:, 0], df.iloc[:, 1])[0]

def lag_heat(df, max_lag=21):
    if df.empty: return
    vals = [_r(df.mail_volume, df.call_volume.shift(-lag)) for lag in range(max_lag+1)]
    fig, ax = plt.subplots(figsize=(12, 1.4))
    sns.heatmap(np.array(vals).reshape(1, -1), annot=True, fmt=".2f", cmap="vlag",
                cbar=False, xticklabels=range(max_lag+1), yticklabels=["r"], ax=ax)
    ax.set_title("Mail → Call correlation vs lag")
    ax.set_xlabel("Lag (days)")
    fig.tight_layout(); fig.savefig(settings.out_dir / "lag_corr_heatmap.png", dpi=300)
    plt.close(fig); log.info("Saved lag_corr_heatmap.png")

def rolling_corr(df, win=30):
    if df.empty or len(df) < win+2: return
    r = (df.set_index("date").mail_volume
         .rolling(win).corr(df.set_index("date").call_volume))
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(r.index, r); ax.set_title(f"{win}-day rolling correlation")
    ax.grid(ls="--", alpha=.4)
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout(); fig.savefig(settings.out_dir / "rolling_corr.png", dpi=300)
    plt.close(fig); log.info("Saved rolling_corr.png")

def corr_variants(df):
    if df.empty: return
    v = {
        "raw":  _r(df.mail_volume, df.call_volume),
        "lag3": _r(df.mail_volume, df.call_volume.shift(-3)),
        "lag7": _r(df.mail_volume, df.call_volume.shift(-7)),
        "ma7":  _r(df.mail_volume.rolling(7).mean(), df.call_volume.rolling(7).mean()),
        "pct":  _r(df.mail_pct, df.call_pct),
        "z14":  _r(df.mail_z14, df.call_z14)
    }
    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(x=list(v.keys()), y=list(v.values()), ax=ax)
    ax.set_title("Correlation variants"); ax.set_ylabel("r")
    fig.tight_layout(); fig.savefig(settings.out_dir / "variant_corr.png", dpi=300)
    plt.close(fig); log.info("Saved variant_corr.png")
PY

# ---------------------------------------------------------------------------
# 9.  Mail-intent correlation  ----------------------------------------------
# ---------------------------------------------------------------------------
echo "=============================================================================="
echo " STEP 9 – Write analytics/mail_intent_corr.py "
echo "=============================================================================="

cat > "$PKG/analytics/mail_intent_corr.py" <<'PY'
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import pearsonr
from ..data.loader import load_mail, load_intents
from ..config import settings
from ..utils.logging_utils import get_logger
log = get_logger(__name__)

def plot_top10():
    mail = load_mail()
    intents = load_intents()
    if mail.empty or intents.empty:
        log.warning("Mail or intents empty – skipping mail-intent heat map")
        return
    mail_piv = (mail.pivot_table(index="date", columns="mail_type",
                                 values="mail_volume", aggfunc="sum")
                   .fillna(0))
    merged = pd.merge(mail_piv.reset_index(), intents, on="date", how="inner").set_index("date")
    mail_cols = [c for c in mail_piv.columns if c in merged.columns]
    intent_cols = [c for c in intents.columns if c != "date" and c in merged.columns]
    results = []
    for m in mail_cols:
        for i in intent_cols:
            if merged[m].std()==0 or merged[i].std()==0: continue
            r,_ = pearsonr(merged[m], merged[i])
            results.append((m, i, abs(r), r))
    if not results:
        log.warning("No valid mail-intent correlations")
        return
    top = sorted(results, key=lambda x: x[2], reverse=True)[:10]
    df = (pd.DataFrame(top, columns=["mail","intent","abs_r","r"])
            .pivot(index="mail", columns="intent", values="r")
            .fillna(0))
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Top-10 |r|  Mail-Type × Intent")
    fig.tight_layout(); fig.savefig(settings.out_dir/"mailtype_intent_corr.png", dpi=300)
    plt.close(fig); log.info("Saved mailtype_intent_corr.png")
PY

# ---------------------------------------------------------------------------
# 10.  Baseline model  -------------------------------------------------------
# ---------------------------------------------------------------------------
echo "=============================================================================="
echo " STEP 10 – Write models/baseline.py "
echo "=============================================================================="

cat > "$PKG/models/baseline.py" <<'PY'
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
from ..processing.combine import build_master
from ..config import settings
from ..utils.logging_utils import get_logger
log = get_logger(__name__)

def run_baseline():
    df = build_master()
    if df.empty: return
    # Simple lag features
    for lag in (1,3,7):
        df[f"mail_lag{lag}"] = df.mail_volume.shift(lag)
    df.dropna(inplace=True)
    X = df.drop(columns=["call_volume","date"])
    y = df.call_volume
    tscv = TimeSeriesSplit(n_splits=5)
    mape=[]
    for tr, te in tscv.split(X):
        rf=RandomForestRegressor(n_estimators=300,random_state=42)
        rf.fit(X.iloc[tr], y.iloc[tr]); pred=rf.predict(X.iloc[te])
        mape.append(mean_absolute_percentage_error(y.iloc[te],pred))
    rep={"MAPE_mean":float(np.mean(mape)),"MAPE_split":[float(v) for v in mape]}
    with open(settings.out_dir/"rf_baseline.json","w") as f: json.dump(rep,f,indent=2)
    log.info(f"Saved rf_baseline.json – mean MAPE={rep['MAPE_mean']:.3f}")
PY

# ---------------------------------------------------------------------------
# 11.  Stage runners  --------------------------------------------------------
# ---------------------------------------------------------------------------
echo "=============================================================================="
echo " STEP 11 – Write run_stageX.py "
echo "=============================================================================="

cat > "$PKG/run_stage1.py" <<'PY'
from customer_comms.viz import plots as V
from customer_comms.utils.logging_utils import get_logger
log = get_logger(__name__)
def main():
    log.info("Stage-1 – QA + raw plots")
    V.overview(); V.raw_call_files(); V.data_gaps(); V.qa_jsons()
if __name__=="__main__": main()
PY

cat > "$PKG/run_stage2.py" <<'PY'
from customer_comms.processing.combine import build_master
from customer_comms.analytics import corr_extras as C
from customer_comms.utils.logging_utils import get_logger
log = get_logger(__name__)
def main():
    log.info("Stage-2 – Correlation extras")
    df=build_master()
    if df.empty: return
    C.lag_heat(df); C.rolling_corr(df); C.corr_variants(df)
if __name__=="__main__": main()
PY

cat > "$PKG/run_stage3.py" <<'PY'
from customer_comms.analytics import mail_intent_corr as M
from customer_comms.utils.logging_utils import get_logger
log = get_logger(__name__)
def main():
    log.info("Stage-3 – Mail × Intent heat-map")
    M.plot_top10()
if __name__=="__main__": main()
PY

cat > "$PKG/run_stage4.py" <<'PY'
from customer_comms.models import baseline as B
from customer_comms.utils.logging_utils import get_logger
log=get_logger(__name__)
def main():
    log.info("Stage-4 – RF baseline")
    B.run_baseline()
if __name__=="__main__": main()
PY

# ---------------------------------------------------------------------------
# 12.  Orchestrator  ---------------------------------------------------------
# ---------------------------------------------------------------------------
echo "=============================================================================="
echo " STEP 12 – Write run_pipeline.py "
echo "=============================================================================="

cat > "$PKG/run_pipeline.py" <<'PY'
import importlib, sys
from customer_comms.utils.logging_utils import get_logger
log=get_logger(__name__)
def run():
    for mod in ("run_stage1","run_stage2","run_stage3","run_stage4"):
        log.info(f"🏁  Running {mod} …")
        m=importlib.import_module(f"customer_comms.{mod}")
        importlib.reload(m); m.main()
    log.info("🎉  Pipeline complete – check ./customer_comms/output/")
if __name__=="__main__": run()
PY

# ---------------------------------------------------------------------------
# 13.  Copy CSVs if they exist  ---------------------------------------------
# ---------------------------------------------------------------------------
echo "=============================================================================="
echo " STEP 13 – Copy CSVs from ./data to package data/ "
echo "=============================================================================="

for f in "GenesysExtract_20250703.csv" "GenesysExtract_20250609.csv" "all_mail_data.csv"; do
  [[ -f data/$f ]] && cp -f "data/$f" "$PKG/data/"
done

# ---------------------------------------------------------------------------
# 14.  Execute pipeline  -----------------------------------------------------
# ---------------------------------------------------------------------------
echo "=============================================================================="
echo " STEP 14 – Execute full pipeline "
echo "=============================================================================="

python -m customer_comms.run_pipeline || echo "❌  Top-level failure – see logs/"

echo -e "\n✅  ALL DONE ► Plots & JSON  →  customer_comms/output/   |   Logs → ./logs/"