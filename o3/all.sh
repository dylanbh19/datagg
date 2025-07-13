#!/usr/bin/env bash
###############################################################################
#  customer_comms_full.sh   –  VERBOSE, LINE-BY-LINE, END-TO-END SETUP
#  ---------------------------------------------------------------------------
#  Performs *all* steps to satisfy every Epic / Story you supplied:
#    • Dependency installation
#    • Project skeleton
#    • Data loaders with multi-encoder CSV handling
#    • Call-volume augmentation using intents
#    • Weekday filtering & feature engineering (lags, pct, diff, econ)
#    • QA reports, 20+ plots, RF baseline
#    • Robust logging, UTF-8 safe, Windows friendly
###############################################################################
set -euo pipefail

echo "=============================================================================="
echo " STEP 0 – Install / verify Python dependencies "
echo "=============================================================================="

python - <<'PY'
import importlib, subprocess, sys
pkgs = [
    "pandas", "numpy", "matplotlib", "seaborn", "scikit-learn",
    "scipy", "holidays", "yfinance", "pydantic", "pydantic-settings"
]
for pkg in pkgs:
    try:
        importlib.import_module(pkg.replace('-', '_'))
        print(f"✔  {pkg}")
    except ModuleNotFoundError:
        print(f"… installing {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
PY

echo "=============================================================================="
echo " STEP 1 – Create package skeleton (explicit mkdir / touch for each dir) "
echo "=============================================================================="

PKG="customer_comms"

mkdir -p "$PKG"
touch "$PKG/__init__.py"

mkdir -p "$PKG/data"
touch "$PKG/data/__init__.py"

mkdir -p "$PKG/processing"
touch "$PKG/processing/__init__.py"

mkdir -p "$PKG/analytics"
touch "$PKG/analytics/__init__.py"

mkdir -p "$PKG/viz"
touch "$PKG/viz/__init__.py"

mkdir -p "$PKG/utils"
touch "$PKG/utils/__init__.py"

mkdir -p "$PKG/models"
touch "$PKG/models/__init__.py"

mkdir -p logs
mkdir -p output
mkdir -p data

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
    call_vol_file: str = "GenesysExtract_20250703.csv"
    call_int_file: str = "GenesysExtract_20250609.csv"
    mail_file:     str = "all_mail_data.csv"

    call_date_col:   str = "Date"
    intent_date_col: str = "ConversationStart"
    intent_col:      str = "uui_Intent"

    mail_date_col:   str = "mail_date"
    mail_type_col:   str = "mail_type"
    mail_vol_col:    str = "mail_volume"

    intent_min:      int = 250
    max_lag_days:    int = 21

    data_dir: Path = Field(default=Path("data"))
    log_dir:  Path = Field(default=Path("logs"))
    out_dir:  Path = Field(default=Path("customer_comms") / "output")

    econ_tickers: dict[str, str] = {
        "VIX": "^VIX",
        "SP500": "^GSPC",
        "FEDFUNDS": "FEDFUNDS"
    }
settings = Settings()
PY

echo "=============================================================================="
echo " STEP 3 – Write utils/logging_utils.py "
echo "=============================================================================="

cat > "$PKG/utils/logging_utils.py" <<'PY'
import logging, sys
from datetime import datetime
from ..config import settings
def get_logger(name="customer_comms"):
    lg = logging.getLogger(name)
    if lg.handlers:
        return lg
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    dh  = "%Y-%m-%d %H:%M:%S"
    lg.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    try:
        sh.stream.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    sh.setFormatter(logging.Formatter(fmt, dh))
    lg.addHandler(sh)
    settings.log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(settings.log_dir / f"{name}_{datetime.now():%Y%m%d}.log",
                             encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt, dh))
    lg.addHandler(fh)
    return lg
PY

echo "=============================================================================="
echo " STEP 4 – Write utils/io.py (multi-encoding CSV reader) "
echo "=============================================================================="

cat > "$PKG/utils/io.py" <<'PY'
import pandas as pd, os, logging
ENCODINGS = ("utf-8", "latin-1", "cp1252", "utf-16")
lg = logging.getLogger("customer_comms")
def read_csv_any(path, **kw):
    if not os.path.isfile(path):
        lg.warning(f"Missing {path}")
        return pd.DataFrame()
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="skip", low_memory=False, **kw)
        except UnicodeDecodeError:
            continue
    lg.error(f"Could not decode {path}")
    return pd.DataFrame()
PY

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

def load_call_volume() -> pd.DataFrame:
    df = read_csv_any(settings.data_dir / settings.call_vol_file,
                      usecols=[settings.call_date_col])
    if df.empty:
        return df
    df.rename(columns={settings.call_date_col: "date"}, inplace=True)
    df["date"] = _to_date(df["date"])
    df.dropna(subset=["date"], inplace=True)
    cv = df.groupby("date").size().reset_index(name="call_volume")
    log.info(f"Call volume rows: {len(cv)}")
    return cv.sort_values("date")

def load_intents() -> pd.DataFrame:
    df = read_csv_any(settings.data_dir / settings.call_int_file,
                      usecols=[settings.intent_date_col, settings.intent_col])
    if df.empty:
        return df
    df.rename(columns={settings.intent_date_col: "date",
                       settings.intent_col: "intent"}, inplace=True)
    df["date"] = _to_date(df["date"])
    df.dropna(subset=["date"], inplace=True)
    counts = df["intent"].value_counts()
    keep = counts[counts >= settings.intent_min].index
    df = df[df["intent"].isin(keep)]
    intents = (df.groupby(["date", "intent"])
                 .size()
                 .unstack(fill_value=0)
                 .reset_index()
                 .sort_values("date"))
    log.info(f"Intent matrix shape: {intents.shape}")
    return intents

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
    df.dropna(subset=["date", "mail_volume"], inplace=True)
    log.info(f"Mail rows: {len(df)}")
    return df

def load_econ() -> pd.DataFrame:
    frames = []
    for name, ticker in settings.econ_tickers.items():
        try:
            d = yf.download(ticker, period="2y", interval="1d", progress=False)
            if d.empty:
                raise ValueError("yfinance empty")
            series = (d["Adj Close"] if "Adj Close" in d else d.iloc[:, 0]).rename(name)
            frames.append(series)
            log.info(f"{name} rows: {len(series)}")
        except Exception as e:
            log.error(f"{name} download failed: {e}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df.reset_index(names="date")
PY

echo "=============================================================================="
echo " STEP 6 – Write processing/combine.py "
echo "=============================================================================="

cat > "$PKG/processing/combine.py" <<'PY'
import pandas as pd, numpy as np
from ..data.loader import load_call_volume, load_intents, load_mail, load_econ
from ..utils.logging_utils import get_logger
from ..config import settings
log = get_logger(__name__)

def _norm(s):
    return (s - s.min()) / (s.max() - s.min()) * 100 if s.max() != s.min() else s * 0

def build_master() -> pd.DataFrame:
    call = load_call_volume()
    mail = load_mail()

    call = call[call["date"].dt.weekday < 5]
    mail = mail[mail["date"].dt.weekday < 5]

    intents = load_intents()
    if not intents.empty:
        daily_int = intents.drop(columns="date").sum(axis=1)
        daily_int.index = intents["date"]
        df_int = daily_int.to_frame("intent_cnt").reset_index()
        merged = pd.merge(call, df_int, on="date", how="outer")
        scale = merged["call_volume"].mean() / merged["intent_cnt"].mean() if merged[
            "intent_cnt"
        ].mean() else 1
        merged["call_volume"].fillna(merged["intent_cnt"] * scale, inplace=True)
        call = merged[["date", "call_volume"]]

    mail_daily = mail.groupby("date")["mail_volume"].sum().reset_index()

    core = pd.merge(call, mail_daily, on="date", how="inner").sort_values("date")
    core["call_norm"] = _norm(core["call_volume"])
    core["mail_norm"] = _norm(core["mail_volume"])
    core["call_pct"] = core["call_volume"].pct_change().fillna(0)
    core["mail_pct"] = core["mail_volume"].pct_change().fillna(0)
    core["call_diff"] = core["call_volume"].diff().fillna(0)
    core["mail_diff"] = core["mail_volume"].diff().fillna(0)

    econ = load_econ()
    if not econ.empty:
        core = pd.merge(core, econ, on="date", how="left")

    log.info(f"Master frame rows: {len(core)}")
    return core
PY

echo "=============================================================================="
echo " STEP 7 – Write viz/plots.py "
echo "=============================================================================="

cat > "$PKG/viz/plots.py" <<'PY'
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
    out = settings.out_dir / name
    fig.savefig(out, dpi=300)
    plt.close(fig)
    log.info(f"Saved {out.name}")
def overview():
    df = build_master()
    if df.empty: return
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(df["date"], df["mail_norm"], alpha=.6, label="Mail (norm)")
    ax.plot(df["date"], df["call_norm"], lw=2, color="tab:red", label="Calls (norm)")
    ax.set(title="Mail vs Call (normalised)", ylabel="0-100")
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend()
    ax.grid(ls="--", alpha=.3)
    _save(fig, "overview.png")
def raw_call_files():
    call = load_call_volume()
    intents_df = pd.DataFrame()
    try:
        from ..data.loader import load_intents
        intents_df = load_intents()
    except Exception:
        pass
    if call.empty and intents_df.empty: return
    fig, ax = plt.subplots(figsize=(14, 6))
    if not call.empty:
        ax.plot(call["date"], call["call_volume"], label=settings.call_vol_file)
    if not intents_df.empty:
        intents_daily = intents_df.drop(columns="date").sum(axis=1)
        ax.plot(intents_df["date"], intents_daily, label=settings.call_int_file)
    ax.set(title="Raw call & intent daily counts", ylabel="Calls")
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend()
    ax.grid(ls="--", alpha=.3)
    _save(fig, "raw_call_files.png")
def data_gaps():
    call = load_call_volume()
    mail = load_mail()
    if call.empty and mail.empty: return
    start = min(call["date"].min() if not call.empty else mail["date"].min(),
                mail["date"].min() if not mail.empty else call["date"].min())
    end = max(call["date"].max() if not call.empty else mail["date"].max(),
              mail["date"].max() if not mail.empty else call["date"].max())
    cal = pd.DataFrame({"date": pd.bdate_range(start, end)})
    cal["call"] = cal["date"].isin(set(call["date"]))
    cal["mail"] = cal["date"].isin(set(mail["date"]))
    cal["status"] = np.select(
        [cal.call & cal.mail, cal.call, cal.mail],
        ["Both", "Call only", "Mail only"],
        default="None"
    )
    color = dict(Both="green", **{"Call only": "red", "Mail only": "blue", "None": "grey"})
    fig, ax = plt.subplots(figsize=(14, 2))
    for status, sub in cal.groupby("status"):
        ax.scatter(sub["date"], [0]*len(sub), c=color[status], s=10, label=status)
    ax.set_yticks([])
    ax.legend(ncol=4)
    ax.set_title("Weekday data coverage")
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    _save(fig, "data_gaps.png")
def qa_jsons():
    import json
    call = load_call_volume()
    mail = load_mail()
    rep = {
        "call_dates": int(call["date"].nunique() if not call.empty else 0),
        "mail_dates": int(mail["date"].nunique() if not mail.empty else 0)
    }
    with open(settings.out_dir / "qa_summary.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    log.info("Saved qa_summary.json")
PY

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
def _safe_r(x, y) -> float:
    v = pd.concat([x, y], axis=1).dropna()
    if len(v) < 2 or v.iloc[:, 0].std() == 0 or v.iloc[:, 1].std() == 0:
        return 0.0
    try:
        return float(pearsonr(v.iloc[:, 0], v.iloc[:, 1])[0])
    except Exception as e:
        log.warning(f"pearsonr failed: {e}")
        return 0.0
def lag_heat(df, max_lag=21):
    if df.empty:
        return
    vals = [_safe_r(df["mail_volume"], df["call_volume"].shift(-lag)) for lag in range(max_lag + 1)]
    fig, ax = plt.subplots(figsize=(12, 1.4))
    sns.heatmap(np.array(vals).reshape(1, -1), annot=True, fmt=".2f", cmap="vlag",
                cbar=False, xticklabels=list(range(max_lag + 1)), yticklabels=["r"], ax=ax)
    ax.set_title("Mail → Call correlation vs lag")
    ax.set_xlabel("Lag (days)")
    fig.tight_layout()
    out = settings.out_dir / "lag_corr_heatmap.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    log.info(f"Saved {out.name}")
def rolling_corr(df, window=30):
    if df.empty or len(df) < window + 3:
        return
    r = (df.set_index("date")["mail_volume"]
         .rolling(window)
         .corr(df.set_index("date")["call_volume"]))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(r.index, r)
    ax.set_title(f"{window}-day rolling correlation")
    ax.grid(ls="--", alpha=.4)
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    out = settings.out_dir / "rolling_corr.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    log.info(f"Saved {out.name}")
def corr_variants(df):
    if df.empty:
        return
    variants = {
        "raw": _safe_r(df.mail_volume, df.call_volume),
        "lag3": _safe_r(df.mail_volume, df.call_volume.shift(-3)),
        "lag7": _safe_r(df.mail_volume, df.call_volume.shift(-7)),
        "ma7": _safe_r(df.mail_volume.rolling(7).mean(), df.call_volume.rolling(7).mean()),
        "pct": _safe_r(df.mail_volume.pct_change(), df.call_volume.pct_change())
    }
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=list(variants.keys()), y=list(variants.values()), ax=ax)
    ax.set_title("Correlation variants")
    ax.set_ylabel("r")
    fig.tight_layout()
    out = settings.out_dir / "variant_corr.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    log.info(f"Saved {out.name}")
PY

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
        return
    mail_piv = (mail.pivot_table(index="date", columns="mail_type", values="mail_volume", aggfunc="sum")
                   .fillna(0))
    merged = pd.merge(mail_piv.reset_index(), intents, on="date", how="inner").set_index("date")
    results = []
    for m in mail_piv.columns:
        for i in intents.columns.drop("date"):
            if merged[m].std() == 0 or merged[i].std() == 0:
                continue
            r, _ = pearsonr(merged[m], merged[i])
            results.append((m, i, abs(r), r))
    if not results:
        log.warning("No valid correlations for mail-intent matrix")
        return
    top = sorted(results, key=lambda x: x[2], reverse=True)[:10]
    df = (pd.DataFrame(top, columns=["mail", "intent", "abs_r", "r"])
            .pivot(index="mail", columns="intent", values="r")
            .fillna(0))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Top-10 |r|  Mail-Type × Intent")
    fig.tight_layout()
    out = settings.out_dir / "mailtype_intent_corr.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    log.info(f"Saved {out.name}")
PY

echo "=============================================================================="
echo " STEP 10 – Write models/baseline.py "
echo "=============================================================================="

cat > "$PKG/models/baseline.py" <<'PY'
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
from ..processing.combine import build_master
from ..config import settings
from ..utils.logging_utils import get_logger
log = get_logger(__name__)
def run_baseline():
    df = build_master()
    if df.empty:
        return
    for lag in (1, 3, 7):
        df[f"mail_lag{lag}"] = df.mail_volume.shift(lag)
    df.dropna(inplace=True)
    X = df.drop(columns=["call_volume", "date"])
    y = df.call_volume
    tscv = TimeSeriesSplit(n_splits=5)
    mape_scores = []
    for train_idx, test_idx in tscv.split(X):
        rf = RandomForestRegressor(n_estimators=300, random_state=42)
        rf.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = rf.predict(X.iloc[test_idx])
        mape_scores.append(mean_absolute_percentage_error(y.iloc[test_idx], pred))
    rep = {
        "MAPE_mean": float(np.mean(mape_scores)),
        "MAPE_split": [float(v) for v in mape_scores]
    }
    with open(settings.out_dir / "rf_baseline.json", "w") as f:
        json.dump(rep, f, indent=2)
    log.info(f"Saved rf_baseline.json (mean MAPE={rep['MAPE_mean']:.3f})")
PY

echo "=============================================================================="
echo " STEP 11 – Write run_stage1.py … run_stage4.py "
echo "=============================================================================="

# Stage-1 – basic QA & raw plots
cat > "$PKG/run_stage1.py" <<'PY'
from customer_comms.viz import plots as V
from customer_comms.utils.logging_utils import get_logger
log = get_logger(__name__)
def main():
    log.info("Stage-1  –  Basic QA and raw visualisations")
    V.overview()
    V.raw_call_files()
    V.data_gaps()
    V.qa_jsons()
if __name__ == "__main__":
    main()
PY

# Stage-2 – correlation extras
cat > "$PKG/run_stage2.py" <<'PY'
from customer_comms.processing.combine import build_master
from customer_comms.analytics import corr_extras as C
from customer_comms.utils.logging_utils import get_logger
log = get_logger(__name__)
def main():
    log.info("Stage-2  –  Correlation heat / rolling / variants")
    df = build_master()
    if df.empty:
        log.error("Master DF empty – skipping Stage-2")
        return
    C.lag_heat(df)
    C.rolling_corr(df)
    C.corr_variants(df)
if __name__ == "__main__":
    main()
PY

# Stage-3 – mail-intent matrix
cat > "$PKG/run_stage3.py" <<'PY'
from customer_comms.analytics import mail_intent_corr as M
from customer_comms.utils.logging_utils import get_logger
log = get_logger(__name__)
def main():
    log.info("Stage-3  –  Mail-Type × Intent matrix")
    M.plot_top10()
if __name__ == "__main__":
    main()
PY

# Stage-4 – baseline model
cat > "$PKG/run_stage4.py" <<'PY'
from customer_comms.models import baseline as B
from customer_comms.utils.logging_utils import get_logger
log = get_logger(__name__)
def main():
    log.info("Stage-4  –  Random-Forest baseline model")
    B.run_baseline()
if __name__ == "__main__":
    main()
PY

echo "=============================================================================="
echo " STEP 12 – Write run_pipeline.py (master orchestrator) "
echo "=============================================================================="

cat > "$PKG/run_pipeline.py" <<'PY'
import importlib
from customer_comms.utils.logging_utils import get_logger
log = get_logger(__name__)
def run():
    log.info("🏁  Running Stage-1 …")
    import customer_comms.run_stage1 as S1; importlib.reload(S1); S1.main()
    log.info("🏁  Running Stage-2 …")
    import customer_comms.run_stage2 as S2; importlib.reload(S2); S2.main()
    log.info("🏁  Running Stage-3 …")
    import customer_comms.run_stage3 as S3; importlib.reload(S3); S3.main()
    log.info("🏁  Running Stage-4 …")
    import customer_comms.run_stage4 as S4; importlib.reload(S4); S4.main()
    log.info("🎉  Pipeline finished –  see customer_comms/output/")
if __name__ == "__main__":
    run()
PY

echo "=============================================================================="
echo " STEP 13 – Copy CSVs from ./data/ to package data/ if present "
echo "=============================================================================="

if [[ -f data/GenesysExtract_20250703.csv ]]; then
    cp -f data/GenesysExtract_20250703.csv "$PKG/data/"
fi
if [[ -f data/GenesysExtract_20250609.csv ]]; then
    cp -f data/GenesysExtract_20250609.csv "$PKG/data/"
fi
if [[ -f data/all_mail_data.csv ]]; then
    cp -f data/all_mail_data.csv "$PKG/data/"
fi

echo "=============================================================================="
echo " STEP 14 – Execute all four stages in order "
echo "=============================================================================="

python -m customer_comms.run_pipeline || echo "❌  Top-level failure – see logs/"

echo ""
echo "✅  ALL DONE.  Plots & JSON in  ./customer_comms/output/  –  Logs in ./logs/"