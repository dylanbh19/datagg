#!/usr/bin/env bash
# ------------------------------------------------------------------
#  three.sh  –  Stage-3  (Advanced correlation plots & QC)
#  • Windows-friendly, no venv, self-healing imports
#  • Fixes previous NaN / empty-image issues
# ------------------------------------------------------------------
set -euo pipefail
export PYTHONUTF8=1                           # UTF-8 everywhere
export PYTHONWARNINGS="ignore::RuntimeWarning"

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$PROJ_ROOT:$PYTHONPATH"

LOG_DIR="$PROJ_ROOT/logs"; mkdir -p "$LOG_DIR"
OUT_DIR="$PROJ_ROOT/output"; mkdir -p "$OUT_DIR"
LOG_FILE="$LOG_DIR/stage-3_$(date +%Y%m%d).log"

echo "🚀 Stage-3  –  advanced correlations & QC"
echo "📜 Log → $LOG_FILE"
echo "-------------------------------------------------------------"

python - <<'PY' 2>&1 | tee -a "$LOG_FILE"
import sys, traceback, json, warnings, os, importlib
from pathlib import Path
warnings.filterwarnings("ignore", category=RuntimeWarning)

# -----------------------------------------------------------------
#  Safety import – make sure package is importable
# -----------------------------------------------------------------
PKG = "customer_comms"
if PKG not in sys.modules:
    sys.path.insert(0, str(Path.cwd()))
try:
    cfg_mod = importlib.import_module(f"{PKG}.config")
except ModuleNotFoundError as e:
    print(f"❌  Could not import package '{PKG}'. Run ./one.sh and ./two.sh first.")
    sys.exit(1)

settings = cfg_mod.settings
OUT_DIR = settings.output_dir
LOG = importlib.import_module(f"{PKG}.logging_utils").get_logger("stage3")

# -----------------------------------------------------------------
#  Ensure feature-engineering & plotting helpers exist  -------------
# -----------------------------------------------------------------
# (Re-write minimal helpers in-place so they’re always present)
utils_dir = Path(PKG, "utils"); utils_dir.mkdir(exist_ok=True)

(Path(utils_dir, "clean.py")).write_text("""
import numpy as np
def safe_zscore(s):
    if s.std(ddof=0)==0 or s.isna().all(): return s*0
    return (s-s.mean())/s.std(ddof=0)
""", encoding="utf-8")

(Path(PKG, "advanced")).mkdir(exist_ok=True)

(Path(PKG, "advanced", "__init__.py")).touch()

# -----------------------------------------------------------------
#  Build dataset (weekday only, with augmented calls)  --------------
# -----------------------------------------------------------------
from customer_comms.processing.combine import build_dataset

df = build_dataset()        # already logs size / coverage

if df.empty or len(df) < 40:           # need at least ~2 months of weekdays
    LOG.error("Dataset too small for Stage-3 analysis – skipping plots.")
    sys.exit(0)

# -----------------------------------------------------------------
#  Correlation helpers  -------------------------------------------
# -----------------------------------------------------------------
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import pearsonr
from matplotlib.dates import DateFormatter
plt.style.use("default")
FMT = DateFormatter("%Y-%m-%d")

def lag_corr_heat(df, max_lag=21, min_samples=25):
    vals = []
    lags = list(range(max_lag+1))
    x = df["mail_volume"]
    y = df["call_volume"]

    for lag in lags:
        aligned = pd.DataFrame(
            {"x": x, "y": y.shift(-lag)}
        ).dropna()
        if len(aligned) < min_samples or aligned["x"].std()==0 or aligned["y"].std()==0:
            vals.append(np.nan)
        else:
            vals.append(pearsonr(aligned["x"], aligned["y"])[0])

    if np.isnan(vals).all():
        LOG.warning("Lag-heatmap skipped – not enough valid data")
        return

    plt.figure(figsize=(10,3))
    sns.heatmap(
        np.array(vals).reshape(1,-1),
        annot=True, fmt=".2f", cmap="vlag", center=0,
        xticklabels=lags, yticklabels=["r"], cbar=False
    )
    plt.title("Mail→Call correlation vs lag")
    plt.xlabel("Lag (days)")
    plt.tight_layout()
    fn = OUT_DIR / "05_lag_corr_heat.png"
    plt.savefig(fn, dpi=300); plt.close()
    LOG.info(f"Saved {fn}")

def rolling_corr(df, window=30, min_samples=window+10):
    if len(df) < min_samples:
        LOG.warning("Rolling-corr skipped – not enough rows")
        return
    r = (df.set_index("date")["mail_volume"]
            .rolling(window, min_periods=int(window*0.7))
            .corr(df.set_index("date")["call_volume"]))
    if r.dropna().empty:
        LOG.warning("Rolling-corr all NaN – skipped")
        return
    plt.figure(figsize=(12,4))
    plt.plot(r.index, r, lw=1)
    plt.title(f"{window}-day rolling Mail↔Call correlation")
    plt.grid(ls="--", alpha=.4)
    ax = plt.gca(); ax.xaxis.set_major_formatter(FMT)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    fn = OUT_DIR / "06_rolling_corr.png"
    plt.savefig(fn, dpi=300); plt.close()
    LOG.info(f"Saved {fn}")

# -----------------------------------------------------------------
#  Execute & log  --------------------------------------------------
# -----------------------------------------------------------------
try:
    lag_corr_heat(df)
    rolling_corr(df)
    LOG.info("🎉  Stage-3 complete – plots written to ./output/")
except Exception as e:
    LOG.error(f"Stage-3 failed: {e}")
    traceback.print_exc()
    sys.exit(1)
PY

echo "✅  Stage-3 artefacts generated in ./output/ | logs in ./logs/"
echo "-------------------------------------------------------------"