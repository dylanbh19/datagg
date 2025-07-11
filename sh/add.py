#!/usr/bin/env bash
# =====================================================================
#  advanced_updates.sh   (bolt-on to the customer_comms scaffold)
#  • Augment call-volume to match intent span & scale
#  • Three feature-engineering tricks → stronger correlations
#  • Extra visualisations
#  • New pipeline4.py – produces 15 artefacts
# =====================================================================
set -euo pipefail
PKG="customer_comms"

echo "🔧  Adding advanced augmentation & feature modules …"
mkdir -p "${PKG}/analytics" "${PKG}/features" "${PKG}/viz"

# ---------------------------------------------------------------------
# 0.  Config additions
# ---------------------------------------------------------------------
python - <<'PY'
from pathlib import Path, re, sys, fileinput
cfg = Path("customer_comms/config.py")
txt = cfg.read_text()
if "augment_gap_limit" not in txt:
    ins = "\n    augment_gap_limit: int = 3  # ≤ n business-days forward-fill\n" \
          "    augment_intent_weight: bool = True  # scale to avg intent/call ratio\n"
    txt = txt.replace("max_lag:", ins+"    max_lag:")
    cfg.write_text(txt)
    print("✅  config.py patched")
else:
    print("ℹ️  config.py already patched")
PY

# ---------------------------------------------------------------------
# 1.  analytics/augment.py
# ---------------------------------------------------------------------
cat > "${PKG}/analytics/augment.py" << 'PY'
"""Augment call-volume so its shape matches the intent series better."""
import pandas as pd, numpy as np
from ..config import settings
from ..logging_utils import get_logger
log = get_logger(__name__)

def augment_call_volume(call_vol: pd.DataFrame,
                        intents: pd.DataFrame) -> pd.DataFrame:
    """Return call_vol with a new column call_volume_aug."""
    if intents.empty or call_vol.empty:
        call_vol["call_volume_aug"] = call_vol["call_volume"]
        return call_vol

    # 1) forward‐fill gaps ≤ cfg.augment_gap_limit
    cv = (call_vol.set_index("date")
                  .asfreq("B")
                  .fillna(method="ffill", limit=settings.augment_gap_limit))

    # 2) weekday median smoothing
    wd_median = cv.groupby(cv.index.weekday)["call_volume"].transform("median")
    cv["call_volume_aug"] = wd_median

    # 3) global scaling so total matches intent total if wanted
    if settings.augment_intent_weight and "date" in intents.columns:
        intent_total = intents.drop(columns=["date"]).to_numpy().sum()
        call_total   = cv["call_volume_aug"].sum()
        if call_total > 0:
            cv["call_volume_aug"] *= intent_total / call_total

    cv = cv.reset_index()
    log.info("Augmented call_volume ({} rows)".format(len(cv)))
    return cv[["date", "call_volume", "call_volume_aug"]]
PY

# ---------------------------------------------------------------------
# 2.  features/enhance.py
# ---------------------------------------------------------------------
cat > "${PKG}/features/enhance.py" << 'PY'
"""Three quick feature-engineering transforms for correlation boosts."""
import pandas as pd, numpy as np
from ..logging_utils import get_logger
log = get_logger(__name__)

def add_fe_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("mail_volume", "call_volume_aug"):
        if col not in out.columns: continue
        out[f"{col}_log"] = np.log1p(out[col])
        out[f"{col}_ma7"] = out[col].rolling(7, min_periods=1).mean()
        out[f"{col}_z"]   = (out[col] - out[col].mean()) / out[col].std(ddof=0)
    log.info("Added FE columns")
    return out
PY

# ---------------------------------------------------------------------
# 3.  viz/aug_overview.py
# ---------------------------------------------------------------------
cat > "${PKG}/viz/aug_overview.py" << 'PY'
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from ..config import settings
from ..logging_utils import get_logger
log = get_logger(__name__)

def save_aug_overview(df, fname="overview_augmented.png"):
    if "call_volume_aug" not in df.columns: return
    plt.figure(figsize=(14,6))
    plt.plot(df["date"], df["call_volume"],      lw=1, alpha=.4,
             label="Call raw", color="tab:red")
    plt.plot(df["date"], df["call_volume_aug"],  lw=2,
             label="Call augmented", color="tab:red")
    plt.bar(df["date"], df["mail_volume"], label="Mail", alpha=.3)
    plt.legend(); plt.grid(ls="--", alpha=.3)
    plt.title("Augmented call-volume vs mail (weekdays)")
    ax=plt.gca(); ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    settings.output_dir.mkdir(exist_ok=True)
    plt.tight_layout(); plt.savefig(settings.output_dir/fname, dpi=300); plt.close()
    log.info(f"Saved {settings.output_dir/fname}")
PY

# ---------------------------------------------------------------------
# 4.  Patch processing/combine.py  – import & call augment
# ---------------------------------------------------------------------
python - <<'PY'
from pathlib import Path, re, sys, textwrap
pp = Path("customer_comms/processing/combine.py")
code = pp.read_text()

if "augment_call_volume" not in code:
    code = code.replace(
        "from ..data.loader import load_call_data, load_mail",
        "from ..data.loader import load_call_data, load_mail\n"
        "from ..analytics.augment import augment_call_volume")
    # insert call after load_call_data()
    pat = re.compile(r"call_vol,_=load_call_data\(\)")
    code = pat.sub("call_vol,intents=load_call_data()\n"
                   "    call_vol = augment_call_volume(call_vol, intents)", code)
    pp.write_text(code)
    print("✅  processing/combine.py patched")
else:
    print("ℹ️  processing/combine.py already patched")
PY

# ---------------------------------------------------------------------
# 5.  Patch viz/raw_calls.py  – overlay augmented series
# ---------------------------------------------------------------------
python - <<'PY'
from pathlib import Path, re, sys
p = Path("customer_comms/viz/raw_calls.py")
txt = p.read_text()
if "call_volume_aug" not in txt:
    txt = txt.replace('plt.plot(sub["date"], sub["calls"], label=f)',
                      'plt.plot(sub["date"], sub["calls"], label=f)\n'
                      '    if "call_volume_aug" in sub.columns:\n'
                      '        plt.plot(sub["date"], sub["call_volume_aug"], '
                      'lw=1, alpha=.4)')
    p.write_text(txt)
    print("✅  viz/raw_calls.py patched")
else:
    print("ℹ️  viz/raw_calls.py already patched")
PY

# ---------------------------------------------------------------------
# 6.  analytics/corr_extras.py – feature_corr_suite()
# ---------------------------------------------------------------------
python - <<'PY'
from pathlib import Path
fn = Path("customer_comms/analytics/corr_extras.py")
txt = fn.read_text()
if "feature_corr_suite" not in txt:
    add = '''
# -----------------------------------------------------------------
def feature_corr_suite(df):
    """Heat-map of Pearson r for engineered features."""
    import seaborn as sns, matplotlib.pyplot as plt, pandas as pd, numpy as np
    feats = [c for c in df.columns if any(s in c for s in ("_log","_ma7","_z"))]
    if not feats: return
    corr = df[["call_volume_aug"]+feats].corr().iloc[1:]
    plt.figure(figsize=(8, min(6,len(feats)*.5)))
    sns.heatmap(corr[["call_volume_aug"]], annot=True, cmap="vlag", center=0, fmt=".2f")
    plt.title("Call-aug vs engineered mail features")
    out = settings.output_dir / "feature_corr_heatmap.png"
    plt.tight_layout(); plt.savefig(out, dpi=300); plt.close()
    log.info(f"Saved {out}")
'''
    txt += add
    fn.write_text(txt)
    print("✅  feature_corr_suite added")
else:
    print("ℹ️  feature_corr_suite already present")
PY

# ---------------------------------------------------------------------
# 7.  pipeline4.py
# ---------------------------------------------------------------------
cat > "${PKG}/pipeline4.py" << 'PY'
from .processing.combine          import build_dataset
from .viz.aug_overview            import save_aug_overview
from .features.enhance            import add_fe_features
from .analytics.corr_extras       import feature_corr_suite
from .pipeline3                   import main as base_pipeline   # keeps all 13 artefacts
from .logging_utils               import get_logger
log=get_logger(__name__)

def main():
    # Run the existing pipeline3 first (13 artefacts)
    base_pipeline()

    # Load the combined dataset with augmented column
    from .processing.combine import build_dataset
    df = build_dataset()
    if df.empty or "call_volume_aug" not in df.columns:
        log.error("Augmented dataset missing – aborting extras"); return

    save_aug_overview(df)
    df_fe = add_fe_features(df)
    feature_corr_suite(df_fe)
    log.info("🚀  Advanced augmentation run complete (15 artefacts).")

if __name__=="__main__":
    main()
PY

echo "✅  Advanced modules added.  Run with:"
echo "   python -m customer_comms.pipeline4"