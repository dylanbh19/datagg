#!/usr/bin/env bash
# =============================================================================
# 03_modelling.sh  –  Stage-3  (feature eng + model-readiness visual pack)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"          # project root

# ---------- 0)  auto-install deps -------------------------------------------
python - <<'PY'
import importlib, subprocess, sys, pathlib, textwrap, json, warnings
pkgs = ["pandas","numpy","matplotlib","seaborn","statsmodels","scipy","sklearn"]
for p in pkgs:
    try: importlib.import_module(p)
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable,"-m","pip","install","-q",p])

# ---------- 1)  self-healing helper module ----------------------------------
utils_dir = pathlib.Path("customer_comms/utils"); utils_dir.mkdir(parents=True, exist_ok=True)
helper = utils_dir/"clean.py"
if not helper.exists():
    helper.write_text(textwrap.dedent("""\
        import numpy as np, pandas as pd
        def safe_zscore(arr):
            arr = np.asarray(arr, dtype='float64')
            std = np.nanstd(arr); mean = np.nanmean(arr)
            return np.zeros_like(arr) if std==0 or np.isnan(std) else (arr-mean)/std
        def drop_all_nan(df):
            return df.dropna(axis=0,how='all').dropna(axis=1,how='all')
        """), encoding="utf-8")

# ---------- 2)  run Stage-3 driver ------------------------------------------
driver = r"""
import warnings, json, itertools as it, numpy as np, pandas as pd, seaborn as sns, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from customer_comms.logging_utils import get_logger
from customer_comms.processing.combine import build_dataset
from customer_comms.data.loader     import load_call_data, load_mail
from customer_comms.utils.clean     import safe_zscore, drop_all_nan
from customer_comms.viz.plots       import save_lag_corr_heat, save_rolling_corr   # <- Stage-2 funcs
from statsmodels.stats.outliers_inflation import variance_inflation_factor

warnings.filterwarnings("ignore", category=RuntimeWarning)
out = Path("output"); out.mkdir(exist_ok=True)
log = get_logger("stage3")

# ===== 2.0 dataset ===========================================================
df = build_dataset()
if df.empty:
    log.error("Stage-3 aborted – combined dataset empty"); raise SystemExit(1)

# ===== 2.1 re-create lag & rolling corr plots (keep artefacts together) ======
try:
    save_lag_corr_heat(df, out / "05_lag_corr_heat.png")
    save_rolling_corr(df, out / "06_rolling_corr.png")
except Exception as e:
    log.warning("Lag/Rolling corr plots skipped: %s", e)

# ===== 2.2 feature engineering ==============================================
feat = pd.DataFrame(index=df["date"])
feat["call_lag1"]  = df["call_volume"].shift(1)
feat["mail_lag1"]  = df["mail_volume"].shift(1)
feat["call_ma7"]   = df["call_volume"].rolling(7).mean()
feat["mail_ma7"]   = df["mail_volume"].rolling(7).mean()
feat["call_norm"]  = safe_zscore(df["call_volume"])
feat["mail_norm"]  = safe_zscore(df["mail_volume"])
feat["ratio"]      = np.where(df["mail_volume"]>0,
                              df["call_volume"]/df["mail_volume"], np.nan)
feat = feat.dropna().replace([np.inf,-np.inf], np.nan).fillna(0)
feat.to_parquet(out/"07_feat_matrix.parquet")
log.info("Feature matrix saved (rows=%s, cols=%s)", *feat.shape)

# ===== 2.3 top correlations bar =============================================
corr = feat.corr().abs()
pairs = (corr.where(np.triu(np.ones(corr.shape),1).astype(bool))
              .stack()
              .sort_values(ascending=False)
              .head(20))
pairs.to_csv(out/"08_top_corr_pairs.csv")
plt.figure(figsize=(9,4)); pairs.plot.bar()
plt.title("Top |r| feature correlations"); plt.tight_layout()
plt.savefig(out/"08_top_corr.png", dpi=300); plt.close()

# ===== 2.4 VIF heat-map ======================================================
try:
    X = feat.assign(const=1)
    vif = pd.DataFrame({"vif":[variance_inflation_factor(X.values,i)
                               for i in range(X.shape[1])]},
                       index=X.columns)
    sns.heatmap(vif.T, annot=True, cmap="Reds"); plt.title("VIF")
    plt.tight_layout(); plt.savefig(out/"09_vif_heat.png", dpi=300); plt.close()
except Exception as e:
    log.warning("VIF skipped: %s", e)

# ===== 2.5 feature-pair scatter-grid =========================================
try:
    cols = ["call_volume","mail_volume","call_ma7","mail_ma7","ratio"]
    sample = df[cols].dropna().sample(n=min(2000,len(df)), random_state=1)
    sns.pairplot(sample, diag_kind="kde", corner=True); plt.tight_layout()
    plt.savefig(out/"11_scatter_matrix.png", dpi=250); plt.close()
except Exception as e:
    log.warning("Pair-plot skipped: %s", e)

# ===== 2.6 model-readiness report ===========================================
ready = {
    "rows"          : len(feat),
    "columns"       : list(feat.columns),
    "pct_missing"   : float(feat.isna().mean().mean()),
    "max_vif"       : float(vif["vif"].max()) if 'vif' in locals() else None,
    "top5_corr_abs" : pairs.iloc[:5].round(3).to_dict()
}
json.dump(ready, open(out/"12_model_ready.json","w",encoding="utf-8"), indent=2)
log.info("Saved 12_model_ready.json")

log.info("🎉 Stage-3 complete – artefacts refreshed in ./output/")
"""
import runpy, textwrap, sys, os
exec(compile(textwrap.dedent(driver), "<stage3>", "exec"))
PY

echo "-------------------------------------------------------------"
echo "✅ Stage-3 artefacts generated in ./output/ | logs in ./logs/"
echo "-------------------------------------------------------------"