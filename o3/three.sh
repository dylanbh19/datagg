#!/usr/bin/env bash
# =============================================================================
# 03_modelling.sh  –  Stage-3 : Feature engineering, auto-correlation scan,
#                    model-readiness checks.  Windows-compatible, UTF-8 logs.
# -----------------------------------------------------------------------------
# * Self-roots to the script’s directory
# * Self-heals missing optional deps
# * Cleans NaN / Inf before plots
# * Generates artefacts: 07_feat_matrix.parquet, 08_top_corr.png,
#   09_vif_heat.png, 10_granger_table.csv
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"               # <- project root

# ---------- 0. ensure deps ----------------------------------------------------
python - <<'PY'
import importlib, subprocess, sys, json, pathlib, os, warnings, logging
pkgs = ["pandas","numpy","statsmodels","scipy","seaborn","matplotlib","sklearn"]
for p in pkgs:
    try: importlib.import_module(p)
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable,"-m","pip","install","-q",p])

# ---------- 1. run stage-3 python driver -------------------------------------
code = r"""
import warnings, sys, json, numpy as np, pandas as pd, seaborn as sns, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from customer_comms.logging_utils import get_logger
from customer_comms.processing.combine import build_dataset
from customer_comms.data.loader import load_call_data, load_mail
from customer_comms.utils.clean import safe_zscore if hasattr(__import__('customer_comms.utils'), 'clean') else None

log = get_logger('stage3')

out_dir = Path("output"); out_dir.mkdir(exist_ok=True)

# ---- 1. assemble master DF ---------------------------------------------------
df = build_dataset()
if df.empty:
    log.error("Stage-3 aborted: combined dataset empty"); sys.exit(1)

# ---- 2. Feature engineering --------------------------------------------------
feat = pd.DataFrame(index=df["date"])
feat["call_lag1"]  = df["call_volume"].shift(1)
feat["mail_lag1"]  = df["mail_volume"].shift(1)
feat["call_ma7"]   = df["call_volume"].rolling(7).mean()
feat["mail_ma7"]   = df["mail_volume"].rolling(7).mean()
feat["call_norm"]  = (df["call_volume"]-df["call_volume"].mean())/df["call_volume"].std()
feat["mail_norm"]  = (df["mail_volume"]-df["mail_volume"].mean())/df["mail_volume"].std()
feat["calls_per_mail"] = np.where(df["mail_volume"]>0,
                                  df["call_volume"]/df["mail_volume"], np.nan)
feat = feat.dropna().replace([np.inf,-np.inf], np.nan).fillna(0)

feat.to_parquet(out_dir/"07_feat_matrix.parquet")
log.info("Feature matrix saved (rows=%s, cols=%s)", *feat.shape)

# ---- 3. correlation scan -----------------------------------------------------
corr = feat.corr(method="pearson").abs()
top_pairs = (corr.where(np.triu(np.ones(corr.shape),1).astype(bool))
                  .stack()
                  .sort_values(ascending=False)
                  .head(20))
top_pairs.to_csv(out_dir/"10_top_corr_pairs.csv")
# bar-plot of top correlations
plt.figure(figsize=(8,4)); top_pairs.plot.bar()
plt.title("Top absolute feature correlations"); plt.tight_layout()
plt.savefig(out_dir/"08_top_corr.png", dpi=300); plt.close()
log.info("Saved 08_top_corr.png")

# ---- 4. VIF heat-map (multicollinearity) -------------------------------------
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X = feat.assign(const=1)
    vif = pd.DataFrame({
        "feature": X.columns,
        "vif": [variance_inflation_factor(X.values, i)
                for i in range(X.shape[1])]
    }).set_index("feature")
    plt.figure(figsize=(6,4))
    sns.heatmap(vif.T, annot=True, cmap="Reds"); plt.title("Variance Inflation")
    plt.tight_layout(); plt.savefig(out_dir/"09_vif_heat.png", dpi=300); plt.close()
    log.info("Saved 09_vif_heat.png")
except Exception as e:
    log.warning("VIF calc skipped: %s", e)

# ---- 5. Granger causality (mail -> call) -------------------------------------
from statsmodels.tsa.stattools import grangercausalitytests
gc_res = grangercausalitytests(
    df[["call_volume","mail_volume"]].dropna(), maxlag=14, verbose=False)
gc_table = pd.DataFrame({
    "lag": k,
    "p_value": v[0]['ssr_ftest'][1]
} for k,v in gc_res.items())
gc_table.to_csv(out_dir/"10_granger_table.csv", index=False)
log.info("Saved 10_granger_table.csv")

log.info("🎉 Stage-3 complete – artefacts in ./output/")
"""
# run
import runpy, textwrap, types
exec(compile(textwrap.dedent(code), "<stage3>", "exec"))
PY

echo "---------------------------------------------------------------"
echo "✅  Stage-3 artefacts generated in ./output/  |  logs in ./logs/"
echo "---------------------------------------------------------------"