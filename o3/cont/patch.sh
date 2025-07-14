#!/usr/bin/env bash
###############################################################################
# apply_patches.sh  –  hot-fixes for production modelling repo
#
# ▶ WHAT IT DOES
#   • fixes missing `date` column in economic loader
#   • fixes NumPy-bool JSON issue in mail-intent plotter
#   • fixes CLI global predictor reference
#
# ▶ USAGE
#   chmod +x apply_patches.sh
#   ./apply_patches.sh
#
# Re-run  ./modelling.sh  or the CLI once the patch completes.
###############################################################################
set -euo pipefail
export PYTHONUTF8=1

PKG="customer_comms"

if [[ ! -d "$PKG" ]]; then
  echo "❌ $PKG directory not found – run from repo root"; exit 1
fi

python - <<'PY'
import pathlib, sys, textwrap, re

root = pathlib.Path('.')
pkg  = root / 'customer_comms'

def patch_file(path: pathlib.Path, search: str, replace: str, label: str):
    txt = path.read_text(encoding='utf-8')
    if replace in txt:
        print(f"• {path.name:<30} already patched – {label}")
        return
    if search not in txt:
        print(f"⚠️  {path.name:<30} marker not found – {label} (skipped)")
        return
    patched = txt.replace(search, replace)
    path.write_text(patched, encoding='utf-8')
    print(f"✅ Patched {path.name:<30} – {label}")

# 1️⃣  economic_data.py  – ensure 'date' column
econ = pkg / 'data' / 'economic_data.py'
search = "combined = combined.fillna(method='ffill')"
inject = """
    # ---- PATCH: ensure 'date' column exists even in FRED-only scenarios ----
    if 'date' not in combined.columns:
        combined = combined.reset_index().rename(columns={combined.columns[0]: 'date'})
"""
patch_file(econ, search, search + inject, "date column fix")

# 2️⃣  mail_intent_correlation.py  – NumPy bool -> bool
mic  = pkg / 'viz'  / 'mail_intent_correlation.py'
patch_file(
    mic,
    "'significant': p_val < 0.05",
    "'significant': bool(p_val < 0.05)",
    "NumPy bool JSON fix"
)

# 3️⃣  predict_cli.py  – replace global predictor after load
cli  = pkg / 'cli' / 'predict_cli.py'
search_cli = "production_predictor.load_model()"
replace_cli = "global production_predictor\n            production_predictor = production_predictor.load_model()"
patch_file(cli, search_cli, replace_cli, "CLI global predictor fix")
PY

echo
echo "🎉 All patches applied.  Re-run your pipeline or CLI – the previous errors should be gone."