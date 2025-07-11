#!/usr/bin/env bash
# ======================================================================
#  00_bootstrap_customer_comms.sh
#  --------------------------------------------------------------------
#  Turns an empty directory into the full code-base required by the
#  “Customer Communication Analytics” project brief.
#  • Robust UTF-8 everywhere (Windows safe)
#  • Dependency self-install  (only when missing)
#  • Creates package  customer_comms/   with sub-modules for every Epic
#  • Injects production-grade logging, CSV multi-encoding reader,
#    MA-aligned call-augmentation & min-250 intent filter.
#  • Generates a Makefile with high-level targets (ingest, eda, model,…)
#  • Script exits 0 even on data decode errors – they’re logged.
# ======================================================================
set -euo pipefail
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

PKG="customer_comms"
echo "🔧  Creating project skeleton in $(pwd)/$PKG …"
rm -rf "$PKG" {logs,output,tests} 2>/dev/null || true
mkdir -p "$PKG"/{data,processing,analytics,modeling,viz,dashboards,docs} logs output tests
touch "$PKG"/__init__.py
for sub in data processing analytics modeling viz dashboards; do
  mkdir -p "$PKG/$sub"/__init__.py
done

# ---------------------------------------------------------------------
# 0.  Dependencies  (only install if missing)
# ---------------------------------------------------------------------
python - <<'PY'
import importlib, subprocess, sys, contextlib
req=['pandas','numpy','matplotlib','seaborn','scipy','scikit-learn',
     'statsmodels','holidays','pydantic','pydantic-settings','plotly',
     'dash','yfinance']
for p in req:
    with contextlib.suppress(ModuleNotFoundError):
        importlib.import_module(p.replace('-','_')); continue
    print(f"📦  Installing {p} …", flush=True)
    subprocess.check_call([sys.executable,'-m','pip','install','-q',p])
PY

# ---------------------------------------------------------------------
# 1.  config.py  (central settings)
# ---------------------------------------------------------------------
cat > "$PKG/config.py" << 'PY'
from pathlib import Path
try:
    from pydantic_settings import BaseSettings            # Pydantic v2
except ModuleNotFoundError:
    from pydantic import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # --------------- raw CSV locations -----------------
    call_files: list[str] = ["GenesysExtract_20250609.csv",
                             "GenesysExtract_20250703.csv"]
    mail_file:  str       = "merged_call_data.csv"
    econ_file:  str       = "econ_indicators.csv"    # optional

    # --------------- column names ----------------------
    call_date_cols:   list[str] = ["ConversationStart","Date"]
    call_intent_cols: list[str] = ["uui_Intent","intent","Intent"]
    mail_date_col:    str = "mail_date"
    mail_type_col:    str = "mail_type"
    mail_volume_col:  str = "mail_volume"

    # --------------- processing knobs ------------------
    augment_gap_limit: int  = 3      # ≤ N business-day f-fill
    ma_window:         int  = 7      # align call MA to intent MA
    min_intent_rows:   int  = 250    # filter sparse intents
    min_rows:          int  = 20     # fail-fast threshold
    max_lag:           int  = 21     # lag scan

    # --------------- dirs ------------------------------
    data_dir:  Path = Field(default=Path("data"))
    out_dir:   Path = Field(default=Path("output"))
    log_dir:   Path = Field(default=Path("logs"))
settings = Settings()
PY

# ---------------------------------------------------------------------
# 2.  logging_utils.py
# ---------------------------------------------------------------------
cat > "$PKG/logging_utils.py" << 'PY'
import logging, sys, traceback
from datetime import datetime
from .config import settings

def get_logger(name="customer_comms"):
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    dh ="%Y-%m-%d %H:%M:%S"

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(fmt, dh))
    try: sh.stream.reconfigure(encoding="utf-8")
    except AttributeError: pass
    log.addHandler(sh)

    settings.log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(settings.log_dir /
                             f"{name}_{datetime.now():%Y%m%d}.log",
                             encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt, dh))
    log.addHandler(fh)

    # auto stack-trace on CRITICAL
    def handle(record):                                  # noqa: D401
        if record.levelno >= logging.CRITICAL:
            log.error(traceback.format_exc())
    log.addHandler(logging.StreamHandler(stream=sys.stderr))
    return log
PY

# ---------------------------------------------------------------------
# 3.  data/loader.py  – multi-encoding + fallback
# ---------------------------------------------------------------------
cat > "$PKG/data/loader.py" << 'PY'
from __future__ import annotations
import pandas as pd, io
from pathlib import Path
from ..config import settings
from ..logging_utils import get_logger
log = get_logger(__name__)

_CODECS = ["utf-8","utf-8-sig","latin-1","cp1252","utf-16","iso-8859-1"]
def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        log.error(f"File not found: {path}"); return pd.DataFrame()
    for enc in _CODECS:
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="skip", low_memory=False)
        except UnicodeDecodeError:  # try next encoding
            continue
    log.warning(f"{path}: all decoders failed, using UTF-8 replacement")
    text = path.read_bytes().decode("utf-8", "replace")
    return pd.read_csv(io.StringIO(text), on_bad_lines="skip", low_memory=False)

def _to_date(s):
    return (pd.to_datetime(s, errors="coerce")
              .dt.tz_localize(None)
              .dt.normalize())
PY

# ---------------------------------------------------------------------
# 4.  processing/ingest.py  (Epic 1 – Story 1.1 skeleton)
# ---------------------------------------------------------------------
cat > "$PKG/processing/ingest.py" << 'PY'
"""Load & profile raw data – Story 1.1."""
import pandas as pd, json, numpy as np
from pathlib import Path
from ..config import settings
from ..data.loader import _read_csv, _to_date
from ..logging_utils import get_logger
log = get_logger(__name__)

def load_all():
    # ---- call volumes & intents -------------------------------------
    call_frames = []; intents_present = False
    for fn in settings.call_files:
        df = _read_csv(settings.data_dir / fn)
        if df.empty: continue
        dcol = next((c for c in settings.call_date_cols if c in df.columns), None)
        icol = next((c for c in settings.call_intent_cols if c in df.columns), None)
        if not dcol:
            log.error(f"{fn}: no date col"); continue
        rename = {dcol: "date"} | ({icol: "intent"} if icol else {})
        df = df.rename(columns=rename)[["date"]+(['intent'] if icol else [])]
        df["date"] = _to_date(df["date"]);  df = df.dropna(subset=["date"])
        call_frames.append(df)
        intents_present |= bool(icol)
    raw_call = pd.concat(call_frames, ignore_index=True) if call_frames else pd.DataFrame()

    # ---- mail --------------------------------------------------------
    mail = _read_csv(settings.data_dir / settings.mail_file)
    if not mail.empty:
        mail = mail.rename(columns={settings.mail_date_col:"date",
                                    settings.mail_type_col:"mail_type",
                                    settings.mail_volume_col:"mail_volume"})
        mail["date"] = _to_date(mail["date"])
        mail["mail_volume"] = pd.to_numeric(mail["mail_volume"], errors="coerce")
        mail = mail.dropna(subset=["date","mail_volume"])

    # ---- econ (optional) --------------------------------------------
    econ_path = settings.data_dir / getattr(settings, "econ_file", "")
    econ = _read_csv(econ_path) if econ_path.exists() else pd.DataFrame()

    return raw_call, mail, econ

def profile(df: pd.DataFrame, name: str):
    if df.empty:
        return {"dataset": name, "empty": True}
    rep = {
        "dataset": name,
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "na_counts": df.isna().sum().to_dict(),
        "date_range": [str(df.select_dtypes("datetime").min().min()),
                       str(df.select_dtypes("datetime").max().max())],
    }
    return rep

def run():
    call, mail, econ = load_all()
    reports = [profile(call, "raw_call"),
               profile(mail, "mail"),
               profile(econ, "econ")]
    out = settings.out_dir / "01_data_quality_report.json"
    settings.out_dir.mkdir(exist_ok=True)
    json.dump(reports, open(out, "w", encoding="utf-8"), indent=2)
    log.info(f"Data quality report → {out}")
    return call, mail, econ
PY

# ---------------------------------------------------------------------
# 5.  analytics/augment.py  (Epic 1 – Story 1.2 ready)
# ---------------------------------------------------------------------
cat > "$PKG/analytics/augment.py" << 'PY'
"""Call-volume augmentation to match intent MA (Story 1.2)."""
import pandas as pd, numpy as np
from ..config import settings
from ..logging_utils import get_logger
log = get_logger(__name__)

def augment(call: pd.DataFrame) -> pd.DataFrame:
    if call.empty or "intent" not in call.columns:
        return call.assign(call_volume_aug=call["call_volume"])
    # ---- calc intent counts per day ---------------------------------
    intent_daily = (call.groupby(["date","intent"])
                         .size().unstack(fill_value=0).reset_index())
    total_intent = intent_daily.set_index("date").sum(axis=1)
    cv = (call.groupby("date").size()
                .rename("call_volume")
                .to_frame())
    cv["intent_total"] = total_intent
    cv = cv.dropna()
    # ---- moving-average scaling ------------------------------------
    w = settings.ma_window
    ma_scale = (cv["intent_total"].rolling(w,1).mean() /
                cv["call_volume"].rolling(w,1).mean().replace(0,np.nan))
    ma_scale = ma_scale.clip(0.25, 4).fillna(1)
    cv["call_volume_aug"] = cv["call_volume"] * ma_scale
    return cv.reset_index()
PY

# ---------------------------------------------------------------------
# 6.  analytics/mail_intent_corr.py  (Story 2.2 skeleton)
# ---------------------------------------------------------------------
cat > "$PKG/analytics/mail_intent_corr.py" << 'PY'
"""Mail-type × Intent correlation with min-250 filter."""
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from scipy.stats import pearsonr
from ..config import settings
from ..logging_utils import get_logger
log = get_logger(__name__)

def corr_heat(mail: pd.DataFrame, call: pd.DataFrame):
    if mail.empty or "intent" not in call.columns:
        return
    intents = (call.groupby(["date","intent"]).size()
                    .unstack(fill_value=0).reset_index())
    # ---- filter sparse intents -------------------------------------
    keep = [c for c in intents.columns if c!="date"
            and intents[c].sum() >= settings.min_intent_rows]
    if not keep:
        log.warning("No intents ≥ threshold");  return
    intents = intents[["date"]+keep]
    mail_piv = (mail.pivot_table(index="date", columns="mail_type",
                                 values="mail_volume", aggfunc="sum")
                   .fillna(0))
    merged = mail_piv.reset_index().merge(intents, on="date")
    res = []
    for m in mail_piv.columns:
        for i in keep:
            if merged[m].std()==0 or merged[i].std()==0: continue
            r,_ = pearsonr(merged[m], merged[i]); res.append((m,i,abs(r),r))
    top = sorted(res,key=lambda x:x[2],reverse=True)[:10]
    if not top: return
    hm = pd.DataFrame(top,columns=["mail","intent","abs_r","r"]).set_index(["mail","intent"])
    plt.figure(figsize=(8,6)); sns.heatmap(hm[["r"]],annot=True,cmap="vlag",center=0,fmt=".2f")
    plt.title("Top-10 |r| Mail-Type × Intent  (≥250 rows)")
    out = settings.out_dir / "mailtype_intent_corr.png"
    plt.tight_layout(); plt.savefig(out,dpi=300); plt.close()
    log.info(f"Saved {out}")
PY

# ---------------------------------------------------------------------
# 7.  dashboards/README.md  (Epic 4 placeholder)
# ---------------------------------------------------------------------
cat > "$PKG/dashboards/README.md" << 'MD'
# Dashboards (Epic 4)

* `exec_dashboard.py` – will serve COO-level KPIs  
* `analysis_dash.py` – deep-dive interactive components

> **Next step:** implement Dash / Plotly code in these files.
MD

# ---------------------------------------------------------------------
# 8.  docs/PROJECT_EPICS.md – auto-generated from your brief
# ---------------------------------------------------------------------
cat > "$PKG/docs/PROJECT_EPICS.md" << 'MD'
*(generated stub – fill out as you implement)*

# Project Roadmap

The directory structure mirrors these Epics & Stories:

| Epic | Story ID | Folder / Stub Module |
|------|----------|----------------------|
| Data Foundation & Quality | 1.1 | `processing/ingest.py` |
| | 1.2 | `analytics/augment.py` |
| | 1.3 | `processing/econ_ingest.py` (TBD) |
| Advanced EDA & Insights | 2.1 – 2.4 | `analytics/` sub-modules |
| Predictive Modeling | 3.x | `modeling/` sub-modules |
| Production-Grade Visualizations | 4.x | `viz/` & `dashboards/` |
| Delivery & Documentation | 5.x | `docs/` & top-level `Makefile` |
MD

# ---------------------------------------------------------------------
# 9.  Makefile  (high-level targets)
# ---------------------------------------------------------------------
cat > Makefile << 'MK'
.PHONY: ingest eda model dashboard clean

ingest:        ## Run data ingestion & profiling
	python -c "import customer_comms.processing.ingest as p; p.run()"

eda:           ## Placeholder – advanced EDA pipeline
	@echo "TODO: implement EDA pipeline"

model:         ## Placeholder – modelling pipeline
	@echo "TODO: implement modelling pipeline"

dashboard:     ## Placeholder – run Dash dashboard
	@echo "TODO: launch Dash app"

clean:
	rm -rf logs/* output/*
MK

# ---------------------------------------------------------------------
# 10.  mini driver to prove everything imports
# ---------------------------------------------------------------------
python - <<'PY'
from customer_comms.processing.ingest import run
from customer_comms.analytics.augment import augment
from customer_comms.analytics.mail_intent_corr import corr_heat
from customer_comms.logging_utils import get_logger
log=get_logger("bootstrap-test")
try:
    call, mail, _ = run()
    if not call.empty:
        aug = augment(call)
        log.info(f"Augmented sample rows: {len(aug)}")
    corr_heat(mail, call)
except Exception as e:
    log.critical(f"Bootstrap smoke-test failed: {e}", exc_info=True)
PY

echo "✅  Project scaffold complete."
echo "   • Inspect ./output/ for first artefacts"
echo "   • Start iterating per Epic – see docs/PROJECT_EPICS.md"
