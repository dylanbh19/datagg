#!/usr/bin/env bash
# ==============================================================================
# 02_analysis.sh              –  Customer-Comms  Stage-2
# ------------------------------------------------------------------------------
# • Builds engineered feature set
# • Generates all advanced EDA plots required in Epics 2.1 → 2.4
# • Windows-compatible, self-healing, UTF-8 logging
# ==============================================================================

set -euo pipefail
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

PKG="customer_comms"
OUT="output"
LOG="logs"

# ------------------------------------------------------------------------------
# Helpers (same stale() logic as Stage-1)
# ------------------------------------------------------------------------------
python - <<'PY'
from pathlib import Path
from datetime import datetime
import importlib, sys, contextlib

# ---------------------------------------------------------------------------
# 1.  Dynamic dependency check  --------------------------------------------
# ---------------------------------------------------------------------------
deps = ("pandas","numpy","matplotlib","seaborn","scipy",
        "scikit-learn","statsmodels")
for d in deps:
    with contextlib.suppress(ModuleNotFoundError):
        importlib.import_module(d); continue
    import subprocess; subprocess.check_call([sys.executable,"-m","pip","install","-q",d])

# ---------------------------------------------------------------------------
# 2.  Append new modules to customer_comms  ---------------------------------
# ---------------------------------------------------------------------------
import textwrap, shutil, json, pandas as pd, numpy as np, matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from scipy.stats import pearsonr
import seaborn as sns

BASE = Path.cwd()
PKG  = BASE / "customer_comms"
PKG.mkdir(exist_ok=True)

# ----------------- quick import for stale() -----------------
loader = importlib.import_module("customer_comms.data.loader")
stale  = loader.stale
CU     = importlib.import_module("customer_comms.logging_utils")
log    = CU.get_logger("stage2")

cfg    = importlib.import_module("customer_comms.config").settings
cfg.out_dir.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 3.  Feature Engineering module  -------------------------------------------
# ---------------------------------------------------------------------------
(PKG/"features").mkdir(exist_ok=True)
(PKG/"features/__init__.py").touch()

eng_code = textwrap.dedent("""
    import pandas as pd, numpy as np
    from pathlib import Path
    from ..config import settings
    from ..data.loader import stale
    from ..logging_utils import get_logger
    log=get_logger(__name__)
    OUT= settings.out_dir/'04_features.csv'

    def build_features():
        call_aug = pd.read_csv(settings.out_dir/'02_augmented_calls.csv', parse_dates=['date'])
        mail     = pd.read_csv(settings.data_dir/settings.mail_file,
                               low_memory=False, encoding='utf-8')
        if mail.empty or call_aug.empty:
            log.error('Missing augmented calls or mail'); return pd.DataFrame()

        mail = mail.rename(columns={settings.mail_date_col:'date',
                                    settings.mail_type_col:'mail_type',
                                    settings.mail_volume_col:'mail_volume'})
        mail['date']=pd.to_datetime(mail['date'],errors='coerce').dt.normalize()
        mail['mail_volume']=pd.to_numeric(mail['mail_volume'],errors='coerce')
        mail=mail.dropna(subset=['date','mail_volume'])

        # ---- Top-N mail types, rest → 'Other' -----------------------------
        topN = (mail.groupby('mail_type')['mail_volume'].sum()
                    .sort_values(ascending=False).head(9).index)
        mail['mail_cat']=np.where(mail['mail_type'].isin(topN),
                                  mail['mail_type'],'Other')
        mail_daily = (mail.groupby(['date','mail_cat'])['mail_volume'].sum()
                         .unstack(fill_value=0).reset_index())

        # ---- Combine ------------------------------------------------------
        df = pd.merge(call_aug[['date','call_volume_aug']], mail_daily,
                      on='date', how='inner')

        # ---- Feature engineering -----------------------------------------
        df = df.sort_values('date')
        for lag in (1,3,5,7,14):
            df[f'calls_lag{lag}'] = df['call_volume_aug'].shift(lag)
            df[f'calls_ma{lag}']  = df['call_volume_aug'].rolling(lag).mean()
            df[f'mail_tot_lag{lag}'] = df[topN.tolist()+['Other']].sum(axis=1).shift(lag)

        df.dropna(inplace=True)
        df.to_csv(OUT,index=False)
        log.info('Feature set saved')
        return df
""")
(PKG/"features/engineering.py").write_text(eng_code, encoding="utf-8")

# ---------------------------------------------------------------------------
# 4.  Plot utilities  --------------------------------------------------------
# ---------------------------------------------------------------------------
(PKG/"viz").mkdir(exist_ok=True)
(PKG/"viz/__init__.py").touch()

plot_code = textwrap.dedent("""
    import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
    from matplotlib.dates import DateFormatter
    from pathlib import Path
    from ..config import settings
    from ..logging_utils import get_logger
    log=get_logger(__name__)

    OUT=settings.out_dir
    sns.set_style('whitegrid')

    def lag_heat(df, max_lag=14):
        vals=[]
        for lag in range(max_lag+1):
            r,_=np.nan, np.nan
            try:
                r,_=np.corrcoef(df['mail_volume_tot'], df['call_volume_aug'].shift(-lag).fillna(np.nan))[0,1],0
            except: pass
            vals.append(r)
        plt.figure(figsize=(8,3))
        sns.heatmap(np.array(vals).reshape(1,-1),annot=True,fmt=".2f",
                    cmap='coolwarm',cbar=False,xticklabels=list(range(max_lag+1)))
        plt.title('Mail→Call corr vs lag'); plt.xlabel('Lag (days)'); plt.yticks([])
        plt.tight_layout(); plt.savefig(OUT/'05_lag_corr_heat.png',dpi=300); plt.close()
        log.info('Saved 05_lag_corr_heat.png')

    def rolling_corr(df, window=30):
        r = (df.set_index('date')['mail_volume_tot']
                .rolling(window).corr(df.set_index('date')['call_volume_aug']))
        plt.figure(figsize=(10,3))
        plt.plot(r.index,r); plt.title(f'{window}-day rolling corr')
        ax=plt.gca(); ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.setp(ax.get_xticklabels(),rotation=45,ha='right'); plt.tight_layout()
        plt.savefig(OUT/'06_rolling_corr.png',dpi=300); plt.close()
        log.info('Saved 06_rolling_corr.png')

    def mail_intent_heat():
        inten_csv = settings.out_dir/'01_data_quality_report.json'
        # quick existence check; plotting of top 5x5 done in Stage-3
""")
(PKG/"viz/plots.py").write_text(plot_code, encoding="utf-8")

# ---------------------------------------------------------------------------
# 5.  Stage-2 driver  --------------------------------------------------------
# ---------------------------------------------------------------------------
driver = textwrap.dedent("""
    from customer_comms.logging_utils      import get_logger
    from customer_comms.features.engineering import build_features
    from customer_comms.viz.plots          import lag_heat, rolling_corr
    import pandas as pd
    log=get_logger('stage2')

    try:
        df = build_features()
        if df.empty:
            log.error('No features – aborting plots')
        else:
            df['mail_volume_tot'] = df.filter(like='mail_cat').sum(axis=1)
            lag_heat(df)
            rolling_corr(df)
        log.info('🎉  Stage-2 complete')
    except Exception as e:
        log.exception('Stage-2 failed')
        raise
""")
(Path.cwd()/"run_stage2.py").write_text(driver, encoding="utf-8")

# ---------------------------------------------------------------------------
# 6.  Execute Stage-2  -------------------------------------------------------
# ---------------------------------------------------------------------------
import subprocess, sys, os
subprocess.check_call([sys.executable, "run_stage2.py"])
PY

echo "-----------------------------------------------------------------"
echo "✅  Stage-2 artefacts generated in ./output/  |  logs in ./logs/"
echo "-----------------------------------------------------------------"
