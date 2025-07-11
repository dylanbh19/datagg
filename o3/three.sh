#!/usr/bin/env bash
# ======================================================================
#  part3_modelling.sh                    (Customer-Comms  –  Stage 3 of 4)
#  --------------------------------------------------------------------
#  Adds Epic 3: feature-engineering, model training, interpretability.
# ======================================================================
set -euo pipefail
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

PKG="customer_comms"
[[ -d $PKG ]] || { echo "❌  Run part1 & part2 first"; exit 1; }

# ---------------------------------------------------------------------
# 0.  Extra dependencies (Prophet, XGBoost, SHAP – silent install)
# ---------------------------------------------------------------------
python - <<'PY'
import importlib, subprocess, sys, contextlib
deps=['prophet','xgboost','shap']
for p in deps:
    with contextlib.suppress(ModuleNotFoundError):
        importlib.import_module(p.replace('-','_')); continue
    subprocess.check_call([sys.executable,'-m','pip','install','-q',p])
PY

# ---------------------------------------------------------------------
# 1.  features/engineer.py  – Story 3.1
# ---------------------------------------------------------------------
cat > "$PKG/features/engineer.py" << 'PY'
"""Extensive feature-engineering pipeline (Story 3.1)."""
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from ..config import settings
from ..logging_utils import get_logger
log=get_logger(__name__)

def build(base: pd.DataFrame) -> pd.DataFrame:
    if base.empty: return pd.DataFrame()
    df = base.copy()
    # ---- time features -------------------------------------------------
    df["dow"]      = df["date"].dt.weekday
    df["week"]     = df["date"].dt.isocalendar().week.astype(int)
    df["month"]    = df["date"].dt.month
    df["quarter"]  = df["date"].dt.quarter
    # ---- lags & rolling ------------------------------------------------
    for l in (1,3,7,14,21):
        df[f"mail_lag_{l}"]  = df["mail_volume"].shift(l)
        df[f"call_lag_{l}"]  = df["call_volume_aug"].shift(l)
    for w in (3,7,14):
        df[f"mail_rmean_{w}"] = df["mail_volume"].rolling(w,1).mean()
        df[f"call_rmean_{w}"] = df["call_volume_aug"].rolling(w,1).mean()
        df[f"mail_rstd_{w}"]  = df["mail_volume"].rolling(w,1).std()
        df[f"call_rstd_{w}"]  = df["call_volume_aug"].rolling(w,1).std()
    # ---- pct change / volatility --------------------------------------
    df["mail_pct_1"]  = df["mail_volume"].pct_change()*100
    df["call_pct_1"]  = df["call_volume_aug"].pct_change()*100
    # ---- interaction ---------------------------------------------------
    df["ratio_call_mail"] = df["call_volume_aug"] / df["mail_volume"].replace(0,np.nan)
    df.replace([np.inf,-np.inf],np.nan,inplace=True)
    df.dropna(inplace=True)
    log.info(f"Feature dataframe rows={len(df)}, cols={df.shape[1]}")
    return df

def variance_filter(df: pd.DataFrame, thresh: float = 1e-5) -> pd.DataFrame:
    num = df.select_dtypes("number")
    vt  = VarianceThreshold(threshold=thresh).fit(num)
    kept = num.columns[vt.get_support()]
    dropped = set(num.columns) - set(kept)
    log.info(f"Variance filter dropped {len(dropped)} cols")
    return df[kept.union(df.select_dtypes(exclude="number").columns)]
PY

# ---------------------------------------------------------------------
# 2.  modeling/train.py  – Story 3.2
# ---------------------------------------------------------------------
cat > "$PKG/modeling/train.py" << 'PY'
"""Model training & CV (Story 3.2)."""
import pandas as pd, numpy as np, json, warnings, itertools
warnings.filterwarnings("ignore")
from pathlib import Path
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.statespace.sarimax import SARIMAX
from ..config import settings
from ..logging_utils import get_logger
log=get_logger(__name__)

def metrics(y, yhat):
    mae = mean_absolute_error(y,yhat)
    rmse= np.sqrt(mean_squared_error(y,yhat))
    mape= np.mean(np.abs((y-yhat)/y))*100
    return dict(mae=mae, rmse=rmse, mape=mape)

def train_sarimax(series: pd.Series):
    order=(1,0,1); seasonal=(1,0,1,7)
    model=SARIMAX(series,order=order,seasonal_order=seasonal,
                  enforce_stationarity=False,enforce_invertibility=False)
    res=model.fit(disp=False)
    return res

def train_prophet(df: pd.DataFrame):
    m=Prophet(daily_seasonality=True,weekly_seasonality=True,yearly_seasonality=True)
    m.fit(df.rename(columns={"date":"ds","call_volume_aug":"y"}))
    return m

def train_xgb(X: pd.DataFrame, y: pd.Series):
    model=XGBRegressor(objective="reg:squarederror",n_estimators=300,
                       max_depth=4,learning_rate=0.05,subsample=0.8,
                       random_state=42)
    model.fit(X,y)
    return model

def cv_loop(df: pd.DataFrame):
    out=[]
    tscv=TimeSeriesSplit(n_splits=3)
    target="call_volume_aug"
    features=[c for c in df.columns if c not in ("date",target)]
    X=df[features]; y=df[target]
    for fold,(tr,te) in enumerate(tscv.split(df)):
        model=train_xgb(X.iloc[tr],y.iloc[tr])
        pred=model.predict(X.iloc[te])
        out.append(metrics(y.iloc[te],pred)|{"fold":fold+1})
    return pd.DataFrame(out)

def pipeline(feature_df: pd.DataFrame):
    if feature_df.empty: return
    # -------- XGBoost CV ----------------------------------------------
    cv=cv_loop(feature_df)
    cv.to_csv(settings.out_dir/"10_xgb_cv_scores.csv",index=False)
    log.info("XGBoost CV done")
    # -------- SARIMAX / Prophet one-shot --------------------------------
    series=feature_df.set_index("date")["call_volume_aug"]
    sar_res=train_sarimax(series)
    sar_fore=sar_res.fittedvalues
    json.dump(metrics(series,sar_fore),
              open(settings.out_dir/"10_sarimax_metrics.json","w"),indent=2)
    try:
        proph=train_prophet(feature_df[["date","call_volume_aug"]])
        fut=proph.make_future_dataframe(periods=0)
        yhat=proph.predict(fut)["yhat"]
        json.dump(metrics(series,yhat),
                  open(settings.out_dir/"10_prophet_metrics.json","w"),indent=2)
        log.info("Prophet complete")
    except Exception as e:
        log.warning(f"Prophet failed: {e}")
PY

# ---------------------------------------------------------------------
# 3.  modeling/interpret.py  – Story 3.3
# ---------------------------------------------------------------------
cat > "$PKG/modeling/interpret.py" << 'PY'
"""SHAP & PDP interpretability for XGBoost (Story 3.3)."""
import pandas as pd, shap, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
from xgboost import XGBRegressor
from ..config import settings
from ..logging_utils import get_logger
log=get_logger(__name__)

def shap_summary(df: pd.DataFrame):
    target="call_volume_aug"
    X=df.drop(columns=["date",target]); y=df[target]
    model=XGBRegressor(n_estimators=300,learning_rate=0.05,
                       max_depth=4,random_state=42,subsample=0.8)
    model.fit(X,y)
    expl=shap.Explainer(model); shap_values=expl(X)
    plt.figure(); shap.summary_plot(shap_values,X,show=False)
    out=settings.out_dir/"11_shap_summary.png"; plt.tight_layout(); plt.savefig(out,dpi=300); plt.close()
    log.info(f"Saved {out}")
    return model

def pdp_top5(model,df:pd.DataFrame):
    feat_imp=sorted(model.get_booster().get_score(importance_type="gain").items(),
                    key=lambda x:x[1],reverse=True)[:5]
    top=[f for f,_ in feat_imp]
    for f in top:
        fig=PartialDependenceDisplay.from_estimator(model,df[top+["call_volume_aug"]],
                                                    features=[f],kind="average")
        out=settings.out_dir/f"11_pdp_{f}.png"
        fig.figure_.savefig(out,dpi=300); plt.close(fig.figure_)
        log.info(f"PDP {f} → {out}")
PY

# ---------------------------------------------------------------------
# 4.  Extend Makefile with modelling targets
# ---------------------------------------------------------------------
awk '/^clean:/ {exit} {print}' Makefile > Makefile.tmp || true
cat Makefile.tmp > Makefile && rm -f Makefile.tmp

cat >> Makefile << 'MK'

features:      ## Story 3.1 – generate engineered feature set
	python - <<'PY'
import customer_comms.processing.ingest as ing
from customer_comms.analytics.augment import augment
from customer_comms.features.engineer import build, variance_filter
call, mail = ing.run(); base=augment(call).merge(mail,on="date",how="inner")
feat=variance_filter(build(base))
feat.to_csv("output/10_feature_matrix.csv",index=False)
print("→ output/10_feature_matrix.csv")
PY

model_train:   ## Story 3.2 – train models & CV
	python - <<'PY'
import pandas as pd, json
from customer_comms.modeling.train import pipeline
df=pd.read_csv("output/10_feature_matrix.csv",parse_dates=["date"])
pipeline(df)
PY

interpret:     ## Story 3.3 – SHAP & PDP plots
	python - <<'PY'
import pandas as pd
from customer_comms.modeling.interpret import shap_summary, pdp_top5
df=pd.read_csv("output/10_feature_matrix.csv",parse_dates=["date"])
model=shap_summary(df); pdp_top5(model,df)
PY
MK

echo "✅  Stage 3 added.  Run:"
echo "   make features      # build 50+ engineered features"
echo "   make model_train   # SARIMAX, Prophet, XGBoost CV"
echo "   make interpret     # SHAP & PDP visuals"
echo "   Artefacts land in ./output/, logs in ./logs/"
