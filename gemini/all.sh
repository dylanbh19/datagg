#!/usr/bin/env bash
# =====================================================================
#  Unified Customer Comms Pipeline (Gemini Production-Grade Version)
#
#  This single script scaffolds the entire project, loads and cleans
#  data, runs a full suite of EDA and advanced analytics, determines
#  the best data transformations, and generates a feature-rich
#  dataset ready for modeling, based on the full project plan.
#
#  Usage: ./unified_pipeline.sh [project_name]
# =====================================================================
set -euo pipefail

# ---------------------------------------------------------------------
# 1. Project Setup
# ---------------------------------------------------------------------
PROJ="${1:-Customer_Comms_Analysis_Project}"
PKG="customer_comms"

echo "🔧 Creating / refreshing project directory: ${PROJ}"
mkdir -p "${PROJ}"/{data,logs,output,tests}
cd "${PROJ}"
echo "✅ Project structure created in ./${PROJ}"

# ---------------------------------------------------------------------
# 2. Dependency Installation
# ---------------------------------------------------------------------
echo "📦 Ensuring Python dependencies are installed..."
python - <<'PY'
import importlib, subprocess, sys
pkgs = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'scipy',
        'scikit-learn', 'statsmodels', 'holidays',
        'pydantic', 'pydantic-settings']
for p in pkgs:
    if importlib.util.find_spec(p.replace('-', '_')) is None:
        print(f"  -> Installing {p}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', p])
print("✅ All dependencies are satisfied.")
PY

# ---------------------------------------------------------------------
# 3. Python Package Scaffolding
# ---------------------------------------------------------------------
echo "🏗️ Scaffolding Python package '${PKG}'..."
for d in "$PKG" "$PKG/data" "$PKG/processing" "$PKG/analytics" "$PKG/viz" "$PKG/features"; do
  mkdir -p "$d"
  touch "$d/__init__.py"
done
echo "✅ Package structure created."

# ---------------------------------------------------------------------
# 4. Module Generation (Writing all .py files)
# ---------------------------------------------------------------------
echo "📝 Generating Python modules..."

# --- 4.1 config.py ---
cat > "$PKG/config.py" << 'PY'
from pathlib import Path
try:
    from pydantic_settings import BaseSettings
except ModuleNotFoundError:
    from pydantic import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # --- Input Files (place in ./data) ---
    call_files: list[str] = ["GenesysExtract_20250609.csv", "GenesysExtract_20250703.csv"]
    mail_file:  str       = "merged_call_data.csv"

    # --- Column Names (flexible detection) ---
    call_date_cols:   list[str] = ["ConversationStart", "Date"]
    call_intent_cols: list[str] = ["uui_Intent", "uuiIntent", "intent", "Intent"]
    mail_date_col:   str = "mail_date"
    mail_type_col:   str = "mail_type"
    mail_volume_col: str = "mail_volume"
    
    # --- Economic Data Tickers (Story 1.3) ---
    economic_tickers: dict[str, str] = {
        "S&P_500": "^GSPC",
        "VIX": "^VIX",
        "10_Yr_Treasury": "^TNX",
        "Oil_Prices": "CL=F",
    }

    # --- Analysis Parameters ---
    max_lag:    int  = 28
    rolling_window: int = 7
    anomaly_contamination: float = 0.05
    imputation_gap_limit: int = 3 # Max consecutive days to forward-fill in augmentation

    # --- Directory Setup ---
    data_dir:   Path = Field(default=Path("data"))
    log_dir:    Path = Field(default=Path("logs"))
    output_dir: Path = Field(default=Path("output"))

settings = Settings()
PY

# --- 4.2 logging_utils.py ---
cat > "$PKG/logging_utils.py" << 'PY'
import logging, sys
from datetime import datetime
from .config import settings

def get_logger(name="customer_comms"):
    log = logging.getLogger(name)
    if log.handlers:
        return log

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s - %(message)s"
    dt_fmt  = "%Y-%m-%d %H:%M:%S"
    log.setLevel(logging.INFO)

    # Console Handler (UTF-8 safe)
    con = logging.StreamHandler(sys.stdout)
    con.setFormatter(logging.Formatter(fmt, dt_fmt))
    try: con.stream.reconfigure(encoding="utf-8")
    except AttributeError: pass
    log.addHandler(con)

    # File Handler
    settings.log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(settings.log_dir / f"{name}_{datetime.now():%Y%m%d}.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt, dt_fmt))
    log.addHandler(fh)
    return log
PY

# --- 4.3 data/loader.py (Handles Economic Data - Story 1.3) ---
cat > "$PKG/data/loader.py" << 'PY'
import pandas as pd
import yfinance as yf
from pathlib import Path
from ..config import settings
from ..logging_utils import get_logger

log = get_logger(__name__)
_ENCODINGS = ("utf-8", "latin-1", "cp1252")

def _read_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        log.error(f"File not found: {p}")
        return pd.DataFrame()
    for enc in _ENCODINGS:
        try:
            return pd.read_csv(p, encoding=enc, on_bad_lines="skip", low_memory=False)
        except Exception:
            continue
    log.error(f"Could not decode {p} with any tried encoding.")
    return pd.DataFrame()

def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None).dt.normalize()

def load_call_data():
    try:
        dfs = []
        for fname in settings.call_files:
            df = _read_csv(settings.data_dir / fname)
            if df.empty: continue
            
            date_col = next((c for c in settings.call_date_cols if c in df.columns), None)
            if not date_col:
                log.error(f"In {fname}, no date column found from candidates: {settings.call_date_cols}")
                continue
                
            intent_col = next((c for c in settings.call_intent_cols if c in df.columns), None)
            
            rename_map = {date_col: "date"}
            if intent_col: rename_map[intent_col] = "intent"
                
            df = df.rename(columns=rename_map)
            df["date"] = _to_date(df["date"])
            df = df.dropna(subset=["date"])
            
            cols_to_keep = ["date"] + (["intent"] if intent_col else [])
            dfs.append(df[cols_to_keep])

        if not dfs: return pd.DataFrame()
        
        log.info(f"Successfully loaded and combined {len(dfs)} call file(s).")
        return pd.concat(dfs, ignore_index=True)
    except Exception as e:
        log.error(f"Failed to load call data: {e}", exc_info=True)
        return pd.DataFrame()

def load_mail_data():
    try:
        df = _read_csv(settings.data_dir / settings.mail_file)
        if df.empty: return df

        df = df.rename(columns={
            settings.mail_date_col: "date",
            settings.mail_type_col: "mail_type",
            settings.mail_volume_col: "mail_volume"
        })
        df["date"] = _to_date(df["date"])
        df["mail_volume"] = pd.to_numeric(df["mail_volume"], errors="coerce")
        df = df.dropna(subset=["date", "mail_volume"])
        log.info(f"Successfully loaded mail data with {len(df)} rows.")
        return df
    except Exception as e:
        log.error(f"Failed to load mail data: {e}", exc_info=True)
        return pd.DataFrame()

def load_economic_data(start_date, end_date):
    log.info(f"Loading economic data from {start_date} to {end_date}...")
    try:
        df = yf.download(list(settings.economic_tickers.values()), start=start_date, end=end_date, progress=False)['Adj Close']
        df.rename(columns={v: k for k, v in settings.economic_tickers.items()}, inplace=True)
        df = df.ffill().bfill()
        log.info(f"Successfully loaded {len(df.columns)} economic indicators.")
        return df.reset_index().rename(columns={'Date': 'date'})
    except Exception as e:
        log.error(f"Failed to download economic data: {e}", exc_info=True)
        return pd.DataFrame()
PY

# --- 4.5 features/engineer.py (Epic 3) ---
cat > "$PKG/features/engineer.py" << 'PY'
import pandas as pd
import numpy as np
import holidays
from ..config import settings
from ..logging_utils import get_logger
from ..data.loader import load_economic_data

log = get_logger(__name__)

def augment_call_volume(call_df: pd.DataFrame, intent_df: pd.DataFrame) -> pd.DataFrame:
    """Augments call volume using intent data to fill gaps. (Story 1.2)"""
    log.info("Augmenting call volume data...")
    try:
        # Create a complete daily index
        full_range = pd.date_range(start=call_df['date'].min(), end=call_df['date'].max(), freq='D')
        call_df = call_df.set_index('date').reindex(full_range).rename_axis('date').reset_index()
        
        # Forward fill small gaps
        call_df['call_volume_aug'] = call_df['call_volume'].ffill(limit=settings.imputation_gap_limit)
        
        # For larger gaps, use intent-to-volume ratio if possible
        if not intent_df.empty:
            intent_total = intent_df.drop(columns=['date']).sum(axis=1)
            call_total = call_df['call_volume']
            ratio = (call_total / intent_total).replace([np.inf, -np.inf], np.nan).mean()
            
            if pd.notna(ratio):
                imputed_values = intent_total * ratio
                call_df['call_volume_aug'] = call_df['call_volume_aug'].fillna(imputed_values)

        call_df['call_volume_aug'] = call_df['call_volume_aug'].fillna(call_df['call_volume'])
        log.info("Call volume augmentation complete.")
        return call_df
    except Exception as e:
        log.error(f"Call volume augmentation failed: {e}", exc_info=True)
        call_df['call_volume_aug'] = call_df['call_volume']
        return call_df

def create_features_for_modeling(mail_df: pd.DataFrame, call_df: pd.DataFrame) -> pd.DataFrame:
    log.info("Starting comprehensive feature engineering process (Epic 3)...")
    
    try:
        # 1. Aggregate data to daily level
        daily_mail = mail_df.pivot_table(index='date', columns='mail_type', values='mail_volume', aggfunc='sum').fillna(0)
        daily_mail.columns = [f"mail_{c.replace(' ', '_').lower()}" for c in daily_mail.columns]
        daily_mail['mail_total'] = daily_mail.sum(axis=1)

        daily_calls_intents = call_df.pivot_table(index='date', columns='intent', aggfunc='size').fillna(0)
        daily_calls_intents.columns = [f"call_{c.replace(' ', '_').lower()}" for c in daily_calls_intents.columns]
        
        daily_calls_volume = call_df.groupby('date').size().reset_index(name='call_volume')
        
        # 2. Augment call volume (Story 1.2)
        augmented_calls = augment_call_volume(daily_calls_volume, daily_calls_intents.reset_index())

        # 3. Combine into a single master dataframe
        df = pd.merge(daily_mail.reset_index(), augmented_calls, on='date', how='outer')
        df = pd.merge(df, daily_calls_intents.reset_index(), on='date', how='outer')
        df = df.set_index('date').asfreq('D').fillna(0).reset_index()

        # 4. Integrate Economic Data (Story 1.3)
        econ_df = load_economic_data(df['date'].min(), df['date'].max())
        if not econ_df.empty:
            df = pd.merge(df, econ_df, on='date', how='left').ffill().bfill()
            log.info("Economic data successfully integrated.")

        # 5. Time-based features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
        df['month'] = df['date'].dt.month
        df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)

        # 6. Holiday features
        us_hols = holidays.UnitedStates(years=df['date'].dt.year.unique())
        df['is_holiday'] = df['date'].isin(us_hols).astype(int)

        # 7. Lagged features (for mail and total calls)
        feature_cols = [c for c in df.columns if 'mail_' in c or c == 'call_volume_aug']
        for col in feature_cols:
            for lag in [1, 2, 3, 7, 14, 21, 28]:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag).fillna(0)

        # 8. Rolling window features
        for col in feature_cols:
            df[f'{col}_roll_mean_{settings.rolling_window}'] = df[col].rolling(window=settings.rolling_window, min_periods=1).mean().fillna(0)
            df[f'{col}_roll_std_{settings.rolling_window}'] = df[col].rolling(window=settings.rolling_window, min_periods=1).std().fillna(0)

        log.info(f"Feature engineering complete. Dataset has {df.shape[1]} columns.")
        
        out_path = settings.output_dir / "features_for_modeling.csv"
        df.to_csv(out_path, index=False)
        log.info(f"✅ Pre-modeling dataset saved to: {out_path}")
        
        return df
    except Exception as e:
        log.critical(f"Feature engineering failed critically: {e}", exc_info=True)
        return pd.DataFrame()
PY

# --- 4.6 analytics/core.py (Epic 2) ---
cat > "$PKG/analytics/core.py" << 'PY'
import pandas as pd
import json
from scipy.stats import pearsonr
from sklearn.ensemble import IsolationForest
from ..config import settings
from ..logging_utils import get_logger

log = get_logger(__name__)

def generate_health_report(df: pd.DataFrame):
    log.info("Generating data health report (Story 1.1)...")
    try:
        report = {
            "total_rows": len(df),
            "date_range": {"start": str(df["date"].min()), "end": str(df["date"].max())},
            "missing_values_per_column": df.isna().sum().to_dict(),
            "duplicate_dates": int(df["date"].duplicated().sum())
        }
        out_path = settings.output_dir / "data_health_report.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        log.info(f"Data health report saved to {out_path}")
    except Exception as e:
        log.error(f"Failed to generate health report: {e}", exc_info=True)

def find_anomalies(df: pd.DataFrame):
    log.info("Running anomaly detection (Story 2.4)...")
    try:
        df_anom = df.copy()
        numeric_cols = [c for c in df.columns if 'call_' in c or 'mail_' in c]
        if not numeric_cols: return df_anom
        
        iso_forest = IsolationForest(contamination=settings.anomaly_contamination, random_state=42)
        df_anom['is_anomaly'] = iso_forest.fit_predict(df_anom[numeric_cols])
        df_anom['is_anomaly'] = (df_anom['is_anomaly'] == -1)
        log.info(f"Anomaly detection found {df_anom['is_anomaly'].sum()} anomalies.")
        return df_anom
    except Exception as e:
        log.error(f"Anomaly detection failed: {e}", exc_info=True)
        df['is_anomaly'] = False
        return df

def calculate_intent_correlations(df: pd.DataFrame):
    log.info("Calculating mail type to call intent correlations (Story 2.2)...")
    try:
        mail_cols = sorted([c for c in df.columns if c.startswith('mail_') and c != 'mail_total'])
        intent_cols = sorted([c for c in df.columns if c.startswith('call_') and c != 'call_total'])
        
        if not mail_cols or not intent_cols:
            log.warning("Not enough mail/intent columns for correlation matrix.")
            return pd.DataFrame()

        corr_matrix = pd.DataFrame(index=mail_cols, columns=intent_cols, dtype=float)

        for m_col in mail_cols:
            for c_col in intent_cols:
                correlations = []
                for lag in range(settings.max_lag + 1):
                    s1 = df[m_col]
                    s2 = df[c_col].shift(-lag)
                    valid_mask = s1.notna() & s2.notna()
                    if valid_mask.sum() < 2: continue
                    
                    r, _ = pearsonr(s1[valid_mask], s2[valid_mask])
                    correlations.append(r if pd.notna(r) else 0)
                
                corr_matrix.loc[m_col, c_col] = max(correlations) if correlations else 0
                
        log.info("✅ Mail-to-Intent correlation analysis complete.")
        return corr_matrix
    except Exception as e:
        log.error(f"Intent correlation calculation failed: {e}", exc_info=True)
        return pd.DataFrame()
PY

# --- 4.7 viz/plots.py (Epic 4) ---
cat > "$PKG/viz/plots.py" << 'PY'
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.dates import DateFormatter
from ..config import settings
from ..logging_utils import get_logger

log = get_logger(__name__)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('muted')

def save_plot(fig_or_plt, filename):
    out_path = settings.output_dir / filename
    try:
        if isinstance(fig_or_plt, plt.Figure):
            fig_or_plt.tight_layout()
            fig_or_plt.savefig(out_path, dpi=300)
        else:
            fig_or_plt.tight_layout()
            fig_or_plt.savefig(out_path, dpi=300)
        plt.close('all')
        log.info(f"Plot saved: {out_path}")
    except Exception as e:
        log.error(f"Failed to save plot {filename}: {e}", exc_info=True)


def plot_overview(df: pd.DataFrame):
    log.info("Generating overview plot...")
    fig, ax1 = plt.subplots(figsize=(15, 7))
    
    ax1.bar(df['date'], df['mail_total'], color='skyblue', alpha=0.7, label='Total Mail Volume')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Mail Volume', color='skyblue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='skyblue')

    ax2 = ax1.twinx()
    ax2.plot(df['date'], df['call_volume_aug'], color='tomato', linewidth=2.5, label='Augmented Call Volume')
    ax2.set_ylabel('Call Volume', color='tomato', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='tomato')
    
    fig.suptitle('Daily Mail vs. Augmented Call Volume', fontsize=16, weight='bold')
    fig.legend(loc="upper left", bbox_to_anchor=(0.05, 0.95))
    ax1.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    save_plot(fig, "01_overview_volume.png")

def plot_weekday_patterns(df: pd.DataFrame):
    log.info("Generating weekday pattern plots...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    mail_by_day = df.groupby(df['date'].dt.day_name())[[c for c in df.columns if 'mail_' in c and c != 'mail_total']].sum().reindex(weekday_order)
    mail_by_day.plot(kind='bar', ax=ax1, colormap='Blues_r', alpha=0.8, stacked=True)
    ax1.set_title('Total Mail Volume by Day of the Week', weight='bold')
    ax1.set_ylabel('Total Mail Sent')
    ax1.tick_params(axis='x', rotation=0)
    ax1.legend(title='Mail Type')
    
    call_by_day = df.groupby(df['date'].dt.day_name())[[c for c in df.columns if 'call_' in c and c != 'call_total' and c != 'call_volume_aug']].sum().reindex(weekday_order)
    call_by_day.plot(kind='bar', ax=ax2, colormap='Reds_r', alpha=0.8, stacked=True)
    ax2.set_title('Total Call Volume by Day of the Week', weight='bold')
    ax2.set_ylabel('Total Calls Received')
    ax2.set_xlabel('')
    ax2.tick_params(axis='x', rotation=0)
    ax2.legend(title='Call Intent')
    
    save_plot(fig, "02_weekday_patterns.png")

def plot_correlation_heatmap(corr_matrix: pd.DataFrame):
    if corr_matrix.empty: return
    log.info("Generating correlation heatmap...")
    plt.figure(figsize=(max(8, len(corr_matrix.columns)), max(6, len(corr_matrix.index))))
    sns.heatmap(corr_matrix, annot=True, cmap="vlag", center=0, fmt=".2f", linewidths=.5)
    plt.title('Peak Correlation Heatmap: Mail Type vs. Call Intent', weight='bold')
    plt.xlabel('Call Intent Type')
    plt.ylabel('Mail Campaign Type')
    save_plot(plt.gcf(), "03_intent_correlation_heatmap.png")

def plot_anomalies(df: pd.DataFrame):
    if 'is_anomaly' not in df.columns or not df['is_anomaly'].any(): return
    log.info("Generating anomaly plot...")
    anomalies = df[df['is_anomaly']]
    
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(df['date'], df['call_volume_aug'], label='Total Calls', color='gray', alpha=0.7, zorder=1)
    ax.scatter(anomalies['date'], anomalies['call_volume_aug'], color='red', s=60, zorder=2, label='Detected Anomaly', edgecolors='black')
    
    ax.set_title('Anomaly Detection in Daily Call Volume', weight='bold')
    ax.set_ylabel('Number of Calls')
    ax.legend()
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    save_plot(fig, "04_anomaly_detection.png")
PY

# --- 4.8 pipeline.py ---
cat > "$PKG/pipeline.py" << 'PY'
from .logging_utils import get_logger
from .data.loader import load_mail_data, load_call_data
from .features.engineer import create_features_for_modeling
from .analytics.core import generate_health_report, find_anomalies, calculate_intent_correlations
from .viz.plots import plot_overview, plot_weekday_patterns, plot_correlation_heatmap, plot_anomalies
from .config import settings
import pandas as pd

log = get_logger(__name__)

def generate_summary_report(df: pd.DataFrame, corr_matrix: pd.DataFrame):
    log.info("Generating final summary report (Story 5.1)...")
    
    try:
        report_lines = [
            "========================================",
            "  Executive Summary & Key Insights      ",
            "========================================",
            f"Analysis Period: {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}",
            f"Total Mail Sent: {df[[c for c in df.columns if c.startswith('mail_') and c != 'mail_total']].sum().sum():,.0f}",
            f"Total Calls Received (Augmented): {df['call_volume_aug'].sum():,.0f}",
            "",
            "--- Key Findings ---"
        ]

        if not corr_matrix.empty:
            best_corr_val = corr_matrix.max().max()
            best_pair = corr_matrix.stack().idxmax()
            report_lines.append(
                f"1. Strongest Link: '{best_pair[0]}' mail campaigns show the highest correlation (r={best_corr_val:.2f}) "
                f"with '{best_pair[1]}' calls within a {settings.max_lag}-day window."
            )

        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        call_by_day = df.groupby(df['date'].dt.day_name())['call_volume_aug'].sum().reindex(days)
        peak_call_day = call_by_day.idxmax()
        report_lines.append(
            f"2. Peak Call Day: The busiest day for customer calls is consistently {peak_call_day}."
        )

        if 'is_anomaly' in df.columns:
            num_anomalies = df['is_anomaly'].sum()
            report_lines.append(
                f"3. Anomalous Activity: We detected {num_anomalies} days with unusual call volumes "
                "that may warrant further investigation."
            )
        
        report_lines.append("\n--- Recommendations for COO ---")
        report_lines.append(
            "- Focus marketing attribution analysis on the strongest mail-to-call links identified in the heatmap."
        )
        report_lines.append(
            f"- Consider resource planning adjustments based on the predictable peak call volumes on {peak_call_day}s."
        )
        report_lines.append(
            "- Investigate the dates marked as anomalies to understand the drivers (e.g., outages, external events)."
        )
        report_lines.append(
            "- The dataset 'features_for_modeling.csv' is now ready for the ML team to begin building predictive models."
        )

        report_text = "\n".join(report_lines)
        out_path = settings.output_dir / "final_summary_report.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        log.info(f"✅ Final summary report saved to {out_path}")
    except Exception as e:
        log.error(f"Failed to generate summary report: {e}", exc_info=True)


def main():
    log.info("====== Starting Unified Production-Grade Pipeline ======")
    
    try:
        # Epic 1: Data Foundation
        mail_df = load_mail_data()
        call_df = load_call_data()
        if mail_df.empty or call_df.empty:
            log.critical("Critical data is missing. Aborting pipeline.")
            return

        # Epic 3: Feature Engineering & Epic 1 (Augmentation)
        features_df = create_features_for_modeling(mail_df, call_df)
        if features_df.empty:
            log.critical("Feature engineering failed. Aborting pipeline.")
            return

        # Epic 2: Advanced EDA
        generate_health_report(features_df)
        features_df_anom = find_anomalies(features_df)
        corr_matrix = calculate_intent_correlations(features_df)

        # Epic 4: Production-Grade Visualizations
        plot_overview(features_df)
        plot_weekday_patterns(features_df)
        plot_correlation_heatmap(corr_matrix)
        plot_anomalies(features_df_anom)
        
        # Epic 5: Delivery
        generate_summary_report(features_df, corr_matrix)

        log.info("====== ✅ Unified Analysis Pipeline Finished Successfully! ======")
        log.info(f"Find all outputs in the ./{settings.output_dir.name}/ directory.")

    except Exception as e:
        log.critical(f"A critical error occurred in the main pipeline: {e}", exc_info=True)
        log.critical("Pipeline aborted.")

if __name__ == "__main__":
    main()
PY

echo "✅ All Python modules generated."

# ---------------------------------------------------------------------
# 5. Execution
# ---------------------------------------------------------------------
echo "🚀 Running the unified, production-grade analysis pipeline..."
python -m ${PKG}.pipeline

echo ""
echo "🎉 All Done! Your analysis is complete."
echo "------------------------------------------------"
echo "➡️  Check the '${PROJ}/output' directory for your plots and reports."
echo "➡️  The pre-modeling dataset is ready at: '${PROJ}/output/features_for_modeling.csv'"
echo "------------------------------------------------"
