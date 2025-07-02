Of course. Here is the complete, final script.
The only part you need to edit is the CONFIG section at the top. I have added comments inside the code to highlight exactly which lines you need to change.
## Complete Analysis & Modeling Script
Below is the entire Python file. You just need to modify the highlighted lines within the CONFIG block to point to your files and match your column names.
#!/usr/bin/env python3
"""
Definitive Marketing & Financial Analysis Pipeline

This script performs a complete, end-to-end analysis of mail and call data.
It includes robust data processing, augmentation of missing data, automated
predictive modeling (SARIMAX, Prophet, RandomForest), and generation of a
comprehensive set of static plots and reports.
"""

# --- Core Libraries ---
import pandas as pd
import numpy as np
import warnings
import os
import sys
import json
import traceback
from datetime import datetime
import logging

# --- Visualization ---
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# --- Data & Modeling ---
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from prophet import Prophet

# Suppress common warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# ENHANCED LOGGING SETUP (WITH COLORS FOR TERMINAL)
# =============================================================================

class CustomFormatter(logging.Formatter):
    """Custom logger formatter with colors for terminal output."""
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    green = "\x1b[32;20m"
    reset = "\x1b[0m"
    format_str = "%(asctime)s - %(levelname)-8s - %(message)s"

    FORMATS = {
        logging.INFO: green + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.grey + self.format_str + self.reset)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger('AdvancedAnalysis')
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    ch = logging.StreamHandler()
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(output_dir, f'analysis_log_{datetime.now().strftime("%Y%m%d")}.txt'))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    
    return logger

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # --- 1. SET YOUR FILE PATHS HERE ---
    'MAIL_FILE_PATH': r'C:\path\to\your\mail_data.csv',        # <-- EDIT THIS
    'CALL_FILE_PATH': r'C:\path\to\your\call_data.csv',        # <-- EDIT THIS
    'OUTPUT_DIR': r'C:\path\to\output\final_analysis',            # <-- EDIT THIS

    # --- 2. MAP YOUR COLUMN NAMES HERE ---
    # The script will use these to find the right columns in your files.
    'MAIL_COLUMNS': {
        'date': 'date',          # <-- EDIT with the name of the date column in your mail file
        'volume': 'volume',      # <-- EDIT with the name of the mail volume/quantity column
        'type': 'mail_type'      # <-- EDIT with the name of the column specifying the mail type
    },
    'CALL_COLUMNS': {
        'date': 'date',          # <-- EDIT with the name of the date column in your call file
        'intent': 'intent'       # <-- EDIT with the name of the column specifying the call reason/intent
    },

    # --- Other Settings (Usually No Need to Change) ---
    'FINANCIAL_DATA': {
        'S&P 500': '^GSPC',
        '10-Yr Treasury Yield': '^TNX'
    },
    'PLOT_STYLE': 'seaborn-v0_8-whitegrid',
    'FIGURE_SIZE': (18, 9),
    'DPI': 300
}

# =============================================================================
# === DO NOT EDIT BELOW THIS LINE =============================================
# =============================================================================

class MarketingAnalyzer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.mail_df = self.call_df = self.financial_df = None
        self.daily_data = self.augmented_data = None
        self.model_results = {}
        
        self.plots_dir = os.path.join(config['OUTPUT_DIR'], 'plots')
        self.reports_dir = os.path.join(config['OUTPUT_DIR'], 'reports')
        for dir_path in [self.plots_dir, self.reports_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        plt.style.use(config['PLOT_STYLE'])

    def run_pipeline(self):
        """Executes the entire analysis and modeling pipeline."""
        self.logger.info("Starting the Advanced Analysis Pipeline...")
        try:
            if not self._load_and_process_data(): return
            if not self._augment_missing_data(): return
            if not self._run_modeling_suite(): return
            if not self._create_visualizations(): return
            self.logger.info("✅ Pipeline completed successfully! Check the output folder.")
        except Exception as e:
            self.logger.critical(f"A critical error stopped the pipeline: {e}")
            self.logger.error(traceback.format_exc())

    def _load_and_process_data(self):
        """Loads, cleans, and integrates all data sources using CONFIG mappings."""
        self.logger.info("STEP 1: Loading and processing initial data...")
        try:
            mail_cols = self.config['MAIL_COLUMNS']
            self.mail_df = pd.read_csv(self.config['MAIL_FILE_PATH'])
            self.mail_df.rename(columns={v: k for k, v in mail_cols.items()}, inplace=True)
            self.mail_df['date'] = pd.to_datetime(self.mail_df['date'])

            call_cols = self.config['CALL_COLUMNS']
            self.call_df = pd.read_csv(self.config['CALL_FILE_PATH'])
            self.call_df.rename(columns={v: k for k, v in call_cols.items()}, inplace=True)
            self.call_df['date'] = pd.to_datetime(self.call_df['date'])

            mail_summary = self.mail_df.groupby(pd.Grouper(key='date', freq='D'))['volume'].sum().rename('mail_volume')
            call_summary = self.call_df.groupby(pd.Grouper(key='date', freq='D')).size().rename('call_volume')
            self.daily_data = pd.concat([mail_summary, call_summary], axis=1).fillna(0)
            
            start, end = self.daily_data.index.min(), self.daily_data.index.max()
            tickers = self.config['FINANCIAL_DATA']
            self.financial_df = yf.download(list(tickers.values()), start=start, end=end, progress=False)['Adj Close']
            self.financial_df.rename(columns={v: k for k, v in tickers.items()}, inplace=True)
            self.daily_data = self.daily_data.join(self.financial_df).ffill().bfill()
            
            self.logger.info("✓ Data loaded and initial table created.")
            return True
        except FileNotFoundError as e:
            self.logger.error(f"File not found. Please check paths in CONFIG. Details: {e}")
            return False
        except KeyError as e:
            self.logger.error(f"Column not found. Please check column name mappings in CONFIG. Details: {e}")
            return False

    def _augment_missing_data(self):
        """Identifies and augments a year-long gap in call data."""
        self.logger.info("STEP 2: Searching for and augmenting missing call data...")
        self.augmented_data = self.daily_data.copy()
        
        full_range = pd.date_range(start=self.augmented_data.index.min(), end=self.augmented_data.index.max())
        self.augmented_data = self.augmented_data.reindex(full_range, fill_value=np.nan)
        
        missing_dates = self.augmented_data[self.augmented_data['call_volume'].isnull()].index
        if len(missing_dates) < 30:
            self.logger.info("No significant data gap found to augment.")
            self.augmented_data.fillna(0, inplace=True)
            self.augmented_dates = pd.Index([])
            return True
            
        self.logger.warning(f"Found a significant data gap of {len(missing_dates)} days. Augmenting now...")
        
        self.augmented_data['call_volume'] = self.augmented_data['call_volume'].interpolate(method='time')
        self.augmented_data.fillna(0, inplace=True)
        self.augmented_dates = missing_dates
        
        self.logger.info("✓ Data augmentation complete.")
        return True

    def _run_modeling_suite(self):
        """Runs multiple predictive models on both original and augmented data."""
        self.logger.info("STEP 3: Running predictive modeling suite...")
        
        datasets = {
            "Original_Data": self.daily_data,
            "Augmented_Data": self.augmented_data
        }

        for name, df in datasets.items():
            self.logger.info(f"--- Modeling on: {name.replace('_', ' ')} ---")
            
            df_ml = df.copy()
            df_ml['mail_volume_lag1'] = df_ml['mail_volume'].shift(1)
            df_ml.dropna(inplace=True)
            
            X = df_ml.drop('call_volume', axis=1)
            y = df_ml['call_volume']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            try:
                sarimax_model = sm.tsa.statespace.SARIMAX(y_train, exog=X_train, order=(1,1,1), seasonal_order=(1,1,1,7)).fit(disp=False)
                sarimax_pred = sarimax_model.predict(start=X_test.index[0], end=X_test.index[-1], exog=X_test)
                self.model_results[f'SARIMAX_{name}'] = {'test': y_test, 'pred': sarimax_pred}
                self.logger.info("✓ SARIMAX model trained successfully.")
            except Exception as e:
                self.logger.warning(f"SARIMAX model failed for {name}: {e}")

            try:
                prophet_df = df.reset_index().rename(columns={'index': 'ds', 'call_volume': 'y'})
                prophet_train = prophet_df.iloc[:len(X_train)]
                
                prophet_model = Prophet()
                for col in X.columns: prophet_model.add_regressor(col)
                prophet_model.fit(prophet_train)
                
                future = prophet_model.make_future_dataframe(periods=len(y_test))
                for col in X.columns: future[col] = df[col].values
                
                prophet_pred_df = prophet_model.predict(future)
                prophet_pred = prophet_pred_df.iloc[-len(y_test):]['yhat']
                prophet_pred.index = y_test.index
                self.model_results[f'Prophet_{name}'] = {'test': y_test, 'pred': prophet_pred}
                self.logger.info("✓ Prophet model trained successfully.")
            except Exception as e:
                self.logger.warning(f"Prophet model failed for {name}: {e}")

            try:
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X_train, y_train)
                rf_pred = rf_model.predict(X_test)
                rf_pred = pd.Series(rf_pred, index=y_test.index)
                self.model_results[f'RandomForest_{name}'] = {'test': y_test, 'pred': rf_pred, 'model': rf_model, 'features': X.columns}
                self.logger.info("✓ Random Forest model trained successfully.")
            except Exception as e:
                self.logger.warning(f"Random Forest model failed for {name}: {e}")
        return True
        
    def _create_visualizations(self):
        """Creates and saves all plots."""
        self.logger.info("STEP 4: Creating and saving all visualizations...")

        # Plot 1: Augmented Data Overview
        fig, ax = plt.subplots(figsize=self.config['FIGURE_SIZE'])
        ax.plot(self.daily_data.index, self.daily_data['call_volume'], label='Original Calls', color='blue', zorder=10)
        if not self.augmented_dates.empty:
            ax.plot(self.augmented_data.loc[self.augmented_dates].index, self.augmented_data.loc[self.augmented_dates]['call_volume'], label='Augmented Calls', color='red', linestyle='--', zorder=11)
        ax.set(title='Call Volume: Original vs. Augmented Data', xlabel='Date', ylabel='Call Volume')
        ax.legend()
        plt.savefig(os.path.join(self.plots_dir, '01_data_augmentation_overview.png'), dpi=self.config['DPI'])
        plt.close()

        # Plot 2: Intent & Mail Type Analysis
        try:
            intent_summary = self.call_df.groupby([pd.Grouper(key='date', freq='W'), 'intent']).size().unstack(fill_value=0)
            mail_type_summary = self.mail_df.groupby([pd.Grouper(key='date', freq='W'), 'type'])['volume'].sum().unstack(fill_value=0)
            
            fig, ax1 = plt.subplots(figsize=self.config['FIGURE_SIZE'])
            intent_summary.plot(kind='area', stacked=True, ax=ax1, alpha=0.8, cmap='Paired', linewidth=0)
            ax1.set_ylabel('Weekly Call Count (by Intent)', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')

            ax2 = ax1.twinx()
            mail_type_summary.plot(kind='line', ax=ax2, alpha=0.9, style='--')
            ax2.set_ylabel('Weekly Mail Volume (by Type)', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            ax1.set(title='Weekly Call Intents (Area) vs. Mail Types (Dashed Lines)', xlabel='Date')
            plt.savefig(os.path.join(self.plots_dir, '02_intent_vs_mailtype.png'), dpi=self.config['DPI'])
            plt.close()
            self.logger.info("✓ Intent vs. Mail Type plot created.")
        except Exception as e:
            self.logger.warning(f"Could not create intent plot: {e}")

        # Plots 3 onwards: Model Forecasts
        plot_index = 3
        for model_name, results in self.model_results.items():
            if 'pred' not in results: continue
            fig, ax = plt.subplots(figsize=self.config['FIGURE_SIZE'])
            ax.plot(self.augmented_data.index, self.augmented_data['call_volume'], label='Full Call Volume History', color='gray', alpha=0.5)
            if not self.augmented_dates.empty:
                 ax.plot(self.augmented_data.loc[self.augmented_dates].index, self.augmented_data.loc[self.augmented_dates]['call_volume'], color='red', linestyle='', marker='.', markersize=2, label='Augmented Data Points')
            ax.plot(results['test'].index, results['test'], label='Actual Test Data', color='black', linewidth=2)
            ax.plot(results['pred'].index, results['pred'], label=f'Forecast', color='cyan', linestyle='--', linewidth=2.5)
            ax.set(title=f'Model Forecast: {model_name.replace("_", " ")}', xlabel='Date', ylabel='Call Volume')
            ax.legend()
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            plt.savefig(os.path.join(self.plots_dir, f'{plot_index:02d}_forecast_{model_name}.png'), dpi=self.config['DPI'])
            plt.close()
            plot_index += 1

        # Plot: RF Feature Importance
        try:
            rf_results = self.model_results['RandomForest_Augmented_Data']
            importances = pd.Series(rf_results['model'].feature_importances_, index=rf_results['features'])
            fig, ax = plt.subplots(figsize=(10, 6))
            importances.sort_values().plot(kind='barh', ax=ax)
            ax.set_title('Feature Importance (Random Forest on Augmented Data)')
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, '99_feature_importance.png'), dpi=self.config['DPI'])
            plt.close()
            self.logger.info("✓ Feature Importance plot created.")
        except KeyError:
            pass
        self.logger.info("✓ All visualizations saved.")
        return True

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================
if __name__ == '__main__':
    if not os.path.exists(CONFIG['MAIL_FILE_PATH']) or not os.path.exists(CONFIG['CALL_FILE_PATH']):
        print(f"\x1b[31;1mERROR: Cannot find data files. Please check paths in CONFIG section.\x1b[0m")
        sys.exit(1)
        
    logger = setup_logging(CONFIG['OUTPUT_DIR'])
    analyzer = MarketingAnalyzer(CONFIG, logger)
    analyzer.run_pipeline()

