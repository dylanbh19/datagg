#!/usr/bin/env python3
"""
Definitive Marketing & Financial Analysis Pipeline

This script performs a complete, end-to-end analysis of mail and call data.
It includes robust data processing with flexible date handling and removal logging,
augmentation of missing data, automated predictive modeling, and generation of a
comprehensive set of static plots and reports.
"""

# --- Core Libraries ---
import pandas as pd
import numpy as np
import warnings
import os
import sys
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
from sklearn.metrics import mean_squared_error
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
    """Sets up dual-output logging to file and a colorful terminal."""
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger('AdvancedAnalysis')
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console Handler (with colors)
    ch = logging.StreamHandler()
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)

    # File Handler (without colors)
    fh = logging.FileHandler(os.path.join(output_dir, f'analysis_log_{datetime.now().strftime("%Y%m%d")}.txt'))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s\n%(pathname)s:%(lineno)d'))
    logger.addHandler(fh)
    
    return logger

# =============================================================================
# CONFIGURATION - UPDATED WITH YOUR SETTINGS
# =============================================================================
CONFIG = {
    # --- Your File Paths ---
    'MAIL_FILE_PATH': r'merged_output.csv',
    'CALL_FILE_PATH': r'data\GenesysExtract_20250609.csv',
    'OUTPUT_DIR': r'output\plots\reports',

    # --- Your Column Mappings ---
    'MAIL_COLUMNS': {
        'date': 'mail date',
        'volume': 'mail_volume',
        'type': 'mail_type'
    },
    'CALL_COLUMNS': {
        'date': 'ConversationStart',
        'intent': 'vui_intent'
    },

    # --- Other Settings ---
    'FINANCIAL_DATA': {
        'S&P 500': '^GSPC',
        '10-Yr Treasury Yield': '^TNX'
    },
    'PLOT_STYLE': 'seaborn-v0_8-whitegrid',
    'FIGURE_SIZE': (20, 10),
    'DPI': 300
}

# =============================================================================
# === DO NOT EDIT BELOW THIS LINE =============================================
# =============================================================================

class MarketingAnalyzer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        # Initialize all data attributes
        self.mail_df = self.call_df = self.financial_df = None
        self.daily_data = self.augmented_data = None
        self.model_results = {}
        
        # Create output directories
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
            self.logger.info("✅ Pipeline completed successfully! Check the output folder: %s", self.config['OUTPUT_DIR'])
        except Exception as e:
            self.logger.critical(f"A critical error stopped the pipeline: {e}")
            self.logger.error(traceback.format_exc())

    def _load_and_process_data(self):
        """Loads, cleans, and integrates all data sources with robust date handling."""
        self.logger.info("STEP 1: Loading and processing initial data...")
        try:
            removed_rows = []

            # --- Load and Process Mail Data ---
            mail_cols = self.config['MAIL_COLUMNS']
            self.mail_df = pd.read_csv(self.config['MAIL_FILE_PATH'])
            self.mail_df.rename(columns={v: k for k, v in mail_cols.items()}, inplace=True)
            
            self.mail_df['original_date'] = self.mail_df['date'] # Keep original for logging
            self.mail_df['date'] = pd.to_datetime(self.mail_df['date'], errors='coerce')
            invalid_mail_rows = self.mail_df[self.mail_df['date'].isnull()]
            if not invalid_mail_rows.empty:
                self.logger.warning(f"Found and removed {len(invalid_mail_rows)} rows with unreadable dates in the mail file.")
                invalid_mail_rows['removal_reason'] = 'Unreadable Date'
                removed_rows.append(invalid_mail_rows)
                self.mail_df.dropna(subset=['date'], inplace=True)

            # --- Load and Process Call Data ---
            call_cols = self.config['CALL_COLUMNS']
            self.call_df = pd.read_csv(self.config['CALL_FILE_PATH'])
            self.call_df.rename(columns={v: k for k, v in call_cols.items()}, inplace=True)
            
            self.call_df['original_date'] = self.call_df['date']
            self.call_df['date'] = pd.to_datetime(self.call_df['date'], errors='coerce')
            invalid_call_rows = self.call_df[self.call_df['date'].isnull()]
            if not invalid_call_rows.empty:
                self.logger.warning(f"Found and removed {len(invalid_call_rows)} rows with unreadable dates in the call file.")
                invalid_call_rows['removal_reason'] = 'Unreadable Date'
                removed_rows.append(invalid_call_rows)
                self.call_df.dropna(subset=['date'], inplace=True)

            # --- Save Removed Rows to File ---
            if removed_rows:
                removed_df = pd.concat(removed_rows, ignore_index=True)
                removed_filepath = os.path.join(self.config['OUTPUT_DIR'], 'removed_rows_log.csv')
                removed_df.to_csv(removed_filepath, index=False)
                self.logger.info(f"✓ Details of {len(removed_df)} removed rows saved to: {removed_filepath}")

            # --- Aggregate Data ---
            mail_summary = self.mail_df.groupby(pd.Grouper(key='date', freq='D'))['volume'].sum().rename('mail_volume')
            call_summary = self.call_df.groupby(pd.Grouper(key='date', freq='D')).size().rename('call_volume')
            self.daily_data = pd.concat([mail_summary, call_summary], axis=1).fillna(0)
            
            # --- Definitive Fix for Financial Data ---
            self.logger.info("Fetching financial data...")
            start, end = self.daily_data.index.min(), self.daily_data.index.max()
            tickers = self.config['FINANCIAL_DATA']
            raw_financial_df = yf.download(list(tickers.values()), start=start, end=end, progress=False)
            
            self.financial_df = pd.DataFrame(index=raw_financial_df.index)
            for friendly_name, ticker in tickers.items():
                # Try to get price data from common column names, making it robust
                for price_type in ['Adj Close', 'Close', 'Open']:
                    if (price_type, ticker) in raw_financial_df.columns:
                        self.financial_df[friendly_name] = raw_financial_df[(price_type, ticker)]
                        break # Found a price, move to next ticker
                else: # This else belongs to the for loop, runs if `break` was not hit
                    self.logger.warning(f"Could not find price data for ticker '{ticker}'. It will be skipped.")
            
            self.daily_data = self.daily_data.join(self.financial_df)
            self.daily_data.ffill(inplace=True)
            self.daily_data.bfill(inplace=True)
            self.daily_data.fillna(0, inplace=True) # Fill any remaining NaNs
            
            self.logger.info("✓ Data loaded and initial table created.")
            return True
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during data loading: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def _augment_missing_data(self):
        """Identifies and augments large gaps in call data."""
        self.logger.info("STEP 2: Searching for and augmenting missing call data...")
        self.augmented_data = self.daily_data.copy()
        
        full_range = pd.date_range(start=self.augmented_data.index.min(), end=self.augmented_data.index.max())
        self.augmented_data = self.augmented_data.reindex(full_range, fill_value=np.nan)
        
        missing_dates = self.augmented_data[self.augmented_data['call_volume'].isnull()].index
        
        if len(missing_dates) < 30:
            self.logger.info("No significant data gap found to augment. Filling minor gaps with 0.")
            self.augmented_data.fillna(0, inplace=True)
            self.augmented_dates = pd.Index([])
            return True
            
        self.logger.warning(f"Found a significant data gap of {len(missing_dates)} days. Augmenting with interpolated values...")
        
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
            df_ml['mail_volume_lag1'] = df_ml['mail_volume'].shift(1).fillna(0)
            df_ml['mail_volume_lag7'] = df_ml['mail_volume'].shift(7).fillna(0)
            df_ml.dropna(inplace=True)
            
            X = df_ml.drop('call_volume', axis=1)
            y = df_ml['call_volume']
            train_size = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
            y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
            
            try:
                self.logger.info("Training SARIMAX model...")
                sarimax_model = sm.tsa.statespace.SARIMAX(y_train, exog=X_train, order=(1,1,1), seasonal_order=(1,1,1,7)).fit(disp=False)
                sarimax_pred = sarimax_model.predict(start=X_test.index[0], end=X_test.index[-1], exog=X_test)
                self.model_results[f'SARIMAX_{name}'] = {'test': y_test, 'pred': sarimax_pred}
                self.logger.info("✓ SARIMAX model trained successfully.")
            except Exception as e:
                self.logger.warning(f"SARIMAX model failed for {name}. Details: {e}")

            try:
                self.logger.info("Training Prophet model...")
                prophet_df = df_ml.reset_index().rename(columns={'index': 'ds', 'call_volume': 'y'})
                prophet_train = prophet_df.iloc[:train_size]
                
                prophet_model = Prophet()
                for col in X.columns: prophet_model.add_regressor(col)
                prophet_model.fit(prophet_train, show_stan_stdout=False)
                
                future = prophet_df.iloc[train_size:][['ds'] + list(X.columns)]
                
                prophet_pred_df = prophet_model.predict(future)
                prophet_pred = prophet_pred_df['yhat']
                prophet_pred.index = y_test.index
                self.model_results[f'Prophet_{name}'] = {'test': y_test, 'pred': prophet_pred}
                self.logger.info("✓ Prophet model trained successfully.")
            except Exception as e:
                self.logger.warning(f"Prophet model failed for {name}. Details: {e}")

            try:
                self.logger.info("Training Random Forest model...")
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                rf_model.fit(X_train, y_train)
                rf_pred = rf_model.predict(X_test)
                rf_pred = pd.Series(rf_pred, index=y_test.index)
                self.model_results[f'RandomForest_{name}'] = {'test': y_test, 'pred': rf_pred, 'model': rf_model, 'features': X.columns}
                self.logger.info("✓ Random Forest model trained successfully.")
            except Exception as e:
                self.logger.warning(f"Random Forest model failed for {name}. Details: {e}")
        return True
            
    def _create_visualizations(self):
        """Creates and saves all plots."""
        self.logger.info("STEP 4: Creating and saving all visualizations...")

        self.logger.info("Creating plot 1: Data Augmentation Overview...")
        fig, ax = plt.subplots(figsize=self.config['FIGURE_SIZE'])
        ax.plot(self.daily_data.index, self.daily_data['call_volume'], label='Original Calls', color='blue', zorder=10)
        if not self.augmented_dates.empty:
            ax.plot(self.augmented_data.loc[self.augmented_dates].index, self.augmented_data.loc[self.augmented_dates]['call_volume'], 
                    label='Augmented Call Data', color='red', linestyle='--', zorder=11)
        ax.set(title='Call Volume: Original vs. Augmented Data', xlabel='Date', ylabel='Call Volume')
        ax.legend()
        plt.savefig(os.path.join(self.plots_dir, '01_data_augmentation_overview.png'), dpi=self.config['DPI'])
        plt.close()

        self.logger.info("Creating plot 2: Intent vs. Mail Type Analysis...")
        try:
            intent_summary = self.call_df.groupby([pd.Grouper(key='date', freq='W'), 'intent']).size().unstack(fill_value=0)
            mail_type_summary = self.mail_df.groupby([pd.Grouper(key='date', freq='W'), 'type'])['volume'].sum().unstack(fill_value=0)
            
            fig, ax1 = plt.subplots(figsize=self.config['FIGURE_SIZE'])
            intent_summary.plot(kind='area', stacked=True, ax=ax1, alpha=0.8, cmap='Paired', linewidth=0)
            ax1.set_ylabel('Weekly Call Count (by Intent)', color='blue', fontsize=14)
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.legend(title='Call Intents', loc='upper left')

            ax2 = ax1.twinx()
            mail_type_summary.plot(kind='line', ax=ax2, alpha=0.9, style='--', marker='o', markersize=4)
            ax2.set_ylabel('Weekly Mail Volume (by Type)', color='red', fontsize=14)
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.legend(title='Mail Types', loc='upper right')
            
            ax1.set(title='Weekly Call Intents (Area) vs. Mail Types (Dashed Lines)', xlabel='Date')
            fig.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, '02_intent_vs_mailtype.png'), dpi=self.config['DPI'])
            plt.close()
        except Exception as e:
            self.logger.warning(f"Could not create intent vs mail type plot. Check 'intent' and 'type' columns. Details: {e}")

        self.logger.info("Creating model forecast plots...")
        plot_index = 3
        for model_name, results in self.model_results.items():
            if 'pred' not in results: continue
            
            fig, ax = plt.subplots(figsize=self.config['FIGURE_SIZE'])
            ax.plot(self.augmented_data.index, self.augmented_data['call_volume'], label='Full Call History', color='gray', alpha=0.4)
            if not self.augmented_dates.empty:
                 ax.plot(self.augmented_data.loc[self.augmented_dates].index, self.augmented_data.loc[self.augmented_dates]['call_volume'], 
                         color='red', linestyle='', marker='.', markersize=2, label='Augmented Data Points')
            ax.plot(results['test'].index, results['test'], label='Actual Test Data', color='black', linewidth=2)
            ax.plot(results['pred'].index, results['pred'], label='Forecast', color='cyan', linestyle='--', linewidth=2.5)
            
            ax.set(title=f'Model Forecast: {model_name.replace("_", " ")}', xlabel='Date', ylabel='Call Volume')
            ax.legend()
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            fig.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, f'{plot_index:02d}_forecast_{model_name}.png'), dpi=self.config['DPI'])
            plt.close()
            plot_index += 1

        self.logger.info("Creating plot: Feature Importance...")
        try:
            rf_results = self.model_results['RandomForest_Augmented_Data']
            importances = pd.Series(rf_results['model'].feature_importances_, index=rf_results['features'])
            fig, ax = plt.subplots(figsize=(10, 8))
            importances.sort_values().plot(kind='barh', ax=ax)
            ax.set_title('Feature Importance (Which factors predict call volume?)')
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, '99_feature_importance.png'), dpi=self.config['DPI'])
            plt.close()
        except KeyError:
            self.logger.warning("Could not create feature importance plot (Random Forest model might have failed).")
            
        self.logger.info("✓ All visualizations saved.")
        return True

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================
if __name__ == '__main__':
    if not os.path.exists(CONFIG['MAIL_FILE_PATH']) or not os.path.exists(CONFIG['CALL_FILE_PATH']):
        print(f"\x1b[31;1mCRITICAL ERROR: Cannot find data files. Please check paths in CONFIG section.\x1b[0m")
        sys.exit(1)
        
    logger = setup_logging(CONFIG['OUTPUT_DIR'])
    analyzer = MarketingAnalyzer(CONFIG, logger)
    analyzer.run_pipeline()
