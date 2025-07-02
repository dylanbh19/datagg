#!/usr/bin/env python3
"""
Comprehensive and Robust Exploratory Data Analysis (EDA) Pipeline

This script performs a deep EDA on mail and call data. It includes corrected
data augmentation with timeline extension, robust date handling, feature
engineering, and a comprehensive set of clear, readable static plots.
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
from sklearn.preprocessing import MinMaxScaler

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
    logger = logging.getLogger('EDA_Analysis')
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    ch = logging.StreamHandler()
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(output_dir, f'eda_log_{datetime.now().strftime("%Y%m%d")}.txt'))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s\n%(pathname)s:%(lineno)d'))
    logger.addHandler(fh)
    
    return logger

# =============================================================================
# CONFIGURATION - Your settings are saved here
# =============================================================================
CONFIG = {
    'MAIL_FILE_PATH': r'merged_output.csv',
    'CALL_FILE_PATH': r'data\GenesysExtract_20250609.csv',
    'OUTPUT_DIR': r'output\plots\reports',

    'MAIL_COLUMNS': {
        'date': 'mail date',
        'volume': 'mail_volume',
        'type': 'mail_type'
    },
    'CALL_COLUMNS': {
        'date': 'ConversationStart',
        'intent': 'vui_intent'
    },

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

class EDAAnalyzer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.mail_df = self.call_df = self.financial_df = None
        self.daily_data = self.augmented_data = self.feature_engineered_data = None
        
        self.plots_dir = os.path.join(config['OUTPUT_DIR'], 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        
        plt.style.use(config['PLOT_STYLE'])

    def run_eda_pipeline(self):
        """Executes the entire Exploratory Data Analysis pipeline."""
        self.logger.info("Starting the Comprehensive EDA Pipeline...")
        try:
            if not self._load_and_process_data(): return
            if not self._augment_call_data(): return
            if not self._feature_engineering(): return
            if not self._create_eda_visualizations(): return
            self.logger.info("✅ EDA Pipeline completed successfully! Check the output folder: %s", self.plots_dir)
        except Exception as e:
            self.logger.critical(f"A critical error stopped the pipeline: {e}")
            self.logger.error(traceback.format_exc())

    def _load_and_process_data(self):
        """Loads, cleans, and integrates all data sources with robust date handling."""
        self.logger.info("STEP 1: Loading and processing initial data...")
        try:
            mail_cols = self.config['MAIL_COLUMNS']
            self.mail_df = pd.read_csv(self.config['MAIL_FILE_PATH'], encoding='utf-8', on_bad_lines='warn')
            self.mail_df.rename(columns={v: k for k, v in mail_cols.items()}, inplace=True)
            self.mail_df['date'] = pd.to_datetime(self.mail_df['date'], errors='coerce').dt.normalize()
            self.mail_df.dropna(subset=['date'], inplace=True)
            self.mail_df.set_index('date', inplace=True)

            call_cols = self.config['CALL_COLUMNS']
            self.call_df = pd.read_csv(self.config['CALL_FILE_PATH'], encoding='utf-8', on_bad_lines='warn')
            self.call_df.rename(columns={v: k for k, v in call_cols.items()}, inplace=True)
            self.call_df['date'] = pd.to_datetime(self.call_df['date'], errors='coerce').dt.normalize()
            self.call_df.dropna(subset=['date'], inplace=True)
            self.call_df.set_index('date', inplace=True)

            mail_summary = self.mail_df.groupby(self.mail_df.index)['volume'].sum().rename('mail_volume')
            call_summary = self.call_df.groupby(self.call_df.index).size().rename('call_volume')
            self.daily_data = pd.concat([mail_summary, call_summary], axis=1).fillna(0)
            
            self.logger.info("Fetching financial data...")
            start, end = self.daily_data.index.min(), self.daily_data.index.max()
            tickers = self.config['FINANCIAL_DATA']
            raw_financial_df = yf.download(list(tickers.values()), start=start, end=end, progress=False)
            
            self.financial_df = pd.DataFrame(index=raw_financial_df.index)
            for friendly_name, ticker in tickers.items():
                for price_type in ['Adj Close', 'Close', 'Open']:
                    if (price_type, ticker) in raw_financial_df.columns:
                        self.financial_df[friendly_name] = raw_financial_df[(price_type, ticker)]
                        break
                else:
                    self.logger.warning(f"Could not find price data for ticker '{ticker}'.")
            
            self.daily_data = self.daily_data.join(self.financial_df).ffill().bfill().fillna(0)
            self.logger.info("✓ Data loaded and initial daily summary created.")
            return True
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during data loading: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def _augment_call_data(self):
        """Correctly augments call data to match the earliest mail date."""
        self.logger.info("STEP 2: Augmenting missing call data...")
        
        # --- START OF FIX: Correct Augmentation Logic ---
        mail_start_date = self.mail_df.index.min()
        combined_start_date = min(mail_start_date, self.daily_data.index.min())
        combined_end_date = self.daily_data.index.max()

        self.logger.info(f"Full analysis date range set from {combined_start_date.date()} to {combined_end_date.date()}.")
        
        # Create a complete date range for the final augmented dataset
        full_range = pd.date_range(start=combined_start_date, end=combined_end_date, freq='D')
        
        # Reindex the daily data to this full range, creating NaNs where data is missing
        self.augmented_data = self.daily_data.reindex(full_range)

        # Identify all dates that were originally NaN, including the new extended period
        self.augmented_dates = self.augmented_data[self.augmented_data['call_volume'].isnull()].index
        
        # First, fill all NaNs with 0 to provide a solid baseline for interpolation
        self.augmented_data.fillna(0, inplace=True)
        
        # If there were any original gaps (not just the extended timeline), interpolate them now
        # This part is more for internal gaps, but is safe to run
        internal_gaps = self.augmented_data.loc[self.daily_data.index.min():]['call_volume'].replace(0, np.nan)
        internal_gaps_interpolated = internal_gaps.interpolate(method='time')
        self.augmented_data.loc[internal_gaps_interpolated.index, 'call_volume'] = internal_gaps_interpolated.values
        # --- END OF FIX ---
        
        self.logger.info("✓ Data augmentation step complete.")
        return True

    def _feature_engineering(self):
        """Creates new features for analysis from the existing data."""
        self.logger.info("STEP 3: Performing feature engineering...")
        df = self.augmented_data.copy()

        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['week_of_year'] = df.index.isocalendar().week
        df['call_volume_rolling_7d'] = df['call_volume'].rolling(window=7).mean()
        df['mail_volume_lag_1d'] = df['mail_volume'].shift(1)
        df['mail_volume_lag_7d'] = df['mail_volume'].shift(7)
        
        self.feature_engineered_data = df.dropna()
        self.logger.info("✓ New features created (day of week, rolling averages, lags).")
        return True

    def _create_eda_visualizations(self):
        """Creates and saves a comprehensive set of EDA plots with fixes."""
        self.logger.info("STEP 4: Creating all EDA visualizations...")

        # Plot 1: Overall Mail and Call Volume (FIXED with dual axis)
        self.logger.info("Creating plot: Overall Mail vs. Call Volume (Dual Axis)...")
        fig, ax1 = plt.subplots(figsize=self.config['FIGURE_SIZE'])
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Daily Call Volume', color='navy', fontsize=14)
        ax1.plot(self.augmented_data.index, self.augmented_data['call_volume'], color='navy', label='Call Volume')
        ax1.tick_params(axis='y', labelcolor='navy')
        ax2 = ax1.twinx()
        ax2.set_ylabel('Daily Mail Volume', color='skyblue', fontsize=14)
        ax2.bar(self.augmented_data.index, self.augmented_data['mail_volume'], color='skyblue', label='Mail Volume', alpha=0.7)
        ax2.tick_params(axis='y', labelcolor='skyblue')
        fig.suptitle('Overall Daily Mail and Call Volumes', fontsize=18, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(self.plots_dir, '01_overall_volumes_dual_axis.png'), dpi=self.config['DPI'])
        plt.close(fig)

        # Plot 2: Call Volume by Intent (FIXED with robust fallback)
        self.logger.info("Creating plot: Call Volume by Intent...")
        if 'intent' in self.call_df.columns:
            intent_summary = self.call_df.groupby([pd.Grouper(freq='W'), 'intent']).size().unstack(fill_value=0)
            if not intent_summary.empty:
                intent_summary.plot(kind='area', stacked=True, figsize=self.config['FIGURE_SIZE'], cmap='Paired', linewidth=0)
                plt.title('Weekly Call Volume by Intent', fontsize=18, fontweight='bold')
                plt.ylabel('Weekly Call Count'); plt.xlabel('Date')
                plt.tight_layout()
                plt.savefig(os.path.join(self.plots_dir, '02_call_volume_by_intent.png'), dpi=self.config['DPI'])
                plt.close()
            else:
                self.logger.warning("Skipping intent plot: No data available after grouping.")
        else:
            self.logger.warning("Skipping intent plot: 'intent' column not found in call data file.")

        # Plot 3: Mail Volume by Type (FIXED with robust fallback)
        self.logger.info("Creating plot: Mail Volume by Type...")
        if 'type' in self.mail_df.columns:
            mail_type_summary = self.mail_df.groupby([pd.Grouper(freq='W'), 'type'])['volume'].sum().unstack(fill_value=0)
            if not mail_type_summary.empty:
                mail_type_summary.plot(kind='line', style='--', marker='o', figsize=self.config['FIGURE_SIZE'], ms=4)
                plt.title('Weekly Mail Volume by Type', fontsize=18, fontweight='bold')
                plt.ylabel('Weekly Mail Volume'); plt.xlabel('Date')
                plt.legend(title='Mail Type')
                plt.tight_layout()
                plt.savefig(os.path.join(self.plots_dir, '03_mail_volume_by_type.png'), dpi=self.config['DPI'])
                plt.close()
            else:
                self.logger.warning("Skipping mail type plot: No data available after grouping.")
        else:
             self.logger.warning("Skipping mail type plot: 'type' column not found in mail data file.")

        # Plot 4: Normalized Trend Comparison (FIXED with rolling average for calls)
        self.logger.info("Creating plot: Normalized Trend Comparison...")
        scaler = MinMaxScaler()
        normalized_df = pd.DataFrame(scaler.fit_transform(self.augmented_data), columns=self.augmented_data.columns, index=self.augmented_data.index)
        normalized_df['call_volume_rolling_7d'] = normalized_df['call_volume'].rolling(window=7, min_periods=1).mean()
        fig, ax = plt.subplots(figsize=self.config['FIGURE_SIZE'])
        ax.plot(normalized_df.index, normalized_df['call_volume_rolling_7d'], label='Call Volume (7-Day Rolling Avg)', color='blue', linewidth=2.5)
        ax.plot(normalized_df.index, normalized_df['mail_volume'], label='Mail Volume', color='skyblue', linewidth=1.5, linestyle='--')
        for col in self.config['FINANCIAL_DATA'].keys():
            if col in normalized_df.columns:
                ax.plot(normalized_df.index, normalized_df[col], label=col, linestyle=':', alpha=0.9)
        ax.set(title='Normalized Trends of All Data (Scaled 0-1)', xlabel='Date', ylabel='Normalized Value')
        ax.legend()
        plt.savefig(os.path.join(self.plots_dir, '04_normalized_trends_readable.png'), dpi=self.config['DPI'])
        plt.close(fig)

        # Plot 5: Data Augmentation Analysis (FIXED to show extended timeline)
        self.logger.info("Creating plot: Data Augmentation Analysis...")
        fig, ax = plt.subplots(figsize=self.config['FIGURE_SIZE'])
        ax.plot(self.augmented_data.index, self.augmented_data['call_volume'], label='Call Volume (including augmentation)', color='navy', zorder=10)
        if not self.augmented_dates.empty:
            highlight_df = self.augmented_data.loc[self.augmented_dates]
            ax.plot(highlight_df.index, highlight_df['call_volume'], label='Augmented Data Section', color='red', linestyle='--', zorder=11, linewidth=2)
        ax.set(title='Call Volume: Final Augmented Timeline', xlabel='Date', ylabel='Call Volume')
        ax.legend()
        plt.savefig(os.path.join(self.plots_dir, '05_augmentation_analysis_corrected.png'), dpi=self.config['DPI'])
        plt.close(fig)

        # Plot 6: Feature Engineering Correlation Matrix
        self.logger.info("Creating plot: Feature Correlation Heatmap...")
        fig, ax = plt.subplots(figsize=(12, 10))
        corr_matrix = self.feature_engineered_data.corr()
        sns.heatmap(corr_matrix[['call_volume']].sort_values(by='call_volume', ascending=False), annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Correlation of Features with Call Volume', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, '06_feature_correlation.png'), dpi=self.config['DPI'])
        plt.close(fig)

        self.logger.info("✓ All EDA visualizations have been saved.")
        return True

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================
if __name__ == '__main__':
    if not os.path.exists(CONFIG['MAIL_FILE_PATH']) or not os.path.exists(CONFIG['CALL_FILE_PATH']):
        print(f"\x1b[31;1mCRITICAL ERROR: Cannot find data files. Please check paths in CONFIG section.\x1b[0m")
        sys.exit(1)
        
    logger = setup_logging(CONFIG['OUTPUT_DIR'])
    analyzer = EDAAnalyzer(CONFIG, logger)
    analyzer.run_eda_pipeline()
