#!/usr/bin/env python3
"""
Call Volume Time Series Analysis Script

This script performs a comprehensive analysis of call volume data in relation to mail campaigns.
It loads mail and call data, processes and cleans it, performs detailed correlation and lag analysis,
and generates a suite of plots and summary reports to uncover insights.

Configure your file paths and column mappings in the CONFIGURATION section below.
All outputs (plots, reports, data) will be saved to the specified output directory.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
import sys
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import traceback
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Statistical libraries
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =============================================================================
# ENHANCED CONFIGURATION SECTION - MODIFY THESE PATHS AND SETTINGS
# =============================================================================

CONFIG = {
    # === FILE PATHS ===
    'MAIL_FILE_PATH': r'C:\path\to\your\mail_data.csv',        # UPDATE THIS PATH
    'CALL_FILE_PATH': r'C:\path\to\your\call_data.csv',        # UPDATE THIS PATH
    'OUTPUT_DIR': r'C:\path\to\output\results',                # UPDATE THIS PATH

    # === MAIL DATA COLUMN MAPPING ===
    'MAIL_COLUMNS': {
        'date': 'date',              # Date column name in your mail file
        'volume': 'volume',          # Volume/quantity column name
        'type': 'mail_type',         # Mail type column name (for legend)
        'source': 'source'           # Source column (optional)
    },

    # === CALL DATA COLUMN MAPPING ===
    # If your call data has individual call records (one row per call):
    'CALL_COLUMNS': {
        'date': 'call_date',         # Date column name in your call file
        'call_id': 'call_id',        # Call ID or unique identifier (optional)
        'phone': 'phone_number',     # Phone number column (optional)
        'duration': 'duration',      # Call duration column (optional)
        'type': 'call_type'          # Call type column (optional)
    },

    # === CALL DATA AGGREGATION SETTINGS ===
    'CALL_AGGREGATION': {
        'method': 'count',           # 'count' = count rows per day, 'sum' = sum a specific column
        'sum_column': None,          # If method='sum', which column to sum (e.g., 'duration')
        'group_by_type': False,      # Whether to also group by call type
    },

    # === DATE PROCESSING SETTINGS ===
    'DATE_PROCESSING': {
        'standardize_mail_dates': True,     # Add 00:00:00 time to date-only entries
        'standardize_call_dates': True,     # Add 00:00:00 time to date-only entries
        'date_format': None,                # e.g., '%Y-%m-%d' or None for auto-detection
        'time_format': '%H:%M:%S',          # Time format for standardization
    },

    # === ANALYSIS SETTINGS ===
    'REMOVE_OUTLIERS': True,         # Remove outliers from analysis
    'MAX_LAG_DAYS': 21,             # Maximum lag days to test
    'MIN_OVERLAP_RECORDS': 10,       # Minimum overlapping records needed
    'MAX_RESPONSE_RATE': 50,         # Maximum realistic response rate (%)

    # === PLOT SETTINGS ===
    'PLOT_STYLE': 'seaborn-v0_8',   # Matplotlib style
    'FIGURE_SIZE': (15, 10),        # Default figure size
    'DPI': 300,                     # Plot resolution
    'FONT_SIZE': 12,                # Default font size
}

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(output_dir: str) -> logging.Logger:
    """Setup comprehensive logging to both file and console"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger('CallVolumeAnalysis')
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')

    # File handler
    log_file = os.path.join(output_dir, f'analysis_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger

# =============================================================================
# ENHANCED DATA PROCESSING FUNCTIONS
# =============================================================================

def standardize_dates(data, date_column, logger):
    """Standardize dates by adding 00:00:00 time to date-only entries"""
    logger.info(f"Standardizing dates in column: {date_column}")
    
    date_strings = data[date_column].astype(str)
    
    # Standardize dates by converting to datetime and back to string with a consistent format
    data[date_column] = pd.to_datetime(date_strings, errors='coerce').dt.normalize()
    
    logger.info("✓ Date standardization completed")
    return data

def aggregate_call_data(call_data, config, logger):
    """Aggregate call data by date to get volume per day"""
    logger.info("Aggregating call data by date...")
    
    agg_config = config['CALL_AGGREGATION']
    method = agg_config['method']
    sum_column = agg_config['sum_column']
    group_by_type = agg_config['group_by_type']

    logger.info(f"  Aggregation method: {method}")
    logger.info(f"  Group by type: {group_by_type}")

    group_cols = ['date']
    if group_by_type and 'type' in call_data.columns:
        group_cols.append('type')
        logger.info("  Including call type in grouping")

    if method == 'count':
        if group_by_type and 'type' in call_data.columns:
            agg_data = call_data.groupby(group_cols).size().reset_index(name='volume')
            total_agg = call_data.groupby('date').size().reset_index(name='volume')
        else:
            agg_data = call_data.groupby('date').size().reset_index(name='volume')
            total_agg = agg_data.copy()
            
    elif method == 'sum' and sum_column and sum_column in call_data.columns:
        if group_by_type and 'type' in call_data.columns:
            agg_data = call_data.groupby(group_cols)[sum_column].sum().reset_index()
            agg_data.columns = group_cols + ['volume']
            total_agg = call_data.groupby('date')[sum_column].sum().reset_index(name='volume')
        else:
            agg_data = call_data.groupby('date')[sum_column].sum().reset_index(name='volume')
            total_agg = agg_data.copy()
        logger.info(f"  Summing column: {sum_column}")
    else:
        logger.warning(f"Invalid aggregation method or column, defaulting to count")
        agg_data = call_data.groupby('date').size().reset_index(name='volume')
        total_agg = agg_data.copy()

    logger.info(f"✓ Call data aggregated: {len(total_agg)} unique dates")
    logger.info(f"  Total call volume: {total_agg['volume'].sum():,}")
    logger.info(f"  Average daily volume: {total_agg['volume'].mean():.1f}")
    
    return total_agg, agg_data if group_by_type and 'type' in call_data.columns else None

# =============================================================================
# MAIN ANALYSIS CLASS
# =============================================================================

class CallVolumeAnalyzer:
    def __init__(self, config: Dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.mail_data = None
        self.call_data = None
        self.mail_data_clean = None
        self.call_data_clean = None
        self.combined_data = None
        self.analysis_results = {}

        # Create output directories
        self.plots_dir = os.path.join(config['OUTPUT_DIR'], 'plots')
        self.data_dir = os.path.join(config['OUTPUT_DIR'], 'data')
        self.reports_dir = os.path.join(config['OUTPUT_DIR'], 'reports')
        
        for dir_path in [self.plots_dir, self.data_dir, self.reports_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Set plot style
        try:
            plt.style.use(config['PLOT_STYLE'])
        except:
            plt.style.use('default')
            self.logger.warning(f"Could not set plot style {config['PLOT_STYLE']}, using default")
        
        plt.rcParams.update({'font.size': config['FONT_SIZE']})

    def run_complete_analysis(self) -> bool:
        """Run the complete analysis pipeline"""
        self.logger.info("=" * 80)
        self.logger.info("CALL VOLUME TIME SERIES ANALYSIS - STARTING")
        self.logger.info("=" * 80)
        self.logger.info(f"Output directory: {self.config['OUTPUT_DIR']}")
        self.logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            if not self._load_data(): return False
            if not self._process_data(): return False
            if not self._clean_data(): return False
            if not self._analyze_correlations(): return False
            if not self._create_all_plots(): return False
            if not self._generate_reports(): return False
            if not self._save_processed_data(): return False
            
            self.logger.info("=" * 80)
            self.logger.info("ANALYSIS COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 80)
            self.logger.info(f"Check output directory: {self.config['OUTPUT_DIR']}")
            return True
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    def _load_data(self) -> bool:
        """Load mail and call data files"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STEP 1: LOADING DATA")
        self.logger.info("=" * 60)
        
        try:
            # Load mail data
            self.logger.info(f"Loading mail data from: {self.config['MAIL_FILE_PATH']}")
            if not os.path.exists(self.config['MAIL_FILE_PATH']):
                raise FileNotFoundError(f"Mail file not found: {self.config['MAIL_FILE_PATH']}")
            self.mail_data = pd.read_csv(self.config['MAIL_FILE_PATH']) if self.config['MAIL_FILE_PATH'].lower().endswith('.csv') else pd.read_excel(self.config['MAIL_FILE_PATH'])
            self.logger.info(f"✓ Mail data loaded: {len(self.mail_data):,} rows")

            # Load call data
            self.logger.info(f"Loading call data from: {self.config['CALL_FILE_PATH']}")
            if not os.path.exists(self.config['CALL_FILE_PATH']):
                raise FileNotFoundError(f"Call file not found: {self.config['CALL_FILE_PATH']}")
            self.call_data = pd.read_csv(self.config['CALL_FILE_PATH']) if self.config['CALL_FILE_PATH'].lower().endswith('.csv') else pd.read_excel(self.config['CALL_FILE_PATH'])
            self.logger.info(f"✓ Call data loaded: {len(self.call_data):,} rows")
            return True
        except Exception as e:
            self.logger.error(f"✗ Error loading data: {str(e)}")
            return False

    def _process_data(self) -> bool:
        """Enhanced process and validate both datasets"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STEP 2: ENHANCED DATA PROCESSING")
        self.logger.info("=" * 60)

        try:
            # Process mail data
            self.logger.info("Processing mail data...")
            mail_mapping = self._map_columns(self.mail_data, self.config['MAIL_COLUMNS'], 'mail')
            if mail_mapping: self.mail_data = self.mail_data.rename(columns=mail_mapping)
            if 'date' not in self.mail_data.columns or 'volume' not in self.mail_data.columns:
                raise ValueError("Date and volume columns are required in mail data")
            if self.config['DATE_PROCESSING']['standardize_mail_dates']:
                self.mail_data = standardize_dates(self.mail_data, 'date', self.logger)
            self.mail_data = self._process_dates_and_volumes(self.mail_data, 'mail')
            self.mail_data_agg = self._aggregate_mail_data()

            # Process call data
            self.logger.info("Processing call data...")
            call_mapping = self._map_call_columns()
            if call_mapping: self.call_data = self.call_data.rename(columns=call_mapping)
            if 'date' not in self.call_data.columns: raise ValueError("Date column not found in call data")
            if self.config['DATE_PROCESSING']['standardize_call_dates']:
                self.call_data = standardize_dates(self.call_data, 'date', self.logger)
            self.call_data = self._process_call_dates_and_data()
            self.call_data_agg, self.call_data_by_type = aggregate_call_data(self.call_data, self.config, self.logger)

            self.logger.info(f"✓ Mail data processed: {len(self.mail_data_agg):,} unique dates")
            self.logger.info(f"✓ Call data processed: {len(self.call_data_agg):,} unique dates")
            return True
        except Exception as e:
            self.logger.error(f"✗ Error processing data: {str(e)}")
            return False

    def _map_columns(self, data: pd.DataFrame, column_config: Dict, data_type: str) -> Dict:
        """Map column names based on configuration"""
        mapping = {}
        for standard_name, config_name in column_config.items():
            if config_name in data.columns:
                if config_name != standard_name: mapping[config_name] = standard_name
            else:
                similar_cols = [col for col in data.columns if config_name.lower() in col.lower() or col.lower() in config_name.lower()]
                if similar_cols:
                    mapping[similar_cols[0]] = standard_name
                    self.logger.warning(f"Using '{similar_cols[0]}' for '{standard_name}' in {data_type} data")
                elif standard_name in ['date', 'volume']:
                    self.logger.warning(f"Required column '{config_name}' not found in {data_type} data")
        if mapping: self.logger.info(f"Column mapping for {data_type} data: {mapping}")
        return mapping

    def _map_call_columns(self) -> Dict:
        """Enhanced call column mapping"""
        mapping = {}
        call_config = self.config['CALL_COLUMNS']
        date_col = call_config.get('date', 'date')
        if date_col in self.call_data.columns:
            if date_col != 'date': mapping[date_col] = 'date'
        else:
            date_candidates = [col for col in self.call_data.columns if any(word in col.lower() for word in ['date', 'time', 'created', 'timestamp'])]
            if date_candidates:
                mapping[date_candidates[0]] = 'date'
                self.logger.warning(f"Using '{date_candidates[0]}' as date column for call data")
            else: self.logger.error("No date column found in call data")
        
        for standard_name, config_name in call_config.items():
            if standard_name != 'date' and config_name in self.call_data.columns and config_name != standard_name:
                mapping[config_name] = standard_name
        
        if mapping: self.logger.info(f"Call data column mapping: {mapping}")
        return mapping

    def _process_dates_and_volumes(self, data: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Process date and volume columns"""
        date_format = self.config['DATE_PROCESSING']['date_format']
        data['date'] = pd.to_datetime(data['date'], format=date_format, errors='coerce')
        data = data.dropna(subset=['date'])
        
        data['volume'] = pd.to_numeric(data['volume'], errors='coerce')
        data = data.dropna(subset=['volume'])
        data = data[data['volume'] >= 0]
        return data
    
    def _process_call_dates_and_data(self) -> pd.DataFrame:
        """Enhanced call data processing"""
        data = self.call_data.copy()
        date_format = self.config['DATE_PROCESSING']['date_format']
        data['date'] = pd.to_datetime(data['date'], format=date_format, errors='coerce')
        data = data.dropna(subset=['date'])
        data['date'] = data['date'].dt.date
        data['date'] = pd.to_datetime(data['date'])
        self.logger.info(f"Call data date processing completed: {len(data):,} records")
        return data

    def _aggregate_mail_data(self) -> pd.DataFrame:
        """Aggregate mail data by date and preserve type information"""
        self.mail_data_with_types = self.mail_data.copy()
        if 'type' in self.mail_data.columns:
            agg_data = self.mail_data.groupby('date').agg({'volume': 'sum', 'type': lambda x: '|'.join(x.astype(str).unique())}).reset_index()
            agg_data.columns = ['date', 'volume', 'types_combined']
            self.logger.info(f"Mail types found: {self.mail_data['type'].nunique()}")
        else:
            agg_data = self.mail_data.groupby('date')['volume'].sum().reset_index()
            self.logger.info("No mail type column found")
        return agg_data

    def _clean_data(self) -> bool:
        """Clean data by detecting and optionally removing outliers"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STEP 3: CLEANING DATA")
        self.logger.info("=" * 60)
        try:
            for data_type, data_agg in [('mail', self.mail_data_agg), ('call', self.call_data_agg)]:
                self.logger.info(f"Analyzing {data_type} data outliers...")
                outliers = self._detect_outliers(data_agg, 'volume')
                self.logger.info(f"{data_type.capitalize()} outliers detected: {outliers.sum():,} ({outliers.sum()/len(data_agg)*100:.1f}%)")
                if self.config['REMOVE_OUTLIERS'] and outliers.sum() > 0:
                    clean_data = data_agg[~outliers].reset_index(drop=True)
                    self.logger.info(f"✓ Removed {outliers.sum():,} {data_type} outliers")
                else:
                    clean_data = data_agg.copy()
                    self.logger.info(f"✓ No {data_type} outliers removed")
                
                if data_type == 'mail': self.mail_data_clean = clean_data
                else: self.call_data_clean = clean_data
            
            self.logger.info(f"Final clean datasets: Mail ({len(self.mail_data_clean):,} records), Call ({len(self.call_data_clean):,} records)")
            return True
        except Exception as e:
            self.logger.error(f"✗ Error cleaning data: {str(e)}")
            return False

    def _detect_outliers(self, data: pd.DataFrame, column: str) -> pd.Series:
        """Detect outliers using multiple methods"""
        outliers = pd.DataFrame(index=data.index)
        # IQR method
        Q1, Q3 = data[column].quantile(0.25), data[column].quantile(0.75)
        outliers['iqr'] = (data[column] < (Q1 - 1.5 * (Q3 - Q1))) | (data[column] > (Q3 + 1.5 * (Q3 - Q1)))
        # Z-score method
        outliers['zscore'] = np.abs(stats.zscore(data[column])) > 3
        # Isolation Forest method
        try:
            iso_forest = IsolationForest(contamination=0.05, random_state=42)
            outliers['isolation'] = iso_forest.fit_predict(data[[column]].values) == -1
        except: outliers['isolation'] = False
        return outliers.sum(axis=1) >= 2 # Consensus if detected by >= 2 methods

    def _analyze_correlations(self) -> bool:
        """Analyze correlations and find optimal lag"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STEP 4: CORRELATION ANALYSIS")
        self.logger.info("=" * 60)
        try:
            combined = pd.merge(self.call_data_clean, self.mail_data_clean, on='date', how='outer', suffixes=('_call', '_mail')).sort_values('date').reset_index(drop=True)
            combined.fillna(0, inplace=True)
            self.combined_data = combined
            
            overlap_data = combined[(combined['volume_call'] > 0) & (combined['volume_mail'] > 0)]
            if len(overlap_data) < self.config['MIN_OVERLAP_RECORDS']:
                self.logger.warning(f"Insufficient overlapping data for correlation analysis (need >= {self.config['MIN_OVERLAP_RECORDS']})")
                return True # Continue without correlation analysis

            corr_pearson, p_pearson = pearsonr(overlap_data['volume_call'], overlap_data['volume_mail'])
            corr_spearman, p_spearman = spearmanr(overlap_data['volume_call'], overlap_data['volume_mail'])

            lag_results = []
            for lag in range(self.config['MAX_LAG_DAYS'] + 1):
                mail_lagged = combined['volume_mail'].shift(lag)
                mask = (combined['volume_call'] > 0) & (mail_lagged > 0)
                if mask.sum() >= self.config['MIN_OVERLAP_RECORDS']:
                    corr, p_val = pearsonr(combined.loc[mask, 'volume_call'], mail_lagged[mask])
                    lag_results.append({'lag': lag, 'correlation': corr, 'p_value': p_val, 'n_obs': mask.sum()})

            self.analysis_results['correlations'] = {'pearson': corr_pearson, 'spearman': corr_spearman, 'pearson_pvalue': p_pearson, 'spearman_pvalue': p_spearman}
            if lag_results:
                lag_df = pd.DataFrame(lag_results)
                best_lag = lag_df.loc[lag_df['correlation'].idxmax()]
                self.analysis_results['lag_analysis'] = lag_df
                self.analysis_results['best_lag'] = best_lag
                self.logger.info(f"✓ Best lag found: {best_lag['lag']} days with correlation {best_lag['correlation']:.4f}")

            overlap_data['response_rate'] = overlap_data['volume_call'] / overlap_data['volume_mail'] * 100
            valid_rates = overlap_data[overlap_data['response_rate'] <= self.config['MAX_RESPONSE_RATE']]['response_rate']
            if len(valid_rates) > 0: self.analysis_results['response_rates'] = valid_rates
            
            self.analysis_results['overlap_stats'] = {'total_records': len(combined), 'overlap_records': len(overlap_data), 'overlap_percentage': len(overlap_data)/len(combined)*100}
            return True
        except Exception as e:
            self.logger.error(f"✗ Error in correlation analysis: {str(e)}")
            return False

    def _create_all_plots(self) -> bool:
        """Create all plots and save to files"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STEP 5: CREATING COMPREHENSIVE EDA PLOTS")
        self.logger.info("=" * 60)
        try:
            plt.rcParams.update({'figure.dpi': self.config['DPI'], 'savefig.dpi': self.config['DPI'], 'font.size': self.config['FONT_SIZE']})
            plots_created = 0
            plot_functions = [
                ("Mail volume time series", self._create_mail_timeseries_plot),
                ("Call volume time series", self._create_call_timeseries_plot),
                ("Combined overlay plot", self._create_combined_overlay_plot),
                ("Distribution plots", self._create_distribution_plots),
                ("Day of week analysis", self._create_day_of_week_plots),
                ("Monthly trends", self._create_monthly_trends_plot),
                ("Seasonal analysis", self._create_seasonal_analysis_plot),
            ]
            if self.combined_data is not None and self.analysis_results.get('correlations'): plot_functions.append(("Correlation scatter plot", self._create_correlation_scatter_plot))
            if self.analysis_results.get('lag_analysis') is not None: plot_functions.append(("Lag analysis plot", self._create_lag_analysis_plot))
            if self.analysis_results.get('response_rates') is not None: plot_functions.append(("Response rate analysis", self._create_response_rate_plot))
            if hasattr(self, 'mail_data_with_types') and 'type' in self.mail_data_with_types.columns: plot_functions.append(("Mail type analysis", self._create_mail_type_analysis_plot))
            
            for name, func in plot_functions:
                self.logger.info(f"Creating {name}...")
                if func(): plots_created += 1; self.logger.info(f"✓ {name} created successfully")
                else: self.logger.warning(f"✗ Failed to create {name}")
            
            self.logger.info(f"✓ Created {plots_created}/{len(plot_functions)} plots in {self.plots_dir}")
            return True
        except Exception as e:
            self.logger.error(f"✗ Error creating plots: {str(e)}")
            return False

    def _create_plot_template(self, title, xlabel, ylabel, figsize=None):
        """Helper to create a standard plot figure and axis."""
        fig, ax = plt.subplots(figsize=figsize if figsize else self.config['FIGURE_SIZE'])
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        return fig, ax

    def _save_plot(self, fig, filename):
        """Helper to save a plot."""
        plot_path = os.path.join(self.plots_dir, filename)
        fig.tight_layout()
        fig.savefig(plot_path, bbox_inches='tight', dpi=self.config['DPI'])
        plt.close(fig)

    def _create_mail_timeseries_plot(self) -> bool:
        try:
            fig, ax = self._create_plot_template('Mail Volume Over Time', 'Date', 'Mail Volume')
            if hasattr(self, 'mail_data_with_types') and 'type' in self.mail_data_with_types.columns:
                unique_types = self.mail_data_with_types['type'].unique()
                colors = plt.cm.Set3(np.linspace(0, 1, len(unique_types)))
                for i, mail_type in enumerate(unique_types):
                    type_data = self.mail_data_with_types[self.mail_data_with_types['type'] == mail_type].groupby('date')['volume'].sum()
                    ax.plot(type_data.index, type_data.values, label=f'{mail_type}', color=colors[i], alpha=0.8)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                ax.plot(self.mail_data_clean['date'], self.mail_data_clean['volume'], color='red', alpha=0.8)
            plt.xticks(rotation=45)
            self._save_plot(fig, '01_mail_volume_timeseries.png')
            return True
        except Exception as e: self.logger.error(f"Error creating mail timeseries plot: {e}"); return False

    def _create_call_timeseries_plot(self) -> bool:
        try:
            fig, ax = self._create_plot_template('Call Volume Over Time', 'Date', 'Call Volume')
            ax.plot(self.call_data_clean['date'], self.call_data_clean['volume'], color='blue', alpha=0.8)
            plt.xticks(rotation=45)
            self._save_plot(fig, '02_call_volume_timeseries.png')
            return True
        except Exception as e: self.logger.error(f"Error creating call timeseries plot: {e}"); return False

    def _create_combined_overlay_plot(self) -> bool:
        try:
            fig, ax = self._create_plot_template('Normalized Mail vs Call Volume Overlay', 'Date', 'Normalized Volume (0-1)')
            mail_norm = self.mail_data_clean['volume'] / self.mail_data_clean['volume'].max()
            call_norm = self.call_data_clean['volume'] / self.call_data_clean['volume'].max()
            ax.plot(self.mail_data_clean['date'], mail_norm, label='Mail (normalized)', color='red', alpha=0.7)
            ax.plot(self.call_data_clean['date'], call_norm, label='Calls (normalized)', color='blue', alpha=0.7)
            ax.legend()
            plt.xticks(rotation=45)
            self._save_plot(fig, '03_combined_overlay.png')
            return True
        except Exception as e: self.logger.error(f"Error creating combined overlay plot: {e}"); return False
        
    def _create_distribution_plots(self) -> bool:
        try:
            fig, axes = plt.subplots(2, 2, figsize=(20, 12))
            sns.histplot(self.mail_data_clean['volume'], bins=50, ax=axes[0,0], color='red', kde=True).set_title('Mail Volume Distribution')
            sns.histplot(self.call_data_clean['volume'], bins=50, ax=axes[0,1], color='blue', kde=True).set_title('Call Volume Distribution')
            sns.boxplot(y=self.mail_data_clean['volume'], ax=axes[1,0], color='red').set_title('Mail Volume Box Plot')
            sns.boxplot(y=self.call_data_clean['volume'], ax=axes[1,1], color='blue').set_title('Call Volume Box Plot')
            self._save_plot(fig, '04_distributions.png')
            return True
        except Exception as e: self.logger.error(f"Error creating distribution plots: {e}"); return False

    def _create_day_of_week_plots(self) -> bool:
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            mail_dow = self.mail_data_clean.copy(); mail_dow['day_of_week'] = mail_dow['date'].dt.day_name()
            call_dow = self.call_data_clean.copy(); call_dow['day_of_week'] = call_dow['date'].dt.day_name()
            sns.boxplot(data=mail_dow, x='day_of_week', y='volume', order=day_order, ax=ax1, color='red').set_title('Mail Volume by Day of Week')
            sns.boxplot(data=call_dow, x='day_of_week', y='volume', order=day_order, ax=ax2, color='blue').set_title('Call Volume by Day of Week')
            ax1.tick_params(axis='x', rotation=45); ax2.tick_params(axis='x', rotation=45)
            self._save_plot(fig, '05_day_of_week_analysis.png')
            return True
        except Exception as e: self.logger.error(f"Error creating day of week plots: {e}"); return False

    def _create_correlation_scatter_plot(self) -> bool:
        try:
            scatter_data = self.combined_data[(self.combined_data['volume_call'] > 0) & (self.combined_data['volume_mail'] > 0)]
            if len(scatter_data) < 2: return False
            fig, ax = self._create_plot_template('Call Volume vs Mail Volume Correlation', 'Mail Volume', 'Call Volume')
            sns.regplot(data=scatter_data, x='volume_mail', y='volume_call', ax=ax, color='purple', line_kws={"color": "red"})
            corr = self.analysis_results['correlations']['pearson']
            ax.text(0.05, 0.95, f'Pearson r = {corr:.4f}', transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
            self._save_plot(fig, '06_correlation_scatter.png')
            return True
        except Exception as e: self.logger.error(f"Error creating correlation scatter plot: {e}"); return False

    def _create_lag_analysis_plot(self) -> bool:
        try:
            lag_df = self.analysis_results['lag_analysis']
            best_lag = self.analysis_results['best_lag']
            fig, ax = self._create_plot_template('Correlation vs Lag Days', 'Lag Days', 'Pearson Correlation')
            ax.plot(lag_df['lag'], lag_df['correlation'], marker='o')
            ax.axvline(x=best_lag['lag'], color='red', linestyle='--', label=f"Best lag: {best_lag['lag']} days (r={best_lag['correlation']:.2f})")
            ax.legend()
            self._save_plot(fig, '07_lag_analysis.png')
            return True
        except Exception as e: self.logger.error(f"Error creating lag analysis plot: {e}"); return False
        
    def _create_response_rate_plot(self) -> bool:
        try:
            rates = self.analysis_results['response_rates']
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            sns.histplot(rates, bins=30, ax=ax1, color='green', kde=True).set_title('Response Rate Distribution (%)')
            ax1.axvline(rates.mean(), color='red', ls='--', label=f'Mean: {rates.mean():.2f}%')
            ax1.legend()
            sns.boxplot(y=rates, ax=ax2, color='green').set_title('Response Rate Box Plot (%)')
            self._save_plot(fig, '08_response_rate_analysis.png')
            return True
        except Exception as e: self.logger.error(f"Error creating response rate plot: {e}"); return False

    def _create_monthly_trends_plot(self) -> bool:
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            mail_monthly = self.mail_data_clean.set_index('date')['volume'].resample('M').sum()
            call_monthly = self.call_data_clean.set_index('date')['volume'].resample('M').sum()
            ax1.plot(mail_monthly.index, mail_monthly.values, marker='o', color='red'); ax1.set_title('Monthly Mail Volume')
            ax2.plot(call_monthly.index, call_monthly.values, marker='o', color='blue'); ax2.set_title('Monthly Call Volume')
            self._save_plot(fig, '09_monthly_trends.png')
            return True
        except Exception as e: self.logger.error(f"Error creating monthly trends plot: {e}"); return False

    def _create_mail_type_analysis_plot(self) -> bool:
        try:
            type_stats = self.mail_data_with_types.groupby('type')['volume'].agg(['sum', 'count', 'mean']).sort_values('sum', ascending=False)
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            sns.barplot(x=type_stats.index, y=type_stats['sum'], ax=axes[0,0], color='orange').set_title('Total Mail Volume by Type')
            sns.barplot(x=type_stats.index, y=type_stats['mean'], ax=axes[0,1], color='purple').set_title('Average Mail Volume by Type')
            sns.barplot(x=type_stats.index, y=type_stats['count'], ax=axes[1,0], color='brown').set_title('Number of Campaigns by Type')
            axes[1,1].pie(type_stats['sum'], labels=type_stats.index, autopct='%1.1f%%', startangle=90)
            axes[1,1].set_title('Mail Volume Distribution by Type')
            for ax in axes.flat[:-1]: ax.tick_params(axis='x', rotation=45, ha='right')
            self._save_plot(fig, '10_mail_type_analysis.png')
            return True
        except Exception as e: self.logger.error(f"Error creating mail type analysis plot: {e}"); return False

    def _create_seasonal_analysis_plot(self) -> bool:
        try:
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            mail_seasonal = self.mail_data_clean.copy(); mail_seasonal['quarter'] = mail_seasonal['date'].dt.quarter; mail_seasonal['month'] = mail_seasonal['date'].dt.month
            call_seasonal = self.call_data_clean.copy(); call_seasonal['quarter'] = call_seasonal['date'].dt.quarter; call_seasonal['month'] = call_seasonal['date'].dt.month
            sns.barplot(data=mail_seasonal, x='quarter', y='volume', ax=axes[0,0], color='red').set_title('Average Mail Volume by Quarter')
            sns.barplot(data=call_seasonal, x='quarter', y='volume', ax=axes[0,1], color='blue').set_title('Average Call Volume by Quarter')
            sns.lineplot(data=mail_seasonal, x='month', y='volume', marker='o', ax=axes[1,0], color='red').set_title('Average Mail Volume by Month')
            sns.lineplot(data=call_seasonal, x='month', y='volume', marker='o', ax=axes[1,1], color='blue').set_title('Average Call Volume by Month')
            axes[1,0].set_xticks(range(1,13)); axes[1,1].set_xticks(range(1,13))
            self._save_plot(fig, '11_seasonal_analysis.png')
            return True
        except Exception as e: self.logger.error(f"Error creating seasonal analysis plot: {e}"); return False

    def _generate_reports(self) -> bool:
        """Generate comprehensive analysis reports"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STEP 6: GENERATING REPORTS")
        self.logger.info("=" * 60)
        try:
            for report_name, report_func in [("analysis_summary.txt", self._create_summary_report), ("detailed_metrics.txt", self._create_metrics_report)]:
                report_content = report_func()
                if report_content:
                    report_path = os.path.join(self.reports_dir, report_name)
                    with open(report_path, 'w', encoding='utf-8') as f: f.write(report_content)
                    self.logger.info(f"✓ {report_name} saved")
            return True
        except Exception as e:
            self.logger.error(f"✗ Error generating reports: {str(e)}")
            return False

    def _create_summary_report(self) -> str:
        """Create a comprehensive summary report"""
        report = ["=" * 80, "CALL VOLUME TIME SERIES ANALYSIS - SUMMARY REPORT", "=" * 80, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"]
        report.extend(["DATA OVERVIEW", "-" * 40, f"Mail Data: {len(self.mail_data_clean):,} records | Total Volume: {self.mail_data_clean['volume'].sum():,} | Avg Daily: {self.mail_data_clean['volume'].mean():.1f}"])
        report.extend([f"Call Data: {len(self.call_data_clean):,} records | Total Volume: {self.call_data_clean['volume'].sum():,} | Avg Daily: {self.call_data_clean['volume'].mean():.1f}\n"])

        if self.analysis_results:
            report.extend(["ANALYSIS RESULTS", "-" * 40])
            if 'best_lag' in self.analysis_results: report.append(f"Optimal Lag: {self.analysis_results['best_lag']['lag']} days (Correlation: {self.analysis_results['best_lag']['correlation']:.4f})")
            if 'response_rates' in self.analysis_results: report.append(f"Response Rate: Mean {self.analysis_results['response_rates'].mean():.2f}%, Median {self.analysis_results['response_rates'].median():.2f}%")
            if 'correlations' in self.analysis_results: report.append(f"Overall Correlation (Pearson): {self.analysis_results['correlations']['pearson']:.4f}\n")

        if hasattr(self, 'mail_data_with_types') and 'type' in self.mail_data_with_types.columns:
            report.extend(["MAIL TYPE ANALYSIS", "-" * 40])
            report.append("Top mail types by volume:\n" + self.mail_data_with_types.groupby('type')['volume'].sum().nlargest(5).to_string() + "\n")
        
        report.extend(["RECOMMENDATIONS", "-" * 40])
        recs = []
        if self.analysis_results.get('best_lag', {}).get('lag', 0) > 0: recs.append(f"A lag of {self.analysis_results['best_lag']['lag']} days shows the strongest correlation; consider this in marketing attribution.")
        corr = self.analysis_results.get('correlations', {}).get('pearson', 0)
        if corr > 0.5: recs.append("Strong positive correlation suggests mail campaigns effectively drive call volume.")
        elif corr < 0.2: recs.append("Weak correlation may indicate other factors drive calls, or data needs further segmentation.")
        if not recs: recs.append("Analysis complete. Review plots for seasonal and weekly trends.")
        report.extend([f"  - {rec}" for rec in recs])

        report.extend(["\n", "END OF REPORT", "="*80])
        return "\n".join(report)

    def _create_metrics_report(self) -> str:
        """Create detailed metrics report"""
        if not self.analysis_results: return ""
        report = ["=" * 80, "DETAILED METRICS REPORT", "=" * 80, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"]
        if 'lag_analysis' in self.analysis_results:
            report.extend(["LAG ANALYSIS DETAILS", "-" * 40, self.analysis_results['lag_analysis'].to_string(), "\n"])
        return "\n".join(report)
        
    def _save_processed_data(self) -> bool:
        """Save processed datasets to files"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STEP 7: SAVING PROCESSED DATA")
        self.logger.info("=" * 60)
        try:
            self.mail_data_clean.to_csv(os.path.join(self.data_dir, 'mail_data_clean.csv'), index=False)
            self.call_data_clean.to_csv(os.path.join(self.data_dir, 'call_data_clean.csv'), index=False)
            self.combined_data.to_csv(os.path.join(self.data_dir, 'combined_data.csv'), index=False)
            self.logger.info("✓ Clean and combined data saved.")

            if self.analysis_results:
                serializable_results = {}
                for key, value in self.analysis_results.items():
                    if isinstance(value, pd.DataFrame): serializable_results[key] = value.to_dict('records')
                    elif isinstance(value, pd.Series): serializable_results[key] = value.tolist()
                    elif isinstance(value, np.integer): serializable_results[key] = int(value)
                    elif isinstance(value, np.floating): serializable_results[key] = float(value)
                    elif isinstance(value, dict):
                         serializable_results[key] = {k: (int(v) if isinstance(v, np.integer) else (float(v) if isinstance(v, np.floating) else v)) for k, v in value.items()}
                    else: serializable_results[key] = value
                
                with open(os.path.join(self.data_dir, 'analysis_results.json'), 'w') as f:
                    json.dump(serializable_results, f, indent=4)
                self.logger.info("✓ Analysis results saved as JSON.")
            return True
        except Exception as e:
            self.logger.error(f"✗ Error saving processed data: {str(e)}")
            return False

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def main():
    """Main execution function"""
    # Validate configuration
    for path_key in ['MAIL_FILE_PATH', 'CALL_FILE_PATH']:
        if not os.path.exists(CONFIG[path_key]):
            print(f"ERROR: File not found: {CONFIG[path_key]}")
            print(f"Please update {path_key} in the configuration section.")
            return False

    try:
        logger = setup_logging(CONFIG['OUTPUT_DIR'])
    except Exception as e:
        print(f"ERROR: Failed to setup logging: {str(e)}")
        return False

    analyzer = CallVolumeAnalyzer(CONFIG, logger)
    return analyzer.run_complete_analysis()

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

if __name__ == "__main__":
    """
    TO RUN THIS SCRIPT:
    1. Update the file paths in the CONFIG section at the top.
    2. Update column mappings if your columns have different names.
    3. Configure call aggregation settings based on your data format.
    4. Run the script from your terminal: python your_script_name.py
    5. Check the specified output directory for results.
    """
    # ASCII art header
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    CALL VOLUME TIME SERIES ANALYSIS                          ║
║                                                                              ║
║  This script analyzes the relationship between mail campaigns and call      ║
║  volumes, providing comprehensive EDA plots, correlation analysis, and      ║
║  actionable insights for predictive modeling.                               ║
║                                                                              ║
║  Configure your file paths in the CONFIG section and run!                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Run main analysis
    success = main()

    if success:
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"Find your reports, plots, and data in: {CONFIG['OUTPUT_DIR']}")
        print("=" * 80)
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("ANALYSIS FAILED!")
        print("Please check the log file in the output directory for details.")
        print("=" * 80)
        sys.exit(1)
