# #!/usr/bin/env python3
“””
Call Volume Time Series Analysis Script

This script performs comprehensive analysis of call volume data in relation to mail campaigns.
Configure your file paths and column mappings in the CONFIGURATION section below.

All outputs (plots, reports, data) will be saved to the specified output directory.
“””

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
warnings.filterwarnings(‘ignore’)

# Statistical libraries

from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =============================================================================

# CONFIGURATION SECTION - MODIFY THESE PATHS AND SETTINGS

# =============================================================================

CONFIG = {
# === FILE PATHS ===
‘MAIL_FILE_PATH’: r’C:\path\to\your\mail_data.csv’,        # UPDATE THIS PATH
‘CALL_FILE_PATH’: r’C:\path\to\your\call_data.csv’,        # UPDATE THIS PATH
‘OUTPUT_DIR’: r’C:\path\to\output\results’,                # UPDATE THIS PATH

```
# === MAIL DATA COLUMN MAPPING ===
'MAIL_COLUMNS': {
    'date': 'date',              # Date column name in your mail file
    'volume': 'volume',          # Volume/quantity column name
    'type': 'mail_type',         # Mail type column name (for legend)
    'source': 'source'           # Source column (optional)
},

# === CALL DATA COLUMN MAPPING ===
'CALL_COLUMNS': {
    'date': 'date',              # Date column name in your call file
    'volume': 'call_volume'      # Call volume column name
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

# === DATA SETTINGS ===
'DATE_FORMAT': None,            # Date format (None for auto-detection)
'DECIMAL_PLACES': 2,            # Decimal places for metrics
```

}

# =============================================================================

# LOGGING SETUP

# =============================================================================

def setup_logging(output_dir: str) -> logging.Logger:
“”“Setup comprehensive logging to both file and console”””

```
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
```

# =============================================================================

# MAIN ANALYSIS CLASS

# =============================================================================

class CallVolumeAnalyzer:
def **init**(self, config: Dict, logger: logging.Logger):
self.config = config
self.logger = logger
self.mail_data = None
self.call_data = None
self.mail_data_clean = None
self.call_data_clean = None
self.combined_data = None
self.analysis_results = {}

```
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
    
    # Set font size
    plt.rcParams.update({'font.size': config['FONT_SIZE']})

def run_complete_analysis(self) -> bool:
    """Run the complete analysis pipeline"""
    
    self.logger.info("=" * 80)
    self.logger.info("CALL VOLUME TIME SERIES ANALYSIS - STARTING")
    self.logger.info("=" * 80)
    self.logger.info(f"Output directory: {self.config['OUTPUT_DIR']}")
    self.logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Step 1: Load data
        if not self._load_data():
            return False
        
        # Step 2: Process data
        if not self._process_data():
            return False
        
        # Step 3: Clean data
        if not self._clean_data():
            return False
        
        # Step 4: Analyze correlations
        if not self._analyze_correlations():
            return False
        
        # Step 5: Create plots
        if not self._create_all_plots():
            return False
        
        # Step 6: Generate reports
        if not self._generate_reports():
            return False
        
        # Step 7: Save processed data
        if not self._save_processed_data():
            return False
        
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
        
        if self.config['MAIL_FILE_PATH'].lower().endswith('.csv'):
            self.mail_data = pd.read_csv(self.config['MAIL_FILE_PATH'])
        elif self.config['MAIL_FILE_PATH'].lower().endswith(('.xlsx', '.xls')):
            self.mail_data = pd.read_excel(self.config['MAIL_FILE_PATH'])
        else:
            raise ValueError("Mail file must be CSV or Excel format")
        
        self.logger.info(f"✓ Mail data loaded: {len(self.mail_data):,} rows, {len(self.mail_data.columns)} columns")
        self.logger.info(f"  Columns: {list(self.mail_data.columns)}")
        
        # Load call data
        self.logger.info(f"Loading call data from: {self.config['CALL_FILE_PATH']}")
        
        if not os.path.exists(self.config['CALL_FILE_PATH']):
            raise FileNotFoundError(f"Call file not found: {self.config['CALL_FILE_PATH']}")
        
        if self.config['CALL_FILE_PATH'].lower().endswith('.csv'):
            self.call_data = pd.read_csv(self.config['CALL_FILE_PATH'])
        elif self.config['CALL_FILE_PATH'].lower().endswith(('.xlsx', '.xls')):
            self.call_data = pd.read_excel(self.config['CALL_FILE_PATH'])
        else:
            raise ValueError("Call file must be CSV or Excel format")
        
        self.logger.info(f"✓ Call data loaded: {len(self.call_data):,} rows, {len(self.call_data.columns)} columns")
        self.logger.info(f"  Columns: {list(self.call_data.columns)}")
        
        return True
        
    except Exception as e:
        self.logger.error(f"✗ Error loading data: {str(e)}")
        return False

def _process_data(self) -> bool:
    """Process and validate both datasets"""
    
    self.logger.info("\n" + "=" * 60)
    self.logger.info("STEP 2: PROCESSING DATA")
    self.logger.info("=" * 60)
    
    try:
        # Process mail data
        self.logger.info("Processing mail data...")
        
        # Map mail columns
        mail_mapping = self._map_columns(self.mail_data, self.config['MAIL_COLUMNS'], 'mail')
        if mail_mapping:
            self.mail_data = self.mail_data.rename(columns=mail_mapping)
        
        # Validate required columns
        if 'date' not in self.mail_data.columns:
            raise ValueError("Date column not found in mail data")
        if 'volume' not in self.mail_data.columns:
            raise ValueError("Volume column not found in mail data")
        
        # Process dates and volumes
        self.mail_data = self._process_dates_and_volumes(self.mail_data, 'mail')
        
        # Aggregate mail data
        self.mail_data_agg = self._aggregate_mail_data()
        
        # Process call data
        self.logger.info("Processing call data...")
        
        # Map call columns
        call_mapping = self._map_columns(self.call_data, self.config['CALL_COLUMNS'], 'call')
        if call_mapping:
            self.call_data = self.call_data.rename(columns=call_mapping)
        
        # Validate required columns
        if 'date' not in self.call_data.columns:
            raise ValueError("Date column not found in call data")
        if 'volume' not in self.call_data.columns:
            raise ValueError("Volume column not found in call data")
        
        # Process dates and volumes
        self.call_data = self._process_dates_and_volumes(self.call_data, 'call')
        
        # Aggregate call data
        self.call_data_agg = self.call_data.groupby('date')['volume'].sum().reset_index()
        
        # Log results
        self.logger.info(f"✓ Mail data processed: {len(self.mail_data_agg):,} unique dates")
        self.logger.info(f"  Date range: {self.mail_data_agg['date'].min()} to {self.mail_data_agg['date'].max()}")
        self.logger.info(f"  Total volume: {self.mail_data_agg['volume'].sum():,}")
        
        self.logger.info(f"✓ Call data processed: {len(self.call_data_agg):,} unique dates")
        self.logger.info(f"  Date range: {self.call_data_agg['date'].min()} to {self.call_data_agg['date'].max()}")
        self.logger.info(f"  Total volume: {self.call_data_agg['volume'].sum():,}")
        
        return True
        
    except Exception as e:
        self.logger.error(f"✗ Error processing data: {str(e)}")
        return False

def _map_columns(self, data: pd.DataFrame, column_config: Dict, data_type: str) -> Dict:
    """Map column names based on configuration"""
    
    mapping = {}
    for standard_name, config_name in column_config.items():
        if config_name in data.columns:
            if config_name != standard_name:
                mapping[config_name] = standard_name
        else:
            # Try fuzzy matching
            similar_cols = [col for col in data.columns 
                          if config_name.lower() in col.lower() or col.lower() in config_name.lower()]
            if similar_cols:
                mapping[similar_cols[0]] = standard_name
                self.logger.warning(f"Using '{similar_cols[0]}' for '{standard_name}' in {data_type} data")
            elif standard_name in ['date', 'volume']:  # Required columns
                self.logger.warning(f"Required column '{config_name}' not found in {data_type} data")
    
    if mapping:
        self.logger.info(f"Column mapping for {data_type} data: {mapping}")
    
    return mapping

def _process_dates_and_volumes(self, data: pd.DataFrame, data_type: str) -> pd.DataFrame:
    """Process date and volume columns"""
    
    # Convert dates
    if self.config['DATE_FORMAT']:
        data['date'] = pd.to_datetime(data['date'], format=self.config['DATE_FORMAT'], errors='coerce')
    else:
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
    
    # Check for invalid dates
    invalid_dates = data['date'].isnull().sum()
    if invalid_dates > 0:
        self.logger.warning(f"Found {invalid_dates:,} invalid dates in {data_type} data")
        data = data.dropna(subset=['date'])
    
    # Convert volumes to numeric
    data['volume'] = pd.to_numeric(data['volume'], errors='coerce')
    invalid_volumes = data['volume'].isnull().sum()
    if invalid_volumes > 0:
        self.logger.warning(f"Found {invalid_volumes:,} invalid volumes in {data_type} data")
        data = data.dropna(subset=['volume'])
    
    # Check for negative volumes
    negative_volumes = (data['volume'] < 0).sum()
    if negative_volumes > 0:
        self.logger.warning(f"Found {negative_volumes:,} negative volumes in {data_type} data")
        data = data[data['volume'] >= 0]
    
    return data

def _aggregate_mail_data(self) -> pd.DataFrame:
    """Aggregate mail data by date and preserve type information"""
    
    # Store original data with types for plotting
    self.mail_data_with_types = self.mail_data.copy()
    
    # Aggregate by date
    if 'type' in self.mail_data.columns:
        agg_data = self.mail_data.groupby('date').agg({
            'volume': 'sum',
            'type': lambda x: '|'.join(x.astype(str).unique())
        }).reset_index()
        agg_data.columns = ['date', 'volume', 'types_combined']
        
        # Log type information
        type_counts = self.mail_data['type'].value_counts()
        self.logger.info(f"Mail types found: {len(type_counts)}")
        for mail_type, count in type_counts.head(10).items():
            self.logger.info(f"  {mail_type}: {count:,} records")
            
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
        # Analyze mail data outliers
        self.logger.info("Analyzing mail data outliers...")
        mail_outliers = self._detect_outliers(self.mail_data_agg, 'volume')
        
        self.logger.info(f"Mail outliers detected: {mail_outliers.sum():,} ({mail_outliers.sum()/len(self.mail_data_agg)*100:.1f}%)")
        
        if mail_outliers.sum() > 0:
            outlier_data = self.mail_data_agg[mail_outliers].nlargest(5, 'volume')
            self.logger.info("Top 5 mail volume outliers:")
            for _, row in outlier_data.iterrows():
                self.logger.info(f"  {row['date'].strftime('%Y-%m-%d')}: {row['volume']:,}")
        
        # Clean mail data
        if self.config['REMOVE_OUTLIERS'] and mail_outliers.sum() > 0:
            self.mail_data_clean = self.mail_data_agg[~mail_outliers].reset_index(drop=True)
            self.logger.info(f"✓ Removed {mail_outliers.sum():,} mail outliers")
        else:
            self.mail_data_clean = self.mail_data_agg.copy()
            self.logger.info("✓ No mail outliers removed")
        
        # Analyze call data outliers
        self.logger.info("Analyzing call data outliers...")
        call_outliers = self._detect_outliers(self.call_data_agg, 'volume')
        
        self.logger.info(f"Call outliers detected: {call_outliers.sum():,} ({call_outliers.sum()/len(self.call_data_agg)*100:.1f}%)")
        
        if call_outliers.sum() > 0:
            outlier_data = self.call_data_agg[call_outliers].nlargest(5, 'volume')
            self.logger.info("Top 5 call volume outliers:")
            for _, row in outlier_data.iterrows():
                self.logger.info(f"  {row['date'].strftime('%Y-%m-%d')}: {row['volume']:,}")
        
        # Clean call data
        if self.config['REMOVE_OUTLIERS'] and call_outliers.sum() > 0:
            self.call_data_clean = self.call_data_agg[~call_outliers].reset_index(drop=True)
            self.logger.info(f"✓ Removed {call_outliers.sum():,} call outliers")
        else:
            self.call_data_clean = self.call_data_agg.copy()
            self.logger.info("✓ No call outliers removed")
        
        # Final statistics
        self.logger.info(f"Final clean datasets:")
        self.logger.info(f"  Mail data: {len(self.mail_data_clean):,} records")
        self.logger.info(f"  Call data: {len(self.call_data_clean):,} records")
        
        return True
        
    except Exception as e:
        self.logger.error(f"✗ Error cleaning data: {str(e)}")
        return False

def _detect_outliers(self, data: pd.DataFrame, column: str) -> pd.Series:
    """Detect outliers using multiple methods"""
    
    outliers = pd.DataFrame(index=data.index)
    
    # IQR method
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers['iqr'] = (data[column] < lower_bound) | (data[column] > upper_bound)
    
    # Z-score method
    z_scores = np.abs(stats.zscore(data[column]))
    outliers['zscore'] = z_scores > 3
    
    # Modified Z-score method
    median = data[column].median()
    mad = np.median(np.abs(data[column] - median))
    if mad > 0:
        modified_z_scores = 0.6745 * (data[column] - median) / mad
        outliers['modified_zscore'] = np.abs(modified_z_scores) > 3.5
    else:
        outliers['modified_zscore'] = False
    
    # Isolation Forest method
    try:
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        outliers['isolation'] = iso_forest.fit_predict(data[[column]].values) == -1
    except:
        outliers['isolation'] = False
    
    # Consensus: outlier if detected by at least 2 methods
    return outliers.sum(axis=1) >= 2

def _analyze_correlations(self) -> bool:
    """Analyze correlations and find optimal lag"""
    
    self.logger.info("\n" + "=" * 60)
    self.logger.info("STEP 4: CORRELATION ANALYSIS")
    self.logger.info("=" * 60)
    
    try:
        # Combine datasets
        combined = pd.merge(self.call_data_clean, self.mail_data_clean, 
                          on='date', how='outer', suffixes=('_call', '_mail'))
        combined = combined.sort_values('date').reset_index(drop=True)
        combined['volume_call'] = combined['volume_call'].fillna(0)
        combined['volume_mail'] = combined['volume_mail'].fillna(0)
        
        # Analyze overlap
        both_exist = (combined['volume_call'] > 0) & (combined['volume_mail'] > 0)
        overlap_data = combined[both_exist]
        
        self.logger.info(f"Combined dataset: {len(combined):,} records")
        self.logger.info(f"Records with both call & mail data: {len(overlap_data):,}")
        self.logger.info(f"Overlap percentage: {len(overlap_data)/len(combined)*100:.1f}%")
        
        if len(overlap_data) < self.config['MIN_OVERLAP_RECORDS']:
            self.logger.warning(f"Insufficient overlapping data for correlation analysis (need ≥{self.config['MIN_OVERLAP_RECORDS']})")
            return False
        
        # Basic correlations
        corr_pearson, p_pearson = pearsonr(overlap_data['volume_call'], overlap_data['volume_mail'])
        corr_spearman, p_spearman = spearmanr(overlap_data['volume_call'], overlap_data['volume_mail'])
        
        self.logger.info(f"Pearson correlation: {corr_pearson:.4f} (p-value: {p_pearson:.4f})")
        self.logger.info(f"Spearman correlation: {corr_spearman:.4f} (p-value: {p_spearman:.4f})")
        
        # Lag analysis
        self.logger.info(f"Testing lag correlations (0-{self.config['MAX_LAG_DAYS']} days)...")
        lag_results = []
        
        for lag in range(0, self.config['MAX_LAG_DAYS'] + 1):
            mail_lagged = combined['volume_mail'].shift(lag)
            mask = (combined['volume_call'] > 0) & (mail_lagged > 0)
            
            if mask.sum() >= self.config['MIN_OVERLAP_RECORDS']:
                try:
                    corr, p_val = pearsonr(combined.loc[mask, 'volume_call'], mail_lagged[mask])
                    lag_results.append({
                        'lag': lag,
                        'correlation': corr,
                        'p_value': p_val,
                        'n_obs': mask.sum()
                    })
                except:
                    continue
        
        # Find best lag
        if lag_results:
            lag_df = pd.DataFrame(lag_results)
            best_lag = lag_df.loc[lag_df['correlation'].idxmax()]
            
            self.logger.info(f"✓ Best lag found: {best_lag['lag']} days")
            self.logger.info(f"  Correlation: {best_lag['correlation']:.4f}")
            self.logger.info(f"  P-value: {best_lag['p_value']:.4f}")
            self.logger.info(f"  Observations: {best_lag['n_obs']:,}")
            
            # Show top 5 lags
            top_lags = lag_df.nlargest(5, 'correlation')
            self.logger.info("Top 5 lag periods:")
            for _, row in top_lags.iterrows():
                self.logger.info(f"  {row['lag']} days: {row['correlation']:.4f} (n={row['n_obs']})")
        
        # Response rate analysis
        overlap_data['response_rate'] = overlap_data['volume_call'] / overlap_data['volume_mail'] * 100
        valid_rates = overlap_data[overlap_data['response_rate'] <= self.config['MAX_RESPONSE_RATE']]['response_rate']
        
        if len(valid_rates) > 0:
            self.logger.info(f"Response rate analysis (n={len(valid_rates):,}):")
            self.logger.info(f"  Mean: {valid_rates.mean():.2f}%")
            self.logger.info(f"  Median: {valid_rates.median():.2f}%")
            self.logger.info(f"  Std Dev: {valid_rates.std():.2f}%")
            self.logger.info(f"  Range: {valid_rates.min():.2f}% - {valid_rates.max():.2f}%")
        
        # Store results
        self.analysis_results = {
            'correlations': {
                'pearson': corr_pearson,
                'spearman': corr_spearman,
                'pearson_pvalue': p_pearson,
                'spearman_pvalue': p_spearman
            },
            'lag_analysis': lag_df if lag_results else None,
            'best_lag': best_lag if lag_results else None,
            'response_rates': valid_rates if len(valid_rates) > 0 else None,
            'overlap_stats': {
                'total_records': len(combined),
                'overlap_records': len(overlap_data),
                'overlap_percentage': len(overlap_data)/len(combined)*100
            }
        }
        
        self.combined_data = combined
        return True
        
    except Exception as e:
        self.logger.error(f"✗ Error in correlation analysis: {str(e)}")
        return False

def _create_all_plots(self) -> bool:
    """Create all plots and save to files"""
    
    self.logger.info("\n" + "=" * 60)
    self.logger.info("STEP 5: CREATING PLOTS")
    self.logger.info("=" * 60)
    
    try:
        # Set plotting parameters
        plt.rcParams['figure.dpi'] = self.config['DPI']
        plt.rcParams['savefig.dpi'] = self.config['DPI']
        plt.rcParams['font.size'] = self.config['FONT_SIZE']
        
        plots_created = 0
        
        # 1. Mail volume time series with type legend
        self.logger.info("Creating mail volume time series plot...")
        if self._create_mail_timeseries_plot():
            plots_created += 1
        
        # 2. Call volume time series
        self.logger.info("Creating call volume time series plot...")
        if self._create_call_timeseries_plot():
            plots_created += 1
        
        # 3. Combined overlay plot
        self.logger.info("Creating combined overlay plot...")
        if self._create_combined_overlay_plot():
            plots_created += 1
        
        # 4. Distribution plots
        self.logger.info("Creating distribution plots...")
        if self._create_distribution_plots():
            plots_created += 1
        
        # 5. Day of week analysis
        self.logger.info("Creating day of week analysis plots...")
        if self._create_day_of_week_plots():
            plots_created += 1
        
        # 6. Correlation scatter plot
        if self.combined_data is not None:
            self.logger.info("Creating correlation scatter plot...")
            if self._create_correlation_scatter_plot():
                plots_created += 1
        
        # 7. Lag analysis plot
        if self.analysis_results.get('lag_analysis') is not None:
            self.logger.info("Creating lag analysis plot...")
            if self._create_lag_analysis_plot():
                plots_created += 1
        
        # 8. Response rate plot
        if self.analysis_results.get('response_rates') is not None:
            self.logger.info("Creating response rate plot...")
            if self._create_response_rate_plot():
                plots_created += 1
        
        self.logger.info(f"✓ Created {plots_created} plots in {self.plots_dir}")
        return True
        
    except Exception as e:
        self.logger.error(f"✗ Error creating plots: {str(e)}")
        return False

def _create_mail_timeseries_plot(self) -> bool:
    """Create mail volume time series plot with type legend"""
    try:
        fig, ax = plt.subplots(figsize=self.config['FIGURE_SIZE'])
        
        # Check if we have type information
        if hasattr(self, 'mail_data_with_types') and 'type' in self.mail_data_with_types.columns:
            # Plot by type with legend
            unique_types = self.mail_data_with_types['type'].unique()
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_types)))
            
            for i, mail_type in enumerate(unique_types):
                type_data = self.mail_data_with_types[self.mail_data_with_types['type'] == mail_type]
                type_agg = type_data.groupby('date')['volume'].sum().reset_index()
                ax.plot(type_agg['date'], type_agg['volume'], 
                       label=f'{mail_type}', color=colors[i], alpha=0.8, linewidth=2)
            
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            title = 'Mail Volume Over Time by Type'
        else:
            # Plot total volume only
            ax.plot(self.mail_data_clean['date'], self.mail_data_clean['volume'], 
                   color='red', alpha=0.8, linewidth=2)
            title = 'Mail Volume Over Time'
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Mail Volume', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plot_path = os.path.join(self.plots_dir, '01_mail_volume_timeseries.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=self.config['DPI'])
        plt.close()
        return True
        
    except Exception as e:
        self.logger.error(f"Error creating mail timeseries plot: {str(e)}")
        return False

def _create_call_timeseries_plot(self) -> bool:
    """Create call volume time series plot"""
    try:
        fig, ax = plt.subplots(figsize=self.config['FIGURE_SIZE'])
        
        ax.plot(self.call_data_clean['date'], self.call_data_clean['volume'], 
               color='blue', alpha=0.8, linewidth=2)
        ax.set_title('Call Volume Over Time', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Call Volume', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plot_path = os.path.join(self.plots_dir, '02_call_volume_timeseries.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=self.config['DPI'])
        plt.close()
        return True
        
    except Exception as e:
        self.logger.error(f"Error creating call timeseries plot: {str(e)}")
        return False

def _create_combined_overlay_plot(self) -> bool:
    """Create combined overlay plot with normalized values"""
    try:
        fig, ax = plt.subplots(figsize=self.config['FIGURE_SIZE'])
        
        # Normalize for comparison
        mail_max = self.mail_data_clean['volume'].max()
        call_max = self.call_data_clean['volume'].max()
        
        if mail_max > 0:
            mail_norm = self.mail_data_clean['volume'] / mail_max
        else:
            mail_norm = self.mail_data_clean['volume']
            
        if call_max > 0:
            call_norm = self.call_data_clean['volume'] / call_max
        else:
            call_norm = self.call_data_clean['volume']
        
        ax.plot(self.mail_data_clean['date'], mail_norm, 
               label='Mail (normalized)', color='red', alpha=0.7, linewidth=2)
        ax.plot(self.call_data_clean['date'], call_norm, 
               label='Calls (normalized)', color='blue', alpha=0.7, linewidth=2)
        
        ax.set_title('Normalized Mail vs Call Volume Overlay', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Normalized Volume (0-1)', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plot_path = os.path.join(self.plots_dir, '03_combined_overlay.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=self.config['DPI'])
        plt.close()
        return True
        
    except Exception as e:
        self.logger.error(f"Error creating combined overlay plot: {str(e)}")
        return False

def _create_distribution_plots(self) -> bool:
    """Create distribution plots for both datasets"""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        
        # Mail volume histogram
        ax1.hist(self.mail_data_clean['volume'], bins=50, alpha=0.7, color='red', edgecolor='black')
        ax1.set_title('Mail Volume Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Mail Volume')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Call volume histogram
        ax2.hist(self.call_data_clean['volume'], bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax2.set_title('Call Volume Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Call Volume')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Mail volume box plot
        ax3.boxplot(self.mail_data_clean['volume'], patch_artist=True, 
                   boxprops=dict(facecolor='red', alpha=0.7))
        ax3.set_title('Mail Volume Box Plot', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Mail Volume')
        ax3.grid(True, alpha=0.3)
        
        # Call volume box plot
        ax4.boxplot(self.call_data_clean['volume'], patch_artist=True, 
                   boxprops=dict(facecolor='blue', alpha=0.7))
        ax4.set_title('Call Volume Box Plot', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Call Volume')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.plots_dir, '04_distributions.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=self.config['DPI'])
        plt.close()
        return True
        
    except Exception as e:
        self.logger.error(f"Error creating distribution plots: {str(e)}")
        return False

def _create_day_of_week_plots(self) -> bool:
    """Create day of week analysis plots"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Prepare data with day of week
        mail_dow = self.mail_data_clean.copy()
        mail_dow['day_of_week'] = mail_dow['date'].dt.day_name()
        
        call_dow = self.call_data_clean.copy()
        call_dow['day_of_week'] = call_dow['date'].dt.day_name()
        
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Mail volume by day of week
        sns.boxplot(data=mail_dow, x='day_of_week', y='volume', order=day_order, ax=ax1)
        ax1.set_title('Mail Volume by Day of Week', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Day of Week')
        ax1.set_ylabel('Mail Volume')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Call volume by day of week
        sns.boxplot(data=call_dow, x='day_of_week', y='volume', order=day_order, ax=ax2)
        ax2.set_title('Call Volume by Day of Week', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Day of Week')
        ax2.set_ylabel('Call Volume')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.plots_dir, '05_day_of_week_analysis.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=self.config['DPI'])
        plt.close()
        return True
        
    except Exception as e:
        self.logger.error(f"Error creating day of week plots: {str(e)}")
        return False

def _create_correlation_scatter_plot(self) -> bool:
    """Create correlation scatter plot"""
    try:
        # Get overlapping data
        both_exist = (self.combined_data['volume_call'] > 0) & (self.combined_data['volume_mail'] > 0)
        scatter_data = self.combined_data[both_exist]
        
        if len(scatter_data) == 0:
            self.logger.warning("No overlapping data for scatter plot")
            return False
        
        fig, ax = plt.subplots(figsize=self.config['FIGURE_SIZE'])
        
        # Create scatter plot
        ax.scatter(scatter_data['volume_mail'], scatter_data['volume_call'], 
                  alpha=0.6, color='purple', s=50)
        
        # Add trend line
        if len(scatter_data) > 1:
            z = np.polyfit(scatter_data['volume_mail'], scatter_data['volume_call'], 1)
            p = np.poly1d(z)
            ax.plot(scatter_data['volume_mail'], p(scatter_data['volume_mail']), 
                   "r--", alpha=0.8, linewidth=2)
        
        # Add correlation info
        if 'correlations' in self.analysis_results:
            corr = self.analysis_results['correlations']['pearson']
            ax.text(0.05, 0.95, f'Pearson r = {corr:.4f}', 
                   transform=ax.transAxes, fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax.set_title('Call Volume vs Mail Volume Correlation', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Mail Volume', fontsize=12)
        ax.set_ylabel('Call Volume', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(self.plots_dir, '06_correlation_scatter.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=self.config['DPI'])
        plt.close()
        return True
        
    except Exception as e:
        self.logger.error(f"Error creating correlation scatter plot: {str(e)}")
        return False

def _create_lag_analysis_plot(self) -> bool:
    """Create lag analysis plot"""
    try:
        lag_df = self.analysis_results['lag_analysis']
        
        fig, ax = plt.subplots(figsize=self.config['FIGURE_SIZE'])
        
        ax.plot(lag_df['lag'], lag_df['correlation'], marker='o', linewidth=2, markersize=6)
        
        # Highlight best lag
        best_lag = self.analysis_results['best_lag']
        ax.axvline(x=best_lag['lag'], color='red', linestyle='--', alpha=0.7, 
                  label=f'Best lag: {best_lag["lag"]} days')
        ax.scatter([best_lag['lag']], [best_lag['correlation']], 
                  color='red', s=100, zorder=5)
        
        ax.set_title('Correlation vs Lag Days', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Lag Days', fontsize=12)
        ax.set_ylabel('Pearson Correlation', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(self.plots_dir, '07_lag_analysis.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=self.config['DPI'])
        plt.close()
        return True
        
    except Exception as e:
        self.logger.error(f"Error creating lag analysis plot: {str(e)}")
        return False

def _create_response_rate_plot(self) -> bool:
    """Create response rate analysis plot"""
    try:
        response_rates = self.analysis_results['response_rates']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Response rate histogram
        ax1.hist(response_rates, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax1.axvline(response_rates.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {response_rates.mean():.2f}%')
        ax1.axvline(response_rates.median(), color='orange', linestyle='--', linewidth=2, 
                   label=f'Median: {response_rates.median():.2f}%')
        ax1.set_title('Response Rate Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Response Rate (%)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Response rate box plot
        ax2.boxplot(response_rates, patch_artist=True, 
                   boxprops=dict(facecolor='green', alpha=0.7))
        ax2.set_title('Response Rate Box Plot', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Response Rate (%)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.plots_dir, '08_response_rate_analysis.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=self.config['DPI'])
        plt.close()
        return True
        
    except Exception as e:
        self.logger.error(f"Error creating response rate plot: {str(e)}")
        return False

def _create_monthly_trends_plot(self) -> bool:
    """Create monthly trends analysis"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Prepare monthly data
        mail_monthly = self.mail_data_clean.copy()
        mail_monthly['month'] = mail_monthly['date'].dt.to_period('M')
        mail_monthly = mail_monthly.groupby('month')['volume'].sum().reset_index()
        mail_monthly['month_str'] = mail_monthly['month'].astype(str)
        
        call_monthly = self.call_data_clean.copy()
        call_monthly['month'] = call_monthly['date'].dt.to_period('M')
        call_monthly = call_monthly.groupby('month')['volume'].sum().reset_index()
        call_monthly['month_str'] = call_monthly['month'].astype(str)
        
        # Mail monthly trends
        ax1.plot(mail_monthly['month_str'], mail_monthly['volume'], 
                marker='o', linewidth=2, markersize=6, color='red')
        ax1.set_title('Monthly Mail Volume Trends', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Mail Volume')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Call monthly trends
        ax2.plot(call_monthly['month_str'], call_monthly['volume'], 
                marker='o', linewidth=2, markersize=6, color='blue')
        ax2.set_title('Monthly Call Volume Trends', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Call Volume')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.plots_dir, '09_monthly_trends.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=self.config['DPI'])
        plt.close()
        return True
        
    except Exception as e:
        self.logger.error(f"Error creating monthly trends plot: {str(e)}")
        return False

def _create_mail_type_analysis_plot(self) -> bool:
    """Create detailed mail type analysis"""
    try:
        if not hasattr(self, 'mail_data_with_types') or 'type' not in self.mail_data_with_types.columns:
            return False
        
        # Mail type volume analysis
        type_stats = self.mail_data_with_types.groupby('type')['volume'].agg(['sum', 'count', 'mean']).reset_index()
        type_stats = type_stats.sort_values('sum', ascending=False)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Total volume by type
        ax1.bar(range(len(type_stats)), type_stats['sum'], color='orange', alpha=0.7)
        ax1.set_title('Total Mail Volume by Type', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Mail Type')
        ax1.set_ylabel('Total Volume')
        ax1.set_xticks(range(len(type_stats)))
        ax1.set_xticklabels(type_stats['type'], rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Average volume by type
        ax2.bar(range(len(type_stats)), type_stats['mean'], color='purple', alpha=0.7)
        ax2.set_title('Average Mail Volume by Type', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Mail Type')
        ax2.set_ylabel('Average Volume')
        ax2.set_xticks(range(len(type_stats)))
        ax2.set_xticklabels(type_stats['type'], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Number of campaigns by type
        ax3.bar(range(len(type_stats)), type_stats['count'], color='brown', alpha=0.7)
        ax3.set_title('Number of Campaigns by Type', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Mail Type')
        ax3.set_ylabel('Number of Campaigns')
        ax3.set_xticks(range(len(type_stats)))
        ax3.set_xticklabels(type_stats['type'], rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Mail type pie chart
        ax4.pie(type_stats['sum'], labels=type_stats['type'], autopct='%1.1f%%', startangle=90)
        ax4.set_title('Mail Volume Distribution by Type', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.plots_dir, '10_mail_type_analysis.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=self.config['DPI'])
        plt.close()
        return True
        
    except Exception as e:
        self.logger.error(f"Error creating mail type analysis plot: {str(e)}")
        return False

def _create_seasonal_analysis_plot(self) -> bool:
    """Create seasonal analysis plots"""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Prepare seasonal data
        mail_seasonal = self.mail_data_clean.copy()
        mail_seasonal['quarter'] = mail_seasonal['date'].dt.quarter
        mail_seasonal['month'] = mail_seasonal['date'].dt.month
        
        call_seasonal = self.call_data_clean.copy()
        call_seasonal['quarter'] = call_seasonal['date'].dt.quarter
        call_seasonal['month'] = call_seasonal['date'].dt.month
        
        # Quarterly analysis - Mail
        mail_quarterly = mail_seasonal.groupby('quarter')['volume'].mean()
        ax1.bar(mail_quarterly.index, mail_quarterly.values, color='red', alpha=0.7)
        ax1.set_title('Average Mail Volume by Quarter', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Quarter')
        ax1.set_ylabel('Average Volume')
        ax1.set_xticks([1, 2, 3, 4])
        ax1.grid(True, alpha=0.3)
        
        # Quarterly analysis - Calls
        call_quarterly = call_seasonal.groupby('quarter')['volume'].mean()
        ax2.bar(call_quarterly.index, call_quarterly.values, color='blue', alpha=0.7)
        ax2.set_title('Average Call Volume by Quarter', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Quarter')
        ax2.set_ylabel('Average Volume')
        ax2.set_xticks([1, 2, 3, 4])
        ax2.grid(True, alpha=0.3)
        
        # Monthly analysis - Mail
        mail_monthly = mail_seasonal.groupby('month')['volume'].mean()
        ax3.plot(mail_monthly.index, mail_monthly.values, marker='o', color='red', linewidth=2)
        ax3.set_title('Average Mail Volume by Month', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Month')
        ax3.set_ylabel('Average Volume')
        ax3.set_xticks(range(1, 13))
        ax3.grid(True, alpha=0.3)
        
        # Monthly analysis - Calls
        call_monthly = call_seasonal.groupby('month')['volume'].mean()
        ax4.plot(call_monthly.index, call_monthly.values, marker='o', color='blue', linewidth=2)
        ax4.set_title('Average Call Volume by Month', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Average Volume')
        ax4.set_xticks(range(1, 13))
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.plots_dir, '11_seasonal_analysis.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=self.config['DPI'])
        plt.close()
        return True
        
    except Exception as e:
        self.logger.error(f"Error creating seasonal analysis plot: {str(e)}")
        return False

def _create_all_plots(self) -> bool:
    """Create all plots and save to files"""
    
    self.logger.info("\n" + "=" * 60)
    self.logger.info("STEP 5: CREATING COMPREHENSIVE EDA PLOTS")
    self.logger.info("=" * 60)
    
    try:
        # Set plotting parameters
        plt.rcParams['figure.dpi'] = self.config['DPI']
        plt.rcParams['savefig.dpi'] = self.config['DPI']
        plt.rcParams['font.size'] = self.config['FONT_SIZE']
        
        plots_created = 0
        plot_functions = [
            ("Mail volume time series with type legend", self._create_mail_timeseries_plot),
            ("Call volume time series", self._create_call_timeseries_plot),
            ("Combined overlay plot", self._create_combined_overlay_plot),
            ("Distribution plots", self._create_distribution_plots),
            ("Day of week analysis", self._create_day_of_week_plots),
            ("Monthly trends", self._create_monthly_trends_plot),
            ("Seasonal analysis", self._create_seasonal_analysis_plot),
        ]
        
        # Add conditional plots
        if self.combined_data is not None:
            plot_functions.append(("Correlation scatter plot", self._create_correlation_scatter_plot))
        
        if self.analysis_results.get('lag_analysis') is not None:
            plot_functions.append(("Lag analysis plot", self._create_lag_analysis_plot))
        
        if self.analysis_results.get('response_rates') is not None:
            plot_functions.append(("Response rate analysis", self._create_response_rate_plot))
        
        if hasattr(self, 'mail_data_with_types') and 'type' in self.mail_data_with_types.columns:
            plot_functions.append(("Mail type analysis", self._create_mail_type_analysis_plot))
        
        # Create all plots
        for plot_name, plot_function in plot_functions:
            self.logger.info(f"Creating {plot_name}...")
            if plot_function():
                plots_created += 1
                self.logger.info(f"✓ {plot_name} created successfully")
            else:
                self.logger.warning(f"✗ Failed to create {plot_name}")
        
        self.logger.info(f"✓ Created {plots_created}/{len(plot_functions)} plots in {self.plots_dir}")
        return True
        
    except Exception as e:
        self.logger.error(f"✗ Error creating plots: {str(e)}")
        return False

def _generate_reports(self) -> bool:
    """Generate comprehensive analysis reports"""
    
    self.logger.info("\n" + "=" * 60)
    self.logger.info("STEP 6: GENERATING REPORTS")
    self.logger.info("=" * 60)
    
    try:
        # Create summary report
        summary_report = self._create_summary_report()
        
        # Save summary report
        report_path = os.path.join(self.reports_dir, 'analysis_summary.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(summary_report)
        
        self.logger.info(f"✓ Summary report saved: {report_path}")
        
        # Create detailed metrics report
        if self.analysis_results:
            metrics_report = self._create_metrics_report()
            metrics_path = os.path.join(self.reports_dir, 'detailed_metrics.txt')
            with open(metrics_path, 'w', encoding='utf-8') as f:
                f.write(metrics_report)
            self.logger.info(f"✓ Metrics report saved: {metrics_path}")
        
        return True
        
    except Exception as e:
        self.logger.error(f"✗ Error generating reports: {str(e)}")
        return False

def _create_summary_report(self) -> str:
    """Create a comprehensive summary report"""
    
    report = []
    report.append("=" * 80)
    report.append("CALL VOLUME TIME SERIES ANALYSIS - SUMMARY REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Data Overview
    report.append("DATA OVERVIEW")
    report.append("-" * 40)
    report.append(f"Mail data file: {self.config['MAIL_FILE_PATH']}")
    report.append(f"Call data file: {self.config['CALL_FILE_PATH']}")
    report.append("")
    
    if self.mail_data_clean is not None:
        report.append(f"Mail Data:")
        report.append(f"  Records: {len(self.mail_data_clean):,}")
        report.append(f"  Date range: {self.mail_data_clean['date'].min()} to {self.mail_data_clean['date'].max()}")
        report.append(f"  Total volume: {self.mail_data_clean['volume'].sum():,}")
        report.append(f"  Average daily volume: {self.mail_data_clean['volume'].mean():.1f}")
        report.append("")
    
    if self.call_data_clean is not None:
        report.append(f"Call Data:")
        report.append(f"  Records: {len(self.call_data_clean):,}")
        report.append(f"  Date range: {self.call_data_clean['date'].min()} to {self.call_data_clean['date'].max()}")
        report.append(f"  Total volume: {self.call_data_clean['volume'].sum():,
```