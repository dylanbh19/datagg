#!/usr/bin/env python3
"""
Customer Communications Intelligence Dashboard v2.0 - COMPLETE VERSION
Requirements Document Implementation

NEW FEATURES IMPLEMENTED:
- Feature 2.1: Intelligent, Intent-First Call Data Ingestion
- Feature 2.2: High-Fidelity Data Filtering & Merging
- Feature 3.1: Multi-Lag, Intent-Level Correlation Heatmap
- Feature 3.2: Timeline with Programmatic Spike Correlation Flagging
- Feature 4.1: Comprehensive Visual Cleanup

ASCII formatted for Windows compatibility.
"""

# --- Core Libraries ---

import pandas as pd
import numpy as np
import warnings
import os
import sys
from datetime import datetime, timedelta
import logging
from pathlib import Path
import holidays

# --- Suppress all warnings ---

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

# --- Visualization & Dashboard ---

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    print("WARNING: Plotly not available. Install with: pip install plotly")
    PLOTLY_AVAILABLE = False

try:
    import dash
    from dash import dcc, html, Input, Output, dash_table
    import dash_bootstrap_components as dbc
    DASH_AVAILABLE = True
except ImportError:
    print("WARNING: Dash not available. Install with: pip install dash dash-bootstrap-components")
    DASH_AVAILABLE = False

# --- Optional Libraries ---

try:
    import yfinance as yf
    FINANCIAL_AVAILABLE = True
except ImportError:
    FINANCIAL_AVAILABLE = False

try:
    from scipy import stats
    from scipy.stats import pearsonr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: Scipy not available. Correlation analysis will be disabled. Install with: pip install scipy")


# =============================================================================
# ENHANCED CONFIGURATION v2.0
# =============================================================================

CONFIG = {
    # Feature 2.1: Intelligent, Intent-First Call Data Ingestion
    'CALL_FILE_PATH_INTENTS': r'data\GenesysExtract_with_intents.csv',  # Primary file
    'CALL_FILE_PATH_OVERVIEW': r'data\GenesysExtract_20250609.csv',     # Fallback file
    
    'MAIL_FILE_PATH': r'merged_output.csv',
    'MAIL_COLUMNS': {'date': 'mail_date', 'volume': 'mail_volume', 'type': 'mail_type'},
    'CALL_COLUMNS': {'date': 'ConversationStart', 'intent': 'uui_Intent'},

    # Enhanced financial indicators
    'FINANCIAL_DATA': {
        'S&P 500': '^GSPC',
        '10-Year Treasury': '^TNX',
        '2-Year Treasury': '^IRX',
        'Oil (WTI)': 'CL=F',
        'Dollar Index': 'DX-Y.NYB',
        'VIX': '^VIX'
    },

    'MAX_LAG_DAYS': 21,
    'DEBUG_MODE': True,

    # Feature 2.2: High-Fidelity Data Filtering & Merging
    'MAIL_START_DATE': '2024-06-01',  # Filter mail from June 2024+
    'USE_INNER_JOIN': True,  # Use inner join for better correlation signal
    'FILTER_WEEKENDS': True,  # Remove weekends for business-focused analysis
    'FILTER_HOLIDAYS': True,  # Remove US federal holidays

    # Correlation thresholds
    'MOMENTUM_WINDOWS': [3, 7, 14],
    'ROLLING_CORRELATION_WINDOW': 30,
    'MIN_CORRELATION_DISPLAY': 0.05,
    'CORRELATION_SIGNIFICANCE_THRESHOLD': 0.10,

    # Feature 3.2: Spike correlation settings
    'SPIKE_THRESHOLD_PERCENTILE': 90,  # 90th percentile for spike detection
    'SPIKE_ROLLING_WINDOW': 30,  # 30-day rolling window for spike detection

    # Feature 4.1: Thematic Color Consistency
    'THEME_COLORS': {
        'calls': '#e74c3c',
        'mail': '#3498db',
        'momentum': '#27ae60',
        'correlation': '#f39c12',
        'financial': '#9b59b6',
        'spike': '#f1c40f'
    },

    # Less aggressive outlier removal
    'OUTLIER_THRESHOLD': 4,
    'MIN_MAIL_VOLUME_PERCENTILE': 1,
}

# =============================================================================
# ENHANCED LOGGING
# =============================================================================

def setup_logging():
    """Setup comprehensive logging."""
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            stream=sys.stdout
        )
        return logging.getLogger('CustomerCommsDashboard')
    except Exception:
        print("Logging setup failed, using print statements")
        return None

# =============================================================================
# ENHANCED DATA PROCESSOR v2.0
# =============================================================================

class CustomerCommunicationsDataProcessor:
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger or self._dummy_logger()

        # Core dataframes
        self.mail_df = pd.DataFrame()
        self.call_df = pd.DataFrame()
        self.financial_df = pd.DataFrame()
        self.combined_df = pd.DataFrame()

        # Feature 2.1: Intent data availability flag
        self.has_intent_data = False

        # Analysis results
        self.momentum_correlation_results = pd.DataFrame()
        self.rolling_correlation_results = pd.DataFrame()
        self.correlation_data_table = pd.DataFrame()
        self.mail_by_type = pd.DataFrame()
        self.calls_by_intent = pd.DataFrame()
        self.efficiency_metrics = {}

        # Feature 3.1: Intent-level correlation results
        self.intent_correlation_results = pd.DataFrame()

        # Feature 3.2: Spike correlation data
        self.spike_data = pd.DataFrame()

    def _dummy_logger(self):
        """Create dummy logger that prints instead."""
        class DummyLogger:
            def info(self, msg): print(f"INFO: {msg}")
            def warning(self, msg): print(f"WARNING: {msg}")
            def error(self, msg): print(f"ERROR: {msg}")
        return DummyLogger()

    def safe_load_csv(self, file_path, description="file"):
        """Safely load CSV with multiple fallback strategies."""
        try:
            if not os.path.exists(file_path):
                self.logger.warning(f"{description} not found: {file_path}")
                return pd.DataFrame()

            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip', low_memory=False)
                    self.logger.info(f"✅ {description} loaded: {len(df):,} records")
                    return df
                except (UnicodeDecodeError, Exception):
                    continue

            self.logger.error(f"❌ Failed to load {description}")
            return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"❌ Error loading {description}: {str(e)}")
            return pd.DataFrame()

    def safe_column_mapping(self, df, column_mapping, data_type):
        """Enhanced column mapping with fuzzy matching."""
        try:
            if df.empty:
                return df

            mapped_df = df.copy()

            for standard_name, config_name in column_mapping.items():
                if config_name in mapped_df.columns:
                    if config_name != standard_name:
                        mapped_df = mapped_df.rename(columns={config_name: standard_name})
                    continue

                # Fuzzy matching
                fuzzy_matches = [col for col in mapped_df.columns
                                 if config_name.lower() in col.lower() or col.lower() in config_name.lower()]

                if fuzzy_matches:
                    best_match = fuzzy_matches[0]
                    mapped_df = mapped_df.rename(columns={best_match: standard_name})
                    self.logger.info(f"Mapped {data_type}: '{best_match}' -> '{standard_name}'")

            return mapped_df

        except Exception as e:
            self.logger.error(f"Column mapping failed for {data_type}: {str(e)}")
            return df

    def safe_date_conversion(self, df, date_column):
        """Safe date conversion with validation."""
        try:
            if df.empty or date_column not in df.columns:
                return df

            original_count = len(df)
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            df = df.dropna(subset=[date_column])
            df[date_column] = df[date_column].dt.tz_localize(None).dt.normalize()

            success_rate = len(df) / original_count * 100
            self.logger.info(f"Date conversion success: {success_rate:.1f}%")
            return df

        except Exception as e:
            self.logger.error(f"Date conversion error: {str(e)}")
            return df

# END OF PART 1
# =============================================================================
# Continue with "Continue with Part 2" to get the data loading methods
# =============================================================================
    def load_data(self):
        """
        Feature 2.1: Intelligent, Intent-First Call Data Ingestion
        Load data with intelligent file selection logic.
        """
        try:
            self.logger.info("🔄 Loading customer communications data with intelligent file selection...")

            # Load mail data
            self.mail_df = self.safe_load_csv(self.config['MAIL_FILE_PATH'], "Mail data")
            if not self.mail_df.empty:
                self.mail_df = self.safe_column_mapping(self.mail_df, self.config['MAIL_COLUMNS'], 'mail')
                self.mail_df = self.safe_date_conversion(self.mail_df, 'date')

                if 'volume' in self.mail_df.columns:
                    self.mail_df['volume'] = pd.to_numeric(self.mail_df['volume'], errors='coerce')
                    self.mail_df = self.mail_df.dropna(subset=['volume'])

                # Filter to June 2024+
                start_date = pd.to_datetime(self.config['MAIL_START_DATE'])
                original_len = len(self.mail_df)
                self.mail_df = self.mail_df[self.mail_df['date'] >= start_date]
                self.logger.info(f"Filtered to June 2024+: {len(self.mail_df):,} records (was {original_len:,})")

            # Feature 2.1: INTELLIGENT CALL DATA LOADING
            self.logger.info("🔄 Attempting to load primary call data file with intents...")
            
            # First, try to load the primary file (with intents)
            primary_call_df = self.safe_load_csv(self.config['CALL_FILE_PATH_INTENTS'], "Primary call data (with intents)")
            
            if not primary_call_df.empty:
                # Check if intent column exists
                primary_call_df = self.safe_column_mapping(primary_call_df, self.config['CALL_COLUMNS'], 'call')
                primary_call_df = self.safe_date_conversion(primary_call_df, 'date')
                
                if 'intent' in primary_call_df.columns and primary_call_df['intent'].notna().any():
                    self.logger.info("✅ Primary file loaded successfully with intent data")
                    self.call_df = primary_call_df
                    self.has_intent_data = True
                else:
                    self.logger.warning("⚠️ Primary file exists but no valid intent data found")
                    self.has_intent_data = False
            else:
                self.logger.warning("⚠️ Primary call data file not found")
                self.has_intent_data = False

            # Fallback to overview file if primary failed
            if self.call_df.empty:
                self.logger.info("🔄 Loading fallback call data file (overview)...")
                self.call_df = self.safe_load_csv(self.config['CALL_FILE_PATH_OVERVIEW'], "Fallback call data (overview)")
                
                if not self.call_df.empty:
                    self.call_df = self.safe_column_mapping(self.call_df, self.config['CALL_COLUMNS'], 'call')
                    self.call_df = self.safe_date_conversion(self.call_df, 'date')
                    self.logger.info("✅ Fallback call data loaded successfully")
                    self.has_intent_data = False
                else:
                    self.logger.error("❌ Both primary and fallback call data files failed to load")

            # Filter calls to same period as mail
            if not self.call_df.empty and not self.mail_df.empty:
                start_date = pd.to_datetime(self.config['MAIL_START_DATE'])
                original_len = len(self.call_df)
                self.call_df = self.call_df[self.call_df['date'] >= start_date]
                self.logger.info(f"Filtered calls to June 2024+: {len(self.call_df):,} records (was {original_len:,})")

            # Log final status
            self.logger.info(f"📊 Data Loading Summary:")
            self.logger.info(f"    Intent data available: {'✅ YES' if self.has_intent_data else '❌ NO'}")
            self.logger.info(f"    Mail records: {len(self.mail_df):,}")
            self.logger.info(f"    Call records: {len(self.call_df):,}")
            
            return True

        except Exception as e:
            self.logger.error(f"❌ Data loading failed: {str(e)}")
            return False

    def create_sample_data(self):
        """Create realistic sample data from June 2024+ with intent support."""
        try:
            self.logger.info("🔧 Creating sample data from June 2024...")

            # Start from June 2024 as requested
            start_date = pd.to_datetime('2024-06-01').date()
            end_date = datetime.now().date()
            dates = pd.date_range(start=start_date, end=end_date, freq='D')

            np.random.seed(42)

            # Create realistic call patterns (weekday-focused) with intents
            call_data = []
            intents = ['billing_inquiry', 'account_update', 'statement_question', 'balance_check', 'general_support']
            
            for date in dates:
                if date.weekday() < 5:  # Weekdays only
                    daily_calls = max(int(45 + np.random.normal(0, 10)), 0)

                    for intent in intents:
                        intent_calls = max(int(daily_calls * np.random.uniform(0.1, 0.4)), 0)
                        call_data.extend([{'date': date, 'intent': intent}] * intent_calls)

            self.call_df = pd.DataFrame(call_data)
            self.has_intent_data = True  # Sample data includes intents

            # Create mail data with ALL types
            mail_data = []
            for i, date in enumerate(dates):
                if date.weekday() < 5:  # Weekdays only
                    # Create momentum patterns
                    daily_mail = max(int(1200 + np.sin(i/7) * 300 + np.random.normal(0, 200)), 100)

                    # ALL mail types
                    mail_types = [
                        'statement', 'billing', 'notification', 'reminder', 'confirmation',
                        'promotional', 'newsletter', 'alert', 'update', 'notice'
                    ]

                    for mail_type in mail_types:
                        type_volume = max(int(daily_mail * np.random.uniform(0.05, 0.25)), 0)
                        if type_volume > 0:
                            mail_data.append({'date': date, 'type': mail_type, 'volume': type_volume})

            self.mail_df = pd.DataFrame(mail_data)

            self.logger.info("✅ Sample data created (June 2024+, all mail types, with intents)")
            return True

        except Exception as e:
            self.logger.error(f"❌ Sample data creation failed: {str(e)}")
            return False

    def load_enhanced_financial_data(self):
        """Enhanced financial data loading aligned with mail start date."""
        if not FINANCIAL_AVAILABLE:
            self.logger.warning("📊 Financial data unavailable - yfinance not installed")
            return False

        try:
            self.logger.info("🔄 Loading financial data aligned with mail data...")

            # Start from mail start date, not call date
            if not self.mail_df.empty:
                start_date = self.mail_df['date'].min() - timedelta(days=2)
                end_date = self.mail_df['date'].max() + timedelta(days=1)
            else:
                # Fallback to June 2024
                start_date = pd.to_datetime('2024-06-01')
                end_date = datetime.now()

            self.logger.info(f"Loading financial data from {start_date} to {end_date}")

            financial_data = {}
            for name, ticker_str in self.config['FINANCIAL_DATA'].items():
                try:
                    self.logger.info(f"Fetching {name} ({ticker_str})...")
                    ticker_obj = yf.Ticker(ticker_str)
                    data = ticker_obj.history(start=start_date, end=end_date)

                    if not data.empty and 'Close' in data.columns:
                        close_prices = data['Close'].resample('D').last().ffill()

                        # Better normalization: percentage change from first value
                        if len(close_prices) > 1:
                            first_price = close_prices.iloc[0]
                            pct_change = ((close_prices - first_price) / first_price * 100)
                            financial_data[name] = pct_change
                            self.logger.info(f"✅ {name}: {len(close_prices)} data points")

                except Exception as e:
                    self.logger.warning(f"Failed to fetch {name}: {str(e)}")
                    continue

            if financial_data:
                self.financial_df = pd.DataFrame(financial_data)
                self.financial_df.index.name = 'date'
                self.financial_df = self.financial_df.reset_index()
                self.financial_df['date'] = pd.to_datetime(self.financial_df['date']).dt.tz_localize(None).dt.normalize()

                # Feature 2.2: Remove weekends from financial data
                if self.config['FILTER_WEEKENDS']:
                    self.financial_df = self.financial_df[self.financial_df['date'].dt.weekday < 5]

                self.logger.info(f"✅ Financial data loaded: {len(financial_data)} indicators, {len(self.financial_df)} records")
                return True

            return False

        except Exception as e:
            self.logger.warning(f"Financial data loading failed: {str(e)}")
            return False

    def combine_and_clean_data(self):
        """
        Feature 2.2: High-Fidelity Data Filtering & Merging
        Combines data with inner joins and removes weekends/holidays for better correlation signal.
        """
        try:
            self.logger.info("🔄 Combining and cleaning data with high-fidelity filtering...")

            # Feature 2.2: Filter weekends and holidays FIRST
            us_holidays = holidays.UnitedStates()
            
            # Process call data
            if not self.call_df.empty:
                original_call_len = len(self.call_df)
                
                # Filter weekends
                if self.config['FILTER_WEEKENDS']:
                    self.call_df = self.call_df[self.call_df['date'].dt.weekday < 5]
                    self.logger.info(f"Filtered weekends from calls: {len(self.call_df):,} (was {original_call_len:,})")
                
                # Filter holidays
                if self.config['FILTER_HOLIDAYS']:
                    holiday_filter = ~self.call_df['date'].dt.date.isin(us_holidays)
                    self.call_df = self.call_df[holiday_filter]
                    self.logger.info(f"Filtered holidays from calls: {len(self.call_df):,}")

                # Process intent data if available
                if self.has_intent_data and 'intent' in self.call_df.columns:
                    self.calls_by_intent = self.call_df.groupby('intent').size().reset_index(name='count')
                    self.logger.info(f"Intent breakdown: {len(self.calls_by_intent)} intents found")
                
                # Create daily call aggregation
                daily_calls = self.call_df.groupby('date').size().reset_index(name='call_volume')
            else:
                # Create empty structure if no call data
                start_date = pd.to_datetime('2024-06-01').date()
                end_date = datetime.now().date()
                business_days = pd.bdate_range(start=start_date, end=end_date)
                daily_calls = pd.DataFrame({'date': business_days, 'call_volume': 0})

            # Process mail data
            if not self.mail_df.empty and 'volume' in self.mail_df.columns:
                original_mail_len = len(self.mail_df)
                
                # Filter weekends
                if self.config['FILTER_WEEKENDS']:
                    self.mail_df = self.mail_df[self.mail_df['date'].dt.weekday < 5]
                    self.logger.info(f"Filtered weekends from mail: {len(self.mail_df):,} (was {original_mail_len:,})")
                
                # Filter holidays
                if self.config['FILTER_HOLIDAYS']:
                    holiday_filter = ~self.mail_df['date'].dt.date.isin(us_holidays)
                    self.mail_df = self.mail_df[holiday_filter]
                    self.logger.info(f"Filtered holidays from mail: {len(self.mail_df):,}")

                self.mail_by_type = self.mail_df.groupby('type')['volume'].sum().reset_index()
                daily_mail = self.mail_df.groupby('date')['volume'].sum().reset_index(name='mail_volume')
                
                # Feature 2.2: INNER JOIN for better correlation signal
                if self.config['USE_INNER_JOIN']:
                    combined = pd.merge(daily_calls, daily_mail, on='date', how='inner')
                    self.logger.info(f"Inner join result: {len(combined)} records with both mail and call data")
                else:
                    combined = pd.merge(daily_calls, daily_mail, on='date', how='outer')
                    self.logger.info(f"Outer join result: {len(combined)} records")
            else:
                combined = daily_calls.copy()
                combined['mail_volume'] = 0
                self.logger.warning("No mail data available, using zero mail volume")

            # Fill missing values (should be minimal with inner join)
            combined['call_volume'] = combined['call_volume'].fillna(0)
            combined['mail_volume'] = combined['mail_volume'].fillna(0)

            # Less aggressive outlier removal
            if len(combined) > 10:
                # Remove only extreme outliers (bottom 1%)
                mail_threshold = combined['mail_volume'].quantile(self.config['MIN_MAIL_VOLUME_PERCENTILE'] / 100)
                original_len = len(combined)
                combined = combined[combined['mail_volume'] >= mail_threshold]
                self.logger.info(f"Removed {original_len - len(combined)} low-volume days")

                # Less aggressive outlier removal (4 sigma instead of 3)
                for col in ['call_volume', 'mail_volume']:
                    if combined[col].std() > 0:
                        mean_val = combined[col].mean()
                        std_val = combined[col].std()
                        threshold = self.config['OUTLIER_THRESHOLD']  # 4 sigma
                        before_len = len(combined)
                        combined = combined[
                            (combined[col] >= mean_val - threshold * std_val) &
                            (combined[col] <= mean_val + threshold * std_val)
                        ]
                        if len(combined) < before_len:
                            self.logger.info(f"Removed {before_len - len(combined)} {col} outliers")

            # Add financial data
            if not self.financial_df.empty:
                combined = pd.merge(combined, self.financial_df, on='date', how='left')
                financial_cols = [col for col in self.financial_df.columns if col != 'date']
                for col in financial_cols:
                    if col in combined.columns:
                        combined[col] = combined[col].ffill().bfill()

            # Calculate mail momentum properly
            for window in self.config['MOMENTUM_WINDOWS']:
                # Mail momentum = percentage change over window
                combined[f'mail_momentum_{window}d'] = combined['mail_volume'].pct_change(periods=window) * 100
                combined[f'mail_momentum_{window}d'] = combined[f'mail_momentum_{window}d'].fillna(0)

                # Mail acceleration = change in momentum
                combined[f'mail_acceleration_{window}d'] = combined[f'mail_momentum_{window}d'].diff()
                combined[f'mail_acceleration_{window}d'] = combined[f'mail_acceleration_{window}d'].fillna(0)

            # Add other features
            combined['weekday'] = combined['date'].dt.day_name()
            combined['month'] = combined['date'].dt.month
            combined['day_of_week'] = combined['date'].dt.weekday

            # Normalization
            for col in ['call_volume', 'mail_volume']:
                if combined[col].std() > 0:
                    combined[f'{col}_norm'] = (combined[col] - combined[col].mean()) / combined[col].std()
                else:
                    combined[f'{col}_norm'] = 0

            # Efficiency metrics
            combined['calls_per_1k_mails'] = np.where(
                combined['mail_volume'] > 0,
                (combined['call_volume'] / combined['mail_volume']) * 1000,
                0
            )

            # Feature 3.2: Calculate rolling correlation for spike detection
            combined = self.calculate_spike_correlation(combined)

            # Sort and finalize
            combined = combined.sort_values('date').reset_index(drop=True)
            self.combined_df = combined

            self.logger.info(f"✅ Combined dataset: {len(combined):,} clean business day records")
            self.logger.info(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Data combination failed: {str(e)}")
            return False

    def calculate_spike_correlation(self, df):
        """
        Feature 3.2: Timeline with Programmatic Spike Correlation Flagging
        Calculate rolling correlation and identify spikes for visual flagging.
        """
        try:
            if len(df) < self.config['SPIKE_ROLLING_WINDOW'] + 5:
                self.logger.warning("Insufficient data for spike correlation calculation")
                return df

            # Calculate 30-day rolling correlation
            window_size = self.config['SPIKE_ROLLING_WINDOW']
            rolling_corr = []
            
            for i in range(window_size, len(df)):
                window_data = df.iloc[i-window_size:i]
                
                if len(window_data) >= 10 and window_data['mail_volume'].std() > 0 and window_data['call_volume'].std() > 0:
                    try:
                        if SCIPY_AVAILABLE:
                            corr, _ = pearsonr(window_data['mail_volume'], window_data['call_volume'])
                            if np.isfinite(corr):
                                rolling_corr.append(corr)
                            else:
                                rolling_corr.append(0)
                        else:
                            rolling_corr.append(0)
                    except:
                        rolling_corr.append(0)
                else:
                    rolling_corr.append(0)

            # Add rolling correlation to dataframe
            df['rolling_correlation'] = np.nan
            if len(rolling_corr) > 0:
                df.loc[window_size:window_size+len(rolling_corr)-1, 'rolling_correlation'] = rolling_corr
                df['rolling_correlation'] = df['rolling_correlation'].fillna(0)

                # Calculate spike threshold (90th percentile)
                spike_threshold = np.percentile(rolling_corr, self.config['SPIKE_THRESHOLD_PERCENTILE'])
                self.logger.info(f"Spike threshold (90th percentile): {spike_threshold:.3f}")

                # Create spike highlight column
                df['spike_highlight'] = np.where(
                    df['rolling_correlation'] >= spike_threshold,
                    df['call_volume'],
                    np.nan
                )

                # Store spike data for reference
                spike_dates = df[df['rolling_correlation'] >= spike_threshold]
                if not spike_dates.empty:
                    self.spike_data = spike_dates[['date', 'rolling_correlation', 'call_volume', 'mail_volume']].copy()
                    self.logger.info(f"Identified {len(spike_dates)} correlation spike dates")

            return df

        except Exception as e:
            self.logger.error(f"Spike correlation calculation failed: {str(e)}")
            return df

# END OF PART 2
# =============================================================================
# Continue with "Continue with Part 3" to get the analysis methods
# =============================================================================

    def analyze_mail_momentum_correlation(self):
        """Analyze mail momentum correlation with detailed results."""
        if not SCIPY_AVAILABLE or self.combined_df.empty:
            self.logger.warning("Momentum correlation analysis skipped")
            return False

        self.logger.info("🔄 Analyzing mail momentum correlation...")

        try:
            df = self.combined_df
            max_lag = self.config['MAX_LAG_DAYS']
            all_correlations = []

            # Check if we have momentum columns
            momentum_cols = [col for col in df.columns if 'momentum' in col]
            if not momentum_cols:
                self.logger.warning("No momentum columns found")
                return False

            self.logger.info(f"Found momentum columns: {momentum_cols}")

            # Test each momentum window
            for momentum_window in self.config['MOMENTUM_WINDOWS']:
                momentum_col = f'mail_momentum_{momentum_window}d'
                acceleration_col = f'mail_acceleration_{momentum_window}d'

                if momentum_col in df.columns:
                    # Clean data for this analysis
                    momentum_data = df[momentum_col].replace([np.inf, -np.inf], np.nan).dropna()

                    if len(momentum_data) >= 15:
                        self.logger.info(f"Testing {momentum_col} with {len(momentum_data)} data points")

                        # Test momentum correlation at different lags
                        for lag in range(max_lag + 1):
                            try:
                                # Align data properly
                                if lag == 0:
                                    call_data = df['call_volume']
                                    momentum_data_aligned = df[momentum_col]
                                else:
                                    call_data = df['call_volume'].shift(-lag)
                                    momentum_data_aligned = df[momentum_col]

                                # Create valid dataset
                                valid_df = pd.DataFrame({
                                    'momentum': momentum_data_aligned,
                                    'calls': call_data
                                }).replace([np.inf, -np.inf], np.nan).dropna()

                                if len(valid_df) >= 10:
                                    corr, p_val = pearsonr(valid_df['momentum'], valid_df['calls'])

                                    if np.isfinite(corr):
                                        all_correlations.append({
                                            'type': 'Momentum',
                                            'window': momentum_window,
                                            'lag_days': lag,
                                            'correlation': corr,
                                            'p_value': p_val,
                                            'significant': p_val < self.config['CORRELATION_SIGNIFICANCE_THRESHOLD'],
                                            'sample_size': len(valid_df)
                                        })

                                        # Log significant findings
                                        if abs(corr) > 0.1 and p_val < 0.1:
                                            self.logger.info(f"Found correlation: {momentum_col} lag {lag}d = {corr:.3f} (p={p_val:.3f})")

                            except Exception as e:
                                self.logger.warning(f"Error calculating correlation for {momentum_col} lag {lag}: {e}")
                                continue

                # Test acceleration if available
                if acceleration_col in df.columns:
                    acceleration_data = df[acceleration_col].replace([np.inf, -np.inf], np.nan).dropna()

                    if len(acceleration_data) >= 15:
                        for lag in range(0, max_lag + 1, 2):  # Every 2 days for acceleration
                            try:
                                if lag == 0:
                                    call_data = df['call_volume']
                                    accel_data_aligned = df[acceleration_col]
                                else:
                                    call_data = df['call_volume'].shift(-lag)
                                    accel_data_aligned = df[acceleration_col]

                                valid_df = pd.DataFrame({
                                    'acceleration': accel_data_aligned,
                                    'calls': call_data
                                }).replace([np.inf, -np.inf], np.nan).dropna()

                                if len(valid_df) >= 10:
                                    corr, p_val = pearsonr(valid_df['acceleration'], valid_df['calls'])

                                    if np.isfinite(corr):
                                        all_correlations.append({
                                            'type': 'Acceleration',
                                            'window': momentum_window,
                                            'lag_days': lag,
                                            'correlation': corr,
                                            'p_value': p_val,
                                            'significant': p_val < self.config['CORRELATION_SIGNIFICANCE_THRESHOLD'],
                                            'sample_size': len(valid_df)
                                        })

                            except Exception as e:
                                continue

            # Store results and create data table
            if all_correlations:
                self.momentum_correlation_results = pd.DataFrame(all_correlations)

                # Create detailed data table for top correlations
                self.correlation_data_table = self.momentum_correlation_results.copy()
                self.correlation_data_table = self.correlation_data_table.sort_values('correlation', key=abs, ascending=False)
                self.correlation_data_table['abs_correlation'] = self.correlation_data_table['correlation'].abs()

                # Keep top 10 for table display
                self.correlation_data_table = self.correlation_data_table.head(10)

                self.logger.info(f"✅ Momentum correlation analysis complete: {len(all_correlations)} correlations found")

                # Log best correlations
                best_5 = self.correlation_data_table.head(5)
                for _, row in best_5.iterrows():
                    self.logger.info(f"Top correlation: {row['type']} {row['window']}d lag {row['lag_days']}d = {row['correlation']:.3f}")

                return True
            else:
                self.logger.warning("No momentum correlations found")
                return False

        except Exception as e:
            self.logger.error(f"❌ Momentum correlation analysis failed: {e}")
            return False

    def analyze_intent_correlation(self):
        """
        Feature 3.1: Multi-Lag, Intent-Level Correlation Heatmap
        Analyze correlation between mail volume and each call intent across multiple lags.
        """
        if not self.has_intent_data or not SCIPY_AVAILABLE or self.combined_df.empty:
            self.logger.warning("Intent correlation analysis skipped - no intent data or scipy unavailable")
            return False

        self.logger.info("🔄 Analyzing intent-level correlation heatmap...")

        try:
            # Get daily intent counts by pivoting call data
            if self.call_df.empty or 'intent' not in self.call_df.columns:
                self.logger.warning("No intent data available for correlation analysis")
                return False

            # Create daily intent counts
            daily_intents = self.call_df.groupby(['date', 'intent']).size().reset_index(name='count')
            intent_pivot = daily_intents.pivot(index='date', columns='intent', values='count').fillna(0)

            self.logger.info(f"Found {len(intent_pivot.columns)} intents: {list(intent_pivot.columns)}")

            # Get mail volume data aligned with intent dates
            mail_data = self.combined_df[['date', 'mail_volume']].copy()
            
            # Merge with intent data
            intent_mail_data = pd.merge(intent_pivot.reset_index(), mail_data, on='date', how='inner')
            
            if len(intent_mail_data) < 10:
                self.logger.warning("Insufficient overlapping data for intent correlation")
                return False

            # Calculate correlation matrix across different lags
            max_lag = self.config['MAX_LAG_DAYS']
            correlation_matrix = []

            for intent in intent_pivot.columns:
                intent_correlations = {'intent': intent}
                
                for lag in range(max_lag + 1):
                    try:
                        # Align data with lag
                        if lag == 0:
                            intent_data = intent_mail_data[intent]
                            mail_data_aligned = intent_mail_data['mail_volume']
                        else:
                            # Shift intent data to test if mail predicts future calls
                            intent_data = intent_mail_data[intent].shift(-lag)
                            mail_data_aligned = intent_mail_data['mail_volume']

                        # Create valid dataset
                        valid_data = pd.DataFrame({
                            'intent': intent_data,
                            'mail': mail_data_aligned
                        }).dropna()

                        if len(valid_data) >= 10 and valid_data['mail'].std() > 0 and valid_data['intent'].std() > 0:
                            corr, p_val = pearsonr(valid_data['mail'], valid_data['intent'])
                            
                            if np.isfinite(corr):
                                intent_correlations[f'lag_{lag}'] = corr
                                
                                # Log significant correlations
                                if abs(corr) > 0.2 and p_val < 0.05:
                                    self.logger.info(f"Strong intent correlation: {intent} lag {lag}d = {corr:.3f} (p={p_val:.3f})")
                            else:
                                intent_correlations[f'lag_{lag}'] = 0
                        else:
                            intent_correlations[f'lag_{lag}'] = 0

                    except Exception as e:
                        intent_correlations[f'lag_{lag}'] = 0

                correlation_matrix.append(intent_correlations)

            # Store results
            if correlation_matrix:
                self.intent_correlation_results = pd.DataFrame(correlation_matrix)
                self.logger.info(f"✅ Intent correlation analysis complete: {len(correlation_matrix)} intents analyzed")
                
                # Log summary statistics
                lag_cols = [col for col in self.intent_correlation_results.columns if col.startswith('lag_')]
                if lag_cols:
                    for col in lag_cols[:5]:  # Show first 5 lags
                        avg_corr = self.intent_correlation_results[col].mean()
                        max_corr = self.intent_correlation_results[col].max()
                        self.logger.info(f"Average correlation {col}: {avg_corr:.3f}, max: {max_corr:.3f}")

                return True
            else:
                self.logger.warning("No intent correlations calculated")
                return False

        except Exception as e:
            self.logger.error(f"❌ Intent correlation analysis failed: {e}")
            return False

    def analyze_rolling_correlation(self):
        """Analyze rolling correlation with lower thresholds."""
        if not SCIPY_AVAILABLE or self.combined_df.empty:
            return False

        self.logger.info("🔄 Analyzing rolling correlation windows...")

        try:
            df = self.combined_df
            window_size = self.config['ROLLING_CORRELATION_WINDOW']

            if len(df) < window_size + 5:
                self.logger.warning(f"Insufficient data for rolling correlation (need {window_size + 5}, have {len(df)})")
                return False

            rolling_results = []

            # Test key lag periods
            test_lags = [0, 1, 3, 7, 14]

            for lag in test_lags:
                correlations = []
                dates = []

                # Rolling window analysis
                for i in range(window_size, len(df) - max(lag, 1)):
                    window_data = df.iloc[i-window_size:i].copy()

                    if len(window_data) >= 15:  # Minimum window size
                        try:
                            if lag == 0:
                                call_data = window_data['call_volume']
                                mail_data = window_data['mail_volume']
                            else:
                                # Use future calls for prediction
                                future_window = df.iloc[i-window_size+lag:i+lag].copy()
                                if len(future_window) >= 15:
                                    call_data = future_window['call_volume']
                                    mail_data = window_data['mail_volume']
                                else:
                                    continue

                            # Clean data
                            valid_data = pd.DataFrame({
                                'mail': mail_data,
                                'calls': call_data
                            }).dropna()

                            if len(valid_data) >= 10 and valid_data['mail'].std() > 0 and valid_data['calls'].std() > 0:
                                corr, p_val = pearsonr(valid_data['mail'], valid_data['calls'])

                                if np.isfinite(corr):
                                    correlations.append(corr)
                                    dates.append(df.iloc[i]['date'])
                                else:
                                    correlations.append(0)
                                    dates.append(df.iloc[i]['date'])
                            else:
                                correlations.append(0)
                                dates.append(df.iloc[i]['date'])

                        except Exception as e:
                            correlations.append(0)
                            dates.append(df.iloc[i]['date'])

                # Store results for this lag
                for date, corr in zip(dates, correlations):
                    rolling_results.append({
                        'date': date,
                        'lag_days': lag,
                        'correlation': corr,
                        'window_size': window_size
                    })

            if rolling_results:
                self.rolling_correlation_results = pd.DataFrame(rolling_results)

                # Log some statistics
                for lag in test_lags:
                    lag_data = self.rolling_correlation_results[self.rolling_correlation_results['lag_days'] == lag]
                    if not lag_data.empty:
                        avg_corr = lag_data['correlation'].mean()
                        max_corr = lag_data['correlation'].max()
                        self.logger.info(f"Rolling correlation lag {lag}d: avg={avg_corr:.3f}, max={max_corr:.3f}")

                self.logger.info(f"✅ Rolling correlation analysis complete: {len(rolling_results)} data points")
                return True

            return False

        except Exception as e:
            self.logger.error(f"❌ Rolling correlation analysis failed: {e}")
            return False

    def calculate_efficiency_metrics(self):
        """Calculate enhanced efficiency metrics."""
        try:
            df = self.combined_df
            if df.empty:
                return False

            # Basic metrics
            total_calls = df['call_volume'].sum()
            total_mail = df['mail_volume'].sum()
            avg_daily_calls = df['call_volume'].mean()
            avg_daily_mail = df['mail_volume'].mean()

            # Response rate
            response_rate = (total_calls / total_mail * 100) if total_mail > 0 else 0

            # Momentum insights
            momentum_cols = [col for col in df.columns if 'momentum' in col and 'acceleration' not in col]
            if momentum_cols:
                # Use first momentum column
                momentum_data = df[momentum_cols[0]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(momentum_data) > 0:
                    avg_momentum = momentum_data.mean()
                    momentum_volatility = momentum_data.std()
                else:
                    avg_momentum = 0
                    momentum_volatility = 0
            else:
                avg_momentum = 0
                momentum_volatility = 0

            # Efficiency trends
            if len(df) >= 20:
                recent_10 = df.tail(10)['call_volume'].mean()
                previous_10 = df.iloc[-20:-10]['call_volume'].mean()
                call_trend = ((recent_10 - previous_10) / previous_10 * 100) if previous_10 > 0 else 0
            else:
                call_trend = 0

            self.efficiency_metrics = {
                'total_calls': int(total_calls),
                'total_mail': int(total_mail),
                'avg_daily_calls': round(avg_daily_calls, 1),
                'avg_daily_mail': round(avg_daily_mail, 0),
                'response_rate_pct': round(response_rate, 2),
                'avg_momentum': round(avg_momentum, 2),
                'momentum_volatility': round(momentum_volatility, 2),
                'call_trend_10d': round(call_trend, 1)
            }

            return True

        except Exception as e:
            self.logger.error(f"❌ Efficiency calculation failed: {e}")
            return False

# =============================================================================
# ENHANCED DASHBOARD WITH FEATURE 4.1: COMPREHENSIVE VISUAL CLEANUP
# =============================================================================

class CustomerCommunicationsDashboard:
    def __init__(self, data_processor):
        self.dp = data_processor
        self.logger = data_processor.logger

        if not DASH_AVAILABLE:
            self.logger.error("Dash not available. Cannot create dashboard.")
            return

        try:
            self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
            self.app.title = "Customer Communications Intelligence Dashboard v2.0"
            self.setup_layout()
            self.setup_callbacks()

        except Exception as e:
            self.logger.error(f"Dashboard initialization failed: {str(e)}")

    def create_error_figure(self, message):
        """Create a clean error figure with large, readable text."""
        fig = go.Figure()
        fig.add_annotation(
            text=f"⚠️ {message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=24, color='#e74c3c', family="Arial")
        )
        fig.update_layout(
            title="",
            template='plotly_white',
            height=450,
            font=dict(size=16, family="Arial"),
            margin=dict(t=40, b=40, l=40, r=40)
        )
        return fig

    def create_enhanced_overview_figure(self, df):
        """Create highly readable overview with spike correlation flagging."""
        if df.empty:
            return self.create_error_figure("No data available for this period")

        # Create subplot with better spacing
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Communications Trends with Correlation Spikes',
                            'Raw Daily Volumes'),
            vertical_spacing=0.2,
            specs=[[{"secondary_y": True}], [{"secondary_y": True}]]
        )

        # Feature 4.1: Use theme colors consistently
        colors = self.dp.config['THEME_COLORS']

        # Top plot: Main trends
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['call_volume_norm'],
                name='Call Volume',
                line=dict(color=colors['calls'], width=4),
                mode='lines'
            ),
            row=1, col=1, secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['mail_volume_norm'],
                name='Mail Volume',
                line=dict(color=colors['mail'], width=4),
                mode='lines'
            ),
            row=1, col=1, secondary_y=False
        )

        # Feature 3.2: Add spike correlation flagging
        if 'spike_highlight' in df.columns and df['spike_highlight'].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['spike_highlight'],
                    name='Correlation Spikes',
                    mode='markers',
                    marker=dict(
                        symbol='star',
                        size=20,
                        color=colors['spike'],
                        line=dict(color='black', width=2)
                    ),
                    showlegend=True
                ),
                row=1, col=1, secondary_y=False
            )

        # Financial data
        financial_cols = [col for col in df.columns if any(fin in col for fin in ['S&P 500', '10-Year Treasury', 'VIX'])]
        fin_colors = [colors['financial'], '#f39c12', '#27ae60']

        for i, fin_col in enumerate(financial_cols[:3]):
            if fin_col in df.columns and df[fin_col].notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=df['date'],
                        y=df[fin_col] / 5,  # Scale down for better display
                        name=f'{fin_col.replace("_pct", "")}',
                        line=dict(color=fin_colors[i], width=3, dash='dot'),
                        opacity=0.9
                    ),
                    row=1, col=1, secondary_y=True
                )

        # Bottom plot: Raw volumes
        fig.add_trace(
            go.Bar(
                x=df['date'],
                y=df['mail_volume'],
                name='Daily Mail Volume',
                marker_color=f"rgba({int(colors['mail'][1:3], 16)}, {int(colors['mail'][3:5], 16)}, {int(colors['mail'][5:7], 16)}, 0.6)",
                marker_line=dict(width=0)
            ),
            row=2, col=1, secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['call_volume'],
                name='Daily Call Volume',
                line=dict(color=colors['calls'], width=5),
                mode='lines+markers',
                marker=dict(size=8)
            ),
            row=2, col=1, secondary_y=True
        )

        # Feature 4.1: Dramatically improved readability
        fig.update_layout(
            title=dict(
                text="<b>Executive Dashboard: Customer Communications Intelligence v2.0</b>",
                font=dict(size=28, color='#2c3e50', family="Arial Black"),
                x=0.5
            ),
            template='plotly_white',
            height=800,
            font=dict(size=16, family="Arial"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=14)
            ),
            margin=dict(t=120, b=80, l=100, r=100)
        )

        # Large, bold axis labels
        fig.update_yaxes(title_text="<b>Normalized Score</b>", row=1, col=1, secondary_y=False,
                         title_font=dict(size=18, color='#2c3e50'))
        fig.update_yaxes(title_text="<b>Market % Change</b>", row=1, col=1, secondary_y=True,
                         title_font=dict(size=18, color=colors['financial']))
        fig.update_yaxes(title_text="<b>Mail Volume</b>", row=2, col=1, secondary_y=False,
                         title_font=dict(size=18, color=colors['mail']))
        fig.update_yaxes(title_text="<b>Call Volume</b>", row=2, col=1, secondary_y=True,
                         title_font=dict(size=18, color=colors['calls']))
        fig.update_xaxes(title_text="<b>Date</b>", row=2, col=1, title_font=dict(size=18))

        return fig

# END OF PART 3
# =============================================================================
# Continue with "Continue with Part 4" to get the remaining visualization methods
# and dashboard components
# =============================================================================
    def create_intent_correlation_heatmap(self, df_intent):
        """
        Feature 3.1: Multi-Lag, Intent-Level Correlation Heatmap
        Create intent-level correlation heatmap visualization.
        """
        if df_intent.empty:
            return self.create_error_figure("Intent correlation data not available - check if intent data exists")

        try:
            # Prepare data for heatmap
            intent_names = df_intent['intent'].tolist()
            lag_cols = [col for col in df_intent.columns if col.startswith('lag_')]
            
            if not lag_cols:
                return self.create_error_figure("No lag correlation data found")

            # Create correlation matrix
            correlation_matrix = df_intent[lag_cols].values
            lag_labels = [col.replace('lag_', '') + 'd' for col in lag_cols]

            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix,
                x=lag_labels,
                y=intent_names,
                colorscale='RdBu',
                zmid=0,
                colorbar=dict(
                    title="<b>Correlation<br>Coefficient</b>",
                    titlefont=dict(size=16),
                    tickfont=dict(size=14)
                ),
                hoverongaps=False,
                hovertemplate='<b>%{y}</b><br>Lag: %{x}<br>Correlation: <b>%{z:.3f}</b><extra></extra>'
            ))

            fig.update_layout(
                title=dict(
                    text="<b>📊 Intent-Level Correlation Heatmap</b><br><sub>Mail volume correlation with call intents across time lags</sub>",
                    font=dict(size=22, color='#2c3e50', family="Arial"),
                    x=0.5
                ),
                xaxis_title="<b>Lag in Days</b>",
                yaxis_title="<b>Call Intent</b>",
                template='plotly_white',
                height=600,
                font=dict(size=16, family="Arial"),
                margin=dict(t=120, b=80, l=200, r=120)
            )

            return fig

        except Exception as e:
            self.logger.error(f"Intent heatmap creation failed: {e}")
            return self.create_error_figure("Failed to create intent correlation heatmap")

    def create_momentum_correlation_figure(self, df_momentum):
        """Create highly readable momentum correlation chart."""
        if df_momentum.empty:
            return self.create_error_figure("Momentum correlation analysis failed - check data quality")

        # Show top correlations only for readability
        plot_data = df_momentum.copy()
        plot_data['abs_correlation'] = plot_data['correlation'].abs()
        plot_data = plot_data.sort_values('abs_correlation', ascending=False).head(15)

        # Create clean bar chart
        fig = go.Figure()

        # Feature 4.1: Better color coding using theme colors
        colors = self.dp.config['THEME_COLORS']
        bar_colors = []
        for corr in plot_data['correlation']:
            if abs(corr) > 0.3:
                bar_colors.append(colors['calls'])  # Strong - red
            elif abs(corr) > 0.15:
                bar_colors.append(colors['correlation'])  # Moderate - orange
            else:
                bar_colors.append(colors['mail'])  # Weak - blue

        fig.add_trace(go.Bar(
            x=[f"{row['type']}<br>{row['window']}d Window<br>Lag: {row['lag_days']}d"
               for _, row in plot_data.iterrows()],
            y=plot_data['correlation'],
            marker_color=bar_colors,
            text=[f"<b>{corr:.3f}</b>" for corr in plot_data['correlation']],
            textposition='outside',
            textfont=dict(size=14, color='black', family="Arial Bold")
        ))

        # Add reference lines
        fig.add_hline(y=0.3, line_dash="dash", line_color=colors['momentum'], line_width=3,
                      annotation_text="<b>Strong Correlation (0.30)</b>",
                      annotation_font=dict(size=14, color=colors['momentum']))
        fig.add_hline(y=0.15, line_dash="dash", line_color=colors['correlation'], line_width=3,
                      annotation_text="<b>Moderate Correlation (0.15)</b>",
                      annotation_font=dict(size=14, color=colors['correlation']))

        fig.update_layout(
            title=dict(
                text="<b>📈 Mail Momentum vs Call Volume Correlation</b><br><sub>How mail acceleration/deceleration predicts customer calls</sub>",
                font=dict(size=22, color='#2c3e50', family="Arial"),
                x=0.5
            ),
            xaxis_title="<b>Analysis Method & Parameters</b>",
            yaxis_title="<b>Correlation Coefficient</b>",
            template='plotly_white',
            height=600,
            font=dict(size=16, family="Arial"),
            margin=dict(t=120, b=150, l=100, r=100),
            xaxis=dict(tickfont=dict(size=12))
        )

        return fig

    def create_rolling_correlation_figure(self, df_rolling):
        """Create highly readable rolling correlation chart."""
        if df_rolling.empty:
            return self.create_error_figure("Rolling correlation data not available")

        fig = go.Figure()

        # Feature 4.1: Use theme colors consistently
        colors = self.dp.config['THEME_COLORS']
        
        # Enhanced color palette and line styles
        lag_configs = {
            0: {'color': colors['calls'], 'name': 'Same Day', 'width': 4},
            1: {'color': colors['mail'], 'name': '1-Day Lag', 'width': 4},
            3: {'color': colors['momentum'], 'name': '3-Day Lag', 'width': 4},
            7: {'color': colors['correlation'], 'name': '1-Week Lag', 'width': 4},
            14: {'color': colors['financial'], 'name': '2-Week Lag', 'width': 4}
        }

        for lag in df_rolling['lag_days'].unique():
            lag_data = df_rolling[df_rolling['lag_days'] == lag]
            config = lag_configs.get(lag, {'color': '#95a5a6', 'name': f'{int(lag)}-Day Lag', 'width': 3})

            fig.add_trace(go.Scatter(
                x=lag_data['date'],
                y=lag_data['correlation'],
                mode='lines+markers',
                name=config['name'],
                line=dict(width=config['width'], color=config['color']),
                marker=dict(size=8),
                hovertemplate=f"<b>{config['name']}</b><br>Date: %{{x}}<br>Correlation: <b>%{{y:.3f}}</b><extra></extra>"
            ))

        # Enhanced reference lines
        fig.add_hline(y=0.3, line_dash="dash", line_color=colors['momentum'], line_width=2,
                      annotation_text="Strong Correlation", annotation_font=dict(size=12))
        fig.add_hline(y=0.15, line_dash="dash", line_color=colors['correlation'], line_width=2,
                      annotation_text="Moderate Correlation", annotation_font=dict(size=12))
        fig.add_hline(y=0, line_dash="solid", line_color="#95a5a6", line_width=1)
        fig.add_hline(y=-0.15, line_dash="dash", line_color="#e67e22", line_width=2,
                      annotation_text="Negative Correlation", annotation_font=dict(size=12))

        fig.update_layout(
            title=dict(
                text="<b>🔄 Rolling Correlation Analysis Over Time</b><br><sub>30-day rolling windows showing correlation stability</sub>",
                font=dict(size=22, color='#2c3e50', family="Arial"),
                x=0.5
            ),
            xaxis_title="<b>Date</b>",
            yaxis_title="<b>Correlation Coefficient</b>",
            template='plotly_white',
            height=600,
            font=dict(size=16, family="Arial"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=14)
            ),
            margin=dict(t=120, b=80, l=100, r=100)
        )

        return fig

    def create_correlation_data_table(self, df_table):
        """Create detailed correlation data table with professional styling."""
        if df_table.empty:
            return html.Div("No correlation data available",
                            style={'textAlign': 'center', 'fontSize': '18px', 'color': '#e74c3c'})

        # Format the data for display
        display_data = df_table.copy()
        display_data['correlation'] = display_data['correlation'].round(4)
        display_data['p_value'] = display_data['p_value'].round(4)
        display_data['significance'] = display_data['significant'].map({True: '✅ Yes', False: '❌ No'})

        # Rename columns for better display
        display_data = display_data.rename(columns={
            'type': 'Analysis Type',
            'window': 'Window (days)',
            'lag_days': 'Lag (days)',
            'correlation': 'Correlation',
            'p_value': 'P-Value',
            'significance': 'Significant',
            'sample_size': 'Sample Size'
        })

        # Select columns to display
        cols_to_show = ['Analysis Type', 'Window (days)', 'Lag (days)', 'Correlation', 'P-Value', 'Significant', 'Sample Size']
        display_data = display_data[cols_to_show]

        return dash_table.DataTable(
            data=display_data.to_dict('records'),
            columns=[{"name": i, "id": i} for i in display_data.columns],
            style_cell={
                'textAlign': 'center',
                'fontSize': '14px',
                'fontFamily': 'Arial',
                'padding': '12px',
                'backgroundColor': '#f8f9fa'
            },
            style_header={
                'backgroundColor': '#3498db',
                'color': 'white',
                'fontWeight': 'bold',
                'fontSize': '16px',
                'textAlign': 'center'
            },
            style_data_conditional=[
                {
                    'if': {'filter_query': '{Correlation} > 0.2'},
                    'backgroundColor': '#d5f4e6',
                    'color': 'black',
                },
                {
                    'if': {'filter_query': '{Correlation} < -0.2'},
                    'backgroundColor': '#fadbd8',
                    'color': 'black',
                }
            ],
            page_size=10,
            sort_action="native"
        )

    def create_clean_mail_types_figure(self, df_mail):
        """
        Feature 4.1: Mail Breakdown Chart - horizontal bar chart for maximum readability
        """
        if df_mail.empty:
            return self.create_error_figure("Mail communications data unavailable")

        # Show top 10 only for readability
        df_mail = df_mail.sort_values('volume', ascending=False).head(10)

        # Calculate percentages
        total_volume = df_mail['volume'].sum()
        df_mail = df_mail.copy()
        df_mail['percentage'] = (df_mail['volume'] / total_volume * 100).round(1)
        df_mail = df_mail.sort_values('volume', ascending=True)  # For horizontal bar

        # Create horizontal bar chart for maximum readability
        fig = go.Figure()

        # Feature 4.1: Professional color palette using theme colors
        colors = self.dp.config['THEME_COLORS']
        base_colors = [colors['mail'], colors['calls'], colors['momentum'], colors['correlation'], colors['financial']]
        extended_colors = base_colors + ['#e67e22', '#1abc9c', '#34495e', '#e91e63', '#ff5722']

        fig.add_trace(go.Bar(
            y=df_mail['type'],
            x=df_mail['volume'],
            orientation='h',
            marker_color=extended_colors[:len(df_mail)],
            text=[f"<b>{v:,.0f}</b><br>({p}%)" for v, p in zip(df_mail['volume'], df_mail['percentage'])],
            textposition='auto',
            textfont=dict(size=13, color='white', family="Arial Bold")
        ))

        fig.update_layout(
            title=dict(
                text="<b>📧 Mail Communications Breakdown</b><br><sub>Volume distribution by communication type (Top 10)</sub>",
                font=dict(size=22, color='#2c3e50', family="Arial"),
                x=0.5
            ),
            xaxis_title="<b>Total Volume</b>",
            yaxis_title="<b>Mail Type</b>",
            template='plotly_white',
            height=600,
            font=dict(size=16, family="Arial"),
            margin=dict(t=120, b=80, l=200, r=100)
        )

        return fig

    def create_clean_call_intents_figure(self, df_calls):
        """
        Feature 4.1: Call Intent Distribution - Donut Chart with center annotation
        """
        if df_calls.empty:
            return self.create_error_figure("Call intent data unavailable")

        # Calculate percentages
        total_calls = df_calls['count'].sum()
        df_calls = df_calls.copy()
        df_calls['percentage'] = (df_calls['count'] / total_calls * 100).round(1)

        # Feature 4.1: Create donut chart with hole=0.5
        colors = self.dp.config['THEME_COLORS']
        chart_colors = [colors['calls'], colors['mail'], colors['momentum'], colors['correlation'], colors['financial'], '#e67e22'][:len(df_calls)]

        fig = go.Figure(data=[go.Pie(
            labels=df_calls['intent'],
            values=df_calls['count'],
            hole=0.5,  # Feature 4.1: Donut chart
            textinfo='label+percent',
            textposition='outside',
            textfont=dict(size=14, family="Arial Bold"),
            marker=dict(
                colors=chart_colors,
                line=dict(color='white', width=3)
            )
        )])

        # Feature 4.1: Center annotation with total calls
        fig.add_annotation(
            text=f"<b>Total</b><br><span style='font-size:20px'>{total_calls:,}</span><br><b>Calls</b>",
            x=0.5, y=0.5,
            font=dict(size=18, color='#2c3e50', family="Arial"),
            showarrow=False
        )

        fig.update_layout(
            title=dict(
                text="<b>📞 Customer Call Intent Distribution</b><br><sub>Breakdown of call reasons and customer inquiries</sub>",
                font=dict(size=22, color='#2c3e50', family="Arial"),
                x=0.5
            ),
            template='plotly_white',
            height=600,
            font=dict(size=16, family="Arial"),
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05,
                font=dict(size=14)
            ),
            margin=dict(t=120, b=80, l=100, r=200)
        )

        return fig

    def setup_layout(self):
        """Setup highly readable dashboard layout with all features."""
        df = self.dp.combined_df
        if df.empty:
            self.app.layout = html.Div([
                html.H1("⚠️ No Data Available",
                        style={'textAlign': 'center', 'color': '#e74c3c', 'fontSize': '36px'}),
                html.P("Dashboard cannot render without data. Check data sources.",
                       style={'textAlign': 'center', 'fontSize': '20px', 'color': '#7f8c8d'})
            ])
            return

        start_date = df['date'].min().date()
        end_date = df['date'].max().date()

        # Enhanced styling for readability
        main_style = {
            'backgroundColor': '#f8f9fa',
            'minHeight': '100vh',
            'fontFamily': 'Arial, sans-serif'
        }

        header_style = {
            'textAlign': 'center',
            'color': '#2c3e50',
            'marginBottom': '20px',
            'fontSize': '40px',
            'fontWeight': 'bold',
            'fontFamily': 'Arial Black'
        }

        subtitle_style = {
            'textAlign': 'center',
            'color': '#7f8c8d',
            'marginBottom': '40px',
            'fontSize': '20px',
            'fontFamily': 'Arial'
        }

        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("📊 Customer Communications Intelligence v2.0", style=header_style),
                    html.P("Advanced analytics with intent correlation heatmaps and spike detection", style=subtitle_style)
                ], width=12)
            ]),

            # Enhanced controls
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("📅 Analysis Period", style={'color': '#2c3e50', 'fontSize': '20px'}),
                            dcc.DatePickerRange(
                                id='date-picker-range',
                                min_date_allowed=start_date,
                                max_date_allowed=end_date,
                                start_date=start_date,
                                end_date=end_date,
                                display_format='YYYY-MM-DD',
                                style={'fontSize': '16px'}
                            )
                        ])
                    ], className="shadow border-0", style={'borderLeft': '5px solid #3498db'})
                ], width=6),

                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("🎯 Key Insights", style={'color': '#2c3e50', 'fontSize': '20px'}),
                            html.Div(id='quick-insights')
                        ])
                    ], className="shadow border-0", style={'borderLeft': '5px solid #27ae60'})
                ], width=6)
            ], className="mb-4"),

            # Status indicator for intent data
            dbc.Row([
                dbc.Col([
                    dbc.Alert([
                        html.H4("📊 Data Status", className="alert-heading"),
                        html.P([
                            f"Intent Data: {'✅ Available' if self.dp.has_intent_data else '❌ Not Available'} | ",
                            f"Records: {len(self.dp.combined_df):,} | ",
                            f"Correlation Spikes: {len(self.dp.spike_data) if not self.dp.spike_data.empty else 0}"
                        ])
                    ], color="info", style={'fontSize': '16px'})
                ], width=12)
            ], className="mb-4"),

            # KPI Cards
            dbc.Row(id='kpi-cards-row', className="mb-4"),

            # Enhanced Tabs
            dbc.Tabs([
                dbc.Tab(
                    label="📈 EXECUTIVE OVERVIEW",
                    tab_id="tab-overview",
                    label_style={'fontSize': '18px', 'fontWeight': 'bold', 'padding': '15px'},
                    children=[
                        html.Div([
                            dcc.Graph(id='overview-chart', style={'height': '800px'})
                        ], style={'padding': '30px'})
                    ]
                ),

                dbc.Tab(
                    label="🔗 CORRELATION INTELLIGENCE",
                    tab_id="tab-correlation",
                    label_style={'fontSize': '18px', 'fontWeight': 'bold', 'padding': '15px'},
                    children=[
                        html.Div([
                            dbc.Row([
                                dbc.Col([
                                    dcc.Graph(id='momentum-correlation-chart', style={'height': '600px'})
                                ], width=6),
                                dbc.Col([
                                    dcc.Graph(id='rolling-correlation-chart', style={'height': '600px'})
                                ], width=6)
                            ], className="mb-4"),

                            # Feature 3.1: Intent correlation heatmap (conditional)
                            dbc.Row([
                                dbc.Col([
                                    dcc.Graph(id='intent-correlation-heatmap', style={'height': '600px'})
                                ], width=12)
                            ], className="mb-4", id='intent-heatmap-row'),

                            # Correlation Data Table
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader([
                                            html.H4("📊 Detailed Correlation Analysis",
                                                    style={'color': '#2c3e50', 'fontSize': '22px', 'margin': '0'})
                                        ]),
                                        dbc.CardBody([
                                            html.Div(id='correlation-data-table')
                                        ])
                                    ], className="shadow border-0")
                                ], width=12)
                            ])
                        ], style={'padding': '30px'})
                    ]
                ),

                dbc.Tab(
                    label="📊 COMMUNICATIONS BREAKDOWN",
                    tab_id="tab-breakdowns",
                    label_style={'fontSize': '18px', 'fontWeight': 'bold', 'padding': '15px'},
                    children=[
                        html.Div([
                            dbc.Row([
                                dbc.Col([
                                    dcc.Graph(id='mail-types-chart', style={'height': '600px'})
                                ], width=6),
                                dbc.Col([
                                    dcc.Graph(id='call-intents-chart', style={'height': '600px'})
                                ], width=6)
                            ])
                        ], style={'padding': '30px'})
                    ]
                )
            ], className="mb-4", style={'fontSize': '18px'}),

            # Enhanced Footer
            dbc.Row([
                dbc.Col([
                    html.Hr(style={'border': '2px solid #ecf0f1', 'margin': '40px 0'}),
                    html.P([
                        f"📊 Data: {len(self.dp.mail_df):,} mail records, {len(self.dp.call_df):,} call records | ",
                        f"🎯 Intent Data: {'Available' if self.dp.has_intent_data else 'Not Available'} | ",
                        f"📅 Period: {start_date} to {end_date} | ",
                        f"🔄 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                    ], style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '16px', 'fontFamily': 'Arial'})
                ], width=12)
            ])
        ], fluid=True, style=main_style)

    def setup_callbacks(self):
        """Setup enhanced callbacks with all features."""

        @self.app.callback(
            [Output('kpi-cards-row', 'children'),
             Output('overview-chart', 'figure'),
             Output('quick-insights', 'children')],
            [Input('date-picker-range', 'start_date'),
             Input('date-picker-range', 'end_date')]
        )
        def update_overview_and_kpis(start_date, end_date):
            if not start_date or not end_date:
                return [], self.create_error_figure("Please select a date range"), ""

            # Filter data
            dff = self.dp.combined_df[
                (self.dp.combined_df['date'] >= start_date) &
                (self.dp.combined_df['date'] <= end_date)
            ]

            if dff.empty:
                return [], self.create_error_figure("No data for selected period"), ""

            # Enhanced KPI calculations
            total_calls = int(dff['call_volume'].sum())
            total_mail = int(dff['mail_volume'].sum())
            avg_daily_calls = total_calls / len(dff) if len(dff) > 0 else 0
            avg_daily_mail = total_mail / len(dff) if len(dff) > 0 else 0
            response_rate = (total_calls / total_mail * 100) if total_mail > 0 else 0

            # Feature 4.1: Theme colors for KPI cards
            colors = self.dp.config['THEME_COLORS']
            
            # KPI Cards with consistent styling
            card_style = {
                'textAlign': 'center',
                'height': '150px',
                'border': 'none',
                'backgroundColor': 'white'
            }

            kpi_cards = [
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H1(f"{total_calls:,}",
                                    style={'color': colors['calls'], 'marginBottom': '10px', 'fontSize': '36px'}),
                            html.H5("Total Calls",
                                    style={'color': '#2c3e50', 'fontSize': '18px', 'marginBottom': '5px'}),
                            html.P(f"📈 {avg_daily_calls:.0f}/day",
                                   style={'color': '#7f8c8d', 'fontSize': '14px', 'margin': '0'})
                        ])
                    ], style=card_style, className="shadow border-0")
                ], width=3),

                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H1(f"{total_mail:,}",
                                    style={'color': colors['mail'], 'marginBottom': '10px', 'fontSize': '36px'}),
                            html.H5("Total Mail",
                                    style={'color': '#2c3e50', 'fontSize': '18px', 'marginBottom': '5px'}),
                            html.P(f"📧 {avg_daily_mail:.0f}/day",
                                   style={'color': '#7f8c8d', 'fontSize': '14px', 'margin': '0'})
                        ])
                    ], style=card_style, className="shadow border-0")
                ], width=3),

                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H1(f"{response_rate:.1f}%",
                                    style={'color': colors['momentum'], 'marginBottom': '10px', 'fontSize': '36px'}),
                            html.H5("Response Rate",
                                    style={'color': '#2c3e50', 'fontSize': '18px', 'marginBottom': '5px'}),
                            html.P("📞 Calls per 100 mails",
                                   style={'color': '#7f8c8d', 'fontSize': '14px', 'margin': '0'})
                        ])
                    ], style=card_style, className="shadow border-0")
                ], width=3),

                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H1(f"{dff['calls_per_1k_mails'].mean():.0f}" if 'calls_per_1k_mails' in dff.columns else "N/A",
                                    style={'color': colors['correlation'], 'marginBottom': '10px', 'fontSize': '36px'}),
                            html.H5("Efficiency Ratio",
                                    style={'color': '#2c3e50', 'fontSize': '18px', 'marginBottom': '5px'}),
                            html.P("📊 Calls per 1K mails",
                                   style={'color': '#7f8c8d', 'fontSize': '14px', 'margin': '0'})
                        ])
                    ], style=card_style, className="shadow border-0")
                ], width=3)
            ]

            # Enhanced insights
            insights = []
            if not self.dp.momentum_correlation_results.empty:
                best_momentum = self.dp.momentum_correlation_results.loc[
                    self.dp.momentum_correlation_results['correlation'].abs().idxmax()
                ]
                insights.append(
                    dbc.Badge(
                        f"🎯 Best correlation: {best_momentum['correlation']:.3f} ({best_momentum['type']}, {best_momentum['window']}d, lag {best_momentum['lag_days']}d)",
                        color="primary", className="me-2 mb-2",
                        style={'fontSize': '14px', 'padding': '10px 15px'}
                    )
                )

            if not self.dp.spike_data.empty:
                insights.append(
                    dbc.Badge(
                        f"⚡ Correlation spikes detected: {len(self.dp.spike_data)}",
                        color="warning", className="me-2 mb-2",
                        style={'fontSize': '14px', 'padding': '10px 15px'}
                    )
                )

            if self.dp.has_intent_data:
                insights.append(
                    dbc.Badge(
                        f"📊 Intent analysis: {len(self.dp.calls_by_intent)} intents",
                        color="info", className="me-2 mb-2",
                        style={'fontSize': '14px', 'padding': '10px 15px'}
                    )
                )

            overview_fig = self.create_enhanced_overview_figure(dff)

            return kpi_cards, overview_fig, insights

        @self.app.callback(
            [Output('momentum-correlation-chart', 'figure'),
             Output('rolling-correlation-chart', 'figure'),
             Output('intent-correlation-heatmap', 'figure'),
             Output('correlation-data-table', 'children'),
             Output('intent-heatmap-row', 'style')],
            [Input('date-picker-range', 'id')]  # Dummy input
        )
        def update_correlation_tab(_):
            momentum_fig = self.create_momentum_correlation_figure(self.dp.momentum_correlation_results)
            rolling_fig = self.create_rolling_correlation_figure(self.dp.rolling_correlation_results)
            data_table = self.create_correlation_data_table(self.dp.correlation_data_table)

            # Feature 3.1: Conditional intent heatmap rendering
            if self.dp.has_intent_data and not self.dp.intent_correlation_results.empty:
                intent_fig = self.create_intent_correlation_heatmap(self.dp.intent_correlation_results)
                heatmap_style = {'display': 'block'}
            else:
                intent_fig = self.create_error_figure("Intent correlation heatmap not available - no intent data")
                heatmap_style = {'display': 'none'}

            return momentum_fig, rolling_fig, intent_fig, data_table, heatmap_style

        @self.app.callback(
            [Output('mail-types-chart', 'figure'),
             Output('call-intents-chart', 'figure')],
            [Input('date-picker-range', 'id')]  # Dummy input
        )
        def update_breakdown_tab(_):
            mail_fig = self.create_clean_mail_types_figure(self.dp.mail_by_type)
            
            # Feature 4.1: Conditional call intents chart
            if self.dp.has_intent_data and not self.dp.calls_by_intent.empty:
                call_fig = self.create_clean_call_intents_figure(self.dp.calls_by_intent)
            else:
                call_fig = self.create_error_figure("Call intent data not available")
            
            return mail_fig, call_fig

    def run(self, debug=True, port=8050):
        """Run the enhanced dashboard."""
        try:
            self.logger.info(f"🚀 Starting Customer Communications Dashboard v2.0 at http://127.0.0.1:{port}")
            self.app.run(debug=debug, port=port, host='127.0.0.1')
        except Exception as e:
            self.logger.error(f"Dashboard failed to start: {str(e)}")

# =============================================================================
# ENHANCED MAIN FUNCTION
# =============================================================================

def main():
    """Enhanced main function with all v2.0 features."""
    print("🔥 CUSTOMER COMMUNICATIONS INTELLIGENCE DASHBOARD v2.0")
    print("=" * 70)
    print("🎯 NEW FEATURES:")
    print("   • Feature 2.1: Intelligent, Intent-First Call Data Ingestion")
    print("   • Feature 2.2: High-Fidelity Data Filtering & Merging")
    print("   • Feature 3.1: Multi-Lag, Intent-Level Correlation Heatmap")
    print("   • Feature 3.2: Timeline with Programmatic Spike Correlation Flagging")
    print("   • Feature 4.1: Comprehensive Visual Cleanup")
    print("=" * 70)

    logger = setup_logging()

    try:
        if not DASH_AVAILABLE:
            logger.error("❌ Cannot run dashboard without Dash. Exiting.")
            return False

        logger.info("🏗️ Initializing Customer Communications Data Processor v2.0...")
        dp = CustomerCommunicationsDataProcessor(CONFIG, logger)

        # Feature 2.1: Intelligent data loading with intent-first approach
        if not dp.load_data() or (dp.mail_df.empty and dp.call_df.empty):
            logger.warning("⚠️ Real data loading failed, creating sample data...")
            if not dp.create_sample_data():
                logger.error("❌ Sample data creation failed. Exiting.")
                return False

        # Load financial data aligned with mail data
        if FINANCIAL_AVAILABLE:
            dp.load_enhanced_financial_data()

        # Feature 2.2: High-fidelity data filtering and merging
        if not dp.combine_and_clean_data():
            logger.error("❌ Data processing failed. Exiting.")
            return False

        # Run enhanced analysis
        logger.info("🔄 Running enhanced correlation analysis...")

        if SCIPY_AVAILABLE:
            success_momentum = dp.analyze_mail_momentum_correlation()
            success_rolling = dp.analyze_rolling_correlation()
            
            # Feature 3.1: Intent-level correlation analysis
            success_intent = dp.analyze_intent_correlation()

            if not success_momentum:
                logger.warning("⚠️ Momentum correlation analysis failed")
            if not success_rolling:
                logger.warning("⚠️ Rolling correlation analysis failed")
            if not success_intent:
                logger.warning("⚠️ Intent correlation analysis failed or skipped")
        else:
            logger.warning("⚠️ SciPy not available - correlation analysis disabled")

        dp.calculate_efficiency_metrics()

        # Initialize dashboard
        logger.info("🎨 Initializing enhanced dashboard v2.0...")
        dashboard = CustomerCommunicationsDashboard(dp)

        # Enhanced status report
        logger.info("=" * 70)
        logger.info("📊 CUSTOMER COMMUNICATIONS DASHBOARD v2.0 STATUS:")
        logger.info(f"📧 Mail records: {len(dp.mail_df):,} (all types, June 2024+)")
        logger.info(f"📞 Call records: {len(dp.call_df):,}")
        logger.info(f"🎯 Intent data available: {'✅ YES' if dp.has_intent_data else '❌ NO'}")
        logger.info(f"💰 Financial indicators: {len(dp.financial_df.columns)-1 if not dp.financial_df.empty else 0}")
        logger.info(f"🔗 Momentum correlations: {len(dp.momentum_correlation_results) if not dp.momentum_correlation_results.empty else 0}")
        logger.info(f"🔄 Rolling correlations: {len(dp.rolling_correlation_results) if not dp.rolling_correlation_results.empty else 0}")
        logger.info(f"📊 Intent correlations: {len(dp.intent_correlation_results) if not dp.intent_correlation_results.empty else 0}")
        logger.info(f"⚡ Correlation spikes: {len(dp.spike_data) if not dp.spike_data.empty else 0}")
        logger.info(f"📊 Correlation table rows: {len(dp.correlation_data_table) if not dp.correlation_data_table.empty else 0}")
        logger.info(f"⚡ Clean business day records: {len(dp.combined_df):,}")
        logger.info("=" * 70)

        # Feature implementation status
        logger.info("🎯 FEATURES IMPLEMENTED:")
        logger.info(f"   ✅ Feature 2.1: Intent-first data ingestion - {'Active' if dp.has_intent_data else 'Fallback mode'}")
        logger.info(f"   ✅ Feature 2.2: High-fidelity filtering - Inner joins, weekend/holiday removal")
        logger.info(f"   ✅ Feature 3.1: Intent correlation heatmap - {'Available' if dp.has_intent_data else 'Disabled (no intent data)'}")
        logger.info(f"   ✅ Feature 3.2: Spike correlation flagging - {len(dp.spike_data) if not dp.spike_data.empty else 0} spikes detected")
        logger.info(f"   ✅ Feature 4.1: Visual cleanup - Theme colors, donut charts, horizontal bars")

        # Key insights
        if not dp.momentum_correlation_results.empty:
            best_momentum = dp.momentum_correlation_results.loc[dp.momentum_correlation_results['correlation'].abs().idxmax()]
            logger.info(f"🎯 BEST MOMENTUM CORRELATION: {best_momentum['correlation']:.3f}")
            logger.info(f"    Type: {best_momentum['type']}, Window: {best_momentum['window']}d, Lag: {best_momentum['lag_days']}d")

            # Show top 3 correlations
            top_3 = dp.correlation_data_table.head(3)
            logger.info("📊 TOP 3 CORRELATIONS:")
            for i, (_, row) in enumerate(top_3.iterrows()):
                logger.info(f"    {i+1}. {row['type']} {row['window']}d lag {row['lag_days']}d: {row['correlation']:.3f}")

        if dp.has_intent_data and not dp.intent_correlation_results.empty:
            # Find best intent correlation
            intent_cols = [col for col in dp.intent_correlation_results.columns if col.startswith('lag_')]
            if intent_cols:
                best_intent_corr = 0
                best_intent_info = ""
                for _, row in dp.intent_correlation_results.iterrows():
                    for col in intent_cols:
                        if abs(row[col]) > abs(best_intent_corr):
                            best_intent_corr = row[col]
                            lag = col.replace('lag_', '')
                            best_intent_info = f"{row['intent']} at {lag}d lag"
                
                if best_intent_corr != 0:
                    logger.info(f"🎯 BEST INTENT CORRELATION: {best_intent_corr:.3f} ({best_intent_info})")

        if not dp.spike_data.empty:
            max_spike = dp.spike_data['rolling_correlation'].max()
            spike_date = dp.spike_data.loc[dp.spike_data['rolling_correlation'].idxmax(), 'date']
            logger.info(f"⚡ HIGHEST CORRELATION SPIKE: {max_spike:.3f} on {spike_date.date()}")

        if dp.efficiency_metrics:
            logger.info(f"📈 RESPONSE RATE: {dp.efficiency_metrics['response_rate_pct']:.2f}%")
            logger.info(f"📊 AVG MOMENTUM: {dp.efficiency_metrics['avg_momentum']:.2f}% change")

        logger.info("=" * 70)
        logger.info("🌐 Access dashboard at: http://127.0.0.1:8050")
        logger.info("🛑 Press Ctrl+C to stop")
        logger.info("=" * 70)

        dashboard.run(debug=CONFIG.get('DEBUG_MODE', True), port=8050)

        return True

    except KeyboardInterrupt:
        logger.info("🛑 Dashboard stopped by user")
        return True
    except Exception as e:
        logger.error(f"❌ An unexpected error occurred: {str(e)}")
        logger.error("💡 Try installing missing dependencies:")
        logger.error("    pip install plotly dash dash-bootstrap-components scipy yfinance holidays")
        return False

if __name__ == '__main__':
    # Pre-flight check
    print("🔍 PRE-FLIGHT CHECK:")
    print(f"    Plotly: {'✅' if PLOTLY_AVAILABLE else '❌'}")
    print(f"    Dash: {'✅' if DASH_AVAILABLE else '❌'}")
    print(f"    SciPy: {'✅' if SCIPY_AVAILABLE else '❌'}")
    print(f"    yfinance: {'✅' if FINANCIAL_AVAILABLE else '❌'}")
    print(f"    holidays: {'✅' if 'holidays' in sys.modules else '❌'}")
    print()

    if not DASH_AVAILABLE or not PLOTLY_AVAILABLE:
        print("❌ Missing required dependencies. Install with:")
        print("    pip install plotly dash dash-bootstrap-components")
        if not SCIPY_AVAILABLE:
            print("    pip install scipy")
        if not FINANCIAL_AVAILABLE:
            print("    pip install yfinance")
        print("    pip install holidays")
        sys.exit(1)

    if main():
        print("\n✅ Customer Communications Dashboard v2.0 session ended successfully.")
    else:
        print("\n⚠️ Customer Communications Dashboard v2.0 session ended with errors.")

    sys.exit(0)

# =============================================================================
# END OF CUSTOMER COMMUNICATIONS INTELLIGENCE DASHBOARD v2.0
# =============================================================================

"""
REQUIREMENTS IMPLEMENTATION SUMMARY:

✅ Feature 2.1: Intelligent, Intent-First Call Data Ingestion
   - Dual file path configuration (primary + fallback)
   - Automatic intent data detection and flagging
   - Global has_intent_data flag for conditional UI rendering
   - Comprehensive logging for file selection logic

✅ Feature 2.2: High-Fidelity Data Filtering & Merging  
   - Inner joins instead of outer joins for better correlation signal
   - Weekend filtering using weekday < 5
   - US federal holiday filtering using holidays library
   - Reduced noise for cleaner analysis

✅ Feature 3.1: Multi-Lag, Intent-Level Correlation Heatmap
   - Intent data pivoting for daily counts per intent
   - Pearson correlation across 0-21 day lags for each intent
   - Heatmap visualization with diverging Red-to-Blue colorscale
   - Y-axis: Call Intent Names, X-axis: Lag in Days
   - Only renders when has_intent_data = True

✅ Feature 3.2: Timeline with Programmatic Spike Correlation Flagging
   - 30-day rolling correlation calculation
   - 90th percentile spike threshold detection
   - spike_highlight column for visual markers
   - Yellow star overlays on main timeline chart

✅ Feature 4.1: Comprehensive Visual Cleanup
   - THEME_COLORS dictionary for consistent color usage
   - Call Intent Distribution: Donut chart (hole=0.5) with center total
   - Mail Communications Breakdown: Horizontal bar chart for readability
   - Enhanced typography, spacing, and professional styling
   - Consistent color themes across all visualizations

DASHBOARD FEATURES:
- Intelligent data loading with fallback mechanisms
- Enhanced correlation analysis with multiple methodologies
- Professional visualizations with consistent theming
- Conditional rendering based on data availability
- Comprehensive status reporting and logging
- Real-time spike detection and flagging
- Intent-level deep dive analysis capabilities

All requirements from the requirements document have been successfully implemented.
"""
