#!/usr/bin/env python3
"""
Executive Customer Communications Intelligence Dashboard - Clean & Readable Version

Focus on mail momentum correlation, rolling windows, enhanced financial data,
outlier removal, and significantly improved readability.
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
# ENHANCED CONFIGURATION FOR CUSTOMER COMMUNICATIONS
# =============================================================================

CONFIG = {
    'MAIL_FILE_PATH': r'merged_output.csv',
    'CALL_FILE_PATH': r'data\GenesysExtract_20250609.csv',
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
    
    'MAX_LAG_DAYS': 21,  # 3 weeks
    'DEBUG_MODE': True,
    
    # Customer account communication types (not campaigns)
    'RELEVANT_MAIL_TYPES': [
        'statement', 'billing', 'account_update', 'notification', 
        'reminder', 'confirmation', 'alert', 'notice'
    ],
    
    # Mail momentum analysis parameters
    'MOMENTUM_WINDOWS': [3, 7, 14],  # Days to calculate momentum
    'ROLLING_CORRELATION_WINDOW': 30,  # Days for rolling correlation
    'MIN_CORRELATION_DISPLAY': 0.15,  # Minimum correlation to show
    
    # Outlier removal parameters
    'OUTLIER_THRESHOLD': 3,  # Standard deviations for outlier detection
    'MIN_MAIL_VOLUME_PERCENTILE': 5,  # Remove bottom 5% mail volume days
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
# ENHANCED DATA PROCESSOR WITH MOMENTUM ANALYSIS
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
        
        # Analysis results
        self.momentum_correlation_results = pd.DataFrame()
        self.rolling_correlation_results = pd.DataFrame()
        self.significant_correlations = pd.DataFrame()
        self.mail_by_type = pd.DataFrame()
        self.calls_by_intent = pd.DataFrame()
        self.efficiency_metrics = {}
        self.alerts = []

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

    def load_data(self):
        """Load and clean customer communications data."""
        try:
            self.logger.info("🔄 Loading customer communications data...")
            
            # Load mail data
            self.mail_df = self.safe_load_csv(self.config['MAIL_FILE_PATH'], "Customer mail data")
            if not self.mail_df.empty:
                self.mail_df = self.safe_column_mapping(self.mail_df, self.config['MAIL_COLUMNS'], 'mail')
                self.mail_df = self.safe_date_conversion(self.mail_df, 'date')
                
                if 'volume' in self.mail_df.columns:
                    self.mail_df['volume'] = pd.to_numeric(self.mail_df['volume'], errors='coerce')
                    self.mail_df = self.mail_df.dropna(subset=['volume'])
                
                # Filter to relevant communication types
                if 'type' in self.mail_df.columns:
                    relevant_types = self.config['RELEVANT_MAIL_TYPES']
                    original_len = len(self.mail_df)
                    self.mail_df = self.mail_df[self.mail_df['type'].isin(relevant_types)]
                    self.logger.info(f"Filtered to account communications: {len(self.mail_df):,} records (was {original_len:,})")
            
            # Load call data
            self.call_df = self.safe_load_csv(self.config['CALL_FILE_PATH'], "Customer call data")
            if not self.call_df.empty:
                self.call_df = self.safe_column_mapping(self.call_df, self.config['CALL_COLUMNS'], 'call')
                self.call_df = self.safe_date_conversion(self.call_df, 'date')
            
            self.logger.info("✅ Data loading completed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Data loading failed: {str(e)}")
            return False

    def create_sample_data(self):
        """Create realistic sample data for customer communications."""
        try:
            self.logger.info("🔧 Creating sample customer communications data...")
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=90)
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            np.random.seed(42)
            
            # Create realistic call patterns (weekday-focused)
            base_calls = 45
            call_data = []
            
            for date in dates:
                if date.weekday() < 5:  # Weekdays only for stronger correlation
                    daily_calls = max(int(base_calls + np.random.normal(0, 8)), 0)
                    intents = ['billing_inquiry', 'account_update', 'statement_question', 'balance_check', 'general_support']
                    
                    for intent in intents:
                        intent_calls = max(int(daily_calls * np.random.uniform(0.1, 0.4)), 0)
                        call_data.extend([{'date': date, 'intent': intent}] * intent_calls)
            
            self.call_df = pd.DataFrame(call_data)
            
            # Create mail data with momentum patterns
            mail_data = []
            base_mail = 800
            momentum = 0
            
            for i, date in enumerate(dates):
                if date.weekday() < 5:  # Weekdays only
                    # Add momentum effect (acceleration/deceleration)
                    momentum_change = np.random.normal(0, 50)
                    momentum = momentum * 0.8 + momentum_change  # Decay momentum over time
                    
                    daily_mail = max(int(base_mail + momentum + np.random.normal(0, 100)), 50)
                    
                    mail_types = ['statement', 'billing', 'account_update', 'notification', 'reminder']
                    for mail_type in mail_types:
                        type_volume = max(int(daily_mail * np.random.uniform(0.15, 0.35)), 0)
                        if type_volume > 0:
                            mail_data.append({'date': date, 'type': mail_type, 'volume': type_volume})
            
            self.mail_df = pd.DataFrame(mail_data)
            
            self.logger.info("✅ Sample customer communications data created")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Sample data creation failed: {str(e)}")
            return False

    def load_enhanced_financial_data(self):
        """Load enhanced financial data with better indicators."""
        if not FINANCIAL_AVAILABLE:
            self.logger.warning("📊 Financial data unavailable - yfinance not installed")
            return False
        
        try:
            self.logger.info("🔄 Loading enhanced financial indicators...")
            
            if not self.call_df.empty:
                start_date = self.call_df['date'].min() - timedelta(days=5)
                end_date = self.call_df['date'].max() + timedelta(days=1)
            else:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=90)
            
            financial_data = {}
            for name, ticker_str in self.config['FINANCIAL_DATA'].items():
                try:
                    self.logger.info(f"Fetching {name}...")
                    ticker_obj = yf.Ticker(ticker_str)
                    data = ticker_obj.history(start=start_date, end=end_date)
                    
                    if not data.empty and 'Close' in data.columns:
                        close_prices = data['Close'].resample('D').last().ffill()
                        
                        # Calculate normalized percentage change from start
                        if len(close_prices) > 1:
                            first_price = close_prices.iloc[0]
                            pct_change = ((close_prices - first_price) / first_price * 100)
                            financial_data[name] = pct_change
                            
                except Exception as e:
                    self.logger.warning(f"Failed to fetch {name}: {str(e)}")
                    continue
            
            if financial_data:
                self.financial_df = pd.DataFrame(financial_data)
                self.financial_df.index.name = 'date'
                self.financial_df = self.financial_df.reset_index()
                self.financial_df['date'] = pd.to_datetime(self.financial_df['date']).dt.tz_localize(None).dt.normalize()
                
                # Remove weekends from financial data too
                self.financial_df = self.financial_df[self.financial_df['date'].dt.weekday < 5]
                
                self.logger.info(f"✅ Enhanced financial data loaded: {len(financial_data)} indicators")
                return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Financial data loading failed: {str(e)}")
            return False

    def combine_and_clean_data(self):
        """Combine data with outlier removal and weekend filtering."""
        try:
            self.logger.info("🔄 Combining and cleaning data...")
            
            # Aggregate call data (weekdays only)
            if not self.call_df.empty:
                # Remove weekends
                self.call_df = self.call_df[self.call_df['date'].dt.weekday < 5]
                self.calls_by_intent = self.call_df.groupby('intent').size().reset_index(name='count')
                daily_calls = self.call_df.groupby('date').size().reset_index(name='call_volume')
            else:
                # Create weekday-only date range
                end_date = datetime.now().date()
                start_date = end_date - timedelta(days=30)
                weekdays = pd.bdate_range(start=start_date, end=end_date)
                daily_calls = pd.DataFrame({'date': weekdays, 'call_volume': 0})

            # Aggregate mail data (weekdays only)
            if not self.mail_df.empty and 'volume' in self.mail_df.columns:
                # Remove weekends
                self.mail_df = self.mail_df[self.mail_df['date'].dt.weekday < 5]
                self.mail_by_type = self.mail_df.groupby('type')['volume'].sum().reset_index()
                daily_mail = self.mail_df.groupby('date')['volume'].sum().reset_index(name='mail_volume')
                combined = pd.merge(daily_calls, daily_mail, on='date', how='outer')
            else:
                combined = daily_calls.copy()
                combined['mail_volume'] = 0
            
            # Create full weekday range
            if not combined.empty:
                start_date = combined['date'].min()
                end_date = combined['date'].max()
                full_weekday_range = pd.bdate_range(start=start_date, end=end_date)
                combined = combined.set_index('date').reindex(full_weekday_range).reset_index()
                combined = combined.rename(columns={'index': 'date'})

            combined['call_volume'] = combined['call_volume'].fillna(0)
            combined['mail_volume'] = combined['mail_volume'].fillna(0)
            
            # CRITICAL: Remove outliers and trailing low-volume data
            if len(combined) > 10:
                # Remove bottom percentile of mail volume days (likely incomplete)
                mail_threshold = combined['mail_volume'].quantile(self.config['MIN_MAIL_VOLUME_PERCENTILE'] / 100)
                combined = combined[combined['mail_volume'] >= mail_threshold]
                
                # Remove statistical outliers (beyond 3 standard deviations)
                for col in ['call_volume', 'mail_volume']:
                    if combined[col].std() > 0:
                        mean_val = combined[col].mean()
                        std_val = combined[col].std()
                        threshold = self.config['OUTLIER_THRESHOLD']
                        combined = combined[
                            (combined[col] >= mean_val - threshold * std_val) &
                            (combined[col] <= mean_val + threshold * std_val)
                        ]
                
                self.logger.info(f"Cleaned data: {len(combined):,} weekday records after outlier removal")
            
            # Add financial data
            if not self.financial_df.empty:
                combined = pd.merge(combined, self.financial_df, on='date', how='left')
                financial_cols = [col for col in self.financial_df.columns if col != 'date']
                for col in financial_cols:
                    if col in combined.columns:
                        combined[col] = combined[col].ffill().bfill()
            
            # Calculate mail momentum (key feature!)
            for window in self.config['MOMENTUM_WINDOWS']:
                # Mail momentum = rate of change in mail volume
                combined[f'mail_momentum_{window}d'] = combined['mail_volume'].pct_change(periods=window) * 100
                
                # Mail acceleration = change in momentum
                combined[f'mail_acceleration_{window}d'] = combined[f'mail_momentum_{window}d'].diff()
            
            # Add other derived features
            combined['weekday'] = combined['date'].dt.day_name()
            combined['month'] = combined['date'].dt.month
            combined['day_of_week'] = combined['date'].dt.weekday
            
            # Normalization for visualization
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
            
            # Sort and finalize
            combined = combined.sort_values('date').reset_index(drop=True)
            self.combined_df = combined
            
            self.logger.info(f"✅ Enhanced dataset created: {len(combined):,} clean weekday records")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Data combination failed: {str(e)}")
            return False

    def analyze_mail_momentum_correlation(self):
        """Analyze correlation between mail momentum and call volume."""
        if not SCIPY_AVAILABLE or self.combined_df.empty:
            self.logger.warning("Momentum correlation analysis skipped")
            return False

        self.logger.info("🔄 Analyzing mail momentum correlation with call volume...")
        
        try:
            df = self.combined_df
            max_lag = self.config['MAX_LAG_DAYS']
            correlations = []
            
            # Test different momentum windows
            for momentum_window in self.config['MOMENTUM_WINDOWS']:
                momentum_col = f'mail_momentum_{momentum_window}d'
                acceleration_col = f'mail_acceleration_{momentum_window}d'
                
                if momentum_col in df.columns:
                    # Clean data for this analysis
                    clean_df = df[[momentum_col, acceleration_col, 'call_volume', 'mail_volume']].dropna()
                    
                    if len(clean_df) >= 20:  # Need sufficient data
                        # 1. Momentum correlation
                        for lag in range(max_lag + 1):
                            call_shifted = clean_df['call_volume'].shift(-lag)
                            momentum_current = clean_df[momentum_col]
                            
                            valid_data = pd.DataFrame({
                                'momentum': momentum_current, 
                                'calls': call_shifted
                            }).dropna()
                            
                            if len(valid_data) >= 10:
                                try:
                                    corr, p_val = pearsonr(valid_data['momentum'], valid_data['calls'])
                                    if np.isfinite(corr):
                                        correlations.append({
                                            'type': 'Momentum',
                                            'window': momentum_window,
                                            'lag_days': lag,
                                            'correlation': corr,
                                            'p_value': p_val,
                                            'significant': p_val < 0.05,
                                            'sample_size': len(valid_data)
                                        })
                                except:
                                    pass
                        
                        # 2. Acceleration correlation (if available)
                        if acceleration_col in df.columns:
                            acceleration_data = clean_df[acceleration_col].dropna()
                            if len(acceleration_data) >= 15:
                                for lag in range(0, max_lag + 1, 2):  # Every 2 days
                                    call_shifted = clean_df['call_volume'].shift(-lag)
                                    acceleration_current = clean_df[acceleration_col]
                                    
                                    valid_data = pd.DataFrame({
                                        'acceleration': acceleration_current, 
                                        'calls': call_shifted
                                    }).dropna()
                                    
                                    if len(valid_data) >= 10:
                                        try:
                                            corr, p_val = pearsonr(valid_data['acceleration'], valid_data['calls'])
                                            if np.isfinite(corr):
                                                correlations.append({
                                                    'type': 'Acceleration',
                                                    'window': momentum_window,
                                                    'lag_days': lag,
                                                    'correlation': corr,
                                                    'p_value': p_val,
                                                    'significant': p_val < 0.05,
                                                    'sample_size': len(valid_data)
                                                })
                                        except:
                                            pass
            
            if correlations:
                self.momentum_correlation_results = pd.DataFrame(correlations)
                
                # Extract significant correlations
                significant = self.momentum_correlation_results[
                    (self.momentum_correlation_results['significant'] == True) &
                    (self.momentum_correlation_results['correlation'].abs() >= self.config['MIN_CORRELATION_DISPLAY'])
                ]
                
                if not significant.empty:
                    self.significant_correlations = significant.copy()
                    self.logger.info(f"✅ Found {len(significant)} significant momentum correlations")
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Momentum correlation analysis failed: {e}")
            return False

    def analyze_rolling_correlation(self):
        """Analyze how correlation strength varies over time."""
        if not SCIPY_AVAILABLE or self.combined_df.empty:
            return False

        self.logger.info("🔄 Analyzing rolling correlation windows...")
        
        try:
            df = self.combined_df
            window_size = self.config['ROLLING_CORRELATION_WINDOW']
            
            if len(df) < window_size + 10:
                self.logger.warning("Insufficient data for rolling correlation")
                return False
            
            rolling_results = []
            
            # Test different lag periods
            test_lags = [0, 3, 7, 14]  # Key lag periods to test
            
            for lag in test_lags:
                correlations = []
                dates = []
                
                # Rolling window analysis
                for i in range(window_size, len(df) - lag):
                    window_data = df.iloc[i-window_size:i]
                    
                    if len(window_data) >= 10:
                        call_shifted = window_data['call_volume'].shift(-lag)
                        mail_current = window_data['mail_volume']
                        
                        valid_data = pd.DataFrame({
                            'mail': mail_current,
                            'calls': call_shifted
                        }).dropna()
                        
                        if len(valid_data) >= 8:
                            try:
                                corr, p_val = pearsonr(valid_data['mail'], valid_data['calls'])
                                if np.isfinite(corr):
                                    correlations.append(corr)
                                    dates.append(df.iloc[i]['date'])
                                else:
                                    correlations.append(0)
                                    dates.append(df.iloc[i]['date'])
                            except:
                                correlations.append(0)
                                dates.append(df.iloc[i]['date'])
                
                if correlations:
                    for date, corr in zip(dates, correlations):
                        rolling_results.append({
                            'date': date,
                            'lag_days': lag,
                            'correlation': corr,
                            'window_size': window_size
                        })
            
            if rolling_results:
                self.rolling_correlation_results = pd.DataFrame(rolling_results)
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
            momentum_cols = [col for col in df.columns if 'momentum' in col]
            if momentum_cols:
                avg_momentum = df[momentum_cols[0]].mean()
                momentum_volatility = df[momentum_cols[0]].std()
            else:
                avg_momentum = 0
                momentum_volatility = 0
            
            # Efficiency trends
            if len(df) >= 30:
                recent_15 = df.tail(15)['call_volume'].mean()
                previous_15 = df.iloc[-30:-15]['call_volume'].mean()
                call_trend = ((recent_15 - previous_15) / previous_15 * 100) if previous_15 > 0 else 0
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
                'call_trend_15d': round(call_trend, 1)
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Efficiency calculation failed: {e}")
            return False

# =============================================================================
# ENHANCED DASHBOARD WITH IMPROVED READABILITY
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
            self.app.title = "Customer Communications Intelligence Dashboard"
            self.setup_layout()
            self.setup_callbacks()
            
        except Exception as e:
            self.logger.error(f"Dashboard initialization failed: {str(e)}")

    def create_error_figure(self, message):
        """Create a clean error figure with larger text."""
        fig = go.Figure()
        fig.add_annotation(
            text=f"⚠️ {message}", 
            xref="paper", yref="paper",
            x=0.5, y=0.5, 
            showarrow=False, 
            font=dict(size=20, color='#666666')
        )
        fig.update_layout(
            title="", 
            template='plotly_white', 
            height=400,
            font=dict(size=14)
        )
        return fig

    def create_enhanced_overview_figure(self, df):
        """Create clean, readable overview with normalized financial data."""
        if df.empty:
            return self.create_error_figure("No data available for this period")

        # Create subplot with improved spacing
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('📈 Normalized Trends (Customer Communications & Market)', 
                          '📊 Raw Volumes with Clear Trends'),
            vertical_spacing=0.15,
            specs=[[{"secondary_y": True}], [{"secondary_y": True}]]
        )

        # Top plot: Normalized trends
        fig.add_trace(
            go.Scatter(
                x=df['date'], 
                y=df['call_volume_norm'], 
                name='Call Volume (Normalized)', 
                line=dict(color='#e74c3c', width=3),
                mode='lines'
            ),
            row=1, col=1, secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['date'], 
                y=df['mail_volume_norm'], 
                name='Mail Volume (Normalized)', 
                line=dict(color='#3498db', width=3),
                mode='lines'
            ),
            row=1, col=1, secondary_y=False
        )

        # Add prominent financial data if available
        financial_cols = [col for col in df.columns if any(fin in col for fin in ['S&P 500', '10-Year Treasury', 'VIX'])]
        colors = ['#27ae60', '#f39c12', '#9b59b6']
        
        for i, fin_col in enumerate(financial_cols[:3]):  # Max 3 for readability
            if fin_col in df.columns and df[fin_col].notna().any():
                # Normalize financial data properly
                fin_data = df[fin_col].dropna()
                if len(fin_data) > 1 and fin_data.std() > 0:
                    fin_norm = (fin_data - fin_data.mean()) / fin_data.std()
                    fig.add_trace(
                        go.Scatter(
                            x=df['date'], 
                            y=fin_norm, 
                            name=f'{fin_col.replace("_pct", "")}',
                            line=dict(color=colors[i], width=2, dash='dot'),
                            opacity=0.8
                        ),
                        row=1, col=1, secondary_y=True
                    )

        # Bottom plot: Raw volumes with clear visualization
        fig.add_trace(
            go.Bar(
                x=df['date'], 
                y=df['mail_volume'], 
                name='Daily Mail Volume', 
                marker_color='rgba(52, 152, 219, 0.4)',
                marker_line=dict(width=0)
            ),
            row=2, col=1, secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['date'], 
                y=df['call_volume'], 
                name='Daily Call Volume', 
                line=dict(color='#e74c3c', width=4),
                mode='lines+markers',
                marker=dict(size=6)
            ),
            row=2, col=1, secondary_y=True
        )

        # Enhanced layout with better readability
        fig.update_layout(
            title=dict(
                text="Executive Overview: Customer Communications & Market Context",
                font=dict(size=24, color='#2c3e50'),
                x=0.5
            ),
            template='plotly_white',
            height=700,
            font=dict(size=14),
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.02, 
                xanchor="center", 
                x=0.5,
                font=dict(size=12)
            ),
            margin=dict(t=100, b=50, l=80, r=80)
        )
        
        # Enhanced axis labels
        fig.update_yaxes(title_text="<b>Normalized Score</b>", row=1, col=1, secondary_y=False, title_font=dict(size=14))
        fig.update_yaxes(title_text="<b>Market Indicators</b>", row=1, col=1, secondary_y=True, title_font=dict(size=14))
        fig.update_yaxes(title_text="<b>Mail Volume</b>", row=2, col=1, secondary_y=False, title_font=dict(size=14))
        fig.update_yaxes(title_text="<b>Call Volume</b>", row=2, col=1, secondary_y=True, title_font=dict(size=14))
        fig.update_xaxes(title_text="<b>Date</b>", row=2, col=1, title_font=dict(size=14))
        
        return fig

    def create_momentum_correlation_figure(self, df_momentum):
        """Create momentum correlation visualization."""
        if df_momentum.empty:
            return self.create_error_figure("Momentum correlation data not available")

        # Filter for most significant correlations
        significant_momentum = df_momentum[
            (df_momentum['significant'] == True) & 
            (df_momentum['correlation'].abs() >= 0.15)
        ]
        
        if significant_momentum.empty:
            # Show all if no significant ones
            plot_data = df_momentum.nlargest(10, 'correlation')
        else:
            plot_data = significant_momentum
        
        # Create clean bar chart
        fig = go.Figure()
        
        # Color code by correlation strength
        colors = ['#e74c3c' if corr > 0.3 else '#f39c12' if corr > 0.2 else '#3498db' 
                 for corr in plot_data['correlation']]
        
        fig.add_trace(go.Bar(
            x=[f"{row['type']}<br>{row['window']}d<br>Lag: {row['lag_days']}" 
               for _, row in plot_data.iterrows()],
            y=plot_data['correlation'],
            marker_color=colors,
            text=[f"{corr:.3f}" for corr in plot_data['correlation']],
            textposition='outside',
            textfont=dict(size=12, color='black')
        ))
        
        # Add significance threshold
        fig.add_hline(
            y=0.2, 
            line_dash="dash", 
            line_color="#27ae60", 
            annotation_text="Strong Correlation Threshold",
            annotation_font=dict(size=12)
        )

        fig.update_layout(
            title=dict(
                text="📈 Mail Momentum vs Call Volume Correlation<br><sub>Significant relationships between mail acceleration and customer calls</sub>",
                font=dict(size=20, color='#2c3e50'),
                x=0.5
            ),
            xaxis_title="<b>Analysis Type & Parameters</b>",
            yaxis_title="<b>Correlation Coefficient</b>",
            template='plotly_white',
            height=500,
            font=dict(size=14),
            margin=dict(t=100, b=100, l=80, r=80)
        )
        
        return fig

    def create_rolling_correlation_figure(self, df_rolling):
        """Create rolling correlation time series."""
        if df_rolling.empty:
            return self.create_error_figure("Rolling correlation data not available")

        fig = go.Figure()
        
        # Plot different lag periods
        lag_colors = {'0': '#e74c3c', '3': '#3498db', '7': '#27ae60', '14': '#f39c12'}
        
        for lag in df_rolling['lag_days'].unique():
            lag_data = df_rolling[df_rolling['lag_days'] == lag]
            
            fig.add_trace(go.Scatter(
                x=lag_data['date'],
                y=lag_data['correlation'],
                mode='lines+markers',
                name=f'{int(lag)}-Day Lag',
                line=dict(width=3, color=lag_colors.get(str(int(lag)), '#95a5a6')),
                marker=dict(size=6),
                hovertemplate=f'<b>{int(lag)}-Day Lag</b><br>Date: %{{x}}<br>Correlation: %{{y:.3f}}<extra></extra>'
            ))
        
        # Add reference lines
        fig.add_hline(y=0.3, line_dash="dash", line_color="#27ae60", 
                     annotation_text="Strong Correlation", annotation_font=dict(size=10))
        fig.add_hline(y=0, line_dash="solid", line_color="#95a5a6", 
                     annotation_text="No Correlation", annotation_font=dict(size=10))

        fig.update_layout(
            title=dict(
                text="🔄 Rolling Correlation Analysis<br><sub>How correlation strength varies over time (30-day windows)</sub>",
                font=dict(size=20, color='#2c3e50'),
                x=0.5
            ),
            xaxis_title="<b>Date</b>",
            yaxis_title="<b>Correlation Coefficient</b>",
            template='plotly_white',
            height=500,
            font=dict(size=14),
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.02, 
                xanchor="center", 
                x=0.5
            ),
            margin=dict(t=100, b=60, l=80, r=80)
        )
        
        return fig

    def create_clean_mail_types_figure(self, df_mail):
        """Create clean account communications breakdown."""
        if df_mail.empty:
            return self.create_error_figure("Account communications data unavailable")

        # Calculate percentages and sort
        total_volume = df_mail['volume'].sum()
        df_mail = df_mail.copy()
        df_mail['percentage'] = (df_mail['volume'] / total_volume * 100).round(1)
        df_mail = df_mail.sort_values('volume', ascending=True)  # Horizontal bar chart

        # Create horizontal bar chart for better readability
        fig = go.Figure()
        
        # Clean color palette
        colors = ['#3498db', '#e74c3c', '#27ae60', '#f39c12', '#9b59b6', '#e67e22', '#1abc9c', '#34495e']
        
        fig.add_trace(go.Bar(
            y=df_mail['type'],
            x=df_mail['volume'],
            orientation='h',
            marker_color=colors[:len(df_mail)],
            text=[f"{v:,.0f}<br>({p}%)" for v, p in zip(df_mail['volume'], df_mail['percentage'])],
            textposition='auto',
            textfont=dict(size=12, color='white')
        ))

        fig.update_layout(
            title=dict(
                text="📧 Customer Account Communications Breakdown<br><sub>Volume distribution by communication type</sub>",
                font=dict(size=20, color='#2c3e50'),
                x=0.5
            ),
            xaxis_title="<b>Total Volume</b>",
            yaxis_title="<b>Communication Type</b>",
            template='plotly_white',
            height=500,
            font=dict(size=14),
            margin=dict(t=100, b=60, l=150, r=80)
        )
        
        return fig

    def create_clean_call_intents_figure(self, df_calls):
        """Create clean call intents visualization."""
        if df_calls.empty:
            return self.create_error_figure("Call intent data unavailable")

        # Calculate percentages
        total_calls = df_calls['count'].sum()
        df_calls = df_calls.copy()
        df_calls['percentage'] = (df_calls['count'] / total_calls * 100).round(1)

        # Create clean donut chart
        fig = go.Figure(data=[go.Pie(
            labels=df_calls['intent'],
            values=df_calls['count'],
            hole=0.5,
            textinfo='label+percent',
            textposition='outside',
            textfont=dict(size=12),
            marker=dict(
                colors=['#3498db', '#e74c3c', '#27ae60', '#f39c12', '#9b59b6'][:len(df_calls)],
                line=dict(color='white', width=2)
            )
        )])

        # Enhanced center annotation
        fig.add_annotation(
            text=f"<b>Total</b><br>{total_calls:,}<br><b>Calls</b>",
            x=0.5, y=0.5,
            font=dict(size=16, color='#2c3e50'),
            showarrow=False
        )

        fig.update_layout(
            title=dict(
                text="📞 Customer Call Intent Distribution<br><sub>Breakdown of call reasons and inquiries</sub>",
                font=dict(size=20, color='#2c3e50'),
                x=0.5
            ),
            template='plotly_white',
            height=500,
            font=dict(size=14),
            showlegend=True,
            legend=dict(
                orientation="v", 
                yanchor="middle", 
                y=0.5,
                xanchor="left",
                x=1.05
            ),
            margin=dict(t=100, b=60, l=80, r=150)
        )
        
        return fig

    def create_significant_correlations_figure(self, df_sig):
        """Create focused significant correlations visualization."""
        if df_sig.empty:
            return self.create_error_figure("No significant correlations found")

        # Sort by correlation strength
        df_sig = df_sig.copy().sort_values('correlation', ascending=True)
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        # Color code by correlation strength
        colors = ['#e74c3c' if abs(corr) > 0.4 else '#f39c12' if abs(corr) > 0.25 else '#3498db' 
                 for corr in df_sig['correlation']]
        
        fig.add_trace(go.Bar(
            y=[f"{row['type']} ({row['window']}d) - Lag {row['lag_days']}d" 
               for _, row in df_sig.iterrows()],
            x=df_sig['correlation'],
            orientation='h',
            marker_color=colors,
            text=[f"{corr:.3f}" for corr in df_sig['correlation']],
            textposition='auto',
            textfont=dict(size=11, color='white')
        ))

        fig.update_layout(
            title=dict(
                text="🎯 Significant Mail-to-Call Correlations<br><sub>Only statistically significant relationships shown</sub>",
                font=dict(size=20, color='#2c3e50'),
                x=0.5
            ),
            xaxis_title="<b>Correlation Coefficient</b>",
            yaxis_title="<b>Analysis Method</b>",
            template='plotly_white',
            height=400,
            font=dict(size=14),
            margin=dict(t=100, b=60, l=200, r=80)
        )
        
        # Add reference line at 0
        fig.add_vline(x=0, line_dash="solid", line_color="#95a5a6")
        
        return fig

    def setup_layout(self):
        """Setup clean, readable dashboard layout."""
        df = self.dp.combined_df
        if df.empty:
            self.app.layout = html.Div([
                html.H1("⚠️ No Data Available", style={'textAlign': 'center', 'color': '#e74c3c'}),
                html.P("Dashboard cannot render without data.", style={'textAlign': 'center', 'fontSize': '18px'})
            ])
            return

        start_date = df['date'].min().date()
        end_date = df['date'].max().date()

        # Enhanced styling
        header_style = {
            'textAlign': 'center',
            'color': '#2c3e50',
            'marginBottom': '10px',
            'fontSize': '32px',
            'fontWeight': 'bold'
        }
        
        subtitle_style = {
            'textAlign': 'center',
            'color': '#7f8c8d',
            'marginBottom': '30px',
            'fontSize': '16px'
        }

        self.app.layout = dbc.Container([
            # Enhanced Header
            dbc.Row([
                dbc.Col([
                    html.H1("📊 Customer Communications Intelligence", style=header_style),
                    html.P("Weekday-focused analysis for actionable business insights", style=subtitle_style)
                ], width=12)
            ]),
            
            # Clean date picker
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("📅 Analysis Period", className="mb-3", style={'color': '#2c3e50'}),
                            dcc.DatePickerRange(
                                id='date-picker-range',
                                min_date_allowed=start_date,
                                max_date_allowed=end_date,
                                start_date=start_date,
                                end_date=end_date,
                                display_format='YYYY-MM-DD',
                                style={'fontSize': '14px'}
                            )
                        ])
                    ], className="shadow-sm")
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("🎯 Key Insights", className="mb-3", style={'color': '#2c3e50'}),
                            html.Div(id='quick-insights')
                        ])
                    ], className="shadow-sm")
                ], width=6)
            ], className="mb-4"),

            # Enhanced KPI Cards
            dbc.Row(id='kpi-cards-row', className="mb-4"),

            # Main Tabs with better spacing
            dbc.Tabs([
                dbc.Tab(
                    label="📈 Executive Overview", 
                    tab_id="tab-overview",
                    label_style={'fontSize': '16px', 'fontWeight': 'bold'},
                    children=[
                        html.Div([
                            dcc.Graph(id='overview-chart', style={'height': '700px'})
                        ], style={'padding': '20px'})
                    ]
                ),
                
                dbc.Tab(
                    label="🔗 Correlation Intelligence", 
                    tab_id="tab-correlation",
                    label_style={'fontSize': '16px', 'fontWeight': 'bold'},
                    children=[
                        html.Div([
                            dbc.Row([
                                dbc.Col([
                                    dcc.Graph(id='momentum-correlation-chart', style={'height': '500px'})
                                ], width=6),
                                dbc.Col([
                                    dcc.Graph(id='rolling-correlation-chart', style={'height': '500px'})
                                ], width=6)
                            ], className="mb-4"),
                            dbc.Row([
                                dbc.Col([
                                    dcc.Graph(id='significant-correlations-chart', style={'height': '450px'})
                                ], width=12)
                            ])
                        ], style={'padding': '20px'})
                    ]
                ),
                
                dbc.Tab(
                    label="📊 Communications Breakdown", 
                    tab_id="tab-breakdowns",
                    label_style={'fontSize': '16px', 'fontWeight': 'bold'},
                    children=[
                        html.Div([
                            dbc.Row([
                                dbc.Col([
                                    dcc.Graph(id='mail-types-chart', style={'height': '500px'})
                                ], width=6),
                                dbc.Col([
                                    dcc.Graph(id='call-intents-chart', style={'height': '500px'})
                                ], width=6)
                            ])
                        ], style={'padding': '20px'})
                    ]
                )
            ], className="mb-4"),
            
            # Enhanced Footer
            dbc.Row([
                dbc.Col([
                    html.Hr(style={'border': '1px solid #ecf0f1'}),
                    html.P([
                        f"📊 Data: {len(self.dp.mail_df):,} mail records, {len(self.dp.call_df):,} call records (weekdays only) | ",
                        f"📅 Period: {start_date} to {end_date} | ",
                        f"🔄 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                    ], style={'textAlign': 'center', 'color': '#95a5a6', 'fontSize': '12px'})
                ], width=12)
            ])
        ], fluid=True, style={'backgroundColor': '#f8f9fa', 'minHeight': '100vh', 'padding': '20px'})

    def setup_callbacks(self):
        """Setup enhanced callbacks with better error handling."""
        
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

            # Enhanced KPI Cards with better styling
            card_style = {'textAlign': 'center', 'height': '120px', 'border': 'none'}
            
            kpi_cards = [
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H2(f"{total_calls:,}", style={'color': '#e74c3c', 'marginBottom': '5px'}),
                            html.P("Total Calls", style={'color': '#2c3e50', 'fontSize': '14px', 'marginBottom': '5px'}),
                            html.Small(f"📈 {avg_daily_calls:.0f}/day", style={'color': '#7f8c8d'})
                        ])
                    ], style=card_style, className="shadow-sm border-left-danger")
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H2(f"{total_mail:,}", style={'color': '#3498db', 'marginBottom': '5px'}),
                            html.P("Total Communications", style={'color': '#2c3e50', 'fontSize': '14px', 'marginBottom': '5px'}),
                            html.Small(f"📧 {avg_daily_mail:.0f}/day", style={'color': '#7f8c8d'})
                        ])
                    ], style=card_style, className="shadow-sm border-left-info")
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H2(f"{response_rate:.1f}%", style={'color': '#27ae60', 'marginBottom': '5px'}),
                            html.P("Response Rate", style={'color': '#2c3e50', 'fontSize': '14px', 'marginBottom': '5px'}),
                            html.Small("📞 Calls per 100 mails", style={'color': '#7f8c8d'})
                        ])
                    ], style=card_style, className="shadow-sm border-left-success")
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H2(f"{dff['calls_per_1k_mails'].mean():.0f}" if 'calls_per_1k_mails' in dff.columns else "N/A", 
                                   style={'color': '#f39c12', 'marginBottom': '5px'}),
                            html.P("Efficiency Ratio", style={'color': '#2c3e50', 'fontSize': '14px', 'marginBottom': '5px'}),
                            html.Small("📊 Calls per 1K mails", style={'color': '#7f8c8d'})
                        ])
                    ], style=card_style, className="shadow-sm border-left-warning")
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
                        f"🎯 Best momentum correlation: {best_momentum['correlation']:.3f} ({best_momentum['type']}, {best_momentum['window']}d)",
                        color="primary", className="me-2 mb-2", style={'fontSize': '12px'}
                    )
                )
            
            trend_badge_color = "success" if avg_daily_calls > dff['call_volume'].iloc[:len(dff)//2].mean() else "danger"
            trend_text = "📈 Growing" if avg_daily_calls > dff['call_volume'].iloc[:len(dff)//2].mean() else "📉 Declining"
            
            insights.append(
                dbc.Badge(
                    f"Trend: {trend_text}",
                    color=trend_badge_color, className="me-2 mb-2", style={'fontSize': '12px'}
                )
            )
            
            overview_fig = self.create_enhanced_overview_figure(dff)
            
            return kpi_cards, overview_fig, insights

        @self.app.callback(
            [Output('momentum-correlation-chart', 'figure'),
             Output('rolling-correlation-chart', 'figure'),
             Output('significant-correlations-chart', 'figure')],
            [Input('date-picker-range', 'id')]  # Dummy input
        )
        def update_correlation_tab(_):
            momentum_fig = self.create_momentum_correlation_figure(self.dp.momentum_correlation_results)
            rolling_fig = self.create_rolling_correlation_figure(self.dp.rolling_correlation_results)
            significant_fig = self.create_significant_correlations_figure(self.dp.significant_correlations)
            
            return momentum_fig, rolling_fig, significant_fig

        @self.app.callback(
            [Output('mail-types-chart', 'figure'),
             Output('call-intents-chart', 'figure')],
            [Input('date-picker-range', 'id')]  # Dummy input
        )
        def update_breakdown_tab(_):
            mail_fig = self.create_clean_mail_types_figure(self.dp.mail_by_type)
            call_fig = self.create_clean_call_intents_figure(self.dp.calls_by_intent)
            return mail_fig, call_fig

    def run(self, debug=True, port=8050):
        """Run the enhanced dashboard."""
        try:
            self.logger.info(f"🚀 Starting Customer Communications Dashboard at http://127.0.0.1:{port}")
            self.app.run(debug=debug, port=port, host='127.0.0.1')
        except Exception as e:
            self.logger.error(f"Dashboard failed to start: {str(e)}")

# =============================================================================
# ENHANCED MAIN FUNCTION
# =============================================================================

def main():
    """Enhanced main function with comprehensive processing."""
    print("🔥 CUSTOMER COMMUNICATIONS INTELLIGENCE DASHBOARD")
    print("=" * 60)
    
    logger = setup_logging()

    try:
        if not DASH_AVAILABLE:
            logger.error("❌ Cannot run dashboard without Dash. Exiting.")
            return False
            
        logger.info("🏗️ Initializing Customer Communications Data Processor...")
        dp = CustomerCommunicationsDataProcessor(CONFIG, logger)
        
        # Load data with fallback
        if not dp.load_data() or (dp.mail_df.empty and dp.call_df.empty):
            logger.warning("⚠️ Real data loading failed, creating sample data...")
            if not dp.create_sample_data():
                logger.error("❌ Sample data creation failed. Exiting.")
                return False
        
        # Load financial data
        if FINANCIAL_AVAILABLE:
            dp.load_enhanced_financial_data()
        
        # Combine and clean data
        if not dp.combine_and_clean_data():
            logger.error("❌ Data processing failed. Exiting.")
            return False
        
        # Run enhanced analysis
        logger.info("🔄 Running enhanced correlation analysis...")
        
        if SCIPY_AVAILABLE:
            dp.analyze_mail_momentum_correlation()
            dp.analyze_rolling_correlation()
        else:
            logger.warning("⚠️ SciPy not available - correlation analysis disabled")
        
        dp.calculate_efficiency_metrics()
        
        # Initialize dashboard
        logger.info("🎨 Initializing enhanced dashboard...")
        dashboard = CustomerCommunicationsDashboard(dp)
        
        # Status report
        logger.info("=" * 50)
        logger.info("📊 CUSTOMER COMMUNICATIONS DASHBOARD STATUS:")
        logger.info(f"📧 Mail records: {len(dp.mail_df):,}")
        logger.info(f"📞 Call records: {len(dp.call_df):,}")
        logger.info(f"💰 Financial indicators: {len(dp.financial_df.columns)-1 if not dp.financial_df.empty else 0}")
        logger.info(f"🔗 Momentum correlations: {'✅ Complete' if not dp.momentum_correlation_results.empty else '❌ Failed'}")
        logger.info(f"🔄 Rolling correlations: {'✅ Complete' if not dp.rolling_correlation_results.empty else '❌ Failed'}")
        logger.info(f"🎯 Significant correlations: {len(dp.significant_correlations) if not dp.significant_correlations.empty else 0}")
        logger.info(f"⚡ Clean weekday records: {len(dp.combined_df):,}")
        logger.info("=" * 50)
        
        # Key insights
        if not dp.momentum_correlation_results.empty:
            best_momentum = dp.momentum_correlation_results.loc[dp.momentum_correlation_results['correlation'].abs().idxmax()]
            logger.info(f"🎯 BEST MOMENTUM CORRELATION: {best_momentum['correlation']:.3f}")
            logger.info(f"   Type: {best_momentum['type']}, Window: {best_momentum['window']}d, Lag: {best_momentum['lag_days']}d")
        
        if dp.efficiency_metrics:
            logger.info(f"📈 RESPONSE RATE: {dp.efficiency_metrics['response_rate_pct']:.2f}%")
            logger.info(f"📊 MOMENTUM: {dp.efficiency_metrics['avg_momentum']:.2f}% avg change")
        
        logger.info("🌐 Access dashboard at: http://127.0.0.1:8050")
        logger.info("🛑 Press Ctrl+C to stop")
        
        dashboard.run(debug=CONFIG.get('DEBUG_MODE', True), port=8050)
        
        return True
        
    except KeyboardInterrupt:
        logger.info("🛑 Dashboard stopped by user")
        return True
    except Exception as e:
        logger.error(f"❌ An unexpected error occurred: {str(e)}")
        logger.error("💡 Try installing missing dependencies:")
        logger.error("   pip install plotly dash dash-bootstrap-components scipy yfinance")
        return False

if __name__ == '__main__':
    # Pre-flight check
    print("🔍 PRE-FLIGHT CHECK:")
    print(f"   Plotly: {'✅' if PLOTLY_AVAILABLE else '❌'}")
    print(f"   Dash: {'✅' if DASH_AVAILABLE else '❌'}")
    print(f"   SciPy: {'✅' if SCIPY_AVAILABLE else '❌'}")
    print(f"   yfinance: {'✅' if FINANCIAL_AVAILABLE else '❌'}")
    print()
    
    if not DASH_AVAILABLE or not PLOTLY_AVAILABLE:
        print("❌ Missing required dependencies. Install with:")
        print("   pip install plotly dash dash-bootstrap-components")
        if not SCIPY_AVAILABLE:
            print("   pip install scipy")
        if not FINANCIAL_AVAILABLE:
            print("   pip install yfinance")
        sys.exit(1)
    
    if main():
        print("\n✅ Customer Communications Dashboard session ended successfully.")
    else:
        print("\n⚠️ Customer Communications Dashboard session ended with errors.")
    
    sys.exit(0)
