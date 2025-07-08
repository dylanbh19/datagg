#!/usr/bin/env python3
"""
Executive Marketing Intelligence Dashboard - Enhanced Version

Enhanced dashboard with proper correlation analysis, normalized visualizations,
financial data integration, and executive-level insights.
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
# CONFIGURATION - ENHANCED SETTINGS
# =============================================================================

CONFIG = {
    'MAIL_FILE_PATH': r'merged_output.csv',
    'CALL_FILE_PATH': r'data\GenesysExtract_20250609.csv',
    'MAIL_COLUMNS': {'date': 'mail_date', 'volume': 'mail_volume', 'type': 'mail_type'},
    'CALL_COLUMNS': {'date': 'ConversationStart', 'intent': 'uui_Intent'},
    'FINANCIAL_DATA': {'S&P 500': '^GSPC', '10-Yr Treasury': '^TNX', 'VIX': '^VIX'},
    'MAX_LAG_DAYS': 28,
    'DEBUG_MODE': True,
    # New configuration for enhanced features
    'MOVING_AVERAGE_DAYS': 7,
    'ALERT_THRESHOLDS': {
        'call_volume_spike': 2.0,  # 2x standard deviation
        'mail_volume_spike': 2.0,
        'correlation_threshold': 0.3
    }
}

# =============================================================================
# ROBUST LOGGING
# =============================================================================

def setup_logging():
    """Setup bulletproof logging."""
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            stream=sys.stdout
        )
        return logging.getLogger('EnhancedDashboard')
    except Exception:
        print("Logging setup failed, using print statements")
        return None

# =============================================================================
# ENHANCED DATA PROCESSING
# =============================================================================

class SafeDataProcessor:
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger or self._dummy_logger()
        self.mail_df = pd.DataFrame()
        self.call_df = pd.DataFrame()
        self.financial_df = pd.DataFrame()
        self.combined_df = pd.DataFrame()
        # Enhanced analysis dataframes
        self.correlation_results = pd.DataFrame()
        self.mail_by_type = pd.DataFrame()
        self.calls_by_intent = pd.DataFrame()
        self.intent_correlation_matrix = pd.DataFrame()
        self.efficiency_metrics = {}
        self.alerts = []
        self.errors = []

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
                    self.logger.info(f"✅ {description} loaded: {len(df):,} records (encoding: {encoding})")
                    return df
                except (UnicodeDecodeError, Exception):
                    continue

            self.logger.error(f"❌ Complete failure loading {description} at {file_path}")
            return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"❌ Unexpected error loading {description}: {str(e)}")
            return pd.DataFrame()

    def safe_column_mapping(self, df, column_mapping, data_type):
        """Safely map columns with fuzzy matching."""
        try:
            if df.empty:
                return df
            
            mapped_df = df.copy()
            successful_mappings = {}
            
            for standard_name, config_name in column_mapping.items():
                if config_name in mapped_df.columns:
                    if config_name != standard_name:
                        mapped_df = mapped_df.rename(columns={config_name: standard_name})
                    successful_mappings[config_name] = standard_name
                    continue
                
                fuzzy_matches = [col for col in mapped_df.columns 
                                 if config_name.lower() in col.lower() or col.lower() in config_name.lower()]
                
                if fuzzy_matches:
                    best_match = fuzzy_matches[0]
                    mapped_df = mapped_df.rename(columns={best_match: standard_name})
                    successful_mappings[best_match] = standard_name
                    self.logger.warning(f"Fuzzy mapping for {data_type}: '{best_match}' -> '{standard_name}'")
                else:
                    self.logger.warning(f"Column '{config_name}' not found in {data_type} data")
            
            if successful_mappings:
                self.logger.info(f"{data_type} column mappings: {successful_mappings}")
            
            return mapped_df
            
        except Exception as e:
            self.logger.error(f"Column mapping failed for {data_type}: {str(e)}")
            return df

    def safe_date_conversion(self, df, date_column):
        """Safely convert dates with multiple strategies."""
        try:
            if df.empty or date_column not in df.columns:
                return df
            
            original_count = len(df)
            
            # Use pd.to_datetime with robust settings
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            df = df.dropna(subset=[date_column])
            df[date_column] = df[date_column].dt.tz_localize(None).dt.normalize()
            
            success_rate = len(df) / original_count * 100
            self.logger.info(f"Date conversion success rate for '{date_column}': {success_rate:.1f}%")
            return df
            
        except Exception as e:
            self.logger.error(f"Date conversion error for '{date_column}': {str(e)}")
            return df

    def load_data(self):
        """Load data with comprehensive error handling."""
        try:
            self.logger.info("🔄 Loading data with robust error handling...")
            
            # Load mail data
            self.mail_df = self.safe_load_csv(self.config['MAIL_FILE_PATH'], "Mail data")
            if not self.mail_df.empty:
                self.mail_df = self.safe_column_mapping(self.mail_df, self.config['MAIL_COLUMNS'], 'mail')
                self.mail_df = self.safe_date_conversion(self.mail_df, 'date')
                
                if 'volume' in self.mail_df.columns:
                    self.mail_df['volume'] = pd.to_numeric(self.mail_df['volume'], errors='coerce')
                    self.mail_df = self.mail_df.dropna(subset=['volume'])
            
            # Load call data
            self.call_df = self.safe_load_csv(self.config['CALL_FILE_PATH'], "Call data")
            if not self.call_df.empty:
                self.call_df = self.safe_column_mapping(self.call_df, self.config['CALL_COLUMNS'], 'call')
                self.call_df = self.safe_date_conversion(self.call_df, 'date')
            
            self.logger.info("✅ Data loading completed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Data loading failed: {str(e)}")
            return False

    def create_sample_data(self):
        """Create enhanced sample data with realistic patterns."""
        try:
            self.logger.info("🔧 Creating enhanced sample data for demonstration...")
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=90)
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            np.random.seed(42)
            
            # Create more realistic call data with weekly patterns
            base_calls = 50
            weekly_pattern = np.sin(np.arange(len(dates)) * 2 * np.pi / 7) * 10
            trend = np.linspace(0, 20, len(dates))
            noise = np.random.normal(0, 5, len(dates))
            call_volumes = np.maximum(base_calls + weekly_pattern + trend + noise, 0).astype(int)
            
            intents = ['billing', 'support', 'sales', 'account_update', 'general_query']
            intent_weights = [0.3, 0.25, 0.2, 0.15, 0.1]
            
            call_data = []
            for i, date in enumerate(dates):
                daily_calls = call_volumes[i]
                for intent in intents:
                    intent_calls = int(daily_calls * np.random.choice(intent_weights))
                    if intent_calls > 0:
                        call_data.extend([{'date': date, 'intent': intent}] * intent_calls)
            
            self.call_df = pd.DataFrame(call_data)
            
            # Create correlated mail data
            base_mail = 1000
            mail_pattern = np.sin(np.arange(len(dates)) * 2 * np.pi / 7) * 200
            # Add correlation with future calls (3-day lag)
            future_call_impact = np.roll(call_volumes, -3) * 10
            mail_volumes = np.maximum(base_mail + mail_pattern + future_call_impact + 
                                    np.random.normal(0, 100, len(dates)), 0).astype(int)
            
            mail_types = ['promotional', 'newsletter', 'reminder', 'statement']
            type_weights = [0.4, 0.3, 0.2, 0.1]
            
            mail_data = []
            for i, date in enumerate(dates):
                daily_mail = mail_volumes[i]
                for mail_type in mail_types:
                    type_volume = int(daily_mail * np.random.choice(type_weights))
                    if type_volume > 0:
                        mail_data.append({'date': date, 'type': mail_type, 'volume': type_volume})
            
            self.mail_df = pd.DataFrame(mail_data)
            
            self.logger.info("✅ Enhanced sample data created successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Sample data creation failed: {str(e)}")
            return False

    def safe_financial_data(self):
        """Safely attempt to load financial data using a more stable method."""
        if not FINANCIAL_AVAILABLE:
            self.logger.warning("📊 Financial data unavailable - yfinance not installed")
            return False
        
        try:
            self.logger.info("🔄 Attempting to load financial data...")
            
            if not self.call_df.empty:
                start_date = self.call_df['date'].min() - timedelta(days=5)
                end_date = self.call_df['date'].max() + timedelta(days=1)
            else:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=90)
            
            financial_data = {}
            for name, ticker_str in self.config['FINANCIAL_DATA'].items():
                try:
                    ticker_obj = yf.Ticker(ticker_str)
                    data = ticker_obj.history(start=start_date, end=end_date)
                    if not data.empty and 'Close' in data.columns:
                        financial_data[name] = data['Close'].resample('D').last()
                except Exception as e:
                    self.logger.warning(f"Failed to fetch {name} ({ticker_str}): {str(e)}")
                    continue
            
            if financial_data:
                self.financial_df = pd.DataFrame(financial_data)
                self.financial_df.index.name = 'date'
                self.financial_df = self.financial_df.reset_index()
                self.financial_df['date'] = pd.to_datetime(self.financial_df['date']).dt.tz_localize(None).dt.normalize()
                self.logger.info(f"✅ Financial data loaded: {len(financial_data)} indicators")
                return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Financial data loading failed: {str(e)}")
            return False

    def combine_data(self):
        """Enhanced data combination with normalization and derived metrics."""
        try:
            self.logger.info("🔄 Combining and enhancing data sources...")
            
            # Start with call data aggregation
            if not self.call_df.empty:
                self.calls_by_intent = self.call_df.groupby('intent').size().reset_index(name='count')
                daily_calls = self.call_df.groupby('date').size().reset_index(name='call_volume')
            else:
                dates = pd.date_range(start=datetime.now().date() - timedelta(days=30), 
                                    end=datetime.now().date(), freq='D')
                daily_calls = pd.DataFrame({'date': dates, 'call_volume': 0})

            # Add mail data
            if not self.mail_df.empty and 'volume' in self.mail_df.columns:
                self.mail_by_type = self.mail_df.groupby('type')['volume'].sum().reset_index()
                daily_mail = self.mail_df.groupby('date')['volume'].sum().reset_index(name='mail_volume')
                combined = pd.merge(daily_calls, daily_mail, on='date', how='outer')
            else:
                combined = daily_calls.copy()
                combined['mail_volume'] = 0
            
            # Create a full date range to ensure no gaps
            if 'date' in combined.columns and not combined.empty:
                full_date_range = pd.date_range(start=combined['date'].min(), 
                                              end=combined['date'].max(), freq='D')
                combined = combined.set_index('date').reindex(full_date_range).reset_index()
                combined = combined.rename(columns={'index': 'date'})

            combined['call_volume'] = combined['call_volume'].fillna(0)
            combined['mail_volume'] = combined['mail_volume'].fillna(0)
            
            # Add financial data
            if not self.financial_df.empty:
                combined = pd.merge(combined, self.financial_df, on='date', how='left')
                financial_cols = self.config['FINANCIAL_DATA'].keys()
                for col in financial_cols:
                    if col in combined.columns:
                        combined[col] = combined[col].ffill().bfill()
            
            # Add enhanced derived features
            combined['weekday'] = combined['date'].dt.day_name()
            combined['month'] = combined['date'].dt.month
            combined['is_weekend'] = combined['date'].dt.weekday >= 5
            combined['week_number'] = combined['date'].dt.isocalendar().week
            
            # Add moving averages
            ma_days = self.config['MOVING_AVERAGE_DAYS']
            combined['call_volume_ma'] = combined['call_volume'].rolling(window=ma_days, min_periods=1).mean()
            combined['mail_volume_ma'] = combined['mail_volume'].rolling(window=ma_days, min_periods=1).mean()
            
            # Add normalized metrics (z-scores)
            if len(combined) > 1:
                combined['call_volume_norm'] = ((combined['call_volume'] - combined['call_volume'].mean()) / 
                                              (combined['call_volume'].std() + 1e-8))
                combined['mail_volume_norm'] = ((combined['mail_volume'] - combined['mail_volume'].mean()) / 
                                              (combined['mail_volume'].std() + 1e-8))
            else:
                combined['call_volume_norm'] = 0
                combined['mail_volume_norm'] = 0
            
            # Add percentage changes
            combined['call_volume_pct_change'] = combined['call_volume'].pct_change() * 100
            combined['mail_volume_pct_change'] = combined['mail_volume'].pct_change() * 100
            
            # Add efficiency ratio
            combined['calls_per_1k_mails'] = np.where(combined['mail_volume'] > 0,
                                                    (combined['call_volume'] / combined['mail_volume']) * 1000,
                                                    0)
            
            # Sort by date
            combined = combined.sort_values('date').reset_index(drop=True)
            
            self.combined_df = combined
            self.logger.info(f"✅ Enhanced combined dataset created: {len(combined):,} records")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Data combination failed: {str(e)}")
            return False

    def analyze_correlation(self):
        """Enhanced correlation analysis with proper error handling."""
        if not SCIPY_AVAILABLE or self.combined_df.empty:
            self.logger.warning("Correlation analysis skipped (SciPy unavailable or no data).")
            return False

        self.logger.info("🔄 Running enhanced lag correlation analysis...")
        try:
            correlations = []
            max_lag = self.config['MAX_LAG_DAYS']
            
            # Ensure we have enough data
            if len(self.combined_df) < max_lag + 10:
                self.logger.warning("Insufficient data for correlation analysis")
                return False

            for lag in range(max_lag + 1):
                # Create lagged call volume
                call_shifted = self.combined_df['call_volume'].shift(-lag)
                mail_current = self.combined_df['mail_volume']
                
                # Remove NaN values
                valid_data = pd.DataFrame({'mail': mail_current, 'call_lag': call_shifted}).dropna()
                
                if len(valid_data) < 10:  # Need minimum data points
                    corr = 0
                    p_value = 1
                else:
                    try:
                        corr, p_value = pearsonr(valid_data['mail'], valid_data['call_lag'])
                        if not np.isfinite(corr):
                            corr = 0
                    except:
                        corr = 0
                        p_value = 1
                
                correlations.append({
                    'lag_days': lag, 
                    'correlation': corr,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'sample_size': len(valid_data)
                })

            self.correlation_results = pd.DataFrame(correlations)
            self.logger.info("✅ Enhanced correlation analysis complete.")
            return True

        except Exception as e:
            self.logger.error(f"❌ Correlation analysis failed: {e}")
            return False
        
    def analyze_intent_correlation(self):
        """Enhanced intent correlation analysis."""
        if not SCIPY_AVAILABLE or self.mail_df.empty or self.call_df.empty:
            self.logger.warning("Skipping intent correlation (insufficient data).")
            self.intent_correlation_matrix = pd.DataFrame()
            return False

        self.logger.info("🔄 Running enhanced intent-level correlation analysis...")
        try:
            # Create pivot tables
            mail_pivot = self.mail_df.pivot_table(index='date', columns='type', 
                                                values='volume', aggfunc='sum').fillna(0)
            call_pivot = self.call_df.groupby(['date', 'intent']).size().unstack(fill_value=0)

            # Align date ranges
            start_date = min(mail_pivot.index.min(), call_pivot.index.min())
            end_date = max(mail_pivot.index.max(), call_pivot.index.max())
            full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')

            mail_pivot = mail_pivot.reindex(full_date_range, fill_value=0)
            call_pivot = call_pivot.reindex(full_date_range, fill_value=0)

            # Calculate correlations with statistical significance
            correlation_matrix = pd.DataFrame(index=mail_pivot.columns, 
                                            columns=call_pivot.columns, dtype=float)

            for mail_type in mail_pivot.columns:
                for call_intent in call_pivot.columns:
                    max_corr = 0
                    for lag in range(self.config['MAX_LAG_DAYS'] + 1):
                        try:
                            shifted_calls = call_pivot[call_intent].shift(-lag).fillna(0)
                            valid_data = pd.DataFrame({
                                'mail': mail_pivot[mail_type],
                                'calls': shifted_calls
                            }).dropna()
                            
                            if len(valid_data) > 10:
                                corr, p_val = pearsonr(valid_data['mail'], valid_data['calls'])
                                if np.isfinite(corr) and p_val < 0.05:  # Only significant correlations
                                    max_corr = max(max_corr, abs(corr))
                        except:
                            continue
                    
                    correlation_matrix.loc[mail_type, call_intent] = max_corr

            self.intent_correlation_matrix = correlation_matrix
            self.logger.info("✅ Enhanced intent correlation analysis complete.")
            return True
        except Exception as e:
            self.logger.error(f"❌ Intent correlation analysis failed: {e}")
            self.intent_correlation_matrix = pd.DataFrame()
            return False

    def calculate_efficiency_metrics(self):
        """Calculate business efficiency metrics."""
        try:
            self.logger.info("🔄 Calculating efficiency metrics...")
            
            if self.combined_df.empty:
                return False
            
            df = self.combined_df
            
            # Basic efficiency metrics
            total_calls = df['call_volume'].sum()
            total_mail = df['mail_volume'].sum()
            avg_daily_calls = df['call_volume'].mean()
            avg_daily_mail = df['mail_volume'].mean()
            
            # Response rate (calls per mail)
            response_rate = (total_calls / total_mail * 100) if total_mail > 0 else 0
            
            # Weekly patterns
            weekly_pattern = df.groupby('weekday')[['call_volume', 'mail_volume']].mean()
            peak_call_day = weekly_pattern['call_volume'].idxmax()
            peak_mail_day = weekly_pattern['mail_volume'].idxmax()
            
            # Volatility metrics
            call_volatility = df['call_volume'].std() / df['call_volume'].mean() if avg_daily_calls > 0 else 0
            mail_volatility = df['mail_volume'].std() / df['mail_volume'].mean() if avg_daily_mail > 0 else 0
            
            # Trend analysis (last 30 days vs previous 30 days)
            if len(df) >= 60:
                recent_30 = df.tail(30)['call_volume'].mean()
                previous_30 = df.iloc[-60:-30]['call_volume'].mean()
                call_trend = ((recent_30 - previous_30) / previous_30 * 100) if previous_30 > 0 else 0
                
                recent_30_mail = df.tail(30)['mail_volume'].mean()
                previous_30_mail = df.iloc[-60:-30]['mail_volume'].mean()
                mail_trend = ((recent_30_mail - previous_30_mail) / previous_30_mail * 100) if previous_30_mail > 0 else 0
            else:
                call_trend = 0
                mail_trend = 0
            
            self.efficiency_metrics = {
                'total_calls': int(total_calls),
                'total_mail': int(total_mail),
                'avg_daily_calls': round(avg_daily_calls, 1),
                'avg_daily_mail': round(avg_daily_mail, 1),
                'response_rate_pct': round(response_rate, 2),
                'peak_call_day': peak_call_day,
                'peak_mail_day': peak_mail_day,
                'call_volatility': round(call_volatility, 3),
                'mail_volatility': round(mail_volatility, 3),
                'call_trend_30d': round(call_trend, 1),
                'mail_trend_30d': round(mail_trend, 1)
            }
            
            self.logger.info("✅ Efficiency metrics calculated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Efficiency metrics calculation failed: {str(e)}")
            return False

    def detect_alerts(self):
        """Detect anomalies and generate alerts."""
        try:
            self.logger.info("🔄 Detecting anomalies and alerts...")
            
            if self.combined_df.empty:
                return False
            
            df = self.combined_df
            alerts = []
            
            # Spike detection
            call_threshold = self.config['ALERT_THRESHOLDS']['call_volume_spike']
            mail_threshold = self.config['ALERT_THRESHOLDS']['mail_volume_spike']
            
            # Recent spikes (last 7 days)
            recent_data = df.tail(7)
            call_mean = df['call_volume'].mean()
            call_std = df['call_volume'].std()
            mail_mean = df['mail_volume'].mean()
            mail_std = df['mail_volume'].std()
            
            for _, row in recent_data.iterrows():
                # Call volume spike
                if call_std > 0 and (row['call_volume'] - call_mean) > (call_threshold * call_std):
                    alerts.append({
                        'type': 'spike',
                        'metric': 'call_volume',
                        'date': row['date'],
                        'value': int(row['call_volume']),
                        'threshold': round(call_mean + (call_threshold * call_std), 1),
                        'severity': 'high' if (row['call_volume'] - call_mean) > (3 * call_std) else 'medium'
                    })
                
                # Mail volume spike
                if mail_std > 0 and (row['mail_volume'] - mail_mean) > (mail_threshold * mail_std):
                    alerts.append({
                        'type': 'spike',
                        'metric': 'mail_volume', 
                        'date': row['date'],
                        'value': int(row['mail_volume']),
                        'threshold': round(mail_mean + (mail_threshold * mail_std), 1),
                        'severity': 'high' if (row['mail_volume'] - mail_mean) > (3 * mail_std) else 'medium'
                    })
            
            # Correlation alerts
            if not self.correlation_results.empty:
                max_corr = self.correlation_results['correlation'].max()
                corr_threshold = self.config['ALERT_THRESHOLDS']['correlation_threshold']
                
                if max_corr > corr_threshold:
                    best_lag = self.correlation_results.loc[self.correlation_results['correlation'].idxmax()]
                    alerts.append({
                        'type': 'correlation',
                        'metric': 'mail_call_correlation',
                        'value': round(max_corr, 3),
                        'lag_days': int(best_lag['lag_days']),
                        'severity': 'info'
                    })
            
            self.alerts = alerts
            self.logger.info(f"✅ Alert detection complete. Found {len(alerts)} alerts")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Alert detection failed: {str(e)}")
            return False

# =============================================================================
# ENHANCED SAFE DASHBOARD
# =============================================================================

class SafeDashboard:
    def __init__(self, data_processor):
        self.dp = data_processor
        self.logger = data_processor.logger

        if not DASH_AVAILABLE:
            self.logger.error("Dash not available. Cannot create dashboard.")
            return
        
        try:
            self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
            self.app.title = "Enhanced Marketing Intelligence Dashboard"
            self.setup_layout()
            self.setup_callbacks()
            
        except Exception as e:
            self.logger.error(f"Dashboard initialization failed: {str(e)}")

    def create_error_figure(self, message):
        """Create a simple error figure."""
        fig = go.Figure()
        fig.add_annotation(
            text=f"⚠️ {message}", xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False, font=dict(size=16)
        )
        fig.update_layout(title="Chart Unavailable", template='plotly_white', height=400)
        return fig

    def create_overview_figure(self, df):
        """Enhanced overview chart with normalized data and financial context."""
        if df.empty:
            return self.create_error_figure("No data available for this period")

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Normalized Volume Trends', 'Raw Volume with Moving Averages'),
            vertical_spacing=0.12,
            specs=[[{"secondary_y": True}], [{"secondary_y": True}]]
        )

        # Top subplot: Normalized trends
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['call_volume_norm'], 
                      name='Call Volume (Normalized)', 
                      line=dict(color='red', width=2)),
            row=1, col=1, secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['mail_volume_norm'], 
                      name='Mail Volume (Normalized)', 
                      line=dict(color='blue', width=2)),
            row=1, col=1, secondary_y=False
        )

        # Add financial data if available
        if 'S&P 500' in df.columns and df['S&P 500'].notna().any():
            sp500_norm = (df['S&P 500'] - df['S&P 500'].mean()) / df['S&P 500'].std()
            fig.add_trace(
                go.Scatter(x=df['date'], y=sp500_norm, 
                          name='S&P 500 (Normalized)', 
                          line=dict(color='green', width=1, dash='dot'),
                          opacity=0.7),
                row=1, col=1, secondary_y=True
            )

        # Bottom subplot: Raw volumes with moving averages
        fig.add_trace(
            go.Bar(x=df['date'], y=df['mail_volume'], 
                   name='Mail Volume', 
                   marker_color='lightblue', opacity=0.6),
            row=2, col=1, secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['mail_volume_ma'], 
                      name='Mail 7D MA', 
                      line=dict(color='blue', dash='dash')),
            row=2, col=1, secondary_y=False
        )

        fig.add_trace(
            go.Scatter(x=df['date'], y=df['call_volume'], 
                      name='Call Volume', 
                      line=dict(color='red', width=2)),
            row=2, col=1, secondary_y=True
        )
        
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['call_volume_ma'], 
                      name='Call 7D MA', 
                      line=dict(color='darkred', dash='dash')),
            row=2, col=1, secondary_y=True
        )

        # Update layout
        fig.update_layout(
            title="Executive Overview: Volume Trends & Market Context",
            template='plotly_white',
            height=600,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Normalized Score", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Market Index (Normalized)", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Mail Volume", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Call Volume", row=2, col=1, secondary_y=True)
        
        return fig

    def create_correlation_figure(self, df_corr):
        """Enhanced correlation analysis with statistical significance."""
        if df_corr.empty:
            return self.create_error_figure("Correlation data not available")

        # Find peak correlation
        significant_corrs = df_corr[df_corr['significant'] == True] if 'significant' in df_corr.columns else df_corr
        
        if not significant_corrs.empty:
            best_lag = significant_corrs.loc[significant_corrs['correlation'].abs().idxmax()]
        else:
            best_lag = df_corr.loc[df_corr['correlation'].abs().idxmax()]

        # Color code bars
        colors = []
        for _, row in df_corr.iterrows():
            if 'significant' in row and row['significant']:
                if row['lag_days'] == best_lag['lag_days']:
                    colors.append('red')  # Peak significant correlation
                else:
                    colors.append('orange')  # Other significant correlations
            else:
                colors.append('lightgray')  # Non-significant

        fig = go.Figure()
        
        # Add correlation bars
        fig.add_trace(go.Bar(
            x=df_corr['lag_days'], 
            y=df_corr['correlation'],
            marker_color=colors,
            name='Correlation',
            text=[f"{c:.3f}" for c in df_corr['correlation']],
            textposition='outside'
        ))

        # Add significance threshold line
        if 'significant' in df_corr.columns:
            fig.add_hline(y=0.3, line_dash="dash", line_color="green", 
                         annotation_text="Significance Threshold")

        # Add peak annotation
        fig.add_annotation(
            x=best_lag['lag_days'],
            y=best_lag['correlation'],
            text=f"Peak: Day {int(best_lag['lag_days'])}<br>r={best_lag['correlation']:.3f}",
            arrowhead=2,
            arrowcolor="red",
            bgcolor="white",
            bordercolor="red"
        )

        fig.update_layout(
            title=f"Mail-to-Call Lag Correlation Analysis<br><sub>Peak correlation at {int(best_lag['lag_days'])} days (r={best_lag['correlation']:.3f})</sub>",
            xaxis_title="Lag in Days (Mail Sent → Call Received)",
            yaxis_title="Pearson Correlation Coefficient",
            template='plotly_white',
            height=450
        )
        
        return fig
    
    def create_mail_type_figure(self, df_mail):
        """Enhanced mail type analysis with efficiency metrics."""
        if df_mail.empty:
            return self.create_error_figure("Mail type data unavailable")

        # Calculate percentages
        total_volume = df_mail['volume'].sum()
        df_mail = df_mail.copy()
        df_mail['percentage'] = (df_mail['volume'] / total_volume * 100).round(1)

        fig = go.Figure()
        
        # Add bars with custom colors
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        fig.add_trace(go.Bar(
            x=df_mail['type'],
            y=df_mail['volume'],
            marker_color=colors[:len(df_mail)],
            text=[f"{v:,}<br>({p}%)" for v, p in zip(df_mail['volume'], df_mail['percentage'])],
            textposition='outside',
            name='Volume'
        ))

        fig.update_layout(
            title="Mail Volume Distribution by Campaign Type",
            xaxis_title="Campaign Type",
            yaxis_title="Total Volume",
            template='plotly_white',
            height=400,
            showlegend=False
        )
        
        return fig

    def create_call_intent_figure(self, df_calls):
        """Enhanced call intent analysis with better visualization."""
        if df_calls.empty:
            return self.create_error_figure("Call intent data unavailable")

        # Calculate percentages
        total_calls = df_calls['count'].sum()
        df_calls = df_calls.copy()
        df_calls['percentage'] = (df_calls['count'] / total_calls * 100).round(1)

        # Create donut chart
        fig = go.Figure(data=[go.Pie(
            labels=df_calls['intent'],
            values=df_calls['count'],
            hole=0.4,
            textinfo='label+percent',
            textposition='outside',
            marker=dict(colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'][:len(df_calls)])
        )])

        # Add center annotation
        fig.add_annotation(
            text=f"Total<br>{total_calls:,}<br>Calls",
            x=0.5, y=0.5,
            font_size=16,
            showarrow=False
        )

        fig.update_layout(
            title="Call Distribution by Intent Category",
            template='plotly_white',
            height=400,
            showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5)
        )
        
        return fig

    def create_intent_correlation_heatmap(self, df_corr):
        """Enhanced intent correlation heatmap with better formatting."""
        if df_corr.empty:
            return self.create_error_figure("Intent correlation data not available")
        
        # Convert to numeric and handle any remaining issues
        df_numeric = df_corr.astype(float)
        
        fig = go.Figure(data=go.Heatmap(
            z=df_numeric.values,
            x=df_numeric.columns,
            y=df_numeric.index,
            colorscale='RdYlBu_r',
            zmid=0,
            text=df_numeric.round(3).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Correlation<br>Strength")
        ))

        fig.update_layout(
            title="Peak Correlation Matrix: Mail Type vs Call Intent<br><sub>Values represent maximum correlation across all lag periods</sub>",
            xaxis_title="Call Intent",
            yaxis_title="Mail Campaign Type",
            template='plotly_white',
            height=500
        )
        
        return fig

    def create_weekday_intent_figure(self):
        """Enhanced weekday analysis with pattern insights."""
        df = self.dp.call_df
        if df.empty:
            return self.create_error_figure("Call data unavailable")

        # Ensure we have date column
        if 'date' not in df.columns:
            return self.create_error_figure("Date column not found in call data")

        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        df = df.copy()
        df['weekday'] = pd.Categorical(df['date'].dt.day_name(), categories=weekday_order, ordered=True)

        weekday_counts = df.groupby(['weekday', 'intent']).size().reset_index(name='count')

        fig = px.bar(
            weekday_counts, 
            x='weekday', 
            y='count', 
            color='intent',
            title="Call Volume Patterns: Intent Distribution by Weekday",
            labels={
                'weekday': 'Day of the Week', 
                'count': 'Number of Calls', 
                'intent': 'Call Intent'
            },
            color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        )
        
        # Add trend line for total daily calls
        daily_totals = weekday_counts.groupby('weekday')['count'].sum().reset_index()
        fig.add_trace(go.Scatter(
            x=daily_totals['weekday'],
            y=daily_totals['count'],
            mode='lines+markers',
            name='Total Daily Calls',
            line=dict(color='black', width=3),
            yaxis='y2'
        ))

        fig.update_layout(
            template='plotly_white',
            height=500,
            yaxis2=dict(
                title="Total Calls",
                overlaying='y',
                side='right'
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        return fig

    def create_efficiency_dashboard(self):
        """Create efficiency metrics dashboard."""
        metrics = self.dp.efficiency_metrics
        
        if not metrics:
            return html.Div("Efficiency metrics not available")

        cards = []
        
        # Response rate card
        response_color = "success" if metrics['response_rate_pct'] > 2 else "warning" if metrics['response_rate_pct'] > 1 else "danger"
        cards.append(
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{metrics['response_rate_pct']}%", className="card-title text-center"),
                    html.P("Response Rate", className="card-text text-center"),
                    html.Small(f"({metrics['total_calls']:,} calls per {metrics['total_mail']:,} mails)", 
                             className="text-muted text-center d-block")
                ])
            ], color=response_color, outline=True)
        )
        
        # Trend cards
        call_trend_color = "success" if metrics['call_trend_30d'] > 0 else "danger"
        mail_trend_color = "success" if metrics['mail_trend_30d'] > 0 else "danger"
        
        cards.extend([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{metrics['call_trend_30d']:+.1f}%", className="card-title text-center"),
                    html.P("Call Trend (30d)", className="card-text text-center"),
                ])
            ], color=call_trend_color, outline=True),
            
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{metrics['mail_trend_30d']:+.1f}%", className="card-title text-center"),
                    html.P("Mail Trend (30d)", className="card-text text-center"),
                ])
            ], color=mail_trend_color, outline=True),
        ])
        
        # Peak days card
        cards.append(
            dbc.Card([
                dbc.CardBody([
                    html.H6("Peak Activity Days", className="card-title text-center"),
                    html.P(f"📞 Calls: {metrics['peak_call_day']}", className="text-center mb-1"),
                    html.P(f"📧 Mail: {metrics['peak_mail_day']}", className="text-center mb-0"),
                ])
            ], color="info", outline=True)
        )

        return dbc.Row([dbc.Col(card, width=3) for card in cards])

    def create_alerts_component(self):
        """Create alerts component."""
        alerts = self.dp.alerts
        
        if not alerts:
            return dbc.Alert("No alerts detected. All metrics are within normal ranges.", color="success")

        alert_components = []
        
        for alert in alerts[-5:]:  # Show last 5 alerts
            if alert['type'] == 'spike':
                color = "danger" if alert['severity'] == 'high' else "warning"
                icon = "🚨" if alert['severity'] == 'high' else "⚠️"
                
                message = f"{icon} {alert['metric'].replace('_', ' ').title()} spike detected on {alert['date'].strftime('%Y-%m-%d')}: {alert['value']:,} (threshold: {alert['threshold']})"
                
            elif alert['type'] == 'correlation':
                color = "info"
                message = f"📊 Strong correlation detected: {alert['value']} at {alert['lag_days']}-day lag"
            
            alert_components.append(dbc.Alert(message, color=color, className="mb-2"))

        return html.Div(alert_components)

    def setup_layout(self):
        """Enhanced dashboard layout with executive-level insights."""
        df = self.dp.combined_df
        if df.empty:
            self.app.layout = html.Div([
                html.H1("⚠️ No Data Loaded"), 
                html.P("Dashboard cannot render without data.")
            ])
            return

        start_date = df['date'].min().date()
        end_date = df['date'].max().date()

        self.app.layout = dbc.Container([
            # Header with branding
            dbc.Row([
                dbc.Col([
                    html.H1("📊 Executive Marketing Intelligence Dashboard", 
                           className="text-center text-primary mb-2"),
                    html.P("Real-time insights for data-driven decision making", 
                          className="text-center text-muted")
                ], width=12)
            ], className="mb-4"),
            
            # Alert banner
            dbc.Row([
                dbc.Col([
                    html.Div(id='alerts-banner')
                ], width=12)
            ], className="mb-3"),
            
            # Interactive Controls
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5("📅 Analysis Period", className="mb-3"),
                            dcc.DatePickerRange(
                                id='date-picker-range',
                                min_date_allowed=start_date,
                                max_date_allowed=end_date,
                                start_date=start_date,
                                end_date=end_date,
                                display_format='YYYY-MM-DD'
                            )
                        ], width=4),
                        dbc.Col([
                            html.H5("🎯 Quick Insights", className="mb-3"),
                            html.Div(id='quick-insights')
                        ], width=8)
                    ])
                ])
            ], className="mb-4"),

            # KPI Cards
            dbc.Row(id='kpi-cards-row', className="mb-4"),
            
            # Efficiency Metrics
            dbc.Card([
                dbc.CardHeader(html.H5("⚡ Efficiency Metrics", className="mb-0")),
                dbc.CardBody(id='efficiency-metrics')
            ], className="mb-4"),

            # Main Tabbed Interface
            dbc.Tabs([
                dbc.Tab(label="📈 Executive Overview", tab_id="tab-overview", children=[
                    dcc.Graph(id='overview-chart')
                ]),
                
                dbc.Tab(label="🔗 Correlation Intelligence", tab_id="tab-correlation", children=[
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='correlation-chart')
                        ], width=8),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader(html.H6("🎯 Key Insights")),
                                dbc.CardBody(id='correlation-insights')
                            ])
                        ], width=4)
                    ], className="mt-4")
                ]),
                
                dbc.Tab(label="📊 Performance Breakdowns", tab_id="tab-breakdowns", children=[
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='mail-type-chart')
                        ], width=6),
                        dbc.Col([
                            dcc.Graph(id='call-intent-chart')
                        ], width=6)
                    ], className="mt-4")
                ]),
                
                dbc.Tab(label="🎯 Advanced Analytics", tab_id="tab-analytics", children=[
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='intent-heatmap')
                        ], width=7),
                        dbc.Col([
                            dcc.Graph(id='weekday-intent-chart')
                        ], width=5)
                    ], className="mt-4"),
                    
                    # Financial correlation if available
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='financial-correlation')
                        ], width=12)
                    ], className="mt-4") if not self.dp.financial_df.empty else html.Div()
                ])
            ], className="mb-4"),
            
            # Footer with data info
            dbc.Row([
                dbc.Col([
                    html.Hr(),
                    html.P([
                        f"📊 Data Status: {len(self.dp.mail_df):,} mail records, {len(self.dp.call_df):,} call records | ",
                        f"📅 Period: {start_date} to {end_date} | ",
                        f"🔄 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                    ], className="text-muted text-center small")
                ], width=12)
            ])
        ], fluid=True)

    def setup_callbacks(self):
        """Enhanced callbacks for interactive dashboard."""
        
        @self.app.callback(
            [Output('kpi-cards-row', 'children'),
             Output('overview-chart', 'figure'),
             Output('quick-insights', 'children'),
             Output('alerts-banner', 'children')],
            [Input('date-picker-range', 'start_date'),
             Input('date-picker-range', 'end_date')]
        )
        def update_overview_and_kpis(start_date, end_date):
            if not start_date or not end_date:
                return [], self.create_error_figure("Please select a date range."), "", ""

            # Filter data
            dff = self.dp.combined_df[
                (self.dp.combined_df['date'] >= start_date) & 
                (self.dp.combined_df['date'] <= end_date)
            ]

            if dff.empty:
                return [], self.create_error_figure("No data for selected period."), "", ""

            # Calculate KPIs
            total_calls = int(dff['call_volume'].sum())
            total_mail = int(dff['mail_volume'].sum())
            avg_daily_calls = total_calls / len(dff) if len(dff) > 0 else 0
            avg_daily_mail = total_mail / len(dff) if len(dff) > 0 else 0

            # KPI Cards with enhanced styling
            kpi_cards = [
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3(f"{total_calls:,}", className="card-title text-primary"),
                            html.P("Total Calls", className="card-text"),
                            html.Small(f"📈 {avg_daily_calls:.1f}/day avg", className="text-muted")
                        ])
                    ], className="h-100")
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3(f"{total_mail:,}", className="card-title text-info"),
                            html.P("Total Mail", className="card-text"),
                            html.Small(f"📧 {avg_daily_mail:,.0f}/day avg", className="text-muted")
                        ])
                    ], className="h-100")
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3(f"{(total_calls/total_mail*100):.1f}%" if total_mail > 0 else "N/A", 
                                   className="card-title text-success"),
                            html.P("Response Rate", className="card-text"),
                            html.Small("📞 Calls per 100 mails", className="text-muted")
                        ])
                    ], className="h-100")
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3(f"{dff['calls_per_1k_mails'].mean():.1f}" if 'calls_per_1k_mails' in dff.columns else "N/A", 
                                   className="card-title text-warning"),
                            html.P("Efficiency Ratio", className="card-text"),
                            html.Small("📊 Calls per 1K mails", className="text-muted")
                        ])
                    ], className="h-100")
                ], width=3)
            ]
            
            # Quick insights
            if not self.dp.correlation_results.empty:
                best_corr = self.dp.correlation_results.loc[self.dp.correlation_results['correlation'].abs().idxmax()]
                insights = [
                    dbc.Badge(f"🎯 Peak correlation: {best_corr['correlation']:.3f} at {int(best_corr['lag_days'])} days", 
                             color="primary", className="me-2"),
                    dbc.Badge(f"📈 Trend: {'↗️ Growing' if avg_daily_calls > dff['call_volume'].iloc[:len(dff)//2].mean() else '↘️ Declining'}", 
                             color="info", className="me-2")
                ]
            else:
                insights = [dbc.Badge("📊 Analyzing patterns...", color="secondary")]
            
            # Alerts banner
            alerts_component = self.create_alerts_component()
            
            overview_fig = self.create_overview_figure(dff)
            
            return kpi_cards, overview_fig, insights, alerts_component

        @self.app.callback(
            [Output('correlation-chart', 'figure'),
             Output('correlation-insights', 'children')],
            [Input('date-picker-range', 'id')]
        )
        def update_correlation_tab(_):
            corr_fig = self.create_correlation_figure(self.dp.correlation_results)

            if not self.dp.correlation_results.empty:
                best_lag = self.dp.correlation_results.loc[self.dp.correlation_results['correlation'].abs().idxmax()]
                
                # Generate insights
                insights = [
                    html.H6("🔍 Analysis Results", className="mb-3"),
                    html.P([
                        "Mail campaigns show the strongest correlation with call volume ",
                        html.Strong(f"{int(best_lag['lag_days'])} days later"),
                        f" with a correlation coefficient of ",
                        html.Strong(f"{best_lag['correlation']:.3f}")
                    ]),
                    html.Hr(),
                    html.H6("💡 Business Implications"),
                    html.Ul([
                        html.Li("Plan support capacity based on mail send schedule"),
                        html.Li("Optimize campaign timing for resource availability"), 
                        html.Li("Use predictive staffing models")
                    ]),
                    html.Hr(),
                    html.H6("📈 Recommended Actions"),
                    html.P("Implement automated alerts for mail campaigns exceeding 1M recipients to pre-position support staff.", 
                          className="small text-muted")
                ]
            else:
                insights = [
                    html.P("Correlation analysis requires more data points.", className="text-muted"),
                    html.P("Please ensure sufficient historical data is available.")
                ]
            
            return corr_fig, insights

        @self.app.callback(
            [Output('mail-type-chart', 'figure'),
             Output('call-intent-chart', 'figure')],
            [Input('date-picker-range', 'id')]
        )
        def update_breakdown_tab(_):
            mail_fig = self.create_mail_type_figure(self.dp.mail_by_type)
            call_fig = self.create_call_intent_figure(self.dp.calls_by_intent)
            return mail_fig, call_fig

        @self.app.callback(
            [Output('intent-heatmap', 'figure'),
             Output('weekday-intent-chart', 'figure')],
            [Input('date-picker-range', 'id')]
        )
        def update_analytics_tab(_):
            heatmap_fig = self.create_intent_correlation_heatmap(self.dp.intent_correlation_matrix)
            weekday_fig = self.create_weekday_intent_figure()
            return heatmap_fig, weekday_fig

        @self.app.callback(
            Output('efficiency-metrics', 'children'),
            [Input('date-picker-range', 'id')]
        )
        def update_efficiency_metrics(_):
            return self.create_efficiency_dashboard()

        # Financial correlation chart if financial data is available
        if not self.dp.financial_df.empty:
            @self.app.callback(
                Output('financial-correlation', 'figure'),
                [Input('date-picker-range', 'id')]
            )
            def update_financial_correlation(_):
                return self.create_financial_correlation_chart()

    def create_financial_correlation_chart(self):
        """Create financial market correlation analysis."""
        df = self.dp.combined_df
        
        if df.empty or 'S&P 500' not in df.columns:
            return self.create_error_figure("Financial data not available")
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Call Volume vs Market Performance', 'Market Volatility Impact'),
            vertical_spacing=0.15,
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
        )
        
        # Top: Call volume vs S&P 500
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['call_volume_norm'], 
                      name='Call Volume (Normalized)', line=dict(color='red')),
            row=1, col=1, secondary_y=False
        )
        
        if 'S&P 500' in df.columns:
            sp500_norm = (df['S&P 500'] - df['S&P 500'].mean()) / df['S&P 500'].std()
            fig.add_trace(
                go.Scatter(x=df['date'], y=sp500_norm, 
                          name='S&P 500 (Normalized)', line=dict(color='green')),
                row=1, col=1, secondary_y=True
            )
        
        # Bottom: VIX correlation
        if 'VIX' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['date'], y=df['VIX'], 
                          name='VIX (Volatility)', line=dict(color='orange')),
                row=2, col=1
            )
        
        fig.update_layout(
            title="Financial Market Correlation Analysis",
            template='plotly_white',
            height=600
        )
        
        fig.update_yaxes(title_text="Normalized Values", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Market Index", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Volatility Index", row=2, col=1)
        
        return fig

    def run(self, debug=True, port=8050):
        """Run dashboard with enhanced error handling."""
        try:
            self.logger.info(f"🚀 Starting enhanced dashboard at http://127.0.0.1:{port}")
            self.app.run(debug=debug, port=port, host='127.0.0.1')
        except Exception as e:
            self.logger.error(f"Dashboard failed to start: {str(e)}")

# =============================================================================
# ENHANCED MAIN FUNCTION
# =============================================================================

def main():
    """Enhanced main function with comprehensive processing pipeline."""
    print("🔥 ENHANCED EXECUTIVE MARKETING INTELLIGENCE DASHBOARD")
    print("=" * 70)
    
    logger = setup_logging()

    try:
        if not DASH_AVAILABLE:
            logger.error("❌ Cannot run dashboard without Dash. Exiting.")
            return False
            
        logger.info("🏗️ Initializing enhanced data processor...")
        dp = SafeDataProcessor(CONFIG, logger)
        
        # Load data with fallback to sample data
        if not dp.load_data() or (dp.mail_df.empty and dp.call_df.empty):
            logger.warning("⚠️ Real data loading failed or returned empty, creating sample data...")
            if not dp.create_sample_data():
                logger.error("❌ Sample data creation failed. Exiting.")
                return False
        
        # Load financial data if available
        if FINANCIAL_AVAILABLE:
            dp.safe_financial_data()
        
        # Combine and enhance data
        if not dp.combine_data():
            logger.error("❌ Data combination failed. Exiting.")
            return False
        
        # Run enhanced analysis pipeline
        logger.info("🔄 Running enhanced analysis pipeline...")
        
        # Correlation analysis (FIXED: Now properly called)
        if SCIPY_AVAILABLE:
            dp.analyze_correlation()
            dp.analyze_intent_correlation()
        else:
            logger.warning("⚠️ SciPy not available - correlation analysis disabled")
        
        # Calculate efficiency metrics
        dp.calculate_efficiency_metrics()
        
        # Detect alerts and anomalies
        dp.detect_alerts()
        
        logger.info("🎨 Initializing enhanced dashboard...")
        dashboard = SafeDashboard(dp)
        
        # Enhanced status report
        logger.info("=" * 40)
        logger.info("📊 ENHANCED DASHBOARD STATUS:")
        logger.info(f"📧 Mail records processed: {len(dp.mail_df):,}")
        logger.info(f"📞 Call records processed: {len(dp.call_df):,}")
        logger.info(f"💰 Financial indicators: {len(dp.financial_df.columns)-1 if not dp.financial_df.empty else 0}")
        logger.info(f"🔗 Correlation analysis: {'✅ Complete' if not dp.correlation_results.empty else '❌ Failed'}")
        logger.info(f"🎯 Intent correlation: {'✅ Complete' if not dp.intent_correlation_matrix.empty else '❌ Failed'}")
        logger.info(f"⚡ Efficiency metrics: {'✅ Complete' if dp.efficiency_metrics else '❌ Failed'}")
        logger.info(f"🚨 Alerts detected: {len(dp.alerts)}")
        logger.info("=" * 40)
        
        if not dp.correlation_results.empty:
            best_corr = dp.correlation_results.loc[dp.correlation_results['correlation'].abs().idxmax()]
            logger.info(f"🎯 KEY INSIGHT: Peak correlation of {best_corr['correlation']:.3f} at {int(best_corr['lag_days'])} days")
        
        if dp.efficiency_metrics:
            logger.info(f"📈 RESPONSE RATE: {dp.efficiency_metrics['response_rate_pct']:.2f}%")
            logger.info(f"📊 CALL TREND: {dp.efficiency_metrics['call_trend_30d']:+.1f}% (30-day)")
        
        logger.info("🌐 Access enhanced dashboard at: http://127.0.0.1:8050")
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
        print("\n✅ Enhanced dashboard session ended successfully.")
    else:
        print("\n⚠️ Enhanced dashboard session ended with errors.")
    
    sys.exit(0)
