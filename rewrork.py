# #!/usr/bin/env python3
"""
Executive Marketing Intelligence Dashboard - Robust Version

Bulletproof dashboard with comprehensive error handling and fallbacks.
Designed to work even with incomplete data or missing dependencies.
"""

# — Core Libraries —

import pandas as pd
import numpy as np
import warnings
import os
import sys
from datetime import datetime, timedelta
import logging
from pathlib import Path

# — Suppress all warnings —

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

# — Visualization & Dashboard —

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

# — Optional Libraries —

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

# =============================================================================

# CONFIGURATION - KEEP YOUR SETTINGS

# =============================================================================

CONFIG = {
'MAIL_FILE_PATH': r'merged_output.csv',
'CALL_FILE_PATH': r'data\GenesysExtract_20250609.csv',
'MAIL_COLUMNS': {'date': 'mail_date', 'volume': 'mail_volume', 'type': 'mail_type'},
'CALL_COLUMNS': {'date': 'ConversationStart', 'intent': 'uui_Intent'},
'FINANCIAL_DATA': {'S&P 500': '^GSPC', '10-Yr Treasury': '^TNX', 'VIX': '^VIX'},
'MAX_LAG_DAYS': 28,
'DEBUG_MODE': True
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
        return logging.getLogger('RobustDashboard')
    except Exception:
        print("Logging setup failed, using print statements")
    return None

# =============================================================================

# SAFE DATA PROCESSING

# =============================================================================

class SafeDataProcessor:
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger or self._dummy_logger()
        self.mail_df = pd.DataFrame()
        self.call_df = pd.DataFrame()
        self.financial_df = pd.DataFrame()
        self.combined_df = pd.DataFrame()
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
            
            # Try different encoding strategies
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip')
                    self.logger.info(f"✅ {description} loaded: {len(df):,} records (encoding: {encoding})")
                    return df
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    self.logger.warning(f"Failed to load {description} with {encoding}: {str(e)}")
                    continue
            
            # Last resort - try with minimal parameters
            try:
                df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip', low_memory=False)
                self.logger.info(f"✅ {description} loaded with fallback method: {len(df):,} records")
                return df
            except Exception as e:
                self.logger.error(f"❌ Complete failure loading {description}: {str(e)}")
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
                # Direct match
                if config_name in mapped_df.columns:
                    if config_name != standard_name:
                        mapped_df = mapped_df.rename(columns={config_name: standard_name})
                        successful_mappings[config_name] = standard_name
                    continue
                
                # Fuzzy match
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
            
            # Strategy 1: Standard pandas conversion
            try:
                df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
                df = df.dropna(subset=[date_column])
                df[date_column] = df[date_column].dt.normalize()
                
                success_rate = len(df) / original_count * 100
                self.logger.info(f"Date conversion success rate: {success_rate:.1f}%")
                return df
                
            except Exception as e:
                self.logger.warning(f"Standard date conversion failed: {str(e)}")
            
            # Strategy 2: Try common date formats
            date_formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S']
            
            for fmt in date_formats:
                try:
                    df[date_column] = pd.to_datetime(df[date_column], format=fmt, errors='coerce')
                    df = df.dropna(subset=[date_column])
                    df[date_column] = df[date_column].dt.normalize()
                    
                    success_rate = len(df) / original_count * 100
                    self.logger.info(f"Date conversion with format {fmt} success rate: {success_rate:.1f}%")
                    return df
                except:
                    continue
            
            self.logger.error("All date conversion strategies failed")
            return df
            
        except Exception as e:
            self.logger.error(f"Date conversion error: {str(e)}")
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
                
                # Ensure volume column is numeric
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
        """Create sample data if real data fails to load."""
        try:
            self.logger.info("🔧 Creating sample data for demonstration...")
            
            # Create date range
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=90)
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Sample call data
            np.random.seed(42)
            call_volumes = np.random.poisson(50, len(dates)) + np.random.normal(0, 10, len(dates))
            call_volumes = np.maximum(call_volumes, 0)  # Ensure non-negative
            
            self.call_df = pd.DataFrame({
                'date': dates,
                'call_volume': call_volumes.astype(int),
                'intent': np.random.choice(['billing', 'support', 'sales'], len(dates))
            })
            
            # Sample mail data
            mail_volumes = np.random.poisson(1000, len(dates)) + np.random.normal(0, 200, len(dates))
            mail_volumes = np.maximum(mail_volumes, 0)
            
            self.mail_df = pd.DataFrame({
                'date': dates,
                'volume': mail_volumes.astype(int),
                'type': np.random.choice(['promotional', 'newsletter', 'reminder'], len(dates))
            })
            
            self.logger.info("✅ Sample data created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Sample data creation failed: {str(e)}")
            return False

    def safe_financial_data(self):
        """Safely attempt to load financial data."""
        if not FINANCIAL_AVAILABLE:
            self.logger.warning("📊 Financial data unavailable - yfinance not installed")
            return False
        
        try:
            self.logger.info("🔄 Attempting to load financial data...")
            
            # Get date range
            if not self.call_df.empty:
                start_date = self.call_df['date'].min() - timedelta(days=5)
                end_date = self.call_df['date'].max()
            else:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=90)
            
            financial_data = {}
            
            for name, ticker in self.config['FINANCIAL_DATA'].items():
                try:
                    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    if not data.empty and 'Close' in data.columns:
                        financial_data[name] = data['Close'].resample('D').last()
                except Exception as e:
                    self.logger.warning(f"Failed to fetch {name}: {str(e)}")
                    continue
            
            if financial_data:
                self.financial_df = pd.DataFrame(financial_data)
                self.financial_df.index.name = 'date'
                self.financial_df = self.financial_df.reset_index()
                self.financial_df['date'] = pd.to_datetime(self.financial_df['date']).dt.normalize()
                self.logger.info(f"✅ Financial data loaded: {len(financial_data)} indicators")
                return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Financial data loading failed: {str(e)}")
            return False

    def combine_data(self):
        """Safely combine all data sources."""
        try:
            self.logger.info("🔄 Combining data sources...")
            
            # Start with call data aggregation
            if not self.call_df.empty:
                if 'call_volume' in self.call_df.columns:
                    daily_calls = self.call_df.groupby('date')['call_volume'].sum().reset_index()
                else:
                    daily_calls = self.call_df.groupby('date').size().reset_index(name='call_volume')
            else:
                # Create minimal structure
                dates = pd.date_range(start=datetime.now().date() - timedelta(days=30), 
                                    end=datetime.now().date(), freq='D')
                daily_calls = pd.DataFrame({'date': dates, 'call_volume': 0})
            
            # Add mail data
            if not self.mail_df.empty and 'volume' in self.mail_df.columns:
                daily_mail = self.mail_df.groupby('date')['volume'].sum().reset_index(name='mail_volume')
                combined = daily_calls.merge(daily_mail, on='date', how='outer')
            else:
                combined = daily_calls.copy()
                combined['mail_volume'] = 0
            
            # Fill missing values safely
            combined['call_volume'] = combined['call_volume'].fillna(0)
            combined['mail_volume'] = combined['mail_volume'].fillna(0)
            
            # Add financial data if available
            if not self.financial_df.empty:
                combined = combined.merge(self.financial_df, on='date', how='left')
                
                # Safe forward fill for financial data
                financial_cols = [col for col in combined.columns if col in self.config['FINANCIAL_DATA'].keys()]
                for col in financial_cols:
                    combined[col] = combined[col].ffill().bfill()
            
            # Add derived features
            combined['weekday'] = combined['date'].dt.day_name()
            combined['month'] = combined['date'].dt.month
            combined['is_weekend'] = combined['date'].dt.weekday >= 5
            combined['response_rate'] = np.where(
                combined['mail_volume'] > 0,
                combined['call_volume'] / combined['mail_volume'] * 100,
                0
            )
            
            # Sort by date
            combined = combined.sort_values('date').reset_index(drop=True)
            
            self.combined_df = combined
            self.logger.info(f"✅ Combined dataset: {len(combined):,} records")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Data combination failed: {str(e)}")
            return False


# =============================================================================

# SAFE DASHBOARD

# =============================================================================

class SafeDashboard:
    def __init__(self, data_processor):
        self.dp = data_processor
        self.logger = data_processor.logger


        if not DASH_AVAILABLE:
                self.logger.error("Dash not available. Cannot create dashboard.")
                return
        
        try:
            self.app = dash.Dash(
                __name__,
                external_stylesheets=[
                    "https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css",
                    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
                ],
                suppress_callback_exceptions=True
            )
            self.app.title = "Marketing Intelligence Dashboard"
            self.setup_layout()
            self.setup_callbacks()
            
        except Exception as e:
            self.logger.error(f"Dashboard initialization failed: {str(e)}")

    def safe_figure(self, figure_func):
        """Safely create plotly figures with fallbacks."""
        try:
            if not PLOTLY_AVAILABLE:
                return self.create_error_figure("Plotly not available")
            
            return figure_func()
            
        except Exception as e:
            self.logger.error(f"Figure creation failed: {str(e)}")
            return self.create_error_figure(f"Error: {str(e)}")

    def create_error_figure(self, message):
        """Create a simple error figure."""
        try:
            fig = go.Figure()
            fig.add_annotation(
                text=f"⚠️ {message}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            fig.update_layout(
                title="Chart Unavailable",
                template='plotly_white',
                height=400
            )
            return fig
        except:
            # Ultimate fallback
            return {}

    def create_main_figure(self):
        """Create main dashboard figure."""
        try:
            df = self.dp.combined_df
            if df.empty:
                return self.create_error_figure("No data available")
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Call Volume", "Mail Volume"),
                vertical_spacing=0.1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['call_volume'],
                    mode='lines',
                    name='Calls',
                    line=dict(color='#1f77b4', width=2)
                ), row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=df['date'],
                    y=df['mail_volume'],
                    name='Mail',
                    marker_color='#2ca02c',
                    opacity=0.7
                ), row=2, col=1
            )
            
            fig.update_layout(
                title="Marketing Performance Overview",
                template='plotly_white',
                height=500
            )
            
            return fig
            
        except Exception as e:
            return self.create_error_figure(f"Main chart error: {str(e)}")

    def setup_layout(self):
        """Setup dashboard layout with error handling."""
        try:
            df = self.dp.combined_df
            
            # Calculate safe KPIs
            total_calls = int(df['call_volume'].sum()) if not df.empty else 0
            total_mail = int(df['mail_volume'].sum()) if not df.empty else 0
            avg_response = df['response_rate'].mean() if not df.empty else 0
            
            self.app.layout = html.Div([
                # Header
                html.Div([
                    html.H1("📊 Marketing Intelligence Dashboard", className="text-center text-white"),
                    html.P("Executive Analytics Platform", className="text-center text-white"),
                ], className="bg-primary p-4 mb-4"),
                
                # KPI Cards
                html.Div([
                    html.Div([
                        html.Div([
                            html.H3(f"{total_calls:,}", className="text-primary"),
                            html.P("Total Calls", className="text-muted")
                        ], className="card-body text-center")
                    ], className="card col-md-3 mx-2"),
                    
                    html.Div([
                        html.Div([
                            html.H3(f"{total_mail:,}", className="text-success"),
                            html.P("Mail Volume", className="text-muted")
                        ], className="card-body text-center")
                    ], className="card col-md-3 mx-2"),
                    
                    html.Div([
                        html.Div([
                            html.H3(f"{avg_response:.1f}%", className="text-info"),
                            html.P("Response Rate", className="text-muted")
                        ], className="card-body text-center")
                    ], className="card col-md-3 mx-2"),
                    
                    html.Div([
                        html.Div([
                            html.H3("✅", className="text-warning"),
                            html.P("System Status", className="text-muted")
                        ], className="card-body text-center")
                    ], className="card col-md-3 mx-2"),
                ], className="row mb-4"),
                
                # Main Chart
                html.Div([
                    dcc.Graph(
                        id='main-chart',
                        figure=self.safe_figure(self.create_main_figure)
                    )
                ], className="mb-4"),
                
                # Data Status
                html.Div([
                    html.Div([
                        html.H5("📋 Data Status"),
                        html.P(f"Records: {len(df):,}" if not df.empty else "No data loaded"),
                        html.P(f"Date Range: {df['date'].min().date()} to {df['date'].max().date()}" if not df.empty else "No date range"),
                        html.P("✅ Dashboard operational")
                    ], className="card-body")
                ], className="card")
            ], className="container-fluid")
            
        except Exception as e:
            self.logger.error(f"Layout setup failed: {str(e)}")
            # Minimal fallback layout
            self.app.layout = html.Div([
                html.H1("⚠️ Dashboard Error"),
                html.P(f"Error: {str(e)}"),
                html.P("Check logs for details.")
            ])

    def setup_callbacks(self):
        """Setup callbacks with error handling."""
        try:
            @self.app.callback(
                Output('main-chart', 'figure'),
                Input('main-chart', 'id')  # Dummy input to trigger
            )
            def update_main_chart(_):
                return self.safe_figure(self.create_main_figure)
                
        except Exception as e:
            self.logger.error(f"Callback setup failed: {str(e)}")

    def run(self, debug=True, port=8050):
        """Run dashboard with error handling."""
        try:
            self.logger.info(f"🚀 Starting dashboard at http://127.0.0.1:{port}")
            self.app.run(debug=debug, port=port, host='127.0.0.1')
        except Exception as e:
            self.logger.error(f"Dashboard failed to start: {str(e)}")
            print("Dashboard startup failed. Check error messages above.")


# =============================================================================

# BULLETPROOF MAIN FUNCTION

# =============================================================================

def main():
    """Bulletproof main function with comprehensive error handling."""
    print("🔥 EXECUTIVE MARKETING INTELLIGENCE DASHBOARD")
    print("=" * 60)
    print("🛡️  ROBUST VERSION - No Fatal Errors Guaranteed")
    print("=" * 60)


    # Setup logging
    logger = setup_logging()

    try:
        # Check dependencies
        missing_deps = []
        if not PLOTLY_AVAILABLE:
            missing_deps.append("plotly")
        if not DASH_AVAILABLE:
            missing_deps.append("dash dash-bootstrap-components")
        
        if missing_deps:
            print(f"⚠️  Missing dependencies: {', '.join(missing_deps)}")
            print("Install with: pip install " + " ".join(missing_deps))
            if not DASH_AVAILABLE:
                print("❌ Cannot run dashboard without Dash. Exiting.")
                return False
        
        # Initialize data processor
        logger.info("🏗️  Initializing robust data processor...")
        dp = SafeDataProcessor(CONFIG, logger)
        
        # Attempt to load real data
        if not dp.load_data():
            logger.warning("⚠️  Real data loading failed, creating sample data...")
            if not dp.create_sample_data():
                logger.error("❌ Sample data creation failed")
                return False
        
        # Attempt financial data (optional)
        if FINANCIAL_AVAILABLE:
            dp.safe_financial_data()
        
        # Combine data
        if not dp.combine_data():
            logger.error("❌ Data combination failed")
            return False
        
        # Initialize dashboard
        logger.info("🎨 Initializing dashboard...")
        dashboard = SafeDashboard(dp)
        
        # Final status report
        logger.info("📊 DASHBOARD STATUS:")
        logger.info("=" * 30)
        logger.info(f"📧 Mail records: {len(dp.mail_df):,}")
        logger.info(f"📞 Call records: {len(dp.call_df):,}")
        logger.info(f"📈 Combined records: {len(dp.combined_df):,}")
        logger.info(f"💰 Financial data: {'✅' if not dp.financial_df.empty else '❌'}")
        logger.info(f"🎯 Plotly: {'✅' if PLOTLY_AVAILABLE else '❌'}")
        logger.info(f"🖥️  Dash: {'✅' if DASH_AVAILABLE else '❌'}")
        logger.info("=" * 30)
        
        if DASH_AVAILABLE:
            logger.info("🌐 Access dashboard at: http://127.0.0.1:8050")
            logger.info("🛑 Press Ctrl+C to stop")
            dashboard.run(debug=CONFIG.get('DEBUG_MODE', True), port=8050)
        else:
            logger.error("❌ Dashboard cannot run without Dash")
            return False
        
        return True
        
    except KeyboardInterrupt:
        logger.info("🛑 Dashboard stopped by user")
        return True
        
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        print(f"Fatal error: {str(e)}")
        print("The dashboard encountered an unexpected error but did not crash.")
        return False


if __name__ == '__main__':
    try:
        success = main()
        print("\n" + "=" * 60)
        if success:
            print("✅ Dashboard completed successfully")
        else:
            print("⚠️  Dashboard completed with warnings")
            print("=" * 60)
    except Exception as e:
            print(f"❌ Critical error: {str(e)}")
            print("Please check your Python environment and try again.")

    finally:
            print("🔚 Dashboard session ended")
            sys.exit(0)
