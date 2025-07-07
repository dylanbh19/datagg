#!/usr/bin/env python3
"""
Executive Marketing Intelligence Dashboard - Robust Version

Bulletproof dashboard with comprehensive error handling and fallbacks.
Designed to work even with incomplete data or missing dependencies.
NEW: Includes correlation analysis, categorical breakdowns, and interactive date filtering.
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

# --- NEW INSIGHT: Added SciPy for correlation calculations ---
try:
    from scipy import stats
    from scipy.stats import pearsonr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: Scipy not available. Correlation analysis will be disabled. Install with: pip install scipy")


# =============================================================================
# CONFIGURATION - KEEP YOUR SETTINGS
# =============================================================================

CONFIG = {
    'MAIL_FILE_PATH': r'merged_output.csv',
    'CALL_FILE_PATH': r'data\GenesysExtract_20250609.csv',
    'MAIL_COLUMNS': {'date': 'mail_date', 'volume': 'mail_volume', 'type': 'mail_type'},
    'CALL_COLUMNS': {'date': 'ConversationStart', 'intent': 'uui_Intent'},
    'FINANCIAL_DATA': {'S&P 500': '^GSPC', '10-Yr Treasury': '^TNX', 'VIX': '^VIX'},
    # --- NEW INSIGHT: MAX_LAG_DAYS controls the correlation analysis window ---
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
        # --- NEW INSIGHT: Added dataframes for new analyses ---
        self.correlation_results = pd.DataFrame()
        self.mail_by_type = pd.DataFrame()
        self.calls_by_intent = pd.DataFrame()
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
            df[date_column] = df[date_column].dt.normalize()
            
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
        """Create sample data if real data fails to load."""
        try:
            self.logger.info("🔧 Creating sample data for demonstration...")
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=90)
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            np.random.seed(42)
            # Sample call data
            call_volumes = np.maximum(np.random.poisson(50, len(dates)) + np.random.normal(0, 10, len(dates)), 0)
            self.call_df = pd.DataFrame({
                'date': dates, 'call_volume_raw': call_volumes.astype(int),
                'intent': np.random.choice(['billing', 'support', 'sales', 'account_update', 'general_query'], len(dates))
            }).rename(columns={'call_volume_raw': 'call_volume'})
            
            # Sample mail data
            mail_volumes = np.maximum(np.random.poisson(1000, len(dates)) + np.random.normal(0, 200, len(dates)), 0)
            self.mail_df = pd.DataFrame({
                'date': dates, 'volume': mail_volumes.astype(int),
                'type': np.random.choice(['promotional', 'newsletter', 'reminder', 'statement'], len(dates))
            })
            
            self.logger.info("✅ Sample data created successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Sample data creation failed: {str(e)}")
            return False

    # --- MODIFIED FUNCTION: Patched yfinance download method ---
    def safe_financial_data(self):
        """Safely attempt to load financial data using a more stable method."""
        if not FINANCIAL_AVAILABLE:
            self.logger.warning("📊 Financial data unavailable - yfinance not installed")
            return False
        
        try:
            self.logger.info("🔄 Attempting to load financial data...")
            
            if not self.call_df.empty:
                start_date = self.call_df['date'].min() - timedelta(days=5)
                end_date = self.call_df['date'].max() + timedelta(days=1) # Add one day to ensure last day is included
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
                # --- NEW INSIGHT: Aggregate by both date and intent for later use ---
                self.calls_by_intent = self.call_df.groupby('intent').size().reset_index(name='count')
                daily_calls = self.call_df.groupby('date').size().reset_index(name='call_volume')
            else:
                dates = pd.date_range(start=datetime.now().date() - timedelta(days=30), end=datetime.now().date(), freq='D')
                daily_calls = pd.DataFrame({'date': dates, 'call_volume': 0})

            # Add mail data
            if not self.mail_df.empty and 'volume' in self.mail_df.columns:
                 # --- NEW INSIGHT: Aggregate by both date and type for later use ---
                self.mail_by_type = self.mail_df.groupby('type')['volume'].sum().reset_index()
                daily_mail = self.mail_df.groupby('date')['volume'].sum().reset_index(name='mail_volume')
                combined = pd.merge(daily_calls, daily_mail, on='date', how='outer')
            else:
                combined = daily_calls.copy()
                combined['mail_volume'] = 0
            
            # Create a full date range to ensure no gaps for time-series analysis
            if 'date' in combined.columns and not combined.empty:
                full_date_range = pd.date_range(start=combined['date'].min(), end=combined['date'].max(), freq='D')
                combined = combined.set_index('date').reindex(full_date_range).reset_index().rename(columns={'index': 'date'})

            combined['call_volume'] = combined['call_volume'].fillna(0)
            combined['mail_volume'] = combined['mail_volume'].fillna(0)
            
            # Add financial data
            if not self.financial_df.empty:
                combined = pd.merge(combined, self.financial_df, on='date', how='left')
                financial_cols = self.config['FINANCIAL_DATA'].keys()
                for col in financial_cols:
                    if col in combined.columns:
                        combined[col] = combined[col].ffill().bfill()
            
            # Add derived features
            combined['weekday'] = combined['date'].dt.day_name()
            combined['month'] = combined['date'].dt.month
            combined['is_weekend'] = combined['date'].dt.weekday >= 5
            
            # Sort by date
            combined = combined.sort_values('date').reset_index(drop=True)
            
            self.combined_df = combined
            self.logger.info(f"✅ Combined dataset created: {len(combined):,} records")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Data combination failed: {str(e)}")
            return False

    # --- NEW INSIGHT: Method to calculate lag correlation ---
    def analyze_correlation(self):
        """Calculate the lagged correlation between mail and call volumes."""
        if not SCIPY_AVAILABLE or self.combined_df.empty:
            self.logger.warning("Correlation analysis skipped (SciPy unavailable or no data).")
            return False

        self.logger.info("🔄 Running lag correlation analysis...")
        try:
            correlations = []
            max_lag = self.config['MAX_LAG_DAYS']

            for lag in range(max_lag + 1):
                # Shift call volume back in time by 'lag' days
                self.combined_df[f'call_volume_lag_{lag}'] = self.combined_df['call_volume'].shift(-lag)

                # Correlate current mail volume with future call volume
                temp_df = self.combined_df[['mail_volume', f'call_volume_lag_{lag}']].dropna()

                if len(temp_df) < 2:
                    corr = 0
                else:
                    corr, _ = pearsonr(temp_df['mail_volume'], temp_df[f'call_volume_lag_{lag}'])
                
                correlations.append({'lag_days': lag, 'correlation': corr if np.isfinite(corr) else 0})

            self.correlation_results = pd.DataFrame(correlations)
            self.logger.info("✅ Correlation analysis complete.")
            return True

        except Exception as e:
            self.logger.error(f"❌ Correlation analysis failed: {e}")
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
            self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
            self.app.title = "Marketing Intelligence Dashboard"
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

    # --- NEW METHOD: Main overview chart ---
    def create_overview_figure(self, df):
        if df.empty:
            return self.create_error_figure("No data available for this period")
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add Mail Volume Bar Chart
        fig.add_trace(go.Bar(
            x=df['date'], y=df['mail_volume'], name='Mail Volume',
            marker_color='lightskyblue', opacity=0.7
        ), secondary_y=False)

        # Add Call Volume Line Chart
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['call_volume'], name='Call Volume',
            mode='lines+markers', line=dict(color='navy', width=3)
        ), secondary_y=True)

        fig.update_layout(
            title="Daily Mail vs. Call Volume", template='plotly_white', height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_yaxes(title_text="<b>Mail Volume</b>", secondary_y=False)
        fig.update_yaxes(title_text="<b>Call Volume</b>", secondary_y=True)
        return fig

    # --- NEW METHOD: Correlation analysis chart ---
    def create_correlation_figure(self, df_corr):
        if df_corr.empty:
            return self.create_error_figure("Correlation data not available")

        best_lag = df_corr.loc[df_corr['correlation'].idxmax()]
        colors = ['red' if x == best_lag['lag_days'] else 'navy' for x in df_corr['lag_days']]

        fig = px.bar(df_corr, x='lag_days', y='correlation',
                     title=f"Mail-to-Call Lag Correlation (Peak at Day {int(best_lag['lag_days'])})",
                     labels={'lag_days': 'Lag in Days (Mail Sent vs. Call Received)', 'correlation': 'Pearson Correlation Coefficient'})
        
        fig.update_traces(marker_color=colors)
        fig.update_layout(template='plotly_white', height=400)
        return fig
    
    # --- NEW METHOD: Mail by type chart ---
    def create_mail_type_figure(self, df_mail):
        if df_mail.empty:
            return self.create_error_figure("Mail type data unavailable")

        fig = px.bar(df_mail, x='type', y='volume', color='type',
                     title='Total Mail Volume by Campaign Type',
                     labels={'type': 'Mail Type', 'volume': 'Total Volume'})
        fig.update_layout(template='plotly_white', height=400, showlegend=False)
        return fig

    # --- NEW METHOD: Calls by intent chart ---
    def create_call_intent_figure(self, df_calls):
        if df_calls.empty:
            return self.create_error_figure("Call intent data unavailable")

        fig = px.pie(df_calls, names='intent', values='count',
                     title='Distribution of Call Intents', hole=0.3)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(template='plotly_white', height=400, showlegend=True)
        return fig

    # --- MAJOR OVERHAUL: Layout is now interactive with tabs and date picker ---
    def setup_layout(self):
        """Setup dashboard layout with interactive controls and tabs."""
        df = self.dp.combined_df
        if df.empty:
            self.app.layout = html.Div([html.H1("⚠️ No Data Loaded"), html.P("Dashboard cannot render without data.")])
            return

        start_date = df['date'].min().date()
        end_date = df['date'].max().date()

        self.app.layout = dbc.Container([
            # Header
            dbc.Row(
                dbc.Col(html.H1("📊 Executive Marketing Intelligence Dashboard", className="text-center text-primary, mb-4"), width=12)
            ),
            
            # Interactive Controls
            dbc.Card(dbc.CardBody([
                dbc.Row([
                    dbc.Col(html.H5("Select Date Range"), width="auto", className="me-3"),
                    dbc.Col(dcc.DatePickerRange(
                        id='date-picker-range',
                        min_date_allowed=start_date,
                        max_date_allowed=end_date,
                        start_date=start_date,
                        end_date=end_date,
                        className="w-100"
                    ), width=4)
                ])
            ]), className="mb-4"),

            # KPI Cards
            dbc.Row(id='kpi-cards-row', className="mb-4"),

            # Tabbed Interface for Charts
            dbc.Tabs([
                dbc.Tab(label="📈 Overview", tab_id="tab-overview", children=[
                    dcc.Graph(id='overview-chart')
                ]),
                dbc.Tab(label="🔗 Correlation Analysis", tab_id="tab-correlation", children=[
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='correlation-chart'), width=8),
                        dbc.Col(dbc.Card(id='correlation-kpi', body=True, color="light"), width=4)
                    ], className="mt-4")
                ]),
                dbc.Tab(label="📊 Data Breakdowns", tab_id="tab-breakdowns", children=[
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='mail-type-chart'), width=6),
                        dbc.Col(dcc.Graph(id='call-intent-chart'), width=6)
                    ], className="mt-4")
                ])
            ])

        ], fluid=True)

    # --- MAJOR OVERHAUL: Callbacks are now interactive ---
    def setup_callbacks(self):
        """Setup callbacks for interactive components."""
        
        @self.app.callback(
            [Output('kpi-cards-row', 'children'),
             Output('overview-chart', 'figure')],
            [Input('date-picker-range', 'start_date'),
             Input('date-picker-range', 'end_date')]
        )
        def update_overview_and_kpis(start_date, end_date):
            if not start_date or not end_date:
                return [], self.create_error_figure("Please select a date range.")

            # Filter data based on date picker
            dff = self.dp.combined_df[
                (self.dp.combined_df['date'] >= start_date) & 
                (self.dp.combined_df['date'] <= end_date)
            ]

            if dff.empty:
                return [], self.create_error_figure("No data for selected period.")

            # Calculate KPIs
            total_calls = int(dff['call_volume'].sum())
            total_mail = int(dff['mail_volume'].sum())
            avg_daily_calls = total_calls / len(dff) if len(dff) > 0 else 0
            avg_daily_mail = total_mail / len(dff) if len(dff) > 0 else 0

            kpi_cards = [
                dbc.Col(dbc.Card(dbc.CardBody([html.H4(f"{total_calls:,}", className="card-title"), html.P("Total Calls", className="card-text")])), width=3),
                dbc.Col(dbc.Card(dbc.CardBody([html.H4(f"{total_mail:,}", className="card-title"), html.P("Total Mail", className="card-text")])), width=3),
                dbc.Col(dbc.Card(dbc.CardBody([html.H4(f"{avg_daily_calls:,.1f}", className="card-title"), html.P("Avg Daily Calls", className="card-text")])), width=3),
                dbc.Col(dbc.Card(dbc.CardBody([html.H4(f"{avg_daily_mail:,.1f}", className="card-title"), html.P("Avg Daily Mail", className="card-text")])), width=3),
            ]
            
            overview_fig = self.create_overview_figure(dff)
            
            return kpi_cards, overview_fig

        # Callback for correlation charts
        @self.app.callback(
            [Output('correlation-chart', 'figure'),
             Output('correlation-kpi', 'children')],
            [Input('date-picker-range', 'id')] # Dummy input, this chart uses all data
        )
        def update_correlation_tab(_):
            corr_fig = self.create_correlation_figure(self.dp.correlation_results)

            if not self.dp.correlation_results.empty:
                best_lag = self.dp.correlation_results.loc[self.dp.correlation_results['correlation'].idxmax()]
                corr_kpi = [
                    html.H4("Key Insight", className="card-title"),
                    html.P("Call volume shows the strongest positive relationship with mail sent..."),
                    html.H2(f"{int(best_lag['lag_days'])} days prior", className="text-center text-primary"),
                    html.P(f"with a correlation score of {best_lag['correlation']:.3f}.", className="text-center"),
                ]
            else:
                corr_kpi = html.P("Correlation analysis could not be run.")
            
            return corr_fig, corr_kpi

        # Callback for breakdown charts
        @self.app.callback(
            [Output('mail-type-chart', 'figure'),
             Output('call-intent-chart', 'figure')],
            [Input('date-picker-range', 'id')] # Dummy input, these charts use all data
        )
        def update_breakdown_tab(_):
            mail_fig = self.create_mail_type_figure(self.dp.mail_by_type)
            call_fig = self.create_call_intent_figure(self.dp.calls_by_intent)
            return mail_fig, call_fig

    def run(self, debug=True, port=8050):
        """Run dashboard with error handling."""
        try:
            self.logger.info(f"🚀 Starting dashboard at http://127.0.0.1:{port}")
            self.app.run(debug=debug, port=port, host='127.0.0.1')
        except Exception as e:
            self.logger.error(f"Dashboard failed to start: {str(e)}")

# =============================================================================
# BULLETPROOF MAIN FUNCTION
# =============================================================================

def main():
    """Bulletproof main function with comprehensive error handling."""
    print("🔥 EXECUTIVE MARKETING INTELLIGENCE DASHBOARD")
    print("=" * 60)
    
    logger = setup_logging()

    try:
        if not DASH_AVAILABLE:
            logger.error("❌ Cannot run dashboard without Dash. Exiting.")
            return False
            
        logger.info("🏗️ Initializing robust data processor...")
        dp = SafeDataProcessor(CONFIG, logger)
        
        if not dp.load_data() or (dp.mail_df.empty and dp.call_df.empty):
            logger.warning("⚠️ Real data loading failed or returned empty, creating sample data...")
            if not dp.create_sample_data():
                logger.error("❌ Sample data creation failed. Exiting.")
                return False
        
        if FINANCIAL_AVAILABLE:
            dp.safe_financial_data()
        
        if not dp.combine_data():
            logger.error("❌ Data combination failed. Exiting.")
            return False
        
        # --- NEW STEP: Run the new analysis method ---
        if SCIPY_AVAILABLE:
            dp.analyze_correlation()

        logger.info("🎨 Initializing enhanced dashboard...")
        dashboard = SafeDashboard(dp)
        
        logger.info("=" * 30)
        logger.info("📊 FINAL DATA STATUS:")
        logger.info(f"📧 Mail records processed: {len(dp.mail_df):,}")
        logger.info(f"📞 Call records processed: {len(dp.call_df):,}")
        logger.info(f"🔗 Correlation analysis run: {'✅' if not dp.correlation_results.empty else '❌'}")
        logger.info("=" * 30)
        
        logger.info("🌐 Access dashboard at: http://127.0.0.1:8050")
        logger.info("🛑 Press Ctrl+C to stop")
        dashboard.run(debug=CONFIG.get('DEBUG_MODE', True), port=8050)
        
        return True
        
    except KeyboardInterrupt:
        logger.info("🛑 Dashboard stopped by user")
        return True
    except Exception as e:
        logger.error(f"❌ An unexpected error occurred in the main function: {str(e)}")
        return False

if __name__ == '__main__':
    if main():
        print("\n✅ Dashboard session ended successfully.")
    else:
        print("\n⚠️ Dashboard session ended with errors.")
    sys.exit(0)
