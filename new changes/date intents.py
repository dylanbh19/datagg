#!/usr/bin/env python3
"""
Executive Marketing Intelligence Dashboard - Robust Version

Bulletproof dashboard with comprehensive error handling and fallbacks.
NEW: Includes moving averages, weekday pattern analysis, and modeling feature suggestions.
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
    from dash import dcc, html, Input, Output
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
        self.logger = logger or logging.getLogger()
        self.mail_df = pd.DataFrame()
        self.call_df = pd.DataFrame()
        self.financial_df = pd.DataFrame()
        self.combined_df = pd.DataFrame()
        # --- ADDITION: Dataframes for new analyses ---
        self.intent_correlation_matrix = pd.DataFrame()

    def safe_load_csv(self, file_path, description="file"):
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
                except Exception:
                    continue
            self.logger.error(f"❌ Complete failure loading {description} at {file_path}")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"❌ Unexpected error loading {description}: {str(e)}")
            return pd.DataFrame()

    def safe_column_mapping(self, df, column_mapping, data_type):
        if df.empty: return df
        mapped_df = df.copy()
        for standard_name, config_name in column_mapping.items():
            if config_name in mapped_df.columns:
                if config_name != standard_name:
                    mapped_df = mapped_df.rename(columns={config_name: standard_name})
            else:
                self.logger.warning(f"Column '{config_name}' not found in {data_type} data, skipping mapping.")
        return mapped_df

    def safe_date_conversion(self, df, date_column):
        if df.empty or date_column not in df.columns: return df
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df = df.dropna(subset=[date_column])
        df[date_column] = df[date_column].dt.normalize()
        return df

    def load_data(self):
        self.logger.info("🔄 Loading data...")
        self.mail_df = self.safe_load_csv(self.config['MAIL_FILE_PATH'], "Mail data")
        if not self.mail_df.empty:
            self.mail_df = self.safe_column_mapping(self.mail_df, self.config['MAIL_COLUMNS'], 'mail')
            self.mail_df = self.safe_date_conversion(self.mail_df, 'date')
            if 'volume' in self.mail_df.columns:
                self.mail_df['volume'] = pd.to_numeric(self.mail_df['volume'], errors='coerce').fillna(0)

        self.call_df = self.safe_load_csv(self.config['CALL_FILE_PATH'], "Call data")
        if not self.call_df.empty:
            self.call_df = self.safe_column_mapping(self.call_df, self.config['CALL_COLUMNS'], 'call')
            self.call_df = self.safe_date_conversion(self.call_df, 'date')
        
        return not self.mail_df.empty or not self.call_df.empty

    def safe_financial_data(self):
        """Safely attempt to load financial data, skipping individual tickers on failure."""
        if not FINANCIAL_AVAILABLE:
            self.logger.warning("📊 Financial data unavailable - yfinance not installed.")
            return False
        
        self.logger.info("🔄 Attempting to load financial data...")
        # Use a date range from combined_df if available, otherwise from call_df
        if not self.combined_df.empty:
            start_date = self.combined_df['date'].min() - timedelta(days=5)
            end_date = self.combined_df['date'].max() + timedelta(days=1)
        elif not self.call_df.empty:
            start_date = self.call_df['date'].min() - timedelta(days=5)
            end_date = self.call_df['date'].max() + timedelta(days=1)
        else:
            self.logger.warning("Cannot fetch financial data without a date range from mail/call data.")
            return False

        all_financial_data = []
        for name, ticker_str in self.config['FINANCIAL_DATA'].items():
            try:
                ticker_obj = yf.Ticker(ticker_str)
                data = ticker_obj.history(start=start_date, end=end_date)
                if not data.empty and 'Close' in data.columns:
                    series = data['Close'].resample('D').last().rename(name)
                    all_financial_data.append(series)
                    self.logger.info(f"✅ Successfully loaded financial ticker: {name}")
                else:
                    self.logger.warning(f"⚠️ No data returned for financial ticker: {name}")
            except Exception as e:
                self.logger.error(f"❌ Failed to fetch financial ticker {name}: {str(e)}")
        
        if all_financial_data:
            self.financial_df = pd.concat(all_financial_data, axis=1).reset_index()
            self.financial_df['date'] = pd.to_datetime(self.financial_df['date']).dt.normalize()
            self.logger.info(f"✅ Financial data loading complete. {len(all_financial_data)} indicators loaded.")
            return True
        else:
            self.logger.error("❌ No financial tickers could be loaded.")
            return False

    def combine_data(self):
        """Safely combine all data sources and handle financial data merge."""
        self.logger.info("🔄 Combining data sources...")
        if self.call_df.empty and self.mail_df.empty:
            self.logger.error("Both mail and call data are empty. Cannot proceed.")
            return False

        # Aggregate daily data
        daily_calls = self.call_df.groupby('date').size().reset_index(name='call_volume') if not self.call_df.empty else pd.DataFrame(columns=['date', 'call_volume'])
        daily_mail = self.mail_df.groupby('date')['volume'].sum().reset_index(name='mail_volume') if not self.mail_df.empty else pd.DataFrame(columns=['date', 'mail_volume'])

        # Merge mail and call data
        if daily_calls.empty:
            combined = daily_mail
        elif daily_mail.empty:
            combined = daily_calls
        else:
            combined = pd.merge(daily_calls, daily_mail, on='date', how='outer')
        
        if combined.empty:
            self.logger.error("Combined mail and call data is empty.")
            return False

        # Create a full date range to ensure no gaps
        full_date_range = pd.date_range(start=combined['date'].min(), end=combined['date'].max(), freq='D')
        combined = combined.set_index('date').reindex(full_date_range).reset_index().rename(columns={'index': 'date'})
        combined.fillna(0, inplace=True)

        # --- FIX for financial data merge ---
        # Ensure date types are identical before merging
        if not self.financial_df.empty:
            self.logger.info("Merging financial data...")
            combined['date'] = pd.to_datetime(combined['date']).dt.normalize()
            self.financial_df['date'] = pd.to_datetime(self.financial_df['date']).dt.normalize()
            combined = pd.merge(combined, self.financial_df, on='date', how='left')
            
            # Forward-fill and back-fill financial data to cover weekends/holidays
            financial_cols = self.config['FINANCIAL_DATA'].keys()
            for col in financial_cols:
                if col in combined.columns:
                    combined[col] = combined[col].ffill().bfill()

        self.combined_df = combined.sort_values('date').reset_index(drop=True)
        self.logger.info(f"✅ Combined dataset created with {len(self.combined_df)} records.")
        return True

    def analyze_intent_correlation(self):
        """Calculate lagged correlation for each mail_type vs each call_intent."""
        if not SCIPY_AVAILABLE or self.mail_df.empty or self.call_df.empty:
            self.logger.warning("Skipping intent correlation (SciPy, mail, or call data unavailable).")
            return False

        self.logger.info("🔄 Running intent-level correlation analysis...")
        mail_pivot = self.mail_df.pivot_table(index='date', columns='type', values='volume', aggfunc='sum').fillna(0)
        call_pivot = self.call_df.groupby(['date', 'intent']).size().unstack(fill_value=0)
        
        start_date = min(mail_pivot.index.min(), call_pivot.index.min())
        end_date = max(mail_pivot.index.max(), call_pivot.index.max())
        full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        mail_pivot = mail_pivot.reindex(full_date_range, fill_value=0)
        call_pivot = call_pivot.reindex(full_date_range, fill_value=0)
        
        correlation_matrix = pd.DataFrame(index=mail_pivot.columns, columns=call_pivot.columns, dtype=float)
        
        for mail_type in mail_pivot.columns:
            for call_intent in call_pivot.columns:
                correlations = []
                for lag in range(self.config['MAX_LAG_DAYS'] + 1):
                    # Correlate mail_type at day T with call_intent at day T+lag
                    corr, _ = pearsonr(mail_pivot[mail_type], call_pivot[call_intent].shift(-lag).fillna(0))
                    correlations.append(corr if np.isfinite(corr) else 0)
                # Store the peak positive correlation in the lag window
                correlation_matrix.loc[mail_type, call_intent] = max(correlations) if correlations else 0

        self.intent_correlation_matrix = correlation_matrix
        self.logger.info("✅ Intent correlation analysis complete.")
        return True

# =============================================================================
# SAFE DASHBOARD
# =============================================================================

class SafeDashboard:
    def __init__(self, data_processor):
        self.dp = data_processor
        self.logger = data_processor.logger
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.app.title = "Marketing Intelligence Dashboard"
        self.setup_layout()
        self.setup_callbacks()

    def create_error_figure(self, message):
        fig = go.Figure()
        fig.add_annotation(text=f"⚠️ {message}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig.update_layout(title="Chart Unavailable", template='plotly_white')

    def create_overview_figure(self, df):
        """Creates the main overview chart with moving averages."""
        if df.empty: return self.create_error_figure("No data for period")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Calculate moving averages
        df['mail_7d_ma'] = df['mail_volume'].rolling(window=7).mean()
        df['call_7d_ma'] = df['call_volume'].rolling(window=7).mean()

        # Mail series
        fig.add_trace(go.Bar(x=df['date'], y=df['mail_volume'], name='Mail Volume', marker_color='lightskyblue', opacity=0.6), secondary_y=False)
        fig.add_trace(go.Scatter(x=df['date'], y=df['mail_7d_ma'], name='Mail 7D MA', mode='lines', line=dict(color='blue', dash='dash')), secondary_y=False)
        
        # Call series
        fig.add_trace(go.Scatter(x=df['date'], y=df['call_volume'], name='Call Volume', mode='lines', line=dict(color='salmon')), secondary_y=True)
        fig.add_trace(go.Scatter(x=df['date'], y=df['call_7d_ma'], name='Call 7D MA', mode='lines', line=dict(color='red', dash='dash')), secondary_y=True)

        fig.update_layout(title="Daily Mail vs. Call Volume (with 7-Day Moving Average)", template='plotly_white', height=450, legend=dict(orientation="h", y=1.15, xanchor="right", x=1))
        fig.update_yaxes(title_text="Mail Volume", secondary_y=False)
        fig.update_yaxes(title_text="Call Volume", secondary_y=True)
        return fig

    def create_intent_correlation_heatmap(self, df_corr):
        """Creates the heatmap for mail type vs. call intent correlation."""
        if df_corr.empty: return self.create_error_figure("Intent correlation not run")
        fig = px.imshow(df_corr, text_auto=".2f", aspect="auto", color_continuous_scale='Blues', title="Peak Correlation: Mail Type vs. Call Intent")
        return fig.update_layout(template='plotly_white', height=500, xaxis_title="Call Intent", yaxis_title="Mail Type")

    def create_weekday_intent_figure(self):
        """Creates the bar chart for call intents by day of the week."""
        df = self.dp.call_df
        if df.empty: return self.create_error_figure("Call data unavailable")

        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        df['weekday'] = pd.Categorical(df['date'].dt.day_name(), categories=weekday_order, ordered=True)
        
        weekday_counts = df.groupby(['weekday', 'intent']).size().reset_index(name='count')
        
        fig = px.bar(weekday_counts, x='weekday', y='count', color='intent', title="Call Volume by Intent and Day of the Week")
        return fig.update_layout(template='plotly_white', height=500, xaxis_title="Day of the Week", yaxis_title="Number of Calls")

    def get_model_features_markdown(self):
        """Returns a Markdown component with feature suggestions for modeling."""
        return dcc.Markdown("""
            #### Features for a Predictive Model
            To predict daily call volumes, you can construct a model using these feature types:

            **1. Lagged Variables (Historical Context):**
            - `mail_volume_lag_1...28`: Mail volume sent over the past 4 weeks.
            - `call_volume_lag_1, 7, 14`: Call volume from yesterday, last week, and two weeks ago to capture auto-correlation and weekly patterns.
            - Lagged volumes for *specific* mail types (e.g., `promotional_mail_lag_5`).
            
            **2. Time-Based Features (Cyclical Patterns):**
            - `Day of Week`, `Day of Month`, `Month of Year`, `Week of Year`.
            - `Is_Weekend`: A binary flag for Saturday/Sunday.

            **3. Rolling Window Statistics (Recent Trends):**
            - **7-day / 28-day Moving Average** of mail and call volumes.
            - **7-day / 28-day Standard Deviation** to measure recent volatility.

            **4. External Features (Wider Context):**
            - **Holiday Flag**: A binary feature indicating if a day is a public holiday.
            - **Financial Data**: S&P 500 (market sentiment), VIX (volatility).
            
            *A model like **XGBoost** or **LightGBM** is well-suited to handle this variety of features.*
        """, className="p-3 border rounded")

    def setup_layout(self):
        """Sets up the full dashboard layout with multiple tabs."""
        if self.dp.combined_df.empty:
            self.app.layout = dbc.Container([html.H1("⚠️ No Data Loaded")], fluid=True)
            return

        start_date, end_date = self.dp.combined_df['date'].min().date(), self.dp.combined_df['date'].max().date()

        self.app.layout = dbc.Container(fluid=True, children=[
            dbc.Row(dbc.Col(html.H1("📊 Executive Marketing Intelligence Dashboard", className="text-primary text-center my-4"))),
            dbc.Card(dbc.CardBody(dcc.DatePickerRange(id='date-picker-range', min_date_allowed=start_date, max_date_allowed=end_date, start_date=start_date, end_date=end_date)), className="mb-4"),
            dbc.Row(id='kpi-cards-row', className="mb-4"),
            
            dbc.Tabs(id="tabs-main", children=[
                dbc.Tab(label="📈 Overview", tab_id="tab-overview", children=[
                    dbc.Row(dbc.Col(dcc.Graph(id='overview-chart')), className="mt-4")
                ]),
                dbc.Tab(label="🎯 Campaign & Intent Analysis", tab_id="tab-intent-analysis", children=[
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='intent-heatmap'), width=12, lg=7),
                        dbc.Col(dcc.Graph(id='weekday-intent-chart'), width=12, lg=5),
                    ], className="mt-4")
                ]),
                dbc.Tab(label="🤖 Modeling Insights", tab_id="tab-modeling", children=[
                    dbc.Row(dbc.Col(self.get_model_features_markdown(), width=12, lg=8, className="mx-auto"), className="mt-4")
                ]),
            ])
        ])

    def setup_callbacks(self):
        """Sets up the callbacks for all interactive components."""
        @self.app.callback(
            [Output('kpi-cards-row', 'children'), Output('overview-chart', 'figure')],
            [Input('date-picker-range', 'start_date'), Input('date-picker-range', 'end_date')]
        )
        def update_overview_and_kpis(start_date, end_date):
            if not start_date or not end_date:
                return [], self.create_error_figure("Please select a date range.")
            
            dff = self.dp.combined_df[(self.dp.combined_df['date'] >= start_date) & (self.dp.combined_df['date'] <= end_date)]
            if dff.empty: return [], self.create_error_figure("No data for selected period.")
            
            total_calls = int(dff['call_volume'].sum())
            total_mail = int(dff['mail_volume'].sum())
            kpi_cards = [
                dbc.Col(dbc.Card(dbc.CardBody([html.H4(f"{total_calls:,}"), html.P("Total Calls", className="text-muted")]))),
                dbc.Col(dbc.Card(dbc.CardBody([html.H4(f"{total_mail:,}"), html.P("Total Mail", className="text-muted")]))),
            ]
            return kpi_cards, self.create_overview_figure(dff)

        @self.app.callback(
            [Output('intent-heatmap', 'figure'), Output('weekday-intent-chart', 'figure')],
            Input('date-picker-range', 'id') # Dummy input to trigger on load
        )
        def update_analysis_tab(_):
            heatmap_fig = self.create_intent_correlation_heatmap(self.dp.intent_correlation_matrix)
            weekday_fig = self.create_weekday_intent_figure()
            return heatmap_fig, weekday_fig

    def run(self, debug=True, port=8050):
        self.app.run_server(debug=debug, port=port)

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    logger = setup_logging()
    if not DASH_AVAILABLE:
        logger.error("❌ Critical dependency 'dash' is not installed. Exiting.")
        return

    dp = SafeDataProcessor(CONFIG, logger)
    if not dp.load_data():
        logger.error("❌ Data loading failed completely. Exiting.")
        return

    # Run data processing and all analyses
    if not dp.combine_data():
        logger.error("❌ Data combination failed. Exiting.")
        return
    
    dp.safe_financial_data()
    dp.analyze_intent_correlation()
    
    logger.info("🎨 Initializing enhanced dashboard...")
    dashboard = SafeDashboard(dp)
    
    logger.info("=" * 30)
    logger.info("📊 FINAL DATA STATUS:")
    logger.info(f"📧 Mail records processed: {len(dp.mail_df):,}")
    logger.info(f"📞 Call records processed: {len(dp.call_df):,}")
    logger.info(f"💰 Financial data loaded: {'✅' if not dp.financial_df.empty else '❌'}")
    logger.info(f"🔗 Intent correlation run: {'✅' if not dp.intent_correlation_matrix.empty else '❌'}")
    logger.info("=" * 30)
    
    logger.info(f"🌐 Access dashboard at: http://127.0.0.1:8050")
    logger.info("🛑 Press Ctrl+C to stop")
    dashboard.run(debug=CONFIG.get('DEBUG_MODE', True))

if __name__ == '__main__':
    if main():
        print("\n✅ Dashboard session ended successfully.")
    else:
        print("\n⚠️ Dashboard session ended with errors.")
    sys.exit(0)
