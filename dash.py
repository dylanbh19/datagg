#!/usr/bin/env python3
"""
Executive-Grade Interactive Dashboard for Advanced Marketing & Call Volume EDA

This script creates a professional, multi-page Dash application suitable for
high-level presentations. It performs advanced, adaptive data augmentation and
provides a suite of sophisticated interactive visualizations for deep,
actionable insights.
"""
# --- Core Libraries ---
import pandas as pd
import numpy as np
import warnings
import os
import sys
from datetime import datetime
import logging

# --- Visualization & Dashboarding ---
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

# --- Data Fetching & Statistical Analysis ---
import yfinance as yf
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    'MAIL_FILE_PATH': r'merged_output.csv',
    'CALL_FILE_PATH': r'data\GenesysExtract_20250609.csv',
    'MAIL_COLUMNS': {'date': 'mail date', 'volume': 'mail_volume', 'type': 'mail_type'},
    'CALL_COLUMNS': {'date': 'ConversationStart', 'intent': 'vui_intent'},
    'FINANCIAL_DATA': {'S&P 500': '^GSPC', '10-Yr Treasury Yield': '^TNX'},
    'MAX_LAG_DAYS': 28
}

# =============================================================================
# SETUP
# =============================================================================

def setup_logging():
    """Sets up a clean logger for terminal feedback."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
    return logging.getLogger('DashboardAnalysis')

# =============================================================================
# DATA PROCESSING & ADVANCED AUGMENTATION
# =============================================================================

def load_data(config, logger):
    """Loads and preprocesses mail and call data from CSV files."""
    try:
        logger.info("STEP 1: Loading and processing source data...")
        mail_df = pd.read_csv(config['MAIL_FILE_PATH'], on_bad_lines='skip')
        mail_df.rename(columns={v: k for k, v in config['MAIL_COLUMNS'].items()}, inplace=True)
        mail_df['date'] = pd.to_datetime(mail_df['date'], errors='coerce').dt.normalize()
        mail_df.dropna(subset=['date', 'volume', 'type'], inplace=True)

        call_df = pd.read_csv(config['CALL_FILE_PATH'], on_bad_lines='skip')
        call_df.rename(columns={v: k for k, v in config['CALL_COLUMNS'].items()}, inplace=True)
        call_df['date'] = pd.to_datetime(call_df['date'], errors='coerce').dt.normalize()
        call_df.dropna(subset=['date', 'intent'], inplace=True)
        
        logger.info("✓ Data loaded successfully.")
        return mail_df, call_df
    except Exception as e:
        logger.critical(f"Failed to load initial data. Aborting. Error: {e}")
        sys.exit(1)

def augment_and_combine_data(call_df, mail_df, config, logger):
    """Performs adaptive data augmentation and combines all data sources."""
    logger.info("STEP 2: Augmenting data and combining sources...")
    daily_calls = call_df.groupby('date').size().rename('call_volume')
    daily_mail = mail_df.groupby('date')['volume'].sum().rename('mail_volume')
    
    # Determine the full date range needed
    min_date = min(mail_df['date'].min(), call_df['date'].min())
    max_date = max(mail_df['date'].max(), call_df['date'].max())
    full_range = pd.date_range(start=min_date, end=max_date, freq='D')

    # Reindex calls to the full range, creating NaNs where data is missing
    augmented_calls = daily_calls.reindex(full_range)
    original_calls_for_analysis = augmented_calls.dropna().copy()
    
    # --- Adaptive Seasonality ---
    # Check if there's enough data for a yearly decomposition, otherwise default to weekly
    if len(original_calls_for_analysis) >= 730:
        logger.info("Sufficient data found for yearly seasonality decomposition.")
        period = 365
    else:
        logger.warning("Insufficient data for yearly seasonality. Decomposing weekly pattern instead.")
        period = 7
    
    call_decomposition = seasonal_decompose(original_calls_for_analysis.asfreq('D', fill_value=0), model='additive', period=period)
    call_seasonal_pattern = call_decomposition.seasonal
    
    # Use mail data trend as a proxy for the missing period's trend
    mail_trend = seasonal_decompose(daily_mail.reindex(full_range).fillna(0), model='additive', period=365).trend
    
    # Fill missing values using the discovered patterns
    missing_dates = augmented_calls[augmented_calls.isnull()].index
    if not missing_dates.empty:
        logger.warning(f"Creating synthetic call data for {len(missing_dates)} days...")
        for date in missing_dates:
            base_value = mail_trend.get(date, original_calls_for_analysis.mean()) / mail_trend.max() * original_calls_for_analysis.mean()
            seasonal_effect = call_seasonal_pattern.get(date.to_period('D').to_timestamp() - pd.DateOffset(years=1) if period == 365 else date, 0)
            augmented_calls[date] = max(0, base_value + seasonal_effect)

    augmented_calls.fillna(0, inplace=True)
    
    # Combine with other data sources
    final_df = augmented_calls.to_frame().join(daily_mail).fillna(0)
    
    # Fetch Financial Data
    financial_df = yf.download(list(config['FINANCIAL_DATA'].values()), start=min_date, end=max_date, progress=False)
    if not financial_df.empty:
        for name, ticker in config['FINANCIAL_DATA'].items():
            for price_type in ['Adj Close', 'Close']:
                if (price_type, ticker) in financial_df.columns:
                    final_df[name] = financial_df[(price_type, ticker)]
                    break
    
    final_df.ffill(inplace=True).bfill(inplace=True).fillna(0, inplace=True)
    logger.info("✓ Data augmentation and combination complete.")
    return final_df

# =============================================================================
# DASHBOARD APPLICATION & PLOTTING
# =============================================================================

def build_dashboard(df, mail_df, call_df):
    """Builds and runs the entire multi-page professional Dash application."""
    logger.info("STEP 3: Initializing executive dashboard...")
    
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SPACELAB], suppress_callback_exceptions=True)
    app.title = "Marketing & Call Volume Intelligence"

    # --- Reusable Components ---
    def create_kpi_card(title, value, change):
        return dbc.Card(dbc.CardBody([
            html.H6(title, className="card-title text-muted"),
            html.H3(f"{value}", className="card-text"),
            html.P(f"{change}", className=f"card-text text-{'success' if change.startswith('+') else 'danger'}")
        ]))

    # --- Sidebar Layout ---
    sidebar = html.Div([
        html.H2("Marketing Intelligence", className="display-6"), html.Hr(),
        html.P("An interactive EDA tool for strategic insights.", className="lead", style={'fontSize': '0.9rem'}),
        dbc.Nav([
            dbc.NavLink("Executive Overview", href="/", active="exact"),
            dbc.NavLink("Campaign & Intent Deep Dive", href="/campaigns", active="exact"),
            dbc.NavLink("Data Diagnostics & Health", href="/diagnostics", active="exact"),
        ], vertical=True, pills=True),
    ], style={"position": "fixed", "top": 0, "left": 0, "bottom": 0, "width": "20rem", "padding": "2rem 1rem", "backgroundColor": "#f8f9fa"})
    
    content = html.Div(id="page-content", style={"marginLeft": "22rem", "marginRight": "2rem", "padding": "2rem 1rem"})
    app.layout = html.Div([dcc.Location(id="url"), sidebar, content])
    
    # --- Register Callbacks ---
    @app.callback(Output("page-content", "children"), Input("url", "pathname"))
    def render_page_content(pathname):
        if pathname == "/": return build_overview_page(df)
        if pathname == "/campaigns": return build_campaign_page(mail_df, call_df)
        if pathname == "/diagnostics": return build_diagnostics_page(df, mail_df, call_df)
        return html.H1("404: Page Not Found", className="text-danger")

    @app.callback(Output('campaign-graph', 'figure'), [Input('mail-type-dropdown', 'value'), Input('intent-dropdown', 'value')])
    def update_campaign_graph(selected_mail_types, selected_intents):
        mail_dff = mail_df[mail_df['type'].isin(selected_mail_types)] if selected_mail_types else mail_df
        call_dff = call_df[call_df['intent'].isin(selected_intents)] if selected_intents else call_df
        mail_agg = mail_dff.groupby(pd.Grouper(key='date', freq='W'))['volume'].sum()
        call_agg = call_dff.groupby(pd.Grouper(key='date', freq='W')).size()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=call_agg.index, y=call_agg.values, name='Call Volume', line=dict(color='#0d6efd', width=3)), secondary_y=False)
        fig.add_trace(go.Bar(x=mail_agg.index, y=mail_agg.values, name='Mail Volume', marker_color='#adb5bd', opacity=0.7), secondary_y=True)
        fig.update_layout(title_text="Weekly Trends for Selected Segments", template='plotly_white', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        return fig
    
    logger.info("✓ Dashboard initialized. Starting server at http://127.0.0.1:8050/")
    return app

# --- Page & Figure Builders ---
def build_overview_page(df):
    """Builds the layout for the main executive overview page."""
    # KPI Calculations
    last_30_days = df.iloc[-30:]
    prev_30_days = df.iloc[-60:-30]
    call_change = (last_30_days['call_volume'].sum() - prev_30_days['call_volume'].sum()) / prev_30_days['call_volume'].sum() * 100 if prev_30_days['call_volume'].sum() > 0 else 0
    mail_change = (last_30_days['mail_volume'].sum() - prev_30_days['mail_volume'].sum()) / prev_30_days['mail_volume'].sum() * 100 if prev_30_days['mail_volume'].sum() > 0 else 0
    
    # Figures
    fig_overview = make_subplots(specs=[[{"secondary_y": True}]])
    fig_overview.add_trace(go.Scatter(x=df.index, y=df['call_volume'], name='Call Volume', line=dict(color='#0d6efd')), secondary_y=False)
    fig_overview.add_trace(go.Bar(x=df.index, y=df['mail_volume'], name='Mail Volume', marker_color='#adb5bd', opacity=0.6), secondary_y=True)
    fig_overview.update_layout(title_text="Overall Mail vs. Call Volume Timeline", template='plotly_white')

    rolling_corr = df['mail_volume'].rolling(window=30).corr(df['call_volume'])
    fig_corr = px.line(rolling_corr, title="30-Day Rolling Correlation Between Mail and Calls", labels={'value': 'Correlation Coefficient'})
    fig_corr.update_layout(template='plotly_white', showlegend=False)

    return dbc.Container([
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([html.H6("Total Calls (Augmented)"), html.H3(f"{df['call_volume'].sum():,.0f}")]))),
            dbc.Col(dbc.Card(dbc.CardBody([html.H6("Total Mail Sent"), html.H3(f"{df['mail_volume'].sum():,.0f}")]))),
            dbc.Col(dbc.Card(dbc.CardBody([html.H6("Call Change (Last 30d)"), html.H3(f"{call_change:+.2f}%", className=f"text-{'success' if call_change >= 0 else 'danger'}")]))),
            dbc.Col(dbc.Card(dbc.CardBody([html.H6("Mail Change (Last 30d)"), html.H3(f"{mail_change:+.2f}%", className=f"text-{'success' if mail_change >= 0 else 'danger'}")]))),
        ]),
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_overview), width=12)], className="mt-4"),
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_corr), width=12)], className="mt-4"),
    ], fluid=True)

def build_campaign_page(mail_df, call_df):
    return dbc.Container([
        dbc.Row(dbc.Col(html.H3("Campaign & Intent Deep Dive"))),
        dbc.Row([
            dbc.Col(dcc.Dropdown(id='mail-type-dropdown', options=[{'label': i, 'value': i} for i in sorted(mail_df['type'].unique())], multi=True, placeholder="Filter by Mail Type...")),
            dbc.Col(dcc.Dropdown(id='intent-dropdown', options=[{'label': i, 'value': i} for i in sorted(call_df['intent'].unique())], multi=True, placeholder="Filter by Call Intent...")),
        ]),
        dbc.Row(dbc.Col(dcc.Graph(id='campaign-graph'), className="mt-3")),
    ], fluid=True)

def build_diagnostics_page(df, mail_df, call_df):
    return dbc.Container([
        dbc.Row(dbc.Col(html.H3("Data Health & Diagnostics"))),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([html.H5("Mail Data"), html.P(f"Date Range: {mail_df['date'].min().date()} to {mail_df['date'].max().date()}"), html.P(f"{len(mail_df['type'].unique())} Unique Mail Types")]))) ,
            dbc.Col(dbc.Card(dbc.CardBody([html.H5("Call Data"), html.P(f"Date Range: {call_df['date'].min().date()} to {call_df['date'].max().date()}"), html.P(f"{len(call_df['intent'].unique())} Unique Call Intents")]))),
        ]),
        dbc.Row(dbc.Col(dcc.Graph(figure=px.imshow(df.corr(), text_auto=".2f", title="Full Correlation Matrix of All Variables").update_layout(template='plotly_white')), className="mt-4")),
    ], fluid=True)
    
# =============================================================================
# SCRIPT EXECUTION
# =============================================================================
if __name__ == '__main__':
    logger = setup_logging()
    
    mail_data, call_data = load_data(CONFIG, logger)
    augmented_df = augment_and_combine_data(call_data, mail_data, CONFIG, logger)
    final_df = feature_engineering(augmented_df, logger)
    
    app = build_dashboard(final_df, mail_data, call_data)
    app.run_server(debug=True)

