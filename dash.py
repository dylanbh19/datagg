#!/usr/bin/env python3
"""
Executive-Grade Interactive Dashboard for Advanced Marketing & Call Volume EDA

This script creates a professional, multi-tab Dash application suitable for
high-level presentations. It performs advanced data augmentation, feature
engineering, and provides a suite of sophisticated interactive visualizations
for deep, actionable insights.
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
    'MAX_LAG_DAYS': 21 # For lag correlation analysis
}

# =============================================================================
# ENHANCED LOGGING SETUP
# =============================================================================

class CustomFormatter(logging.Formatter):
    """Custom logger formatter with colors for terminal output."""
    grey, yellow, red, bold_red, green, reset = "\x1b[38;20m", "\x1b[33;20m", "\x1b[31;20m", "\x1b[31;1m", "\x1b[32;20m", "\x1b[0m"
    format_str = "%(asctime)s - %(levelname)-8s - %(message)s"
    FORMATS = { logging.INFO: green + format_str + reset, logging.WARNING: yellow + format_str + reset, logging.ERROR: red + format_str + reset, logging.CRITICAL: bold_red + format_str + reset }
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.grey + self.format_str + self.reset)
        return logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S').format(record)

def setup_logging():
    """Sets up a colorful terminal logger."""
    logger = logging.getLogger('DashboardAnalysis')
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): logger.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    return logger

# =============================================================================
# DATA PROCESSING & FEATURE ENGINEERING
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
    """Performs data augmentation and combines all data sources."""
    logger.info("STEP 2: Augmenting and combining data sources...")
    daily_calls = call_df.groupby('date').size().rename('call_volume')
    daily_mail = mail_df.groupby('date')['volume'].sum().rename('mail_volume')
    
    min_date = min(mail_df['date'].min(), call_df['date'].min())
    max_date = max(mail_df['date'].max(), call_df['date'].max())
    full_range = pd.date_range(start=min_date, end=max_date, freq='D')
    
    augmented_calls = daily_calls.reindex(full_range)
    original_call_data = augmented_calls.dropna().copy()
    
    if augmented_calls.isnull().sum() > 0:
        logger.warning("Creating synthetic call data for missing periods...")
        call_trend = seasonal_decompose(original_call_data, model='additive', period=365).trend.fillna(0)
        call_weekly_pattern = seasonal_decompose(original_call_data, model='additive', period=7).seasonal
        
        for date in augmented_calls[augmented_calls.isnull()].index:
            base_value = call_trend.get(date - pd.DateOffset(years=1), 0)
            seasonal_effect = call_weekly_pattern.get(date, 0)
            augmented_calls[date] = max(0, base_value + seasonal_effect)

    final_df = augmented_calls.to_frame().join(daily_mail).fillna(0)
    
    financial_df = yf.download(list(config['FINANCIAL_DATA'].values()), start=min_date, end=max_date, progress=False)
    if not financial_df.empty:
        processed_financial = pd.DataFrame(index=financial_df.index)
        for name, ticker in config['FINANCIAL_DATA'].items():
            for price_type in ['Adj Close', 'Close']:
                if (price_type, ticker) in financial_df.columns:
                    processed_financial[name] = financial_df[(price_type, ticker)]
                    break
        final_df = final_df.join(processed_financial)
        final_df.ffill(inplace=True).bfill(inplace=True)
    
    final_df.fillna(0, inplace=True)
    logger.info("✓ Data augmentation and combination complete.")
    return final_df

def feature_engineering(df, logger):
    """Creates a rich set of features for deeper analysis."""
    logger.info("STEP 3: Engineering advanced features...")
    df['day_of_week'] = df.index.day_name()
    df['month'] = df.index.month_name()
    df['week_of_year'] = df.index.isocalendar().week
    df['quarter'] = df.index.quarter
    df['call_volume_roll_7d'] = df['call_volume'].rolling(window=7).mean()
    df['mail_volume_roll_7d'] = df['mail_volume'].rolling(window=7).mean()
    
    for i in [1, 3, 7, 14]:
        df[f'mail_lag_{i}d'] = df['mail_volume'].shift(i)
        
    df.fillna(0, inplace=True)
    logger.info("✓ Feature engineering complete.")
    return df

# =============================================================================
# DASHBOARD APPLICATION & PLOTTING
# =============================================================================

def build_dashboard(df, mail_df, call_df):
    """Builds and runs the entire multi-page professional Dash application."""
    logger = setup_logging()
    logger.info("STEP 4: Initializing executive dashboard...")
    
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX], suppress_callback_exceptions=True)
    app.title = "Marketing & Call Volume Analysis"

    sidebar = html.Div([
        html.H3("Marketing Analysis", className="display-6"),
        html.Hr(),
        html.P("An interactive EDA tool for strategic insights.", className="lead", style={'fontSize': '1rem'}),
        dbc.Nav([
            dbc.NavLink("Executive Overview", href="/", active="exact"),
            dbc.NavLink("Campaign Deep Dive", href="/campaigns", active="exact"),
            dbc.NavLink("Seasonality & Trends", href="/seasonality", active="exact"),
            dbc.NavLink("Lag Correlation Analysis", href="/lags", active="exact"),
        ], vertical=True, pills=True),
    ], style={"position": "fixed", "top": 0, "left": 0, "bottom": 0, "width": "20rem", "padding": "2rem 1rem", "backgroundColor": "#f8f9fa"})
    
    content = html.Div(id="page-content", style={"marginLeft": "22rem", "marginRight": "2rem", "padding": "2rem 1rem"})
    app.layout = html.Div([dcc.Location(id="url"), sidebar, content])
    
    register_callbacks(app, df, mail_df, call_df)
    
    logger.info("✓ Dashboard initialized. Starting server at http://127.0.0.1:8050/")
    return app

def register_callbacks(app, df, mail_df, call_df):
    """Registers all callbacks for the Dash application."""
    @app.callback(Output("page-content", "children"), Input("url", "pathname"))
    def render_page_content(pathname):
        if pathname == "/":
            return build_overview_page(df)
        elif pathname == "/campaigns":
            return build_campaign_page(mail_df, call_df)
        elif pathname == "/seasonality":
            return build_seasonality_page(df)
        elif pathname == "/lags":
            return build_lag_page(df)
        return dbc.Container([
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised...")
        ])

    @app.callback(Output('campaign-graph', 'figure'), [Input('mail-type-dropdown', 'value'), Input('intent-dropdown', 'value')])
    def update_campaign_graph(selected_mail_types, selected_intents):
        mail_dff = mail_df[mail_df['type'].isin(selected_mail_types)] if selected_mail_types else mail_df
        call_dff = call_df[call_df['intent'].isin(selected_intents)] if selected_intents else call_df
        mail_agg = mail_dff.groupby(pd.Grouper(key='date', freq='W'))['volume'].sum()
        call_agg = call_dff.groupby(pd.Grouper(key='date', freq='W')).size()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=call_agg.index, y=call_agg.values, name='Call Volume', line=dict(color='#0d6efd')), secondary_y=False)
        fig.add_trace(go.Bar(x=mail_agg.index, y=mail_agg.values, name='Mail Volume', marker_color='#adb5bd', opacity=0.6), secondary_y=True)
        fig.update_layout(title_text="Weekly Trends for Selected Mail Types and Call Intents", template='plotly_white')
        return fig

# --- Page Layout & Figure Builders ---
def build_overview_page(df):
    kpi_card = lambda title, value: dbc.Card(dbc.CardBody([html.H4(title), html.P(f"{value}", className="card-text", style={'fontSize': '1.5rem', 'fontWeight': 'bold'})]))
    return dbc.Container([
        dbc.Row([
            dbc.Col(kpi_card("Total Calls (Augmented)", f"{df['call_volume'].sum():,.0f}"), width=4),
            dbc.Col(kpi_card("Total Mail Sent", f"{df['mail_volume'].sum():,.0f}"), width=4),
            dbc.Col(kpi_card("Busiest Call Day", df.groupby('day_of_week')['call_volume'].sum().idxmax()), width=4),
        ]),
        dbc.Row(dbc.Col(dcc.Graph(figure=px.line(df, y=['call_volume_roll_7d', 'mail_volume_roll_7d'], title="7-Day Rolling Average: Mail vs. Calls").update_layout(template='plotly_white')), width=12), style={'marginTop': '2rem'}),
        dbc.Row(dbc.Col(dcc.Graph(figure=px.imshow(df[['call_volume', 'mail_volume'] + list(CONFIG['FINANCIAL_DATA'].keys())].corr(), text_auto=True, title="Correlation Matrix").update_layout(template='plotly_white')), width=12), style={'marginTop': '2rem'}),
    ], fluid=True)

def build_campaign_page(mail_df, call_df):
    return dbc.Container([
        dbc.Row(dbc.Col(html.H3("Campaign & Intent Deep Dive"))),
        dbc.Row([
            dbc.Col(dcc.Dropdown(id='mail-type-dropdown', options=[{'label': i, 'value': i} for i in sorted(mail_df['type'].unique())], multi=True, placeholder="Filter by Mail Type...")),
            dbc.Col(dcc.Dropdown(id='intent-dropdown', options=[{'label': i, 'value': i} for i in sorted(call_df['intent'].unique())], multi=True, placeholder="Filter by Call Intent...")),
        ]),
        dbc.Row(dbc.Col(dcc.Graph(id='campaign-graph'), style={'marginTop': '1rem'})),
    ], fluid=True)

def build_seasonality_page(df):
    res_calls = seasonal_decompose(df['call_volume'], model='additive', period=365)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=("Observed", "Trend", "Seasonal", "Residual"))
    fig.add_trace(go.Scatter(x=df.index, y=res_calls.observed, name='Observed'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=res_calls.trend, name='Trend'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=res_calls.seasonal, name='Seasonal'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=res_calls.resid, name='Residual', mode='markers'), row=4, col=1)
    fig.update_layout(height=800, title_text="Call Volume Seasonality Decomposition", template='plotly_white')
    return dbc.Container([
        dbc.Row(dbc.Col(html.H3("Seasonality & Trend Analysis"))),
        dbc.Row(dbc.Col(dcc.Graph(figure=fig))),
    ], fluid=True)

def build_lag_page(df):
    corrs = [df['call_volume'].corr(df['mail_volume'].shift(lag)) for lag in range(CONFIG['MAX_LAG_DAYS'] + 1)]
    fig = px.bar(x=list(range(CONFIG['MAX_LAG_DAYS'] + 1)), y=corrs, labels={'x': 'Lag (Days)', 'y': 'Correlation Coefficient'}, title="Mail Volume Lag vs. Call Volume Correlation")
    fig.update_layout(template='plotly_white')
    return dbc.Container([
        dbc.Row(dbc.Col(html.H3("Lag Correlation Analysis"))),
        dbc.Row(dbc.Col(dcc.Graph(figure=fig))),
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
