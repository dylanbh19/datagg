#!/usr/bin/env python3
"""
Executive-Level Interactive Dashboard for Marketing & Call Volume Analysis

A professional, multi-page Dash application designed for high-level presentations.
This script performs advanced data augmentation and provides a suite of interactive
visualizations for deep, actionable insights.
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

# --- Data Fetching & Modeling ---
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
    'FINANCIAL_DATA': {'S&P 500': '^GSPC', '10-Yr Treasury Yield': '^TNX'}
}

# =============================================================================
# DATA PROCESSING & ADVANCED AUGMENTATION
# =============================================================================

def load_data(config):
    """Loads and preprocesses mail and call data from CSV files."""
    try:
        mail_df = pd.read_csv(config['MAIL_FILE_PATH'], on_bad_lines='skip')
        mail_df.rename(columns={v: k for k, v in config['MAIL_COLUMNS'].items()}, inplace=True)
        mail_df['date'] = pd.to_datetime(mail_df['date'], errors='coerce').dt.normalize()
        mail_df.dropna(subset=['date', 'volume', 'type'], inplace=True)

        call_df = pd.read_csv(config['CALL_FILE_PATH'], on_bad_lines='skip')
        call_df.rename(columns={v: k for k, v in config['CALL_COLUMNS'].items()}, inplace=True)
        call_df['date'] = pd.to_datetime(call_df['date'], errors='coerce').dt.normalize()
        call_df.dropna(subset=['date', 'intent'], inplace=True)
        
        return mail_df, call_df
    except Exception as e:
        print(f"CRITICAL: Failed to load initial data. Error: {e}")
        sys.exit(1)

def augment_calls_analytically(call_df, mail_df):
    """
    Performs advanced augmentation by creating a synthetic year of call data
    based on existing call patterns and mail seasonality.
    """
    daily_calls = call_df.groupby('date').size().rename('call_volume')
    daily_mail = mail_df.groupby('date')['volume'].sum().rename('mail_volume')
    
    min_mail_date = mail_df['date'].min()
    max_date = max(mail_df['date'].max(), call_df['date'].max())
    full_range = pd.date_range(start=min_mail_date, end=max_date, freq='D')

    # Decompose the existing call data to find its weekly pattern
    existing_calls = daily_calls.reindex(pd.date_range(start=daily_calls.index.min(), end=max_date, freq='D'), fill_value=0)
    decomposition = seasonal_decompose(existing_calls, model='additive', period=7)
    weekly_pattern = decomposition.seasonal
    
    # Use the mail data's trend as a proxy for the call trend in the missing year
    mail_trend = seasonal_decompose(daily_mail.reindex(full_range, fill_value=0), model='additive', period=365).trend
    
    # Create the synthetic data
    synthetic_calls = pd.Series(index=full_range, dtype=float)
    synthetic_calls = synthetic_calls.combine_first(daily_calls) # Add original data
    
    # Apply the patterns to create the synthetic portion
    for date in synthetic_calls[synthetic_calls.isnull()].index:
        base_value = mail_trend[date] / mail_trend.max() * (original_calls_mean := daily_calls.mean())
        seasonal_effect = weekly_pattern.get(date.day_of_week, 0) # Use dayofweek to get pattern
        synthetic_calls[date] = max(0, base_value + seasonal_effect)

    synthetic_df = synthetic_calls[daily_calls.index.min() - pd.DateOffset(days=1):].to_frame('call_volume')
    original_df = daily_calls.to_frame('call_volume')
    
    return synthetic_calls.to_frame('call_volume'), original_df, synthetic_df[synthetic_df.index < original_df.index.min()]


def create_final_df(augmented_calls, mail_df, config):
    """Combines all data into a final dataframe for plotting."""
    daily_mail = mail_df.groupby('date')['volume'].sum().rename('mail_volume')
    final_df = augmented_calls.join(daily_mail).fillna(0)
    
    start, end = final_df.index.min(), final_df.index.max()
    financial_df = yf.download(list(config['FINANCIAL_DATA'].values()), start=start, end=end, progress=False)
    
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
    return final_df

# =============================================================================
# DASHBOARD APPLICATION
# =============================================================================

def build_dashboard(final_df, mail_df, call_df, original_calls, synthetic_calls):
    """Builds and runs the entire multi-page professional Dash application."""
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    app.title = "Marketing & Call Volume Analysis"

    SIDEBAR_STYLE = {"position": "fixed", "top": 0, "left": 0, "bottom": 0, "width": "22rem", "padding": "2rem 1rem", "backgroundColor": "#f8f9fa"}
    CONTENT_STYLE = {"marginLeft": "24rem", "marginRight": "2rem", "padding": "2rem 1rem"}
    
    sidebar = html.Div([
        html.H2("Analysis Dashboard", className="display-4"),
        html.Hr(),
        html.P("A professional dashboard for analyzing marketing and call center data.", className="lead"),
        dbc.Nav([
            dbc.NavLink("Executive Overview", href="/", active="exact"),
            dbc.NavLink("Campaign Deep Dive", href="/page-campaigns", active="exact"),
            dbc.NavLink("Data Augmentation", href="/page-augmentation", active="exact"),
        ], vertical=True, pills=True),
    ], style=SIDEBAR_STYLE)
    
    content = html.Div(id="page-content", style=CONTENT_STYLE)
    app.layout = html.Div([dcc.Location(id="url"), sidebar, content])
    
    # --- Register Callbacks ---
    @app.callback(Output("page-content", "children"), [Input("url", "pathname")])
    def render_page_content(pathname):
        if pathname == "/":
            return build_overview_page(final_df)
        elif pathname == "/page-campaigns":
            return build_campaign_page(final_df, mail_df, call_df)
        elif pathname == "/page-augmentation":
            return build_augmentation_page(final_df, original_calls, synthetic_calls)
        return html.P("404: Page not found")
        
    return app
    
# --- Page Layout Builders ---
def build_overview_page(df):
    """Builds the layout for the main executive overview page."""
    total_calls = df['call_volume'].sum()
    total_mail = df['mail_volume'].sum()
    avg_calls = df['call_volume'].mean()
    
    kpi_card_style = {'padding': '1rem', 'textAlign': 'center', 'borderRadius': '5px', 'backgroundColor': '#e9ecef'}

    return html.Div([
        dbc.Row([
            dbc.Col(html.Div([html.H4("Total Calls"), html.P(f"{total_calls:,.0f}")], style=kpi_card_style)),
            dbc.Col(html.Div([html.H4("Total Mail Sent"), html.P(f"{total_mail:,.0f}")], style=kpi_card_style)),
            dbc.Col(html.Div([html.H4("Avg Daily Calls"), html.P(f"{avg_calls:,.2f}")], style=kpi_card_style)),
        ]),
        dbc.Row(dbc.Col(dcc.Graph(figure=create_overview_figure(df)), width=12), style={'marginTop': '2rem'}),
        dbc.Row(dbc.Col(dcc.Graph(figure=create_correlation_heatmap(df)), width=12), style={'marginTop': '2rem'}),
    ])

def build_campaign_page(df, mail_df, call_df):
    """Builds the layout for the campaign and intent analysis page."""
    return html.Div([
        html.H3("Campaign & Intent Deep Dive"),
        html.P("Select mail and call types to see their weekly trends."),
        dbc.Row([
            dbc.Col(dcc.Dropdown(id='mail-type-dropdown', options=[{'label': i, 'value': i} for i in mail_df['type'].unique()], multi=True, placeholder="Filter by Mail Type...")),
            dbc.Col(dcc.Dropdown(id='intent-dropdown', options=[{'label': i, 'value': i} for i in call_df['intent'].unique()], multi=True, placeholder="Filter by Call Intent...")),
        ]),
        dcc.Graph(id='campaign-graph'),
    ])

def build_augmentation_page(df, original, synthetic):
    """Builds the layout for the data augmentation analysis page."""
    return html.Div([
        html.H3("Data Augmentation Analysis"),
        html.P("The chart below shows the original 6 months of call data and the 12 months of synthetic data created to enable year-over-year analysis."),
        dcc.Graph(figure=create_augmentation_figure(original, synthetic)),
    ])
    
# --- Plotting Functions ---
def create_overview_figure(df):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df.index, y=df['call_volume'], name='Call Volume', line=dict(color='#0d6efd')), secondary_y=False)
    fig.add_trace(go.Bar(x=df.index, y=df['mail_volume'], name='Mail Volume', marker_color='#adb5bd', opacity=0.6), secondary_y=True)
    fig.update_layout(title_text="Overall Mail vs. Call Volume", template='plotly_white')
    fig.update_yaxes(title_text="Call Volume", secondary_y=False)
    fig.update_yaxes(title_text="Mail Volume", secondary_y=True)
    return fig

def create_correlation_heatmap(df):
    monthly_df = df.resample('M').sum()
    corr_matrix = monthly_df[['call_volume', 'mail_volume']].corr()
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Monthly Correlation: Mail vs. Calls")
    return fig

def create_augmentation_figure(original, synthetic):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=original.index, y=original['call_volume'], name='Original Call Data', mode='lines', line=dict(color='#0d6efd')))
    fig.add_trace(go.Scatter(x=synthetic.index, y=synthetic['call_volume'], name='Synthetic Call Data', mode='lines', line=dict(color='#dc3545', dash='dash')))
    fig.update_layout(title_text="Original vs. Synthetic Call Data", template='plotly_white')
    return fig
    
# =============================================================================
# SCRIPT EXECUTION
# =============================================================================
if __name__ == '__main__':
    try:
        import dash_bootstrap_components as dbc
    except ImportError:
        print("Dash Bootstrap Components not found. Please install it: pip install dash-bootstrap-components")
        sys.exit(1)
        
    logger = setup_logging()
    
    mail_data, call_data = load_data(CONFIG)
    augmented_calls, original_calls, synthetic_calls = augment_calls_analytically(call_data, mail_data, logger)
    final_df = create_final_df(augmented_calls, mail_data, CONFIG, logger)
    
    app = build_dashboard(final_df, mail_data, call_data, original_calls, synthetic_calls)

    # This callback needs to be defined in the main scope after the app is created
    @app.callback(
        Output('campaign-graph', 'figure'),
        [Input('mail-type-dropdown', 'value'),
         Input('intent-dropdown', 'value')]
    )
    def update_campaign_graph(selected_mail_types, selected_intents):
        # Filter data based on selection
        mail_dff = mail_data[mail_data['type'].isin(selected_mail_types)] if selected_mail_types else mail_data
        call_dff = call_data[call_data['intent'].isin(selected_intents)] if selected_intents else call_data
        
        # Aggregate to weekly
        mail_agg = mail_dff.groupby(pd.Grouper(key='date', freq='W'))['volume'].sum()
        call_agg = call_dff.groupby(pd.Grouper(key='date', freq='W')).size()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=call_agg.index, y=call_agg.values, name='Call Volume', line=dict(color='#0d6efd')), secondary_y=False)
        fig.add_trace(go.Bar(x=mail_agg.index, y=mail_agg.values, name='Mail Volume', marker_color='#adb5bd', opacity=0.6), secondary_y=True)
        
        fig.update_layout(title_text="Weekly Trends for Selected Mail Types and Call Intents", template='plotly_white')
        fig.update_yaxes(title_text="Weekly Call Count", secondary_y=False)
        fig.update_yaxes(title_text="Weekly Mail Volume", secondary_y=True)
        return fig
    
    logger.info("✓ Dashboard initialized. Starting server at http://127.0.0.1:8050/")
    app.run_server(debug=True)

