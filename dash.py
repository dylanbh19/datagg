#!/usr/bin/env python3
"""
Professional Interactive Dashboard for Marketing & Financial Analysis

This script creates a multi-tab, interactive Dash application suitable for
high-level presentations. It performs advanced data augmentation to create a
synthetic year of call data and provides clear, filterable visualizations for
deep exploratory data analysis.
"""

# --- Core Libraries ---
import pandas as pd
import numpy as np
import warnings
import os
import sys
import traceback
from datetime import datetime
import logging

# --- Visualization & Dashboarding ---
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output

# --- Data Fetching ---
import yfinance as yf

# Suppress common warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# ENHANCED LOGGING SETUP
# =============================================================================

class CustomFormatter(logging.Formatter):
    """Custom logger formatter with colors for terminal output."""
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    green = "\x1b[32;20m"
    reset = "\x1b[0m"
    format_str = "%(asctime)s - %(levelname)-8s - %(message)s"

    FORMATS = {
        logging.INFO: green + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.grey + self.format_str + self.reset)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

def setup_logging():
    """Sets up a colorful terminal logger."""
    logger = logging.getLogger('DashboardAnalysis')
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    ch = logging.StreamHandler()
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    return logger

# =============================================================================
# CONFIGURATION - Your settings are saved here
# =============================================================================
CONFIG = {
    'MAIL_FILE_PATH': r'merged_output.csv',
    'CALL_FILE_PATH': r'data\GenesysExtract_20250609.csv',
    'MAIL_COLUMNS': {'date': 'mail date', 'volume': 'mail_volume', 'type': 'mail_type'},
    'CALL_COLUMNS': {'date': 'ConversationStart', 'intent': 'vui_intent'},
    'FINANCIAL_DATA': {'S&P 500': '^GSPC', '10-Yr Treasury Yield': '^TNX'}
}

# =============================================================================
# DATA PROCESSING & AUGMENTATION
# =============================================================================

def load_and_process_data(config, logger):
    """Loads, processes, and combines all data sources."""
    try:
        logger.info("STEP 1: Loading and processing data...")
        # Load and process mail data
        mail_df = pd.read_csv(config['MAIL_FILE_PATH'], on_bad_lines='skip')
        mail_df.rename(columns={v: k for k, v in config['MAIL_COLUMNS'].items()}, inplace=True)
        mail_df['date'] = pd.to_datetime(mail_df['date'], errors='coerce').dt.normalize()
        mail_df.dropna(subset=['date'], inplace=True)

        # Load and process call data
        call_df = pd.read_csv(config['CALL_FILE_PATH'], on_bad_lines='skip')
        call_df.rename(columns={v: k for k, v in config['CALL_COLUMNS'].items()}, inplace=True)
        call_df['date'] = pd.to_datetime(call_df['date'], errors='coerce').dt.normalize()
        call_df.dropna(subset=['date'], inplace=True)
        
        logger.info("✓ Data loaded successfully.")
        return mail_df, call_df
    except Exception as e:
        logger.critical(f"Failed to load initial data. Aborting. Error: {e}")
        sys.exit(1)

def augment_data_intelligently(call_df, mail_df, logger):
    """Performs advanced data augmentation by creating a synthetic year."""
    logger.info("STEP 2: Augmenting call data with synthetic year...")
    
    daily_calls = call_df.groupby('date').size().rename('call_volume')
    min_mail_date = mail_df['date'].min()
    max_date = max(mail_df['date'].max(), call_df['date'].max())
    full_date_range = pd.date_range(start=min_mail_date, end=max_date, freq='D')
    augmented_calls = daily_calls.reindex(full_date_range)
    
    original_call_data = augmented_calls.dropna()
    synthetic_dates = augmented_calls[augmented_calls.isnull()].index
    
    if len(synthetic_dates) < 30:
        logger.info("No significant gap found for synthetic data generation.")
        augmented_calls.fillna(0, inplace=True)
        return augmented_calls.to_frame(), augmented_calls.to_frame().iloc[0:0]

    logger.warning(f"Creating synthetic call data for {len(synthetic_dates)} days...")
    
    synthetic_data = original_call_data.copy()
    try:
        synthetic_data.index = synthetic_data.index - pd.DateOffset(years=1)
    except Exception as e:
        logger.error(f"Could not apply date offset for synthetic data generation. Error: {e}")
        synthetic_data.index = synthetic_data.index - pd.Timedelta(days=365) # Fallback
        
    augmented_calls.fillna(synthetic_data, inplace=True)
    augmented_calls.fillna(0, inplace=True)
    
    synthetic_df = augmented_calls.loc[synthetic_dates].to_frame()
    
    logger.info("✓ Synthetic data generation complete.")
    return augmented_calls.to_frame(), synthetic_df

def create_final_dataframe(augmented_calls, mail_df, config, logger):
    """Combines all data sources into a final dataframe for plotting."""
    logger.info("STEP 3: Combining all data sources...")
    
    daily_mail = mail_df.groupby('date')['volume'].sum().rename('mail_volume')
    final_df = augmented_calls.join(daily_mail).fillna(0)
    
    start, end = final_df.index.min(), final_df.index.max()
    tickers = config['FINANCIAL_DATA']
    financial_df = yf.download(list(tickers.values()), start=start, end=end, progress=False)
    
    if not financial_df.empty:
        processed_financial = pd.DataFrame(index=financial_df.index)
        for name, ticker in tickers.items():
            for price_type in ['Adj Close', 'Close']:
                if (price_type, ticker) in financial_df.columns:
                    processed_financial[name] = financial_df[(price_type, ticker)]
                    break
        
        # --- START OF FIX ---
        # The join operation is now assigned to a new variable to ensure final_df is not overwritten with None.
        combined_df = final_df.join(processed_financial)
        if combined_df is not None:
             final_df = combined_df
             # Fill forward first, then backward to handle weekends and holidays robustly
             final_df.ffill(inplace=True)
             final_df.bfill(inplace=True)
        else:
            logger.warning("Joining financial data failed. Proceeding without it.")
        # --- END OF FIX ---
    else:
        logger.warning("Could not download financial data.")

    # Final check for any remaining NaN values
    final_df.fillna(0, inplace=True)
    logger.info("✓ Final dataframe for dashboard created.")
    return final_df

# =============================================================================
# DASHBOARD APPLICATION
# =============================================================================

def create_dashboard(final_df, mail_df, call_df, original_calls, synthetic_calls):
    """Initializes and runs the multi-tab Dash application."""
    logger = setup_logging()
    logger.info("STEP 4: Initializing professional dashboard...")
    
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    app.title = "Marketing & Call Volume Analysis"

    app.layout = html.Div(style={'backgroundColor': '#f9f9f9', 'fontFamily': 'Arial, sans-serif'}, children=[
        html.Div(style={'backgroundColor': '#1A2E44', 'padding': '20px', 'color': 'white'}, children=[
            html.H1("Marketing & Call Volume Analysis Dashboard", style={'textAlign': 'center', 'margin': '0'}),
            html.P("An interactive overview for strategic decision-making", style={'textAlign': 'center', 'margin': '0'})
        ]),
        dcc.Tabs(id="tabs-main", value='tab-overview', children=[
            dcc.Tab(label='High-Level Overview', value='tab-overview'),
            dcc.Tab(label='Campaign & Intent Analysis', value='tab-breakdown'),
            dcc.Tab(label='Data Augmentation Deep Dive', value='tab-augmentation'),
            dcc.Tab(label='Normalized Market Trends', value='tab-trends'),
        ], style={'height': '44px'}, colors={"border": "white", "primary": "#1A2E44", "background": "#f0f0f0"}),
        html.Div(id='tabs-content', style={'padding': '20px'})
    ])

    @app.callback(Output('tabs-content', 'children'), Input('tabs-main', 'value'))
    def render_content(tab):
        if tab == 'tab-overview':
            return overview_tab_layout(final_df)
        elif tab == 'tab-breakdown':
            return breakdown_tab_layout(mail_df, call_df)
        elif tab == 'tab-augmentation':
            return augmentation_tab_layout(original_calls, synthetic_calls)
        elif tab == 'tab-trends':
            return trends_tab_layout(final_df)

    def overview_tab_layout(df):
        min_date, max_date = df.index.min().date(), df.index.max().date()
        return html.Div([
            html.H3("Overall Volume Trends", style={'textAlign': 'center'}),
            dcc.Graph(id='overview-graph'),
            dcc.RangeSlider(
                id='overview-date-slider',
                min=0, max=len(df.index) - 1,
                value=[0, len(df.index) - 1],
                marks={i: date.strftime('%Y-%m') for i, date in enumerate(df.index) if date.day == 1 and date.month % 3 == 1},
                step=7, tooltip={"placement": "bottom", "always_visible": True}
            )
        ])

    def breakdown_tab_layout(mail_data, call_data):
        return html.Div([
            html.H3("Campaign & Intent Breakdown", style={'textAlign': 'center'}),
            html.Div(style={'display': 'flex', 'justifyContent': 'center', 'gap': '30px', 'padding': '10px'}, children=[
                dcc.Dropdown(id='mail-type-dropdown', options=[{'label': i, 'value': i} for i in mail_data['type'].unique()], multi=True, placeholder="Filter by Mail Type...", style={'width': '400px'}),
                dcc.Dropdown(id='intent-dropdown', options=[{'label': i, 'value': i} for i in call_data['intent'].unique()], multi=True, placeholder="Filter by Call Intent...", style={'width': '400px'})
            ]),
            dcc.Graph(id='breakdown-graph')
        ])
    
    def augmentation_tab_layout(original, synthetic):
        return html.Div([
            html.H3("Data Augmentation Analysis", style={'textAlign': 'center'}),
            dcc.Graph(figure=create_augmentation_figure(original, synthetic))
        ])

    def trends_tab_layout(df):
        return html.Div([
            html.H3("Normalized Market & Business Trends", style={'textAlign': 'center'}),
            dcc.Graph(figure=create_normalized_trends_figure(df))
        ])
    
    @app.callback(Output('overview-graph', 'figure'), Input('overview-date-slider', 'value'))
    def update_overview_graph(date_indices):
        dff = final_df.iloc[date_indices[0]:date_indices[1]]
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=dff.index, y=dff['call_volume'], name='Call Volume', line=dict(color='#007BFF')), secondary_y=False)
        fig.add_trace(go.Bar(x=dff.index, y=dff['mail_volume'], name='Mail Volume', marker_color='#17A2B8', opacity=0.6), secondary_y=True)
        fig.update_layout(title_text="Mail vs. Call Volume", template='plotly_white')
        fig.update_yaxes(title_text="Call Volume", secondary_y=False); fig.update_yaxes(title_text="Mail Volume", secondary_y=True)
        return fig

    @app.callback(Output('breakdown-graph', 'figure'), [Input('mail-type-dropdown', 'value'), Input('intent-dropdown', 'value')])
    def update_breakdown_graph(selected_mail_types, selected_intents):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        mail_dff = mail_df[mail_df['type'].isin(selected_mail_types)] if selected_mail_types else mail_df
        mail_agg = mail_dff.groupby('date')['volume'].sum()
        fig.add_trace(go.Bar(x=mail_agg.index, y=mail_agg.values, name='Mail Volume', marker_color='#17A2B8', opacity=0.6), secondary_y=True)
        call_dff = call_df[call_df['intent'].isin(selected_intents)] if selected_intents else call_df
        call_agg = call_dff.groupby('date').size()
        fig.add_trace(go.Scatter(x=call_agg.index, y=call_agg.values, name='Call Volume', line=dict(color='#007BFF')), secondary_y=False)
        fig.update_layout(title_text="Filtered Campaign & Intent Analysis", template='plotly_white')
        fig.update_yaxes(title_text="Call Volume", secondary_y=False); fig.update_yaxes(title_text="Mail Volume", secondary_y=True)
        return fig

    def create_augmentation_figure(original, synthetic):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=original.index, y=original['call_volume'], name='Original Call Data', mode='lines', line=dict(color='#007BFF')))
        fig.add_trace(go.Scatter(x=synthetic.index, y=synthetic['call_volume'], name='Synthetic Call Data', mode='lines', line=dict(color='#FF5733', dash='dash')))
        fig.update_layout(title_text="Original vs. Synthetic Call Data", template='plotly_white')
        return fig
        
    def create_normalized_trends_figure(df):
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        normalized_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
        normalized_df['call_volume_rolling'] = normalized_df['call_volume'].rolling(window=7).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=normalized_df.index, y=normalized_df['call_volume_rolling'], name='Call Volume (Smoothed)', line=dict(color='#007BFF', width=3)))
        fig.add_trace(go.Scatter(x=normalized_df.index, y=normalized_df['mail_volume'], name='Mail Volume', line=dict(color='#17A2B8', dash='dash')))
        for col in CONFIG['FINANCIAL_DATA']:
             if col in normalized_df.columns:
                fig.add_trace(go.Scatter(x=normalized_df.index, y=normalized_df[col], name=col, line=dict(dash='dot', opacity=0.7)))
        fig.update_layout(title_text="Normalized Business & Market Trends", template='plotly_white', yaxis_title="Normalized Value (0 to 1)")
        return fig

    logger.info("✓ Dashboard initialized. Starting server at http://127.0.0.1:8050/")
    app.run_server(debug=True)

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================
if __name__ == '__main__':
    logger = setup_logging()
    
    mail_data, call_data = load_and_process_data(CONFIG, logger)
    augmented_calls, synthetic_calls = augment_data_intelligently(call_data, mail_data, logger)
    final_dataframe = create_final_dataframe(augmented_calls, mail_data, CONFIG, logger)
    
    create_dashboard(final_dataframe, mail_data, call_data, augmented_calls.loc[call_data['date'].min():], synthetic_calls)
