#!/usr/bin/env python3
"""
Interactive Call & Mail Volume Analysis Dashboard

This script creates an interactive web dashboard to analyze the relationship 
between mail campaigns, call volumes, and broader financial market indicators.

It fetches historical market data automatically and allows users to overlay
it onto the primary call/mail volume chart using interactive controls.
"""

import pandas as pd
import numpy as np
import warnings
import os
import yfinance as yf
from datetime import datetime

# Dashboarding & Plotting Libraries
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION SECTION - MODIFY THESE PATHS AND SETTINGS
# =============================================================================
CONFIG = {
    # === FILE PATHS (UPDATE THESE) ===
    'MAIL_FILE_PATH': r'C:\path\to\your\mail_data.csv',
    'CALL_FILE_PATH': r'C:\path\to\your\call_data.csv',

    # === DATA COLUMN MAPPING ===
    'MAIL_COLUMNS': {'date': 'date', 'volume': 'volume'},
    'CALL_COLUMNS': {'date': 'call_date', 'volume': 'volume'}, # Assuming call data needs aggregation
    'CALL_AGGREGATION_METHOD': 'count', # Use 'count' for raw call logs, or 'sum' if volume is pre-calculated

    # === FINANCIAL DATA TICKERS (FROM YAHOO FINANCE) ===
    'FINANCIAL_DATA': {
        'S&P 500': '^GSPC',
        '10-Yr Treasury Yield': '^TNX',
        'Crude Oil': 'CL=F' # Proxy for inflation/economic activity
    }
}

# =============================================================================
# DATA LOADING AND PROCESSING FUNCTIONS
# =============================================================================

def load_and_process_data(file_path, columns, is_call_data=False):
    """Loads and processes either mail or call data from a file."""
    if not os.path.exists(file_path):
        print(f"WARNING: File not found at {file_path}. Skipping.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"ERROR reading {file_path}: {e}")
        return pd.DataFrame()

    # Rename columns to standard names
    rename_map = {v: k for k, v in columns.items() if v in df.columns}
    df.rename(columns=rename_map, inplace=True)

    if 'date' not in df.columns:
        print(f"ERROR: Date column not found in {file_path}")
        return pd.DataFrame()
        
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    # Aggregate call data to daily volume
    if is_call_data:
        if CONFIG['CALL_AGGREGATION_METHOD'] == 'count':
            return df.resample('D').size().to_frame('volume')
        elif 'volume' in df.columns:
            return df.resample('D')['volume'].sum().to_frame('volume')
    
    # For mail data, ensure daily aggregation
    if 'volume' in df.columns:
        return df.resample('D')['volume'].sum().to_frame('volume')
    
    return pd.DataFrame()

def fetch_financial_data(start_date, end_date):
    """Fetches all financial data from Yahoo Finance for the given date range."""
    print("Fetching financial data...")
    financial_df = pd.DataFrame()
    tickers = CONFIG['FINANCIAL_DATA']
    
    data = yf.download(list(tickers.values()), start=start_date, end=end_date, progress=False)
    
    if not data.empty:
        # Extract closing prices and rename columns to be user-friendly
        close_prices = data['Adj Close']
        # Handle cases where only one ticker is returned
        if isinstance(close_prices, pd.Series):
             close_prices = close_prices.to_frame(data.columns[0])

        financial_df = close_prices.rename(columns={v: k for k, v in tickers.items()})
        financial_df.ffill(inplace=True) # Forward-fill to handle non-trading days
    
    print("Financial data fetched successfully.")
    return financial_df

# =============================================================================
# LOAD INITIAL DATA
# =============================================================================
mail_df = load_and_process_data(CONFIG['MAIL_FILE_PATH'], CONFIG['MAIL_COLUMNS'])
call_df = load_and_process_data(CONFIG['CALL_FILE_PATH'], CONFIG['CALL_COLUMNS'], is_call_data=True)

# Combine and find the full date range for fetching financial data
combined_df = pd.concat([mail_df, call_df]).sort_index()
if not combined_df.empty:
    min_date, max_date = combined_df.index.min(), combined_df.index.max()
    financial_data = fetch_financial_data(min_date, max_date)
else:
    print("Could not load mail or call data. Dashboard may be empty.")
    financial_data = pd.DataFrame()
    min_date, max_date = datetime.now(), datetime.now()


# =============================================================================
# SETUP & LAYOUT THE DASHBOARD
# =============================================================================
app = dash.Dash(__name__)
app.title = "Call & Mail Volume Analysis"

app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f9f9f9', 'padding': '20px'}, children=[
    html.H1(
        "Call & Mail Volume Analysis Dashboard",
        style={'textAlign': 'center', 'color': '#2c3e50'}
    ),
    html.P(
        "Analyze call and mail volumes alongside key financial market indicators.",
        style={'textAlign': 'center', 'color': '#555'}
    ),

    html.Div(className='controls-container', style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'marginBottom': '20px'}, children=[
        html.H3("Chart Controls", style={'borderBottom': '2px solid #ddd', 'paddingBottom': '10px'}),
        html.P("Select financial data to overlay on the chart:", style={'fontWeight': 'bold'}),
        dcc.Checklist(
            id='financial-checklist',
            options=[{'label': key, 'value': key} for key in CONFIG['FINANCIAL_DATA'].keys()],
            value=[], # Default to no overlays
            inline=True,
            style={'display': 'flex', 'gap': '20px', 'marginTop': '10px'}
        )
    ]),

    dcc.Graph(id='main-analysis-graph', style={'height': '600px'}),

    html.Footer(
        f"Data analysis from {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}. Fetched on {datetime.now().strftime('%Y-%m-%d %H:%M')}.",
        style={'textAlign': 'center', 'marginTop': '30px', 'fontSize': '12px', 'color': 'gray'}
    )
])

# =============================================================================
# DASHBOARD INTERACTIVITY (CALLBACKS)
# =============================================================================
@app.callback(
    Output('main-analysis-graph', 'figure'),
    [Input('financial-checklist', 'value')]
)
def update_graph(selected_financial_data):
    # Initialize a figure with a secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. Add Mail Volume Data
    if not mail_df.empty:
        fig.add_trace(
            go.Bar(x=mail_df.index, y=mail_df['volume'], name='Mail Volume', marker_color='rgba(255, 159, 64, 0.7)'),
            secondary_y=False,
        )

    # 2. Add Call Volume Data
    if not call_df.empty:
        fig.add_trace(
            go.Scatter(x=call_df.index, y=call_df['volume'], name='Call Volume', mode='lines', line=dict(color='rgba(54, 162, 235, 1)', width=2)),
            secondary_y=False,
        )

    # 3. Add Selected Financial Data Overlays
    if selected_financial_data and not financial_data.empty:
        colors = ['#e74c3c', '#9b59b6', '#16a085'] # Colors for financial traces
        for i, key in enumerate(selected_financial_data):
            if key in financial_data.columns:
                fig.add_trace(
                    go.Scatter(x=financial_data.index, y=financial_data[key], name=key, mode='lines', line=dict(color=colors[i % len(colors)], dash='dot')),
                    secondary_y=True,
                )

    # 4. Update Figure Layout
    fig.update_layout(
        title_text="Mail & Call Volume vs. Financial Markets",
        barmode='stack',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_white',
        xaxis=dict(title_text="Date"),
        yaxis=dict(title_text="Mail & Call Volume", side='left'),
        yaxis2=dict(title_text="Financial Data Values", side='right', overlaying='y', showgrid=False, visible=bool(selected_financial_data))
    )
    
    return fig

# =============================================================================
# RUN THE DASHBOARD
# =============================================================================
if __name__ == '__main__':
    # Check if data files exist before trying to run server
    if not os.path.exists(CONFIG['MAIL_FILE_PATH']) or not os.path.exists(CONFIG['CALL_FILE_PATH']):
        print("\nERROR: One or both data files are not found.")
        print("Please update 'MAIL_FILE_PATH' and 'CALL_FILE_PATH' in the script.")
    else:
        app.run_server(debug=True)
