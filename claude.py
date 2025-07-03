#!/usr/bin/env python3
“””
Executive-Grade Interactive Marketing Intelligence Dashboard

Professional multi-page Dash application with advanced data augmentation,
financial overlays, and boardroom-quality visualizations for strategic insights.
“””

# — Core Libraries —

import pandas as pd
import numpy as np
import warnings
import os
import sys
from datetime import datetime, timedelta
import logging
from pathlib import Path

# — Visualization & Dashboard —

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc

# — Statistical Analysis —

from scipy import stats
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# — Financial Data —

try:
import yfinance as yf
FINANCIAL_AVAILABLE = True
except ImportError:
FINANCIAL_AVAILABLE = False
print(“Warning: yfinance not available. Financial overlays will be disabled.”)

warnings.filterwarnings(‘ignore’)

# =============================================================================

# CONFIGURATION - KEEP YOUR SETTINGS

# =============================================================================

CONFIG = {
‘MAIL_FILE_PATH’: r’merged_output.csv’,
‘CALL_FILE_PATH’: r’data\GenesysExtract_20250609.csv’,
‘MAIL_COLUMNS’: {‘date’: ‘mail date’, ‘volume’: ‘mail_volume’, ‘type’: ‘mail_type’},
‘CALL_COLUMNS’: {‘date’: ‘ConversationStart’, ‘intent’: ‘vui_intent’},
‘FINANCIAL_DATA’: {‘S&P 500’: ‘^GSPC’, ‘10-Yr Treasury’: ‘^TNX’, ‘VIX’: ‘^VIX’},
‘MAX_LAG_DAYS’: 28
}

# =============================================================================

# LOGGING SETUP

# =============================================================================

def setup_logging():
“”“Sets up comprehensive logging for the dashboard.”””
logging.basicConfig(
level=logging.INFO,
format=’%(asctime)s - %(levelname)s - %(message)s’,
stream=sys.stdout
)
return logging.getLogger(‘ExecutiveDashboard’)

# =============================================================================

# DATA LOADING & PROCESSING

# =============================================================================

class DataProcessor:
def **init**(self, config, logger):
self.config = config
self.logger = logger
self.mail_df = None
self.call_df = None
self.financial_df = None
self.combined_df = None

```
def load_data(self):
    """Load and preprocess mail and call data."""
    try:
        self.logger.info("🔄 STEP 1: Loading source data...")
        
        # Load mail data
        if not os.path.exists(self.config['MAIL_FILE_PATH']):
            raise FileNotFoundError(f"Mail file not found: {self.config['MAIL_FILE_PATH']}")
            
        self.mail_df = pd.read_csv(self.config['MAIL_FILE_PATH'], on_bad_lines='skip')
        self.logger.info(f"  📧 Mail data: {len(self.mail_df):,} records loaded")
        
        # Map mail columns
        mail_col_mapping = {v: k for k, v in self.config['MAIL_COLUMNS'].items()}
        self.mail_df = self.mail_df.rename(columns=mail_col_mapping)
        
        # Process mail dates and clean data
        self.mail_df['date'] = pd.to_datetime(self.mail_df['date'], errors='coerce')
        self.mail_df = self.mail_df.dropna(subset=['date'])
        self.mail_df['date'] = self.mail_df['date'].dt.normalize()
        
        # Load call data
        if not os.path.exists(self.config['CALL_FILE_PATH']):
            raise FileNotFoundError(f"Call file not found: {self.config['CALL_FILE_PATH']}")
            
        self.call_df = pd.read_csv(self.config['CALL_FILE_PATH'], on_bad_lines='skip')
        self.logger.info(f"  📞 Call data: {len(self.call_df):,} records loaded")
        
        # Map call columns
        call_col_mapping = {v: k for k, v in self.config['CALL_COLUMNS'].items()}
        self.call_df = self.call_df.rename(columns=call_col_mapping)
        
        # Process call dates
        self.call_df['date'] = pd.to_datetime(self.call_df['date'], errors='coerce')
        self.call_df = self.call_df.dropna(subset=['date'])
        self.call_df['date'] = self.call_df['date'].dt.normalize()
        
        self.logger.info("✅ Data loading completed successfully")
        return True
        
    except Exception as e:
        self.logger.error(f"❌ Failed to load data: {str(e)}")
        return False

def augment_call_data(self):
    """Perform advanced data augmentation for call volumes."""
    self.logger.info("🔄 STEP 2: Augmenting call data...")
    
    try:
        # Aggregate calls by date
        daily_calls = self.call_df.groupby('date').size().reset_index(name='call_volume')
        
        # Get full date range
        if hasattr(self.mail_df, 'date') and len(self.mail_df) > 0:
            min_date = min(self.mail_df['date'].min(), daily_calls['date'].min())
            max_date = max(self.mail_df['date'].max(), daily_calls['date'].max())
        else:
            min_date = daily_calls['date'].min()
            max_date = daily_calls['date'].max()
        
        # Create complete date range
        full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        complete_calls = pd.DataFrame({'date': full_date_range})
        complete_calls = complete_calls.merge(daily_calls, on='date', how='left')
        
        # Identify missing data periods
        missing_mask = complete_calls['call_volume'].isna()
        missing_count = missing_mask.sum()
        
        if missing_count > 0:
            self.logger.info(f"  🔧 Augmenting {missing_count:,} missing call volume days...")
            
            # Advanced augmentation strategy
            complete_calls['day_of_week'] = complete_calls['date'].dt.dayofweek
            complete_calls['month'] = complete_calls['date'].dt.month
            complete_calls['is_weekend'] = complete_calls['day_of_week'].isin([5, 6])
            
            # Calculate baseline patterns from existing data
            existing_data = complete_calls.dropna()
            if len(existing_data) > 0:
                # Weekly patterns
                weekly_pattern = existing_data.groupby('day_of_week')['call_volume'].mean()
                monthly_pattern = existing_data.groupby('month')['call_volume'].mean()
                overall_mean = existing_data['call_volume'].mean()
                overall_std = existing_data['call_volume'].std()
                
                # Fill missing values with intelligent estimates
                for idx, row in complete_calls[missing_mask].iterrows():
                    base_value = weekly_pattern.get(row['day_of_week'], overall_mean)
                    seasonal_factor = monthly_pattern.get(row['month'], overall_mean) / overall_mean
                    
                    # Add realistic variation
                    noise = np.random.normal(0, overall_std * 0.1)
                    estimated_value = max(0, base_value * seasonal_factor + noise)
                    
                    complete_calls.loc[idx, 'call_volume'] = estimated_value
            
            # Mark augmented data
            complete_calls['is_augmented'] = missing_mask
        else:
            complete_calls['is_augmented'] = False
        
        self.call_df_augmented = complete_calls
        self.logger.info("✅ Call data augmentation completed")
        return True
        
    except Exception as e:
        self.logger.error(f"❌ Call augmentation failed: {str(e)}")
        return False

def load_financial_data(self):
    """Load financial indicator data."""
    if not FINANCIAL_AVAILABLE:
        self.logger.warning("⚠️  Financial data unavailable - yfinance not installed")
        return False
        
    self.logger.info("🔄 STEP 3: Loading financial indicators...")
    
    try:
        # Get date range from existing data
        start_date = self.call_df_augmented['date'].min() - timedelta(days=30)
        end_date = self.call_df_augmented['date'].max() + timedelta(days=1)
        
        financial_data = {}
        
        for name, ticker in self.config['FINANCIAL_DATA'].items():
            try:
                self.logger.info(f"  📈 Fetching {name} ({ticker})...")
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if not data.empty:
                    # Use adjusted close price
                    if 'Adj Close' in data.columns:
                        financial_data[name] = data['Adj Close'].resample('D').last()
                    elif 'Close' in data.columns:
                        financial_data[name] = data['Close'].resample('D').last()
                    
            except Exception as e:
                self.logger.warning(f"  ⚠️  Failed to fetch {name}: {str(e)}")
                continue
        
        if financial_data:
            self.financial_df = pd.DataFrame(financial_data)
            self.financial_df.index.name = 'date'
            self.financial_df = self.financial_df.reset_index()
            self.financial_df['date'] = pd.to_datetime(self.financial_df['date']).dt.normalize()
            
            self.logger.info(f"✅ Financial data loaded: {len(financial_data)} indicators")
            return True
        else:
            self.logger.warning("⚠️  No financial data could be loaded")
            return False
            
    except Exception as e:
        self.logger.error(f"❌ Financial data loading failed: {str(e)}")
        return False

def combine_all_data(self):
    """Combine all data sources into final dataset."""
    self.logger.info("🔄 STEP 4: Combining all data sources...")
    
    try:
        # Start with augmented call data
        combined = self.call_df_augmented.copy()
        
        # Add mail data
        if hasattr(self.mail_df, 'date') and len(self.mail_df) > 0:
            mail_daily = self.mail_df.groupby('date')['volume'].sum().reset_index()
            mail_daily = mail_daily.rename(columns={'volume': 'mail_volume'})
            combined = combined.merge(mail_daily, on='date', how='left')
            combined['mail_volume'] = combined['mail_volume'].fillna(0)
        else:
            combined['mail_volume'] = 0
        
        # Add financial data
        if hasattr(self, 'financial_df') and self.financial_df is not None:
            combined = combined.merge(self.financial_df, on='date', how='left')
            
            # Forward fill financial data for weekends/holidays
            financial_cols = [col for col in combined.columns if col in self.config['FINANCIAL_DATA'].keys()]
            for col in financial_cols:
                combined[col] = combined[col].fillna(method='ffill').fillna(method='bfill')
        
        # Add derived features
        combined['weekday'] = combined['date'].dt.day_name()
        combined['month'] = combined['date'].dt.month_name()
        combined['quarter'] = combined['date'].dt.quarter
        combined['is_weekend'] = combined['date'].dt.weekday >= 5
        
        # Calculate response rates where possible
        combined['response_rate'] = np.where(
            combined['mail_volume'] > 0,
            (combined['call_volume'] / combined['mail_volume'] * 100).round(2),
            0
        )
        
        self.combined_df = combined
        self.logger.info(f"✅ Final dataset created: {len(combined):,} records, {len(combined.columns)} features")
        return True
        
    except Exception as e:
        self.logger.error(f"❌ Data combination failed: {str(e)}")
        return False
```

# =============================================================================

# DASHBOARD APPLICATION

# =============================================================================

class ExecutiveDashboard:
def **init**(self, data_processor):
self.dp = data_processor
self.app = dash.Dash(
**name**,
external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
suppress_callback_exceptions=True
)
self.app.title = “Marketing Intelligence Dashboard”

```
def create_sidebar(self):
    """Create the navigation sidebar."""
    return html.Div([
        html.Div([
            html.H2("📊 Marketing Intelligence", className="text-white mb-0"),
            html.P("Executive Dashboard", className="text-light small mb-0")
        ], className="sidebar-header"),
        
        html.Hr(className="border-light"),
        
        dbc.Nav([
            dbc.NavLink([
                html.I(className="fas fa-chart-line me-2"),
                "Executive Overview"
            ], href="/", active="exact", className="nav-link-custom"),
            
            dbc.NavLink([
                html.I(className="fas fa-bullseye me-2"),
                "Campaign Analytics"
            ], href="/campaigns", active="exact", className="nav-link-custom"),
            
            dbc.NavLink([
                html.I(className="fas fa-chart-area me-2"),
                "Financial Correlation"
            ], href="/financial", active="exact", className="nav-link-custom"),
            
            dbc.NavLink([
                html.I(className="fas fa-cogs me-2"),
                "Data Diagnostics"
            ], href="/diagnostics", active="exact", className="nav-link-custom"),
        ], vertical=True, pills=True, className="flex-column"),
        
        html.Hr(className="border-light mt-4"),
        
        html.Div([
            html.P([
                html.I(className="fas fa-database me-2"),
                f"Records: {len(self.dp.combined_df):,}"
            ], className="text-light small mb-1"),
            html.P([
                html.I(className="fas fa-calendar me-2"),
                f"Updated: {datetime.now().strftime('%Y-%m-%d')}"
            ], className="text-light small mb-0"),
        ], className="sidebar-footer")
    ], className="sidebar")

def create_kpi_card(self, title, value, change=None, icon="fas fa-chart-line", color="primary"):
    """Create a KPI card component."""
    card_content = [
        html.Div([
            html.I(className=f"{icon} fa-2x text-{color}"),
            html.Div([
                html.H6(title, className="card-title text-muted mb-1"),
                html.H3(value, className="card-text mb-0"),
            ], className="ms-3")
        ], className="d-flex align-items-center")
    ]
    
    if change is not None:
        change_color = "success" if str(change).startswith('+') else "danger" if str(change).startswith('-') else "muted"
        card_content.append(
            html.P(change, className=f"text-{change_color} small mb-0 mt-2")
        )
    
    return dbc.Card(dbc.CardBody(card_content), className="h-100 shadow-sm")

def build_overview_page(self):
    """Build the executive overview page."""
    df = self.dp.combined_df
    
    # Calculate KPIs
    total_calls = int(df['call_volume'].sum())
    total_mail = int(df['mail_volume'].sum())
    avg_response_rate = df[df['mail_volume'] > 0]['response_rate'].mean()
    
    # Calculate changes (last 30 days vs previous 30)
    recent_data = df.tail(30)
    previous_data = df.tail(60).head(30)
    
    call_change = ((recent_data['call_volume'].sum() - previous_data['call_volume'].sum()) / 
                  previous_data['call_volume'].sum() * 100) if previous_data['call_volume'].sum() > 0 else 0
    
    # Create main time series chart
    fig_main = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Call Volume Timeline", "Mail Volume Timeline"),
        vertical_spacing=0.1,
        shared_xaxes=True
    )
    
    # Add call volume (highlight augmented data)
    real_data = df[~df['is_augmented']]
    aug_data = df[df['is_augmented']]
    
    fig_main.add_trace(
        go.Scatter(
            x=real_data['date'], 
            y=real_data['call_volume'],
            name='Actual Calls',
            line=dict(color='#1f77b4', width=3)
        ), row=1, col=1
    )
    
    if len(aug_data) > 0:
        fig_main.add_trace(
            go.Scatter(
                x=aug_data['date'],
                y=aug_data['call_volume'],
                name='Augmented Calls',
                line=dict(color='#ff7f0e', width=2, dash='dash')
            ), row=1, col=1
        )
    
    fig_main.add_trace(
        go.Bar(
            x=df['date'],
            y=df['mail_volume'],
            name='Mail Volume',
            marker_color='#2ca02c',
            opacity=0.7
        ), row=2, col=1
    )
    
    fig_main.update_layout(
        title="Marketing Performance Overview",
        template='plotly_white',
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Correlation heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        title="Feature Correlation Matrix",
        color_continuous_scale="RdBu_r"
    )
    fig_corr.update_layout(template='plotly_white')
    
    return dbc.Container([
        # KPI Row
        dbc.Row([
            dbc.Col([
                self.create_kpi_card(
                    "Total Calls", f"{total_calls:,}",
                    f"{call_change:+.1f}% vs prev 30d",
                    "fas fa-phone", "primary"
                )
            ], width=3),
            dbc.Col([
                self.create_kpi_card(
                    "Total Mail Sent", f"{total_mail:,}",
                    None, "fas fa-envelope", "success"
                )
            ], width=3),
            dbc.Col([
                self.create_kpi_card(
                    "Avg Response Rate", f"{avg_response_rate:.2f}%",
                    None, "fas fa-percentage", "info"
                )
            ], width=3),
            dbc.Col([
                self.create_kpi_card(
                    "Data Quality", f"{(~df['is_augmented']).mean()*100:.0f}%",
                    "Actual vs Augmented", "fas fa-check-circle", "warning"
                )
            ], width=3),
        ], className="mb-4"),
        
        # Main Chart
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=fig_main)
            ], width=12)
        ], className="mb-4"),
        
        # Correlation Matrix
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=fig_corr)
            ], width=12)
        ])
    ], fluid=True)

def build_campaign_page(self):
    """Build the campaign analytics page."""
    return dbc.Container([
        html.H2("Campaign Analytics", className="mb-4"),
        
        # Controls
        dbc.Row([
            dbc.Col([
                html.Label("Mail Type Filter:"),
                dcc.Dropdown(
                    id='mail-type-dropdown',
                    options=[{'label': t, 'value': t} for t in sorted(self.dp.mail_df['type'].unique())] if hasattr(self.dp.mail_df, 'type') else [],
                    multi=True,
                    placeholder="Select mail types..."
                )
            ], width=6),
            dbc.Col([
                html.Label("Call Intent Filter:"),
                dcc.Dropdown(
                    id='intent-dropdown',
                    options=[{'label': i, 'value': i} for i in sorted(self.dp.call_df['intent'].unique())] if hasattr(self.dp.call_df, 'intent') else [],
                    multi=True,
                    placeholder="Select call intents..."
                )
            ], width=6)
        ], className="mb-4"),
        
        # Main Chart
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='campaign-analysis-chart')
            ], width=12)
        ])
    ], fluid=True)

def build_financial_page(self):
    """Build the financial correlation page."""
    df = self.dp.combined_df
    
    # Check if financial data is available
    financial_cols = [col for col in df.columns if col in self.dp.config['FINANCIAL_DATA'].keys()]
    
    if not financial_cols:
        return dbc.Container([
            dbc.Alert([
                html.H4("Financial Data Unavailable", className="alert-heading"),
                html.P("Financial indicators could not be loaded. This may be due to:"),
                html.Ul([
                    html.Li("Missing yfinance package"),
                    html.Li("Network connectivity issues"),
                    html.Li("Market data API limitations")
                ]),
                html.P("Install yfinance with: pip install yfinance", className="mb-0 font-monospace")
            ], color="warning")
        ], fluid=True)
    
    # Create financial overlay charts
    fig_financial = make_subplots(
        rows=len(financial_cols) + 1, cols=1,
        subplot_titles=["Call Volume"] + financial_cols,
        vertical_spacing=0.08,
        shared_xaxes=True
    )
    
    # Add call volume
    fig_financial.add_trace(
        go.Scatter(
            x=df['date'], 
            y=df['call_volume'],
            name='Call Volume',
            line=dict(color='#1f77b4', width=2)
        ), row=1, col=1
    )
    
    # Add financial indicators
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i, col in enumerate(financial_cols):
        fig_financial.add_trace(
            go.Scatter(
                x=df['date'],
                y=df[col],
                name=col,
                line=dict(color=colors[i % len(colors)], width=2)
            ), row=i+2, col=1
        )
    
    fig_financial.update_layout(
        title="Financial Indicators vs Call Volume",
        template='plotly_white',
        height=200 * (len(financial_cols) + 1),
        showlegend=True
    )
    
    # Calculate correlations
    correlations = []
    for col in financial_cols:
        corr, p_value = pearsonr(df['call_volume'].fillna(0), df[col].fillna(0))
        correlations.append({
            'Indicator': col,
            'Correlation': f"{corr:.4f}",
            'P-Value': f"{p_value:.4f}",
            'Strength': 'Strong' if abs(corr) > 0.7 else 'Moderate' if abs(corr) > 0.3 else 'Weak'
        })
    
    return dbc.Container([
        html.H2("Financial Market Correlation", className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=fig_financial)
            ], width=8),
            dbc.Col([
                html.H5("Correlation Analysis"),
                dash_table.DataTable(
                    data=correlations,
                    columns=[{"name": i, "id": i} for i in correlations[0].keys()] if correlations else [],
                    style_cell={'textAlign': 'left'},
                    style_data_conditional=[
                        {
                            'if': {'filter_query': '{Strength} = Strong'},
                            'backgroundColor': '#d4edda',
                            'color': 'black',
                        },
                        {
                            'if': {'filter_query': '{Strength} = Moderate'},
                            'backgroundColor': '#fff3cd',
                            'color': 'black',
                        }
                    ]
                )
            ], width=4)
        ])
    ], fluid=True)

def build_diagnostics_page(self):
    """Build the data diagnostics page."""
    df = self.dp.combined_df
    
    # Data quality metrics
    total_records = len(df)
    augmented_records = df['is_augmented'].sum()
    real_records = total_records - augmented_records
    
    quality_metrics = [
        {"Metric": "Total Records", "Value": f"{total_records:,}", "Status": "✅"},
        {"Metric": "Real Data Points", "Value": f"{real_records:,}", "Status": "✅"},
        {"Metric": "Augmented Points", "Value": f"{augmented_records:,}", "Status": "⚠️" if augmented_records > 0 else "✅"},
        {"Metric": "Data Completeness", "Value": f"{(real_records/total_records)*100:.1f}%", "Status": "✅"},
        {"Metric": "Date Range", "Value": f"{df['date'].min().date()} to {df['date'].max().date()}", "Status": "✅"}
    ]
    
    return dbc.Container([
        html.H2("Data Quality Diagnostics", className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                html.H5("Data Quality Metrics"),
                dash_table.DataTable(
                    data=quality_metrics,
                    columns=[{"name": i, "id": i} for i in quality_metrics[0].keys()],
                    style_cell={'textAlign': 'left'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
                )
            ], width=6),
            dbc.Col([
                html.H5("Missing Data Pattern"),
                dcc.Graph(
                    figure=px.bar(
                        x=['Real Data', 'Augmented Data'],
                        y=[real_records, augmented_records],
                        title="Data Composition",
                        color=['Real Data', 'Augmented Data'],
                        color_discrete_map={'Real Data': '#2ca02c', 'Augmented Data': '#ff7f0e'}
                    ).update_layout(template='plotly_white', showlegend=False)
                )
            ], width=6)
        ])
    ], fluid=True)

def setup_layout(self):
    """Setup the main layout."""
    self.app.layout = html.Div([
        dcc.Location(id="url"),
        self.create_sidebar(),
        html.Div(id="page-content", className="content")
    ])
    
    # Add custom CSS
    self.app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
            .sidebar {
                position: fixed;
                top: 0;
                left: 0;
                bottom: 0;
                width: 280px;
                padding: 2rem 1rem;
                background: linear-gradient(180deg, #1e3a8a 0%, #3b82f6 100%);
                color: white;
                overflow-y: auto;
            }
            .content {
                margin-left: 300px;
                padding: 2rem;
                min-height: 100vh;
                background-color: #f8fafc;
            }
            .nav-link-custom {
                color: #e2e8f0 !important;
                border-radius: 0.5rem;
                margin-bottom: 0.5rem;
                transition: all 0.2s;
            }
            .nav-link-custom:hover, .nav-link-custom.active {
                background-color: rgba(255,255,255,0.1) !important;
                color: white !important;
            }
            .sidebar-header {
                text-align: center;
                margin-bottom: 1rem;
            }
            .sidebar-footer {
                position: absolute;
                bottom: 1rem;
                left: 1rem;
                right: 1rem;
            }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''

def setup_callbacks(self):
    """Setup dashboard callbacks."""
    
    @self.app.callback(
        Output("page-content", "children"),
        Input("url", "pathname")
    )
    def render_page_content(pathname):
        if pathname == "/":
            return self.build_overview_page()
        elif pathname == "/campaigns":
            return self.build_campaign_page()
        elif pathname == "/financial":
            return self.build_financial_page()
        elif pathname == "/diagnostics":
            return self.build_diagnostics_page()
        else:
            return html.Div([
                dbc.Alert([
                    html.H4("404 - Page Not Found", className="alert-heading"),
                    html.P("The requested page could not be found."),
                    dbc.Button("Return to Dashboard", href="/", color="primary")
                ], color="danger")
            ])
    
    @self.app.callback(
        Output('campaign-analysis-chart', 'figure'),
        [Input('mail-type-dropdown', 'value'),
         Input('intent-dropdown', 'value')]
    )
    def update_campaign_analysis(selected_mail_types, selected_intents):
        # Filter data based on selections
        filtered_mail = self.dp.mail_df.copy()
        filtered_calls = self.dp.call_df.copy()
        
        if selected_mail_types:
            if hasattr(filtered_mail, 'type'):
                filtered_mail = filtered_mail[filtered_mail['type'].isin(selected_mail_types)]
        
        if selected_intents:
            if hasattr(filtered_calls, 'intent'):
                filtered_calls = filtered_calls[filtered_calls['intent'].isin(selected_intents)]
        
        # Aggregate data
        mail_weekly = filtered_mail.groupby(pd.Grouper(key='date', freq='W'))['volume'].sum().reset_index()
        calls_weekly = filtered_calls.groupby(pd.Grouper(key='date', freq='W')).size().reset_index(name='call_count')
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Weekly Call Volume", "Weekly Mail Volume"),
            vertical_spacing=0.1,
            shared_xaxes=True
        )
        
        # Add traces
        fig.add_trace(
            go.Scatter(
                x=calls_weekly['date'],
                y=calls_weekly['call_count'],
                name='Call Volume',
                line=dict(color='#1f77b4', width=3),
                fill='tonexty'
            ), row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=mail_weekly['date'],
                y=mail_weekly['volume'],
                name='Mail Volume',
                marker_color='#2ca02c',
                opacity=0.7
            ), row=2, col=1
        )
        
        fig.update_layout(
            title="Campaign Performance Analysis",
            template='plotly_white',
            height=500,
            showlegend=True
        )
        
        return fig

def run_server(self, debug=True, port=8050):
    """Run the dashboard server."""
    self.dp.logger.info(f"🚀 Starting Executive Dashboard at http://127.0.0.1:{port}")
    self.app.run_server(debug=debug, port=port, host='127.0.0.1')
```

# =============================================================================

# MAIN EXECUTION

# =============================================================================

def main():
“”“Main execution function.”””
print(“🔥 EXECUTIVE MARKETING INTELLIGENCE DASHBOARD”)
print(”=” * 60)

```
# Setup logging
logger = setup_logging()

try:
    # Initialize data processor
    logger.info("🏗️  Initializing data processor...")
    dp = DataProcessor(CONFIG, logger)
    
    # Load and process data
    if not dp.load_data():
        logger.error("❌ Failed to load data. Exiting.")
        return False
    
    if not dp.augment_call_data():
        logger.error("❌ Failed to augment call data. Exiting.")
        return False
    
    # Load financial data (optional)
    dp.load_financial_data()
    
    if not dp.combine_all_data():
        logger.error("❌ Failed to combine data. Exiting.")
        return False
    
    # Initialize dashboard
    logger.info("🎨 Initializing executive dashboard...")
    dashboard = ExecutiveDashboard(dp)
    dashboard.setup_layout()
    dashboard.setup_callbacks()
    
    # Display summary
    logger.info("📊 DASHBOARD READY!")
    logger.info("=" * 40)
    logger.info(f"📧 Mail records: {len(dp.mail_df):,}")
    logger.info(f"📞 Call records: {len(dp.call_df):,}")
    logger.info(f"📈 Final dataset: {len(dp.combined_df):,} records")
    logger.info(f"🔧 Augmented records: {dp.combined_df['is_augmented'].sum():,}")
    logger.info(f"📊 Features: {len(dp.combined_df.columns)}")
    
    if hasattr(dp, 'financial_df') and dp.financial_df is not None:
        financial_indicators = [col for col in dp.combined_df.columns if col in CONFIG['FINANCIAL_DATA'].keys()]
        logger.info(f"💰 Financial indicators: {len(financial_indicators)}")
    
    logger.info("=" * 40)
    logger.info("🌐 Access your dashboard at: http://127.0.0.1:8050")
    logger.info("📱 Dashboard features:")
    logger.info("   • Executive Overview with KPIs")
    logger.info("   • Campaign Performance Analytics")
    logger.info("   • Financial Market Correlations")
    logger.info("   • Data Quality Diagnostics")
    logger.info("=" * 40)
    
    # Run dashboard
    dashboard.run_server(debug=True, port=8050)
    
    return True
    
except KeyboardInterrupt:
    logger.info("🛑 Dashboard stopped by user")
    return True
    
except Exception as e:
    logger.error(f"❌ Fatal error: {str(e)}")
    logger.error("Full traceback:", exc_info=True)
    return False
```

if **name** == ‘**main**’:
success = main()
sys.exit(0 if success else 1)