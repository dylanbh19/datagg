#!/usr/bin/env python3
"""
Customer Communications Intelligence Plot Generator v1.0
Generates and saves key analysis plots as PNG files.
"""

# --- Core Libraries ---
import pandas as pd
import numpy as np
import warnings
import os
from datetime import datetime, timedelta
import logging
from pathlib import Path
import holidays

# --- Suppress all warnings ---
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

# --- Visualization ---
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    print("FATAL: Plotly is required. Install with: pip install plotly")
    exit()

# --- Optional Libraries ---
try:
    import yfinance as yf
    FINANCIAL_AVAILABLE = True
except ImportError:
    FINANCIAL_AVAILABLE = False
    print("WARNING: yfinance not available. Financial data will be skipped.")

try:
    from scipy.stats import pearsonr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("FATAL: Scipy is required for correlation analysis. Install with: pip install scipy")
    exit()

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # --- File Paths ---
    # Use this file for general volume analysis (Visuals 1 & 2)
    'CALL_FILE_PATH_VOLUME': r'data\GenesysExtract_20250609.csv',
    # Use this file specifically for intent analysis (Visual 3)
    'CALL_FILE_PATH_INTENTS': r'data\GenesysExtract_with_intents.csv',
    'MAIL_FILE_PATH': r'merged_output.csv',

    # --- Output Directory ---
    'OUTPUT_DIR': 'plots',

    # --- Column Mappings ---
    'MAIL_COLUMNS': {'date': 'mail_date', 'volume': 'mail_volume', 'type': 'mail_type'},
    'CALL_COLUMNS': {'date': 'ConversationStart', 'intent': 'uui_Intent'},

    # --- Financial Indicators ---
    'FINANCIAL_DATA': {
        'S&P 500': '^GSPC',
        'VIX': '^VIX'
    },

    # --- Analysis Settings ---
    'MAX_LAG_DAYS': 14,
    'MAIL_START_DATE': '2024-06-01',
    'USE_INNER_JOIN': True,  # Ensures we only analyze dates with both mail and calls
    'FILTER_WEEKENDS': True,
    'FILTER_HOLIDAYS': True,

    # --- Visual Settings ---
    'THEME_COLORS': {
        'calls': '#e74c3c',
        'mail': '#3498db',
        'financial': '#9b59b6',
        'correlation_positive': '#27ae60',
        'correlation_negative': '#c0392b',
    }
}

# =============================================================================
# SETUP
# =============================================================================
def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger('PlotGenerator')

def safe_load_csv(file_path, description="file"):
    """Safely load a CSV file."""
    if not os.path.exists(file_path):
        LOGGER.warning(f"{description} not found: {file_path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip', low_memory=False)
        LOGGER.info(f"✅ {description} loaded: {len(df):,} records from {file_path}")
        return df
    except Exception as e:
        LOGGER.error(f"❌ Failed to load {description} from {file_path}: {e}")
        return pd.DataFrame()

def preprocess_data(df, column_mapping, date_column):
    """Standardize columns and convert dates."""
    if df.empty:
        return df
    # Rename columns based on mapping
    rename_dict = {v: k for k, v in column_mapping.items() if v in df.columns}
    df = df.rename(columns=rename_dict)
    # Convert date column
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df = df.dropna(subset=[date_column])
        df[date_column] = df[date_column].dt.tz_localize(None).dt.normalize()
    return df

# =============================================================================
# DATA PROCESSING
# =============================================================================

def get_base_data(config):
    """Loads and prepares the primary mail and call volume data."""
    LOGGER.info("--- Loading Base Data for Volume Analysis ---")
    mail_df = safe_load_csv(config['MAIL_FILE_PATH'], "Mail data")
    call_df_volume = safe_load_csv(config['CALL_FILE_PATH_VOLUME'], "Call volume data")

    mail_df = preprocess_data(mail_df, config['MAIL_COLUMNS'], 'date')
    call_df_volume = preprocess_data(call_df_volume, config['CALL_COLUMNS'], 'date')

    # Aggregate daily volumes
    daily_mail = mail_df.groupby('date')['volume'].sum().reset_index(name='mail_volume')
    daily_calls = call_df_volume.groupby('date').size().reset_index(name='call_volume')

    # Merge data
    how_join = 'inner' if config['USE_INNER_JOIN'] else 'outer'
    combined_df = pd.merge(daily_mail, daily_calls, on='date', how=how_join).fillna(0)

    # Filter by date and remove weekends/holidays
    combined_df = combined_df[combined_df['date'] >= pd.to_datetime(config['MAIL_START_DATE'])]
    if config['FILTER_WEEKENDS']:
        combined_df = combined_df[combined_df['date'].dt.weekday < 5]
    if config['FILTER_HOLIDAYS']:
        us_holidays = holidays.UnitedStates()
        combined_df = combined_df[~combined_df['date'].dt.date.isin(us_holidays)]

    return combined_df.sort_values('date').reset_index(drop=True)

def add_financial_data(df, config):
    """Adds financial data to the dataframe."""
    if not FINANCIAL_AVAILABLE or df.empty:
        return df
    LOGGER.info("--- Adding Financial Data ---")
    start_date = df['date'].min() - timedelta(days=5)
    end_date = df['date'].max() + timedelta(days=5)
    
    for name, ticker in config['FINANCIAL_DATA'].items():
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not data.empty:
                data = data[['Close']].rename(columns={'Close': name})
                df = pd.merge(df, data, left_on='date', right_index=True, how='left')
                df[name] = df[name].ffill().bfill()
        except Exception as e:
            LOGGER.warning(f"Could not download financial data for {name}: {e}")
    return df

def normalize_data(df):
    """Normalize data using Z-score for plotting."""
    LOGGER.info("--- Normalizing Data ---")
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            # Remove outliers beyond 3 standard deviations before normalizing
            mean, std = df[col].mean(), df[col].std()
            if std > 0:
                df = df[np.abs(df[col] - mean) <= (3 * std)]
                # Z-score normalization
                df[f'{col}_norm'] = (df[col] - df[col].mean()) / df[col].std()
    return df

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def save_plot(fig, filename, output_dir):
    """Saves a Plotly figure as a PNG."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    path = os.path.join(output_dir, filename)
    try:
        fig.write_image(path, width=1600, height=900)
        LOGGER.info(f"✅ Plot saved successfully: {path}")
    except Exception as e:
        LOGGER.error(f"❌ Failed to save plot {filename}. Ensure you have 'kaleido' installed (`pip install kaleido`): {e}")

### VISUAL 1: Overlay Timeline ###
def create_and_save_overlay_plot(df, config):
    """Creates and saves the normalized overlay plot of mail, calls, and financials."""
    LOGGER.info("--- Creating Visual 1: Overlay Timeline ---")
    if df.empty or 'call_volume_norm' not in df.columns:
        LOGGER.warning("Not enough data for overlay plot.")
        return

    fig = go.Figure()
    colors = config['THEME_COLORS']

    # Add traces
    fig.add_trace(go.Scatter(x=df['date'], y=df['mail_volume_norm'], name='Mail Volume (Normalized)', line=dict(color=colors['mail'], width=3)))
    fig.add_trace(go.Scatter(x=df['date'], y=df['call_volume_norm'], name='Call Volume (Normalized)', line=dict(color=colors['calls'], width=3)))
    
    for fin_col in config['FINANCIAL_DATA'].keys():
        if f'{fin_col}_norm' in df.columns:
            fig.add_trace(go.Scatter(x=df['date'], y=df[f'{fin_col}_norm'], name=f'{fin_col} (Normalized)', line=dict(color=colors['financial'], width=2, dash='dash')))

    fig.update_layout(
        title='<b>Normalized Mail, Call, and Financial Trends</b><br><sub>Outliers removed and data normalized to identify patterns</sub>',
        xaxis_title='Date',
        yaxis_title='Normalized Value (Z-score)',
        template='plotly_white',
        font=dict(size=14, family="Arial"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    save_plot(fig, "1_overlay_trends.png", config['OUTPUT_DIR'])

### VISUAL 2: Lag Correlation ###
def create_and_save_lag_correlation_plot(df, config):
    """Calculates and plots the correlation between mail and calls at different lags."""
    LOGGER.info("--- Creating Visual 2: Lag Correlation Analysis ---")
    if df.empty or len(df) < config['MAX_LAG_DAYS']:
        LOGGER.warning("Not enough data for lag correlation plot.")
        return

    lags = range(0, config['MAX_LAG_DAYS'] + 1)
    correlations = []
    
    for lag in lags:
        if len(df) > lag:
            # Shift call volume back to see if mail volume predicts future calls
            corr, _ = pearsonr(df['mail_volume'].iloc[:-lag] if lag > 0 else df['mail_volume'], df['call_volume'].iloc[lag:])
            correlations.append(corr)
        else:
            correlations.append(np.nan)

    fig = go.Figure()
    colors = config['THEME_COLORS']
    bar_colors = [colors['correlation_positive'] if c >= 0 else colors['correlation_negative'] for c in correlations]

    fig.add_trace(go.Bar(
        x=[f'{lag} Days' for lag in lags],
        y=correlations,
        marker_color=bar_colors,
        text=[f'{c:.3f}' for c in correlations],
        textposition='auto'
    ))
    fig.update_layout(
        title='<b>Mail-to-Call Correlation by Day Lag</b><br><sub>Correlation coefficient between mail volume and call volume on subsequent days</sub>',
        xaxis_title='Lag (Mail Volume leads Call Volume by X days)',
        yaxis_title='Pearson Correlation Coefficient',
        template='plotly_white',
        font=dict(size=14, family="Arial")
    )
    save_plot(fig, "2_lag_correlation.png", config['OUTPUT_DIR'])

### VISUAL 3: Intent Correlation ###
def create_and_save_intent_correlation_plot(config):
    """Loads intent data and plots the top correlations between mail types and call intents."""
    LOGGER.info("--- Creating Visual 3: Intent Correlation ---")
    mail_df = safe_load_csv(config['MAIL_FILE_PATH'], "Mail data")
    call_df_intents = safe_load_csv(config['CALL_FILE_PATH_INTENTS'], "Call intent data")

    if mail_df.empty or call_df_intents.empty:
        LOGGER.warning("Missing mail or call intent data. Skipping Visual 3.")
        return

    # Preprocess and aggregate data
    mail_df = preprocess_data(mail_df, config['MAIL_COLUMNS'], 'date')
    call_df_intents = preprocess_data(call_df_intents, config['CALL_COLUMNS'], 'date')
    
    # Pivot to get daily counts for each mail type and call intent
    daily_mail_types = mail_df.pivot_table(index='date', columns='type', values='volume', aggfunc='sum').fillna(0)
    daily_call_intents = call_df_intents.pivot_table(index='date', columns='intent', aggfunc='size', fill_value=0)
    
    # Merge on date, keeping only dates where both exist
    merged_intents = pd.merge(daily_mail_types, daily_call_intents, on='date', how='inner')
    
    if len(merged_intents) < 10:
        LOGGER.warning("Not enough overlapping intent data to analyze. Skipping Visual 3.")
        return

    # Calculate all correlations
    all_correlations = []
    for mail_type in daily_mail_types.columns:
        for call_intent in daily_call_intents.columns:
            if merged_intents[mail_type].std() > 0 and merged_intents[call_intent].std() > 0:
                corr, _ = pearsonr(merged_intents[mail_type], merged_intents[call_intent])
                all_correlations.append({
                    'mail_type': mail_type,
                    'call_intent': call_intent,
                    'correlation': corr
                })
    
    if not all_correlations:
        LOGGER.warning("Could not calculate any intent correlations.")
        return

    # Get top 10 absolute correlations
    corr_df = pd.DataFrame(all_correlations)
    corr_df['abs_corr'] = corr_df['correlation'].abs()
    top_10 = corr_df.sort_values('abs_corr', ascending=False).head(10)
    top_10 = top_10.sort_values('correlation', ascending=True) # Sort for plotting

    # Create plot
    fig = go.Figure()
    colors = config['THEME_COLORS']
    bar_colors = [colors['correlation_positive'] if c >= 0 else colors['correlation_negative'] for c in top_10['correlation']]
    
    fig.add_trace(go.Bar(
        y=[f"{row['mail_type']} →<br>{row['call_intent']}" for _, row in top_10.iterrows()],
        x=top_10['correlation'],
        orientation='h',
        marker_color=bar_colors,
        text=[f'{c:.3f}' for c in top_10['correlation']],
        textposition='auto'
    ))
    fig.update_layout(
        title='<b>Top 10 Correlations: Mail Type vs. Call Intent</b><br><sub>Which mail types are most correlated with specific customer calls on the same day</sub>',
        xaxis_title='Pearson Correlation Coefficient',
        yaxis_title='Mail Type → Call Intent',
        template='plotly_white',
        font=dict(size=14, family="Arial"),
        margin=dict(l=250) # Add left margin for long labels
    )
    save_plot(fig, "3_intent_correlations.png", config['OUTPUT_DIR'])

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == '__main__':
    LOGGER = setup_logging()
    LOGGER.info("🚀 Starting Customer Communications Plot Generator...")

    # --- Visuals 1 & 2 ---
    base_df = get_base_data(CONFIG)
    if not base_df.empty:
        # Visual 1
        base_with_financials = add_financial_data(base_df.copy(), CONFIG)
        normalized_df = normalize_data(base_with_financials)
        create_and_save_overlay_plot(normalized_df, CONFIG)
        
        # Visual 2
        create_and_save_lag_correlation_plot(base_df.copy(), CONFIG)
    else:
        LOGGER.error("Base data for visuals 1 & 2 is empty. Halting execution for these plots.")

    # --- Visual 3 ---
    create_and_save_intent_correlation_plot(CONFIG)

    LOGGER.info("✅ Plot generation process complete.")

