#!/usr/bin/env python3
"""
Customer Communications Intelligence Plot Generator v1.3
Speed enhancements from v2.0 integrated for high performance.
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
    print("FATAL: Plotly is required. Install with: pip install plotly kaleido")
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
    print("WARNING: Scipy not available. Correlation analysis will be limited.")

# =============================================================================
# ENHANCED CONFIGURATION
# =============================================================================
CONFIG = {
    # --- File Paths (Intelligent Loading Logic) ---
    'CALL_FILE_PATH_INTENTS': r'data\GenesysExtract_with_intents.csv', # Primary file with intent data
    'CALL_FILE_PATH_OVERVIEW': r'data\GenesysExtract_20250703.csv',  # Fallback for volume only
    'MAIL_FILE_PATH': r'merged_output.csv',

    # --- Output Directory ---
    'OUTPUT_DIR': 'plots',

    # --- Column Mappings ---
    'MAIL_COLUMNS': {'date': 'mail_date', 'volume': 'mail_volume', 'type': 'mail_type'},
    'CALL_INTENT_COLUMNS': {'date': 'ConversationStart', 'intent': 'uui_Intent'}, # For intent file
    'CALL_OVERVIEW_COLUMNS': {'date': 'Date'}, # For overview/volume file

    # --- Financial Indicators ---
    'FINANCIAL_DATA': {
        'S&P 500': '^GSPC',
        'VIX': '^VIX',
        'Dollar Index': 'DX-Y.NYB'
    },

    # --- Analysis & SPEED Settings ---
    'MAX_LAG_DAYS': 14,
    'MAIL_START_DATE': '2024-06-01',
    'USE_INNER_JOIN': True,  # 🚀 KEY SPEED SETTING: Ensures only matching dates are analyzed.
    'FILTER_WEEKENDS': True,
    'FILTER_HOLIDAYS': True,
    'LAG_CORR_SAMPLE_SIZE': 50000, # 🚀 KEY SPEED SETTING: Uses a sample for slow calculations if data is large.

    # --- Visual Settings ---
    'THEME_COLORS': {
        'calls': '#e74c3c',
        'mail': '#3498db',
        'financial': '#9b59b6',
        'correlation_positive': '#27ae60',
        'correlation_negative': '#c0392b',
    },
    'PLOT_WIDTH': 1600,
    'PLOT_HEIGHT': 900,
    'FONT_SIZE': 14,
    'TITLE_SIZE': 20
}

# =============================================================================
# SETUP (Unchanged)
# =============================================================================
def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger('PlotGenerator')

def safe_load_csv(file_path, description="file"):
    if not os.path.exists(file_path):
        LOGGER.warning(f"{description} not found: {file_path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip', encoding='utf-8', low_memory=False)
        LOGGER.info(f"✅ {description} loaded: {len(df):,} records from {file_path}")
        return df
    except Exception:
        try:
            df = pd.read_csv(file_path, on_bad_lines='skip', encoding='latin-1', low_memory=False)
            LOGGER.info(f"✅ {description} loaded with latin-1: {len(df):,} records")
            return df
        except Exception as e:
            LOGGER.error(f"❌ Failed to load {description}: {e}")
            return pd.DataFrame()

def preprocess_data(df, column_mapping, date_column):
    if df.empty: return df
    df = df.rename(columns=column_mapping)
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df.dropna(subset=[date_column], inplace=True)
        df[date_column] = df[date_column].dt.tz_localize(None).dt.normalize()
    return df

# =============================================================================
# DATA PROCESSING (Upgraded for Speed)
# =============================================================================
def get_processed_data(config):
    """
    Loads, processes, and filters data using the high-performance logic.
    Returns three dataframes:
    1. aggregated_df: For general plots (volumes merged and filtered).
    2. mail_df_raw: Raw mail data for intent analysis.
    3. call_df_raw: Raw call data (with intents if available) for intent analysis.
    """
    LOGGER.info("--- 🚀 Starting High-Performance Data Processing ---")

    # 1. Load Mail Data
    mail_df = safe_load_csv(config['MAIL_FILE_PATH'], "Mail data")
    mail_df = preprocess_data(mail_df, config['MAIL_COLUMNS'], 'date')

    # 2. Intelligent Call Data Loading
    call_df = pd.DataFrame()
    has_intent_data = False
    
    # Try primary file (with intents) first
    intent_df = safe_load_csv(config['CALL_FILE_PATH_INTENTS'], "Primary call data (with intents)")
    if not intent_df.empty:
        intent_df = preprocess_data(intent_df, config['CALL_INTENT_COLUMNS'], 'date')
        if 'intent' in intent_df.columns:
            call_df = intent_df
            has_intent_data = True
            LOGGER.info("✅ Using primary call file with intent data.")
        
    # Fallback to overview file if primary fails or lacks intents
    if call_df.empty:
        LOGGER.warning("⚠️ Primary call file failed or lacks intents. Using fallback overview file.")
        overview_df = safe_load_csv(config['CALL_FILE_PATH_OVERVIEW'], "Fallback call data (overview)")
        call_df = preprocess_data(overview_df, config['CALL_OVERVIEW_COLUMNS'], 'date')
        has_intent_data = False

    if mail_df.empty or call_df.empty:
        LOGGER.error("❌ Critical mail or call data missing. Halting.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), False

    # 3. Aggregate for merging
    daily_mail = mail_df.groupby('date')['volume'].sum().reset_index(name='mail_volume')
    daily_calls = call_df.groupby('date').size().reset_index(name='call_volume')

    # 4. High-Fidelity Merge (Inner join is much faster and cleaner)
    how_join = 'inner' if config['USE_INNER_JOIN'] else 'outer'
    aggregated_df = pd.merge(daily_mail, daily_calls, on='date', how=how_join).fillna(0)
    LOGGER.info(f"Merged data using '{how_join}' join, resulting in {len(aggregated_df)} initial records.")

    # 5. Apply Filters on the smaller, merged dataset
    aggregated_df = aggregated_df[aggregated_df['date'] >= pd.to_datetime(config['MAIL_START_DATE'])]
    if config['FILTER_WEEKENDS']:
        aggregated_df = aggregated_df[aggregated_df['date'].dt.weekday < 5]
    if config['FILTER_HOLIDAYS']:
        us_holidays = holidays.UnitedStates()
        aggregated_df = aggregated_df[~aggregated_df['date'].dt.date.isin(us_holidays)]

    aggregated_df = aggregated_df.sort_values('date').reset_index(drop=True)
    LOGGER.info(f"✅ Final processed dataset contains {len(aggregated_df)} clean business day records.")
    
    return aggregated_df, mail_df, call_df, has_intent_data

def add_financial_data(df, config):
    if not FINANCIAL_AVAILABLE or df.empty: return df
    LOGGER.info("--- Adding Financial Data ---")
    start_date, end_date = df['date'].min() - timedelta(days=5), df['date'].max() + timedelta(days=5)
    tickers = list(config['FINANCIAL_DATA'].values())
    names = list(config['FINANCIAL_DATA'].keys())
    ticker_map = dict(zip(tickers, names))
    try:
        fin_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        if fin_data.empty: return df
        close_prices = fin_data['Close']
        if isinstance(close_prices, pd.Series): close_prices = close_prices.to_frame(name=tickers[0])
        close_prices = close_prices.rename(columns=ticker_map)
        df = pd.merge(df, close_prices, left_on='date', right_index=True, how='left')
        for name in names:
            if name in df.columns: df[name] = df[name].ffill().bfill()
        LOGGER.info(f"✅ Successfully added financial data for: {', '.join(names)}")
    except Exception as e:
        LOGGER.warning(f"Could not download or process financial data: {e}")
    return df

def normalize_data(df):
    LOGGER.info("--- Normalizing Data ---")
    df_norm = df.copy()
    for col in df_norm.columns:
        if pd.api.types.is_numeric_dtype(df_norm[col]) and col != 'date':
            mean, std = df_norm[col].mean(), df_norm[col].std()
            if std > 0:
                df_norm[f'{col}_norm'] = (df_norm[col] - mean) / std
    return df_norm

# =============================================================================
# VISUALIZATION FUNCTIONS (Updated for Speed)
# =============================================================================
def save_plot(fig, filename, output_dir):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    path = os.path.join(output_dir, filename)
    try:
        fig.write_image(path, width=CONFIG['PLOT_WIDTH'], height=CONFIG['PLOT_HEIGHT'])
        LOGGER.info(f"✅ Plot saved successfully: {path}")
    except Exception as e:
        LOGGER.error(f"❌ Failed to save plot {filename}: {e}")

def create_and_save_overlay_plot(df, config):
    LOGGER.info("--- Creating Visual 1: Enhanced Overlay Timeline ---")
    if df.empty: return LOGGER.warning("No data for overlay plot.")
    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], vertical_spacing=0.1, subplot_titles=('📊 Normalized Communications & Market Trends', '📈 Raw Daily Volumes'), specs=[[{"secondary_y": True}], [{"secondary_y": True}]])
    colors = config['THEME_COLORS']

    # Top plot: Normalized data
    if 'mail_volume_norm' in df.columns: fig.add_trace(go.Scatter(x=df['date'], y=df['mail_volume_norm'], name='Mail Volume', line=dict(color=colors['mail'], width=4)), row=1, col=1, secondary_y=False)
    if 'call_volume_norm' in df.columns: fig.add_trace(go.Scatter(x=df['date'], y=df['call_volume_norm'], name='Call Volume', line=dict(color=colors['calls'], width=4)), row=1, col=1, secondary_y=False)
    
    financial_added = False
    for fin_col in config['FINANCIAL_DATA'].keys():
        if f'{fin_col}_norm' in df.columns:
            fig.add_trace(go.Scatter(x=df['date'], y=df[f'{fin_col}_norm'], name=fin_col, line=dict(color=colors['financial'], width=2, dash='dash'), opacity=0.8), row=1, col=1, secondary_y=True)
            financial_added = True

    # Bottom plot: Raw volumes (Fast Version)
    fig.add_trace(go.Scatter(x=df['date'], y=df['mail_volume'], name='Daily Mail', mode='lines', line=dict(width=0.5, color=colors['mail']), fill='tozeroy', fillcolor='rgba(52, 152, 219, 0.7)'), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=df['date'], y=df['call_volume'], name='Daily Calls', line=dict(color=colors['calls'], width=3), mode='lines'), row=2, col=1, secondary_y=True)

    fig.update_layout(title_text='<b>Customer Communications Intelligence Dashboard</b><br><sub>Comprehensive analysis of mail and call patterns with market indicators</sub>', template='plotly_white', height=config['PLOT_HEIGHT'], font_size=config['FONT_SIZE'])
    fig.update_yaxes(title_text="<b>Mail Volume</b>", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="<b>Call Volume</b>", row=2, col=1, secondary_y=True)

    LOGGER.info("Rendering '1_enhanced_overlay_trends.png'...")
    save_plot(fig, "1_enhanced_overlay_trends.png", config['OUTPUT_DIR'])

def create_and_save_lag_correlation_plot(df, config):
    LOGGER.info("--- Creating Visual 2: Lag Correlation Analysis ---")
    if df.empty or not SCIPY_AVAILABLE or len(df) < config['MAX_LAG_DAYS']: return LOGGER.warning("Not enough data for lag correlation plot.")

    # 🚀 SPEED ENHANCEMENT: Use a sample for the calculation if the dataset is large.
    sample_size = config['LAG_CORR_SAMPLE_SIZE']
    if len(df) > sample_size:
        LOGGER.warning(f"Dataset is large ({len(df)} rows). Using a random sample of {sample_size} for lag correlation to improve performance.")
        df = df.sample(n=sample_size, random_state=42)

    lags = range(config['MAX_LAG_DAYS'] + 1)
    correlations = []
    for lag in lags:
        try:
            # The calculation is now on a smaller (or original size) dataframe, making it fast.
            corr, _ = pearsonr(df['mail_volume'].iloc[:-lag if lag > 0 else None], df['call_volume'].iloc[lag:])
            correlations.append(corr if np.isfinite(corr) else 0)
        except:
            correlations.append(0)

    fig = go.Figure(go.Bar(x=[f'{lag}' for lag in lags], y=correlations, text=[f'{c:.3f}' for c in correlations], textposition='auto', marker_color=[config['THEME_COLORS']['correlation_positive'] if c > 0 else config['THEME_COLORS']['correlation_negative'] for c in correlations]))
    fig.update_layout(title_text='<b>Mail-to-Call Predictive Correlation</b>', xaxis_title='Lag Days', yaxis_title='Pearson Correlation', template='plotly_white')
    
    LOGGER.info("Rendering '2_enhanced_lag_correlation.png'...")
    save_plot(fig, "2_enhanced_lag_correlation.png", config['OUTPUT_DIR'])

def create_and_save_intent_correlation_plot(mail_df, call_df, config):
    LOGGER.info("--- Creating Visual 3: Intent Correlation ---")
    if mail_df.empty or call_df.empty or 'intent' not in call_df.columns: return LOGGER.warning("Missing data or intents for Visual 3.")

    # This function now receives data directly, not re-reading from disk
    daily_mail_types = mail_df.pivot_table(index='date', columns='type', values='volume', aggfunc='sum').fillna(0)
    daily_call_intents = call_df.pivot_table(index='date', columns='intent', aggfunc='size', fill_value=0)
    
    merged_intents = pd.merge(daily_mail_types, daily_call_intents, on='date', how='inner')
    if len(merged_intents) < 10: return LOGGER.warning("Not enough overlapping data for intent analysis.")

    all_correlations = []
    if SCIPY_AVAILABLE:
        for mail_type in daily_mail_types.columns:
            for call_intent in daily_call_intents.columns:
                if merged_intents[mail_type].std() > 0 and merged_intents[call_intent].std() > 0:
                    try:
                        corr, p_val = pearsonr(merged_intents[mail_type], merged_intents[call_intent])
                        if np.isfinite(corr): all_correlations.append({'mail_type': mail_type, 'call_intent': call_intent, 'correlation': corr, 'p_value': p_val})
                    except: continue
    
    if not all_correlations: return LOGGER.warning("Could not calculate any intent correlations.")

    corr_df = pd.DataFrame(all_correlations)
    corr_df['abs_corr'] = corr_df['correlation'].abs()
    top_correlations = corr_df.sort_values('abs_corr', ascending=False).head(15).sort_values('correlation')

    fig = go.Figure(go.Bar(y=[f"{row['mail_type']} → {row['call_intent']}" for _, row in top_correlations.iterrows()], x=top_correlations['correlation'], orientation='h', text=[f'{c:.3f}' for c in top_correlations['correlation']]))
    fig.update_layout(title_text='<b>Top Mail Type → Call Intent Correlations</b>', template='plotly_white', margin=dict(l=300))
    
    LOGGER.info("Rendering '3_enhanced_intent_correlations.png'...")
    save_plot(fig, "3_enhanced_intent_correlations.png", config['OUTPUT_DIR'])

def create_and_save_summary_dashboard(df, config):
    LOGGER.info("--- Creating Visual 4: Summary Dashboard ---")
    if df.empty: return LOGGER.warning("No data for summary dashboard.")
    fig = make_subplots(rows=2, cols=2, subplot_titles=('Daily Volumes', 'Volume Distribution', '7-Day Rolling Averages', 'Key Statistics'), specs=[[{"secondary_y": True}, {}], [{}, {"type": "table"}]])
    colors = config['THEME_COLORS']

    # Plot 1
    fig.add_trace(go.Scatter(x=df['date'], y=df['mail_volume'], name='Mail Volume', line=dict(color=colors['mail'])), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=df['date'], y=df['call_volume'], name='Call Volume', line=dict(color=colors['calls'])), row=1, col=1, secondary_y=True)
    # Plot 2
    fig.add_trace(go.Histogram(x=df['mail_volume'], name='Mail Dist.', marker_color=colors['mail'], opacity=0.7), row=1, col=2)
    fig.add_trace(go.Histogram(x=df['call_volume'], name='Call Dist.', marker_color=colors['calls'], opacity=0.7), row=1, col=2)
    # Plot 3
    if len(df) >= 7:
        fig.add_trace(go.Scatter(x=df['date'], y=df['mail_volume'].rolling(7).mean(), name='Mail 7D Avg', line=dict(color=colors['mail'], width=4)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['call_volume'].rolling(7).mean(), name='Call 7D Avg', line=dict(color=colors['calls'], width=4)), row=2, col=1)
    # Plot 4
    total_mail, total_calls = df['mail_volume'].sum(), df['call_volume'].sum()
    stats_data = {'Metric': ['Total Mail', 'Total Calls', 'Avg Daily Mail', 'Avg Daily Calls', 'Response Rate %'],
                  'Value': [f"{total_mail:,.0f}", f"{total_calls:,.0f}", f"{df['mail_volume'].mean():.0f}", f"{df['call_volume'].mean():.0f}", f"{(total_calls / total_mail * 100):.2f}" if total_mail > 0 else "0"]}
    fig.add_trace(go.Table(header=dict(values=['<b>Metric</b>', '<b>Value</b>'], fill_color=colors['mail'], font=dict(color='white')), cells=dict(values=[stats_data['Metric'], stats_data['Value']])), row=2, col=2)
    
    fig.update_layout(title_text='<b>Customer Communications Summary Dashboard</b>', template='plotly_white', height=config['PLOT_HEIGHT'])
    
    LOGGER.info("Rendering '4_summary_dashboard.png'...")
    save_plot(fig, "4_summary_dashboard.png", config['OUTPUT_DIR'])

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == '__main__':
    LOGGER = setup_logging()
    LOGGER.info("🚀 Starting Plot Generator v1.3 (High-Performance)...")
    os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)

    # 1. Process all data using the new high-performance function
    aggregated_df, mail_df_raw, call_df_raw, has_intent_data = get_processed_data(CONFIG)
    
    if not aggregated_df.empty:
        # 2. Add financial data and normalize
        with_financials = add_financial_data(aggregated_df.copy(), CONFIG)
        normalized_df = normalize_data(with_financials)
        
        # 3. Create all visualizations with the processed data
        create_and_save_overlay_plot(normalized_df, CONFIG)
        create_and_save_lag_correlation_plot(aggregated_df.copy(), CONFIG)
        if has_intent_data:
            create_and_save_intent_correlation_plot(mail_df_raw, call_df_raw, CONFIG)
        else:
            LOGGER.warning("Skipping intent correlation plot as no intent data was found.")
        create_and_save_summary_dashboard(aggregated_df.copy(), CONFIG)
        
        LOGGER.info("✅ All plots generated successfully!")
        LOGGER.info(f"📁 Check the '{CONFIG['OUTPUT_DIR']}' directory.")
    else:
        LOGGER.error("❌ No data available for analysis after processing. Please check file paths and data content.")

    LOGGER.info("🚀 Analysis complete.")
