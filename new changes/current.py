#!/usr/bin/env python3
"""
Customer Communications Intelligence Plot Generator v1.2
Performance and logging enhancements included.
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
# FIXED CONFIGURATION
# =============================================================================
CONFIG = {
    # --- File Paths ---
    'CALL_FILE_PATH_VOLUME': r'data\GenesysExtract_20250703.csv',
    'CALL_FILE_PATH_INTENTS': r'data\GenesysExtract_20250609.csv',
    'MAIL_FILE_PATH': r'merged_output.csv',

    # --- Output Directory ---
    'OUTPUT_DIR': 'plots',

    # --- FIXED Column Mappings ---
    'MAIL_COLUMNS': {'date': 'mail_date', 'volume': 'mail_volume', 'type': 'mail_type'},
    
    # Different date columns for different call files
    'CALL_VOLUME_COLUMNS': {'date': 'Date'},  # For volume file
    'CALL_INTENT_COLUMNS': {'date': 'ConversationStart', 'intent': 'uui_Intent'},  # For intent file

    # --- Financial Indicators ---
    'FINANCIAL_DATA': {
        'S&P 500': '^GSPC',
        'VIX': '^VIX',
        'Dollar Index': 'DX-Y.NYB'
    },

    # --- Analysis Settings ---
    'MAX_LAG_DAYS': 14,
    'MAIL_START_DATE': '2024-06-01',
    'USE_INNER_JOIN': True,
    'FILTER_WEEKENDS': True,
    'FILTER_HOLIDAYS': True,

    # --- Visual Settings ---
    'THEME_COLORS': {
        'calls': '#e74c3c',
        'mail': '#3498db',
        'financial': '#9b59b6',
        'correlation_positive': '#27ae60',
        'correlation_negative': '#c0392b',
        'momentum': '#f39c12',
        'spike': '#f1c40f'
    },

    # Plot settings
    'PLOT_WIDTH': 1600,
    'PLOT_HEIGHT': 900,
    'FONT_SIZE': 14,
    'TITLE_SIZE': 20
}

# =============================================================================
# SETUP
# =============================================================================
def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger('PlotGenerator')

def safe_load_csv(file_path, description="file"):
    """Safely load a CSV file with multiple encoding attempts."""
    if not os.path.exists(file_path):
        LOGGER.warning(f"{description} not found: {file_path}")
        return pd.DataFrame()
    
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip', low_memory=False)
            LOGGER.info(f"✅ {description} loaded: {len(df):,} records from {file_path}")
            return df
        except Exception as e:
            continue
    
    LOGGER.error(f"❌ Failed to load {description} from {file_path}")
    return pd.DataFrame()

def preprocess_data(df, column_mapping, date_column, data_type="data"):
    """Standardize columns and convert dates with better error handling."""
    if df.empty:
        return df
    
    LOGGER.info(f"Processing {data_type} with columns: {list(df.columns)}")
    
    # Rename columns based on mapping
    rename_dict = {}
    for standard_name, config_name in column_mapping.items():
        if config_name in df.columns:
            rename_dict[config_name] = standard_name
        else:
            LOGGER.warning(f"Column '{config_name}' not found in {data_type}. Available columns: {list(df.columns)}")
    
    if rename_dict:
        df = df.rename(columns=rename_dict)
        LOGGER.info(f"Renamed columns for {data_type}: {rename_dict}")
    
    # Convert date column
    if date_column in df.columns:
        LOGGER.info(f"Converting date column '{date_column}' for {data_type}")
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        before_count = len(df)
        df = df.dropna(subset=[date_column])
        after_count = len(df)
        LOGGER.info(f"Date conversion: {before_count} -> {after_count} records")
        df[date_column] = df[date_column].dt.tz_localize(None).dt.normalize()
    else:
        LOGGER.error(f"Date column '{date_column}' not found in {data_type}")
    
    return df

# =============================================================================
# ENHANCED DATA PROCESSING
# =============================================================================

def get_base_data(config):
    """Loads and prepares the primary mail and call volume data with fixed column handling."""
    LOGGER.info("--- Loading Base Data for Volume Analysis ---")
    
    # Load mail data
    mail_df = safe_load_csv(config['MAIL_FILE_PATH'], "Mail data")
    mail_df = preprocess_data(mail_df, config['MAIL_COLUMNS'], 'date', "mail data")

    # Load call volume data (using the volume file configuration)
    call_df_volume = safe_load_csv(config['CALL_FILE_PATH_VOLUME'], "Call volume data")
    call_df_volume = preprocess_data(call_df_volume, config['CALL_VOLUME_COLUMNS'], 'date', "call volume data")

    # Aggregate daily volumes
    if not mail_df.empty and 'volume' in mail_df.columns:
        daily_mail = mail_df.groupby('date')['volume'].sum().reset_index(name='mail_volume')
    else:
        daily_mail = pd.DataFrame(columns=['date', 'mail_volume'])
    
    if not call_df_volume.empty:
        daily_calls = call_df_volume.groupby('date').size().reset_index(name='call_volume')
    else:
        daily_calls = pd.DataFrame(columns=['date', 'call_volume'])

    # Merge data
    if not daily_mail.empty and not daily_calls.empty:
        how_join = 'inner' if config['USE_INNER_JOIN'] else 'outer'
        combined_df = pd.merge(daily_mail, daily_calls, on='date', how=how_join).fillna(0)
    elif not daily_mail.empty:
        combined_df = daily_mail.copy()
        combined_df['call_volume'] = 0
    elif not daily_calls.empty:
        combined_df = daily_calls.copy()
        combined_df['mail_volume'] = 0
    else:
        LOGGER.error("No data to combine")
        return pd.DataFrame()

    # Filter by date and remove weekends/holidays
    if not combined_df.empty:
        combined_df = combined_df[combined_df['date'] >= pd.to_datetime(config['MAIL_START_DATE'])]
        
        if config['FILTER_WEEKENDS']:
            combined_df = combined_df[combined_df['date'].dt.weekday < 5]
        
        if config['FILTER_HOLIDAYS']:
            us_holidays = holidays.UnitedStates()
            combined_df = combined_df[~combined_df['date'].dt.date.isin(us_holidays)]

    return combined_df.sort_values('date').reset_index(drop=True) if not combined_df.empty else pd.DataFrame()

def add_financial_data(df, config):
    """Adds financial data to the dataframe."""
    if not FINANCIAL_AVAILABLE or df.empty:
        return df
    
    LOGGER.info("--- Adding Financial Data ---")
    start_date = df['date'].min() - timedelta(days=5)
    end_date = df['date'].max() + timedelta(days=5)
    
    # Get tickers and their friendly names from config
    tickers = list(config['FINANCIAL_DATA'].values())
    names = list(config['FINANCIAL_DATA'].keys())
    ticker_to_name_map = dict(zip(tickers, names))

    try:
        # --- EFFICIENT CHANGE: Download all tickers at once ---
        fin_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        
        if fin_data.empty:
            LOGGER.warning("Financial data download returned an empty dataframe.")
            return df

        # --- ROBUST FIX: Isolate 'Close' prices and handle single/multi ticker cases ---
        close_prices = fin_data['Close']
        
        # If only one ticker is downloaded, it returns a Series; convert it to a DataFrame
        if isinstance(close_prices, pd.Series):
            close_prices = close_prices.to_frame(name=tickers[0])

        # Rename columns from ticker symbols (e.g., '^GSPC') to friendly names (e.g., 'S&P 500')
        close_prices = close_prices.rename(columns=ticker_to_name_map)
        
        # --- CORRECTED MERGE: Merge the clean data ---
        df = pd.merge(df, close_prices, left_on='date', right_index=True, how='left')
        
        # Forward-fill and back-fill any missing values for all new columns
        for name in names:
            if name in df.columns:
                df[name] = df[name].ffill().bfill()
        
        LOGGER.info(f"✅ Successfully added financial data for: {', '.join(names)}")

    except Exception as e:
        LOGGER.warning(f"Could not download or process financial data: {e}")
    
    return df

def normalize_data(df):
    """Normalize data using Z-score for plotting."""
    LOGGER.info("--- Normalizing Data ---")
    
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64'] and col != 'date':
            # Remove outliers beyond 3 standard deviations before normalizing
            mean, std = df[col].mean(), df[col].std()
            if std > 0:
                df = df[np.abs(df[col] - mean) <= (3 * std)]
                # Z-score normalization
                df[f'{col}_norm'] = (df[col] - df[col].mean()) / df[col].std()
    
    return df

# =============================================================================
# ENHANCED VISUALIZATION FUNCTIONS
# =============================================================================

def save_plot(fig, filename, output_dir):
    """Saves a Plotly figure as a PNG."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    path = os.path.join(output_dir, filename)
    try:
        fig.write_image(path, width=CONFIG['PLOT_WIDTH'], height=CONFIG['PLOT_HEIGHT'])
        LOGGER.info(f"✅ Plot saved successfully: {path}")
    except Exception as e:
        LOGGER.error(f"❌ Failed to save plot {filename}. Ensure you have 'kaleido' installed (`pip install kaleido`): {e}")

### VISUAL 1: Enhanced Overlay Timeline ###
def create_and_save_overlay_plot(df, config):
    """Creates and saves an enhanced overlay plot with better readability."""
    LOGGER.info("--- Creating Visual 1: Enhanced Overlay Timeline ---")
    
    if df.empty:
        LOGGER.warning("No data for overlay plot.")
        return

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.1,
        subplot_titles=('📊 Normalized Communications & Market Trends', '📈 Raw Daily Volumes'),
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )

    colors = config['THEME_COLORS']

    # Top plot: Normalized data
    if 'mail_volume_norm' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['date'], 
                y=df['mail_volume_norm'], 
                name='Mail Volume',
                line=dict(color=colors['mail'], width=4),
                mode='lines'
            ),
            row=1, col=1, secondary_y=False
        )

    if 'call_volume_norm' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['date'], 
                y=df['call_volume_norm'], 
                name='Call Volume',
                line=dict(color=colors['calls'], width=4),
                mode='lines'
            ),
            row=1, col=1, secondary_y=False
        )

    # Add financial data
    financial_added = False
    for fin_col in config['FINANCIAL_DATA'].keys():
        if f'{fin_col}_norm' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['date'], 
                    y=df[f'{fin_col}_norm'], 
                    name=f'{fin_col}',
                    line=dict(color=colors['financial'], width=2, dash='dash'),
                    opacity=0.8
                ),
                row=1, col=1, secondary_y=True
            )
            financial_added = True

    # --- 🚀 PERFORMANCE FIX APPLIED HERE 🚀 ---
    # Bottom plot: Raw volumes using faster Scatter instead of Bar
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['mail_volume'],
            name='Daily Mail',
            mode='lines',
            line=dict(width=0.5, color=colors['mail']),
            fill='tozeroy',  # Fills the area to look like a bar/area chart
            fillcolor='rgba(52, 152, 219, 0.7)', # Explicit color with opacity
            yaxis='y3'
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['call_volume'],
            name='Daily Calls',
            line=dict(color=colors['calls'], width=3),
            mode='lines',
            yaxis='y3'
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text='<b>Customer Communications Intelligence Dashboard</b><br><sub>Comprehensive analysis of mail and call patterns with market indicators</sub>',
            font=dict(size=config['TITLE_SIZE'], color='#2c3e50'),
            x=0.5
        ),
        template='plotly_white',
        height=config['PLOT_HEIGHT'],
        font=dict(size=config['FONT_SIZE']),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=120, b=60, l=80, r=80)
    )

    # Update axes
    fig.update_yaxes(title_text="<b>Normalized Score</b>", row=1, col=1, secondary_y=False)
    if financial_added:
        fig.update_yaxes(title_text="<b>Market Indicators</b>", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="<b>Volume</b>", row=2, col=1)
    fig.update_xaxes(title_text="<b>Date</b>", row=2, col=1)

    # --- 🔍 LOGGING FIX APPLIED HERE 🔍 ---
    LOGGER.info("Building complete. Now rendering '1_enhanced_overlay_trends.png'...")
    save_plot(fig, "1_enhanced_overlay_trends.png", config['OUTPUT_DIR'])
    LOGGER.info("✅ Visual 1 finished.")


### VISUAL 2: Enhanced Lag Correlation ###
def create_and_save_lag_correlation_plot(df, config):
    """Creates an enhanced lag correlation plot."""
    LOGGER.info("--- Creating Visual 2: Enhanced Lag Correlation Analysis ---")
    
    if df.empty or len(df) < config['MAX_LAG_DAYS'] or not SCIPY_AVAILABLE:
        LOGGER.warning("Not enough data for lag correlation plot or scipy not available.")
        return

    lags = range(0, config['MAX_LAG_DAYS'] + 1)
    correlations = []
    p_values = []
    
    for lag in lags:
        if len(df) > lag:
            try:
                # Shift call volume back to see if mail volume predicts future calls
                if lag > 0:
                    corr, p_val = pearsonr(df['mail_volume'].iloc[:-lag], df['call_volume'].iloc[lag:])
                else:
                    corr, p_val = pearsonr(df['mail_volume'], df['call_volume'])
                
                correlations.append(corr)
                p_values.append(p_val)
            except:
                correlations.append(0)
                p_values.append(1)
        else:
            correlations.append(0)
            p_values.append(1)

    # Create enhanced bar chart
    fig = go.Figure()
    colors = config['THEME_COLORS']
    
    # Color bars based on correlation strength and significance
    bar_colors = []
    for corr, p_val in zip(correlations, p_values):
        if p_val < 0.05:  # Significant
            if corr > 0:
                bar_colors.append(colors['correlation_positive'])
            else:
                bar_colors.append(colors['correlation_negative'])
        else:  # Not significant
            bar_colors.append('#95a5a6')

    fig.add_trace(go.Bar(
        x=[f'{lag}' for lag in lags],
        y=correlations,
        marker_color=bar_colors,
        text=[f'{c:.3f}' for c in correlations],
        textposition='auto',
        textfont=dict(size=12, color='white'),
        hovertemplate='<b>Lag: %{x} days</b><br>Correlation: %{y:.3f}<br>P-value: %{customdata:.3f}<extra></extra>',
        customdata=p_values
    ))

    # Add reference lines
    fig.add_hline(y=0, line_dash="solid", line_color="#000000", line_width=1)
    fig.add_hline(y=0.3, line_dash="dash", line_color=colors['correlation_positive'], line_width=2)
    fig.add_hline(y=-0.3, line_dash="dash", line_color=colors['correlation_negative'], line_width=2)

    fig.update_layout(
        title=dict(
            text='<b>📊 Mail-to-Call Predictive Correlation Analysis</b><br><sub>How well does mail volume predict future call volume? (Green = significant positive, Red = significant negative, Gray = not significant)</sub>',
            font=dict(size=config['TITLE_SIZE'], color='#2c3e50'),
            x=0.5
        ),
        xaxis_title='<b>Lag Days (Mail leads Calls by X days)</b>',
        yaxis_title='<b>Pearson Correlation Coefficient</b>',
        template='plotly_white',
        height=config['PLOT_HEIGHT'],
        font=dict(size=config['FONT_SIZE']),
        margin=dict(t=120, b=80, l=80, r=80)
    )

    LOGGER.info("Building complete. Now rendering '2_enhanced_lag_correlation.png'...")
    save_plot(fig, "2_enhanced_lag_correlation.png", config['OUTPUT_DIR'])
    LOGGER.info("✅ Visual 2 finished.")


### VISUAL 3: Enhanced Intent Correlation ###
def create_and_save_intent_correlation_plot(config):
    """Creates an enhanced intent correlation plot with proper date column handling."""
    LOGGER.info("--- Creating Visual 3: Enhanced Intent Correlation ---")
    
    # Load data with correct column mappings
    mail_df = safe_load_csv(config['MAIL_FILE_PATH'], "Mail data")
    call_df_intents = safe_load_csv(config['CALL_FILE_PATH_INTENTS'], "Call intent data")

    if mail_df.empty or call_df_intents.empty:
        LOGGER.warning("Missing mail or call intent data. Skipping Visual 3.")
        return

    # Preprocess with correct column mappings
    mail_df = preprocess_data(mail_df, config['MAIL_COLUMNS'], 'date', "mail data")
    call_df_intents = preprocess_data(call_df_intents, config['CALL_INTENT_COLUMNS'], 'date', "call intent data")
    
    if mail_df.empty or call_df_intents.empty:
        LOGGER.warning("Data preprocessing failed. Skipping Visual 3.")
        return

    # Check if intent column exists
    if 'intent' not in call_df_intents.columns:
        LOGGER.warning("Intent column not found in call data. Skipping Visual 3.")
        return

    # Pivot to get daily counts for each mail type and call intent
    if 'type' in mail_df.columns and 'volume' in mail_df.columns:
        daily_mail_types = mail_df.pivot_table(index='date', columns='type', values='volume', aggfunc='sum').fillna(0)
    else:
        daily_mail_types = mail_df.groupby('date')['volume'].sum().reset_index()
        daily_mail_types = daily_mail_types.set_index('date')
        daily_mail_types.columns = ['total_mail']

    daily_call_intents = call_df_intents.pivot_table(index='date', columns='intent', aggfunc='size', fill_value=0)
    
    # Merge on date
    merged_intents = pd.merge(daily_mail_types, daily_call_intents, on='date', how='inner')
    
    if len(merged_intents) < 10:
        LOGGER.warning("Not enough overlapping intent data to analyze. Skipping Visual 3.")
        return

    # Calculate correlations
    all_correlations = []
    
    if SCIPY_AVAILABLE:
        for mail_type in daily_mail_types.columns:
            for call_intent in daily_call_intents.columns:
                if merged_intents[mail_type].std() > 0 and merged_intents[call_intent].std() > 0:
                    try:
                        corr, p_val = pearsonr(merged_intents[mail_type], merged_intents[call_intent])
                        all_correlations.append({
                            'mail_type': mail_type,
                            'call_intent': call_intent,
                            'correlation': corr,
                            'p_value': p_val
                        })
                    except:
                        continue
    
    if not all_correlations:
        LOGGER.warning("Could not calculate any intent correlations.")
        return

    # Get top 15 absolute correlations for better readability
    corr_df = pd.DataFrame(all_correlations)
    corr_df['abs_corr'] = corr_df['correlation'].abs()
    top_correlations = corr_df.sort_values('abs_corr', ascending=False).head(15)
    top_correlations = top_correlations.sort_values('correlation', ascending=True)

    # Create enhanced horizontal bar chart
    fig = go.Figure()
    colors = config['THEME_COLORS']
    
    # Color bars based on correlation strength and significance
    bar_colors = []
    for _, row in top_correlations.iterrows():
        if row['p_value'] < 0.05:  # Significant
            if row['correlation'] > 0:
                bar_colors.append(colors['correlation_positive'])
            else:
                bar_colors.append(colors['correlation_negative'])
        else:  # Not significant
            bar_colors.append('#95a5a6')
    
    fig.add_trace(go.Bar(
        y=[f"{row['mail_type']} → {row['call_intent']}" for _, row in top_correlations.iterrows()],
        x=top_correlations['correlation'],
        orientation='h',
        marker_color=bar_colors,
        text=[f'{c:.3f}' for c in top_correlations['correlation']],
        textposition='auto',
        textfont=dict(size=11, color='white'),
        hovertemplate='<b>%{y}</b><br>Correlation: %{x:.3f}<br>P-value: %{customdata:.3f}<extra></extra>',
        customdata=top_correlations['p_value']
    ))

    fig.update_layout(
        title=dict(
            text='<b>📊 Top Mail Type → Call Intent Correlations</b><br><sub>Which mail communications correlate with specific customer call types? (Green = significant positive, Red = significant negative, Gray = not significant)</sub>',
            font=dict(size=config['TITLE_SIZE'], color='#2c3e50'),
            x=0.5
        ),
        xaxis_title='<b>Pearson Correlation Coefficient</b>',
        yaxis_title='<b>Mail Type → Call Intent</b>',
        template='plotly_white',
        height=config['PLOT_HEIGHT'],
        font=dict(size=config['FONT_SIZE']),
        margin=dict(t=120, b=80, l=300, r=80)
    )
    
    LOGGER.info("Building complete. Now rendering '3_enhanced_intent_correlations.png'...")
    save_plot(fig, "3_enhanced_intent_correlations.png", config['OUTPUT_DIR'])
    LOGGER.info("✅ Visual 3 finished.")


### VISUAL 4: Combined Summary Dashboard ###
def create_and_save_summary_dashboard(df, config):
    """Creates a comprehensive summary dashboard."""
    LOGGER.info("--- Creating Visual 4: Summary Dashboard ---")
    
    if df.empty:
        LOGGER.warning("No data for summary dashboard.")
        return

    # Create 2x2 subplot layout
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '📈 Daily Volumes Over Time',
            '📊 Volume Distribution',
            '📉 7-Day Rolling Averages',
            '🔍 Key Statistics'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    colors = config['THEME_COLORS']

    # Plot 1: Daily volumes
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['mail_volume'],
            name='Mail Volume',
            line=dict(color=colors['mail'], width=3),
            mode='lines'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['call_volume'],
            name='Call Volume',
            line=dict(color=colors['calls'], width=3),
            mode='lines',
            yaxis='y2'
        ),
        row=1, col=1
    )

    # Plot 2: Histograms
    fig.add_trace(
        go.Histogram(
            x=df['mail_volume'],
            name='Mail Distribution',
            marker_color=colors['mail'],
            opacity=0.7,
            nbinsx=20
        ),
        row=1, col=2
    )

    # Plot 3: Rolling averages
    if len(df) >= 7:
        df['mail_7d'] = df['mail_volume'].rolling(window=7).mean()
        df['call_7d'] = df['call_volume'].rolling(window=7).mean()
        
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['mail_7d'],
                name='Mail 7-Day Avg',
                line=dict(color=colors['mail'], width=4),
                mode='lines'
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['call_7d'],
                name='Call 7-Day Avg',
                line=dict(color=colors['calls'], width=4),
                mode='lines'
            ),
            row=2, col=1
        )

    # Plot 4: Key statistics table
    stats_data = {
        'Metric': ['Total Mail', 'Total Calls', 'Avg Daily Mail', 'Avg Daily Calls', 'Response Rate %'],
        'Value': [
            f"{df['mail_volume'].sum():,.0f}",
            f"{df['call_volume'].sum():,.0f}",
            f"{df['mail_volume'].mean():.0f}",
            f"{df['call_volume'].mean():.0f}",
            f"{(df['call_volume'].sum() / df['mail_volume'].sum() * 100):.2f}" if df['mail_volume'].sum() > 0 else "0"
        ]
    }

    fig.add_trace(
        go.Table(
            header=dict(
                values=['<b>Metric</b>', '<b>Value</b>'],
                fill_color=colors['mail'],
                font=dict(color='white', size=14)
            ),
            cells=dict(
                values=[stats_data['Metric'], stats_data['Value']],
                fill_color='white',
                font=dict(size=12)
            )
        ),
        row=2, col=2
    )

    fig.update_layout(
        title=dict(
            text='<b>📊 Customer Communications Summary Dashboard</b><br><sub>Comprehensive overview of mail and call patterns</sub>',
            font=dict(size=config['TITLE_SIZE'], color='#2c3e50'),
            x=0.5
        ),
        template='plotly_white',
        height=config['PLOT_HEIGHT'],
        font=dict(size=config['FONT_SIZE']),
        showlegend=True,
        margin=dict(t=120, b=60, l=80, r=80)
    )

    LOGGER.info("Building complete. Now rendering '4_summary_dashboard.png'...")
    save_plot(fig, "4_summary_dashboard.png", config['OUTPUT_DIR'])
    LOGGER.info("✅ Visual 4 finished.")

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == '__main__':
    LOGGER = setup_logging()
    LOGGER.info("🚀 Starting Customer Communications Plot Generator v1.2...")

    # Create output directory
    os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)

    # Load and process data
    base_df = get_base_data(CONFIG)
    
    if not base_df.empty:
        LOGGER.info(f"Loaded {len(base_df)} records for analysis")
        
        # Add financial data if available
        base_with_financials = add_financial_data(base_df.copy(), CONFIG)
        
        # Normalize for overlay plot
        normalized_df = normalize_data(base_with_financials)
        
        # Create all visualizations
        create_and_save_overlay_plot(normalized_df, CONFIG)
        create_and_save_lag_correlation_plot(base_df.copy(), CONFIG)
        create_and_save_intent_correlation_plot(CONFIG)
        create_and_save_summary_dashboard(base_df.copy(), CONFIG)
        
        LOGGER.info("✅ All plots generated successfully!")
        LOGGER.info(f"📁 Check the '{CONFIG['OUTPUT_DIR']}' directory for the following plots:")
        LOGGER.info("   📈 1_enhanced_overlay_trends.png - Normalized communications & market trends")
        LOGGER.info("   📊 2_enhanced_lag_correlation.png - Predictive correlation analysis")
        LOGGER.info("   🎯 3_enhanced_intent_correlations.png - Mail type to call intent correlations")
        LOGGER.info("   📋 4_summary_dashboard.png - Comprehensive overview dashboard")
        
    else:
        LOGGER.error("❌ No data available for analysis. Please check:")
        LOGGER.error("   - Mail file exists and has data")
        LOGGER.error("   - Call volume file exists and has data")
        LOGGER.error("   - Date columns are properly formatted")
        LOGGER.error("   - Files are accessible and not corrupted")

    LOGGER.info("🎯 PLOT DESCRIPTIONS:")
    LOGGER.info("=" * 60)
    LOGGER.info("📈 PLOT 1: Enhanced Overlay Trends")
    LOGGER.info("   • Top panel: Normalized mail/call volumes + financial indicators")
    LOGGER.info("   • Bottom panel: Raw daily volumes (area chart + line for performance)")
    LOGGER.info("   • Shows patterns and correlations over time")
    LOGGER.info("")
    LOGGER.info("📊 PLOT 2: Enhanced Lag Correlation")
    LOGGER.info("   • Bar chart showing correlation at different time lags")
    LOGGER.info("   • Green bars = significant positive correlation")
    LOGGER.info("   • Red bars = significant negative correlation")
    LOGGER.info("   • Gray bars = not statistically significant")
    LOGGER.info("   • Shows how well mail volume predicts future calls")
    LOGGER.info("")
    LOGGER.info("🎯 PLOT 3: Enhanced Intent Correlations")
    LOGGER.info("   • Horizontal bar chart of top 15 correlations")
    LOGGER.info("   • Shows which mail types correlate with specific call intents")
    LOGGER.info("   • Format: 'Mail Type → Call Intent'")
    LOGGER.info("   • Color coded for significance (same as Plot 2)")
    LOGGER.info("")
    LOGGER.info("📋 PLOT 4: Summary Dashboard")
    LOGGER.info("   • 2x2 grid with comprehensive overview")
    LOGGER.info("   • Daily volumes, distribution histograms")
    LOGGER.info("   • 7-day rolling averages, key statistics table")
    LOGGER.info("   • Perfect for executive presentations")
    LOGGER.info("=" * 60)

    LOGGER.info("✅ Plot generation process complete.")
    LOGGER.info(f"💾 All plots saved as high-resolution PNG files ({CONFIG['PLOT_WIDTH']}x{CONFIG['PLOT_HEIGHT']})")
    
    # Print file locations for easy access
    plot_files = [
        "1_enhanced_overlay_trends.png",
        "2_enhanced_lag_correlation.png", 
        "3_enhanced_intent_correlations.png",
        "4_summary_dashboard.png"
    ]
    
    LOGGER.info("📂 Direct file paths:")
    for plot_file in plot_files:
        full_path = os.path.join(CONFIG['OUTPUT_DIR'], plot_file)
        if os.path.exists(full_path):
            LOGGER.info(f"   ✅ {full_path}")
        else:
            LOGGER.info(f"   ❌ {full_path} (not created)")
    
    LOGGER.info("🚀 Ready for analysis and presentation!")
