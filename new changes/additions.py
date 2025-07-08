Understood. My apologies, the previous response was not what you asked for. Let's go through the file you provided and make the specific line-by-line adjustments needed.
Here are the exact changes to fix the date merge error and add the new visualizations.
1. Fix the Date Merge Error
The error occurs because the financial data has a timezone (UTC) while your CSV data does not. We need to make them consistent.
File: your_script_name.py
Change 1: In the safe_date_conversion function
This ensures your mail and call dates are timezone-naive from the start.
 * Line ~202 (Before):
   df[date_column] = df[date_column].dt.normalize()

 * Line ~202 (After):
   df[date_column] = df[date_column].dt.tz_localize(None).dt.normalize()

Change 2: In the safe_financial_data function
This removes the timezone from the financial data right after it's loaded.
 * Line ~281 (Before):
   self.financial_df['date'] = pd.to_datetime(self.financial_df['date']).dt.normalize()

 * Line ~281 (After):
   # FIX: Remove timezone information to allow merging
self.financial_df['date'] = pd.to_datetime(self.financial_df['date']).dt.tz_localize(None).dt.normalize()

2. Add New Plots and Analysis
Step 1: Add a new analysis function to the SafeDataProcessor class.
Place this new function right after the analyze_correlation method (around line 386).
 * Add New Function:
   # --- ADDITION: New function for intent-level correlation ---
def analyze_intent_correlation(self):
    """Calculate lagged correlation for each mail_type vs each call_intent."""
    if not SCIPY_AVAILABLE or self.mail_df.empty or self.call_df.empty:
        self.logger.warning("Skipping intent correlation (SciPy, mail, or call data unavailable).")
        self.intent_correlation_matrix = pd.DataFrame() # Ensure it's an empty dataframe
        return False

    self.logger.info("🔄 Running intent-level correlation analysis...")
    try:
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
                correlations = [pearsonr(mail_pivot[mail_type], call_pivot[call_intent].shift(-lag).fillna(0))[0] for lag in range(self.config['MAX_LAG_DAYS'] + 1)]
                valid_correlations = [c for c in correlations if np.isfinite(c)]
                correlation_matrix.loc[mail_type, call_intent] = max(valid_correlations) if valid_correlations else 0

        self.intent_correlation_matrix = correlation_matrix
        self.logger.info("✅ Intent correlation analysis complete.")
        return True
    except Exception as e:
        self.logger.error(f"❌ Intent correlation analysis failed: {e}")
        self.intent_correlation_matrix = pd.DataFrame()
        return False

Step 2: Add the new plot functions to the SafeDashboard class.
Place these two new functions right after the create_call_intent_figure method (around line 495).
 * Add New Functions:
   # --- ADDITION: Heatmap for type-to-intent correlation ---
def create_intent_correlation_heatmap(self, df_corr):
    if df_corr.empty:
        return self.create_error_figure("Intent correlation data not available.")
    fig = px.imshow(df_corr,
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale='Blues',
                    title="Peak Correlation: Mail Type vs. Call Intent")
    fig.update_layout(template='plotly_white', height=500, xaxis_title="Call Intent", yaxis_title="Mail Type")
    return fig

# --- ADDITION: Chart for call intents by weekday ---
def create_weekday_intent_figure(self):
    df = self.dp.call_df
    if df.empty:
        return self.create_error_figure("Call data unavailable.")

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df['weekday'] = pd.Categorical(df['date'].dt.day_name(), categories=weekday_order, ordered=True)

    weekday_counts = df.groupby(['weekday', 'intent']).size().reset_index(name='count')

    fig = px.bar(weekday_counts, x='weekday', y='count', color='intent',
                 title="Call Volume by Intent and Day of the Week",
                 labels={'weekday': 'Day of the Week', 'count': 'Number of Calls', 'intent': 'Call Intent'})
    fig.update_layout(template='plotly_white', height=500)
    return fig

Step 3: Update the create_overview_figure to include moving averages.
 * Lines ~440-456 (Before):
   def create_overview_figure(self, df):
    if df.empty:
        return self.create_error_figure("No data available for this period")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add Mail Volume Bar Chart
    fig.add_trace(go.Bar(
        x=df['date'], y=df['mail_volume'], name='Mail Volume',
        marker_color='lightskyblue', opacity=0.7
    ), secondary_y=False)

    # Add Call Volume Line Chart
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['call_volume'], name='Call Volume',
        mode='lines+markers', line=dict(color='navy', width=3)
    ), secondary_y=True)

    fig.update_layout(
        title="Daily Mail vs. Call Volume", template='plotly_white', height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="<b>Mail Volume</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Call Volume</b>", secondary_y=True)
    return fig

 * Lines ~440-461 (After):
   def create_overview_figure(self, df):
    if df.empty:
        return self.create_error_figure("No data available for this period")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # --- ADDITION: Calculate moving averages ---
    df['mail_7d_ma'] = df['mail_volume'].rolling(window=7).mean()
    df['call_7d_ma'] = df['call_volume'].rolling(window=7).mean()

    # Add Mail Volume Bar Chart
    fig.add_trace(go.Bar(x=df['date'], y=df['mail_volume'], name='Mail Volume', marker_color='lightskyblue', opacity=0.6), secondary_y=False)
    # --- ADDITION: Add mail moving average line ---
    fig.add_trace(go.Scatter(x=df['date'], y=df['mail_7d_ma'], name='Mail 7D MA', mode='lines', line=dict(color='blue', dash='dash')), secondary_y=False)

    # Add Call Volume Line Chart
    fig.add_trace(go.Scatter(x=df['date'], y=df['call_volume'], name='Call Volume', mode='lines', line=dict(color='salmon')), secondary_y=True)
    # --- ADDITION: Add call moving average line ---
    fig.add_trace(go.Scatter(x=df['date'], y=df['call_7d_ma'], name='Call 7D MA', mode='lines', line=dict(color='red', dash='dash')), secondary_y=True)

    fig.update_layout(
        title="Daily Mail vs. Call Volume (with 7-Day Moving Average)", template='plotly_white', height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="<b>Mail Volume</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Call Volume</b>", secondary_y=True)
    return fig

Step 4: Update the dashboard layout to include the new tab.
 * Lines ~542-561 (Before):
   # Tabbed Interface for Charts
dbc.Tabs([
    dbc.Tab(label="📈 Overview", tab_id="tab-overview", children=[
        dcc.Graph(id='overview-chart')
    ]),
    dbc.Tab(label="🔗 Correlation Analysis", tab_id="tab-correlation", children=[
        dbc.Row([
            dbc.Col(dcc.Graph(id='correlation-chart'), width=8),
            dbc.Col(dbc.Card(id='correlation-kpi', body=True, color="light"), width=4)
        ], className="mt-4")
    ]),
    dbc.Tab(label="📊 Data Breakdowns", tab_id="tab-breakdowns", children=[
        dbc.Row([
            dbc.Col(dcc.Graph(id='mail-type-chart'), width=6),
            dbc.Col(dcc.Graph(id='call-intent-chart'), width=6)
        ], className="mt-4")
    ])
])

 * Lines ~542-555 (After):
   # Tabbed Interface for Charts
dbc.Tabs([
    dbc.Tab(label="📈 Overview", tab_id="tab-overview", children=[
        dbc.Row(dbc.Col(dcc.Graph(id='overview-chart')), className="mt-4")
    ]),
    # --- ADDITION: New, more insightful analysis tab ---
    dbc.Tab(label="🎯 Campaign & Intent Analysis", tab_id="tab-intent-analysis", children=[
        dbc.Row([
            dbc.Col(dcc.Graph(id='intent-heatmap'), width=12, lg=7),
            dbc.Col(dcc.Graph(id='weekday-intent-chart'), width=12, lg=5),
        ], className="mt-4")
    ])
])

Step 5: Update the callbacks to drive the new tab.
Replace the two old callback functions (update_correlation_tab and update_breakdown_tab) with this single new one.
 * Lines ~607-635 (Before - DELETE these two functions):
   # Callback for correlation charts
@self.app.callback(...)
def update_correlation_tab(_):
    # ... (old function content)

# Callback for breakdown charts
@self.app.callback(...)
def update_breakdown_tab(_):
    # ... (old function content)

 * Add New Callback (After the update_overview_and_kpis callback):
   # --- ADDITION: Callback for the new analysis tab ---
@self.app.callback(
    [Output('intent-heatmap', 'figure'),
     Output('weekday-intent-chart', 'figure')],
    [Input('date-picker-range', 'id')] # Dummy input to trigger on load
)
def update_analysis_tab(_):
    heatmap_fig = self.create_intent_correlation_heatmap(self.dp.intent_correlation_matrix)
    weekday_fig = self.create_weekday_intent_figure()
    return heatmap_fig, weekday_fig

Step 6: Call the new analysis function in main.
 * Lines ~662 (Before):
   if SCIPY_AVAILABLE:
    dp.analyze_correlation()

 * Lines ~662 (After):
   # --- ADDITION: Run all analyses ---
if SCIPY_AVAILABLE:
    dp.analyze_intent_correlation()

These adjustments will fix the error and add the new, more powerful analysis to your dashboard.
