Of course. I can help you fix those issues.

The `yfinance` error is a common pandas issue when the structure of the data you're downloading doesn't match the structure you're merging it into. The slow loading of the first visual is almost certainly caused by rendering too many individual markers on a line chart.

Here are the specific fixes for your script.

-----

### 1\. Fix for the `yfinance` Merge Error

The error "not allowed to merge between different levels" happens because `yfinance` can return a DataFrame with a MultiIndex on the columns, especially when fetching multiple tickers. Your current code fetches tickers one by one in a loop, which is inefficient and likely causing this issue.

A much better approach is to download all financial data at once and then process it.

#### **Action:**

Replace the entire `add_financial_data` function with this more robust version. It downloads all tickers in a single request and correctly handles the resulting DataFrame.

```python
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
```

-----

### 2\. Fix for Slow Plot Rendering

The first visual (`create_and_save_overlay_plot`) is slow because you are drawing a scatter plot with both lines and markers (`mode='lines+markers'`) for the raw call volume. If you have hundreds or thousands of data points, rendering a unique marker for each point is computationally expensive and slows down generation significantly.

#### **Action:**

In the `create_and_save_overlay_plot` function, change the `mode` for the 'Daily Calls' trace from `lines+markers` to just `lines`. The line itself is enough to show the trend.

**Find this block of code:**

```python
# Inside create_and_save_overlay_plot()

    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['call_volume'],
            name='Daily Calls',
            line=dict(color=colors['calls'], width=3),
            mode='lines+markers',  # <--- THIS IS THE SLOW PART
            marker=dict(size=6),
            yaxis='y3'
        ),
        row=2, col=1
    )
```

**And change `mode='lines+markers'` to `mode='lines'` like this:**

```python
# Inside create_and_save_overlay_plot()

    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['call_volume'],
            name='Daily Calls',
            line=dict(color=colors['calls'], width=3),
            mode='lines',  # <--- FIXED: Much faster rendering
            yaxis='y3'
        ),
        row=2, col=1
    )
```

By making these two changes, your script should now run without the merge error and generate the plots much more quickly.
