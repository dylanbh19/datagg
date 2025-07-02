# Insert this inside your main CONFIG = { ... } dictionary

    # === FINANCIAL DATA (NEW ADDITION) ===
    'FINANCIAL_DATA': {
        'S&P 500': '^GSPC',
        '10-Yr Treasury Yield': '^TNX',
        'Inflation (Crude Oil)': 'CL=F'
    },




# In your CallVolumeAnalyzer class, replace the existing _create_financial_overlay_plot method with this one.

def _create_financial_overlay_plot(self) -> bool:
    """Creates a static plot comparing mail/call volume to NORMALISED financial indicators."""
    if self.financial_data is None or self.financial_data.empty:
        self.logger.warning("Skipping financial overlay plot: no financial data available.")
        return False

    # --- START OF NEW CODE: Normalization ---
    # Create a copy to avoid modifying the original dataframe
    normalized_financial_data = self.financial_data.copy()
    
    # Normalize each financial metric to a base of 100
    for col in normalized_financial_data.columns:
        # Find the first valid data point to use as the baseline
        first_value = normalized_financial_data[col].dropna().iloc[0]
        if first_value > 0:
            normalized_financial_data[col] = (normalized_financial_data[col] / first_value) * 100
    # --- END OF NEW CODE ---

    fig, ax1 = plt.subplots(figsize=self.config['FIGURE_SIZE'])
    
    # Plot Mail and Call Volume on primary y-axis
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Mail & Call Volume', color='tab:blue', fontsize=12)
    
    # --- START OF MODIFICATION: Add Mail Data ---
    # Add Mail data as a bar chart
    ax1.bar(self.mail_data_clean['date'], self.mail_data_clean['volume'], color='lightsteelblue', label='Mail Volume', alpha=0.6)
    # --- END OF MODIFICATION ---

    ax1.plot(self.call_data_clean['date'], self.call_data_clean['volume'], color='tab:blue', label='Call Volume', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(False) # Keep primary axis grid off for clarity

    # Create a secondary y-axis for the NORMALIZED financial data
    ax2 = ax1.twinx()
    
    # --- START OF MODIFICATION: Update axis label and plot normalized data ---
    ax2.set_ylabel('Normalized Value (Base 100)', color='gray', fontsize=12)
    colors = ['tab:red', 'tab:green', 'tab:purple']
    for i, col in enumerate(normalized_financial_data.columns):
        ax2.plot(normalized_financial_data.index, normalized_financial_data[col], color=colors[i % len(colors)], linestyle='--', alpha=0.8, label=col)
    # --- END OF MODIFICATION ---
    
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.axhline(100, color='gray', linestyle=':', linewidth=1, alpha=0.8) # Add a line for the base 100

    fig.suptitle('Mail/Call Volume vs. Normalized Financial Trends', fontsize=16, fontweight='bold')
    
    # Combine legends from both axes into one box
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = os.path.join(self.plots_dir, '12_financial_overlay.png')
    plt.savefig(plot_path, dpi=self.config['DPI'])
    plt.close()
    
    self.logger.info("✓ Financial overlay plot with normalized data has been created.")
    return True



    # Insert this entire method into the CallVolumeAnalyzer # In your CallVolumeAnalyzer class, replace the old _fetch_market_data method with this one.

def _fetch_market_data(self) -> bool:
    """New function to download financial data."""
    self.logger.info("\n" + "=" * 60)
    self.logger.info("STEP 5: FETCHING FINANCIAL MARKET DATA")
    self.logger.info("=" * 60)
    try:
        if self.combined_data is None or self.combined_data.empty:
            self.logger.warning("Combined data not available, cannot determine date range for financial data.")
            return False

        start_date = self.combined_data['date'].min()
        end_date = self.combined_data['date'].max()
        tickers = self.config['FINANCIAL_DATA']
        
        self.logger.info(f"Downloading market data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        data = yf.download(list(tickers.values()), start=start_date, end=end_date, progress=False)
        
        if data.empty:
            self.logger.warning("Could not download financial data for the specified date range.")
            self.financial_data = pd.DataFrame()
            return True

        # --- START OF FIX ---
        # This block now handles both multi-ticker and single-ticker results
        processed_data = pd.DataFrame()
        try:
            # Try the multi-ticker format first
            processed_data = data['Adj Close']
        except KeyError:
            # If 'Adj Close' key fails, it's likely a single-ticker format
            self.logger.info("Single ticker format detected, processing accordingly.")
            processed_data = data[['Close']] # Use 'Close' price for single tickers
            
            # Since it's one ticker, we need to manually assign its name
            single_ticker_symbol = list(tickers.values())[0]
            single_ticker_name = list(tickers.keys())[0]
            if len(tickers) == 1:
                 processed_data.rename(columns={'Close': single_ticker_name}, inplace=True)

        self.financial_data = processed_data.rename(columns={v: k for k, v in tickers.items()})
        # --- END OF FIX ---
        
        self.financial_data.ffill(inplace=True) # Fill non-trading days
        self.logger.info("✓ Financial data fetched successfully.")
        return True
    except Exception as e:
        self.logger.error(f"Failed to fetch financial data: {e}")
        return False






    # Insert this entire method into the CallVolumeAnalyzer class

    def _create_financial_overlay_plot(self) -> bool:
        """Creates a static plot comparing call volume to financial indicators."""
        if self.financial_data is None or self.financial_data.empty:
            self.logger.warning("Skipping financial overlay plot: no financial data available.")
            return False

        fig, ax1 = plt.subplots(figsize=self.config['FIGURE_SIZE'])
        
        # Plot Call Volume on primary y-axis
        color = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Call Volume', color=color)
        ax1.plot(self.call_data_clean['date'], self.call_data_clean['volume'], color=color, label='Call Volume')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(False) # Turn off grid for the primary axis
        
        # Create a secondary y-axis for financial data
        ax2 = ax1.twinx()
        
        colors = ['tab:red', 'tab:green', 'tab:purple']
        for i, col in enumerate(self.financial_data.columns):
            ax2.plot(self.financial_data.index, self.financial_data[col], color=colors[i % len(colors)], linestyle='--', alpha=0.7, label=col)
        
        ax2.set_ylabel('Financial Index / Price', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        
        fig.suptitle('Call Volume vs. Financial Market Indicators', fontsize=16)
        
        # Combine legends from both axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')

        fig.tight_layout()
        plot_path = os.path.join(self.plots_dir, '12_financial_overlay.png')
        plt.savefig(plot_path, dpi=self.config['DPI'])
        plt.close()
        return True





self.financial_data = None 


# Add this line after the call to _analyze_correlations
if not self._fetch_market_data():
    return False



# Add this tuple to the plot_functions list
("Financial market overlay", self._create_financial_overlay_plot),

