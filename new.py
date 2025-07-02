Of course. Adding this level of functionality requires several new components. Here are the exact code snippets and clear instructions on where to add or replace them in your script.
This process will transform the final output of your script from static images into a single, powerful interactive dashboard.
Step 1: Add New Imports
At the top of your file with the other import statements (around line 15), add the following imports for data processing and the interactive dashboard.
# Add these imports to the top of your file
from sklearn.preprocessing import MinMaxScaler
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go

Step 2: Add New Data Processing Methods
To handle the data augmentation and normalization, add these two new methods inside your CallVolumeAnalyzer class. A good place is after the _fetch_market_data method (around line 450).
    # Add these two entire methods into the CallVolumeAnalyzer class

    def _augment_call_data(self) -> bool:
        """Finds and fills a missing year of call data using interpolation."""
        self.logger.info("STEP 5b: Augmenting missing call data...")
        try:
            # First, create a daily summary of all calls
            self.call_daily_summary = self.call_data.groupby(pd.Grouper(key='date', freq='D'))['volume'].sum().to_frame()

            # Create a full date range to find gaps
            full_range = pd.date_range(start=self.call_daily_summary.index.min(), end=self.call_daily_summary.index.max(), freq='D')
            self.call_daily_summary = self.call_daily_summary.reindex(full_range)

            # Identify which dates will be augmented
            missing_dates = self.call_daily_summary[self.call_daily_summary['volume'].isnull()].index
            if len(missing_dates) == 0:
                self.logger.info("✓ No significant gaps in call data found to augment.")
                self.augmented_dates = pd.Index([]) # Ensure attribute exists
                return True

            # Fill missing data using a linear method
            self.call_daily_summary['volume'].interpolate(method='linear', inplace=True)
            self.augmented_dates = missing_dates
            self.logger.info(f"✓ Augmented {len(self.augmented_dates)} days of missing call data.")
            return True
        except Exception as e:
            self.logger.error(f"Failed during data augmentation: {e}")
            return False

    def _normalize_data(self) -> bool:
        """Normalizes all data columns to a 0-1 scale for comparison."""
        self.logger.info("STEP 5c: Normalizing all data for comparison...")
        try:
            # Combine all data sources into one table
            self.final_table = self.call_daily_summary.rename(columns={'volume': 'Call Volume'})
            mail_summary = self.mail_data.groupby(pd.Grouper(key='date', freq='D'))['volume'].sum().rename('Mail Volume')
            self.final_table = self.final_table.join(mail_summary)
            self.final_table = self.final_table.join(self.financial_data)
            self.final_table.fillna(0, inplace=True)

            # Use MinMaxScaler to scale all columns between 0 and 1
            scaler = MinMaxScaler()
            self.normalized_table = pd.DataFrame(scaler.fit_transform(self.final_table), columns=self.final_table.columns, index=self.final_table.index)
            self.logger.info("✓ All data has been normalized.")
            return True
        except Exception as e:
            self.logger.error(f"Failed during data normalization: {e}")
            return False

Step 3: Replace Static Plots with the Interactive Dashboard
This is the most important change. The new features like zooming and filtering require an interactive dashboard, which works differently than saving static plot images.
You must DELETE your entire _create_all_plots method and all the individual plotting methods (e.g., _create_mail_vs_call_plot, _create_financial_overlay_plot, etc.).
REPLACE them with this single new method:
    # DELETE your old _create_all_plots method and all other _create_*_plot methods.
    # ADD this single new method in their place.

    def _launch_interactive_dashboard(self):
        """Launches a Dash web application for interactive data exploration."""
        self.logger.info("STEP 6: Launching Interactive Dashboard...")
        
        app = dash.Dash(__name__)
        app.title = "Marketing & Financial Analysis Dashboard"

        # Prepare data for filters
        mail_types = ['All'] + sorted(self.mail_data['mail_type'].unique())
        call_intents = ['All'] + sorted(self.call_data['intent'].unique())

        app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif'}, children=[
            html.H1("Interactive Marketing & Financial Analysis", style={'textAlign': 'center'}),
            html.P("All data is normalized to a 0-1 scale for trend comparison. Use filters to explore the data.", style={'textAlign': 'center'}),
            
            html.Div(style={'padding': '20px', 'display': 'flex', 'justifyContent': 'center', 'gap': '30px'}, children=[
                html.Div([
                    html.Label("Filter by Mail Type:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(id='mail-type-filter', options=mail_types, value='All', clearable=False)
                ], style={'width': '350px'}),
                html.Div([
                    html.Label("Filter by Call Intent:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(id='call-intent-filter', options=call_intents, value='All', clearable=False)
                ], style={'width': '350px'})
            ]),
            
            dcc.Graph(id='interactive-plot', style={'height': '75vh'})
        ])

        @app.callback(
            Output('interactive-plot', 'figure'),
            [Input('mail-type-filter', 'value'),
             Input('call-intent-filter', 'value')]
        )
        def update_plot(selected_mail_type, selected_call_intent):
            # Base normalized data
            plot_data = self.normalized_table.copy()
            
            # Filter mail data
            if selected_mail_type != 'All':
                filtered_dates = self.mail_data[self.mail_data['mail_type'] == selected_mail_type]['date'].dt.date
                plot_data['Mail Volume'] = plot_data.index.to_series().dt.date.isin(filtered_dates).astype(float) * plot_data['Mail Volume']

            # Filter call data
            if selected_call_intent != 'All':
                filtered_dates = self.call_data[self.call_data['intent'] == selected_call_intent]['date'].dt.date
                plot_data['Call Volume'] = plot_data.index.to_series().dt.date.isin(filtered_dates).astype(float) * plot_data['Call Volume']

            fig = go.Figure()

            # Plot financial data first (background)
            for col in self.config['FINANCIAL_DATA'].keys():
                fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data[col], name=col, mode='lines', line=dict(dash='dot'), opacity=0.7))

            # Plot mail and call data
            fig.add_trace(go.Bar(x=plot_data.index, y=plot_data['Mail Volume'], name='Mail Volume', marker_color='skyblue', opacity=0.8))
            fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['Call Volume'], name='Call Volume', mode='lines', line=dict(color='navy', width=2.5)))

            # Overlay augmented data highlight
            augmented_trace_data = plot_data.loc[self.augmented_dates]
            fig.add_trace(go.Scatter(x=augmented_trace_data.index, y=augmented_trace_data['Call Volume'], name='Augmented Call Data', mode='markers', marker=dict(color='red', symbol='x', size=5)))

            fig.update_layout(title_text=f"Normalized Trends for Mail: '{selected_mail_type}' & Calls: '{selected_call_intent}'", yaxis_title="Normalized Value (0 to 1)", template='plotly_white')
            return fig

        self.logger.info("Dashboard is running. Press CTRL+C in the terminal to stop.")
        app.run_server(debug=True)
        return True

Step 4: Update the Main Execution Pipeline
Finally, modify your run_complete_analysis method to call the new functions in the correct order and launch the dashboard instead of creating static files.
REPLACE your existing run_complete_analysis method (around line 145) with this new version:
    # REPLACE your old run_complete_analysis method with this new one.

    def run_complete_analysis(self) -> bool:
        """Run the complete analysis pipeline and launch the dashboard."""
        self.logger.info("=" * 80)
        self.logger.info("CALL VOLUME TIME SERIES ANALYSIS - STARTING")
        try:
            # --- DATA PROCESSING STEPS ---
            if not self._load_data(): return False
            if not self._process_data(): return False # Assumes intent/mail_type are processed here
            if not self._fetch_market_data(): return False
            if not self._augment_call_data(): return False
            if not self._normalize_data(): return False
            
            # --- FINAL OUTPUT STEP ---
            # The dashboard replaces all static plotting and reporting
            if not self._launch_interactive_dashboard(): return False
            
            self.logger.info("=" * 80)
            self.logger.info("ANALYSIS COMPLETED SUCCESSFULLY")
            return True
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}\n{traceback.format_exc()}")
            return False







# In your CallVolumeAnalyzer class, REPLACE the existing _fetch_market_data method with this new one.

def _fetch_market_data(self) -> bool:
    """New function to download financial data."""
    self.logger.info("\n" + "=" * 60)
    self.logger.info("STEP 5: FETCHING FINANCIAL MARKET DATA")
    self.logger.info("=" * 60)
    try:
        # --- START OF FIX ---
        # Determine date range directly from the loaded mail and call data
        if self.mail_data is None or self.call_data is None:
            self.logger.error("Mail or Call data not loaded, cannot determine date range.")
            return False

        start_date = min(self.mail_data['date'].min(), self.call_data['date'].min())
        end_date = max(self.mail_data['date'].max(), self.call_data['date'].max())
        # --- END OF FIX ---

        tickers = self.config['FINANCIAL_DATA']
        
        self.logger.info(f"Downloading market data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        data = yf.download(list(tickers.values()), start=start_date, end=end_date, progress=False)
        
        if data.empty:
            self.logger.warning("Could not download financial data for the specified date range.")
            self.financial_data = pd.DataFrame()
            return True

        # This block now handles both multi-ticker and single-ticker results
        try:
            # Try the multi-ticker format first
            processed_data = data['Adj Close']
        except KeyError:
            # If 'Adj Close' key fails, it's likely a single-ticker format
            self.logger.info("Single ticker format detected, processing accordingly.")
            processed_data = data[['Close']] 
            if len(tickers) == 1:
                 single_ticker_name = list(tickers.keys())[0]
                 processed_data.rename(columns={'Close': single_ticker_name}, inplace=True)
        
        # Rename columns to their friendly names
        self.financial_data = processed_data.rename(columns={v: k for k, v in tickers.items()})
        self.financial_data.ffill(inplace=True) # Fill non-trading days
        
        self.logger.info("✓ Financial data fetched successfully.")
        return True
    except Exception as e:
        self.logger.error(f"Failed to fetch financial data: {e}")
        self.logger.error(traceback.format_exc())
        return False
