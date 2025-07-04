# =============================================================================
# SCRIPT EXECUTION
# =============================================================================
def main():
    """Main execution function."""
    print("🔥 EXECUTIVE MARKETING INTELLIGENCE DASHBOARD")
    print("=" * 60)
    
    logger = setup_logging()
    try:
        # The main pipeline calls each function sequentially
        mail_data, call_data = load_data(CONFIG, logger)
        augmented_df, original_calls, synthetic_calls = augment_and_combine_data(call_data, mail_data, CONFIG, logger)
        final_df = feature_engineering(augmented_df, logger)
        
        # Initialize the dashboard with the processed data
        logger.info("🎨 Initializing executive dashboard...")
        app = build_dashboard(final_df, mail_data, call_data, original_calls, synthetic_calls)
        
        # --- START OF ADDED LOGGING ---
        # Log a summary of the processed data before launching the server
        logger.info("📊 DASHBOARD READY! Data Summary:")
        logger.info("=" * 40)
        logger.info(f"Mail records processed: {len(mail_data):,}")
        logger.info(f"Call records processed: {len(call_data):,}")
        logger.info(f"Total analysis records: {len(final_df):,}")
        logger.info(f"Augmented records created: {len(synthetic_calls):,}")
        
        financial_indicators = [col for col in final_df.columns if col in CONFIG['FINANCIAL_DATA'].keys()]
        if financial_indicators:
            logger.info(f"Financial indicators loaded: {', '.join(financial_indicators)}")
        
        logger.info("=" * 40)
        # --- END OF ADDED LOGGING ---

        app.run_server(debug=True)
        return True

    except KeyboardInterrupt:
        logger.info("🛑 Dashboard stopped by user.")
        return True
    except Exception as e:
        logger.error(f"❌ A fatal error occurred in the main execution block: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == '__main__':
    # A check for dash_bootstrap_components, as it's a key dependency for the UI
    try:
        import dash_bootstrap_components as dbc
    except ImportError:
        print("ERROR: dash-bootstrap-components is not installed. Please run: pip install dash-bootstrap-components")
        sys.exit(1)
        
    success = main()
    sys.exit(0 if success else 1)
