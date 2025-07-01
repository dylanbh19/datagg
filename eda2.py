# Create all plots

```
        for plot_name, plot_function in plot_functions:
            self.logger.info(f"Creating {plot_name}...")
            if plot_function():
                plots_created += 1
                self.logger.info(f"✓ {plot_name} created successfully")
            else:
                self.logger.warning(f"✗ Failed to create {plot_name}")
        
        self.logger.info(f"✓ Created {plots_created}/{len(plot_functions)} plots in {self.plots_dir}")
        return True
        
    except Exception as e:
        self.logger.error(f"✗ Error creating plots: {str(e)}")
        return False

def _generate_reports(self) -> bool:
    """Generate comprehensive analysis reports"""
    
    self.logger.info("\n" + "=" * 60)
    self.logger.info("STEP 6: GENERATING REPORTS")
    self.logger.info("=" * 60)
    
    try:
        # Create summary report
        summary_report = self._create_summary_report()
        
        # Save summary report
        report_path = os.path.join(self.reports_dir, 'analysis_summary.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(summary_report)
        
        self.logger.info(f"✓ Summary report saved: {report_path}")
        
        # Create detailed metrics report
        if self.analysis_results:
            metrics_report = self._create_metrics_report()
            metrics_path = os.path.join(self.reports_dir, 'detailed_metrics.txt')
            with open(metrics_path, 'w', encoding='utf-8') as f:
                f.write(metrics_report)
            self.logger.info(f"✓ Metrics report saved: {metrics_path}")
        
        return True
        
    except Exception as e:
        self.logger.error(f"✗ Error generating reports: {str(e)}")
        return False

def _create_summary_report(self) -> str:
    """Create a comprehensive summary report"""
    
    report = []
    report.append("=" * 80)
    report.append("CALL VOLUME TIME SERIES ANALYSIS - SUMMARY REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Data Overview
    report.append("DATA OVERVIEW")
    report.append("-" * 40)
    report.append(f"Mail data file: {self.config['MAIL_FILE_PATH']}")
    report.append(f"Call data file: {self.config['CALL_FILE_PATH']}")
    report.append("")
    
    if self.mail_data_clean is not None:
        report.append(f"Mail Data:")
        report.append(f"  Records: {len(self.mail_data_clean):,}")
        report.append(f"  Date range: {self.mail_data_clean['date'].min()} to {self.mail_data_clean['date'].max()}")
        report.append(f"  Total volume: {self.mail_data_clean['volume'].sum():,}")
        report.append(f"  Average daily volume: {self.mail_data_clean['volume'].mean():.1f}")
        report.append("")
    
    if self.call_data_clean is not None:
        report.append(f"Call Data:")
        report.append(f"  Records: {len(self.call_data_clean):,}")
        report.append(f"  Date range: {self.call_data_clean['date'].min()} to {self.call_data_clean['date'].max()}")
        report.append(f"  Total volume: {self.call_data_clean['volume'].sum():,}")
        report.append(f"  Average daily volume: {self.call_data_clean['volume'].mean():.1f}")
        report.append("")
    
    # Analysis Results
    if self.analysis_results:
        report.append("ANALYSIS RESULTS")
        report.append("-" * 40)
        
        # Correlation results
        if 'correlations' in self.analysis_results:
            corr = self.analysis_results['correlations']
            report.append(f"Correlations:")
            report.append(f"  Pearson correlation: {corr['pearson']:.4f} (p-value: {corr['pearson_pvalue']:.4f})")
            report.append(f"  Spearman correlation: {corr['spearman']:.4f} (p-value: {corr['spearman_pvalue']:.4f})")
            report.append("")
        
        # Best lag
        if 'best_lag' in self.analysis_results and self.analysis_results['best_lag'] is not None:
            best_lag = self.analysis_results['best_lag']
            report.append(f"Optimal Lag Analysis:")
            report.append(f"  Best lag: {best_lag['lag']} days")
            report.append(f"  Correlation at best lag: {best_lag['correlation']:.4f}")
            report.append(f"  P-value: {best_lag['p_value']:.4f}")
            report.append(f"  Observations: {best_lag['n_obs']:,}")
            report.append("")
        
        # Response rates
        if 'response_rates' in self.analysis_results and self.analysis_results['response_rates'] is not None:
            rates = self.analysis_results['response_rates']
            report.append(f"Response Rate Analysis:")
            report.append(f"  Mean response rate: {rates.mean():.2f}%")
            report.append(f"  Median response rate: {rates.median():.2f}%")
            report.append(f"  Standard deviation: {rates.std():.2f}%")
            report.append(f"  Range: {rates.min():.2f}% - {rates.max():.2f}%")
            report.append("")
        
        # Overlap statistics
        if 'overlap_stats' in self.analysis_results:
            overlap = self.analysis_results['overlap_stats']
            report.append(f"Data Overlap:")
            report.append(f"  Total combined records: {overlap['total_records']:,}")
            report.append(f"  Records with both call & mail: {overlap['overlap_records']:,}")
            report.append(f"  Overlap percentage: {overlap['overlap_percentage']:.1f}%")
            report.append("")
    
    # Mail type analysis
    if hasattr(self, 'mail_data_with_types') and 'type' in self.mail_data_with_types.columns:
        type_counts = self.mail_data_with_types['type'].value_counts()
        report.append("MAIL TYPE ANALYSIS")
        report.append("-" * 40)
        report.append(f"Total mail types: {len(type_counts)}")
        report.append("Top mail types:")
        for mail_type, count in type_counts.head(10).items():
            report.append(f"  {mail_type}: {count:,} records")
        report.append("")
    
    # Recommendations
    report.append("RECOMMENDATIONS")
    report.append("-" * 40)
    
    recommendations = []
    
    if self.analysis_results.get('overlap_stats', {}).get('overlap_percentage', 0) < 50:
        recommendations.append("Low data overlap detected - consider reviewing date alignment between datasets")
    
    if 'best_lag' in self.analysis_results and self.analysis_results['best_lag'] is not None:
        lag = self.analysis_results['best_lag']['lag']
        if lag > 0:
            recommendations.append(f"Consider applying {lag}-day lag when modeling mail impact on calls")
        else:
            recommendations.append("No significant lag detected - mail and calls appear to be simultaneous")
    
    if 'correlations' in self.analysis_results:
        corr = abs(self.analysis_results['correlations']['pearson'])
        if corr > 0.7:
            recommendations.append("Strong correlation detected - good candidate for predictive modeling")
        elif corr > 0.3:
            recommendations.append("Moderate correlation detected - modeling possible with additional features")
        else:
            recommendations.append("Weak correlation detected - consider external factors or data quality issues")
    
    if not recommendations:
        recommendations.append("Data appears suitable for time series modeling")
    
    for i, rec in enumerate(recommendations, 1):
        report.append(f"  {i}. {rec}")
    
    report.append("")
    report.append("END OF REPORT")
    report.append("=" * 80)
    
    return "\n".join(report)

def _create_metrics_report(self) -> str:
    """Create detailed metrics report"""
    
    report = []
    report.append("=" * 80)
    report.append("DETAILED METRICS REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Mail data statistics
    if self.mail_data_clean is not None:
        report.append("MAIL DATA STATISTICS")
        report.append("-" * 40)
        stats = self.mail_data_clean['volume'].describe()
        for stat, value in stats.items():
            report.append(f"  {stat}: {value:.2f}")
        report.append("")
    
    # Call data statistics
    if self.call_data_clean is not None:
        report.append("CALL DATA STATISTICS")
        report.append("-" * 40)
        stats = self.call_data_clean['volume'].describe()
        for stat, value in stats.items():
            report.append(f"  {stat}: {value:.2f}")
        report.append("")
    
    # Lag analysis details
    if 'lag_analysis' in self.analysis_results and self.analysis_results['lag_analysis'] is not None:
        lag_df = self.analysis_results['lag_analysis']
        report.append("LAG ANALYSIS DETAILS")
        report.append("-" * 40)
        report.append("Lag (days) | Correlation | P-value | Observations")
        report.append("-" * 50)
        for _, row in lag_df.iterrows():
            report.append(f"{row['lag']:9d} | {row['correlation']:11.4f} | {row['p_value']:7.4f} | {row['n_obs']:12,d}")
        report.append("")
    
    return "\n".join(report)

def _save_processed_data(self) -> bool:
    """Save processed datasets to files"""
    
    self.logger.info("\n" + "=" * 60)
    self.logger.info("STEP 7: SAVING PROCESSED DATA")
    self.logger.info("=" * 60)
    
    try:
        # Save clean mail data
        if self.mail_data_clean is not None:
            mail_path = os.path.join(self.data_dir, 'mail_data_clean.csv')
            self.mail_data_clean.to_csv(mail_path, index=False, encoding='utf-8')
            self.logger.info(f"✓ Clean mail data saved: {mail_path}")
        
        # Save clean call data
        if self.call_data_clean is not None:
            call_path = os.path.join(self.data_dir, 'call_data_clean.csv')
            self.call_data_clean.to_csv(call_path, index=False, encoding='utf-8')
            self.logger.info(f"✓ Clean call data saved: {call_path}")
        
        # Save combined data
        if self.combined_data is not None:
            combined_path = os.path.join(self.data_dir, 'combined_data.csv')
            self.combined_data.to_csv(combined_path, index=False, encoding='utf-8')
            self.logger.info(f"✓ Combined data saved: {combined_path}")
        
        # Save analysis results
        if self.analysis_results:
            import json
            results_path = os.path.join(self.data_dir, 'analysis_results.json')
            
            # Convert pandas objects to serializable format
            serializable_results = {}
            for key, value in self.analysis_results.items():
                if key == 'lag_analysis' and value is not None:
                    serializable_results[key] = value.to_dict('records')
                elif key == 'response_rates' and value is not None:
                    serializable_results[key] = value.tolist()
                else:
                    serializable_results[key] = value
            
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            self.logger.info(f"✓ Analysis results saved: {results_path}")
        
        return True
        
    except Exception as e:
        self.logger.error(f"✗ Error saving processed data: {str(e)}")
        return False
```

# =============================================================================

# MAIN EXECUTION FUNCTION

# =============================================================================

def main():
“”“Main execution function”””

```
print("=" * 80)
print("CALL VOLUME TIME SERIES ANALYSIS")
print("=" * 80)
print("Starting analysis...")
print()

# Validate configuration
if not os.path.exists(CONFIG['MAIL_FILE_PATH']):
    print(f"ERROR: Mail file not found: {CONFIG['MAIL_FILE_PATH']}")
    print("Please update MAIL_FILE_PATH in the configuration section.")
    return False

if not os.path.exists(CONFIG['CALL_FILE_PATH']):
    print(f"ERROR: Call file not found: {CONFIG['CALL_FILE_PATH']}")
    print("Please update CALL_FILE_PATH in the configuration section.")
    return False

# Setup logging
try:
    logger = setup_logging(CONFIG['OUTPUT_DIR'])
    logger.info("Logging initialized successfully")
except Exception as e:
    print(f"ERROR: Failed to setup logging: {str(e)}")
    return False

# Run analysis
try:
    analyzer = CallVolumeAnalyzer(CONFIG, logger)
    success = analyzer.run_complete_analysis()
    
    if success:
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Output directory: {CONFIG['OUTPUT_DIR']}")
        print(f"Plots directory: {os.path.join(CONFIG['OUTPUT_DIR'], 'plots')}")
        print(f"Reports directory: {os.path.join(CONFIG['OUTPUT_DIR'], 'reports')}")
        print(f"Data directory: {os.path.join(CONFIG['OUTPUT_DIR'], 'data')}")
        print("\nPlease review the generated plots and reports.")
        return True
    else:
        print("\n" + "=" * 80)
        print("ANALYSIS FAILED!")
        print("=" * 80)
        print("Please check the log file for detailed error information.")
        return False
        
except Exception as e:
    print(f"\nCRITICAL ERROR: {str(e)}")
    print("Please check your configuration and try again.")
    return False
```

# =============================================================================

# SCRIPT EXECUTION

# =============================================================================

if **name** == “**main**”:
“””
TO RUN THIS SCRIPT:

```
1. Update the file paths in the CONFIG section at the top
2. Update column mappings if your columns have different names
3. Run the script: python call_volume_analysis.py
4. Check the output directory for results

OUTPUTS:
- plots/: All EDA plots including mail type legends
- reports/: Summary and detailed analysis reports  
- data/: Cleaned datasets and analysis results
- analysis_log_[timestamp].txt: Detailed execution log
"""

# ASCII art header
print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    CALL VOLUME TIME SERIES ANALYSIS                          ║
║                                                                              ║
║  This script analyzes the relationship between mail campaigns and call      ║
║  volumes, providing comprehensive EDA plots, correlation analysis, and      ║
║  actionable insights for predictive modeling.                               ║
║                                                                              ║
║  Configure your file paths in the CONFIG section and run!                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

# Run main analysis
success = main()

# Exit with appropriate code
sys.exit(0 if success else 1)
```