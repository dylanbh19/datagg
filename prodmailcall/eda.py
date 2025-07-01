import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings(‘ignore’)

# Statistical libraries

from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

# Time series libraries

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class CallVolumeAnalyzer:
def **init**(self):
self.call_data = None
self.mail_data = None
self.combined_data = None
self.outliers_removed = False

```
def load_and_aggregate_calls(self, call_file_path, date_col='date', volume_col='calls'):
    """
    Load call data and aggregate by date
    """
    print("Loading and aggregating call data...")
    
    # Load call data
    if call_file_path.endswith('.csv'):
        df = pd.read_csv(call_file_path)
    elif call_file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(call_file_path)
    
    # Convert date column to datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Aggregate calls by date
    self.call_data = df.groupby(date_col)[volume_col].sum().reset_index()
    self.call_data.columns = ['date', 'call_volume']
    self.call_data = self.call_data.sort_values('date').reset_index(drop=True)
    
    print(f"Call data loaded: {len(self.call_data)} unique dates")
    print(f"Date range: {self.call_data['date'].min()} to {self.call_data['date'].max()}")
    print(f"Total calls: {self.call_data['call_volume'].sum():,}")
    
    return self.call_data

def load_and_aggregate_mail(self, mail_files, date_col='date', volume_col='mail_volume'):
    """
    Load multiple mail files and aggregate by date
    mail_files can be a single file path or list of file paths
    """
    print("Loading and aggregating mail data...")
    
    if isinstance(mail_files, str):
        mail_files = [mail_files]
    
    all_mail_data = []
    
    for file_path in mail_files:
        print(f"Processing: {file_path}")
        
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        
        # Convert date column to datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # If volume column doesn't exist, try common variations
        if volume_col not in df.columns:
            possible_cols = ['volume', 'mail', 'pieces', 'count', 'quantity']
            for col in possible_cols:
                if col in df.columns:
                    df = df.rename(columns={col: volume_col})
                    break
        
        all_mail_data.append(df[[date_col, volume_col]])
    
    # Combine all mail data
    combined_mail = pd.concat(all_mail_data, ignore_index=True)
    
    # Aggregate by date across all sources
    self.mail_data = combined_mail.groupby(date_col)[volume_col].sum().reset_index()
    self.mail_data.columns = ['date', 'mail_volume']
    self.mail_data = self.mail_data.sort_values('date').reset_index(drop=True)
    
    print(f"Mail data loaded: {len(self.mail_data)} unique dates")
    print(f"Date range: {self.mail_data['date'].min()} to {self.mail_data['date'].max()}")
    print(f"Total mail pieces: {self.mail_data['mail_volume'].sum():,}")
    
    return self.mail_data

def detect_outliers_multiple_methods(self, data, column):
    """
    Detect outliers using multiple methods and return consensus
    """
    outliers = pd.DataFrame(index=data.index)
    
    # Method 1: IQR
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers['iqr'] = (data[column] < lower_bound) | (data[column] > upper_bound)
    
    # Method 2: Z-score
    z_scores = np.abs(stats.zscore(data[column]))
    outliers['zscore'] = z_scores > 3
    
    # Method 3: Modified Z-score (using median)
    median = data[column].median()
    mad = np.median(np.abs(data[column] - median))
    modified_z_scores = 0.6745 * (data[column] - median) / mad
    outliers['modified_zscore'] = np.abs(modified_z_scores) > 3.5
    
    # Method 4: Isolation Forest
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    outliers['isolation'] = iso_forest.fit_predict(data[[column]].values) == -1
    
    # Consensus: outlier if detected by at least 2 methods
    outliers['consensus'] = outliers.sum(axis=1) >= 2
    
    return outliers['consensus']

def perform_call_eda(self, remove_outliers=True):
    """
    Comprehensive EDA for call data
    """
    print("\n" + "="*50)
    print("CALL DATA - EXPLORATORY DATA ANALYSIS")
    print("="*50)
    
    data = self.call_data.copy()
    
    # Basic statistics
    print("\nBASIC STATISTICS:")
    print(f"Total observations: {len(data)}")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")
    print(f"Duration: {(data['date'].max() - data['date'].min()).days} days")
    print("\nCall Volume Statistics:")
    print(data['call_volume'].describe())
    
    # Check for missing dates
    date_range = pd.date_range(start=data['date'].min(), end=data['date'].max())
    missing_dates = set(date_range) - set(data['date'])
    print(f"\nMissing dates: {len(missing_dates)}")
    if len(missing_dates) > 0 and len(missing_dates) <= 10:
        print("Missing dates:", sorted(missing_dates))
    
    # Outlier detection
    outliers = self.detect_outliers_multiple_methods(data, 'call_volume')
    print(f"\nOutliers detected: {outliers.sum()} ({outliers.sum()/len(data)*100:.1f}%)")
    
    if outliers.sum() > 0:
        print("\nOutlier dates and volumes:")
        outlier_data = data[outliers][['date', 'call_volume']].sort_values('call_volume', ascending=False)
        print(outlier_data.head(10))
    
    # Remove outliers if requested
    if remove_outliers and outliers.sum() > 0:
        print(f"\nRemoving {outliers.sum()} outliers...")
        data = data[~outliers].reset_index(drop=True)
        self.call_data_clean = data
        print(f"Clean dataset: {len(data)} observations")
    else:
        self.call_data_clean = data
    
    # Time-based patterns
    data['day_of_week'] = data['date'].dt.day_name()
    data['month'] = data['date'].dt.month
    data['year'] = data['date'].dt.year
    data['quarter'] = data['date'].dt.quarter
    
    print("\nTIME-BASED PATTERNS:")
    print("\nAverage calls by day of week:")
    day_avg = data.groupby('day_of_week')['call_volume'].mean().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ])
    print(day_avg.round(1))
    
    print("\nAverage calls by month:")
    month_avg = data.groupby('month')['call_volume'].mean()
    print(month_avg.round(1))
    
    # Seasonality check
    if len(data) > 365:
        print("\nYearly comparison:")
        yearly_avg = data.groupby('year')['call_volume'].agg(['mean', 'std', 'count'])
        print(yearly_avg.round(1))
    
    return data

def perform_mail_eda(self, remove_outliers=True):
    """
    Comprehensive EDA for mail data
    """
    print("\n" + "="*50)
    print("MAIL DATA - EXPLORATORY DATA ANALYSIS")
    print("="*50)
    
    data = self.mail_data.copy()
    
    # Basic statistics
    print("\nBASIC STATISTICS:")
    print(f"Total observations: {len(data)}")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")
    print(f"Duration: {(data['date'].max() - data['date'].min()).days} days")
    print("\nMail Volume Statistics:")
    print(data['mail_volume'].describe())
    
    # Check for missing dates
    date_range = pd.date_range(start=data['date'].min(), end=data['date'].max())
    missing_dates = set(date_range) - set(data['date'])
    print(f"\nMissing dates: {len(missing_dates)}")
    if len(missing_dates) > 0 and len(missing_dates) <= 10:
        print("Missing dates:", sorted(missing_dates))
    
    # Outlier detection
    outliers = self.detect_outliers_multiple_methods(data, 'mail_volume')
    print(f"\nOutliers detected: {outliers.sum()} ({outliers.sum()/len(data)*100:.1f}%)")
    
    if outliers.sum() > 0:
        print("\nOutlier dates and volumes:")
        outlier_data = data[outliers][['date', 'mail_volume']].sort_values('mail_volume', ascending=False)
        print(outlier_data.head(10))
    
    # Remove outliers if requested
    if remove_outliers and outliers.sum() > 0:
        print(f"\nRemoving {outliers.sum()} outliers...")
        data = data[~outliers].reset_index(drop=True)
        self.mail_data_clean = data
        print(f"Clean dataset: {len(data)} observations")
    else:
        self.mail_data_clean = data
    
    # Time-based patterns
    data['day_of_week'] = data['date'].dt.day_name()
    data['month'] = data['date'].dt.month
    data['year'] = data['date'].dt.year
    data['quarter'] = data['date'].dt.quarter
    
    print("\nTIME-BASED PATTERNS:")
    print("\nAverage mail volume by day of week:")
    day_avg = data.groupby('day_of_week')['mail_volume'].mean().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ])
    print(day_avg.round(1))
    
    print("\nAverage mail volume by month:")
    month_avg = data.groupby('month')['mail_volume'].mean()
    print(month_avg.round(1))
    
    # Seasonality check
    if len(data) > 365:
        print("\nYearly comparison:")
        yearly_avg = data.groupby('year')['mail_volume'].agg(['mean', 'std', 'count'])
        print(yearly_avg.round(1))
    
    return data

def combine_datasets(self, lag_days=None):
    """
    Combine call and mail datasets with optional lag
    """
    print("\n" + "="*50)
    print("COMBINING DATASETS")
    print("="*50)
    
    call_data = self.call_data_clean.copy()
    mail_data = self.mail_data_clean.copy()
    
    if lag_days:
        print(f"Applying {lag_days} day lag to mail data...")
        mail_data['date'] = mail_data['date'] + timedelta(days=lag_days)
    
    # Merge datasets
    self.combined_data = pd.merge(call_data, mail_data, on='date', how='outer')
    self.combined_data = self.combined_data.sort_values('date').reset_index(drop=True)
    
    # Fill missing values
    self.combined_data['call_volume'] = self.combined_data['call_volume'].fillna(0)
    self.combined_data['mail_volume'] = self.combined_data['mail_volume'].fillna(0)
    
    print(f"Combined dataset: {len(self.combined_data)} observations")
    print(f"Date range: {self.combined_data['date'].min()} to {self.combined_data['date'].max()}")
    print(f"Rows with both call and mail data: {((self.combined_data['call_volume'] > 0) & (self.combined_data['mail_volume'] > 0)).sum()}")
    
    return self.combined_data

def comparative_analysis(self):
    """
    Perform comparative analysis between calls and mail
    """
    print("\n" + "="*50)
    print("COMPARATIVE ANALYSIS - CALLS vs MAIL")
    print("="*50)
    
    if self.combined_data is None:
        self.combine_datasets()
    
    data = self.combined_data.copy()
    
    # Filter data where both values exist
    both_exist = (data['call_volume'] > 0) & (data['mail_volume'] > 0)
    analysis_data = data[both_exist].copy()
    
    if len(analysis_data) == 0:
        print("No overlapping data found for comparative analysis!")
        return
    
    print(f"Analyzing {len(analysis_data)} days with both call and mail data")
    
    # Correlation analysis
    correlations = {}
    
    # Pearson correlation
    corr_pearson, p_pearson = pearsonr(analysis_data['call_volume'], analysis_data['mail_volume'])
    correlations['Pearson'] = {'correlation': corr_pearson, 'p_value': p_pearson}
    
    # Spearman correlation
    corr_spearman, p_spearman = spearmanr(analysis_data['call_volume'], analysis_data['mail_volume'])
    correlations['Spearman'] = {'correlation': corr_spearman, 'p_value': p_spearman}
    
    print("\nCORRELATION ANALYSIS:")
    for method, results in correlations.items():
        print(f"{method} correlation: {results['correlation']:.4f} (p-value: {results['p_value']:.4f})")
    
    # Test different lags
    print("\nLAG ANALYSIS:")
    print("Testing different lag periods...")
    
    lag_results = []
    for lag in range(0, 15):  # Test 0-14 day lags
        mail_lagged = data['mail_volume'].shift(lag)
        mask = (data['call_volume'] > 0) & (mail_lagged > 0)
        
        if mask.sum() > 10:  # Need at least 10 observations
            corr, p_val = pearsonr(data.loc[mask, 'call_volume'], mail_lagged[mask])
            lag_results.append({'lag': lag, 'correlation': corr, 'p_value': p_val, 'n_obs': mask.sum()})
    
    if lag_results:
        lag_df = pd.DataFrame(lag_results)
        best_lag = lag_df.loc[lag_df['correlation'].idxmax()]
        print(f"\nBest lag period: {best_lag['lag']} days")
        print(f"Correlation: {best_lag['correlation']:.4f}")
        print(f"P-value: {best_lag['p_value']:.4f}")
        print(f"Observations: {best_lag['n_obs']}")
        
        # Show top 5 lags
        print("\nTop 5 lag periods:")
        top_lags = lag_df.nlargest(5, 'correlation')
        print(top_lags)
    
    # Response rate analysis
    print("\nRESPONSE RATE ANALYSIS:")
    analysis_data['response_rate'] = analysis_data['call_volume'] / analysis_data['mail_volume'] * 100
    
    # Remove extreme response rates (likely data issues)
    valid_rates = analysis_data['response_rate'] < 50  # Assume >50% response rate is unrealistic
    clean_rates = analysis_data[valid_rates]['response_rate']
    
    print(f"Response rate statistics (excluding extreme values):")
    print(clean_rates.describe())
    print(f"Median response rate: {clean_rates.median():.2f}%")
    
    return analysis_data, correlations, lag_results if 'lag_results' in locals() else None

def create_visualizations(self):
    """
    Create comprehensive visualizations
    """
    print("\n" + "="*50)
    print("CREATING VISUALIZATIONS")
    print("="*50)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 24))
    
    # 1. Call volume time series
    ax1 = plt.subplot(4, 2, 1)
    plt.plot(self.call_data_clean['date'], self.call_data_clean['call_volume'], 
            linewidth=1, alpha=0.7, color='blue')
    plt.title('Call Volume Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Call Volume')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 2. Mail volume time series
    ax2 = plt.subplot(4, 2, 2)
    plt.plot(self.mail_data_clean['date'], self.mail_data_clean['mail_volume'], 
            linewidth=1, alpha=0.7, color='red')
    plt.title('Mail Volume Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Mail Volume')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 3. Call volume distribution
    ax3 = plt.subplot(4, 2, 3)
    plt.hist(self.call_data_clean['call_volume'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Call Volume Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Call Volume')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # 4. Mail volume distribution
    ax4 = plt.subplot(4, 2, 4)
    plt.hist(self.mail_data_clean['mail_volume'], bins=50, alpha=0.7, color='red', edgecolor='black')
    plt.title('Mail Volume Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Mail Volume')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # 5. Box plots by day of week (calls)
    ax5 = plt.subplot(4, 2, 5)
    call_data = self.call_data_clean.copy()
    call_data['day_of_week'] = call_data['date'].dt.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    sns.boxplot(data=call_data, x='day_of_week', y='call_volume', order=day_order, ax=ax5)
    plt.title('Call Volume by Day of Week', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 6. Box plots by day of week (mail)
    ax6 = plt.subplot(4, 2, 6)
    mail_data = self.mail_data_clean.copy()
    mail_data['day_of_week'] = mail_data['date'].dt.day_name()
    sns.boxplot(data=mail_data, x='day_of_week', y='mail_volume', order=day_order, ax=ax6)
    plt.title('Mail Volume by Day of Week', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 7. Overlay time series
    ax7 = plt.subplot(4, 2, 7)
    if self.combined_data is not None:
        # Normalize both series for comparison
        call_norm = self.combined_data['call_volume'] / self.combined_data['call_volume'].max()
        mail_norm = self.combined_data['mail_volume'] / self.combined_data['mail_volume'].max()
        
        plt.plot(self.combined_data['date'], call_norm, label='Calls (normalized)', 
                alpha=0.7, color='blue', linewidth=1)
        plt.plot(self.combined_data['date'], mail_norm, label='Mail (normalized)', 
                alpha=0.7, color='red', linewidth=1)
        plt.title('Normalized Call vs Mail Volume Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Normalized Volume')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    # 8. Scatter plot (if combined data exists)
    ax8 = plt.subplot(4, 2, 8)
    if self.combined_data is not None:
        mask = (self.combined_data['call_volume'] > 0) & (self.combined_data['mail_volume'] > 0)
        subset = self.combined_data[mask]
        
        if len(subset) > 0:
            plt.scatter(subset['mail_volume'], subset['call_volume'], alpha=0.6, color='purple')
            
            # Add trend line
            z = np.polyfit(subset['mail_volume'], subset['call_volume'], 1)
            p = np.poly1d(z)
            plt.plot(subset['mail_volume'], p(subset['mail_volume']), "r--", alpha=0.8)
            
            plt.title('Call Volume vs Mail Volume', fontsize=14, fontweight='bold')
            plt.xlabel('Mail Volume')
            plt.ylabel('Call Volume')
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def generate_data_quality_report(self):
    """
    Generate comprehensive data quality report
    """
    print("\n" + "="*60)
    print("DATA QUALITY ASSESSMENT REPORT")
    print("="*60)
    
    report = {
        'call_data': {},
        'mail_data': {},
        'combined_data': {},
        'recommendations': []
    }
    
    # Call data quality
    call_data = self.call_data_clean
    report['call_data'] = {
        'total_records': len(call_data),
        'date_range': f"{call_data['date'].min()} to {call_data['date'].max()}",
        'missing_values': call_data.isnull().sum().sum(),
        'zero_values': (call_data['call_volume'] == 0).sum(),
        'negative_values': (call_data['call_volume'] < 0).sum(),
        'outliers_removed': getattr(self, 'outliers_removed', 0)
    }
    
    # Mail data quality
    mail_data = self.mail_data_clean
    report['mail_data'] = {
        'total_records': len(mail_data),
        'date_range': f"{mail_data['date'].min()} to {mail_data['date'].max()}",
        'missing_values': mail_data.isnull().sum().sum(),
        'zero_values': (mail_data['mail_volume'] == 0).sum(),
        'negative_values': (mail_data['mail_volume'] < 0).sum(),
        'outliers_removed': getattr(self, 'mail_outliers_removed', 0)
    }
    
    # Combined data quality
    if self.combined_data is not None:
        combined = self.combined_data
        overlap = ((combined['call_volume'] > 0) & (combined['mail_volume'] > 0)).sum()
        report['combined_data'] = {
            'total_records': len(combined),
            'overlapping_records': overlap,
            'overlap_percentage': overlap / len(combined) * 100,
            'call_only_records': ((combined['call_volume'] > 0) & (combined['mail_volume'] == 0)).sum(),
            'mail_only_records': ((combined['call_volume'] == 0) & (combined['mail_volume'] > 0)).sum()
        }
    
    # Print report
    print("\nCALL DATA QUALITY:")
    for key, value in report['call_data'].items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print("\nMAIL DATA QUALITY:")
    for key, value in report['mail_data'].items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    if report['combined_data']:
        print("\nCOMBINED DATA QUALITY:")
        for key, value in report['combined_data'].items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Generate recommendations
    recommendations = []
    
    if report['call_data']['zero_values'] > len(call_data) * 0.1:
        recommendations.append("High number of zero call volume days - investigate if these are true zeros or missing data")
    
    if report['mail_data']['zero_values'] > len(mail_data) * 0.1:
        recommendations.append("High number of zero mail volume days - verify mail campaign scheduling")
    
    if report['combined_data'] and report['combined_data']['overlap_percentage'] < 50:
        recommendations.append("Low overlap between call and mail data - consider data alignment issues")
    
    if report['call_data']['negative_values'] > 0:
        recommendations.append("Negative call volumes detected - data cleaning required")
    
    if report['mail_data']['negative_values'] > 0:
        recommendations.append("Negative mail volumes detected - data cleaning required")
    
    print("\nRECOMMendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    return report

def prepare_for_modeling(self, target_lag_days=None):
    """
    Prepare final dataset for time series modeling
    """
    print("\n" + "="*50)
    print("PREPARING DATA FOR TIME SERIES MODELING")
    print("="*50)
    
    if self.combined_data is None:
        self.combine_datasets(lag_days=target_lag_days)
    
    # Create modeling dataset
    modeling_data = self.combined_data.copy()
    
    # Add time-based features
    modeling_data['year'] = modeling_data['date'].dt.year
    modeling_data['month'] = modeling_data['date'].dt.month
    modeling_data['day'] = modeling_data['date'].dt.day
    modeling_data['day_of_week'] = modeling_data['date'].dt.dayofweek
    modeling_data['day_of_year'] = modeling_data['date'].dt.dayofyear
    modeling_data['week_of_year'] = modeling_data['date'].dt.isocalendar().week
    modeling_data['quarter'] = modeling_data['date'].dt.quarter
    modeling_data['is_weekend'] = modeling_data['day_of_week'].isin([5, 6]).astype(int)
    
    # Add lag features for mail volume
    for lag in [1, 2, 3, 5, 7, 14]:
        modeling_data[f'mail_volume_lag_{lag}'] = modeling_data['mail_volume'].shift(lag)
    
    # Add rolling averages
    for window in [3, 7, 14, 30]:
        modeling_data[f'mail_volume_ma_{window}'] = modeling_data['mail_volume'].rolling(window=window).mean()
        modeling_data[f'call_volume_ma_{window}'] = modeling_data['call_volume'].rolling(window=window).mean()
    
    # Remove rows with NaN values created by lag/rolling features
    modeling_data_clean = modeling_data.dropna().reset_index(drop=True)
    
    print(f"Final modeling dataset: {len(modeling_data_clean)} observations")
    print(f"Features available: {len(modeling_data_clean.columns)} columns")
    print(f"Date range: {modeling_data_clean['date'].min()} to {modeling_data_clean['date'].max()}")
    
    # Split features and target
    feature_cols = [col for col in modeling_data_clean.columns if col not in ['date', 'call_volume']]
    X = modeling_data_clean[feature_cols]
    y = modeling_data_clean['call_volume']
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    print(f"\nFeature columns:")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {col}")
    
    self.modeling_data = modeling_data_clean
    self.X = X
    self.y = y
    
    return modeling_data_clean, X, y
```

# Example usage

def main():
“””
Example usage of the CallVolumeAnalyzer
“””
# Initialize analyzer
analyzer = CallVolumeAnalyzer()

```
# Load and process data
print("Step 1: Loading and aggregating call data...")
# analyzer.load_and_aggregate_calls('path/to/call_data.csv', date_col='call_date', volume_col='call_count')

print("Step 2: Loading and aggregating mail data...")
# mail_files = ['path/to/mail_source1.csv', 'path/to/mail_source2.csv', 'path/to/mail_source3.csv']
# analyzer.load_and_aggregate_mail(mail_files, date_col='mail_date', volume_col='pieces_sent')

print("Step 3: Performing EDA on call data...")
# analyzer.perform_call_eda(remove_outliers=True)

print("Step 4: Performing EDA on mail data...")
# analyzer.perform_mail_eda(remove_outliers=True)

print("Step 5: Combining datasets and comparative analysis...")
# analyzer.combine_datasets()
# analyzer.comparative_analysis()

print("Step 6: Creating visualizations...")
# analyzer.create_visualizations()

print("Step 7: Generating data quality report...")
# analyzer.generate_data_quality_report()

print("Step 8: Preparing data for modeling...")
# modeling_data, X, y = analyzer.prepare_for_modeling()

print("Analysis complete! Ready for time series modeling.")
```

# Advanced Time Series Modeling Functions

class TimeSeriesModeler:
def **init**(self, data, target_col=‘call_volume’, date_col=‘date’):
self.data = data
self.target_col = target_col
self.date_col = date_col
self.models = {}
self.predictions = {}
self.metrics = {}

```
def split_data(self, test_size=0.2, validation_size=0.1):
    """
    Split data into train, validation, and test sets chronologically
    """
    from sklearn.model_selection import train_test_split
    
    n = len(self.data)
    test_start = int(n * (1 - test_size))
    val_start = int(n * (1 - test_size - validation_size))
    
    self.train_data = self.data[:val_start].copy()
    self.val_data = self.data[val_start:test_start].copy()
    self.test_data = self.data[test_start:].copy()
    
    print(f"Data split:")
    print(f"  Training: {len(self.train_data)} samples ({self.train_data[self.date_col].min()} to {self.train_data[self.date_col].max()})")
    print(f"  Validation: {len(self.val_data)} samples ({self.val_data[self.date_col].min()} to {self.val_data[self.date_col].max()})")
    print(f"  Test: {len(self.test_data)} samples ({self.test_data[self.date_col].min()} to {self.test_data[self.date_col].max()})")
    
    return self.train_data, self.val_data, self.test_data

def train_linear_regression(self, feature_cols=None):
    """
    Train linear regression model
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    if feature_cols is None:
        feature_cols = [col for col in self.train_data.columns if col not in [self.date_col, self.target_col]]
    
    # Prepare data
    X_train = self.train_data[feature_cols]
    y_train = self.train_data[self.target_col]
    X_val = self.val_data[feature_cols]
    y_val = self.val_data[self.target_col]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)
    
    # Calculate metrics
    train_metrics = {
        'mae': mean_absolute_error(y_train, y_train_pred),
        'mse': mean_squared_error(y_train, y_train_pred),
        'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'r2': r2_score(y_train, y_train_pred)
    }
    
    val_metrics = {
        'mae': mean_absolute_error(y_val, y_val_pred),
        'mse': mean_squared_error(y_val, y_val_pred),
        'rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
        'r2': r2_score(y_val, y_val_pred)
    }
    
    self.models['linear_regression'] = {
        'model': model,
        'scaler': scaler,
        'features': feature_cols
    }
    
    self.metrics['linear_regression'] = {
        'train': train_metrics,
        'validation': val_metrics
    }
    
    self.predictions['linear_regression'] = {
        'train': y_train_pred,
        'validation': y_val_pred
    }
    
    print("Linear Regression Results:")
    print(f"  Training RMSE: {train_metrics['rmse']:.2f}")
    print(f"  Validation RMSE: {val_metrics['rmse']:.2f}")
    print(f"  Training R²: {train_metrics['r2']:.4f}")
    print(f"  Validation R²: {val_metrics['r2']:.4f}")
    
    return model, scaler

def train_random_forest(self, feature_cols=None, n_estimators=100):
    """
    Train Random Forest model
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    if feature_cols is None:
        feature_cols = [col for col in self.train_data.columns if col not in [self.date_col, self.target_col]]
    
    # Prepare data
    X_train = self.train_data[feature_cols]
    y_train = self.train_data[self.target_col]
    X_val = self.val_data[feature_cols]
    y_val = self.val_data[self.target_col]
    
    # Train model
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Calculate metrics
    train_metrics = {
        'mae': mean_absolute_error(y_train, y_train_pred),
        'mse': mean_squared_error(y_train, y_train_pred),
        'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'r2': r2_score(y_train, y_train_pred)
    }
    
    val_metrics = {
        'mae': mean_absolute_error(y_val, y_val_pred),
        'mse': mean_squared_error(y_val, y_val_pred),
        'rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
        'r2': r2_score(y_val, y_val_pred)
    }
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    self.models['random_forest'] = {
        'model': model,
        'features': feature_cols,
        'feature_importance': feature_importance
    }
    
    self.metrics['random_forest'] = {
        'train': train_metrics,
        'validation': val_metrics
    }
    
    self.predictions['random_forest'] = {
        'train': y_train_pred,
        'validation': y_val_pred
    }
    
    print("Random Forest Results:")
    print(f"  Training RMSE: {train_metrics['rmse']:.2f}")
    print(f"  Validation RMSE: {val_metrics['rmse']:.2f}")
    print(f"  Training R²: {train_metrics['r2']:.4f}")
    print(f"  Validation R²: {val_metrics['r2']:.4f}")
    print("\nTop 10 Feature Importances:")
    print(feature_importance.head(10))
    
    return model

def train_arima_model(self, order=(1,1,1), seasonal_order=(1,1,1,7)):
    """
    Train ARIMA model for time series forecasting
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        # Prepare time series data
        ts_data = self.train_data.set_index(self.date_col)[self.target_col]
        ts_val = self.val_data.set_index(self.date_col)[self.target_col]
        
        # Fit SARIMAX model (includes seasonal component)
        model = SARIMAX(ts_data, order=order, seasonal_order=seasonal_order)
        fitted_model = model.fit(disp=False)
        
        # Make predictions
        train_pred = fitted_model.fittedvalues
        val_pred = fitted_model.forecast(steps=len(ts_val))
        
        # Calculate metrics
        train_metrics = {
            'mae': mean_absolute_error(ts_data, train_pred),
            'mse': mean_squared_error(ts_data, train_pred),
            'rmse': np.sqrt(mean_squared_error(ts_data, train_pred))
        }
        
        val_metrics = {
            'mae': mean_absolute_error(ts_val, val_pred),
            'mse': mean_squared_error(ts_val, val_pred),
            'rmse': np.sqrt(mean_squared_error(ts_val, val_pred))
        }
        
        self.models['arima'] = {
            'model': fitted_model,
            'order': order,
            'seasonal_order': seasonal_order
        }
        
        self.metrics['arima'] = {
            'train': train_metrics,
            'validation': val_metrics
        }
        
        self.predictions['arima'] = {
            'train': train_pred,
            'validation': val_pred
        }
        
        print("ARIMA Model Results:")
        print(f"  Training RMSE: {train_metrics['rmse']:.2f}")
        print(f"  Validation RMSE: {val_metrics['rmse']:.2f}")
        print(f"  Model Order: {order}")
        print(f"  Seasonal Order: {seasonal_order}")
        
        return fitted_model
        
    except ImportError:
        print("statsmodels not available. Install with: pip install statsmodels")
        return None

def compare_models(self):
    """
    Compare all trained models
    """
    if not self.metrics:
        print("No models trained yet!")
        return
    
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    comparison_df = []
    for model_name, metrics in self.metrics.items():
        comparison_df.append({
            'Model': model_name,
            'Train_RMSE': metrics['train']['rmse'],
            'Val_RMSE': metrics['validation']['rmse'],
            'Train_R2': metrics['train'].get('r2', 'N/A'),
            'Val_R2': metrics['validation'].get('r2', 'N/A')
        })
    
    comparison_df = pd.DataFrame(comparison_df)
    print(comparison_df.to_string(index=False))
    
    # Best model by validation RMSE
    best_model = comparison_df.loc[comparison_df['Val_RMSE'].idxmin(), 'Model']
    print(f"\nBest model by validation RMSE: {best_model}")
    
    return comparison_df

def plot_predictions(self, model_name=None):
    """
    Plot actual vs predicted values
    """
    if model_name is None:
        model_names = list(self.models.keys())
    else:
        model_names = [model_name]
    
    fig, axes = plt.subplots(len(model_names), 1, figsize=(15, 5*len(model_names)))
    if len(model_names) == 1:
        axes = [axes]
    
    for i, model_name in enumerate(model_names):
        ax = axes[i]
        
        # Plot training data
        train_dates = self.train_data[self.date_col]
        train_actual = self.train_data[self.target_col]
        train_pred = self.predictions[model_name]['train']
        
        ax.plot(train_dates, train_actual, label='Actual (Train)', color='blue', alpha=0.7)
        ax.plot(train_dates, train_pred, label='Predicted (Train)', color='red', alpha=0.7)
        
        # Plot validation data
        val_dates = self.val_data[self.date_col]
        val_actual = self.val_data[self.target_col]
        val_pred = self.predictions[model_name]['validation']
        
        ax.plot(val_dates, val_actual, label='Actual (Val)', color='blue', alpha=0.7, linestyle='--')
        ax.plot(val_dates, val_pred, label='Predicted (Val)', color='red', alpha=0.7, linestyle='--')
        
        ax.set_title(f'{model_name.replace("_", " ").title()} - Predictions vs Actual')
        ax.set_xlabel('Date')
        ax.set_ylabel('Call Volume')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add vertical line to separate train/val
        ax.axvline(x=val_dates.iloc[0], color='black', linestyle=':', alpha=0.5, label='Train/Val Split')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def forecast_future(self, model_name, days_ahead=30, mail_volume_forecast=None):
    """
    Make future predictions
    """
    if model_name not in self.models:
        print(f"Model {model_name} not found!")
        return None
    
    model_info = self.models[model_name]
    
    if model_name == 'arima':
        # ARIMA forecasting
        model = model_info['model']
        forecast = model.forecast(steps=days_ahead)
        
        # Create future dates
        last_date = self.data[self.date_col].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_ahead)
        
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'predicted_calls': forecast
        })
        
    else:
        # ML model forecasting (requires future mail volume data)
        if mail_volume_forecast is None:
            print("Mail volume forecast required for ML models!")
            return None
        
        model = model_info['model']
        features = model_info['features']
        
        # Create future feature matrix (simplified - would need proper feature engineering)
        last_date = self.data[self.date_col].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_ahead)
        
        # This is a simplified example - in practice, you'd need to properly engineer all features
        future_features = pd.DataFrame({
            'date': future_dates,
            'mail_volume': mail_volume_forecast[:days_ahead] if len(mail_volume_forecast) >= days_ahead else [0]*days_ahead
        })
        
        # Add time-based features
        future_features['year'] = future_features['date'].dt.year
        future_features['month'] = future_features['date'].dt.month
        future_features['day'] = future_features['date'].dt.day
        future_features['day_of_week'] = future_features['date'].dt.dayofweek
        future_features['day_of_year'] = future_features['date'].dt.dayofyear
        future_features['week_of_year'] = future_features['date'].dt.isocalendar().week
        future_features['quarter'] = future_features['date'].dt.quarter
        future_features['is_weekend'] = future_features['day_of_week'].isin([5, 6]).astype(int)
        
        # For features not available, fill with historical means or use simple imputation
        for feature in features:
            if feature not in future_features.columns:
                future_features[feature] = self.data[feature].mean()
        
        X_future = future_features[features]
        
        if model_name == 'linear_regression':
            scaler = model_info['scaler']
            X_future_scaled = scaler.transform(X_future)
            predictions = model.predict(X_future_scaled)
        else:
            predictions = model.predict(X_future)
        
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'predicted_calls': predictions
        })
    
    print(f"Future forecast for {days_ahead} days:")
    print(forecast_df.head(10))
    
    return forecast_df
```

if **name** == “**main**”:
main()

# Data Quality Checks and Recommendations

def perform_data_quality_checks(call_data, mail_data):
“””
Perform comprehensive data quality checks
“””
print(”\n” + “=”*60)
print(“COMPREHENSIVE DATA QUALITY ASSESSMENT”)
print(”=”*60)

```
issues = []
recommendations = []

# Check 1: Date consistency
call_date_range = (call_data['date'].min(), call_data['date'].max())
mail_date_range = (mail_data['date'].min(), mail_data['date'].max())

print(f"Call data date range: {call_date_range[0]} to {call_date_range[1]}")
print(f"Mail data date range: {mail_date_range[0]} to {mail_date_range[1]}")

if call_date_range[0] > mail_date_range[1] or mail_date_range[0] > call_date_range[1]:
    issues.append("No date overlap between call and mail data")
    recommendations.append("Verify data collection periods and ensure temporal alignment")

# Check 2: Data gaps
call_date_gaps = pd.date_range(call_date_range[0], call_date_range[1]).difference(call_data['date'])
mail_date_gaps = pd.date_range(mail_date_range[0], mail_date_range[1]).difference(mail_data['date'])

if len(call_date_gaps) > 0:
    issues.append(f"Call data has {len(call_date_gaps)} missing dates")
    recommendations.append("Fill missing call dates with zeros or interpolated values")

if len(mail_date_gaps) > 0:
    issues.append(f"Mail data has {len(mail_date_gaps)} missing dates")
    recommendations.append("Verify mail campaign schedule for missing dates")

# Check 3: Extreme values
call_q99 = call_data['call_volume'].quantile(0.99)
call_outliers = (call_data['call_volume'] > call_q99 * 3).sum()

if call_outliers > 0:
    issues.append(f"Call data has {call_outliers} extreme outliers")
    recommendations.append("Investigate extreme call volume spikes for data quality issues")

# Check 4: Business logic validation
weekend_mail = mail_data[mail_data['date'].dt.dayofweek.isin([5, 6])]['mail_volume'].sum()
total_mail = mail_data['mail_volume'].sum()

if weekend_mail > total_mail * 0.1:
    issues.append("Significant mail volume on weekends detected")
    recommendations.append("Verify if weekend mail delivery is expected for your business")

# Check 5: Seasonal patterns
if len(call_data) > 365:
    monthly_variance = call_data.groupby(call_data['date'].dt.month)['call_volume'].var()
    if monthly_variance.max() > monthly_variance.mean() * 10:
        issues.append("Extremely high seasonal variance detected")
        recommendations.append("Consider seasonal adjustment or separate seasonal models")

# Print summary
print(f"\nISSUES IDENTIFIED: {len(issues)}")
for i, issue in enumerate(issues, 1):
    print(f"  {i}. {issue}")

print(f"\nRECOMMENDATIONS: {len(recommendations)}")
for i, rec in enumerate(recommendations, 1):
    print(f"  {i}. {rec}")

return issues, recommendations
```

# Additional utility functions for advanced analysis

def calculate_response_rates(call_data, mail_data, lag_days=None):
“””
Calculate response rates with optional lag
“””
if lag_days:
mail_data = mail_data.copy()
mail_data[‘date’] = mail_data[‘date’] + timedelta(days=lag_days)

```
merged = pd.merge(call_data, mail_data, on='date', how='inner')
merged = merged[(merged['call_volume'] > 0) & (merged['mail_volume'] > 0)]

if len(merged) == 0:
    return None

merged['response_rate'] = merged['call_volume'] / merged['mail_volume'] * 100

# Remove extreme response rates (likely data issues)
clean_rates = merged[merged['response_rate'] <= 50]

stats = {
    'mean_response_rate': clean_rates['response_rate'].mean(),
    'median_response_rate': clean_rates['response_rate'].median(),
    'std_response_rate': clean_rates['response_rate'].std(),
    'min_response_rate': clean_rates['response_rate'].min(),
    'max_response_rate': clean_rates['response_rate'].max(),
    'total_observations': len(clean_rates)
}

return stats, clean_rates
```

def find_optimal_lag(call_data, mail_data, max_lag=21):
“””
Find optimal lag between mail and calls
“””
from scipy.stats import pearsonr

```
lag_results = []

for lag in range(0, max_lag + 1):
    mail_shifted = mail_data.copy()
    mail_shifted['date'] = mail_shifted['date'] + timedelta(days=lag)
    
    merged = pd.merge(call_data, mail_shifted, on='date', how='inner')
    merged = merged[(merged['call_volume'] > 0) & (merged['mail_volume'] > 0)]
    
    if len(merged) > 10:  # Need sufficient observations
        corr, p_value = pearsonr(merged['call_volume'], merged['mail_volume'])
        lag_results.append({
            'lag_days': lag,
            'correlation': corr,
            'p_value': p_value,
            'n_observations': len(merged)
        })

if not lag_results:
    return None

lag_df = pd.DataFrame(lag_results)
optimal_lag = lag_df.loc[lag_df['correlation'].idxmax()]

return optimal_lag, lag_df
```

print(“Time Series Call Volume Prediction System Ready!”)
print(”=”*60)
print(“Available Classes:”)
print(“1. CallVolumeAnalyzer - Main analysis class”)
print(“2. TimeSeriesModeler - Advanced modeling class”)
print(”\nKey Features:”)
print(“✓ Data loading and aggregation”)
print(“✓ Comprehensive EDA with outlier detection”)
print(“✓ Multi-method outlier detection”)
print(“✓ Comparative analysis between calls and mail”)
print(“✓ Lag analysis and optimization”)
print(“✓ Multiple modeling approaches (Linear, RF, ARIMA)”)
print(“✓ Data quality assessment”)
print(“✓ Future forecasting capabilities”)
print(“✓ Comprehensive visualizations”)
print(”\nTo get started, instantiate CallVolumeAnalyzer() and follow the example usage!”)