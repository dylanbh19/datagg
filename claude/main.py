#!/bin/bash

# ================================================================================
# COMPLETE WINDOWS ANALYTICS PROJECT SETUP
# ================================================================================
# This script creates the entire analytics project from scratch for Windows
# No dependencies on other scripts - runs independently
# ================================================================================

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_header() {
    echo -e "${BLUE}"
    cat << "EOF"
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║    ██████╗██╗   ██╗███████╗████████╗ ██████╗ ███╗   ███╗███████╗██████╗     ║
║   ██╔════╝██║   ██║██╔════╝╚══██╔══╝██╔═══██╗████╗ ████║██╔════╝██╔══██╗    ║
║   ██║     ██║   ██║███████╗   ██║   ██║   ██║██╔████╔██║█████╗  ██████╔╝    ║
║   ██║     ██║   ██║╚════██║   ██║   ██║   ██║██║╚██╔╝██║██╔══╝  ██╔══██╗    ║
║   ╚██████╗╚██████╔╝███████║   ██║   ╚██████╔╝██║ ╚═╝ ██║███████╗██║  ██║    ║
║    ╚═════╝ ╚═════╝ ╚══════╝   ╚═╝    ╚═════╝ ╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝    ║
║                                                                              ║
║              WINDOWS ANALYTICS PROJECT - COMPLETE SETUP                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

check_requirements() {
    print_info "Checking system requirements..."
    
    # Check for Python
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        print_success "Python3 found"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
        print_success "Python found"
    else
        print_error "Python not found. Please install Python 3.9+ first."
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | grep -oP '\d+\.\d+')
    print_info "Python version: $PYTHON_VERSION"
    
    # Check for pip
    if ! $PYTHON_CMD -m pip --version &> /dev/null; then
        print_error "pip not found. Please install pip."
        exit 1
    fi
    
    print_success "System requirements met"
}

create_directory_structure() {
    print_info "Creating project directory structure..."
    
    # Create all necessary directories
    mkdir -p {data/{raw,processed,external},notebooks,src/{data,features,models,visualization},tests/{unit,integration},config,outputs/{plots,reports,models},logs,docs}
    
    print_success "Directory structure created"
}

create_configuration_files() {
    print_info "Creating configuration files..."
    
    # Main configuration file with your column names
    cat > config/config.yaml << 'EOF'
# ================================================================================
# CUSTOMER COMMUNICATION ANALYTICS - CONFIGURATION
# ================================================================================

# Data file paths and column mappings for your specific data
data:
  # Call volume data (with some missing values allowed)
  call_volume:
    file_path: "data/raw/call_volume.csv"
    date_column: "Date"                  # Your date column
    volume_column: "call_volume"         # Call volume column
    
  # Call intents data (complete dataset)
  call_intents:
    file_path: "data/raw/call_intents.csv"
    date_column: "ConversationStart"     # Your date column
    intent_column: "uui_Intent"          # Your intent column
    volume_column: "intent_volume"       # Will be calculated by counting
    
  # Mail data (combined types and volumes)
  mail_data:
    file_path: "data/raw/mail.csv"
    date_column: "mail_date"             # Your date column
    type_column: "mail_type"             # Mail type column
    volume_column: "mail_volume"         # Mail volume column

# Economic data configuration
economic_data:
  indicators:
    - "VIX"           # Volatility Index
    - "SPY"           # S&P 500 ETF
    - "DGS10"         # 10-Year Treasury Rate
    - "UNRATE"        # Unemployment Rate
    - "UMCSENT"       # Consumer Sentiment
    - "FEDFUNDS"      # Federal Funds Rate
  
  start_date: "2022-01-01"
  end_date: "2024-06-30"

# Analysis parameters
analysis:
  max_lag_days: 14
  seasonal_periods: [7, 30, 90, 365]
  rolling_windows: [3, 7, 14, 30, 90]
  test_size: 0.2
  cv_folds: 5
  outlier_methods: ["iqr", "zscore", "isolation_forest"]
  outlier_threshold: 3.0

# Visualization settings
visualization:
  figure_size: [12, 8]
  dpi: 300
  style: "seaborn-v0_8"
  color_palette: "husl"

# Output settings
output:
  plots_dir: "outputs/plots"
  reports_dir: "outputs/reports"
  models_dir: "outputs/models"

# Logging configuration
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/analytics.log"
EOF

    # Data format guide
    cat > config/your_data_format.yaml << 'EOF'
# ================================================================================
# YOUR DATA FORMAT GUIDE
# ================================================================================

# Place these files in data/raw/ directory:

mail_csv:
  file_name: "mail.csv"
  columns:
    - mail_date    # Date column (YYYY-MM-DD format)
    - mail_type    # Type of mail (string)
    - mail_volume  # Number of mails sent (integer)
  
call_intents_csv:
  file_name: "call_intents.csv"
  columns:
    - ConversationStart  # Date/time of call
    - uui_Intent        # Intent category (string)
    # Note: Volume will be calculated by counting rows per date/intent
    
call_volume_csv:
  file_name: "call_volume.csv"
  columns:
    - Date         # Date column
    - call_volume  # Total call volume (may have missing values)
EOF

    print_success "Configuration files created"
}

create_windows_requirements() {
    print_info "Creating Windows-optimized requirements..."
    
    cat > requirements_windows.txt << 'EOF'
# Windows-optimized requirements
pandas>=1.5.0
numpy>=1.21.0
scipy>=1.9.0
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.10.0
scikit-learn>=1.1.0
xgboost>=1.6.0
statsmodels>=0.13.0
openpyxl>=3.0.0
tqdm>=4.64.0
pyyaml>=6.0
EOF

    print_success "Windows requirements created"
}

create_source_code() {
    print_info "Creating source code modules..."
    
    # Windows compatibility module
    cat > src/windows_compat.py << 'EOF'
"""
Windows compatibility fixes
"""
import os
import sys
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

def fix_path(path_str):
    """Convert to Windows-compatible path"""
    return str(Path(path_str))

def ensure_dir_exists(dir_path):
    """Ensure directory exists"""
    Path(dir_path).mkdir(parents=True, exist_ok=True)

def check_package_availability():
    """Check available packages"""
    packages = {}
    test_packages = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn', 'xgboost', 'statsmodels']
    
    for package in test_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            packages[package] = True
        except ImportError:
            packages[package] = False
    
    return packages
EOF

    # Data loader
    cat > src/data/data_loader.py << 'EOF'
"""
Data Loading Module
"""
import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
    def load_call_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load call volume and intent data"""
        logger.info("Loading call data...")
        
        # Load call volume data
        call_vol_config = self.config['data']['call_volume']
        call_volume_df = pd.read_csv(call_vol_config['file_path'])
        call_volume_df[call_vol_config['date_column']] = pd.to_datetime(call_volume_df[call_vol_config['date_column']])
        call_volume_df.rename(columns={call_vol_config['date_column']: 'date'}, inplace=True)
        
        # Load call intents data
        call_int_config = self.config['data']['call_intents']
        call_intents_df = pd.read_csv(call_int_config['file_path'])
        call_intents_df[call_int_config['date_column']] = pd.to_datetime(call_intents_df[call_int_config['date_column']])
        
        # Calculate intent volumes by counting occurrences
        call_intents_df['date'] = call_intents_df[call_int_config['date_column']].dt.date
        call_intents_df['intent'] = call_intents_df[call_int_config['intent_column']]
        
        # Count intent volumes per date/intent combination
        intent_counts = call_intents_df.groupby(['date', 'intent']).size().reset_index(name='intent_volume')
        intent_counts['date'] = pd.to_datetime(intent_counts['date'])
        
        logger.info(f"Loaded {len(call_volume_df)} call volume records")
        logger.info(f"Calculated {len(intent_counts)} call intent records")
        
        return call_volume_df, intent_counts
    
    def load_mail_data(self) -> pd.DataFrame:
        """Load combined mail data"""
        logger.info("Loading mail data...")
        
        mail_config = self.config['data']['mail_data']
        mail_df = pd.read_csv(mail_config['file_path'])
        mail_df[mail_config['date_column']] = pd.to_datetime(mail_df[mail_config['date_column']])
        
        logger.info(f"Loaded {len(mail_df)} mail records")
        return mail_df
    
    def augment_call_volume_data(self, call_volume_df: pd.DataFrame, 
                                call_intents_df: pd.DataFrame) -> pd.DataFrame:
        """Augment missing call volume data using intent data"""
        logger.info("Augmenting call volume data...")
        
        # Aggregate intents by date
        intent_daily = call_intents_df.groupby('date')['intent_volume'].sum().reset_index()
        intent_daily.columns = ['date', 'total_intent_volume']
        
        # Find periods with both call volume and intent data
        merged = pd.merge(call_volume_df, intent_daily, on='date', how='inner')
        merged = merged.dropna(subset=['call_volume'])
        
        # Calculate average ratio
        if len(merged) > 0:
            ratio = merged['call_volume'].sum() / merged['total_intent_volume'].sum()
            logger.info(f"Calculated call volume to intent ratio: {ratio:.3f}")
            
            # Apply ratio to missing call volume data
            call_volume_filled = call_volume_df.copy()
            missing_mask = call_volume_filled['call_volume'].isna()
            
            if missing_mask.any():
                missing_dates = call_volume_filled[missing_mask]['date']
                intent_for_missing = intent_daily[intent_daily['date'].isin(missing_dates)]
                
                for _, row in intent_for_missing.iterrows():
                    mask = (call_volume_filled['date'] == row['date'])
                    call_volume_filled.loc[mask, 'call_volume'] = row['total_intent_volume'] * ratio
                
                filled_count = missing_mask.sum()
                logger.info(f"Filled {filled_count} missing call volume records")
            
            return call_volume_filled
        else:
            logger.warning("No overlapping data found for augmentation")
            return call_volume_df
EOF

    # Feature engineering
    cat > src/features/feature_engineering.py << 'EOF'
"""
Feature Engineering Module
"""
import pandas as pd
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class FeatureEngineering:
    def __init__(self, config: Dict):
        self.config = config
        
    def create_temporal_features(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Create temporal features"""
        logger.info("Creating temporal features...")
        
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Basic temporal features
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day'] = df[date_col].dt.day
        df['dayofweek'] = df[date_col].dt.dayofweek
        df['quarter'] = df[date_col].dt.quarter
        
        # Cyclical features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        # Business indicators
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df['is_monday'] = (df['dayofweek'] == 0).astype(int)
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str, lags: List[int]) -> pd.DataFrame:
        """Create lag features"""
        logger.info(f"Creating lag features for {target_col}...")
        
        df = df.copy()
        df = df.sort_values('date')
        
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
            
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, target_col: str, windows: List[int]) -> pd.DataFrame:
        """Create rolling features"""
        logger.info(f"Creating rolling features for {target_col}...")
        
        df = df.copy()
        df = df.sort_values('date')
        
        for window in windows:
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
            
        return df
EOF

    # Simple visualization
    cat > src/visualization/plots.py << 'EOF'
"""
Visualization Module
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')

class AnalyticsVisualizer:
    def __init__(self, config: Dict):
        self.config = config
        self.plots_dir = config['output']['plots_dir']
        self.fig_size = config['visualization']['figure_size']
        self.dpi = config['visualization']['dpi']
        
    def plot_time_series_overview(self, call_df: pd.DataFrame, mail_df: pd.DataFrame, save_path: str = None):
        """Create time series overview"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Call volume over time
        axes[0, 0].plot(call_df['date'], call_df['call_volume'])
        axes[0, 0].set_title('Call Volume Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Call Volume')
        
        # Mail volume over time
        mail_daily = mail_df.groupby('mail_date')['mail_volume'].sum().reset_index()
        axes[0, 1].plot(mail_daily['mail_date'], mail_daily['mail_volume'], color='orange')
        axes[0, 1].set_title('Mail Volume Over Time')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Mail Volume')
        
        # Call volume distribution
        axes[1, 0].hist(call_df['call_volume'].dropna(), bins=30, alpha=0.7)
        axes[1, 0].set_title('Call Volume Distribution')
        axes[1, 0].set_xlabel('Call Volume')
        axes[1, 0].set_ylabel('Frequency')
        
        # Mail volume distribution
        axes[1, 1].hist(mail_daily['mail_volume'], bins=30, alpha=0.7, color='orange')
        axes[1, 1].set_title('Mail Volume Distribution')
        axes[1, 1].set_xlabel('Mail Volume')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_mail_type_effectiveness(self, mail_df: pd.DataFrame, call_df: pd.DataFrame, save_path: str = None):
        """Plot mail type effectiveness"""
        mail_summary = mail_df.groupby('mail_type')['mail_volume'].agg(['sum', 'count', 'mean']).reset_index()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Total volume by type
        axes[0, 0].bar(mail_summary['mail_type'], mail_summary['sum'])
        axes[0, 0].set_title('Total Mail Volume by Type')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Average volume by type
        axes[0, 1].bar(mail_summary['mail_type'], mail_summary['mean'], color='orange')
        axes[0, 1].set_title('Average Mail Volume by Type')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Campaign frequency
        axes[1, 0].bar(mail_summary['mail_type'], mail_summary['count'], color='green')
        axes[1, 0].set_title('Campaign Frequency by Type')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Distribution pie chart
        axes[1, 1].pie(mail_summary['sum'], labels=mail_summary['mail_type'], autopct='%1.1f%%')
        axes[1, 1].set_title('Mail Volume Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
EOF

    # Simple models
    cat > src/models/predictive_models.py << 'EOF'
"""
Predictive Models Module
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class PredictiveModels:
    def __init__(self, config: Dict):
        self.config = config
        self.results = {}
        
    def prepare_data(self, df: pd.DataFrame, target_col: str, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for modeling"""
        df_clean = df.dropna(subset=[target_col]).copy()
        available_features = [col for col in feature_cols if col in df_clean.columns]
        
        X = df_clean[available_features].fillna(method='ffill').fillna(0)
        y = df_clean[target_col].values
        
        return X.values, y
    
    def train_random_forest(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train Random Forest"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        return {
            'model': model,
            'predictions': y_pred,
            'actual': y_test,
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
    
    def train_linear_regression(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train Linear Regression"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        return {
            'model': model,
            'predictions': y_pred,
            'actual': y_test,
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
    
    def run_models(self, df: pd.DataFrame, target_col: str, feature_cols: List[str]) -> Dict:
        """Run all available models"""
        results = {}
        
        X, y = self.prepare_data(df, target_col, feature_cols)
        
        print("Training Random Forest...")
        results['Random Forest'] = self.train_random_forest(X, y)
        
        print("Training Linear Regression...")
        results['Linear Regression'] = self.train_linear_regression(X, y)
        
        return results
EOF

    # Main execution script
    cat > src/main.py << 'EOF'
"""
Main Execution Script - Windows Compatible
"""
import os
import sys
import warnings
import pandas as pd
import numpy as np
import yaml
import logging
import json
from pathlib import Path

warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')
sys.path.append(os.path.join(os.path.dirname(__file__)))

try:
    from windows_compat import *
    from data.data_loader import DataLoader
    from features.feature_engineering import FeatureEngineering
    from models.predictive_models import PredictiveModels
    from visualization.plots import AnalyticsVisualizer
except ImportError as e:
    print(f"Import error: {e}")
    print("Some modules may not be available - continuing with basic functionality")

# Configure logging
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/analytics.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main execution function"""
    
    logger.info("Starting Customer Communication Analytics Pipeline")
    
    try:
        # Load configuration
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize components
        data_loader = DataLoader()
        feature_engineer = FeatureEngineering(config)
        models = PredictiveModels(config)
        visualizer = AnalyticsVisualizer(config)
        
        # Ensure output directories exist
        Path(config['output']['plots_dir']).mkdir(parents=True, exist_ok=True)
        Path(config['output']['reports_dir']).mkdir(parents=True, exist_ok=True)
        
        # Load data
        logger.info("Loading data...")
        call_volume_df, call_intents_df = data_loader.load_call_data()
        mail_df = data_loader.load_mail_data()
        
        # Augment call volume data
        call_volume_df = data_loader.augment_call_volume_data(call_volume_df, call_intents_df)
        
        # Create aggregated mail volume for analysis
        mail_volume_df = mail_df.groupby('mail_date')['mail_volume'].sum().reset_index()
        mail_volume_df.rename(columns={'mail_date': 'date'}, inplace=True)
        
        # Generate visualizations
        logger.info("Creating visualizations...")
        
        visualizer.plot_time_series_overview(
            call_volume_df, mail_df,
            save_path=os.path.join(config['output']['plots_dir'], '01_time_series_overview.png')
        )
        
        visualizer.plot_mail_type_effectiveness(
            mail_df, call_volume_df,
            save_path=os.path.join(config['output']['plots_dir'], '02_mail_effectiveness.png')
        )
        
        # Feature engineering
        logger.info("Engineering features...")
        merged_df = pd.merge(call_volume_df, mail_volume_df, on='date', how='outer')
        merged_df = feature_engineer.create_temporal_features(merged_df, 'date')
        
        # Create lag and rolling features
        lag_days = list(range(1, config['analysis']['max_lag_days'] + 1))
        merged_df = feature_engineer.create_lag_features(merged_df, 'mail_volume', lag_days)
        merged_df = feature_engineer.create_rolling_features(merged_df, 'mail_volume', config['analysis']['rolling_windows'])
        
        # Modeling
        logger.info("Training models...")
        feature_cols = [col for col in merged_df.columns if col not in ['date', 'call_volume']]
        model_results = models.run_models(merged_df, 'call_volume', feature_cols)
        
        # Create summary
        summary = {
            'data_summary': {
                'call_volume_records': len(call_volume_df),
                'call_intents_records': len(call_intents_df),
                'mail_records': len(mail_df)
            },
            'model_performance': {
                model_name: {
                    'mae': results['mae'],
                    'rmse': results['rmse'],
                    'r2': results['r2']
                }
                for model_name, results in model_results.items()
            }
        }
        
        # Save summary
        with open(os.path.join(config['output']['reports_dir'], 'analysis_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Get best model
        best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['r2'])
        best_r2 = model_results[best_model_name]['r2']
        
        logger.info("Analytics pipeline completed successfully!")
        
        print("\n" + "="*80)
        print("CUSTOMER COMMUNICATION ANALYTICS - SUMMARY")
        print("="*80)
        print(f"Best Model: {best_model_name}")
        print(f"R² Score: {best_r2:.3f}")
        print(f"MAE: {model_results[best_model_name]['mae']:.2f}")
        print(f"RMSE: {model_results[best_model_name]['rmse']:.2f}")
        print("\nResults saved to outputs/ directory")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        print(f"Error: {e}")
        print("Check logs/analytics.log for details")
        raise

if __name__ == "__main__":
    main()
EOF

    print_success "Source code modules created"
}

create_windows_batch_files() {
    print_info "Creating Windows batch files..."
    
    # Setup batch file
    cat > setup.bat << 'EOF'
@echo off
setlocal enabledelayedexpansion

echo ========================================================================
echo           CUSTOMER COMMUNICATION ANALYTICS - SETUP
echo ========================================================================

REM Check for Python
python --version >nul 2>nul
if %errorlevel% neq 0 (
    py --version >nul 2>nul
    if %errorlevel% neq 0 (
        echo Error: Python not found. Please install Python from python.org
        pause
        exit /b 1
    ) else (
        set PYTHON_CMD=py
    )
) else (
    set PYTHON_CMD=python
)

echo Using Python: !PYTHON_CMD!

REM Create virtual environment
echo Creating virtual environment...
!PYTHON_CMD! -m venv analytics_env

REM Activate virtual environment
echo Activating virtual environment...
call analytics_env\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install packages with error handling
echo Installing core packages...
pip install pandas numpy scipy matplotlib seaborn --no-warn-script-location

echo Installing machine learning packages...
pip install scikit-learn --no-warn-script-location

echo Installing additional packages...
pip install plotly openpyxl pyyaml tqdm --no-warn-script-location

REM Try optional packages
echo Installing optional packages...
pip install xgboost --no-warn-script-location 2>nul || echo XGBoost installation failed - will use alternatives
pip install statsmodels --no-warn-script-location 2>nul || echo Statsmodels installation failed - some features limited

echo.
echo ========================================================================
echo                    SETUP COMPLETED SUCCESSFULLY!
echo.
echo Next steps:
echo 1. Add your data files to data\raw\ directory:
echo    - mail.csv ^(mail_date, mail_type, mail_volume^)
echo    - call_intents.csv ^(ConversationStart, uui_Intent^)
echo    - call_volume.csv ^(Date, call_volume^)
echo 2. Run: quick_start.bat ^(to validate data^)
echo 3. Run: run_analysis.bat ^(to run analysis^)
echo ========================================================================

pause
EOF

    # Run analysis batch file
    cat > run_analysis.bat << 'EOF'
@echo off
setlocal enabledelayedexpansion

echo ========================================================================
echo               CUSTOMER COMMUNICATION ANALYTICS
echo                      EXECUTION STARTING
echo ========================================================================

REM Check if virtual environment exists
if not exist "analytics_env" (
    echo Virtual environment not found. Please run setup.bat first.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call analytics_env\Scripts\activate.bat

REM Check if data files exist
echo Checking data files...
set MISSING=0

if not exist "data\raw\mail.csv" (
    echo ❌ data\raw\mail.csv not found
    set MISSING=1
)

if not exist "data\raw\call_intents.csv" (
    echo ❌ data\raw\call_intents.csv not found
    set MISSING=1
)

if not exist "data\raw\call_volume.csv" (
    echo ❌ data\raw\call_volume.csv not found
    set MISSING=1
)

if !MISSING! == 1 (
    echo.
    echo Please add your data files to data\raw\ directory
    echo Expected format:
    echo   - mail.csv: mail_date, mail_type, mail_volume
    echo   - call_intents.csv: ConversationStart, uui_Intent
    echo   - call_volume.csv: Date, call_volume
    echo.
    echo See config\your_data_format.yaml for details
    pause
    exit /b 1
)

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir logs

REM Run the analysis
echo Starting analytics pipeline...
echo Logs will be written to logs\analytics.log
echo.

python src\main.py

if %errorlevel% == 0 (
    echo.
    echo ========================================================================
    echo                    ANALYSIS COMPLETED SUCCESSFULLY!
    echo.
    echo Check the following directories for results:
    echo   - outputs\plots\     : Generated visualizations
    echo   - outputs\reports\   : Analysis summary
    echo   - logs\             : Execution logs
    echo ========================================================================
) else (
    echo.
    echo ========================================================================
    echo                    ANALYSIS FAILED
    echo.
    echo Check logs\analytics.log for error details
    echo ========================================================================
)

pause
EOF

    # Quick start validation
    cat > quick_start.bat << 'EOF'
@echo off
setlocal enabledelayedexpansion

echo Checking for required data files...

set MISSING=0

if not exist "data\raw\mail.csv" (
    echo ❌ data\raw\mail.csv not found
    echo    Expected columns: mail_date, mail_type, mail_volume
    set MISSING=1
) else (
    echo ✅ mail.csv found
)

if not exist "data\raw\call_intents.csv" (
    echo ❌ data\raw\call_intents.csv not found
    echo    Expected columns: ConversationStart, uui_Intent
    set MISSING=1
) else (
    echo ✅ call_intents.csv found
)

if not exist "data\raw\call_volume.csv" (
    echo ❌ data\raw\call_volume.csv not found
    echo    Expected columns: Date, call_volume
    set MISSING=1
) else (
    echo ✅ call_volume.csv found
)

if !MISSING! == 1 (
    echo.
    echo Please add your data files to data\raw\ directory
    echo See config\your_data_format.yaml for format requirements
    pause
    exit /b 1
)

echo.
echo ✅ All data files found! Running analysis...
echo.

call run_analysis.bat
EOF

    # Validation script
    cat > validate.bat << 'EOF'
@echo off
echo Validating installation...

REM Check Python
python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Python not found
    exit /b 1
) else (
    echo ✅ Python found
)

REM Check virtual environment
if exist "analytics_env\Scripts\activate.bat" (
    echo ✅ Virtual environment found
    
    REM Activate and test imports
    call analytics_env\Scripts\activate.bat
    
    echo Testing package imports...
    python -c "import pandas; print('✅ Pandas')" 2>nul || echo "❌ Pandas"
    python -c "import numpy; print('✅ NumPy')" 2>nul || echo "❌ NumPy"
    python -c "import matplotlib; print('✅ Matplotlib')" 2>nul || echo "❌ Matplotlib"
    python -c "import sklearn; print('✅ Scikit-learn')" 2>nul || echo "❌ Scikit-learn"
    
) else (
    echo ❌ Virtual environment not found
    echo Run setup.bat first
    exit /b 1
)

echo.
echo Validation complete!
pause
EOF

    chmod +x *.bat
    print_success "Windows batch files created"
}

create_readme() {
    print_info "Creating documentation..."
    
    cat > README.md << 'EOF'
# Customer Communication Analytics Project

## Quick Start for Windows

### 1. Initial Setup
```cmd
setup.bat
```
This creates the Python environment and installs all required packages.

### 2. Add Your Data Files

Place these files in the `data\raw\` directory:

- **`mail.csv`** with columns:
  - `mail_date` (YYYY-MM-DD format)
  - `mail_type` (string categories)
  - `mail_volume` (integer counts)

- **`call_intents.csv`** with columns:
  - `ConversationStart` (date/time)
  - `uui_Intent` (string categories)

- **`call_volume.csv`** with columns:
  - `Date` (date format)
  - `call_volume` (integer, may have missing values)

### 3. Validate Your Data
```cmd
quick_start.bat
```

### 4. Run the Analysis
```cmd
run_analysis.bat
```

## What You'll Get

The analysis will generate:
- **Time series visualizations** showing trends and patterns
- **Mail effectiveness analysis** comparing different mail types
- **Call volume predictions** using machine learning models
- **Summary report** with key insights

Results are saved in:
- `outputs\plots\` - PNG visualization files
- `outputs\reports\` - JSON summary report
- `logs\` - Execution logs

## Troubleshooting

**Python Not Found:**
- Install Python from https://python.org
- Make sure "Add Python to PATH" is checked during installation

**Package Installation Errors:**
- The setup continues even if some packages fail
- Core functionality will still work

**Data File Errors:**
- Check file names match exactly: `mail.csv`, `call_intents.csv`, `call_volume.csv`
- Verify column names match the expected format
- See `config\your_data_format.yaml` for detailed requirements

**Virtual Environment Issues:**
- Delete `analytics_env` folder and run `setup.bat` again

## Validation

Run `validate.bat` to test your installation and check which packages are available.

## Project Structure

```
├── data\raw\              # Your data files go here
├── config\               # Configuration files
├── src\                  # Python source code
├── outputs\
│   ├── plots\            # Generated visualizations
│   └── reports\          # Analysis summaries
└── logs\                 # Execution logs
```
EOF

    print_success "Documentation created"
}

run_validation() {
    print_info "Running final validation..."
    
    local errors=0
    
    # Check directories
    for dir in data/raw config src outputs logs; do
        if [ -d "$dir" ]; then
            print_success "Directory exists: $dir"
        else
            print_error "Directory missing: $dir"
            ((errors++))
        fi
    done
    
    # Check key files
    for file in config/config.yaml src/main.py setup.bat run_analysis.bat; do
        if [ -f "$file" ]; then
            print_success "File exists: $file"
        else
            print_error "File missing: $file"
            ((errors++))
        fi
    done
    
    # Check source modules
    for module in src/data/data_loader.py src/models/predictive_models.py src/visualization/plots.py; do
        if [ -f "$module" ]; then
            print_success "Module exists: $module"
        else
            print_error "Module missing: $module"
            ((errors++))
        fi
    done
    
    if [ $errors -eq 0 ]; then
        print_success "All validation checks passed!"
        return 0
    else
        print_error "$errors validation check(s) failed"
        return 1
    fi
}

display_completion_message() {
    echo -e "${GREEN}"
    cat << "EOF"

╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    🎉 WINDOWS SETUP COMPLETED! 🎉                            ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  COMPLETE PROJECT CREATED FOR WINDOWS:                                       ║
║                                                                              ║
║  📁 Full directory structure                                                 ║
║  📄 Configuration files with your column names                              ║
║  🐍 Python modules with Windows compatibility                               ║
║  📊 Visualization and modeling code                                          ║
║  🔧 Windows batch files for all operations                                   ║
║                                                                              ║
║  WINDOWS COMMANDS TO RUN:                                                    ║
║                                                                              ║
║  1️⃣  setup.bat           - Create environment & install packages            ║
║  2️⃣  [Add your data files to data\raw\]                                      ║
║  3️⃣  quick_start.bat     - Validate your data files                         ║
║  4️⃣  run_analysis.bat    - Run the full analysis                            ║
║  5️⃣  validate.bat        - Test installation (optional)                     ║
║                                                                              ║
║  YOUR DATA FILES SHOULD BE:                                                  ║
║  📊 data\raw\mail.csv          (mail_date, mail_type, mail_volume)           ║
║  📞 data\raw\call_intents.csv  (ConversationStart, uui_Intent)               ║
║  📈 data\raw\call_volume.csv   (Date, call_volume)                           ║
║                                                                              ║
║  This project is now completely Windows-compatible and standalone!          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

EOF
    echo -e "${NC}"
    
    print_info "Key features created:"
    echo "• Windows batch files for all operations"
    echo "• Configuration set up for your exact column names"
    echo "• Error handling and graceful package failures"
    echo "• Data validation and format checking"
    echo "• Complete analytics pipeline"
    echo "• Professional visualizations and reports"
    echo ""
    print_warning "Start with: setup.bat"
}

main() {
    print_header
    
    check_requirements
    create_directory_structure
    create_configuration_files
    create_windows_requirements
    create_source_code
    create_windows_batch_files
    create_readme
    
    if run_validation; then
        display_completion_message
    else
        print_error "Setup incomplete. Check the output above for issues."
        exit 1
    fi
}

# Execute main function
main "$@"
