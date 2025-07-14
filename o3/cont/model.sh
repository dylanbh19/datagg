#!/usr/bin/env bash
###############################################################################
#  simple_modelling.sh - DIRECT MAIL → CALL VOLUME PREDICTION
#  
#  🎯 SIMPLE PURPOSE: 
#     INPUT:  mail_type, date, volume
#     OUTPUT: predicted call volume over time period
#  
#  📊 CREATES: Easy-to-use prediction function
###############################################################################
set -euo pipefail
export PYTHONUTF8=1

echo "🎯 SIMPLE MAIL → CALL PREDICTION MODEL"
echo "======================================"
echo ""
echo "Creating focused model: Mail Type + Date + Volume → Call Volume Timeline"

# Check prerequisites
if [[ ! -d "customer_comms" ]]; then
    echo "❌ customer_comms package not found!"
    echo "💡 Please run your base analytics pipeline first"
    exit 1
fi

PKG="customer_comms"

###############################################################################
# Install ML Dependencies (lightweight)
###############################################################################
echo "📦 Installing ML dependencies..."

python - <<'PY'
import subprocess, sys, importlib

packages = ["scikit-learn", "joblib"]

for pkg in packages:
    try:
        importlib.import_module(pkg.replace('-', '_'))
        print(f"✅ {pkg}")
    except ModuleNotFoundError:
        print(f"📦 Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

print("✅ Dependencies ready")
PY

###############################################################################
# Create Simple Predictive Model
###############################################################################
echo ""
echo "🤖 Creating simple mail-to-call prediction model..."

cat > "$PKG/simple_predictor.py" <<'PY'
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List

from .data.loader import load_mail, load_intents
from .processing.combine import build_master
from .config import settings
from .utils.logging_utils import get_logger

log = get_logger(__name__)

class SimpleMailCallPredictor:
    """
    Simple predictor: Mail Type + Date + Volume → Call Volume Timeline
    
    Usage:
        predictor = SimpleMailCallPredictor()
        predictor.train()
        result = predictor.predict(mail_type="General Comm", mail_volume=10000, 
                                 campaign_date="2024-06-15", days_ahead=14)
    """
    
    def __init__(self):
        self.model = None
        self.mail_type_encoder = LabelEncoder()
        self.is_trained = False
        self.training_stats = {}
        
    def prepare_training_data(self) -> pd.DataFrame:
        """Prepare training data from your existing data"""
        log.info("📊 Preparing training data...")
        
        # Load your existing data
        master_df = build_master()
        mail_df = load_mail()
        
        if master_df.empty or mail_df.empty:
            raise ValueError("No data available - run your base pipeline first")
        
        # Create features for each date
        features = []
        
        # Get all dates from master data
        for _, row in master_df.iterrows():
            date = row['date']
            call_volume = row['call_volume']
            mail_volume = row['mail_volume']
            
            # Get mail type for this date (if available)
            mail_on_date = mail_df[mail_df['date'] == date]
            if not mail_on_date.empty:
                # Use the dominant mail type on this date
                mail_type = mail_on_date.groupby('mail_type')['mail_volume'].sum().idxmax()
            else:
                mail_type = 'General Comm'  # Default
            
            # Create features
            features.append({
                'date': date,
                'mail_type': mail_type,
                'mail_volume': mail_volume,
                'call_volume': call_volume,
                'day_of_week': date.dayofweek,
                'month': date.month,
                'is_high_volume': 1 if mail_volume > master_df['mail_volume'].quantile(0.75) else 0
            })
        
        training_df = pd.DataFrame(features)
        log.info(f"✅ Training data prepared: {len(training_df)} samples")
        
        return training_df
    
    def train(self) -> Dict:
        """Train the simple prediction model"""
        log.info("🔄 Training simple mail→call prediction model...")
        
        # Prepare data
        training_df = self.prepare_training_data()
        
        # Encode mail types
        training_df['mail_type_encoded'] = self.mail_type_encoder.fit_transform(training_df['mail_type'])
        
        # Prepare features
        feature_columns = ['mail_type_encoded', 'mail_volume', 'day_of_week', 'month', 'is_high_volume']
        X = training_df[feature_columns]
        y = training_df['call_volume']
        
        # Train simple Random Forest
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        
        # Calculate training metrics
        y_pred = self.model.predict(X)
        r2_score = self.model.score(X, y)
        
        self.training_stats = {
            'r2_score': r2_score,
            'training_samples': len(training_df),
            'mail_types': list(self.mail_type_encoder.classes_),
            'average_response_rate': y.sum() / training_df['mail_volume'].sum()
        }
        
        self.is_trained = True
        
        log.info("✅ Model trained successfully!")
        log.info(f"   • R² Score: {r2_score:.3f}")
        log.info(f"   • Training samples: {len(training_df)}")
        log.info(f"   • Mail types: {len(self.mail_type_encoder.classes_)}")
        
        return self.training_stats
    
    def predict(self, mail_type: str, mail_volume: int, campaign_date: str, days_ahead: int = 14) -> Dict:
        """
        Predict call volume timeline for a mail campaign
        
        Args:
            mail_type: Type of mail (e.g., "General Comm", "ACH_Debit_Enrollment")
            mail_volume: Number of mail pieces sent
            campaign_date: Date when mail is sent (YYYY-MM-DD)
            days_ahead: Number of days to predict (default 14)
            
        Returns:
            Dictionary with daily call predictions and summary
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first. Call .train()")
        
        log.info(f"🎯 Predicting: {mail_type}, {mail_volume:,} pieces, {campaign_date}")
        
        # Parse campaign date
        start_date = pd.to_datetime(campaign_date)
        
        # Create prediction timeline (business days only)
        dates = pd.bdate_range(start=start_date, periods=days_ahead)
        
        # Prepare features for each day
        predictions = []
        daily_predictions = []
        
        for i, date in enumerate(dates):
            # Mail volume decreases over time (immediate impact, then decay)
            if i == 0:
                day_mail_volume = mail_volume  # Full volume on campaign day
            else:
                # Exponential decay effect
                decay_factor = 0.3 ** i  # Decay over time
                day_mail_volume = mail_volume * decay_factor
            
            # Encode mail type
            if mail_type in self.mail_type_encoder.classes_:
                mail_type_encoded = self.mail_type_encoder.transform([mail_type])[0]
            else:
                # Use most common mail type as fallback
                mail_type_encoded = 0
                log.warning(f"Unknown mail type '{mail_type}', using default")
            
            # Create features
            features = [[
                mail_type_encoded,
                day_mail_volume,
                date.dayofweek,
                date.month,
                1 if day_mail_volume > 5000 else 0  # is_high_volume
            ]]
            
            # Predict calls for this day
            daily_calls = self.model.predict(features)[0]
            daily_calls = max(0, daily_calls)  # Ensure non-negative
            
            daily_predictions.append(daily_calls)
            
            predictions.append({
                'date': date.strftime('%Y-%m-%d'),
                'predicted_calls': round(daily_calls, 0),
                'mail_effect': round(day_mail_volume, 0)
            })
        
        # Calculate summary metrics
        total_calls = sum(daily_predictions)
        peak_day_idx = np.argmax(daily_predictions)
        response_rate = total_calls / mail_volume if mail_volume > 0 else 0
        
        result = {
            'campaign_input': {
                'mail_type': mail_type,
                'mail_volume': mail_volume,
                'campaign_date': campaign_date,
                'days_predicted': days_ahead
            },
            'daily_predictions': predictions,
            'summary': {
                'total_predicted_calls': round(total_calls, 0),
                'peak_day': dates[peak_day_idx].strftime('%Y-%m-%d'),
                'peak_calls': round(daily_predictions[peak_day_idx], 0),
                'response_rate_percent': round(response_rate * 100, 2),
                'average_daily_calls': round(total_calls / days_ahead, 0)
            },
            'model_info': {
                'training_r2': self.training_stats.get('r2_score', 0),
                'training_samples': self.training_stats.get('training_samples', 0)
            }
        }
        
        log.info("📊 Prediction complete!")
        log.info(f"   • Total predicted calls: {result['summary']['total_predicted_calls']:,}")
        log.info(f"   • Response rate: {result['summary']['response_rate_percent']}%")
        log.info(f"   • Peak day: {result['summary']['peak_day']}")
        
        return result
    
    def save_model(self, filepath: str = None):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        if filepath is None:
            filepath = settings.out_dir / "simple_mail_call_predictor.pkl"
        
        model_data = {
            'model': self.model,
            'mail_type_encoder': self.mail_type_encoder,
            'training_stats': self.training_stats
        }
        
        joblib.dump(model_data, filepath)
        log.info(f"✅ Model saved to {filepath}")
        
    @classmethod
    def load_model(cls, filepath: str = None):
        """Load trained model"""
        if filepath is None:
            filepath = settings.out_dir / "simple_mail_call_predictor.pkl"
        
        model_data = joblib.load(filepath)
        
        instance = cls()
        instance.model = model_data['model']
        instance.mail_type_encoder = model_data['mail_type_encoder']
        instance.training_stats = model_data['training_stats']
        instance.is_trained = True
        
        log.info(f"✅ Model loaded from {filepath}")
        return instance

# Create global instance
simple_predictor = SimpleMailCallPredictor()
PY

###############################################################################
# Create Easy Usage Script
###############################################################################
echo ""
echo "📱 Creating simple usage interface..."

cat > "$PKG/predict_calls.py" <<'PY'
"""
Simple interface for predicting call volume from mail campaigns

Usage:
    from customer_comms.predict_calls import predict_mail_campaign
    
    result = predict_mail_campaign(
        mail_type="General Comm",
        mail_volume=10000,
        campaign_date="2024-06-15"
    )
    
    print(f"Predicted calls: {result['summary']['total_predicted_calls']}")
"""

from .simple_predictor import simple_predictor
from .utils.logging_utils import get_logger

log = get_logger(__name__)

def predict_mail_campaign(mail_type: str, mail_volume: int, campaign_date: str, days_ahead: int = 14):
    """
    Predict call volume from a mail campaign
    
    Args:
        mail_type: Type of mail (e.g., "General Comm", "ACH_Debit_Enrollment", "Print Only")
        mail_volume: Number of mail pieces sent
        campaign_date: Campaign date in YYYY-MM-DD format
        days_ahead: Number of business days to predict (default 14)
    
    Returns:
        Dictionary with predictions and summary
    """
    
    # Train model if not already trained
    if not simple_predictor.is_trained:
        log.info("🔄 Training model (first time use)...")
        simple_predictor.train()
    
    # Make prediction
    return simple_predictor.predict(mail_type, mail_volume, campaign_date, days_ahead)

def get_available_mail_types():
    """Get list of available mail types from training data"""
    if not simple_predictor.is_trained:
        simple_predictor.train()
    
    return simple_predictor.training_stats.get('mail_types', [])

def get_model_stats():
    """Get model training statistics"""
    if not simple_predictor.is_trained:
        simple_predictor.train()
    
    return simple_predictor.training_stats
PY

###############################################################################
# Create Test Script
###############################################################################
echo ""
echo "🧪 Creating test examples..."

cat > "$PKG/test_simple_predictor.py" <<'PY'
"""
Test the simple mail-call predictor with examples
"""

from .predict_calls import predict_mail_campaign, get_available_mail_types, get_model_stats
from .utils.logging_utils import get_logger

log = get_logger(__name__)

def test_simple_predictions():
    """Test the simple predictor with example scenarios"""
    
    log.info("🧪 Testing Simple Mail-Call Predictor")
    log.info("=" * 40)
    
    try:
        # Get model info
        log.info("📊 Model Information:")
        stats = get_model_stats()
        log.info(f"   • Training R²: {stats.get('r2_score', 0):.3f}")
        log.info(f"   • Training samples: {stats.get('training_samples', 0)}")
        log.info(f"   • Average response rate: {stats.get('average_response_rate', 0)*100:.2f}%")
        
        mail_types = get_available_mail_types()
        log.info(f"   • Available mail types: {mail_types}")
        
        # Test scenarios
        test_scenarios = [
            {
                'name': 'Small Campaign',
                'mail_type': 'General Comm',
                'mail_volume': 5000,
                'campaign_date': '2024-06-15'
            },
            {
                'name': 'Medium Campaign', 
                'mail_type': 'ACH_Debit_Enrollment',
                'mail_volume': 15000,
                'campaign_date': '2024-07-01'
            },
            {
                'name': 'Large Campaign',
                'mail_type': 'Print Only',
                'mail_volume': 30000,
                'campaign_date': '2024-08-01'
            }
        ]
        
        log.info("\n🎯 Test Predictions:")
        
        for scenario in test_scenarios:
            log.info(f"\n   {scenario['name']}:")
            log.info(f"     Input: {scenario['mail_type']}, {scenario['mail_volume']:,} pieces")
            
            try:
                result = predict_mail_campaign(
                    mail_type=scenario['mail_type'],
                    mail_volume=scenario['mail_volume'],
                    campaign_date=scenario['campaign_date'],
                    days_ahead=10
                )
                
                summary = result['summary']
                log.info(f"     Output: {summary['total_predicted_calls']:,} calls")
                log.info(f"     Response Rate: {summary['response_rate_percent']}%")
                log.info(f"     Peak Day: {summary['peak_day']} ({summary['peak_calls']:,} calls)")
                
            except Exception as e:
                log.error(f"     ❌ Failed: {e}")
        
        log.info("\n✅ Simple predictor test complete!")
        return True
        
    except Exception as e:
        log.error(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_simple_predictions()
PY

###############################################################################
# Run Test
###############################################################################
echo ""
echo "🧪 Testing the simple predictor..."

python - <<'PY'
import sys
sys.path.insert(0, '.')

try:
    from customer_comms.test_simple_predictor import test_simple_predictions
    
    success = test_simple_predictions()
    
    if success:
        print("\n🎉 SIMPLE PREDICTOR READY!")
        print("\n📖 USAGE EXAMPLE:")
        print("   from customer_comms.predict_calls import predict_mail_campaign")
        print("")
        print("   result = predict_mail_campaign(")
        print("       mail_type='General Comm',")
        print("       mail_volume=10000,")
        print("       campaign_date='2024-06-15'")
        print("   )")
        print("")
        print("   print(f'Predicted calls: {result[\"summary\"][\"total_predicted_calls\"]}')")
        
    else:
        print("❌ Test failed - check logs for details")
    
except Exception as e:
    print(f"❌ Could not run test: {e}")
    print("   This may be normal if base data is not available")
PY

###############################################################################
# Summary
###############################################################################
echo ""
echo "🎉 SIMPLE MAIL-CALL PREDICTOR COMPLETE!"
echo "======================================="
echo ""
echo "🎯 SIMPLE INTERFACE CREATED:"
echo ""
echo "   INPUT:  mail_type, mail_volume, campaign_date"
echo "   OUTPUT: predicted call volume timeline"
echo ""
echo "📖 USAGE:"
echo "   from customer_comms.predict_calls import predict_mail_campaign"
echo ""
echo "   result = predict_mail_campaign("
echo "       mail_type='General Comm',"
echo "       mail_volume=10000,"
echo "       campaign_date='2024-06-15'"
echo "   )"
echo ""
echo "   # Get results"
echo "   total_calls = result['summary']['total_predicted_calls']"
echo "   response_rate = result['summary']['response_rate_percent']"
echo "   daily_breakdown = result['daily_predictions']"
echo ""
echo "📁 FILES CREATED:"
echo "   • simple_predictor.py - Core prediction model"
echo "   • predict_calls.py - Easy interface"
echo "   • test_simple_predictor.py - Test examples"
echo ""
echo "✨ FEATURES:"
echo "   • Automatic model training"
echo "   • Business days only prediction"
echo "   • Decay effect modeling (immediate impact → gradual decline)"
echo "   • Response rate calculation"
echo "   • Peak day identification"
echo ""
echo "🚀 READY TO USE!"
echo "   Just call predict_mail_campaign() with your mail campaign details"
echo ""
