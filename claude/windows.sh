#!/bin/bash

# ================================================================================
# WINDOWS COMPATIBILITY PATCH SCRIPT (Bash Version)
# ================================================================================
# This script converts the entire analytics project to be Windows-compatible
# Run this AFTER setup_analytics_project.sh and patch_mail_data_logic.sh
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
║                    WINDOWS COMPATIBILITY PATCH                              ║
║                                                                              ║
║  Converting analytics project for Windows compatibility                      ║
║  • Virtual environment activation                                            ║
║  • Script file extensions                                                    ║
║  • Path separators                                                           ║
║  • Error handling                                                            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

check_project_structure() {
    print_info "Checking project structure..."
    
    if [ ! -f "config/config.yaml" ]; then
        print_error "config/config.yaml not found. Please run setup_analytics_project.sh first."
        exit 1
    fi
    
    if [ ! -f "src/main.py" ]; then
        print_error "src/main.py not found. Please run setup_analytics_project.sh first."
        exit 1
    fi
    
    print_success "Project structure validated"
}

fix_virtual_environment() {
    print_info "Fixing virtual environment activation..."
    
    # Update setup script to use Scripts instead of bin on Windows
    if [ -f "setup_analytics_project.sh" ]; then
        cp setup_analytics_project.sh setup_analytics_project.sh.backup
        
        # Replace bin/activate with Scripts/activate
        sed -i 's|analytics_env/bin/activate|analytics_env/Scripts/activate|g' setup_analytics_project.sh
        sed -i 's|source "$VENV_NAME/bin/activate"|. "$VENV_NAME/Scripts/activate"|g' setup_analytics_project.sh
        
        print_success "Setup script updated for Windows venv paths"
    fi
    
    # Update any existing run scripts
    if [ -f "run_analysis.sh" ]; then
        cp run_analysis.sh run_analysis.sh.backup
        sed -i 's|analytics_env/bin/activate|analytics_env/Scripts/activate|g' run_analysis.sh
        print_success "Run analysis script updated for Windows"
    fi
    
    if [ -f "quick_start.sh" ]; then
        cp quick_start.sh quick_start.sh.backup
        sed -i 's|analytics_env/bin/activate|analytics_env/Scripts/activate|g' quick_start.sh
        print_success "Quick start script updated for Windows"
    fi
}

create_windows_batch_files() {
    print_info "Creating Windows batch file equivalents..."
    
    # Create run_analysis.bat
    cat > run_analysis.bat << 'EOF'
@echo off
setlocal enabledelayedexpansion

echo ========================================================================
echo               CUSTOMER COMMUNICATION ANALYTICS
echo                      EXECUTION STARTING
echo ========================================================================

REM Check if virtual environment exists
if not exist "analytics_env" (
    echo Virtual environment not found. Please run setup first.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call analytics_env\Scripts\activate.bat

REM Check if data files exist
echo Checking data files...
if not exist "data\raw\mail.csv" (
    if not exist "data\raw\sample_mail_data.csv" (
        echo No mail data found. Please add your data files.
        echo Expected files:
        echo   - data\raw\mail.csv ^(mail_date, mail_type, mail_volume^)
        echo   - data\raw\call_intents.csv ^(ConversationStart, uui_Intent^)
        echo   - data\raw\call_volume.csv ^(Date, call_volume^)
        pause
        exit /b 1
    )
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
    echo   - outputs\plots\     : All generated visualizations
    echo   - outputs\reports\   : Analysis summary and insights
    echo   - outputs\models\    : Trained model objects
    echo   - logs\             : Execution logs
    echo ========================================================================
) else (
    echo.
    echo ========================================================================
    echo                    ANALYSIS FAILED
    echo.
    echo Check logs\analytics.log for detailed error information
    echo ========================================================================
)

pause
EOF

    chmod +x run_analysis.bat
    print_success "run_analysis.bat created"
    
    # Create quick_start.bat
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
    echo.
    echo File requirements:
    echo   mail.csv: mail_date, mail_type, mail_volume
    echo   call_intents.csv: ConversationStart, uui_Intent  
    echo   call_volume.csv: Date, call_volume
    pause
    exit /b 1
)

echo.
echo ✅ All data files found! Running analysis...
echo.

call run_analysis.bat
EOF

    chmod +x quick_start.bat
    print_success "quick_start.bat created"
    
    # Create setup.bat for initial setup
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
        echo Python not found. Please install Python from python.org
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
pip install pandas numpy scipy matplotlib seaborn

echo Installing machine learning packages...
pip install scikit-learn xgboost

echo Installing additional packages...
pip install statsmodels plotly openpyxl pyyaml tqdm

REM Try optional packages
echo Installing optional packages...
pip install prophet 2>nul || echo Prophet installation failed - will skip Prophet models
pip install pmdarima 2>nul || echo pmdarima installation failed - will use basic ARIMA
pip install yfinance fredapi 2>nul || echo Economic data packages failed - will skip economic analysis

echo.
echo ========================================================================
echo                    SETUP COMPLETED SUCCESSFULLY!
echo.
echo Next steps:
echo 1. Add your data files to data\raw\ directory
echo 2. Run: quick_start.bat ^(to validate data^)
echo 3. Run: run_analysis.bat ^(to run analysis^)
echo ========================================================================

pause
EOF

    chmod +x setup.bat
    print_success "setup.bat created"
}

create_windows_requirements() {
    print_info "Creating Windows-optimized requirements..."
    
    cat > requirements_windows.txt << 'EOF'
# Windows-optimized requirements file
# Core data processing
pandas>=1.5.0
numpy>=1.21.0
scipy>=1.9.0

# Machine learning
scikit-learn>=1.1.0
xgboost>=1.6.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.10.0

# Time series analysis
statsmodels>=0.13.0

# Data manipulation
openpyxl>=3.0.0

# Utilities
tqdm>=4.64.0
pyyaml>=6.0

# Optional packages (may fail on some Windows systems)
# prophet>=1.1.0
# pmdarima>=2.0.0
# yfinance>=0.1.70
# fredapi>=0.4.3
EOF

    print_success "Windows requirements file created"
}

fix_python_scripts() {
    print_info "Fixing Python scripts for Windows compatibility..."
    
    # Create Windows compatibility module
    cat > src/windows_compat.py << 'EOF'
"""
Windows compatibility fixes for the analytics project
"""
import os
import sys
import warnings
from pathlib import Path

# Suppress common Windows warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

def fix_path(path_str):
    """Convert Unix-style paths to Windows-compatible paths"""
    return str(Path(path_str))

def ensure_dir_exists(dir_path):
    """Ensure directory exists, create if not"""
    Path(dir_path).mkdir(parents=True, exist_ok=True)

def check_package_availability():
    """Check which packages are available and return compatibility info"""
    packages = {
        'pandas': False,
        'numpy': False,
        'matplotlib': False,
        'seaborn': False,
        'sklearn': False,
        'xgboost': False,
        'statsmodels': False,
        'plotly': False,
        'prophet': False,
        'yfinance': False,
        'fredapi': False
    }
    
    for package in packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            packages[package] = True
        except ImportError:
            pass
    
    return packages

def get_available_models():
    """Return list of available model types based on installed packages"""
    packages = check_package_availability()
    models = []
    
    if packages['sklearn']:
        models.extend(['RandomForest', 'LinearRegression'])
    
    if packages['xgboost']:
        models.append('XGBoost')
    
    if packages['statsmodels']:
        models.append('ARIMA')
    
    if packages['prophet']:
        models.append('Prophet')
    
    return models

def safe_import(package_name, fallback=None):
    """Safely import a package with fallback"""
    try:
        return __import__(package_name)
    except ImportError:
        print(f"Warning: {package_name} not available, using fallback")
        return fallback

# Set up proper paths for Windows
if os.name == 'nt':  # Windows
    os.environ['PYTHONIOENCODING'] = 'utf-8'
EOF

    print_success "Windows compatibility module created"
    
    # Update main.py to use Windows compatibility
    if [ -f "src/main.py" ]; then
        # Backup original
        cp src/main.py src/main.py.backup
        
        # Add Windows compatibility imports at the top
        cat > temp_main.py << 'EOF'
"""
Main Execution Script for Customer Communication Analytics
Windows-compatible version
"""
import os
import sys
import warnings

# Add src to path and import Windows compatibility
sys.path.append('src')
try:
    from windows_compat import *
    print("Windows compatibility module loaded")
except ImportError:
    print("Warning: Windows compatibility module not found")
    def fix_path(p): return p
    def ensure_dir_exists(p): os.makedirs(p, exist_ok=True)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

EOF
        
        # Append the rest of the original main.py (skip the original imports)
        tail -n +10 src/main.py >> temp_main.py
        mv temp_main.py src/main.py
        
        print_success "main.py updated for Windows compatibility"
    fi
    
    # Update data loader for Windows paths
    if [ -f "src/data/data_loader.py" ]; then
        cp src/data/data_loader.py src/data/data_loader.py.backup
        
        # Add Windows path handling
        sed -i '1i import os\nfrom pathlib import Path' src/data/data_loader.py
        sed -i 's|file_path|Path(file_path)|g' src/data/data_loader.py
        
        print_success "data_loader.py updated for Windows paths"
    fi
}

fix_configuration() {
    print_info "Updating configuration for Windows..."
    
    if [ -f "config/config.yaml" ]; then
        cp config/config.yaml config/config.yaml.backup
        
        # Update paths to use forward slashes (Python handles conversion)
        sed -i 's|\\|/|g' config/config.yaml
        
        # Update log file path
        sed -i 's|logs/analytics.log|logs\\analytics.log|g' config/config.yaml
        
        print_success "Configuration updated for Windows"
    fi
}

create_error_handling() {
    print_info "Adding comprehensive error handling..."
    
    # Create error handling module
    cat > src/error_handler.py << 'EOF'
"""
Comprehensive error handling for Windows environment
"""
import sys
import traceback
import logging
from pathlib import Path

def setup_error_logging():
    """Setup error logging for Windows"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'analytics.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def handle_package_import_error(package_name, error):
    """Handle package import errors gracefully"""
    logger = logging.getLogger(__name__)
    logger.warning(f"Package {package_name} not available: {error}")
    logger.info(f"Some features may be limited without {package_name}")
    
    # Provide specific guidance for common packages
    guidance = {
        'prophet': "Prophet models will be skipped. Install with: pip install prophet",
        'xgboost': "XGBoost models will be skipped. Install with: pip install xgboost", 
        'yfinance': "Economic data download may fail. Install with: pip install yfinance",
        'fredapi': "FRED economic data will be skipped. Install with: pip install fredapi"
    }
    
    if package_name in guidance:
        logger.info(guidance[package_name])

def handle_data_file_error(file_path, error):
    """Handle data file errors with helpful messages"""
    logger = logging.getLogger(__name__)
    logger.error(f"Data file error for {file_path}: {error}")
    
    suggestions = [
        f"1. Check if file exists: {file_path}",
        "2. Verify file format matches expected columns",
        "3. Ensure file is not corrupted or locked",
        "4. Check file permissions",
        "5. See config/your_data_format.yaml for format requirements"
    ]
    
    for suggestion in suggestions:
        logger.info(suggestion)

def safe_execute(func, *args, **kwargs):
    """Execute function with comprehensive error handling"""
    logger = logging.getLogger(__name__)
    
    try:
        return func(*args, **kwargs)
    except FileNotFoundError as e:
        handle_data_file_error(str(e), e)
        return None
    except ImportError as e:
        package_name = str(e).split("'")[1] if "'" in str(e) else "unknown"
        handle_package_import_error(package_name, e)
        return None
    except Exception as e:
        logger.error(f"Unexpected error in {func.__name__}: {e}")
        logger.debug(traceback.format_exc())
        return None
EOF

    print_success "Error handling module created"
}

update_readme() {
    print_info "Updating README for Windows..."
    
    if [ -f "README.md" ]; then
        cp README.md README.md.backup
        
        # Add Windows-specific instructions
        cat >> README.md << 'EOF'

## Windows-Specific Instructions

### Quick Setup for Windows

1. **Run the setup batch file:**
   ```cmd
   setup.bat
   ```

2. **Add your data files to `data\raw\` directory:**
   - `mail.csv` with columns: `mail_date`, `mail_type`, `mail_volume`
   - `call_intents.csv` with columns: `ConversationStart`, `uui_Intent`
   - `call_volume.csv` with columns: `Date`, `call_volume`

3. **Validate your data:**
   ```cmd
   quick_start.bat
   ```

4. **Run the analysis:**
   ```cmd
   run_analysis.bat
   ```

### Windows Files Created

- `setup.bat` - Initial environment setup
- `run_analysis.bat` - Run the full analysis
- `quick_start.bat` - Validate data and quick start
- `requirements_windows.txt` - Windows-optimized package list

### Troubleshooting Windows Issues

**Virtual Environment Issues:**
- Use `setup.bat` instead of bash scripts
- Ensure Python is installed and in PATH

**Package Installation Issues:**
- Some packages (like Prophet) may fail on Windows
- The system will continue with available packages
- Install Visual C++ Build Tools if needed

**Path Issues:**
- Use backslashes in Windows: `data\raw\file.csv`
- Or use forward slashes - Python converts automatically

**DLL Errors:**
- Install Microsoft Visual C++ Redistributables
- Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe

EOF

        print_success "README updated with Windows instructions"
    fi
}

create_validation_script() {
    print_info "Creating Windows validation script..."
    
    cat > validate_windows.bat << 'EOF'
@echo off
echo Validating Windows setup...

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
) else (
    echo ❌ Virtual environment not found
    echo Run setup.bat first
    exit /b 1
)

REM Activate and test imports
call analytics_env\Scripts\activate.bat

echo Testing package imports...
python -c "import pandas; print('✅ Pandas')" 2>nul || echo "❌ Pandas"
python -c "import numpy; print('✅ NumPy')" 2>nul || echo "❌ NumPy" 
python -c "import matplotlib; print('✅ Matplotlib')" 2>nul || echo "❌ Matplotlib"
python -c "import sklearn; print('✅ Scikit-learn')" 2>nul || echo "❌ Scikit-learn"

echo.
echo Validation complete!
pause
EOF

    chmod +x validate_windows.bat
    print_success "Windows validation script created"
}

run_final_validation() {
    print_info "Running final validation..."
    
    local errors=0
    
    # Check that Windows batch files were created
    for file in run_analysis.bat quick_start.bat setup.bat validate_windows.bat; do
        if [ -f "$file" ]; then
            print_success "$file created"
        else
            print_error "$file not created"
            ((errors++))
        fi
    done
    
    # Check Windows compatibility module
    if [ -f "src/windows_compat.py" ]; then
        print_success "Windows compatibility module created"
    else
        print_error "Windows compatibility module missing"
        ((errors++))
    fi
    
    # Check requirements
    if [ -f "requirements_windows.txt" ]; then
        print_success "Windows requirements file created"
    else
        print_error "Windows requirements file missing"
        ((errors++))
    fi
    
    if [ $errors -eq 0 ]; then
        print_success "All Windows compatibility patches applied successfully"
        return 0
    else
        print_error "$errors Windows patch(es) failed"
        return 1
    fi
}

display_completion_message() {
    echo -e "${GREEN}"
    cat << "EOF"

╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    🎉 WINDOWS COMPATIBILITY COMPLETE! 🎉                     ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  WINDOWS FILES CREATED:                                                      ║
║                                                                              ║
║  📁 setup.bat               - Initial environment setup                     ║
║  📁 run_analysis.bat        - Run full analysis                             ║
║  📁 quick_start.bat         - Validate data and quick start                 ║
║  📁 validate_windows.bat    - Test installation                             ║
║  📁 requirements_windows.txt - Windows-optimized packages                   ║
║                                                                              ║
║  PYTHON MODULES UPDATED:                                                     ║
║                                                                              ║
║  📄 src/windows_compat.py   - Windows compatibility helpers                 ║
║  📄 src/error_handler.py    - Comprehensive error handling                  ║
║  📄 src/main.py             - Updated with Windows support                  ║
║                                                                              ║
║  NEXT STEPS FOR WINDOWS:                                                     ║
║                                                                              ║
║  1. Run: setup.bat          (creates environment & installs packages)       ║
║  2. Add your data files to data\raw\ directory                              ║
║  3. Run: quick_start.bat    (validates your data)                           ║
║  4. Run: run_analysis.bat   (runs full analysis)                            ║
║                                                                              ║
║  All scripts now have proper Windows error handling and fallbacks!          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

EOF
    echo -e "${NC}"
    
    print_info "Windows-specific features:"
    echo "• Virtual environment uses Scripts/activate.bat"
    echo "• Batch files for all operations"  
    echo "• Windows path handling"
    echo "• Comprehensive error messages"
    echo "• Package availability checking"
    echo "• Graceful fallbacks for missing packages"
    echo ""
    print_warning "Use .bat files on Windows, .sh files on Linux/Mac"
}

main() {
    print_header
    
    check_project_structure
    fix_virtual_environment
    create_windows_batch_files
    create_windows_requirements
    fix_python_scripts
    fix_configuration
    create_error_handling
    update_readme
    create_validation_script
    
    if run_final_validation; then
        display_completion_message
    else
        print_error "Some Windows patches failed. Check the output above."
        exit 1
    fi
}

# Execute main function
main "$@"
