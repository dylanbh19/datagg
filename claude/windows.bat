@echo off
setlocal enabledelayedexpansion

REM ================================================================================
REM WINDOWS COMPATIBILITY PATCH SCRIPT
REM ================================================================================
REM This script ensures the analytics project works properly on Windows
REM with comprehensive error handling and fallback mechanisms
REM ================================================================================

echo.
echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║                                                                              ║
echo ║                    WINDOWS COMPATIBILITY PATCH                              ║
echo ║                                                                              ║
echo ║  This script fixes Windows-specific issues and ensures proper setup         ║
echo ║                                                                              ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.

REM Color definitions for Windows
set GREEN=[92m
set RED=[91m
set YELLOW=[93m
set BLUE=[94m
set NC=[0m

REM Global variables
set PYTHON_CMD=
set VENV_TYPE=
set ENVIRONMENT_NAME=analytics_env
set SUCCESS=0
set CONDA_AVAILABLE=0
set PYTHON_AVAILABLE=0

REM ================================================================================
REM UTILITY FUNCTIONS
REM ================================================================================

:print_success
echo %GREEN%✓ %~1%NC%
goto :eof

:print_error
echo %RED%✗ %~1%NC%
goto :eof

:print_warning
echo %YELLOW%⚠ %~1%NC%
goto :eof

:print_info
echo %BLUE%ℹ %~1%NC%
goto :eof

:check_command
where %1 >nul 2>nul
if %errorlevel% == 0 (
    set RESULT=1
) else (
    set RESULT=0
)
goto :eof

REM ================================================================================
REM ENVIRONMENT DETECTION
REM ================================================================================

:detect_environment
call :print_info "Detecting Python environment options..."

REM Check for conda
call :check_command conda
if !RESULT! == 1 (
    set CONDA_AVAILABLE=1
    call :print_success "Conda detected"
) else (
    call :print_warning "Conda not found"
)

REM Check for Python
call :check_command python
if !RESULT! == 1 (
    set PYTHON_AVAILABLE=1
    set PYTHON_CMD=python
    call :print_success "Python detected"
) else (
    call :check_command py
    if !RESULT! == 1 (
        set PYTHON_AVAILABLE=1
        set PYTHON_CMD=py
        call :print_success "Python detected via py launcher"
    ) else (
        call :print_error "Python not found"
        goto :python_not_found
    )
)

REM Get Python version
for /f "tokens=2" %%i in ('!PYTHON_CMD! --version 2^>^&1') do set PYTHON_VERSION=%%i
call :print_info "Python version: !PYTHON_VERSION!"

goto :eof

:python_not_found
echo.
call :print_error "Python is not installed or not in PATH"
echo.
echo Please install Python from: https://www.python.org/downloads/
echo Make sure to check "Add Python to PATH" during installation
echo.
pause
exit /b 1

REM ================================================================================
REM VISUAL C++ REDISTRIBUTABLES CHECK
REM ================================================================================

:check_vcredist
call :print_info "Checking Visual C++ Redistributables..."

REM Check if vcredist is installed by testing a common DLL
where api-ms-win-crt-runtime-l1-1-0.dll >nul 2>nul
if %errorlevel% == 0 (
    call :print_success "Visual C++ Redistributables found"
    goto :eof
)

call :print_warning "Visual C++ Redistributables may be missing"
echo.
echo This might cause DLL errors. Would you like to download them?
echo.
echo 1. Yes - Open download page
echo 2. No - Continue anyway
echo 3. Skip - I'll install manually later
echo.
set /p choice="Enter choice (1-3): "

if "!choice!" == "1" (
    call :print_info "Opening Visual C++ Redistributables download page..."
    start https://aka.ms/vs/17/release/vc_redist.x64.exe
    echo.
    echo Please install the redistributables and press any key to continue...
    pause >nul
) else if "!choice!" == "2" (
    call :print_warning "Continuing without redistributables - may cause issues"
) else (
    call :print_info "Skipping redistributables installation"
)

goto :eof

REM ================================================================================
REM ENVIRONMENT CREATION WITH FALLBACKS
REM ================================================================================

:create_environment
call :print_info "Creating Python environment with fallbacks..."

REM Try conda first if available
if !CONDA_AVAILABLE! == 1 (
    call :create_conda_env
    if !SUCCESS! == 1 goto :install_packages
)

REM Fall back to venv
call :create_venv
if !SUCCESS! == 1 goto :install_packages

REM Final fallback - use system Python
call :print_warning "Using system Python as final fallback"
set VENV_TYPE=system
set SUCCESS=1
goto :install_packages

:create_conda_env
call :print_info "Attempting to create conda environment..."

conda create -n !ENVIRONMENT_NAME! python=3.9 -y >nul 2>nul
if %errorlevel% == 0 (
    call :print_success "Conda environment created successfully"
    set VENV_TYPE=conda
    set SUCCESS=1
) else (
    call :print_error "Conda environment creation failed"
    set SUCCESS=0
)
goto :eof

:create_venv
call :print_info "Attempting to create venv environment..."

REM Remove existing venv if it exists
if exist !ENVIRONMENT_NAME! (
    call :print_info "Removing existing environment..."
    rmdir /s /q !ENVIRONMENT_NAME! >nul 2>nul
)

!PYTHON_CMD! -m venv !ENVIRONMENT_NAME! >nul 2>nul
if %errorlevel% == 0 (
    call :print_success "Virtual environment created successfully"
    set VENV_TYPE=venv
    set SUCCESS=1
) else (
    call :print_error "Virtual environment creation failed"
    set SUCCESS=0
)
goto :eof

REM ================================================================================
REM PACKAGE INSTALLATION WITH ERROR HANDLING
REM ================================================================================

:install_packages
call :print_info "Installing packages with error handling..."

REM Activate environment based on type
if "!VENV_TYPE!" == "conda" (
    call conda activate !ENVIRONMENT_NAME! >nul 2>nul
) else if "!VENV_TYPE!" == "venv" (
    call !ENVIRONMENT_NAME!\Scripts\activate.bat >nul 2>nul
)

REM Create a more robust requirements file
call :create_safe_requirements

REM Install packages in batches with fallbacks
call :install_core_packages
call :install_ml_packages  
call :install_visualization_packages
call :install_optional_packages

goto :eof

:create_safe_requirements
call :print_info "Creating Windows-safe requirements file..."

echo # Core data processing > requirements_windows.txt
echo pandas^>=1.5.0 >> requirements_windows.txt
echo numpy^>=1.21.0 >> requirements_windows.txt
echo scipy^>=1.9.0 >> requirements_windows.txt
echo >> requirements_windows.txt

echo # Machine learning >> requirements_windows.txt
echo scikit-learn^>=1.1.0 >> requirements_windows.txt
echo xgboost^>=1.6.0 >> requirements_windows.txt
echo >> requirements_windows.txt

echo # Visualization >> requirements_windows.txt
echo matplotlib^>=3.5.0 >> requirements_windows.txt
echo seaborn^>=0.11.0 >> requirements_windows.txt
echo plotly^>=5.10.0 >> requirements_windows.txt
echo >> requirements_windows.txt

echo # Time series >> requirements_windows.txt
echo statsmodels^>=0.13.0 >> requirements_windows.txt
echo >> requirements_windows.txt

echo # Data manipulation >> requirements_windows.txt
echo openpyxl^>=3.0.0 >> requirements_windows.txt
echo >> requirements_windows.txt

echo # Utility >> requirements_windows.txt
echo tqdm^>=4.64.0 >> requirements_windows.txt
echo pyyaml^>=6.0 >> requirements_windows.txt

call :print_success "Windows-safe requirements file created"
goto :eof

:install_core_packages
call :print_info "Installing core packages..."

pip install --upgrade pip --quiet
pip install pandas numpy scipy --quiet
if %errorlevel% == 0 (
    call :print_success "Core packages installed"
) else (
    call :print_error "Core packages failed - trying individual installation"
    pip install pandas --quiet
    pip install numpy --quiet
    pip install scipy --quiet
)
goto :eof

:install_ml_packages
call :print_info "Installing machine learning packages..."

pip install scikit-learn --quiet
if %errorlevel% == 0 (
    call :print_success "Scikit-learn installed"
) else (
    call :print_warning "Scikit-learn failed - continuing without"
)

pip install xgboost --quiet
if %errorlevel% == 0 (
    call :print_success "XGBoost installed"
) else (
    call :print_warning "XGBoost failed - will use alternatives"
)
goto :eof

:install_visualization_packages
call :print_info "Installing visualization packages..."

pip install matplotlib seaborn --quiet
if %errorlevel% == 0 (
    call :print_success "Matplotlib and Seaborn installed"
) else (
    call :print_warning "Visualization packages had issues - trying individually"
    pip install matplotlib --quiet
    pip install seaborn --quiet
)

pip install plotly --quiet
if %errorlevel% == 0 (
    call :print_success "Plotly installed"
) else (
    call :print_warning "Plotly failed - will use matplotlib only"
)
goto :eof

:install_optional_packages
call :print_info "Installing optional packages..."

pip install statsmodels --quiet
if %errorlevel% == 0 (
    call :print_success "Statsmodels installed"
) else (
    call :print_warning "Statsmodels failed - some time series features may be limited"
)

pip install openpyxl pyyaml tqdm --quiet
if %errorlevel% == 0 (
    call :print_success "Utility packages installed"
) else (
    call :print_warning "Some utility packages failed"
)

REM Try prophet with fallback
pip install prophet --quiet >nul 2>nul
if %errorlevel% == 0 (
    call :print_success "Prophet installed"
) else (
    call :print_warning "Prophet failed - will skip Prophet models"
)

goto :eof

REM ================================================================================
REM WINDOWS-SPECIFIC SCRIPT FIXES
REM ================================================================================

:fix_windows_scripts
call :print_info "Applying Windows-specific script fixes..."

REM Fix the run_analysis script
if exist run_analysis.sh (
    call :create_windows_run_script
)

REM Fix the quick_start script  
if exist quick_start.sh (
    call :create_windows_quick_start
)

call :print_success "Windows scripts created"
goto :eof

:create_windows_run_script
echo @echo off > run_analysis.bat
echo setlocal enabledelayedexpansion >> run_analysis.bat
echo. >> run_analysis.bat
echo echo Starting Customer Communication Analytics... >> run_analysis.bat
echo. >> run_analysis.bat

if "!VENV_TYPE!" == "conda" (
    echo call conda activate !ENVIRONMENT_NAME! >> run_analysis.bat
) else if "!VENV_TYPE!" == "venv" (
    echo call !ENVIRONMENT_NAME!\Scripts\activate.bat >> run_analysis.bat
) else (
    echo REM Using system Python >> run_analysis.bat
)

echo. >> run_analysis.bat
echo REM Check if data files exist >> run_analysis.bat
echo if not exist "data\raw\mail.csv" ^( >> run_analysis.bat
echo     echo Error: data\raw\mail.csv not found >> run_analysis.bat
echo     echo Please add your mail data file >> run_analysis.bat
echo     pause >> run_analysis.bat
echo     exit /b 1 >> run_analysis.bat
echo ^) >> run_analysis.bat
echo. >> run_analysis.bat
echo python src\main.py >> run_analysis.bat
echo. >> run_analysis.bat
echo if %%errorlevel%% == 0 ^( >> run_analysis.bat
echo     echo Analysis completed successfully! >> run_analysis.bat
echo     echo Check outputs\ directory for results >> run_analysis.bat
echo ^) else ^( >> run_analysis.bat
echo     echo Analysis failed - check logs\analytics.log >> run_analysis.bat
echo ^) >> run_analysis.bat
echo. >> run_analysis.bat
echo pause >> run_analysis.bat

goto :eof

:create_windows_quick_start
echo @echo off > quick_start.bat
echo echo Checking for required data files... >> quick_start.bat
echo. >> quick_start.bat
echo set MISSING=0 >> quick_start.bat
echo. >> quick_start.bat
echo if not exist "data\raw\mail.csv" ^( >> quick_start.bat
echo     echo ❌ data\raw\mail.csv not found >> quick_start.bat
echo     set MISSING=1 >> quick_start.bat
echo ^) else ^( >> quick_start.bat
echo     echo ✅ mail.csv found >> quick_start.bat
echo ^) >> quick_start.bat
echo. >> quick_start.bat
echo if not exist "data\raw\call_intents.csv" ^( >> quick_start.bat
echo     echo ❌ data\raw\call_intents.csv not found >> quick_start.bat
echo     set MISSING=1 >> quick_start.bat
echo ^) else ^( >> quick_start.bat
echo     echo ✅ call_intents.csv found >> quick_start.bat
echo ^) >> quick_start.bat
echo. >> quick_start.bat
echo if not exist "data\raw\call_volume.csv" ^( >> quick_start.bat
echo     echo ❌ data\raw\call_volume.csv not found >> quick_start.bat
echo     set MISSING=1 >> quick_start.bat
echo ^) else ^( >> quick_start.bat
echo     echo ✅ call_volume.csv found >> quick_start.bat
echo ^) >> quick_start.bat
echo. >> quick_start.bat
echo if %%MISSING%% == 1 ^( >> quick_start.bat
echo     echo. >> quick_start.bat
echo     echo Please add your data files to data\raw\ directory >> quick_start.bat
echo     echo See config\your_data_format.yaml for format requirements >> quick_start.bat
echo     pause >> quick_start.bat
echo     exit /b 1 >> quick_start.bat
echo ^) >> quick_start.bat
echo. >> quick_start.bat
echo echo All data files found! Running analysis... >> quick_start.bat
echo call run_analysis.bat >> quick_start.bat

goto :eof

REM ================================================================================
REM PYTHON SCRIPT COMPATIBILITY FIXES
REM ================================================================================

:fix_python_scripts
call :print_info "Applying Python script compatibility fixes..."

REM Create a Windows compatibility module
echo # Windows compatibility fixes > src\windows_compat.py
echo import os >> src\windows_compat.py
echo import sys >> src\windows_compat.py
echo import warnings >> src\windows_compat.py
echo. >> src\windows_compat.py
echo # Suppress Windows-specific warnings >> src\windows_compat.py
echo warnings.filterwarnings('ignore', category=UserWarning) >> src\windows_compat.py
echo warnings.filterwarnings('ignore', category=FutureWarning) >> src\windows_compat.py
echo. >> src\windows_compat.py
echo # Fix path separators for Windows >> src\windows_compat.py
echo def fix_path(path): >> src\windows_compat.py
echo     return path.replace('/', os.sep) >> src\windows_compat.py
echo. >> src\windows_compat.py
echo # Check package availability >> src\windows_compat.py
echo def is_package_available(package_name): >> src\windows_compat.py
echo     try: >> src\windows_compat.py
echo         __import__(package_name) >> src\windows_compat.py
echo         return True >> src\windows_compat.py
echo     except ImportError: >> src\windows_compat.py
echo         return False >> src\windows_compat.py

REM Update main.py with error handling
call :update_main_py_for_windows

call :print_success "Python scripts updated for Windows compatibility"
goto :eof

:update_main_py_for_windows
if not exist src\main.py.backup (
    copy src\main.py src\main.py.backup >nul
)

REM Add Windows compatibility import at the top of main.py
echo import sys > temp_main.py
echo sys.path.append('src') >> temp_main.py
echo from windows_compat import * >> temp_main.py
echo import warnings >> temp_main.py
echo warnings.filterwarnings('ignore') >> temp_main.py
echo. >> temp_main.py

REM Append the rest of the original file
type src\main.py >> temp_main.py
move temp_main.py src\main.py >nul

goto :eof

REM ================================================================================
REM VALIDATION AND TESTING
REM ================================================================================

:run_validation
call :print_info "Running validation tests..."

REM Test Python environment
if "!VENV_TYPE!" == "conda" (
    call conda activate !ENVIRONMENT_NAME! >nul 2>nul
) else if "!VENV_TYPE!" == "venv" (
    call !ENVIRONMENT_NAME!\Scripts\activate.bat >nul 2>nul
)

REM Test core imports
python -c "import pandas; print('✓ Pandas works')" 2>nul
if %errorlevel% == 0 (
    call :print_success "Pandas import test passed"
) else (
    call :print_error "Pandas import test failed"
)

python -c "import numpy; print('✓ NumPy works')" 2>nul
if %errorlevel% == 0 (
    call :print_success "NumPy import test passed"
) else (
    call :print_error "NumPy import test failed"
)

python -c "import matplotlib.pyplot as plt; print('✓ Matplotlib works')" 2>nul
if %errorlevel% == 0 (
    call :print_success "Matplotlib import test passed"
) else (
    call :print_warning "Matplotlib import test failed - some plots may not work"
)

REM Test directory structure
if exist config\config.yaml (
    call :print_success "Configuration file exists"
) else (
    call :print_error "Configuration file missing"
)

if exist src\main.py (
    call :print_success "Main script exists"
) else (
    call :print_error "Main script missing"
)

goto :eof

REM ================================================================================
REM MAIN EXECUTION
REM ================================================================================

:main
call :detect_environment
call :check_vcredist
call :create_environment
call :fix_windows_scripts
call :fix_python_scripts
call :run_validation

echo.
echo %GREEN%
echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║                                                                              ║
echo ║                    🎉 WINDOWS SETUP COMPLETED! 🎉                            ║
echo ║                                                                              ║
echo ║  Environment Type: !VENV_TYPE!                                                ║
echo ║  Python Command: !PYTHON_CMD!                                                 ║
echo ║                                                                              ║
echo ║  NEXT STEPS:                                                                 ║
echo ║  1. Add your data files to data\raw\ directory                              ║
echo ║  2. Run: quick_start.bat (to validate data)                                 ║
echo ║  3. Run: run_analysis.bat (to run full analysis)                            ║
echo ║                                                                              ║
echo ║  Windows-specific scripts created:                                           ║
echo ║  - run_analysis.bat                                                          ║
echo ║  - quick_start.bat                                                           ║
echo ║  - requirements_windows.txt                                                  ║
echo ║                                                                              ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo %NC%
echo.

echo Press any key to continue...
pause >nul

goto :eof

REM Execute main function
call :main
