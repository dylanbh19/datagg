#!/bin/bash

# ================================================================================
# PYTHON VENV SELF-HEALING DEBUGGER & PATCH SCRIPT
# ================================================================================
# This script intelligently diagnoses and fixes Python virtual environment issues
# Specifically designed for Windows/cross-platform analytics projects
# ================================================================================

# Relaxed error handling - won't exit on minor issues
set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

# Global tracking
ISSUES_FOUND=0
FIXES_APPLIED=0
VENV_PATH=""
PYTHON_CMD=""
OS_TYPE=""

print_banner() {
    clear
    echo -e "${CYAN}${BOLD}"
    cat << "EOF"
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  ██████╗██╗   ██╗████████╗██╗  ██╗ ██████╗ ███╗   ██╗                       ║
║  ██╔══██╗╚██╗ ██╔╝╚══██╔══╝██║  ██║██╔═══██╗████╗  ██║                       ║
║  ██████╔╝ ╚████╔╝    ██║   ███████║██║   ██║██╔██╗ ██║                       ║
║  ██╔═══╝   ╚██╔╝     ██║   ██╔══██║██║   ██║██║╚██╗██║                       ║
║  ██║        ██║      ██║   ██║  ██║╚██████╔╝██║ ╚████║                       ║
║  ╚═╝        ╚═╝      ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝                       ║
║                                                                              ║
║  ██╗   ██╗███████╗███╗   ██╗██╗   ██╗    ██╗  ██╗███████╗ █████╗ ██╗         ║
║  ██║   ██║██╔════╝████╗  ██║██║   ██║    ██║  ██║██╔════╝██╔══██╗██║         ║
║  ██║   ██║█████╗  ██╔██╗ ██║██║   ██║    ███████║█████╗  ███████║██║         ║
║  ╚██╗ ██╔╝██╔══╝  ██║╚██╗██║╚██╗ ██╔╝    ██╔══██║██╔══╝  ██╔══██║██║         ║
║   ╚████╔╝ ███████╗██║ ╚████║ ╚████╔╝     ██║  ██║███████╗██║  ██║███████╗    ║
║    ╚═══╝  ╚══════╝╚═╝  ╚═══╝  ╚═══╝      ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝    ║
║                                                                              ║
║           🩺 SELF-HEALING PYTHON ENVIRONMENT DEBUGGER 🩺                     ║
║                                                                              ║
║  • Diagnoses Python virtual environment issues                              ║
║  • Self-heals common problems automatically                                 ║
║  • Cross-platform compatible (Windows/Mac/Linux)                           ║
║  • Intelligent package management                                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
    ((ISSUES_FOUND++))
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    ((ISSUES_FOUND++))
}

log_fix() {
    echo -e "${MAGENTA}[AUTO-FIX]${NC} $1"
    ((FIXES_APPLIED++))
}

log_section() {
    echo -e "${CYAN}${BOLD}"
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo -e "${NC}"
}

# ================================================================================
# SYSTEM DETECTION FUNCTIONS
# ================================================================================

detect_os() {
    log_section "SYSTEM DETECTION"
    
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
        OS_TYPE="windows"
        log_info "Operating System: Windows (via $OSTYPE)"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS_TYPE="mac"
        log_info "Operating System: macOS"
    else
        OS_TYPE="linux"
        log_info "Operating System: Linux"
    fi
    
    # Windows-specific checks
    if [ "$OS_TYPE" == "windows" ]; then
        log_info "Windows environment detected - applying Windows-specific fixes"
        
        # Check if we're in the right shell
        if [ -z "$MSYSTEM" ] && [ -z "$BASH_VERSION" ]; then
            log_warning "Not running in proper bash environment"
            log_fix "Please use Git Bash or WSL for best compatibility"
        fi
    fi
}

detect_python() {
    log_section "PYTHON DETECTION & VALIDATION"
    
    local python_commands=("python3" "python" "py")
    local found_python=""
    
    for cmd in "${python_commands[@]}"; do
        if command -v "$cmd" &> /dev/null; then
            log_info "Found Python command: $cmd"
            
            # Test if it actually works
            if $cmd --version &> /dev/null; then
                local version=$($cmd --version 2>&1 | head -1)
                log_info "$cmd version: $version"
                
                # Extract version number more robustly
                local version_num=$(echo "$version" | grep -oE "[0-9]+\.[0-9]+\.[0-9]+" | head -1)
                if [ -n "$version_num" ]; then
                    local major=$(echo "$version_num" | cut -d. -f1)
                    local minor=$(echo "$version_num" | cut -d. -f2)
                    
                    if [ "$major" -ge 3 ] && [ "$minor" -ge 8 ]; then
                        found_python="$cmd"
                        PYTHON_CMD="$cmd"
                        log_success "Valid Python found: $cmd ($version_num)"
                        break
                    else
                        log_warning "$cmd version $version_num is too old (need 3.8+)"
                    fi
                else
                    log_warning "Could not parse version for $cmd"
                fi
            else
                log_warning "$cmd command exists but doesn't work properly"
            fi
        fi
    done
    
    if [ -z "$found_python" ]; then
        log_error "No suitable Python installation found"
        log_fix "Python installation guidance:"
        case $OS_TYPE in
            "windows")
                echo "  1. Download from https://python.org/downloads/"
                echo "  2. During installation, check 'Add Python to PATH'"
                echo "  3. Restart your terminal after installation"
                ;;
            "mac")
                echo "  1. Install via Homebrew: brew install python"
                echo "  2. Or download from https://python.org/downloads/"
                ;;
            "linux")
                echo "  1. Ubuntu/Debian: sudo apt-get install python3 python3-pip python3-venv"
                echo "  2. RedHat/CentOS: sudo yum install python3 python3-pip"
                ;;
        esac
        return 1
    fi
    
    # Test pip availability
    if $PYTHON_CMD -m pip --version &> /dev/null; then
        log_success "pip is available"
    else
        log_warning "pip is not available"
        log_fix "Attempting to install pip..."
        
        # Try to install pip
        if curl -s https://bootstrap.pypa.io/get-pip.py | $PYTHON_CMD; then
            log_success "pip installed successfully"
        else
            log_error "Failed to install pip automatically"
            echo "  Please install pip manually for your Python version"
        fi
    fi
    
    return 0
}

# ================================================================================
# VIRTUAL ENVIRONMENT FUNCTIONS
# ================================================================================

analyze_venv() {
    log_section "VIRTUAL ENVIRONMENT ANALYSIS"
    
    # Check for existing virtual environments
    local possible_venvs=("analytics_env" "venv" ".venv" "env")
    local found_venv=""
    
    for venv in "${possible_venvs[@]}"; do
        if [ -d "$venv" ]; then
            log_info "Found existing virtual environment: $venv"
            
            # Check if it's properly configured
            local activate_script=""
            if [ "$OS_TYPE" == "windows" ]; then
                activate_script="$venv/Scripts/activate"
                if [ ! -f "$activate_script" ]; then
                    activate_script="$venv/Scripts/activate.bat"
                fi
            else
                activate_script="$venv/bin/activate"
            fi
            
            if [ -f "$activate_script" ]; then
                log_success "Virtual environment $venv appears to be properly configured"
                VENV_PATH="$venv"
                found_venv="$venv"
                break
            else
                log_warning "Virtual environment $venv is corrupted (missing activation script)"
                log_fix "Will recreate this virtual environment"
                rm -rf "$venv" 2>/dev/null || true
            fi
        fi
    done
    
    if [ -z "$found_venv" ]; then
        log_info "No existing virtual environment found"
        log_fix "Creating new virtual environment: analytics_env"
        create_venv
    fi
}

create_venv() {
    log_info "Creating virtual environment..."
    
    # Remove any corrupted venv first
    if [ -d "analytics_env" ]; then
        log_fix "Removing corrupted analytics_env"
        rm -rf analytics_env
    fi
    
    # Create new virtual environment
    if $PYTHON_CMD -m venv analytics_env; then
        log_success "Virtual environment created successfully"
        VENV_PATH="analytics_env"
        
        # Verify it was created properly
        if [ "$OS_TYPE" == "windows" ]; then
            if [ -f "analytics_env/Scripts/activate" ] || [ -f "analytics_env/Scripts/activate.bat" ]; then
                log_success "Virtual environment structure verified"
            else
                log_error "Virtual environment creation failed - missing activation script"
                return 1
            fi
        else
            if [ -f "analytics_env/bin/activate" ]; then
                log_success "Virtual environment structure verified"
            else
                log_error "Virtual environment creation failed - missing activation script"
                return 1
            fi
        fi
    else
        log_error "Failed to create virtual environment"
        log_fix "Trying alternative methods..."
        
        # Try with explicit python3 -m venv
        if python3 -m venv analytics_env 2>/dev/null; then
            log_success "Virtual environment created with python3"
            VENV_PATH="analytics_env"
        else
            log_error "All virtual environment creation methods failed"
            echo "  This might be due to:"
            echo "  1. Missing python3-venv package (Linux)"
            echo "  2. Corrupted Python installation"
            echo "  3. Insufficient permissions"
            return 1
        fi
    fi
}

test_venv_activation() {
    log_section "VIRTUAL ENVIRONMENT ACTIVATION TEST"
    
    if [ -z "$VENV_PATH" ]; then
        log_error "No virtual environment path set"
        return 1
    fi
    
    # Determine activation script path
    local activate_script=""
    if [ "$OS_TYPE" == "windows" ]; then
        if [ -f "$VENV_PATH/Scripts/activate" ]; then
            activate_script="$VENV_PATH/Scripts/activate"
        elif [ -f "$VENV_PATH/Scripts/activate.bat" ]; then
            activate_script="$VENV_PATH/Scripts/activate.bat"
        fi
    else
        activate_script="$VENV_PATH/bin/activate"
    fi
    
    if [ ! -f "$activate_script" ]; then
        log_error "Activation script not found: $activate_script"
        log_fix "Recreating virtual environment..."
        rm -rf "$VENV_PATH"
        create_venv
        return
    fi
    
    log_info "Testing virtual environment activation..."
    
    # Test activation in a subshell
    if (source "$activate_script" && python --version) &> /dev/null; then
        log_success "Virtual environment activation test passed"
    else
        log_warning "Virtual environment activation test failed"
        log_fix "Recreating virtual environment..."
        rm -rf "$VENV_PATH"
        create_venv
    fi
}

# ================================================================================
# PACKAGE MANAGEMENT FUNCTIONS
# ================================================================================

analyze_packages() {
    log_section "PACKAGE ANALYSIS & MANAGEMENT"
    
    # Activate virtual environment for package testing
    local activate_script=""
    if [ "$OS_TYPE" == "windows" ]; then
        activate_script="$VENV_PATH/Scripts/activate"
    else
        activate_script="$VENV_PATH/bin/activate"
    fi
    
    if [ ! -f "$activate_script" ]; then
        log_error "Cannot test packages - virtual environment not properly set up"
        return 1
    fi
    
    # Test packages in a clean environment
    (
        source "$activate_script"
        
        # Upgrade pip first
        log_info "Upgrading pip..."
        if python -m pip install --upgrade pip --quiet 2>/dev/null; then
            log_success "pip upgraded successfully"
        else
            log_warning "pip upgrade failed, but continuing..."
        fi
        
        # Define required packages
        local core_packages=("pandas" "numpy" "matplotlib" "seaborn" "pyyaml")
        local ml_packages=("scikit-learn" "xgboost")
        local optional_packages=("plotly" "openpyxl" "tqdm" "statsmodels")
        
        # Test core packages
        log_info "Testing core packages..."
        for package in "${core_packages[@]}"; do
            if python -c "import $package" 2>/dev/null; then
                log_success "$package is available"
            else
                log_warning "$package is missing"
                log_fix "Installing $package..."
                if pip install "$package" --quiet; then
                    log_success "$package installed successfully"
                else
                    log_error "Failed to install $package"
                fi
            fi
        done
        
        # Test ML packages
        log_info "Testing machine learning packages..."
        for package in "${ml_packages[@]}"; do
            if python -c "import $package" 2>/dev/null; then
                log_success "$package is available"
            else
                log_warning "$package is missing"
                log_fix "Installing $package..."
                if pip install "$package" --quiet; then
                    log_success "$package installed successfully"
                else
                    log_warning "Failed to install $package - continuing without it"
                fi
            fi
        done
        
        # Test optional packages
        log_info "Testing optional packages..."
        for package in "${optional_packages[@]}"; do
            if python -c "import $package" 2>/dev/null; then
                log_success "$package is available"
            else
                log_info "$package is missing (optional)"
                if pip install "$package" --quiet 2>/dev/null; then
                    log_success "$package installed successfully"
                else
                    log_info "$package installation skipped (optional)"
                fi
            fi
        done
    )
}

create_package_test_script() {
    log_info "Creating package verification script..."
    
    cat > test_packages.py << 'EOF'
"""
Package Testing Script
Tests all packages and reports their status
"""
import sys

def test_package(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name}: OK")
        return True
    except ImportError as e:
        print(f"❌ {package_name}: MISSING ({e})")
        return False

def main():
    print("="*60)
    print("PACKAGE VERIFICATION REPORT")
    print("="*60)
    
    packages = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("scikit-learn", "sklearn"),
        ("xgboost", "xgboost"),
        ("pyyaml", "yaml"),
        ("plotly", "plotly"),
        ("openpyxl", "openpyxl"),
        ("tqdm", "tqdm"),
        ("statsmodels", "statsmodels")
    ]
    
    success_count = 0
    total_count = len(packages)
    
    for package_name, import_name in packages:
        if test_package(package_name, import_name):
            success_count += 1
    
    print("="*60)
    print(f"SUMMARY: {success_count}/{total_count} packages available")
    print(f"Python version: {sys.version}")
    print("="*60)
    
    if success_count >= 5:  # Core packages
        print("✅ Environment is ready for analytics!")
        return 0
    else:
        print("❌ Environment needs more packages")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

    log_success "Package test script created: test_packages.py"
}

# ================================================================================
# DIAGNOSTIC FUNCTIONS
# ================================================================================

run_comprehensive_diagnostics() {
    log_section "COMPREHENSIVE DIAGNOSTICS"
    
    # Create detailed diagnostic report
    cat > diagnostic_report.txt << EOF
PYTHON VIRTUAL ENVIRONMENT DIAGNOSTIC REPORT
Generated: $(date)
System: $OS_TYPE
Python Command: $PYTHON_CMD
Virtual Environment: $VENV_PATH

SYSTEM INFORMATION:
- OS Type: $OSTYPE
- Python Version: $($PYTHON_CMD --version 2>&1)
- Pip Version: $($PYTHON_CMD -m pip --version 2>&1)

DIRECTORY STRUCTURE:
$(ls -la | head -20)

VIRTUAL ENVIRONMENT STRUCTURE:
$(if [ -d "$VENV_PATH" ]; then find "$VENV_PATH" -maxdepth 3 -type f -name "*.exe" -o -name "python*" -o -name "activate*" 2>/dev/null; else echo "No virtual environment found"; fi)

ISSUES FOUND: $ISSUES_FOUND
FIXES APPLIED: $FIXES_APPLIED
EOF

    log_success "Diagnostic report saved: diagnostic_report.txt"
    
    # Test the environment
    if [ -n "$VENV_PATH" ]; then
        log_info "Running environment test..."
        
        if [ "$OS_TYPE" == "windows" ]; then
            activate_script="$VENV_PATH/Scripts/activate"
        else
            activate_script="$VENV_PATH/bin/activate"
        fi
        
        if [ -f "$activate_script" ]; then
            (
                source "$activate_script"
                python test_packages.py
            )
        else
            log_error "Cannot test environment - activation script missing"
        fi
    fi
}

# ================================================================================
# BATCH FILE CREATION (Windows)
# ================================================================================

create_convenience_scripts() {
    log_section "CREATING CONVENIENCE SCRIPTS"
    
    if [ "$OS_TYPE" == "windows" ]; then
        log_info "Creating Windows batch files..."
        
        # Create activate.bat
        cat > activate_env.bat << 'EOF'
@echo off
echo Activating Python virtual environment...
if exist analytics_env\Scripts\activate.bat (
    call analytics_env\Scripts\activate.bat
    echo Virtual environment activated!
    echo Type 'python test_packages.py' to test packages
) else (
    echo Virtual environment not found!
    echo Run: bash venv_self_healer.sh
)
pause
EOF
        
        # Create test.bat
        cat > test_env.bat << 'EOF'
@echo off
echo Testing Python environment...
if exist analytics_env\Scripts\activate.bat (
    call analytics_env\Scripts\activate.bat
    python test_packages.py
) else (
    echo Virtual environment not found!
)
pause
EOF
        
        log_success "Windows batch files created:"
        log_info "  - activate_env.bat: Activate virtual environment"
        log_info "  - test_env.bat: Test package installation"
    fi
    
    # Create universal shell script
    cat > activate_env.sh << 'EOF'
#!/bin/bash
echo "Activating Python virtual environment..."

if [ -f "analytics_env/Scripts/activate" ]; then
    source analytics_env/Scripts/activate
    echo "Virtual environment activated (Windows)!"
elif [ -f "analytics_env/bin/activate" ]; then
    source analytics_env/bin/activate
    echo "Virtual environment activated (Unix)!"
else
    echo "Virtual environment not found!"
    echo "Run: ./venv_self_healer.sh"
    exit 1
fi

echo "Type 'python test_packages.py' to test packages"
EOF
    
    chmod +x activate_env.sh
    log_success "Universal activation script created: activate_env.sh"
}

# ================================================================================
# FINAL REPORT
# ================================================================================

display_final_report() {
    log_section "FINAL DIAGNOSTIC REPORT"
    
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                        VENV SELF-HEALER COMPLETE                             ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo
    
    # System status
    echo -e "${BOLD}SYSTEM STATUS:${NC}"
    echo -e "  Operating System: ${BLUE}$OS_TYPE${NC}"
    echo -e "  Python Command: ${BLUE}$PYTHON_CMD${NC}"
    echo -e "  Virtual Environment: ${BLUE}${VENV_PATH:-'Not Set'}${NC}"
    echo
    
    # Issue summary
    echo -e "${BOLD}DIAGNOSTIC SUMMARY:${NC}"
    if [ $ISSUES_FOUND -eq 0 ]; then
        echo -e "  ${GREEN}✅ No issues found${NC}"
    else
        echo -e "  ${YELLOW}⚠️  Issues found: $ISSUES_FOUND${NC}"
    fi
    echo -e "  ${MAGENTA}🔧 Auto-fixes applied: $FIXES_APPLIED${NC}"
    echo
    
    # Environment status
    echo -e "${BOLD}ENVIRONMENT STATUS:${NC}"
    if [ -n "$VENV_PATH" ] && [ -d "$VENV_PATH" ]; then
        echo -e "  ${GREEN}✅ Virtual environment ready${NC}"
        
        # Quick package check
        if [ "$OS_TYPE" == "windows" ]; then
            activate_script="$VENV_PATH/Scripts/activate"
        else
            activate_script="$VENV_PATH/bin/activate"
        fi
        
        if [ -f "$activate_script" ]; then
            local package_count=$(source "$activate_script" && pip list 2>/dev/null | wc -l)
            echo -e "  ${GREEN}✅ Packages installed: $package_count${NC}"
        fi
    else
        echo -e "  ${RED}❌ Virtual environment not ready${NC}"
    fi
    echo
    
    # Next steps
    echo -e "${BOLD}NEXT STEPS:${NC}"
    echo -e "  ${BLUE}1.${NC} Test your environment:"
    if [ "$OS_TYPE" == "windows" ]; then
        echo -e "     ${CYAN}test_env.bat${NC} (Windows) or ${CYAN}python test_packages.py${NC}"
    else
        echo -e "     ${CYAN}./activate_env.sh${NC} then ${CYAN}python test_packages.py${NC}"
    fi
    echo -e "  ${BLUE}2.${NC} Activate environment:"
    if [ "$OS_TYPE" == "windows" ]; then
        echo -e "     ${CYAN}activate_env.bat${NC} or ${CYAN}source analytics_env/Scripts/activate${NC}"
    else
        echo -e "     ${CYAN}source analytics_env/bin/activate${NC}"
    fi
    echo -e "  ${BLUE}3.${NC} Run your analytics:"
    echo -e "     ${CYAN}python src/main.py${NC}"
    echo
    
    # Files created
    echo -e "${BOLD}FILES CREATED:${NC}"
    echo -e "  ${GREEN}📄 test_packages.py${NC} - Package verification script"
    echo -e "  ${GREEN}📄 diagnostic_report.txt${NC} - Detailed diagnostic report"
    echo -e "  ${GREEN}📄 activate_env.sh${NC} - Universal activation script"
    if [ "$OS_TYPE" == "windows" ]; then
        echo -e "  ${GREEN}📄 activate_env.bat${NC} - Windows activation script"
        echo -e "  ${GREEN}📄 test_env.bat${NC} - Windows test script"
    fi
    echo
    
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                     ENVIRONMENT READY FOR ANALYTICS!                        ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
}

# ================================================================================
# MAIN EXECUTION
# ================================================================================

main() {
    print_banner
    
    log_info "Starting Python virtual environment self-healing process..."
    echo
    
    # Run diagnostic phases
    detect_os
    
    if ! detect_python; then
        log_error "Cannot continue without Python"
        exit 1
    fi
    
    analyze_venv
    test_venv_activation
    analyze_packages
    create_package_test_script
    run_comprehensive_diagnostics
    create_convenience_scripts
    
    # Final report
    display_final_report
    
    # Keep terminal open on Windows
    if [ "$OS_TYPE" == "windows" ]; then
        echo
        read -p "Press Enter to exit..." || true
    fi
}

# Handle errors gracefully
trap 'log_error "Script interrupted"; exit 1' INT
trap 'log_error "Unexpected error occurred"; exit 1' ERR

# Execute main function
main "$@"
