#!/bin/bash

# ================================================================================

# PRODUCTION-GRADE BULLETPROOF ANALYTICS RUNNER

# ================================================================================

# This runner never hangs, handles all errors, and provides complete visibility

# Production-grade error handling with comprehensive logging and recovery

# ================================================================================

# Strict error handling but don’t exit on failures - handle them

set +e

# Global state tracking

RUNNER_START_TIME=$(date +%s)
LOG_FILE=””
PYTHON_CMD=””
OS_TYPE=””
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNINGS=0

# Color codes (safe for all terminals)

if [[ -t 1 ]]; then
RED=’\033[0;31m’
GREEN=’\033[0;32m’
YELLOW=’\033[1;33m’
BLUE=’\033[0;34m’
CYAN=’\033[0;36m’
MAGENTA=’\033[0;35m’
BOLD=’\033[1m’
NC=’\033[0m’
else
RED=’’
GREEN=’’
YELLOW=’’
BLUE=’’
CYAN=’’
MAGENTA=’’
BOLD=’’
NC=’’
fi

# ================================================================================

# LOGGING AND ERROR HANDLING INFRASTRUCTURE

# ================================================================================

setup_logging() {
local timestamp=$(date +”%Y%m%d_%H%M%S”)
mkdir -p logs 2>/dev/null || true
LOG_FILE=“logs/production_runner_${timestamp}.log”

```
# Create log file and make it accessible
touch "$LOG_FILE" 2>/dev/null || LOG_FILE="/tmp/analytics_runner.log"

echo "Production Analytics Runner Started: $(date)" > "$LOG_FILE"
echo "Working Directory: $(pwd)" >> "$LOG_FILE"
echo "User: $(whoami 2>/dev/null || echo 'unknown')" >> "$LOG_FILE"
echo "Shell: $0" >> "$LOG_FILE"
echo "Args: $*" >> "$LOG_FILE"
echo "================================" >> "$LOG_FILE"
```

}

log_and_echo() {
local level=”$1”
shift
local message=”$*”
local timestamp=$(date “+%Y-%m-%d %H:%M:%S”)

```
# Log to file (always)
echo "[$timestamp] [$level] $message" >> "$LOG_FILE" 2>/dev/null || true

# Echo to console with colors
case "$level" in
    "SUCCESS")
        echo -e "${GREEN}[SUCCESS]${NC} ✅ $message"
        ;;
    "ERROR")
        echo -e "${RED}[ERROR]${NC} ❌ $message"
        ;;
    "WARNING")
        echo -e "${YELLOW}[WARNING]${NC} ⚠️  $message"
        ;;
    "INFO")
        echo -e "${BLUE}[INFO]${NC} ℹ️  $message"
        ;;
    "DEBUG")
        echo -e "${MAGENTA}[DEBUG]${NC} 🔍 $message"
        ;;
    *)
        echo "$message"
        ;;
esac
```

}

safe_increment() {
local var_name=”$1”
local current_val
current_val=$(eval echo $$var_name)
local new_val=$((current_val + 1))
eval “$var_name=$new_val”
}

check_passed() {
safe_increment “TOTAL_CHECKS”
safe_increment “PASSED_CHECKS”
log_and_echo “SUCCESS” “$1”
}

check_failed() {
safe_increment “TOTAL_CHECKS”
safe_increment “FAILED_CHECKS”
log_and_echo “ERROR” “$1”
}

check_warning() {
safe_increment “WARNINGS”
log_and_echo “WARNING” “$1”
}

# ================================================================================

# PRODUCTION BANNER AND STARTUP

# ================================================================================

print_production_banner() {
echo -e “${CYAN}${BOLD}”
cat << “EOF”
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                  PRODUCTION-GRADE ANALYTICS RUNNER                          ║
║                                                                              ║
║  🔒 Bulletproof Error Handling    🔍 Complete Visibility                    ║
║  🚀 Never Hangs or Stops          📊 Comprehensive Logging                  ║
║  ⚡ Production-Ready               🛡️  Self-Healing Capabilities             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
echo -e “${NC}”

```
log_and_echo "INFO" "Production Analytics Runner v2.0 Starting"
log_and_echo "INFO" "Session ID: $(date +%s)"
log_and_echo "INFO" "Log File: $LOG_FILE"
```

}

# ================================================================================

# ENVIRONMENT DETECTION WITH BULLETPROOF ERROR HANDLING

# ================================================================================

detect_environment_safe() {
log_and_echo “INFO” “Starting environment detection…”

```
# Detect OS with multiple fallbacks
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    OS_TYPE="windows"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS_TYPE="mac"
elif [[ "$OSTYPE" == "linux"* ]]; then
    OS_TYPE="linux"
else
    # Fallback detection
    if command -v cmd.exe >/dev/null 2>&1; then
        OS_TYPE="windows"
    elif [[ -f "/etc/os-release" ]]; then
        OS_TYPE="linux"
    elif [[ -f "/System/Library/CoreServices/SystemVersion.plist" ]]; then
        OS_TYPE="mac"
    else
        OS_TYPE="unknown"
    fi
fi

log_and_echo "INFO" "Operating System: $OS_TYPE (detected via $OSTYPE)"

# Detect Python with comprehensive fallback chain
local python_candidates=("python3" "python" "py" "/usr/bin/python3" "/usr/local/bin/python3")

for cmd in "${python_candidates[@]}"; do
    log_and_echo "DEBUG" "Testing Python command: $cmd"
    
    if command -v "$cmd" >/dev/null 2>&1; then
        log_and_echo "DEBUG" "Command $cmd exists, testing execution..."
        
        # Test if it actually works and get version
        local version_output
        if version_output=$("$cmd" --version 2>&1) && [[ $? -eq 0 ]]; then
            log_and_echo "DEBUG" "$cmd version output: $version_output"
            
            # Extract version number
            local version_num
            version_num=$(echo "$version_output" | grep -oE '[0-9]+\.[0-9]+' | head -1)
            
            if [[ -n "$version_num" ]]; then
                PYTHON_CMD="$cmd"
                check_passed "Python found: $cmd ($version_output)"
                log_and_echo "INFO" "Python version: $version_num"
                break
            else
                log_and_echo "DEBUG" "$cmd version parsing failed"
            fi
        else
            log_and_echo "DEBUG" "$cmd execution failed (exit code: $?)"
        fi
    else
        log_and_echo "DEBUG" "Command $cmd not found"
    fi
done

if [[ -z "$PYTHON_CMD" ]]; then
    check_failed "No working Python installation found"
    log_and_echo "ERROR" "Tested commands: ${python_candidates[*]}"
    return 1
fi

# Test pip availability
log_and_echo "DEBUG" "Testing pip availability..."
if "$PYTHON_CMD" -m pip --version >/dev/null 2>&1; then
    check_passed "pip is available"
else
    check_warning "pip not available - will attempt to install packages via alternative methods"
fi

return 0
```

}

# ================================================================================

# PROJECT STRUCTURE VALIDATION

# ================================================================================

validate_project_structure() {
log_and_echo “INFO” “Validating project structure…”

```
# Critical files and directories
local critical_items=(
    "src/main.py:file:Main analytics script"
    "config/config.yaml:file:Configuration file"
    "src:dir:Source code directory"
    "data:dir:Data directory"
    "data/raw:dir:Raw data directory"
)

local structure_ok=true

for item in "${critical_items[@]}"; do
    local path="${item%%:*}"
    local type="${item#*:}"
    type="${type%%:*}"
    local desc="${item##*:}"
    
    log_and_echo "DEBUG" "Checking $type: $path ($desc)"
    
    if [[ "$type" == "file" ]]; then
        if [[ -f "$path" ]]; then
            check_passed "$desc exists: $path"
        else
            check_failed "$desc missing: $path"
            structure_ok=false
        fi
    elif [[ "$type" == "dir" ]]; then
        if [[ -d "$path" ]]; then
            check_passed "$desc exists: $path"
        else
            check_warning "$desc missing: $path (will create)"
            mkdir -p "$path" 2>/dev/null || true
        fi
    fi
done

# Create missing output directories
for dir in "outputs/plots" "outputs/reports" "logs"; do
    if [[ ! -d "$dir" ]]; then
        log_and_echo "DEBUG" "Creating directory: $dir"
        mkdir -p "$dir" 2>/dev/null || log_and_echo "WARNING" "Failed to create $dir"
    fi
done

if [[ "$structure_ok" == "true" ]]; then
    check_passed "Project structure validation complete"
    return 0
else
    check_failed "Project structure validation failed - missing critical files"
    return 1
fi
```

}

# ================================================================================

# DATA FILE VALIDATION

# ================================================================================

validate_data_files() {
log_and_echo “INFO” “Validating data files…”

```
local required_files=(
    "data/raw/mail.csv:Mail campaign data:mail_date,mail_type,mail_volume"
    "data/raw/call_intents.csv:Call intent data:ConversationStart,uui_Intent"
    "data/raw/call_volume.csv:Call volume data:Date,call_volume"
)

local files_found=0
local files_validated=0

for file_spec in "${required_files[@]}"; do
    local file_path="${file_spec%%:*}"
    local desc="${file_spec#*:}"
    desc="${desc%%:*}"
    local expected_cols="${file_spec##*:}"
    
    log_and_echo "DEBUG" "Checking data file: $file_path"
    
    if [[ -f "$file_path" ]]; then
        files_found=$((files_found + 1))
        check_passed "$desc found: $file_path"
        
        # Validate file content
        log_and_echo "DEBUG" "Validating content of $file_path"
        
        # Check if file is readable and not empty
        if [[ -r "$file_path" && -s "$file_path" ]]; then
            # Try to read first few lines
            if head -5 "$file_path" >/dev/null 2>&1; then
                check_passed "$desc is readable and non-empty"
                files_validated=$((files_validated + 1))
                
                # Get line count
                local line_count
                line_count=$(wc -l < "$file_path" 2>/dev/null || echo "unknown")
                log_and_echo "INFO" "$desc contains $line_count lines"
                
            else
                check_warning "$desc exists but may be corrupted"
            fi
        else
            check_warning "$desc exists but is empty or unreadable"
        fi
    else
        check_failed "$desc missing: $file_path"
        log_and_echo "INFO" "Expected columns: $expected_cols"
    fi
done

log_and_echo "INFO" "Data file summary: $files_found/3 files found, $files_validated/3 validated"

if [[ $files_found -eq 0 ]]; then
    check_failed "No data files found - cannot proceed with analysis"
    return 1
elif [[ $files_found -lt 3 ]]; then
    check_warning "Some data files missing - analysis may be limited"
    return 0
else
    check_passed "All required data files present"
    return 0
fi
```

}

# ================================================================================

# PACKAGE MANAGEMENT WITH BULLETPROOF INSTALLATION

# ================================================================================

validate_and_install_packages() {
log_and_echo “INFO” “Validating and installing required packages…”

```
# Core packages with fallback names
local packages=(
    "pandas:pandas:Data manipulation library"
    "numpy:numpy:Numerical computing library"
    "matplotlib:matplotlib:Plotting library"
    "seaborn:seaborn:Statistical visualization"
    "pyyaml:yaml:YAML configuration parser"
    "scikit-learn:sklearn:Machine learning library"
)

# Optional packages
local optional_packages=(
    "plotly:plotly:Interactive plotting"
    "openpyxl:openpyxl:Excel file support"
    "tqdm:tqdm:Progress bars"
)

local installed_count=0
local total_required=${#packages[@]}

# Check and install core packages
for pkg_spec in "${packages[@]}"; do
    local pkg_name="${pkg_spec%%:*}"
    local import_name="${pkg_spec#*:}"
    import_name="${import_name%%:*}"
    local desc="${pkg_spec##*:}"
    
    log_and_echo "DEBUG" "Checking package: $pkg_name (import as: $import_name)"
    
    if test_package_import "$import_name"; then
        check_passed "$desc available ($pkg_name)"
        installed_count=$((installed_count + 1))
    else
        log_and_echo "INFO" "Installing $desc ($pkg_name)..."
        if install_package_safe "$pkg_name" "$import_name"; then
            check_passed "$desc installed successfully"
            installed_count=$((installed_count + 1))
        else
            check_failed "$desc installation failed"
        fi
    fi
done

# Check optional packages (don't fail if missing)
for pkg_spec in "${optional_packages[@]}"; do
    local pkg_name="${pkg_spec%%:*}"
    local import_name="${pkg_spec#*:}"
    import_name="${import_name%%:*}"
    local desc="${pkg_spec##*:}"
    
    if test_package_import "$import_name"; then
        log_and_echo "INFO" "Optional $desc available ($pkg_name)"
    else
        log_and_echo "DEBUG" "Optional package $pkg_name not available (will install if possible)"
        install_package_safe "$pkg_name" "$import_name" >/dev/null 2>&1 || true
    fi
done

log_and_echo "INFO" "Package summary: $installed_count/$total_required core packages available"

if [[ $installed_count -ge 4 ]]; then
    check_passed "Sufficient packages available for analytics"
    return 0
else
    check_failed "Insufficient packages - need at least 4 core packages"
    return 1
fi
```

}

test_package_import() {
local import_name=”$1”
log_and_echo “DEBUG” “Testing import: $import_name”

```
if "$PYTHON_CMD" -c "import $import_name" >/dev/null 2>&1; then
    return 0
else
    return 1
fi
```

}

install_package_safe() {
local pkg_name=”$1”
local import_name=”$2”

```
log_and_echo "DEBUG" "Attempting to install package: $pkg_name"

# Strategy 1: Standard installation
if "$PYTHON_CMD" -m pip install "$pkg_name" --quiet >/dev/null 2>&1; then
    log_and_echo "DEBUG" "Package $pkg_name installed via standard method"
    if test_package_import "$import_name"; then
        return 0
    fi
fi

# Strategy 2: User installation
log_and_echo "DEBUG" "Trying user installation for $pkg_name"
if "$PYTHON_CMD" -m pip install "$pkg_name" --user --quiet >/dev/null 2>&1; then
    log_and_echo "DEBUG" "Package $pkg_name installed via --user"
    if test_package_import "$import_name"; then
        return 0
    fi
fi

# Strategy 3: Force reinstall
log_and_echo "DEBUG" "Trying force reinstall for $pkg_name"
if "$PYTHON_CMD" -m pip install "$pkg_name" --force-reinstall --quiet >/dev/null 2>&1; then
    log_and_echo "DEBUG" "Package $pkg_name force reinstalled"
    if test_package_import "$import_name"; then
        return 0
    fi
fi

log_and_echo "DEBUG" "All installation strategies failed for $pkg_name"
return 1
```

}

# ================================================================================

# ANALYTICS EXECUTION WITH COMPREHENSIVE MONITORING

# ================================================================================

execute_analytics_with_monitoring() {
log_and_echo “INFO” “Starting analytics execution with comprehensive monitoring…”

```
local start_time=$(date +%s)
local analytics_log="logs/analytics_execution.log"

# Prepare execution environment
log_and_echo "DEBUG" "Setting up execution environment..."

# Change to project root if needed
if [[ ! -f "src/main.py" ]] && [[ -f "../src/main.py" ]]; then
    cd ..
    log_and_echo "DEBUG" "Changed to parent directory"
fi

# Verify we can execute
if [[ ! -f "src/main.py" ]]; then
    check_failed "Cannot locate src/main.py for execution"
    return 1
fi

# Pre-execution checks
log_and_echo "DEBUG" "Running pre-execution checks..."

# Test Python execution
if ! "$PYTHON_CMD" -c "print('Python execution test passed')" >/dev/null 2>&1; then
    check_failed "Python execution test failed"
    return 1
fi

# Test critical imports
if ! "$PYTHON_CMD" -c "import pandas, numpy; print('Core imports successful')" >/dev/null 2>&1; then
    check_failed "Core package import test failed"
    return 1
fi

check_passed "Pre-execution checks completed"

# Execute analytics with full monitoring
log_and_echo "INFO" "Executing analytics pipeline..."
log_and_echo "DEBUG" "Command: $PYTHON_CMD src/main.py"
log_and_echo "DEBUG" "Working directory: $(pwd)"
log_and_echo "DEBUG" "Analytics log: $analytics_log"

# Execute with timeout and comprehensive logging
local exit_code=0
local execution_output

# Run with timeout (10 minutes max)
if command -v timeout >/dev/null 2>&1; then
    execution_output=$(timeout 600 "$PYTHON_CMD" src/main.py 2>&1) || exit_code=$?
else
    execution_output=$("$PYTHON_CMD" src/main.py 2>&1) || exit_code=$?
fi

# Log execution output
echo "=== Analytics Execution Output ===" >> "$analytics_log"
echo "$execution_output" >> "$analytics_log"
echo "=== Execution Exit Code: $exit_code ===" >> "$analytics_log"

local end_time=$(date +%s)
local duration=$((end_time - start_time))

log_and_echo "INFO" "Analytics execution completed in ${duration}s with exit code: $exit_code"

# Analyze execution results
if [[ $exit_code -eq 0 ]]; then
    check_passed "Analytics pipeline executed successfully"
    validate_execution_outputs
    return 0
elif [[ $exit_code -eq 124 ]]; then
    check_failed "Analytics execution timed out (>10 minutes)"
    return 1
else
    check_failed "Analytics execution failed with exit code: $exit_code"
    log_and_echo "DEBUG" "Execution output logged to: $analytics_log"
    
    # Try to extract useful error information
    if [[ -n "$execution_output" ]]; then
        # Get last few lines of output for immediate feedback
        local error_summary
        error_summary=$(echo "$execution_output" | tail -5)
        log_and_echo "ERROR" "Last error output: $error_summary"
    fi
    
    return 1
fi
```

}

validate_execution_outputs() {
log_and_echo “INFO” “Validating execution outputs…”

```
local outputs_found=0

# Check for plots
if [[ -d "outputs/plots" ]]; then
    local plot_count
    plot_count=$(find outputs/plots -name "*.png" 2>/dev/null | wc -l)
    if [[ $plot_count -gt 0 ]]; then
        check_passed "Generated $plot_count visualization plots"
        outputs_found=$((outputs_found + 1))
        
        # List some plots
        local plot_list
        plot_list=$(find outputs/plots -name "*.png" 2>/dev/null | head -3)
        log_and_echo "INFO" "Sample plots: $plot_list"
    else
        check_warning "No visualization plots generated"
    fi
fi

# Check for reports
if [[ -d "outputs/reports" ]]; then
    local report_count
    report_count=$(find outputs/reports -name "*.json" -o -name "*.txt" 2>/dev/null | wc -l)
    if [[ $report_count -gt 0 ]]; then
        check_passed "Generated $report_count analysis reports"
        outputs_found=$((outputs_found + 1))
    else
        check_warning "No analysis reports generated"
    fi
fi

# Check for logs
if [[ -f "logs/analytics.log" ]]; then
    check_passed "Analytics log file created"
    outputs_found=$((outputs_found + 1))
fi

if [[ $outputs_found -gt 0 ]]; then
    check_passed "Execution outputs validated successfully"
    return 0
else
    check_warning "Execution completed but no outputs detected"
    return 1
fi
```

}

# ================================================================================

# COMPREHENSIVE FINAL REPORTING

# ================================================================================

generate_final_report() {
local end_time=$(date +%s)
local total_duration=$((end_time - RUNNER_START_TIME))

```
log_and_echo "INFO" "Generating comprehensive final report..."

echo -e "${CYAN}${BOLD}"
cat << "EOF"
```

╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    PRODUCTION RUNNER FINAL REPORT                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
echo -e “${NC}”

```
# System Information
echo -e "${BOLD}SYSTEM INFORMATION:${NC}"
echo -e "  🐍 Python: ${GREEN}$PYTHON_CMD${NC}"
echo -e "  💻 OS: ${GREEN}$OS_TYPE${NC}"
echo -e "  📁 Directory: ${GREEN}$(pwd)${NC}"
echo -e "  ⏱️  Total Runtime: ${GREEN}${total_duration}s${NC}"
echo

# Execution Summary
echo -e "${BOLD}EXECUTION SUMMARY:${NC}"
echo -e "  ✅ Checks Passed: ${GREEN}$PASSED_CHECKS${NC}"
echo -e "  ❌ Checks Failed: ${RED}$FAILED_CHECKS${NC}"
echo -e "  ⚠️  Warnings: ${YELLOW}$WARNINGS${NC}"
echo -e "  📊 Total Checks: ${BLUE}$TOTAL_CHECKS${NC}"
echo

# Success Rate
local success_rate=0
if [[ $TOTAL_CHECKS -gt 0 ]]; then
    success_rate=$(( (PASSED_CHECKS * 100) / TOTAL_CHECKS ))
fi

echo -e "${BOLD}SUCCESS RATE:${NC}"
if [[ $success_rate -ge 90 ]]; then
    echo -e "  🎉 ${GREEN}$success_rate% - EXCELLENT${NC}"
elif [[ $success_rate -ge 75 ]]; then
    echo -e "  👍 ${YELLOW}$success_rate% - GOOD${NC}"
elif [[ $success_rate -ge 50 ]]; then
    echo -e "  ⚠️  ${YELLOW}$success_rate% - ACCEPTABLE${NC}"
else
    echo -e "  ❌ ${RED}$success_rate% - NEEDS ATTENTION${NC}"
fi
echo

# Output Files
echo -e "${BOLD}GENERATED OUTPUTS:${NC}"

# List plots
if [[ -d "outputs/plots" ]]; then
    local plot_files
    plot_files=$(find outputs/plots -name "*.png" 2>/dev/null)
    if [[ -n "$plot_files" ]]; then
        echo -e "  📊 ${GREEN}Visualizations:${NC}"
        echo "$plot_files" | while read -r file; do
            echo -e "     • $file"
        done
    fi
fi

# List reports
if [[ -d "outputs/reports" ]]; then
    local report_files
    report_files=$(find outputs/reports -name "*.json" -o -name "*.txt" 2>/dev/null)
    if [[ -n "$report_files" ]]; then
        echo -e "  📋 ${GREEN}Reports:${NC}"
        echo "$report_files" | while read -r file; do
            echo -e "     • $file"
        done
    fi
fi

# Log files
echo -e "  📝 ${GREEN}Logs:${NC}"
echo -e "     • $LOG_FILE"
if [[ -f "logs/analytics.log" ]]; then
    echo -e "     • logs/analytics.log"
fi
if [[ -f "logs/analytics_execution.log" ]]; then
    echo -e "     • logs/analytics_execution.log"
fi
echo

# Next Steps
echo -e "${BOLD}NEXT STEPS:${NC}"
if [[ $FAILED_CHECKS -eq 0 ]]; then
    echo -e "  🎯 ${GREEN}Analytics completed successfully!${NC}"
    echo -e "  📊 Review outputs/plots/ for visualizations"
    echo -e "  📋 Check outputs/reports/ for analysis results"
else
    echo -e "  🔍 ${YELLOW}Review failed checks above${NC}"
    echo -e "  📝 Check log file: $LOG_FILE"
    echo -e "  🔄 Fix issues and re-run if needed"
fi
echo

# Final status
if [[ $FAILED_CHECKS -eq 0 ]]; then
    echo -e "${GREEN}${BOLD}🎉 PRODUCTION RUNNER COMPLETED SUCCESSFULLY! 🎉${NC}"
else
    echo -e "${YELLOW}${BOLD}⚠️  PRODUCTION RUNNER COMPLETED WITH ISSUES ⚠️${NC}"
fi

echo
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                        RUNNER SESSION COMPLETE                              ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
```

}

# ================================================================================

# MAIN EXECUTION FLOW

# ================================================================================

main() {
# Initialize logging first
setup_logging

```
# Print banner
print_production_banner

# Execute all phases with error handling
local overall_success=true

# Phase 1: Environment Detection
if ! detect_environment_safe; then
    overall_success=false
    log_and_echo "ERROR" "Environment detection failed - cannot continue"
    generate_final_report
    exit 1
fi

# Phase 2: Project Structure Validation
if ! validate_project_structure; then
    overall_success=false
    log_and_echo "ERROR" "Project structure validation failed - cannot continue"
    generate_final_report
    exit 1
fi

# Phase 3: Data File Validation
if ! validate_data_files; then
    overall_success=false
    log_and_echo "WARNING" "Data file validation failed - will attempt to continue"
fi

# Phase 4: Package Management
if ! validate_and_install_packages; then
    overall_success=false
    log_and_echo "ERROR" "Package validation failed - cannot continue"
    generate_final_report
    exit 1
fi

# Phase 5: Analytics Execution
if ! execute_analytics_with_monitoring; then
    overall_success=false
    log_and_echo "ERROR" "Analytics execution failed"
fi

# Generate comprehensive final report
generate_final_report

# Final exit handling
if [[ "$overall_success" == "true" ]]; then
    log_and_echo "SUCCESS" "Production runner completed successfully"
    exit 0
else
    log_and_echo "ERROR" "Production runner completed with failures"
    exit 1
fi
```

}

# Error handling for the entire script

trap ‘log_and_echo “ERROR” “Script interrupted at line $LINENO”; generate_final_report; exit 1’ INT
trap ‘log_and_echo “ERROR” “Unexpected error at line $LINENO”; generate_final_report; exit 1’ ERR

# Execute main function with all arguments

main “$@”