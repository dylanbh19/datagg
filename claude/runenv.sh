#!/bin/bash

# ================================================================================
# SIMPLE ANALYTICS RUNNER - NO VENV, WINDOWS COMPATIBLE
# ================================================================================
# Checks environment, installs missing packages, runs your existing analytics code
# Uses your existing project structure and code
# ================================================================================

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Find Python
PYTHON_CMD=""
if command -v python3 &> /dev/null && python3 --version &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null && python --version &> /dev/null; then
    PYTHON_CMD="python"
elif command -v py &> /dev/null && py --version &> /dev/null; then
    PYTHON_CMD="py"
else
    echo -e "${RED}❌ Python not found. Please install Python 3.8+${NC}"
    exit 1
fi

echo -e "${CYAN}"
cat << "EOF"
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    SIMPLE ANALYTICS RUNNER                                  ║
║                                                                              ║
║  • Checks your environment                                                  ║
║  • Installs missing packages                                                ║
║  • Runs your existing analytics code                                        ║
║  • Windows compatible, no virtual environments                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo -e "${BLUE}[INFO]${NC} Using Python: $PYTHON_CMD"
echo -e "${BLUE}[INFO]${NC} Python version: $($PYTHON_CMD --version)"
echo

# Check if project exists
echo -e "${CYAN}════════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  CHECKING PROJECT STRUCTURE${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════════════════════${NC}"

if [ ! -f "src/main.py" ]; then
    echo -e "${RED}❌ src/main.py not found${NC}"
    echo "Please run complete_windows_setup.sh first to create the project structure"
    exit 1
fi

echo -e "${GREEN}✅ Project structure exists${NC}"

# Check data files
echo -e "${BLUE}[INFO]${NC} Checking for data files..."

data_files_exist=0
if [ -f "data/raw/mail.csv" ]; then
    echo -e "${GREEN}✅ mail.csv found${NC}"
    ((data_files_exist++))
else
    echo -e "${YELLOW}⚠️  data/raw/mail.csv not found${NC}"
fi

if [ -f "data/raw/call_intents.csv" ]; then
    echo -e "${GREEN}✅ call_intents.csv found${NC}"
    ((data_files_exist++))
else
    echo -e "${YELLOW}⚠️  data/raw/call_intents.csv not found${NC}"
fi

if [ -f "data/raw/call_volume.csv" ]; then
    echo -e "${GREEN}✅ call_volume.csv found${NC}"
    ((data_files_exist++))
else
    echo -e "${YELLOW}⚠️  data/raw/call_volume.csv not found${NC}"
fi

if [ $data_files_exist -eq 0 ]; then
    echo -e "${RED}❌ No data files found. Please add your data files to data/raw/ directory${NC}"
    echo "Required files:"
    echo "  - data/raw/mail.csv (mail_date, mail_type, mail_volume)"
    echo "  - data/raw/call_intents.csv (ConversationStart, uui_Intent)"
    echo "  - data/raw/call_volume.csv (Date, call_volume)"
    exit 1
fi

# Check and install packages
echo
echo -e "${CYAN}════════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  CHECKING & INSTALLING PACKAGES${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════════════════════${NC}"

# Required packages
packages=("pandas" "numpy" "matplotlib" "seaborn" "scikit-learn" "pyyaml")

echo -e "${BLUE}[INFO]${NC} Checking required packages..."

missing_packages=()
for package in "${packages[@]}"; do
    # Special case for scikit-learn
    import_name="$package"
    if [ "$package" == "scikit-learn" ]; then
        import_name="sklearn"
    elif [ "$package" == "pyyaml" ]; then
        import_name="yaml"
    fi
    
    if $PYTHON_CMD -c "import $import_name" 2>/dev/null; then
        echo -e "${GREEN}✅ $package${NC}"
    else
        echo -e "${YELLOW}⚠️  $package missing${NC}"
        missing_packages+=("$package")
    fi
done

# Install missing packages
if [ ${#missing_packages[@]} -gt 0 ]; then
    echo
    echo -e "${BLUE}[INFO]${NC} Installing missing packages: ${missing_packages[*]}"
    
    # Upgrade pip first
    $PYTHON_CMD -m pip install --upgrade pip --quiet 2>/dev/null || echo -e "${YELLOW}⚠️  pip upgrade failed, continuing...${NC}"
    
    for package in "${missing_packages[@]}"; do
        echo -e "${BLUE}[INFO]${NC} Installing $package..."
        
        if $PYTHON_CMD -m pip install "$package" --quiet 2>/dev/null; then
            echo -e "${GREEN}✅ $package installed${NC}"
        elif $PYTHON_CMD -m pip install "$package" --user --quiet 2>/dev/null; then
            echo -e "${GREEN}✅ $package installed with --user${NC}"
        else
            echo -e "${YELLOW}⚠️  $package installation failed, will try to continue${NC}"
        fi
    done
else
    echo -e "${GREEN}✅ All required packages are available${NC}"
fi

# Final package verification
echo
echo -e "${BLUE}[INFO]${NC} Final package verification..."
working_packages=0
for package in "${packages[@]}"; do
    import_name="$package"
    if [ "$package" == "scikit-learn" ]; then
        import_name="sklearn"
    elif [ "$package" == "pyyaml" ]; then
        import_name="yaml"
    fi
    
    if $PYTHON_CMD -c "import $import_name" 2>/dev/null; then
        ((working_packages++))
    fi
done

echo -e "${GREEN}✅ $working_packages/${#packages[@]} packages working${NC}"

if [ $working_packages -lt 4 ]; then
    echo -e "${RED}❌ Too many packages missing. Analytics may not work properly.${NC}"
    echo "Please install packages manually: pip install pandas numpy matplotlib pyyaml"
    exit 1
fi

# Run analytics
echo
echo -e "${CYAN}════════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  RUNNING ANALYTICS PIPELINE${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════════════════════${NC}"

echo -e "${BLUE}[INFO]${NC} Starting analytics pipeline..."
echo

# Ensure output directories exist
mkdir -p outputs/{plots,reports} logs

# Run the main analytics
if $PYTHON_CMD src/main.py; then
    echo
    echo -e "${GREEN}🎉 ANALYTICS COMPLETED SUCCESSFULLY!${NC}"
    echo
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                           RESULTS SUMMARY                                   ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo
    
    echo -e "${GREEN}📊 Generated Files:${NC}"
    
    # Check for generated plots
    if [ -d "outputs/plots" ]; then
        plot_count=$(find outputs/plots -name "*.png" 2>/dev/null | wc -l)
        if [ $plot_count -gt 0 ]; then
            echo -e "  ${GREEN}✅ Plots: $plot_count PNG files in outputs/plots/${NC}"
            find outputs/plots -name "*.png" 2>/dev/null | head -5 | while read file; do
                echo -e "     • $file"
            done
            if [ $plot_count -gt 5 ]; then
                echo -e "     • ... and $(($plot_count - 5)) more"
            fi
        fi
    fi
    
    # Check for generated reports
    if [ -d "outputs/reports" ]; then
        report_count=$(find outputs/reports -name "*.json" -o -name "*.txt" 2>/dev/null | wc -l)
        if [ $report_count -gt 0 ]; then
            echo -e "  ${GREEN}✅ Reports: $report_count files in outputs/reports/${NC}"
            find outputs/reports -name "*.json" -o -name "*.txt" 2>/dev/null | while read file; do
                echo -e "     • $file"
            done
        fi
    fi
    
    # Check logs
    if [ -f "logs/analytics.log" ]; then
        echo -e "  ${GREEN}✅ Logs: logs/analytics.log${NC}"
    fi
    
    echo
    echo -e "${BLUE}🎯 Next Steps:${NC}"
    echo -e "  • Open outputs/plots/ to view visualizations"
    echo -e "  • Review outputs/reports/ for analysis results"
    echo -e "  • Check logs/analytics.log for detailed execution info"
    
else
    echo
    echo -e "${RED}❌ ANALYTICS FAILED${NC}"
    echo
    echo -e "${YELLOW}Troubleshooting:${NC}"
    echo -e "  • Check logs/analytics.log for detailed errors"
    echo -e "  • Verify your data files have the correct format"
    echo -e "  • Ensure all required columns are present"
    echo
    echo -e "${BLUE}Expected data format:${NC}"
    echo -e "  • mail.csv: mail_date, mail_type, mail_volume"
    echo -e "  • call_intents.csv: ConversationStart, uui_Intent"
    echo -e "  • call_volume.csv: Date, call_volume"
    
    exit 1
fi

echo
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                        RUNNER COMPLETE                                      ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"

# Keep terminal open on Windows
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    echo
    read -p "Press Enter to exit..." || true
fi
