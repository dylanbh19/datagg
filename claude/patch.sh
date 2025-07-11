#!/bin/bash

# ================================================================================
# MAIL DATA LOGIC PATCH SCRIPT
# ================================================================================
# This script patches the analytics project to handle a single mail file
# containing both mail types and volumes instead of separate files
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

print_header() {
    echo -e "${BLUE}"
    cat << "EOF"
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    MAIL DATA LOGIC PATCH SCRIPT                             ║
║                                                                              ║
║  This script modifies the analytics project to handle a single mail file    ║
║  containing both mail types and volumes in one CSV file.                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

check_project_structure() {
    print_info "Checking project structure..."
    
    # Check if we're in the right directory
    if [ ! -f "config/config.yaml" ]; then
        print_error "config/config.yaml not found. Are you in the project root directory?"
        print_info "Please run this script from the same directory where you ran setup_analytics_project.sh"
        exit 1
    fi
    
    if [ ! -f "src/data/data_loader.py" ]; then
        print_error "src/data/data_loader.py not found. Please run setup_analytics_project.sh first."
        exit 1
    fi
    
    print_success "Project structure validated"
}

patch_config_file() {
    print_info "Patching configuration file with your column names..."
    
    # Create backup
    cp config/config.yaml config/config.yaml.backup
    
    # Create new config with correct column names
    python3 << 'EOF'
import yaml

# Read the original config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Update data section with correct file paths and column names
config['data'] = {
    'call_volume': {
        'file_path': "data/raw/call_volume.csv",
        'date_column': "Date",
        'volume_column': "call_volume"
    },
    'call_intents': {
        'file_path': "data/raw/call_intents.csv", 
        'date_column': "ConversationStart",
        'intent_column': "uui_Intent",
        'volume_column': "intent_volume"
    },
    'mail_data': {
        'file_path': "data/raw/mail.csv",
        'date_column': "mail_date",
        'type_column': "mail_type", 
        'volume_column': "mail_volume"
    }
}

# Write back to file
with open('config/config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, indent=2)
EOF
    
    print_success "Configuration file patched with your column names"
}

patch_data_loader() {
    print_info "Patching data loader..."
    
    # Create backup
    cp src/data/data_loader.py src/data/data_loader.py.backup
    
    # Replace the load_mail_data method
    cat > temp_mail_loader.py << 'EOF'
    def load_mail_data(self) -> pd.DataFrame:
        """Load combined mail data with types and volumes"""
        logger.info("Loading mail data...")
        
        # Load combined mail data
        mail_config = self.config['data']['mail_data']
        mail_df = pd.read_csv(mail_config['file_path'])
        mail_df[mail_config['date_column']] = pd.to_datetime(mail_df[mail_config['date_column']])
        
        logger.info(f"Loaded {len(mail_df)} mail records")
        
        return mail_df
EOF

    # Replace the load_mail_data method in the file
    python3 << 'EOF'
import re

# Read the original file
with open('src/data/data_loader.py', 'r') as f:
    content = f.read()

# Replace the load_mail_data method
new_method = '''    def load_mail_data(self) -> pd.DataFrame:
        """Load combined mail data with types and volumes"""
        logger.info("Loading mail data...")
        
        # Load combined mail data
        mail_config = self.config['data']['mail_data']
        mail_df = pd.read_csv(mail_config['file_path'])
        mail_df[mail_config['date_column']] = pd.to_datetime(mail_df[mail_config['date_column']])
        
        logger.info(f"Loaded {len(mail_df)} mail records")
        
        return mail_df'''

# Use regex to replace the method
pattern = r'    def load_mail_data\(self\).*?return mail_volume_df, mail_types_df'
content = re.sub(pattern, new_method, content, flags=re.DOTALL)

# Also fix the return type annotation
content = content.replace('-> Tuple[pd.DataFrame, pd.DataFrame]:', '-> pd.DataFrame:')

# Write back to file
with open('src/data/data_loader.py', 'w') as f:
    f.write(content)
EOF

    print_success "Data loader patched"
}

patch_main_script() {
    print_info "Patching main execution script for your data..."
    
    # Create backup
    cp src/main.py src/main.py.backup
    
    # Update the main script
    python3 << 'EOF'
import re

# Read the original file
with open('src/main.py', 'r') as f:
    content = f.read()

# Replace mail data loading section
old_pattern = r'        # Load mail data\n        mail_volume_df, mail_types_df = data_loader\.load_mail_data\(\)'
new_code = '''        # Load mail data
        mail_df = data_loader.load_mail_data()
        
        # Create aggregated mail volume for time series analysis
        mail_volume_df = mail_df.groupby('mail_date')['mail_volume'].sum().reset_index()
        mail_volume_df.rename(columns={'mail_date': 'date'}, inplace=True)'''

content = re.sub(old_pattern, new_code, content)

# Update visualization calls to use mail_df instead of mail_types_df
content = content.replace('mail_types_df, call_volume_df', 'mail_df, call_volume_df')
content = content.replace('mail_types_df,', 'mail_df,')

# Write back to file
with open('src/main.py', 'w') as f:
    f.write(content)
EOF

    print_success "Main script patched for your data"
}

patch_visualization() {
    print_info "Patching visualization module..."
    
    # Create backup
    cp src/visualization/plots.py src/visualization/plots.py.backup
    
    # Update the visualization function signature
    python3 << 'EOF'
import re

# Read the original file
with open('src/visualization/plots.py', 'r') as f:
    content = f.read()

# Update the function signature and internal logic
old_signature = r'def plot_mail_type_effectiveness\(self, mail_types_df: pd\.DataFrame,'
new_signature = 'def plot_mail_type_effectiveness(self, mail_df: pd.DataFrame,'

content = re.sub(old_signature, new_signature, content)

# Update the function body to work with combined data
old_aggregation = r'mail_summary = mail_types_df\.groupby\(\'mail_type\'\)\[\'mail_volume\'\]\.agg\(\[\'sum\', \'count\', \'mean\'\]\)\.reset_index\(\)'
new_aggregation = 'mail_summary = mail_df.groupby(\'mail_type\')[\'mail_volume\'].agg([\'sum\', \'count\', \'mean\']).reset_index()'

content = re.sub(old_aggregation, new_aggregation, content)

# Write back to file
with open('src/visualization/plots.py', 'w') as f:
    f.write(content)
EOF

    print_success "Visualization module patched"
}

create_sample_mail_data() {
    print_info "Removing sample data files..."
    
    # Remove sample data files since we're using real data
    rm -f data/raw/sample_*.csv
    
    print_success "Sample data files removed - ready for your real data"
}

update_quick_start_script() {
    print_info "Updating run scripts for your data..."
    
    # Update the quick_start.sh to not copy sample files
    if [ -f "quick_start.sh" ]; then
        cp quick_start.sh quick_start.sh.backup
        
        # Replace the entire script to just check for real data
        cat > quick_start.sh << 'EOF'
#!/bin/bash

# ================================================================================
# DATA VALIDATION AND ANALYSIS SCRIPT
# ================================================================================

set -euo pipefail

echo "Checking for required data files..."

# Check if real data files exist
if [ ! -f "data/raw/mail.csv" ]; then
    echo "❌ data/raw/mail.csv not found"
    echo "Please add your mail data file with columns: mail_date, mail_volume, mail_type"
    exit 1
fi

if [ ! -f "data/raw/call_intents.csv" ]; then
    echo "❌ data/raw/call_intents.csv not found" 
    echo "Please add your call intents file with columns: ConversationStart, uui_Intent"
    exit 1
fi

if [ ! -f "data/raw/call_volume.csv" ]; then
    echo "❌ data/raw/call_volume.csv not found"
    echo "Please add your call volume file with column: Date"
    exit 1
fi

echo "✅ All data files found. Running analysis..."

# Run the analysis
./run_analysis.sh
EOF
        chmod +x quick_start.sh
        
        print_success "Quick start script updated for real data validation"
    fi
}-14,promotional,440
2023-01-14,statement,280
2023-01-14,reminder,170
EOF

    print_success "Sample mail data created"
}

update_quick_start_script() {
    print_info "Updating quick start script..."
    
    # Update the quick_start.sh script to use the new file
    if [ -f "quick_start.sh" ]; then
        # Create backup
        cp quick_start.sh quick_start.sh.backup
        
        # Update the script
        sed -i 's/cp data\/raw\/sample_mail_volume\.csv data\/raw\/mail_volume\.csv/cp data\/raw\/sample_mail_data.csv data\/raw\/mail_data.csv/' quick_start.sh
        sed -i 's/cp data\/raw\/sample_mail_types\.csv data\/raw\/mail_types\.csv/# Mail data now combined in single file/' quick_start.sh
        
        print_success "Quick start script updated"
    fi
}

update_documentation() {
    print_info "Updating documentation for your data format..."
    
    # Update README.md
    if [ -f "README.md" ]; then
        cp README.md README.md.backup
        
        # Update data requirements section
        sed -i 's/- \*\*Mail Types Data\*\*: Date, mail type, volume by type/- **Mail Data**: mail_date, mail_type, mail_volume (combined in single file)/' README.md
        sed -i '/- \*\*Mail Volume Data\*\*: Date, total mail volume/d' README.md
        
        print_success "Documentation updated"
    fi
}

create_data_format_guide() {
    print_info "Creating data format guide for your column names..."
    
    cat > config/your_data_format.yaml << 'EOF'
# ================================================================================
# YOUR DATA FORMAT GUIDE
# ================================================================================

# Expected format for your actual data files:

mail_data:
  file_name: "mail.csv"
  columns:
    - mail_date    # Your date column
    - mail_type    # Mail type categories 
    - mail_volume  # Volume per type
  
call_intents_data:
  file_name: "call_intents.csv" 
  columns:
    - ConversationStart  # Your date column
    - uui_Intent        # Intent categories
    # Note: intent_volume will be calculated by counting rows per date/intent
    
call_volume_data:
  file_name: "call_volume.csv"
  columns:
    - Date         # Your date column
    - call_volume  # Total call volume (may have missing values)

# File placement:
# Place these files in data/raw/ directory:
# - data/raw/mail.csv
# - data/raw/call_intents.csv  
# - data/raw/call_volume.csv
EOF

    print_success "Data format guide created for your column names"
}

patch_call_intents_logic() {
    print_info "Patching call intents logic for your data structure..."
    
    # Update data loader to handle intent counting
    python3 << 'EOF'
# Read the data loader file
with open('src/data/data_loader.py', 'r') as f:
    content = f.read()

# Add intent volume calculation logic
new_intent_method = '''    def load_call_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        
        return call_volume_df, intent_counts'''

# Replace the load_call_data method
import re
pattern = r'    def load_call_data\(self\).*?return call_volume_df, call_intents_df'
content = re.sub(pattern, new_intent_method, content, flags=re.DOTALL)

# Write back to file
with open('src/data/data_loader.py', 'w') as f:
    f.write(content)
EOF

    print_success "Call intents logic patched to count occurrences"
}

run_validation() {
    print_info "Running validation checks for your data format..."
    
    # Check if all files were patched correctly
    local errors=0
    
    # Check config file
    if ! grep -q "mail_data:" config/config.yaml; then
        print_error "Config file patch failed"
        ((errors++))
    fi
    
    # Check if correct column names are in config
    if ! grep -q "mail_date" config/config.yaml; then
        print_error "Mail date column not updated in config"
        ((errors++))
    fi
    
    if ! grep -q "ConversationStart" config/config.yaml; then
        print_error "Call intent date column not updated in config"
        ((errors++))
    fi
    
    # Check data loader
    if ! grep -q "Load combined mail data" src/data/data_loader.py; then
        print_error "Data loader patch failed"
        ((errors++))
    fi
    
    # Check main script
    if ! grep -q "mail_df = data_loader.load_mail_data()" src/main.py; then
        print_error "Main script patch failed"
        ((errors++))
    fi
    
    if [ $errors -eq 0 ]; then
        print_success "All patches applied successfully for your data format"
        return 0
    else
        print_error "$errors patch(es) failed"
        return 1
    fi
}

display_completion_message() {
    echo -e "${GREEN}"
    cat << "EOF"

╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                       🎉 PATCH APPLIED SUCCESSFULLY! 🎉                      ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  UPDATES FOR YOUR DATA FORMAT:                                               ║
║                                                                              ║
║  ✓ Configuration updated for your column names                              ║
║  ✓ Data loader modified for combined mail data                              ║
║  ✓ Call intents logic updated to count occurrences                          ║
║  ✓ Main script updated to handle your format                                ║
║  ✓ Visualization functions patched                                           ║
║  ✓ Sample data removed - ready for your real data                           ║
║                                                                              ║
║  PLACE YOUR DATA FILES:                                                      ║
║  📁 data/raw/mail.csv           (mail_date, mail_type, mail_volume)          ║
║  📁 data/raw/call_intents.csv   (ConversationStart, uui_Intent)              ║
║  📁 data/raw/call_volume.csv    (Date, call_volume)                          ║
║                                                                              ║
║  🔍 The system will automatically count intent occurrences per date         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

EOF
    echo -e "${NC}"
    
    print_info "Next steps:"
    echo "1. Place your 3 data files in data/raw/ directory:"
    echo "   - mail.csv (mail_date, mail_type, mail_volume)" 
    echo "   - call_intents.csv (ConversationStart, uui_Intent)"
    echo "   - call_volume.csv (Date, call_volume)"
    echo "2. Run validation check: ./quick_start.sh"
    echo "3. Run full analysis: ./run_analysis.sh"
    echo ""
    echo -e "${YELLOW}Your data format guide is in: config/your_data_format.yaml${NC}"
    echo -e "${YELLOW}Backup files created with .backup extension if you need to revert${NC}"
}

main() {
    print_header
    
    check_project_structure
    patch_config_file
    patch_data_loader
    patch_call_intents_logic
    patch_main_script
    patch_visualization
    create_sample_mail_data
    update_quick_start_script
    update_documentation
    create_data_format_guide
    
    if run_validation; then
        display_completion_message
    else
        print_error "Some patches failed. Check the output above for details."
        exit 1
    fi
}

# Execute main function
main "$@"
