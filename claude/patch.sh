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
    print_info "Patching configuration file..."
    
    # Create backup
    cp config/config.yaml config/config.yaml.backup
    
    # Replace mail configuration section
    cat > temp_mail_config.yaml << 'EOF'
  # Mail data (combined types and volumes in single file)
  mail_data:
    file_path: "data/raw/mail_data.csv"
    date_column: "date"              # Column containing dates
    type_column: "mail_type"         # Column containing mail type categories (strings)
    volume_column: "mail_volume"     # Column containing volume per type (integers)
EOF

    # Use sed to replace the mail_volume and mail_types sections
    sed '/^  # Mail volume data/,/^  # Mail types data/c\
  # Mail data (combined types and volumes in single file)\
  mail_data:\
    file_path: "data/raw/mail_data.csv"\
    date_column: "date"              # Column containing dates\
    type_column: "mail_type"         # Column containing mail type categories (strings)\
    volume_column: "mail_volume"     # Column containing volume per type (integers)' config/config.yaml > config/config_temp.yaml
    
    # Remove the remaining mail_types section
    sed '/^  # Mail types data/,/^$/d' config/config_temp.yaml > config/config.yaml
    rm config/config_temp.yaml
    
    print_success "Configuration file patched"
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
    print_info "Patching main execution script..."
    
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
        mail_volume_df = mail_df.groupby('date')['mail_volume'].sum().reset_index()'''

content = re.sub(old_pattern, new_code, content)

# Update visualization calls
content = content.replace('mail_types_df, call_volume_df', 'mail_df, call_volume_df')
content = content.replace('mail_types_df,', 'mail_df,')

# Write back to file
with open('src/main.py', 'w') as f:
    f.write(content)
EOF

    print_success "Main script patched"
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
    print_info "Creating sample mail data file..."
    
    # Create the combined sample mail data
    cat > data/raw/sample_mail_data.csv << 'EOF'
date,mail_type,mail_volume
2023-01-01,promotional,500
2023-01-01,statement,300
2023-01-01,reminder,200
2023-01-02,promotional,600
2023-01-02,statement,350
2023-01-02,reminder,250
2023-01-03,promotional,450
2023-01-03,statement,300
2023-01-03,reminder,200
2023-01-04,promotional,550
2023-01-04,statement,320
2023-01-04,reminder,230
2023-01-05,promotional,650
2023-01-05,statement,400
2023-01-05,reminder,250
2023-01-06,promotional,400
2023-01-06,statement,250
2023-01-06,reminder,150
2023-01-07,promotional,380
2023-01-07,statement,220
2023-01-07,reminder,150
2023-01-08,promotional,520
2023-01-08,statement,330
2023-01-08,reminder,200
2023-01-09,promotional,580
2023-01-09,statement,370
2023-01-09,reminder,200
2023-01-10,promotional,620
2023-01-10,statement,380
2023-01-10,reminder,250
2023-01-11,promotional,540
2023-01-11,statement,340
2023-01-11,reminder,200
2023-01-12,promotional,610
2023-01-12,statement,390
2023-01-12,reminder,220
2023-01-13,promotional,590
2023-01-13,statement,370
2023-01-13,reminder,220
2023-01-14,promotional,440
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
    print_info "Updating documentation..."
    
    # Update README.md
    if [ -f "README.md" ]; then
        cp README.md README.md.backup
        
        # Update data requirements section
        sed -i 's/- \*\*Mail Types Data\*\*: Date, mail type, volume by type/- **Mail Data**: Date, mail type, volume by type (combined in single file)/' README.md
        sed -i '/- \*\*Mail Volume Data\*\*: Date, total mail volume/d' README.md
        
        print_success "Documentation updated"
    fi
}

create_data_format_guide() {
    print_info "Creating updated data format guide..."
    
    cat > config/mail_data_format.yaml << 'EOF'
# ================================================================================
# MAIL DATA FORMAT GUIDE (UPDATED)
# ================================================================================

# Expected format for the combined mail data file:

mail_data_sample:
  file_name: "mail_data.csv"
  columns:
    - date         # Format: YYYY-MM-DD
    - mail_type    # String: type of mail sent (e.g., "promotional", "statement", "reminder")
    - mail_volume  # Integer: number of mails of this type sent on this date
  
  example_rows:
    - date: "2023-01-01"
      mail_type: "promotional"
      mail_volume: 500
    - date: "2023-01-01"
      mail_type: "statement"
      mail_volume: 300
    - date: "2023-01-01"
      mail_type: "reminder"
      mail_volume: 200
    - date: "2023-01-02"
      mail_type: "promotional"
      mail_volume: 600
  
  notes:
    - "Each row represents the volume of a specific mail type sent on a specific date"
    - "Multiple rows per date are expected (one for each mail type)"
    - "Mail types should be consistent strings (avoid variations like 'promo' vs 'promotional')"
    - "Total daily mail volume will be calculated by summing all types per date"
EOF

    print_success "Data format guide created"
}

run_validation() {
    print_info "Running validation checks..."
    
    # Check if all files were patched correctly
    local errors=0
    
    # Check config file
    if ! grep -q "mail_data:" config/config.yaml; then
        print_error "Config file patch failed"
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
    
    # Check sample data
    if [ ! -f "data/raw/sample_mail_data.csv" ]; then
        print_error "Sample mail data not created"
        ((errors++))
    fi
    
    if [ $errors -eq 0 ]; then
        print_success "All patches applied successfully"
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
║  MAIL DATA CHANGES:                                                          ║
║                                                                              ║
║  ✓ Configuration updated to use single mail file                            ║
║  ✓ Data loader modified for combined mail data                              ║
║  ✓ Main script updated to handle new format                                 ║
║  ✓ Visualization functions patched                                           ║
║  ✓ Sample data created in correct format                                     ║
║                                                                              ║
║  YOUR MAIL DATA FILE SHOULD NOW HAVE:                                       ║
║  - date column (YYYY-MM-DD format)                                          ║
║  - mail_type column (string categories)                                      ║
║  - mail_volume column (integer volumes)                                      ║
║                                                                              ║
║  Multiple rows per date are expected (one per mail type).                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

EOF
    echo -e "${NC}"
    
    print_info "Next steps:"
    echo "1. Place your mail data file at: data/raw/mail_data.csv"
    echo "2. Ensure it follows the format in: config/mail_data_format.yaml"
    echo "3. Run the analysis: ./run_analysis.sh"
    echo "4. Or test with sample data: ./quick_start.sh"
    echo ""
    echo -e "${YELLOW}Backup files created with .backup extension if you need to revert${NC}"
}

main() {
    print_header
    
    check_project_structure
    patch_config_file
    patch_data_loader
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
