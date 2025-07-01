# =============================================================================

# ENHANCED CONFIGURATION SECTION - MODIFY THESE PATHS AND SETTINGS

# =============================================================================

CONFIG = {
# === FILE PATHS ===
‘MAIL_FILE_PATH’: r’C:\path\to\your\mail_data.csv’,        # UPDATE THIS PATH
‘CALL_FILE_PATH’: r’C:\path\to\your\call_data.csv’,        # UPDATE THIS PATH
‘OUTPUT_DIR’: r’C:\path\to\output\results’,                # UPDATE THIS PATH

```
# === MAIL DATA COLUMN MAPPING ===
'MAIL_COLUMNS': {
    'date': 'date',              # Date column name in your mail file
    'volume': 'volume',          # Volume/quantity column name
    'type': 'mail_type',         # Mail type column name (for legend)
    'source': 'source'           # Source column (optional)
},

# === CALL DATA COLUMN MAPPING ===
# If your call data has individual call records (one row per call):
'CALL_COLUMNS': {
    'date': 'call_date',         # Date column name in your call file
    'call_id': 'call_id',        # Call ID or unique identifier (optional)
    'phone': 'phone_number',     # Phone number column (optional)
    'duration': 'duration',      # Call duration column (optional)
    'type': 'call_type'          # Call type column (optional)
},

# === CALL DATA AGGREGATION SETTINGS ===
'CALL_AGGREGATION': {
    'method': 'count',           # 'count' = count rows per day, 'sum' = sum a specific column
    'sum_column': None,          # If method='sum', which column to sum (e.g., 'duration')
    'group_by_type': False,      # Whether to also group by call type
},

# === DATE PROCESSING SETTINGS ===
'DATE_PROCESSING': {
    'standardize_mail_dates': True,     # Add 00:00:00 time to date-only entries
    'standardize_call_dates': True,     # Add 00:00:00 time to date-only entries
    'date_format': None,                # e.g., '%Y-%m-%d' or None for auto-detection
    'time_format': '%H:%M:%S',          # Time format for standardization
},

# === ANALYSIS SETTINGS ===
'REMOVE_OUTLIERS': True,         # Remove outliers from analysis
'MAX_LAG_DAYS': 21,             # Maximum lag days to test
'MIN_OVERLAP_RECORDS': 10,       # Minimum overlapping records needed
'MAX_RESPONSE_RATE': 50,         # Maximum realistic response rate (%)

# === PLOT SETTINGS ===
'PLOT_STYLE': 'seaborn-v0_8',   # Matplotlib style
'FIGURE_SIZE': (15, 10),        # Default figure size
'DPI': 300,                     # Plot resolution
'FONT_SIZE': 12,                # Default font size
```

}

# =============================================================================

# ENHANCED DATA PROCESSING FUNCTIONS

# =============================================================================

def standardize_dates(data, date_column, logger):
“””
Standardize dates by adding 00:00:00 time to date-only entries
“””
logger.info(f”Standardizing dates in column: {date_column}”)

```
# Convert to string first to check formats
date_strings = data[date_column].astype(str)

# Count different date formats
date_only_pattern = r'^\d{4}-\d{2}-\d{2}$'  # YYYY-MM-DD
datetime_pattern = r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'  # YYYY-MM-DD HH:MM:SS

date_only_count = date_strings.str.match(date_only_pattern).sum()
datetime_count = date_strings.str.match(datetime_pattern).sum()

logger.info(f"  Date-only entries: {date_only_count}")
logger.info(f"  DateTime entries: {datetime_count}")

# Standardize dates
standardized_dates = []
for date_str in date_strings:
    try:
        # If it's just a date, add time
        if pd.to_datetime(date_str).time() == pd.to_datetime('00:00:00').time():
            # Already has time or is date-only
            dt = pd.to_datetime(date_str)
            if dt.time() == pd.to_datetime('00:00:00').time() and ' ' not in str(date_str):
                # Date only, add time
                standardized_dates.append(dt.strftime('%Y-%m-%d 00:00:00'))
            else:
                # Already has time
                standardized_dates.append(str(date_str))
        else:
            standardized_dates.append(str(date_str))
    except:
        standardized_dates.append(str(date_str))

# Convert back to datetime
data[date_column] = pd.to_datetime(standardized_dates, errors='coerce')

logger.info(f"✓ Date standardization completed")
return data
```

def aggregate_call_data(call_data, config, logger):
“””
Aggregate call data by date to get volume per day
“””
logger.info(“Aggregating call data by date…”)

```
# Get aggregation settings
agg_config = config['CALL_AGGREGATION']
method = agg_config['method']
sum_column = agg_config['sum_column']
group_by_type = agg_config['group_by_type']

logger.info(f"  Aggregation method: {method}")
logger.info(f"  Group by type: {group_by_type}")

# Prepare grouping columns
group_cols = ['date']
if group_by_type and 'type' in call_data.columns:
    group_cols.append('type')
    logger.info("  Including call type in grouping")

# Perform aggregation
if method == 'count':
    # Count number of calls per day
    if group_by_type and 'type' in call_data.columns:
        agg_data = call_data.groupby(group_cols).size().reset_index(name='volume')
        # Also create a total volume per date
        total_agg = call_data.groupby('date').size().reset_index(name='volume')
    else:
        agg_data = call_data.groupby('date').size().reset_index(name='volume')
        total_agg = agg_data.copy()
        
elif method == 'sum' and sum_column and sum_column in call_data.columns:
    # Sum a specific column per day
    if group_by_type and 'type' in call_data.columns:
        agg_data = call_data.groupby(group_cols)[sum_column].sum().reset_index()
        agg_data.columns = group_cols + ['volume']
        total_agg = call_data.groupby('date')[sum_column].sum().reset_index(name='volume')
    else:
        agg_data = call_data.groupby('date')[sum_column].sum().reset_index(name='volume')
        total_agg = agg_data.copy()
        
    logger.info(f"  Summing column: {sum_column}")
else:
    # Default to count
    logger.warning(f"Invalid aggregation method or column, defaulting to count")
    agg_data = call_data.groupby('date').size().reset_index(name='volume')
    total_agg = agg_data.copy()

logger.info(f"✓ Call data aggregated: {len(total_agg)} unique dates")
logger.info(f"  Total call volume: {total_agg['volume'].sum():,}")
logger.info(f"  Average daily volume: {total_agg['volume'].mean():.1f}")

# Store both aggregated data
return total_agg, agg_data if group_by_type and 'type' in call_data.columns else None
```

# =============================================================================

# ENHANCED PROCESSING METHODS FOR CallVolumeAnalyzer CLASS

# =============================================================================

# Replace the _process_data method with this enhanced version:

def _process_data(self) -> bool:
“”“Enhanced process and validate both datasets”””

```
self.logger.info("\n" + "=" * 60)
self.logger.info("STEP 2: ENHANCED DATA PROCESSING")
self.logger.info("=" * 60)

try:
    # Process mail data
    self.logger.info("Processing mail data...")
    
    # Map mail columns
    mail_mapping = self._map_columns(self.mail_data, self.config['MAIL_COLUMNS'], 'mail')
    if mail_mapping:
        self.mail_data = self.mail_data.rename(columns=mail_mapping)
    
    # Validate required columns
    if 'date' not in self.mail_data.columns:
        raise ValueError("Date column not found in mail data")
    if 'volume' not in self.mail_data.columns:
        raise ValueError("Volume column not found in mail data")
    
    # Standardize mail dates
    if self.config['DATE_PROCESSING']['standardize_mail_dates']:
        self.mail_data = standardize_dates(self.mail_data, 'date', self.logger)
    
    # Process dates and volumes
    self.mail_data = self._process_dates_and_volumes(self.mail_data, 'mail')
    
    # Aggregate mail data
    self.mail_data_agg = self._aggregate_mail_data()
    
    # Process call data
    self.logger.info("Processing call data...")
    
    # Map call columns - enhanced mapping
    call_mapping = self._map_call_columns()
    if call_mapping:
        self.call_data = self.call_data.rename(columns=call_mapping)
    
    # Validate required columns
    if 'date' not in self.call_data.columns:
        raise ValueError("Date column not found in call data")
    
    # Standardize call dates
    if self.config['DATE_PROCESSING']['standardize_call_dates']:
        self.call_data = standardize_dates(self.call_data, 'date', self.logger)
    
    # Process dates
    self.call_data = self._process_call_dates_and_data()
    
    # Aggregate call data by date
    self.call_data_agg, self.call_data_by_type = aggregate_call_data(
        self.call_data, self.config, self.logger
    )
    
    # Log results
    self.logger.info(f"✓ Mail data processed: {len(self.mail_data_agg):,} unique dates")
    self.logger.info(f"  Date range: {self.mail_data_agg['date'].min()} to {self.mail_data_agg['date'].max()}")
    self.logger.info(f"  Total volume: {self.mail_data_agg['volume'].sum():,}")
    
    self.logger.info(f"✓ Call data processed: {len(self.call_data_agg):,} unique dates")
    self.logger.info(f"  Date range: {self.call_data_agg['date'].min()} to {self.call_data_agg['date'].max()}")
    self.logger.info(f"  Total volume: {self.call_data_agg['volume'].sum():,}")
    
    return True
    
except Exception as e:
    self.logger.error(f"✗ Error processing data: {str(e)}")
    return False
```

def _map_call_columns(self) -> Dict:
“”“Enhanced call column mapping”””

```
mapping = {}
call_config = self.config['CALL_COLUMNS']

# Find date column
date_col = call_config.get('date', 'date')
if date_col in self.call_data.columns:
    if date_col != 'date':
        mapping[date_col] = 'date'
else:
    # Try fuzzy matching for date
    date_candidates = [col for col in self.call_data.columns 
                      if any(word in col.lower() for word in ['date', 'time', 'created', 'timestamp'])]
    if date_candidates:
        mapping[date_candidates[0]] = 'date'
        self.logger.warning(f"Using '{date_candidates[0]}' as date column for call data")
    else:
        self.logger.error("No date column found in call data")

# Map other columns if they exist
for standard_name, config_name in call_config.items():
    if standard_name != 'date' and config_name in self.call_data.columns:
        if config_name != standard_name:
            mapping[config_name] = standard_name

if mapping:
    self.logger.info(f"Call data column mapping: {mapping}")

return mapping
```

def _process_call_dates_and_data(self) -> pd.DataFrame:
“”“Enhanced call data processing”””

```
data = self.call_data.copy()

# Convert dates
if self.config['DATE_PROCESSING']['date_format']:
    data['date'] = pd.to_datetime(data['date'], format=self.config['DATE_PROCESSING']['date_format'], errors='coerce')
else:
    data['date'] = pd.to_datetime(data['date'], errors='coerce')

# Check for invalid dates
invalid_dates = data['date'].isnull().sum()
if invalid_dates > 0:
    self.logger.warning(f"Found {invalid_dates:,} invalid dates in call data")
    data = data.dropna(subset=['date'])

# Extract date only (remove time for daily aggregation)
data['date'] = data['date'].dt.date
data['date'] = pd.to_datetime(data['date'])

self.logger.info(f"Call data date processing completed: {len(data):,} records")

return data
```

# =============================================================================

# USAGE INSTRUCTIONS

# =============================================================================

“””
CONFIGURATION INSTRUCTIONS:

1. CALL DATA COLUMN MAPPING:
   Update ‘CALL_COLUMNS’ in CONFIG to match your actual column names:
   
   ‘CALL_COLUMNS’: {
   ‘date’: ‘your_actual_date_column_name’,    # e.g., ‘call_timestamp’, ‘created_date’
   ‘call_id’: ‘your_call_id_column’,          # Optional: unique call identifier
   ‘phone’: ‘your_phone_column’,              # Optional: phone number
   ‘duration’: ‘your_duration_column’,        # Optional: call duration
   ‘type’: ‘your_call_type_column’            # Optional: call type/category
   }
1. CALL AGGREGATION SETTINGS:
   Configure how to aggregate individual call records into daily volumes:
   
   ‘CALL_AGGREGATION’: {
   ‘method’: ‘count’,        # ‘count’ = count calls per day
   # ‘sum’ = sum a column per day (e.g., total duration)
   ‘sum_column’: None,       # If method=‘sum’, specify column name
   ‘group_by_type’: False,   # True = also analyze by call type
   }
1. DATE STANDARDIZATION:
   Automatically handles mixed date formats in your data:
   
   ‘DATE_PROCESSING’: {
   ‘standardize_mail_dates’: True,    # Add 00:00:00 to date-only entries
   ‘standardize_call_dates’: True,    # Add 00:00:00 to date-only entries
   }

EXAMPLE CONFIGURATIONS:

# If your call data has individual call records:

‘CALL_COLUMNS’: {
‘date’: ‘call_timestamp’,
‘call_id’: ‘unique_id’,
‘phone’: ‘customer_phone’,
‘duration’: ‘call_duration_minutes’,
‘type’: ‘call_category’
}

‘CALL_AGGREGATION’: {
‘method’: ‘count’,           # Count number of calls per day
‘sum_column’: None,
‘group_by_type’: True,       # Also analyze by call type
}

# If you want to sum call durations instead of counting calls:

‘CALL_AGGREGATION’: {
‘method’: ‘sum’,             # Sum total duration per day
‘sum_column’: ‘call_duration_minutes’,
‘group_by_type’: False,
}
“””