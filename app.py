"""
Product Price Checker v4.3.21
Smart brand matching and price comparison tool for cannabis retail products
With automatic CSV type detection, shop filtering, and Blaze POS export

CHANGELOG:
v4.3.22 (2025-01-26)
- CHANGED: Default catalog price sources now Active, New Product, DNO
- Updated from Active/New Price/New Product to include DNO status by default
- Streamlines workflow by including DNO (Do Not Order) products in price comparisons

v4.3.21 (2025-12-13)
- CRITICAL FIX: Placeholder and wildcard matching now respects categories
- Previously matched across categories (Preroll → Flower templates)
- Example: "Blem - Boasy Preroll 5pk 3.5g" now correctly matches to Preroll templates only
- Prevents dangerous cross-category price mismatches

v4.3.20 (2025-12-13)
- FIXED: Troubleshooting tab now shows ALL products including unmatched
- Previously only showed matched products, missing critical debug info
- Now properly tracks and displays products that don't find matches
- Shows brand availability in catalog for unmatched products

v4.3.18 (2025-12-13)
- FIXED: Wildcard matching for complex patterns with "/" characters
- FIXED: Turn products now match "STRAIN Turn Up/Turn Down" templates
- Improved regex to handle special characters in strain names
- Added special handling for Turn Up/Turn Down pattern variations

v4.3.17 (2025-12-13)
- ADDED: Apparel matching logic with size handling
- Strips size in parentheses (e.g., "(XS)") for matching
- Matches "California Shirt (XS)" to "California Shirt" template

v4.3.16 (2025-12-13)
- CHANGED: Apparel category now INCLUDED in processing
- Previously excluded, now processes apparel products
- All other excluded categories remain filtered

v4.3.15 (2025-12-09)
- IMPROVED: Price Difference Type filter now uses multi-select
- Users can select any combination of Retail, Sale, and Cost differences
- Defaults to Retail + Sale (most common use case)
- More flexible filtering for typical workflows

v4.3.14 (2025-12-09)
- FIXED: Price Inspector now properly displays Unit Price and Unit Sale Price
- ADDED: Cost per Unit tracking from Company Products  
- ADDED: Max Unit Cost comparison from Product Catalog
- Enhanced price string to numeric conversion for all display tables

v4.3.13 (2025-11-27)
- RESTORED: Full application code (v4.3.11-12 were accidentally truncated)
- FIXED: Unit Price display by converting '$34.98' strings to numeric
- FIXED: Blaze POS headers exactly: ProductId,Retail Price,Sale Price
- All features from v4.3.10 preserved and working

v4.3.10 (2025-11-27)
- CRITICAL FIX: Unit Price now displays correctly in Streamlit dataframe
- Issue: Data exists (exports correctly) but shows as "None" in UI
- Solution: Convert price columns to strings before display
- This is a Streamlit rendering issue, not a data issue
- All price columns now display properly

v4.3.9 (2025-11-27)
- CRITICAL FIX: Unit Price columns now created if missing from source data
- Properly handles shop exports where Unit Price columns don't exist
- Fallback now works even when main price columns are completely missing
- Removed duplicate fallback logic that was running after columns were deleted

v4.3.7 (2025-11-27)
- CRITICAL FIX: Unit Price and Unit Sale Price columns now display correctly
- Fixed mixed price format issue ('49.98' vs '$49.98') causing display problems
- All prices now normalized to consistent $XX.XX format
- Enhanced Stiiizy STRAIN pattern matching from v4.3.5

v4.3.4 (2025-11-21)
- ENHANCEMENT: Added Hawthorne store mapping ('HAVEN - Hawthorne South Bay' → 'Hawthorne')
- FIX: Restored missing Price Inspector, Troubleshooting, and Product Catalog tabs
- FIX: Price Inspector formatting error - now converts columns to numeric before styling
- Total stores supported: 12

v4.3.3 (2025-11-19)
- CODE: Major cleanup and reorganization for maintainability
- CODE: Consolidated changelog, removed duplicate entries
- CODE: Improved section organization and documentation
- No functional changes - all matching logic preserved

v4.3.2 (2025-11-14)
- ENHANCEMENT: Added fallback to .5 Unit Price fields when standard fields are blank
- Automatically uses ".5 Unit Price" when "Unit Price" is empty
- Automatically uses ".5 Unit Sale Price" when "Unit Sale Price" is empty

v4.3.1 (2025-11-14)
- ENHANCEMENT: Inventory filter now supports All/In Stock/Out of Stock
- Can now filter to show ONLY out-of-stock products

v4.3.0 (2025-11-14)
- ENHANCEMENT: Added Price Difference Type filter in Price Inspector
- Can now filter by: Any, Retail Only, Sale Only, or Both Retail and Sale differences

v4.2.8 (2025-11-14)
- ENHANCEMENT: Price Inspector filters now support Include/Exclude modes
- Can now exclude specific brands/categories/templates/locations

v4.2.7 (2025-11-14)
- ENHANCEMENT: Blaze POS export now fills blank Sale Price with Retail Price
- Required for Blaze bulk updater tool compatibility

v4.2.6 (2025-11-14)
- CRITICAL FIX: Sale price comparison now checks numeric match FIRST
- Logic order: 1) Check numeric match, 2) Handle mismatches, 3) Apply sale=retail logic

v4.2.5 (2025-11-13)
- CRITICAL FIX: Changed Status filter from single to multi-select
- Default includes Active + New Price + New Product

v4.2.3 (2025-11-13)
- CRITICAL FIX: Deduplicate catalog templates to fix Stiiizy and other multi-status brands
- Added wildcard matching for COLOR, STRAIN, FLAVOR patterns

Previous versions consolidated - see git history for details
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import re
from google.oauth2.service_account import Credentials
import gspread
from gspread_dataframe import get_as_dataframe

# ============================================================================
# CONFIGURATION
# ============================================================================

# Page configuration
st.set_page_config(
    page_title="Product Price Checker v4.3.22",
    page_icon="🛒",
    layout="wide"
)

# Version and URLs
VERSION = "4.3.22"
CONNECT_CATALOG_URL = "https://docs.google.com/spreadsheets/d/1FG3K7Rj-a9xw-UegJ4yxM8DAyn1LhmxwopYn67ja5iI/edit?gid=172177068#gid=172177068"

# Shop name mapping between Company Products and Product Catalog
SHOP_NAME_MAPPING = {
    'HAVEN - Maywood': 'Maywood',
    'HAVEN - LB#1 - Los Alamitos': 'Los Alamitos', 
    'HAVEN - LB#2 - Paramount': 'Paramount',
    'HAVEN - LB#3 - Downtown LB': 'DTLB',
    'HAVEN - LB#4 - Belmont': 'Belmont',
    'HAVEN - San Bernardino': 'San Bernardino',
    'HAVEN - Porterville': 'Porterville',
    'HAVEN - Lakewood': 'Lakewood',
    'HAVEN - Orange County': 'Stanton',
    'HAVEN - Fresno': 'Fresno',
    'HAVEN - Corona': 'Corona',
    'HAVEN - Hawthorne South Bay': 'Hawthorne'
}

# Brands requiring exact product-level matching
EXACT_PRODUCT_MATCH_BRANDS = {
    'Blazy Susan', 'Camino', 'Crave', 'Daily Dose', "Dr. Norm's", 'Good Tide', 
    'Happy Fruit', 'High Gorgeous', 'Kiva', 'Lost Farm', 'Made From Dirt', 
    'Papa & Barkley', 'Sip Elixirs', 'St. Ides', "Uncle Arnie's", 'Vet CBD', 
    'Wyld', 'Yummi Karma', "Not Your Father's"
}

# Categories to exclude from processing
EXCLUDED_CATEGORIES = [
    'Display', 'Clones', 'Sample', 'Promo', 'Compassion', 
    'Donation', 'Boxes', 'Non-Cannabis', 'Gift Cards', 'xxxDONOTUSE-Buzzers'
]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def detect_csv_type(df):
    """
    Detect if CSV is a Company export (multi-shop) or Shop export (single shop)
    
    Returns:
        str: 'company' if Shop column exists, 'shop' if not, None if invalid
    """
    if df is None or df.empty:
        return None
    
    first_col = str(df.columns[0]).strip()
    
    if first_col == 'Shop':
        return 'company'
    elif first_col == 'SKU':
        return 'shop'
    else:
        return None

def get_unique_shops(df):
    """
    Extract unique shop names from company export
    
    Args:
        df: DataFrame with Shop column
        
    Returns:
        list: Sorted list of unique shop names
    """
    if df is None or 'Shop' not in df.columns:
        return []
    
    shops = df['Shop'].dropna().unique()
    return sorted([str(shop).strip() for shop in shops if str(shop).strip() and str(shop) != 'nan'])

def clean_price(price_str):
    """
    Clean and convert price string to float
    
    Args:
        price_str: Price as string, float, or None (e.g., "$12.99", "12.99", 12.99)
        
    Returns:
        float or None: Cleaned price value
    """
    if pd.isna(price_str) or price_str == '':
        return None
    try:
        cleaned = str(price_str).replace('$', '').replace(',', '').strip()
        return float(cleaned)
    except:
        return None

def clean_price_for_display(x):
    """
    Convert price strings with $ to numeric for Streamlit display
    
    Args:
        x: Price value (could be '$34.98', 34.98, None, etc.)
        
    Returns:
        float or np.nan: Numeric price value for display
    """
    if pd.isna(x) or str(x).strip() in ['', 'None', 'nan', 'NaN']:
        return np.nan
    try:
        # Remove $ and commas, convert to float
        cleaned = str(x).replace('$', '').replace(',', '').strip()
        return float(cleaned)
    except:
        return np.nan

def extract_gid_from_url(sheet_url):
    """
    Extract the gid (worksheet ID) from a Google Sheets URL
    
    Args:
        sheet_url: Google Sheets URL with optional gid parameter
        
    Returns:
        int or None: Worksheet ID if found
    """
    try:
        if 'gid=' in sheet_url:
            gid_part = sheet_url.split('gid=')[1]
            if '&' in gid_part:
                gid = gid_part.split('&')[0]
            elif '#' in gid_part:
                gid = gid_part.split('#')[0]
            else:
                gid = gid_part
            return int(gid)
    except:
        pass
    return None

# ============================================================================
# EXTRACTION FUNCTIONS - CORE MATCHING ENGINE (PRESERVED AS-IS)
# ============================================================================

def extract_weight_from_item(item_text):
    """Extract weight from item text (e.g., "Blue Dream 3.5g" → "3.5g")"""
    if pd.isna(item_text):
        return None
    
    item_str = str(item_text).strip()
    
    # Try patterns at end of string first (most common)
    end_patterns = [
        r'(\d+\.?\d*g)$',
        r'(\d+\.\d+\s?oz?)$',
        r'(\d+\s?oz?)$',
        r'(1/8\s?oz?)$',
        r'(1/4\s?oz?)$',
        r'(1/2\s?oz?)$',
    ]
    
    for pattern in end_patterns:
        match = re.search(pattern, item_str, re.IGNORECASE)
        if match:
            return match.group(1).lower().replace(' ', '')
    
    # If not found at end, try finding anywhere in string (for cases like "3.75g 5pk")
    anywhere_patterns = [
        r'(\d+\.?\d*g)',
        r'(\d+\.\d+\s?oz?)',
        r'(\d+\s?oz?)',
        r'(1/8\s?oz?)',
        r'(1/4\s?oz?)',
        r'(1/2\s?oz?)',
    ]
    
    for pattern in anywhere_patterns:
        match = re.search(pattern, item_str, re.IGNORECASE)
        if match:
            return match.group(1).lower().replace(' ', '')
    
    return None

def extract_pack_size_from_item(item_text):
    """Extract pack size from item text (e.g., "OG Kush 3pk 1.5g" or "OG Kush 1.5g 3pk" → "3pk")"""
    if pd.isna(item_text):
        return None
    
    item_str = str(item_text).strip()
    
    # Pattern 1: Pack before weight (e.g., "3pk 1.5g")
    pack_before_weight_patterns = [
        r'(\d+pk)\s+\d+\.?\d*g',
        r'(\d+pk)\s+\d+\s?oz',
        r'(\d+pk)\s+1/[248]\s?oz',
    ]
    
    for pattern in pack_before_weight_patterns:
        match = re.search(pattern, item_str, re.IGNORECASE)
        if match:
            return match.group(1).lower()
    
    # Pattern 2: Weight before pack (e.g., "1.5g 3pk" or "1.5g3pk")
    weight_before_pack_patterns = [
        r'\d+\.?\d*g\s*(\d+pk)',
        r'\d+\s?oz\s*(\d+pk)',
        r'1/[248]\s?oz\s*(\d+pk)',
    ]
    
    for pattern in weight_before_pack_patterns:
        match = re.search(pattern, item_str, re.IGNORECASE)
        if match:
            return match.group(1).lower()
    
    # Pattern 3: Standalone pack size anywhere (fallback)
    standalone_pattern = r'(\d+pk)'
    match = re.search(standalone_pattern, item_str, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    
    return None

def extract_category_keywords(item_text, category):
    """Extract category-specific distinguishing keywords from item text"""
    if pd.isna(item_text) or pd.isna(category):
        return None
    
    item_str = str(item_text).lower()
    category_lower = str(category).lower()
    
    if category_lower == 'vape':
        vape_keywords = ['originals', 'ascnd', 'dna', 'exotics', 'disposable', 'live resin', 'reload', 'rtu', 'curepen', 'curebar']
        found_keywords = [keyword for keyword in vape_keywords if keyword in item_str]
        return ', '.join(found_keywords) if found_keywords else None
    
    if 'flower' in category_lower:
        quality_tiers = ['top shelf', 'headstash', 'exotic', 'premium', 'private reserve', 'reserve']
        found_keywords = [tier for tier in quality_tiers if tier in item_str]
        return ', '.join(found_keywords) if found_keywords else None
    
    if category_lower == 'extract':
        found_keywords = []
        
        # Primary extract types (hierarchical)
        if 'live rosin' in item_str:
            found_keywords.append('live rosin')
        elif 'live resin' in item_str:
            found_keywords.append('live resin')
        elif 'hash rosin' in item_str:
            found_keywords.append('hash rosin')
        elif 'rosin' in item_str:
            found_keywords.append('rosin')
        elif 'resin' in item_str:
            found_keywords.append('resin')
        
        # Brand-specific tiers
        if any(brand in item_str for brand in ['bear labs', 'west coast cure']):
            tier_match = re.search(r'tier\s*([1-4])', item_str)
            if tier_match:
                found_keywords.append(f"tier {tier_match.group(1)}")
        
        # Processing modifiers
        modifiers = ['cold cure', 'fresh press', 'curated', 'hte blend', 'dino eggz']
        found_keywords.extend([modifier for modifier in modifiers if modifier in item_str])
        
        # Consistency types
        consistencies = ['diamonds', 'budder', 'badder', 'sauce', 'sugar', 'jam']
        found_keywords.extend([consistency for consistency in consistencies if consistency in item_str])
        
        # Product types
        product_types = ['rso', 'syringe']
        found_keywords.extend([product_type for product_type in product_types if product_type in item_str])
        
        return ', '.join(found_keywords) if found_keywords else None
    
    if category_lower == 'preroll':
        found_keywords = []
        
        # Preroll types
        preroll_types = ['blunts', 'preroll', 'prerolls', 'joints', 'mini']
        found_keywords.extend([preroll_type for preroll_type in preroll_types if preroll_type in item_str])
        
        # Special attributes
        if 'infused' in item_str:
            found_keywords.append('infused')
        
        return ', '.join(found_keywords) if found_keywords else None
    
    return None

# ============================================================================
# PATTERN MATCHING FUNCTIONS (PRESERVED AS-IS)
# ============================================================================

def match_placeholder_pattern(product_name, template_name):
    """
    Check if a product name matches a template with placeholder patterns
    
    Handles placeholders like:
    - STRAIN (any strain name)
    - COLOR (any color) 
    - FLAVOR (any flavor)
    - SIZE (any size)
    
    Special handling for Stiiizy patterns where STRAIN comes BEFORE bag type:
    - "Stiiizy - Black Bag Black Cherry 3.5g" matches "Stiiizy - STRAIN Black Bag 3.5g"
    
    Args:
        product_name: Actual product name (e.g., "Plug Play - Blue Steel Battery")
        template_name: Template with placeholder (e.g., "Plug Play - COLOR Steel Battery")
    
    Returns:
        bool: True if product matches template pattern
    """
    if pd.isna(product_name) or pd.isna(template_name):
        return False
    
    # Known placeholder patterns
    placeholders = ['STRAIN', 'COLOR', 'FLAVOR', 'SIZE', 'VARIANT']
    
    # Check if template has any placeholders
    has_placeholder = any(placeholder in str(template_name).upper() for placeholder in placeholders)
    if not has_placeholder:
        return False
    
    # Convert to uppercase for comparison
    product_upper = str(product_name).upper()
    template_upper = str(template_name).upper()
    
    # SPECIAL HANDLING FOR "Turn Up/Turn Down" PATTERN (v4.3.18)
    # Handle templates with "/" options where product might match just one variant
    if "TURN UP/TURN DOWN" in template_upper and "TURN" in product_upper:
        # Normalize the template to handle either variant
        if "TURN UP" in product_upper:
            template_upper = template_upper.replace("TURN UP/TURN DOWN", "TURN UP")
        elif "TURN DOWN" in product_upper:
            template_upper = template_upper.replace("TURN UP/TURN DOWN", "TURN DOWN")
    
    # SPECIAL HANDLING FOR STIIIZY STRAIN PATTERNS (v4.3.7)
    # Template: "Stiiizy - STRAIN Black Bag 3.5g" 
    # Product:  "Stiiizy - Black Bag Black Cherry 3.5g"
    if 'STIIIZY' in product_upper and 'STIIIZY' in template_upper and 'STRAIN' in template_upper:
        # Known Stiiizy bag types
        bag_types = ['BLACK BAG', 'WHITE BAG', 'BLUE BAG', 'GOLD BAG', 'SILVER BAG', 'PURPLE BAG']
        
        for bag_type in bag_types:
            if bag_type in product_upper and bag_type in template_upper:
                # Template pattern: "STIIIZY - STRAIN [BAG_TYPE] [WEIGHT]"
                # Product pattern:  "STIIIZY - [BAG_TYPE] [STRAIN_NAME] [WEIGHT]"
                
                # Extract the parts after "STIIIZY -"
                product_after_brand = product_upper.split('STIIIZY -')[-1].strip()
                template_after_brand = template_upper.split('STIIIZY -')[-1].strip()
                
                # Check if template has "STRAIN [BAG_TYPE]"
                if template_after_brand.startswith('STRAIN ' + bag_type):
                    # Extract weight from both
                    weight_pattern = r'(\d+\.?\d*G)'
                    product_weight = re.search(weight_pattern, product_after_brand)
                    template_weight = re.search(weight_pattern, template_after_brand)
                    
                    if product_weight and template_weight:
                        # Check if weights match
                        if product_weight.group(1) == template_weight.group(1):
                            # Check if product has the bag type
                            if bag_type in product_after_brand:
                                # This is a match!
                                return True
    
    # STANDARD PLACEHOLDER MATCHING (for non-Stiiizy or other patterns)
    for placeholder in placeholders:
        if placeholder in template_upper:
            # Split template by placeholder
            parts = template_upper.split(placeholder)
            
            if len(parts) != 2:
                continue  # Skip if placeholder appears multiple times
            
            prefix, suffix = parts
            
            # Check if product starts with prefix and ends with suffix
            if product_upper.startswith(prefix) and product_upper.endswith(suffix):
                # Extract what's in place of placeholder
                placeholder_value = product_upper[len(prefix):-len(suffix) if suffix else len(product_upper)]
                
                # Placeholder value should not be empty and should be reasonable length
                if placeholder_value and len(placeholder_value.strip()) > 0 and len(placeholder_value.strip()) < 50:
                    return True
    
    return False

def match_wildcard_template(item_text, template, wildcards=['COLOR', 'STRAIN', 'FLAVOR']):
    """
    Match item against template with wildcards (COLOR, STRAIN, FLAVOR)
    Returns (match_found, extracted_values) tuple
    
    Improved in v4.3.18 to handle complex patterns like "Turn Up/Turn Down"
    """
    if pd.isna(item_text) or pd.isna(template):
        return False, {}
    
    item_str = str(item_text).strip()
    template_str = str(template).strip()
    
    # Find all wildcard positions in template
    wildcard_positions = {}
    for wildcard in wildcards:
        if wildcard in template_str:
            wildcard_positions[wildcard] = template_str.find(wildcard)
    
    if not wildcard_positions:
        return False, {}
    
    # Create regex pattern from template
    pattern = re.escape(template_str)
    
    # Replace escaped wildcards with capture groups
    # Updated regex to handle more characters including /, (), &, etc.
    for wildcard in wildcards:
        escaped_wildcard = re.escape(wildcard)
        if escaped_wildcard in pattern:
            # More permissive capture group that handles special characters
            pattern = pattern.replace(escaped_wildcard, r'([^,]+?)', 1)
    
    # Try to match
    match = re.match(pattern + r'\s*$', item_str, re.IGNORECASE)
    
    if match:
        # Extract the wildcard values
        extracted_values = {}
        wildcard_list = sorted(wildcard_positions.items(), key=lambda x: x[1])
        for i, (wildcard, _) in enumerate(wildcard_list, 1):
            if i <= len(match.groups()):
                extracted_values[wildcard] = match.group(i).strip()
        return True, extracted_values
    
    return False, {}

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_data
def load_google_sheet_data(sheet_url, load_all_for_matching=False):
    """
    Load data from Google Sheets using service account authentication
    
    Args:
        sheet_url: Google Sheets URL
        load_all_for_matching: If True, load ALL statuses for matching.
                              If False, load only Active + New Price for price comparison
    
    Returns:
        tuple: (DataFrame, worksheet_name) or (None, None) if error
    """
    try:
        credentials_dict = st.secrets["google_sheets"]
        creds = Credentials.from_service_account_info(
            credentials_dict,
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets.readonly",
                "https://www.googleapis.com/auth/drive.readonly"
            ]
        )
        
        client = gspread.authorize(creds)
        sheet = client.open_by_url(sheet_url)
        
        # Try to find worksheet by GID first
        gid = extract_gid_from_url(sheet_url)
        worksheet = None
        
        if gid:
            try:
                for ws in sheet.worksheets():
                    if ws.id == gid:
                        worksheet = ws
                        break
                if not worksheet:
                    worksheet = sheet.get_worksheet(0)
            except:
                worksheet = sheet.get_worksheet(0)
        else:
            worksheet = sheet.get_worksheet(0)
        
        # Try different loading methods
        df = None
        try:
            df = get_as_dataframe(worksheet, parse_dates=True, header=0)
            if len(df.columns) > 0:
                first_col = str(df.columns[0]).lower()
                if first_col == 'active' and 'almora' in str(df.columns).lower():
                    df = None  # Headers are wrong
        except:
            pass
        
        if df is None or df.empty:
            try:
                df = get_as_dataframe(worksheet, parse_dates=True, header=1)
            except:
                pass
        
        if df is None or df.empty:
            try:
                all_values = worksheet.get_all_values()
                if len(all_values) > 1:
                    headers = all_values[0]
                    data_rows = all_values[1:]
                    df = pd.DataFrame(data_rows, columns=headers)
            except:
                return None, None
        
        if df is not None:
            df = df.dropna(how='all').dropna(axis=1, how='all')
            
            # Filter based on purpose
            if 'Status' in df.columns:
                original_count = len(df)
                
                if load_all_for_matching:
                    # For matching: Load ALL statuses
                    st.info(f"📋 Loaded {len(df)} products (ALL statuses) for matching")
                else:
                    # For price display: Only Active + New Price
                    valid_statuses = ['Active', 'New Price']
                    df = df[df['Status'].isin(valid_statuses)].copy()
                    st.info(f"📋 Loaded {len(df)} products with valid pricing status (filtered from {original_count})")
            
            return df, worksheet.title
        else:
            return None, None
        
    except Exception as e:
        st.error(f"Error loading Google Sheet: {str(e)}")
        return None, None

def load_csv_data(uploaded_file):
    """
    Load data from uploaded CSV file
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        tuple: (DataFrame, name) or (None, None) if error
    """
    try:
        df = pd.read_csv(uploaded_file, skiprows=1, low_memory=False)
        return df, "Company Products"
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        return None, None

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def filter_company_products(df, connect_catalog_df=None, selected_shop=None, csv_type='company'):
    """
    Filter company products data by Active field, shop, categories, and brands
    
    Args:
        df: Raw products DataFrame
        connect_catalog_df: Product catalog for brand validation
        selected_shop: Shop to filter to (if company export)
        csv_type: 'company' or 'shop'
        
    Returns:
        DataFrame: Filtered and processed products
    """
    if df is None or df.empty:
        return None
    
    st.write(f"**CSV Type Detected**: {csv_type.upper()}")
    st.write(f"**Original data shape**: {df.shape}")
    
    # Filter by selected shop if company export
    if csv_type == 'company' and selected_shop and selected_shop != 'All Shops':
        if 'Shop' in df.columns:
            before_shop_filter = len(df)
            df = df[df['Shop'] == selected_shop].copy()
            after_shop_filter = len(df)
            st.write(f"**After filtering to '{selected_shop}'**: {df.shape}")
            st.info(f"🏪 Filtered to {selected_shop}: {after_shop_filter:,} products (removed {before_shop_filter - after_shop_filter:,})")
    
    # Filter by Active status
    if 'Active' in df.columns:
        df['Active'] = df['Active'].astype(str).str.strip()
        active_df = df[~df['Active'].isin(['No', 'False', 'no', 'false', 'NO', 'FALSE', 'N', 'n'])]
        st.write(f"**After filtering by Active field**: {active_df.shape}")
    else:
        st.warning("No 'Active' column found. Using all data.")
        active_df = df.copy()
    
    # Exclude unwanted categories
    if 'Category' in active_df.columns:
        before_category_filter = len(active_df)
        active_df = active_df[~active_df['Category'].isin(EXCLUDED_CATEGORIES)]
        after_category_filter = len(active_df)
        removed_count = before_category_filter - after_category_filter
        st.write(f"**After excluding unwanted categories**: {active_df.shape}")
        if removed_count > 0:
            st.info(f"🚫 Excluded {removed_count} products from categories: {', '.join(EXCLUDED_CATEGORIES)}")
    
    # Keep only essential columns
    columns_to_keep = [
        'Shop', 'SKU', 'Item', 'Category', 'Cannabis', 'Measurement',
        'Cost per Unit', 'Unit Price', 'Unit Sale Price', 'Product ID',
        'Brand', 'Cannabis Type', 'Weight Per Unit', 'Custom Weight Measurement',
        'Custom Weight Type', 'Active', 'Available Online', 'Sell Type',
        'Master Product ID', 'Company Product ID', 'Inventory Available',
        '.5 Unit Price', '.5 Unit Sale Price'  # Fallback price fields
    ]
    
    existing_columns = [col for col in columns_to_keep if col in active_df.columns]
    missing_columns = [col for col in columns_to_keep if col not in active_df.columns]
    
    if missing_columns:
        st.warning(f"Missing columns: {missing_columns}")
    
    filtered_df = active_df[existing_columns].copy()
    st.write(f"**After column filtering**: {filtered_df.shape}")
    
    # Create Unit Price columns if they don't exist (v4.3.9 fix)
    # This ensures we have columns to copy the fallback values into
    if 'Unit Price' not in filtered_df.columns:
        filtered_df['Unit Price'] = None
        st.info("📝 Created Unit Price column for fallback values")
    
    if 'Unit Sale Price' not in filtered_df.columns:
        filtered_df['Unit Sale Price'] = None  
        st.info("📝 Created Unit Sale Price column for fallback values")
    
    # Use .5 price fields as fallback when standard price fields are blank (v4.3.9)
    if '.5 Unit Price' in filtered_df.columns:
        # Check where Unit Price is None/NaN or empty
        unit_price_is_empty = (
            filtered_df['Unit Price'].isna() | 
            (filtered_df['Unit Price'] == None) |
            (filtered_df['Unit Price'] == 'None') |
            (filtered_df['Unit Price'] == '')
        )
        
        # Copy .5 Unit Price values to Unit Price where Unit Price is empty
        fallback_count = (unit_price_is_empty & filtered_df['.5 Unit Price'].notna()).sum()
        if fallback_count > 0:
            filtered_df.loc[unit_price_is_empty, 'Unit Price'] = filtered_df.loc[unit_price_is_empty, '.5 Unit Price']
            st.success(f"✅ Applied .5 Unit Price values to {fallback_count:,} products")
    
    if '.5 Unit Sale Price' in filtered_df.columns:
        # Check where Unit Sale Price is None/NaN or empty  
        sale_price_is_empty = (
            filtered_df['Unit Sale Price'].isna() | 
            (filtered_df['Unit Sale Price'] == None) |
            (filtered_df['Unit Sale Price'] == 'None') |
            (filtered_df['Unit Sale Price'] == '')
        )
        
        # Copy .5 Unit Sale Price values to Unit Sale Price where Unit Sale Price is empty
        fallback_count = (sale_price_is_empty & filtered_df['.5 Unit Sale Price'].notna()).sum()
        if fallback_count > 0:
            filtered_df.loc[sale_price_is_empty, 'Unit Sale Price'] = filtered_df.loc[sale_price_is_empty, '.5 Unit Sale Price']
            st.success(f"✅ Applied .5 Unit Sale Price values to {fallback_count:,} products")
    
    # NOW normalize price columns to ensure consistent display
    # Fixes issue where mixed formats like '49.98' and '$49.98' caused display problems
    price_columns = ['Unit Price', 'Unit Sale Price', 'Cost per Unit']
    
    for col in price_columns:
        if col in filtered_df.columns:
            # Convert each value to consistent $XX.XX format
            def normalize_price(x):
                if pd.isna(x) or str(x).strip() in ['', 'nan', 'None', 'NaN']:
                    return None
                # Remove $ and convert to string
                price_str = str(x).replace('$', '').replace(',', '').strip()
                try:
                    # Try to convert to float to validate it's a number
                    price_float = float(price_str)
                    # Return formatted with $ 
                    return f"${price_float:.2f}"
                except:
                    # If not a valid number, return original
                    return x
            
            filtered_df[col] = filtered_df[col].apply(normalize_price)
    
    # Remove the .5 price columns after using them - they're not needed for display
    columns_to_remove = []
    if '.5 Unit Price' in filtered_df.columns:
        columns_to_remove.append('.5 Unit Price')
    if '.5 Unit Sale Price' in filtered_df.columns:
        columns_to_remove.append('.5 Unit Sale Price')
    
    if columns_to_remove:
        filtered_df = filtered_df.drop(columns=columns_to_remove)
        st.info(f"🧹 Removed .5 price columns from display")
    
    # Filter by valid brands from catalog
    if connect_catalog_df is not None and not connect_catalog_df.empty and 'Brand' in filtered_df.columns:
        if 'Brand' in connect_catalog_df.columns:
            valid_brands = connect_catalog_df['Brand'].dropna().unique()
            valid_brands = [str(brand).strip() for brand in valid_brands if str(brand).strip() and str(brand) != 'nan']
            
            before_brand_filter = len(filtered_df)
            filtered_df = filtered_df[filtered_df['Brand'].isin(valid_brands)]
            after_brand_filter = len(filtered_df)
            
            st.write(f"**After brand filtering**: {filtered_df.shape}")
            st.info(f"🎯 Filtered to only include {len(valid_brands)} brands from Product Catalog. Removed {before_brand_filter - after_brand_filter} products.")
    
    # Add data source identifier
    filtered_df['Data_Source'] = csv_type.title()
    
    # Extract enhanced matching data
    st.info("🔍 Extracting Weight, Pack Size, and Category Keywords for enhanced matching...")
    filtered_df['Extracted_Weight'] = filtered_df['Item'].apply(extract_weight_from_item)
    filtered_df['Extracted_Pack_Size'] = filtered_df['Item'].apply(extract_pack_size_from_item)
    filtered_df['Extracted_Category_Keywords'] = filtered_df.apply(
        lambda row: extract_category_keywords(row['Item'], row['Category']), axis=1
    )
    
    # Show extraction stats
    weight_extracted_count = filtered_df['Extracted_Weight'].notna().sum()
    pack_extracted_count = filtered_df['Extracted_Pack_Size'].notna().sum()
    keywords_extracted_count = filtered_df['Extracted_Category_Keywords'].notna().sum()
    
    st.info(f"🔍 Extracted weights from {weight_extracted_count:,} products")
    st.info(f"📦 Extracted pack sizes from {pack_extracted_count:,} products")
    st.info(f"🔤 Extracted category keywords from {keywords_extracted_count:,} products")
    
    # Show keyword extraction breakdown by category
    if keywords_extracted_count > 0:
        st.write("**📊 Keyword Extraction by Category:**")
        for category in filtered_df['Category'].unique():
            if pd.notna(category):
                category_mask = filtered_df['Category'] == category
                category_keywords = filtered_df[category_mask]['Extracted_Category_Keywords'].notna().sum()
                category_total = category_mask.sum()
                if category_keywords > 0:
                    percentage = (category_keywords / category_total * 100)
                    st.write(f"• **{category}**: {category_keywords:,} / {category_total:,} products ({percentage:.1f}%)")
    
    st.write(f"**Final filtered data shape**: {filtered_df.shape}")
    return filtered_df

def add_catalog_location_mapping(df, csv_type='company', selected_shop=None):
    """
    Add a 'Catalog_Location' column to products for catalog price lookup
    
    Args:
        df: Products DataFrame
        csv_type: 'company' or 'shop'
        selected_shop: Shop name for mapping
        
    Returns:
        DataFrame: With added Catalog_Location column
    """
    if df is None or df.empty:
        return df
    
    df_copy = df.copy()
    
    if csv_type == 'company':
        if 'Shop' not in df_copy.columns:
            st.warning("⚠️ Company export missing 'Shop' column")
            return df_copy
        
        df_copy['Catalog_Location'] = df_copy['Shop'].map(SHOP_NAME_MAPPING)
        
        unmapped_shops = df_copy[df_copy['Catalog_Location'].isna()]['Shop'].unique()
        if len(unmapped_shops) > 0:
            st.warning(f"⚠️ Unmapped shops found: {list(unmapped_shops)}")
        
        mapped_count = df_copy['Catalog_Location'].notna().sum()
        total_count = len(df_copy)
        st.info(f"✅ Shop mapping: {mapped_count}/{total_count} products mapped to catalog locations")
    
    elif csv_type == 'shop':
        # For shop exports, use the selected shop name for mapping
        if selected_shop and selected_shop in SHOP_NAME_MAPPING:
            catalog_location = SHOP_NAME_MAPPING[selected_shop]
            df_copy['Catalog_Location'] = catalog_location
            st.info(f"✅ Mapped all {len(df_copy):,} products to catalog location: {catalog_location}")
        else:
            st.warning(f"⚠️ Cannot map shop '{selected_shop}' to catalog location")
    
    return df_copy

def normalize_categories(df):
    """
    Normalize category names to match Product Catalog categories
    
    Args:
        df: Products DataFrame with Category column
        
    Returns:
        DataFrame: With normalized categories
    """
    if df is None or df.empty or 'Category' not in df.columns:
        return df
    
    df_copy = df.copy()
    
    flower_mapping = {
        'Flower (Indica)': 'Flower',
        'Flower (Sativa)': 'Flower', 
        'Flower (Hybrid)': 'Flower'
    }
    
    original_categories = df_copy['Category'].value_counts()
    df_copy['Category'] = df_copy['Category'].replace(flower_mapping)
    
    normalized_count = 0
    for old_cat, new_cat in flower_mapping.items():
        if old_cat in original_categories:
            count = original_categories[old_cat]
            normalized_count += count
            st.info(f"📂 Normalized {count:,} products: '{old_cat}' → '{new_cat}'")
    
    if normalized_count > 0:
        st.success(f"✅ Category normalization: {normalized_count:,} products updated")
    
    return df_copy

# ============================================================================
# SMART MATCHING ENGINE (PRESERVED AS-IS)
# ============================================================================

def add_smart_brand_matching(company_df, catalog_df):
    """Smart brand-based matching using actual catalog structure"""
    if company_df is None or catalog_df is None:
        return company_df
    
    st.info("🧠 Starting smart brand structure matching...")
    
    matched_df = company_df.copy()
    matched_df['Catalog_Match_Found'] = False
    matched_df['Catalog_Template'] = None
    matched_df['Match_Type'] = None
    matched_df['Match_Strategy'] = None
    matched_df['Match_Keywords'] = None
    
    # Build brand and category mappings
    brand_catalog_map = {}
    brand_category_catalog_map = {}
    
    for _, cat_row in catalog_df.iterrows():
        brand = cat_row['Brand']
        template = cat_row['Profile Template']
        category = cat_row.get('Category', 'Unknown')
        
        if pd.notna(brand) and pd.notna(template) and str(template).strip():
            # Only add template if not already in list (deduplicates across statuses)
            if brand not in brand_catalog_map:
                brand_catalog_map[brand] = []
            if template not in brand_catalog_map[brand]:
                brand_catalog_map[brand].append(template)
            
            brand_category_key = f"{brand}|{category}"
            if brand_category_key not in brand_category_catalog_map:
                brand_category_catalog_map[brand_category_key] = []
            if template not in brand_category_catalog_map[brand_category_key]:
                brand_category_catalog_map[brand_category_key].append(template)
    
    # Categorize brands by complexity
    single_entry_brands = {}
    multiple_entry_brands = {}
    
    for brand, templates in brand_catalog_map.items():
        if len(templates) == 1:
            single_entry_brands[brand] = templates[0]
        else:
            multiple_entry_brands[brand] = templates
    
    single_entry_brand_categories = {}
    multiple_entry_brand_categories = {}
    
    for brand_category_key, templates in brand_category_catalog_map.items():
        if len(templates) == 1:
            single_entry_brand_categories[brand_category_key] = templates[0]
        else:
            multiple_entry_brand_categories[brand_category_key] = templates
    
    # Filter out exact match brands from auto-matching
    filtered_single_entry_brands = {brand: template for brand, template in single_entry_brands.items() 
                                  if brand not in EXACT_PRODUCT_MATCH_BRANDS}
    filtered_single_entry_brand_categories = {key: template for key, template in single_entry_brand_categories.items() 
                                            if key.split('|')[0] not in EXACT_PRODUCT_MATCH_BRANDS}
    
    # Show matching strategy overview
    total_brands = len(brand_catalog_map)
    single_count = len(filtered_single_entry_brands)
    multiple_count = len(multiple_entry_brands)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📋 Total Brands", total_brands)
    with col2:
        st.metric("1️⃣ Single Entry", single_count)
    with col3:
        st.metric("🔀 Multiple Entry", multiple_count)
    
    # Perform matching
    exact_matches = 0
    wildcard_matches = 0
    single_entry_matches = 0
    brand_category_matches = 0
    flower_weight_matches = 0
    preroll_matches = 0
    vape_extract_matches = 0
    apparel_matches = 0
    no_matches = 0
    
    progress_bar = st.progress(0)
    troubleshooting_data = []
    
    for counter, (idx, row) in enumerate(matched_df.iterrows()):
        if counter % 1000 == 0:
            progress_bar.progress(counter / len(matched_df))
        
        brand = row['Brand']
        item = row['Item']
        category = row.get('Category', 'Unknown')
        shop = row.get('Shop', 'Unknown')
        
        if pd.isna(brand) or pd.isna(item):
            no_matches += 1
            troubleshooting_data.append({
                'Brand': brand,
                'Item': item,
                'Shop': shop,
                'Match_Status': 'Missing brand or item',
                'Catalog_Options': 'N/A',
                'Notes': 'Data quality issue'
            })
            continue
        
        match_found = False
        
        # 1. Try exact match
        if brand in brand_catalog_map:
            for template in brand_catalog_map[brand]:
                if item.lower() == template.lower():
                    matched_df.at[idx, 'Catalog_Match_Found'] = True
                    matched_df.at[idx, 'Catalog_Template'] = template
                    matched_df.at[idx, 'Match_Type'] = 'exact'
                    matched_df.at[idx, 'Match_Strategy'] = 'exact'
                    exact_matches += 1
                    match_found = True
                    troubleshooting_data.append({
                        'Brand': brand,
                        'Item': item,
                        'Shop': shop,
                        'Match_Status': 'Exact match',
                        'Catalog_Template': template,
                        'Catalog_Options': f"{len(brand_catalog_map[brand])} options",
                        'Notes': 'Perfect match (case insensitive)'
                    })
                    break
        
        # 2. Try placeholder pattern match - WITH SPECIFICITY SCORING AND CATEGORY CHECKING
        if not match_found and brand in brand_catalog_map:
            placeholder_candidates = []
            
            # CRITICAL FIX: Only check templates from the SAME CATEGORY
            brand_category_key = f"{brand}|{category}"
            templates_to_check = brand_category_catalog_map.get(brand_category_key, [])
            
            # Collect ALL placeholder pattern matches FROM SAME CATEGORY
            for template in templates_to_check:
                if match_placeholder_pattern(item, template):
                    # Score by number of matching words (more specific = higher score)
                    item_words = set(item.lower().split())
                    template_words = set(template.lower().split())
                    
                    # Count how many non-placeholder words from template are in item
                    placeholder_keywords = {'strain', 'color', 'flavor', 'size', 'variant'}
                    template_meaningful_words = template_words - placeholder_keywords
                    matching_word_count = len(template_meaningful_words & item_words)
                    
                    # Prefer templates with more matching words (more specific)
                    specificity_score = matching_word_count
                    
                    placeholder_candidates.append({
                        'template': template,
                        'score': specificity_score,
                        'matching_words': matching_word_count
                    })
            
            # If we found matches, pick the MOST SPECIFIC one
            if placeholder_candidates:
                # Sort by specificity score (descending)
                placeholder_candidates.sort(key=lambda x: x['score'], reverse=True)
                best_match = placeholder_candidates[0]
                
                matched_df.at[idx, 'Catalog_Match_Found'] = True
                matched_df.at[idx, 'Catalog_Template'] = best_match['template']
                matched_df.at[idx, 'Match_Type'] = 'placeholder_pattern'
                matched_df.at[idx, 'Match_Strategy'] = 'placeholder_specific'
                exact_matches += 1
                match_found = True
                
                # Note in troubleshooting if multiple candidates existed
                notes = 'Matched via placeholder (COLOR, STRAIN, FLAVOR, etc.)'
                if len(placeholder_candidates) > 1:
                    notes += f' (chose most specific from {len(placeholder_candidates)} matches)'
                
                troubleshooting_data.append({
                    'Brand': brand,
                    'Item': item,
                    'Shop': shop,
                    'Match_Status': 'Placeholder pattern match (specific)',
                    'Catalog_Template': best_match['template'],
                    'Catalog_Options': f"{len(brand_catalog_map[brand])} options",
                    'Notes': notes
                })
        
        # 3. Try wildcard match - WITH SPECIFICITY SCORING AND CATEGORY CHECKING
        if not match_found and brand in brand_catalog_map:
            wildcard_candidates = []
            
            # CRITICAL FIX: Only check templates from the SAME CATEGORY
            brand_category_key = f"{brand}|{category}"
            templates_to_check = brand_category_catalog_map.get(brand_category_key, [])
            
            # Collect ALL wildcard matches FROM SAME CATEGORY
            for template in templates_to_check:
                is_wildcard_match, extracted_wildcards = match_wildcard_template(item, template)
                if is_wildcard_match:
                    # Score by number of matching words (more specific = higher score)
                    item_words = set(item.lower().split())
                    template_words = set(template.lower().split())
                    
                    # Count how many non-wildcard words from template are in item
                    wildcard_placeholders = {'strain', 'color', 'flavor', 'size', 'variant'}
                    template_meaningful_words = template_words - wildcard_placeholders
                    matching_word_count = len(template_meaningful_words & item_words)
                    
                    # Prefer templates with more matching words (more specific)
                    specificity_score = matching_word_count
                    
                    wildcard_candidates.append({
                        'template': template,
                        'wildcards': extracted_wildcards,
                        'score': specificity_score,
                        'matching_words': matching_word_count
                    })
            
            # If we found wildcard matches, pick the MOST SPECIFIC one
            if wildcard_candidates:
                # Sort by specificity score (descending)
                wildcard_candidates.sort(key=lambda x: x['score'], reverse=True)
                best_match = wildcard_candidates[0]
                
                matched_df.at[idx, 'Catalog_Match_Found'] = True
                matched_df.at[idx, 'Catalog_Template'] = best_match['template']
                matched_df.at[idx, 'Match_Type'] = 'wildcard'
                matched_df.at[idx, 'Match_Strategy'] = 'wildcard_specific'
                matched_df.at[idx, 'Match_Keywords'] = ', '.join([f"{k}={v}" for k, v in best_match['wildcards'].items()])
                wildcard_matches += 1
                match_found = True
                
                # Note in troubleshooting if multiple candidates existed
                notes = f'Matched wildcards: {", ".join([f"{k}={v}" for k, v in best_match["wildcards"].items()])}'
                if len(wildcard_candidates) > 1:
                    notes += f' (chose most specific from {len(wildcard_candidates)} matches)'
                
                troubleshooting_data.append({
                    'Brand': brand,
                    'Item': item,
                    'Shop': shop,
                    'Match_Status': 'Wildcard match (specific)',
                    'Catalog_Template': best_match['template'],
                    'Catalog_Options': f"{len(brand_catalog_map[brand])} options",
                    'Notes': notes
                })
                        
        # Skip auto-matching for exact product match brands
        skip_auto_matching = brand in EXACT_PRODUCT_MATCH_BRANDS
        
        # 4. Try single entry brand auto-match
        if not match_found and not skip_auto_matching and brand in filtered_single_entry_brands:
            template = filtered_single_entry_brands[brand]
            matched_df.at[idx, 'Catalog_Match_Found'] = True
            matched_df.at[idx, 'Catalog_Template'] = template
            matched_df.at[idx, 'Match_Type'] = 'brand_auto'
            matched_df.at[idx, 'Match_Strategy'] = 'single_entry'
            single_entry_matches += 1
            match_found = True
            troubleshooting_data.append({
                'Brand': brand,
                'Item': item,
                'Shop': shop,
                'Match_Status': 'Single entry auto-match',
                'Catalog_Template': template,
                'Catalog_Options': '1 option',
                'Notes': 'Auto-matched to only catalog option'
            })
        
        # 5. Try brand+category auto-match
        if not match_found and not skip_auto_matching:
            brand_category_key = f"{brand}|{category}"
            if brand_category_key in filtered_single_entry_brand_categories:
                template = filtered_single_entry_brand_categories[brand_category_key]
                matched_df.at[idx, 'Catalog_Match_Found'] = True
                matched_df.at[idx, 'Catalog_Template'] = template
                matched_df.at[idx, 'Match_Type'] = 'brand_category_auto'
                matched_df.at[idx, 'Match_Strategy'] = 'brand_category_single'
                brand_category_matches += 1
                match_found = True
                troubleshooting_data.append({
                    'Brand': brand,
                    'Item': item,
                    'Shop': shop,
                    'Match_Status': 'Brand+Category auto-match',
                    'Catalog_Template': template,
                    'Catalog_Options': '1 option for this category',
                    'Notes': f'Auto-matched to only {category} option for {brand}'
                })
        
        # 6. Try advanced weight/keyword matching for complex categories
        if not match_found and category in ['Flower', 'Preroll', 'Vape', 'Extract', 'Apparel'] and brand in multiple_entry_brands:
            brand_category_key = f"{brand}|{category}"
            if brand_category_key in multiple_entry_brand_categories:
                templates = multiple_entry_brand_categories[brand_category_key]
                
                # Advanced matching logic for different categories
                matched_template = None
                match_steps = []
                
                if category == 'Flower':
                    matched_template, match_steps = match_flower_products(row, templates)
                elif category == 'Preroll':
                    matched_template, match_steps = match_preroll_products(row, templates)
                elif category in ['Vape', 'Extract']:
                    matched_template, match_steps = match_vape_extract_products(row, templates, category)
                elif category == 'Apparel':
                    matched_template, match_steps = match_apparel_products(row, templates)
                
                if matched_template:
                    matched_df.at[idx, 'Catalog_Match_Found'] = True
                    matched_df.at[idx, 'Catalog_Template'] = matched_template
                    matched_df.at[idx, 'Match_Type'] = f'{category.lower()}_weight_keywords'
                    matched_df.at[idx, 'Match_Strategy'] = f'{category.lower()}_weight_keywords'
                    matched_df.at[idx, 'Match_Keywords'] = ', '.join(match_steps)
                    
                    if category == 'Flower':
                        flower_weight_matches += 1
                    elif category == 'Preroll':
                        preroll_matches += 1
                    elif category == 'Apparel':
                        apparel_matches += 1
                    else:
                        vape_extract_matches += 1
                    
                    match_found = True
                    troubleshooting_data.append({
                        'Brand': brand,
                        'Item': item,
                        'Shop': shop,
                        'Match_Status': f'{category} weight+keywords match',
                        'Catalog_Template': matched_template,
                        'Catalog_Options': f"{len(templates)} total, 1 after filtering",
                        'Notes': f'Matched by: {", ".join(match_steps)}'
                    })
        
        if not match_found:
            no_matches += 1
            # ADD UNMATCHED PRODUCTS TO TROUBLESHOOTING DATA!
            troubleshooting_data.append({
                'Brand': brand,
                'Item': item,
                'Shop': shop,
                'Match_Status': 'No match found',
                'Catalog_Template': None,
                'Catalog_Options': f"{len(brand_catalog_map.get(brand, []))} options" if brand in brand_catalog_map else '0 options',
                'Notes': f'Brand {"found" if brand in brand_catalog_map else "NOT"} in catalog'
            })
    
    progress_bar.progress(1.0)
    
    # Show results
    total_matches = exact_matches + wildcard_matches + single_entry_matches + brand_category_matches + flower_weight_matches + preroll_matches + vape_extract_matches + apparel_matches
    total_match_rate = (total_matches / len(matched_df)) * 100 if len(matched_df) > 0 else 0
    
    st.success(f"🎉 Enhanced Matching Results:")
    
    # Count placeholder pattern matches separately
    placeholder_pattern_count = matched_df[matched_df['Match_Type'] == 'placeholder_pattern'].shape[0]
    exact_only_count = exact_matches - placeholder_pattern_count
    
    col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)
    with col1:
        st.metric("🎯 Exact", f"{exact_only_count:,}")
    with col2:
        st.metric("🔤 Pattern", f"{placeholder_pattern_count:,}", help="COLOR, STRAIN, FLAVOR placeholders")
    with col3:
        st.metric("1️⃣ Single", f"{single_entry_matches:,}")
    with col4:
        st.metric("📂 Category", f"{brand_category_matches:,}")
    with col5:
        st.metric("🌸 Flower", f"{flower_weight_matches:,}")
    with col6:
        st.metric("🚬 Preroll", f"{preroll_matches:,}")
    with col7:
        st.metric("💨 Vape/Ext", f"{vape_extract_matches:,}")
    with col8:
        st.metric("👕 Apparel", f"{apparel_matches:,}")
    with col9:
        st.metric("📊 Total", f"{total_matches:,} ({total_match_rate:.1f}%)")
    
    # Store troubleshooting data in session state
    st.session_state['troubleshooting_data'] = pd.DataFrame(troubleshooting_data)
    
    return matched_df

# ============================================================================
# CATEGORY-SPECIFIC MATCHING (PRESERVED AS-IS)
# ============================================================================

def match_flower_products(row, templates):
    """Advanced matching for flower products using weight and keywords"""
    current_templates = templates
    match_steps = []
    
    # Filter by weight
    company_weight = row.get('Extracted_Weight')
    if company_weight:
        weight_matched_templates = []
        for template in current_templates:
            catalog_weight = extract_weight_from_item(template)
            if catalog_weight == company_weight:
                weight_matched_templates.append(template)
        
        if weight_matched_templates:
            current_templates = weight_matched_templates
            match_steps.append(f"weight: {company_weight}")
    
    # Filter by keywords if still multiple options
    company_keywords = row.get('Extracted_Category_Keywords')
    if company_keywords and len(current_templates) > 1:
        company_keyword_list = [kw.strip() for kw in str(company_keywords).split(',')]
        
        template_scores = []
        for template in current_templates:
            catalog_keywords = extract_category_keywords(template, 'Flower')
            if catalog_keywords:
                catalog_keyword_list = [kw.strip() for kw in catalog_keywords.split(',')]
                matches = sum(1 for ck in company_keyword_list if ck in catalog_keyword_list)
                template_scores.append((template, matches, len(catalog_keyword_list), catalog_keyword_list))
            else:
                template_scores.append((template, 0, 0, []))
        
        max_score = max(score for _, score, _, _ in template_scores)
        if max_score > 0:
            best_scored_templates = [(template, score, total_kw, kw_list) for template, score, total_kw, kw_list in template_scores if score == max_score]
            
            if len(best_scored_templates) == 1:
                current_templates = [best_scored_templates[0][0]]
                matched_keywords = [ck for ck in company_keyword_list if ck in best_scored_templates[0][3]]
                match_steps.append(f"keywords: {', '.join(matched_keywords)}")
            else:
                # Tiebreaker: prefer fewer total keywords
                min_total_keywords = min(total_kw for _, _, total_kw, _ in best_scored_templates)
                final_candidates = [template for template, score, total_kw, kw_list in best_scored_templates if total_kw == min_total_keywords]
                
                if len(final_candidates) == 1:
                    current_templates = final_candidates
                    winner_keywords = [kw_list for template, score, total_kw, kw_list in best_scored_templates if template == final_candidates[0]][0]
                    matched_keywords = [ck for ck in company_keyword_list if ck in winner_keywords]
                    match_steps.append(f"keywords: {', '.join(matched_keywords)} (tiebreaker)")
    
    return (current_templates[0], match_steps) if len(current_templates) == 1 else (None, [])

def match_preroll_products(row, templates):
    """Advanced matching for preroll products using infused status, pack size, weight, and keywords"""
    # Filter by infused status first
    company_has_infused = 'infused' in str(row['Item']).lower()
    
    infused_filtered_templates = []
    for template in templates:
        template_has_infused = 'infused' in str(template).lower()
        if company_has_infused == template_has_infused:
            infused_filtered_templates.append(template)
    
    current_templates = infused_filtered_templates if infused_filtered_templates else templates
    match_steps = []
    if infused_filtered_templates:
        match_steps.append(f"infused: {'yes' if company_has_infused else 'no'}")
    
    # Filter by pack size FIRST (before weight) - pack size is very distinctive for prerolls
    company_pack = row.get('Extracted_Pack_Size')
    if company_pack and len(current_templates) > 1:
        pack_matched_templates = []
        for template in current_templates:
            catalog_pack = extract_pack_size_from_item(template)
            if catalog_pack == company_pack:
                pack_matched_templates.append(template)
        
        if pack_matched_templates:
            current_templates = pack_matched_templates
            match_steps.append(f"pack: {company_pack}")
    
    # Filter by weight
    company_weight = row.get('Extracted_Weight')
    if company_weight and len(current_templates) > 1:
        weight_matched_templates = []
        for template in current_templates:
            catalog_weight = extract_weight_from_item(template)
            if catalog_weight == company_weight:
                weight_matched_templates.append(template)
        
        if weight_matched_templates:
            current_templates = weight_matched_templates
            match_steps.append(f"weight: {company_weight}")
    
    # If no pack size in company product but still multiple templates, prefer templates without pack
    if not company_pack and len(current_templates) > 1:
        # Fallback: prefer templates without pack sizes
        no_pack_templates = []
        for template in current_templates:
            catalog_pack = extract_pack_size_from_item(template)
            if not catalog_pack:
                no_pack_templates.append(template)
        
        if len(no_pack_templates) == 1:
            current_templates = no_pack_templates
            match_steps.append("no pack (fallback)")
    
    # Filter by type keywords (excluding 'infused')
    company_keywords = row.get('Extracted_Category_Keywords')
    if company_keywords and len(current_templates) > 1:
        company_keyword_list = [kw.strip() for kw in str(company_keywords).split(',')]
        company_type_keywords = [kw for kw in company_keyword_list if kw != 'infused']
        
        if company_type_keywords:
            template_scores = []
            for template in current_templates:
                catalog_keywords = extract_category_keywords(template, 'Preroll')
                if catalog_keywords:
                    catalog_keyword_list = [kw.strip() for kw in catalog_keywords.split(',')]
                    catalog_type_keywords = [kw for kw in catalog_keyword_list if kw != 'infused']
                    matches = sum(1 for ck in company_type_keywords if ck in catalog_type_keywords)
                    template_scores.append((template, matches, len(catalog_type_keywords), catalog_type_keywords))
                else:
                    template_scores.append((template, 0, 0, []))
            
            max_score = max(score for _, score, _, _ in template_scores)
            if max_score > 0:
                best_scored_templates = [(template, score, total_kw, kw_list) for template, score, total_kw, kw_list in template_scores if score == max_score]
                
                if len(best_scored_templates) == 1:
                    current_templates = [best_scored_templates[0][0]]
                    matched_keywords = [ck for ck in company_type_keywords if ck in best_scored_templates[0][3]]
                    match_steps.append(f"type: {', '.join(matched_keywords)}")
                else:
                    # Tiebreaker: prefer fewer total keywords
                    min_total_keywords = min(total_kw for _, _, total_kw, _ in best_scored_templates)
                    final_candidates = [template for template, score, total_kw, kw_list in best_scored_templates if total_kw == min_total_keywords]
                    
                    if len(final_candidates) == 1:
                        current_templates = final_candidates
                        winner_keywords = [kw_list for template, score, total_kw, kw_list in best_scored_templates if template == final_candidates[0]][0]
                        matched_keywords = [ck for ck in company_type_keywords if ck in winner_keywords]
                        match_steps.append(f"type: {', '.join(matched_keywords)} (tiebreaker)")
    
    return (current_templates[0], match_steps) if len(current_templates) == 1 else (None, [])

def match_vape_extract_products(row, templates, category):
    """Advanced matching for vape and extract products using weight and keywords"""
    current_templates = templates
    match_steps = []
    
    # Filter by weight
    company_weight = row.get('Extracted_Weight')
    if company_weight and len(current_templates) > 1:
        weight_matched_templates = []
        for template in current_templates:
            catalog_weight = extract_weight_from_item(template)
            if catalog_weight == company_weight:
                weight_matched_templates.append(template)
        
        if weight_matched_templates:
            current_templates = weight_matched_templates
            match_steps.append(f"weight: {company_weight}")
    
    # Filter by keywords
    company_keywords = row.get('Extracted_Category_Keywords')
    if company_keywords and len(current_templates) > 1:
        company_keyword_list = [kw.strip() for kw in str(company_keywords).split(',')]
        
        template_scores = []
        for template in current_templates:
            catalog_keywords = extract_category_keywords(template, category)
            if catalog_keywords:
                catalog_keyword_list = [kw.strip() for kw in catalog_keywords.split(',')]
                matches = sum(1 for ck in company_keyword_list if ck in catalog_keyword_list)
                template_scores.append((template, matches, len(catalog_keyword_list), catalog_keyword_list))
            else:
                template_scores.append((template, 0, 0, []))
        
        max_score = max(score for _, score, _, _ in template_scores)
        if max_score > 0:
            best_scored_templates = [(template, score, total_kw, kw_list) for template, score, total_kw, kw_list in template_scores if score == max_score]
            
            if len(best_scored_templates) == 1:
                current_templates = [best_scored_templates[0][0]]
                matched_keywords = [ck for ck in company_keyword_list if ck in best_scored_templates[0][3]]
                match_steps.append(f"keywords: {', '.join(matched_keywords)}")
            else:
                # Tiebreaker: prefer fewer total keywords
                min_total_keywords = min(total_kw for _, _, total_kw, _ in best_scored_templates)
                final_candidates = [template for template, score, total_kw, kw_list in best_scored_templates if total_kw == min_total_keywords]
                
                if len(final_candidates) == 1:
                    current_templates = final_candidates
                    winner_keywords = [kw_list for template, score, total_kw, kw_list in best_scored_templates if template == final_candidates[0]][0]
                    matched_keywords = [ck for ck in company_keyword_list if ck in winner_keywords]
                    match_steps.append(f"keywords: {', '.join(matched_keywords)} (tiebreaker)")
    elif not company_keywords and len(current_templates) > 1:
        # Fallback: prefer templates without keywords
        no_keyword_templates = []
        for template in current_templates:
            catalog_keywords = extract_category_keywords(template, category)
            if not catalog_keywords:
                no_keyword_templates.append(template)
        
        if len(no_keyword_templates) == 1:
            current_templates = no_keyword_templates
            match_steps.append("no keywords (fallback)")
    
    return (current_templates[0], match_steps) if len(current_templates) == 1 else (None, [])

def match_apparel_products(row, templates):
    """
    Advanced matching for apparel products by stripping size suffixes
    
    Apparel products in Company Products have sizes in parentheses:
    - "Haven - California Shirt (XS)" → matches "Haven - California Shirt"
    - "Haven - Hoodie (Large)" → matches "Haven - Hoodie"
    
    Args:
        row: Product row from company data
        templates: List of possible catalog templates for this brand
    
    Returns:
        tuple: (matched_template, match_steps) or (None, [])
    """
    company_item = row['Item']
    match_steps = []
    
    # Strip size from company item (anything in parentheses at the end)
    import re
    size_pattern = r'\s*\([^)]+\)\s*$'
    company_item_no_size = re.sub(size_pattern, '', str(company_item)).strip()
    
    # Check if we removed a size
    if company_item_no_size != str(company_item).strip():
        size_match = re.search(r'\(([^)]+)\)$', str(company_item))
        if size_match:
            extracted_size = size_match.group(1)
            match_steps.append(f"size: {extracted_size}")
    
    # Now try to match the size-stripped item against templates
    matched_template = None
    for template in templates:
        # Clean comparison - case insensitive
        if company_item_no_size.lower() == str(template).lower():
            matched_template = template
            match_steps.append("exact match (without size)")
            break
    
    # If no exact match, try partial matching for complex apparel names
    if not matched_template and len(templates) == 1:
        # If only one template for this brand/category, more lenient matching
        template = templates[0]
        # Check if the core product name is in both
        if company_item_no_size.lower() in str(template).lower() or str(template).lower() in company_item_no_size.lower():
            matched_template = template
            match_steps.append("partial match (single template)")
    
    return (matched_template, match_steps) if matched_template else (None, [])

# ============================================================================
# PRICE COMPARISON FUNCTIONS
# ============================================================================

def add_simple_price_comparison(company_df, catalog_df, selected_statuses=['Active']):
    """
    Simple price comparison - add basic price difference columns
    Now includes Cost per Unit comparison with Max Unit Cost from catalog
    
    Args:
        company_df: Products dataframe
        catalog_df: Full catalog (all statuses)
        selected_statuses: List of statuses to use for pricing (default: ['Active'])
        
    Returns:
        DataFrame: With price comparison columns added
    """
    if company_df is None or catalog_df is None:
        return company_df
    
    matched_products = company_df[company_df['Catalog_Match_Found'] == True].copy()
    
    if len(matched_products) == 0:
        st.warning("⚠️ No matched products found for price comparison")
        return company_df
    
    st.info(f"💰 Adding price comparison for {len(matched_products):,} matched products...")
    
    # Format status list for display
    if len(selected_statuses) == 1:
        status_display = f"**{selected_statuses[0]}**"
    else:
        status_display = f"**{', '.join(selected_statuses)}**"
    st.info(f"📅 Using prices from catalog statuses: {status_display}")
    
    # Initialize price comparison columns
    company_df['Catalog_Retail_Price'] = None
    company_df['Catalog_Sale_Price'] = None
    company_df['Retail_Price_Diff'] = None
    company_df['Sale_Price_Diff'] = None
    company_df['Catalog_Status_Used'] = None
    
    # NEW: Add cost comparison columns
    company_df['Catalog_Max_Unit_Cost'] = None
    company_df['Cost_Diff'] = None
    
    # Build catalog lookup - filter by selected statuses
    catalog_for_pricing = catalog_df[catalog_df['Status'].isin(selected_statuses)].copy() if 'Status' in catalog_df.columns else catalog_df
    
    st.info(f"🔍 {len(catalog_for_pricing)} catalog products have selected status(es) for pricing")
    
    catalog_lookup = {}
    for _, cat_row in catalog_for_pricing.iterrows():
        template = cat_row['Profile Template']
        if pd.notna(template):
            catalog_lookup[template] = {
                'data': cat_row,
                'status': cat_row.get('Status', 'Unknown')
            }
    
    pricing_issues = 0
    cost_issues = 0
    matched_but_no_pricing = 0
    
    # Compare prices for each matched product
    for idx, row in matched_products.iterrows():
        catalog_template = row['Catalog_Template']
        catalog_location = row['Catalog_Location']
        
        if pd.isna(catalog_template) or pd.isna(catalog_location):
            continue
        
        catalog_entry = catalog_lookup.get(catalog_template)
        
        if catalog_entry is None:
            # Product matched but template doesn't have selected statuses
            matched_but_no_pricing += 1
            company_df.at[idx, 'Catalog_Status_Used'] = f"No matching status"
            continue
        
        # Get the catalog data and status
        catalog_data = catalog_entry['data']
        catalog_status = catalog_entry['status']
        company_df.at[idx, 'Catalog_Status_Used'] = catalog_status
        
        # Get catalog prices for this location
        retail_price_col = f"{catalog_location} Retail Price"
        sale_price_col = f"{catalog_location} Sale Price"
        
        catalog_retail = clean_price(catalog_data.get(retail_price_col))
        catalog_sale = clean_price(catalog_data.get(sale_price_col))
        company_retail = clean_price(row.get('Unit Price'))
        company_sale = clean_price(row.get('Unit Sale Price'))
        
        # Store catalog prices
        company_df.at[idx, 'Catalog_Retail_Price'] = catalog_retail
        company_df.at[idx, 'Catalog_Sale_Price'] = catalog_sale
        
        # NEW: Get and compare costs
        catalog_max_cost = clean_price(catalog_data.get('Max Unit Cost'))
        company_cost = clean_price(row.get('Cost per Unit'))
        
        if catalog_max_cost is not None:
            company_df.at[idx, 'Catalog_Max_Unit_Cost'] = catalog_max_cost
            
            if company_cost is not None:
                cost_diff = company_cost - catalog_max_cost
                company_df.at[idx, 'Cost_Diff'] = cost_diff
                
                # Track if cost exceeds max
                if cost_diff > 0.01:
                    cost_issues += 1
        
        # Calculate differences - IMPROVED LOGIC
        # Retail Price Comparison
        if catalog_retail is not None and company_retail is not None:
            company_df.at[idx, 'Retail_Price_Diff'] = company_retail - catalog_retail
        elif catalog_retail is not None and company_retail is None:
            company_df.at[idx, 'Retail_Price_Diff'] = -catalog_retail
        elif catalog_retail is None and company_retail is not None:
            company_df.at[idx, 'Retail_Price_Diff'] = company_retail
        
        # Sale Price Comparison - v4.2.6 FIX: Check numeric match FIRST
        if catalog_sale is not None and company_sale is not None:
            # Both have sale price values - check if they match numerically
            if abs(company_sale - catalog_sale) <= 0.01:
                # Values match (within penny) - no difference
                company_df.at[idx, 'Sale_Price_Diff'] = 0
            else:
                # Values don't match - show difference
                company_df.at[idx, 'Sale_Price_Diff'] = company_sale - catalog_sale
        elif catalog_sale is not None and company_sale is None:
            # Catalog has sale, company doesn't
            company_df.at[idx, 'Sale_Price_Diff'] = -catalog_sale
        elif catalog_sale is None and company_sale is not None:
            # Check if company's sale price is a "real" sale
            company_has_real_sale = (company_retail is not None and 
                                     abs(company_sale - company_retail) > 0.01)
            if company_has_real_sale:
                # Company has real sale, catalog doesn't
                company_df.at[idx, 'Sale_Price_Diff'] = company_sale
        
        # Count pricing issues
        retail_diff = company_df.at[idx, 'Retail_Price_Diff']
        sale_diff = company_df.at[idx, 'Sale_Price_Diff']
        
        has_retail_issue = (retail_diff is not None and not pd.isna(retail_diff) and abs(retail_diff) > 0.01)
        has_sale_issue = (sale_diff is not None and not pd.isna(sale_diff) and abs(sale_diff) > 0.01)
        
        if has_retail_issue or has_sale_issue:
            pricing_issues += 1
    
    if matched_but_no_pricing > 0:
        status_list = ', '.join([f"'{s}'" for s in selected_statuses])
        st.warning(f"⚠️ {matched_but_no_pricing} matched products don't have any of the selected statuses ({status_list})")
    
    st.success(f"💰 Price comparison complete! Found {pricing_issues:,} products with price differences > $0.01")
    
    if cost_issues > 0:
        st.warning(f"⚠️ Found {cost_issues:,} products where Cost per Unit exceeds Max Unit Cost")
    
    return company_df

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.title(f"🛒 Product Price Checker v{VERSION}")
    st.markdown("Filter your products and connect to Product Catalog data with automatic CSV type detection")
    
    st.sidebar.header("📊 Data Sources")
    
    # Add Status Filter at the top of sidebar
    st.sidebar.subheader("📋 Catalog Price Sources")
    
    # Define available statuses
    available_statuses = ['Active', 'New Price', 'New Product', 'DNO', 'REVIEW', 'On-boarding', 'Inactive']
    
    catalog_statuses = st.sidebar.multiselect(
        "Which catalog statuses to use for pricing:",
        options=available_statuses,
        default=['Active', 'New Product', 'DNO'],
        help="Select one or more catalog statuses. Products with these statuses will be used for price comparison."
    )
    
    if not catalog_statuses:
        st.sidebar.warning("⚠️ Please select at least one status")
        catalog_statuses = ['Active']
    
    # Show selected statuses
    if len(catalog_statuses) == 1:
        st.sidebar.info(f"✅ Using {catalog_statuses[0]} catalog prices")
    else:
        st.sidebar.info(f"✅ Using {len(catalog_statuses)} statuses: {', '.join(catalog_statuses)}")
    
    # Add Brand Filtering Option
    st.sidebar.subheader("🏷️ Brand Filtering")
    filter_by_catalog_brands = st.sidebar.checkbox(
        "Only include brands in Product Catalog",
        value=True,
        help="When checked, only processes brands that exist in the Product Catalog. "
             "Uncheck to include ALL brands (non-catalog brands won't have price matching)."
    )
    
    if filter_by_catalog_brands:
        st.sidebar.info("🎯 Only catalog brands will be processed")
    else:
        st.sidebar.warning("⚠️ Including ALL brands (non-catalog brands won't match)")
    
    st.sidebar.subheader("📄 Upload Products CSV")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV (Company or Single Shop):",
        type=['csv'],
        help="Upload either a Company Products CSV (multi-shop) or a Single Shop CSV. Will be automatically detected."
    )
    
    # Initialize session state for CSV type and shops
    if 'csv_type' not in st.session_state:
        st.session_state['csv_type'] = None
    if 'available_shops' not in st.session_state:
        st.session_state['available_shops'] = []
    if 'selected_shop' not in st.session_state:
        st.session_state['selected_shop'] = None
    if 'detection_complete' not in st.session_state:
        st.session_state['detection_complete'] = False
    if 'catalog_statuses' not in st.session_state:
        st.session_state['catalog_statuses'] = ['Active', 'New Product', 'DNO']
    
    # Update catalog statuses in session state
    if st.session_state.get('catalog_statuses') != catalog_statuses:
        st.session_state['catalog_statuses'] = catalog_statuses
        # Clear cached catalog data when statuses change
        st.cache_data.clear()
    
    # Auto-detect CSV type when file is uploaded
    if uploaded_file is not None:
        # Only detect once per file upload
        file_name = uploaded_file.name
        if st.session_state.get('last_uploaded_file') != file_name:
            st.session_state['last_uploaded_file'] = file_name
            st.session_state['detection_complete'] = False
            
            try:
                # Read CSV to detect type
                temp_df = pd.read_csv(uploaded_file, skiprows=1, nrows=5)
                detected_type = detect_csv_type(temp_df)
                
                st.session_state['csv_type'] = detected_type
                
                # If company export, get available shops
                if detected_type == 'company':
                    uploaded_file.seek(0)
                    full_df = pd.read_csv(uploaded_file, skiprows=1, low_memory=False)
                    shops = get_unique_shops(full_df)
                    st.session_state['available_shops'] = shops
                    if not st.session_state.get('selected_shop'):
                        st.session_state['selected_shop'] = 'All Shops'
                elif detected_type == 'shop':
                    # For shop exports, set default to first option
                    st.session_state['available_shops'] = []
                    shop_options = list(SHOP_NAME_MAPPING.keys())
                    if not st.session_state.get('selected_shop') or st.session_state['selected_shop'] not in shop_options:
                        st.session_state['selected_shop'] = shop_options[0]
                else:
                    st.session_state['available_shops'] = []
                    st.session_state['selected_shop'] = None
                
                st.session_state['detection_complete'] = True
                
                # Reset file pointer
                uploaded_file.seek(0)
            except Exception as e:
                st.sidebar.error(f"Error detecting CSV type: {str(e)}")
                st.sidebar.error(f"Debug info: First column = {temp_df.columns[0] if 'temp_df' in locals() else 'Could not read'}")
    
    # Show CSV type indicator
    if uploaded_file is not None and st.session_state['csv_type']:
        csv_type_display = st.session_state['csv_type'].upper()
        if st.session_state['csv_type'] == 'company':
            st.sidebar.success(f"✅ Detected: {csv_type_display} Export (Multi-Shop)")
        elif st.session_state['csv_type'] == 'shop':
            st.sidebar.success(f"✅ Detected: {csv_type_display} Export (Single Shop)")
    
    # Show shop selector for company exports
    if st.session_state['csv_type'] == 'company' and st.session_state['available_shops']:
        st.sidebar.subheader("🏪 Shop Selection")
        shop_options = ['All Shops'] + st.session_state['available_shops']
        selected_shop = st.sidebar.selectbox(
            "Select Shop to Process:",
            options=shop_options,
            index=shop_options.index(st.session_state['selected_shop']) if st.session_state['selected_shop'] in shop_options else 0,
            help="Choose a specific shop or 'All Shops' to process all shops together"
        )
        st.session_state['selected_shop'] = selected_shop
        
        if selected_shop != 'All Shops':
            st.sidebar.info(f"🔍 Will process only: {selected_shop}")
    
    # For shop exports, ask which shop this data is from
    elif st.session_state['csv_type'] == 'shop':
        st.sidebar.subheader("🏪 Shop Identification")
        st.sidebar.info("Please identify which shop this single-shop export is from:")
        shop_options = list(SHOP_NAME_MAPPING.keys())
        
        # Get the current index
        current_index = 0
        if st.session_state['selected_shop'] and st.session_state['selected_shop'] in shop_options:
            current_index = shop_options.index(st.session_state['selected_shop'])
        
        selected_shop = st.sidebar.selectbox(
            "Which shop is this data from?",
            options=shop_options,
            index=current_index,
            help="Select the shop this single-shop CSV export is from for proper catalog mapping"
        )
        st.session_state['selected_shop'] = selected_shop
        st.sidebar.success(f"🔍 Data source: {selected_shop}")
    
    google_sheets_available = "google_sheets" in st.secrets
    
    if google_sheets_available:
        st.sidebar.subheader("🔗 Reference Data Sources")
        st.sidebar.success("✅ Connect Product Catalog - Configured")
        st.sidebar.info("Product catalog will be automatically loaded from configured Google Sheet")
    else:
        st.sidebar.warning("⚠️ Google Sheets API not configured")
        st.sidebar.info("Product filtering will work, but brand cross-referencing will be skipped")
    
    # Add changelog expander
    with st.sidebar.expander("📋 Version History & Changelog"):
        st.markdown("""
        **v4.3.21** (Current - 2025-12-13)
        - 🚨 CRITICAL: Fixed cross-category matching bug
        - ✅ Placeholder/wildcard matching now respects categories
        - 🎯 Prerolls only match to Preroll templates
        - 🔒 Prevents dangerous price mismatches
        
        **v4.3.20** (2025-12-13)
        - 🔧 FIXED: Troubleshooting tab shows ALL products
        - ✅ Now includes unmatched products for debugging
        - 📊 Shows brand availability in catalog
        - 🎯 Better visibility into matching failures
        
        **v4.3.18** (2025-12-13)
        - 🔧 FIXED: Wildcard matching for "/" patterns
        - ✅ Turn products now match Turn Up/Turn Down templates
        - 🎯 Improved regex for special characters in strains
        - 📊 Better handling of complex template patterns
        
        **v4.3.17** (2025-12-13)
        - 👕 ADDED: Apparel matching logic with size handling
        - ✂️ Strips size in parentheses for matching
        - 🎯 Matches "California Shirt (XS)" to "California Shirt"
        - 📊 Added apparel counter to results display
        
        **v4.3.16** (2025-12-13)
        - 👕 CHANGED: Apparel category now INCLUDED
        - ✅ Previously excluded, now processes apparel products
        - 📋 All other excluded categories remain filtered
        
        **v4.3.15** (2025-12-09)
        - 🎯 IMPROVED: Multi-select filter for price differences
        - 📊 Choose any combination of Retail, Sale, Cost
        - ✅ Defaults to Retail + Sale (most common)
        - 🔧 More flexible filtering workflow
        
        **v4.3.14** (2025-12-09)
        - 🔧 FIXED: Price Inspector displays Unit Price correctly
        - 💰 ADDED: Cost per Unit tracking
        - 📊 ADDED: Max Unit Cost comparison
        - 🎯 Enhanced price string conversion
        
        **v4.3.13** (2025-11-27)
        - ✅ Full application restored after truncation
        - 💰 Fixed Unit Price display in main Products tab
        - 🔥 Fixed Blaze POS headers format
        
        **v4.3.4** (2025-11-21)
        - 🏪 Added Hawthorne store mapping
        - 📍 Now supports 12 store locations
        
        [Earlier versions omitted - see git history]
        """)
    
    # Add version at bottom of sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Version {VERSION}**")
    if len(catalog_statuses) == 1:
        st.sidebar.markdown(f"**Catalog Status: {catalog_statuses[0]}**")
    else:
        st.sidebar.markdown(f"**Catalog Statuses: {', '.join(catalog_statuses)}**")
    
    if st.sidebar.button("🚀 Load Data", type="primary", disabled=(uploaded_file is None or st.session_state['csv_type'] is None or st.session_state['selected_shop'] is None)):
        with st.spinner("Loading data from all sources..."):
            
            # Load Product Catalog with ALL statuses for matching
            connect_catalog_df = None
            if google_sheets_available:
                st.info(f"📊 Loading Connect Product Catalog...")
                # Load ALL statuses for matching purposes
                catalog_df, catalog_ws_name = load_google_sheet_data(CONNECT_CATALOG_URL, load_all_for_matching=True)
                if catalog_df is not None:
                    connect_catalog_df = catalog_df
                    st.session_state['df_catalog'] = connect_catalog_df
                    st.session_state['df_catalog_name'] = f"Connect Product Catalog ({catalog_ws_name})"
                    st.session_state['df_catalog_statuses'] = catalog_statuses
                    st.success(f"✅ Loaded Product Catalog: {connect_catalog_df.shape[0]} records (all statuses for matching)")
                else:
                    st.error("❌ Failed to load Connect Product Catalog")
            
            # Load and process Products CSV
            filtered_csv = None
            if uploaded_file is not None and st.session_state['csv_type']:
                csv_type = st.session_state['csv_type']
                selected_shop = st.session_state.get('selected_shop')
                
                st.info(f"📄 Processing {csv_type.upper()} CSV...")
                df_csv, csv_name = load_csv_data(uploaded_file)
                if df_csv is not None:
                    filtered_csv = filter_company_products(
                        df_csv, 
                        connect_catalog_df, 
                        selected_shop=selected_shop if csv_type == 'company' else None,
                        csv_type=csv_type
                    )
                    if filtered_csv is not None:
                        filtered_csv = add_catalog_location_mapping(
                            filtered_csv, 
                            csv_type=csv_type,
                            selected_shop=selected_shop
                        )
                        filtered_csv = normalize_categories(filtered_csv)
                        
                        if connect_catalog_df is not None:
                            filtered_csv = add_smart_brand_matching(filtered_csv, connect_catalog_df)
                            # Pass the selected catalog statuses for price comparison
                            filtered_csv = add_simple_price_comparison(filtered_csv, connect_catalog_df, selected_statuses=catalog_statuses)
                        
                        st.session_state['df_csv'] = filtered_csv
                        csv_display_name = f"{csv_type.title()} Products"
                        if csv_type == 'company' and selected_shop and selected_shop != 'All Shops':
                            csv_display_name += f" - {selected_shop}"
                        elif csv_type == 'shop' and selected_shop:
                            csv_display_name += f" - {selected_shop}"
                        st.session_state['df_csv_name'] = csv_display_name
                        st.success(f"✅ Processed {csv_display_name}: {filtered_csv.shape[0]} products after filtering, matching, and price comparison")
                    else:
                        st.error("❌ Failed to process Products CSV")
                else:
                    st.error("❌ Failed to load Products CSV")
            
            # Summary
            loaded_sources = 0
            if connect_catalog_df is not None:
                loaded_sources += 1
            if filtered_csv is not None:
                loaded_sources += 1
            
            if loaded_sources > 0:
                st.success(f"🎉 Successfully loaded {loaded_sources} data source(s)!")
            else:
                st.error("❌ No data could be loaded. Check your files/URLs and permissions.")
    
    # Display data tabs if any data is loaded
    if any(key in st.session_state for key in ['df_csv', 'df_catalog']):
        
        # Build tab list dynamically
        tab_names = ["📊 Overview"]
        if 'df_csv' in st.session_state:
            tab_names.append("📄 Products")
            if 'Catalog_Match_Found' in st.session_state['df_csv'].columns and st.session_state['df_csv']['Catalog_Match_Found'].sum() > 0:
                tab_names.append("💰 Price Inspector")
            if 'troubleshooting_data' in st.session_state:
                tab_names.append("🔧 Troubleshooting")
        if 'df_catalog' in st.session_state:
            tab_names.append("📋 Product Catalog")
        
        tabs = st.tabs(tab_names)
        tab_index = 0
        
        # Overview Tab
        with tabs[tab_index]:
            st.subheader("📊 Data Overview")
            
            # Show which catalog status is being used
            if 'df_catalog_statuses' in st.session_state:
                status_list = st.session_state['df_catalog_statuses']
                if len(status_list) == 1:
                    if status_list[0] == 'New Price':
                        st.info("📅 Using **NEW PRICE** catalog (effective 11/14/2025)")
                    else:
                        st.info(f"✅ Using **{status_list[0]}** catalog")
                else:
                    st.info(f"✅ Using **{len(status_list)} statuses**: {', '.join(status_list)}")
            
            if 'df_csv' in st.session_state:
                df_csv = st.session_state['df_csv']
                
                # Main metrics (including cost tracking)
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Total Products", f"{len(df_csv):,}")
                with col2:
                    if 'Catalog_Match_Found' in df_csv.columns:
                        matched_count = df_csv['Catalog_Match_Found'].sum()
                        match_rate = (matched_count / len(df_csv) * 100) if len(df_csv) > 0 else 0
                        st.metric("Match Rate", f"{match_rate:.1f}%")
                with col3:
                    if 'Retail_Price_Diff' in df_csv.columns:
                        retail_diff_numeric = pd.to_numeric(df_csv['Retail_Price_Diff'], errors='coerce').fillna(0)
                        sale_diff_numeric = pd.to_numeric(df_csv['Sale_Price_Diff'], errors='coerce').fillna(0) if 'Sale_Price_Diff' in df_csv.columns else pd.Series([0] * len(df_csv))
                        
                        retail_issues = retail_diff_numeric.abs() > 0.01
                        sale_issues = sale_diff_numeric.abs() > 0.01
                        price_issues = (retail_issues | sale_issues).sum()
                        st.metric("Price Issues", f"{price_issues:,}")
                with col4:
                    if 'Cost_Diff' in df_csv.columns:
                        cost_diff_numeric = pd.to_numeric(df_csv['Cost_Diff'], errors='coerce').fillna(0)
                        cost_issues = (cost_diff_numeric > 0.01).sum()
                        st.metric("Cost Issues", f"{cost_issues:,}", help="Products where Cost exceeds Max Unit Cost")
                with col5:
                    if 'Retail_Price_Diff' in df_csv.columns and 'matched_count' in locals():
                        consistency_rate = ((matched_count - price_issues) / matched_count * 100) if matched_count > 0 else 0
                        st.metric("Price Consistency", f"{consistency_rate:.1f}%")
                
                # Matching breakdown
                if 'Match_Type' in df_csv.columns:
                    st.write("**🎯 Enhanced Matching Breakdown:**")
                    match_type_counts = df_csv[df_csv['Catalog_Match_Found'] == True]['Match_Type'].value_counts()
                    
                    col1, col2, col3, col4, col5, col6 = st.columns(6)
                    with col1:
                        exact_count = match_type_counts.get('exact', 0)
                        st.metric("🎯 Exact", f"{exact_count:,}")
                    with col2:
                        pattern_count = match_type_counts.get('placeholder_pattern', 0)
                        st.metric("🔤 Pattern", f"{pattern_count:,}")
                    with col3:
                        auto_count = match_type_counts.get('brand_auto', 0) + match_type_counts.get('brand_category_auto', 0)
                        st.metric("⚡ Auto", f"{auto_count:,}")
                    with col4:
                        flower_preroll = match_type_counts.get('flower_weight_keywords', 0) + match_type_counts.get('preroll_weight_keywords', 0)
                        st.metric("🌸 Weight+KW", f"{flower_preroll:,}")
                    with col5:
                        vape_extract_count = match_type_counts.get('vape_weight_keywords', 0) + match_type_counts.get('extract_weight_keywords', 0)
                        st.metric("💨 Vape/Ext", f"{vape_extract_count:,}")
                    with col6:
                        total_matched = match_type_counts.sum()
                        st.metric("📊 Total", f"{total_matched:,}")
                
                # Pricing summary
                if 'Retail_Price_Diff' in df_csv.columns:
                    st.write("**🎯 Pricing Analysis Summary:**")
                    matched_with_prices = df_csv[(df_csv['Catalog_Match_Found'] == True) & 
                                                (df_csv['Retail_Price_Diff'].notna() | df_csv['Sale_Price_Diff'].notna())]
                    
                    if len(matched_with_prices) > 0:
                        st.write(f"• **{len(matched_with_prices):,} products** have catalog price comparisons")
                        st.write(f"• **{price_issues:,} products** need price adjustments (>$0.01 difference)")
                        st.write(f"• **{len(matched_with_prices) - price_issues:,} products** have consistent pricing")
                        
                        # Add cost analysis summary
                        if 'Cost_Diff' in df_csv.columns and cost_issues > 0:
                            st.write(f"• **{cost_issues:,} products** have costs exceeding Max Unit Cost")
                        
                        st.write(f"• Use the **Price Inspector** tab to review and export products needing fixes")
            else:
                st.info("Upload your Products CSV to see pricing analysis")
        
        tab_index += 1
        
        # Products Tab
        if 'df_csv' in st.session_state:
            with tabs[tab_index]:
                st.subheader(f"📄 {st.session_state['df_csv_name']}")
                st.info("Filtered and processed product data with smart matching and price comparison")
                
                df_csv = st.session_state['df_csv']
                
                # Summary metrics (including cost)
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("📦 Total Products", f"{len(df_csv):,}")
                with col2:
                    if 'Catalog_Location' in df_csv.columns:
                        mapped_count = df_csv['Catalog_Location'].notna().sum()
                        st.metric("🔗 Location Mapped", f"{mapped_count:,}")
                with col3:
                    if 'Catalog_Match_Found' in df_csv.columns:
                        matched_count = df_csv['Catalog_Match_Found'].sum()
                        match_rate = (matched_count / len(df_csv) * 100) if len(df_csv) > 0 else 0
                        st.metric("🎯 Catalog Matched", f"{matched_count:,} ({match_rate:.1f}%)")
                with col4:
                    if 'Retail_Price_Diff' in df_csv.columns:
                        # Convert to numeric first to avoid fillna downcasting warning
                        retail_diff_numeric = pd.to_numeric(df_csv['Retail_Price_Diff'], errors='coerce')
                        retail_issues = retail_diff_numeric.fillna(0).abs() > 0.01
                        
                        if 'Sale_Price_Diff' in df_csv.columns:
                            sale_diff_numeric = pd.to_numeric(df_csv['Sale_Price_Diff'], errors='coerce')
                            sale_issues = sale_diff_numeric.fillna(0).abs() > 0.01
                        else:
                            sale_issues = pd.Series([False] * len(df_csv))
                        
                        price_issues = (retail_issues | sale_issues).sum()
                        st.metric("💰 Price Issues", f"{price_issues:,}")
                with col5:
                    if 'Cost_Diff' in df_csv.columns:
                        cost_diff_numeric = pd.to_numeric(df_csv['Cost_Diff'], errors='coerce')
                        cost_issues = (cost_diff_numeric.fillna(0) > 0.01).sum()
                        st.metric("📈 Cost Issues", f"{cost_issues:,}")
                
                # Data table
                # v4.3.13 Fix: Convert price strings with $ to numeric for proper display
                # The data has '$34.98' strings, Streamlit needs numeric 34.98
                display_df = df_csv.copy()
                
                # Convert price columns from '$34.98' strings to 34.98 numeric
                price_columns = ['Unit Price', 'Unit Sale Price', 'Cost per Unit']
                for col in price_columns:
                    if col in display_df.columns:
                        # Convert string prices to numeric
                        def clean_price_for_display(x):
                            if pd.isna(x) or str(x).strip() in ['', 'None', 'nan']:
                                return np.nan
                            try:
                                # Remove $ and commas, convert to float
                                cleaned = str(x).replace('$', '').replace(',', '').strip()
                                return float(cleaned)
                            except:
                                return np.nan
                        
                        display_df[col] = display_df[col].apply(clean_price_for_display)
                
                st.dataframe(display_df, use_container_width=True)
                
                # Download button
                csv_buffer = io.StringIO()
                df_csv.to_csv(csv_buffer, index=False)
                
                filename = "products_with_price_comparison.csv"
                if st.session_state.get('csv_type') == 'company' and st.session_state.get('selected_shop'):
                    shop_name = st.session_state['selected_shop'].replace(' ', '_').replace('-', '_')
                    filename = f"products_{shop_name}_with_price_comparison.csv"
                elif st.session_state.get('csv_type') == 'shop' and st.session_state.get('selected_shop'):
                    shop_name = st.session_state['selected_shop'].replace(' ', '_').replace('-', '_')
                    filename = f"products_{shop_name}_with_price_comparison.csv"
                
                st.download_button(
                    label="📥 Download Processed Products",
                    data=csv_buffer.getvalue(),
                    file_name=filename,
                    mime="text/csv"
                )
            tab_index += 1
        
        # PRICE INSPECTOR TAB - FIXING THE EMPTY TAB ISSUE
        if 'df_csv' in st.session_state and 'Catalog_Match_Found' in st.session_state['df_csv'].columns:
            if st.session_state['df_csv']['Catalog_Match_Found'].sum() > 0:
                with tabs[tab_index]:
                    st.subheader("💰 Price Inspector")
                    st.info("Review products with price differences and export for bulk update")
                    
                    df_csv = st.session_state['df_csv']
                    
                    # Filter to matched products with price data
                    matched_with_prices = df_csv[
                        (df_csv['Catalog_Match_Found'] == True) & 
                        (df_csv['Retail_Price_Diff'].notna() | df_csv['Sale_Price_Diff'].notna())
                    ].copy()
                    
                    if len(matched_with_prices) == 0:
                        st.warning("No matched products with price comparisons found.")
                    else:
                        # Calculate differences with proper numeric conversion
                        retail_diff_numeric = pd.to_numeric(matched_with_prices['Retail_Price_Diff'], errors='coerce').fillna(0)
                        sale_diff_numeric = pd.to_numeric(matched_with_prices['Sale_Price_Diff'], errors='coerce').fillna(0)
                        
                        # Find products with any price difference
                        has_retail_diff = retail_diff_numeric.abs() > 0.01
                        has_sale_diff = sale_diff_numeric.abs() > 0.01
                        has_any_diff = has_retail_diff | has_sale_diff
                        
                        # Check for cost differences
                        if 'Cost_Diff' in matched_with_prices.columns:
                            cost_diff_numeric = pd.to_numeric(matched_with_prices['Cost_Diff'], errors='coerce').fillna(0)
                            has_cost_diff = cost_diff_numeric > 0.01
                        else:
                            has_cost_diff = pd.Series([False] * len(matched_with_prices))
                        
                        # Initially show products with price differences (not cost-only differences)
                        # Users can add cost differences using the filter if needed
                        products_with_differences = matched_with_prices[has_any_diff].copy()
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Matched with Prices", f"{len(matched_with_prices):,}")
                        with col2:
                            st.metric("Products with Price Issues", f"{has_any_diff.sum():,}")
                        with col3:
                            st.metric("Products with Cost Issues", f"{has_cost_diff.sum():,}")
                        with col4:
                            consistency_rate = ((len(matched_with_prices) - has_any_diff.sum()) / len(matched_with_prices) * 100) if len(matched_with_prices) > 0 else 100
                            st.metric("Price Consistency", f"{consistency_rate:.1f}%")
                        
                        # Advanced Filters
                        st.subheader("🔍 Advanced Filters")
                        
                        filter_col1, filter_col2, filter_col3 = st.columns(3)
                        
                        with filter_col1:
                            # Price Difference Type filter - now multi-select for flexibility
                            price_diff_types = st.multiselect(
                                "Show products with these differences:",
                                options=['Retail Price', 'Sale Price', 'Cost'],
                                default=['Retail Price', 'Sale Price'],  # Default to price differences only
                                help="Select which types of differences to show. Most common: Retail + Sale"
                            )
                            
                            # Inventory Status filter
                            inventory_filter = st.selectbox(
                                "Inventory Status:",
                                options=['All', 'In Stock Only', 'Out of Stock Only'],
                                help="Filter by inventory availability"
                            )
                        
                        with filter_col2:
                            # Brand filter with Include/Exclude
                            brand_mode = st.radio("Brand Filter Mode:", ['Include', 'Exclude'], horizontal=True)
                            available_brands = sorted(products_with_differences['Brand'].dropna().unique())
                            selected_brands = st.multiselect(
                                f"{'Include' if brand_mode == 'Include' else 'Exclude'} Brands:",
                                options=available_brands,
                                default=[]
                            )
                            
                            # Category filter with Include/Exclude
                            category_mode = st.radio("Category Filter Mode:", ['Include', 'Exclude'], horizontal=True)
                            available_categories = sorted(products_with_differences['Category'].dropna().unique())
                            selected_categories = st.multiselect(
                                f"{'Include' if category_mode == 'Include' else 'Exclude'} Categories:",
                                options=available_categories,
                                default=[]
                            )
                        
                        with filter_col3:
                            # Location filter with Include/Exclude
                            location_mode = st.radio("Location Filter Mode:", ['Include', 'Exclude'], horizontal=True)
                            if 'Catalog_Location' in products_with_differences.columns:
                                available_locations = sorted(products_with_differences['Catalog_Location'].dropna().unique())
                                selected_locations = st.multiselect(
                                    f"{'Include' if location_mode == 'Include' else 'Exclude'} Locations:",
                                    options=available_locations,
                                    default=[]
                                )
                            else:
                                selected_locations = []
                            
                            # Template filter with Include/Exclude
                            template_mode = st.radio("Template Filter Mode:", ['Include', 'Exclude'], horizontal=True)
                            available_templates = sorted(products_with_differences['Catalog_Template'].dropna().unique())
                            
                            # Limit display to first 50 for performance
                            if len(available_templates) > 50:
                                st.info(f"Showing first 50 of {len(available_templates)} templates")
                                available_templates = available_templates[:50]
                            
                            selected_templates = st.multiselect(
                                f"{'Include' if template_mode == 'Include' else 'Exclude'} Templates:",
                                options=available_templates,
                                default=[]
                            )
                        
                        # Apply filters
                        display_df = products_with_differences.copy()
                        
                        # Apply price difference type filter - now supports multiple selections
                        if price_diff_types:  # If any types are selected
                            # Build a mask for selected difference types
                            filter_mask = pd.Series([False] * len(display_df), index=display_df.index)
                            
                            if 'Retail Price' in price_diff_types:
                                filter_mask |= has_retail_diff[display_df.index]
                            
                            if 'Sale Price' in price_diff_types:
                                filter_mask |= has_sale_diff[display_df.index]
                            
                            if 'Cost' in price_diff_types and 'Cost_Diff' in display_df.columns:
                                filter_mask |= has_cost_diff[display_df.index]
                            
                            # Apply the combined filter
                            display_df = display_df[filter_mask]
                        else:
                            # If nothing selected, show nothing (or could show all)
                            display_df = display_df.iloc[:0]  # Empty dataframe
                        
                        # Apply inventory filter
                        if 'Inventory Available' in display_df.columns:
                            if inventory_filter == 'In Stock Only':
                                display_df = display_df[display_df['Inventory Available'] > 0]
                            elif inventory_filter == 'Out of Stock Only':
                                display_df = display_df[display_df['Inventory Available'] == 0]
                        
                        # Apply brand filter with Include/Exclude mode
                        if selected_brands:
                            if brand_mode == 'Include':
                                display_df = display_df[display_df['Brand'].isin(selected_brands)]
                            else:  # Exclude mode
                                display_df = display_df[~display_df['Brand'].isin(selected_brands)]
                        
                        # Apply category filter with Include/Exclude mode
                        if selected_categories:
                            if category_mode == 'Include':
                                display_df = display_df[display_df['Category'].isin(selected_categories)]
                            else:  # Exclude mode
                                display_df = display_df[~display_df['Category'].isin(selected_categories)]
                        
                        # Apply location filter with Include/Exclude mode
                        if selected_locations and 'Catalog_Location' in display_df.columns:
                            if location_mode == 'Include':
                                display_df = display_df[display_df['Catalog_Location'].isin(selected_locations)]
                            else:  # Exclude mode
                                display_df = display_df[~display_df['Catalog_Location'].isin(selected_locations)]
                        
                        # Apply template filter with Include/Exclude mode
                        if selected_templates:
                            if template_mode == 'Include':
                                display_df = display_df[display_df['Catalog_Template'].isin(selected_templates)]
                            else:  # Exclude mode
                                display_df = display_df[~display_df['Catalog_Template'].isin(selected_templates)]
                        
                        # Show filtered count
                        st.info(f"📊 Showing {len(display_df):,} of {len(products_with_differences):,} products with price differences")
                        
                        # Display the data
                        if len(display_df) > 0:
                            # Select display columns (including Cost columns)
                            display_columns = [
                                'Product ID', 'Brand', 'Item', 'Catalog_Template', 
                                'Catalog_Location', 'Inventory Available',
                                'Unit Price', 'Catalog_Retail_Price', 'Retail_Price_Diff',
                                'Unit Sale Price', 'Catalog_Sale_Price', 'Sale_Price_Diff',
                                'Cost per Unit', 'Catalog_Max_Unit_Cost', 'Cost_Diff',
                                'Catalog_Status_Used'
                            ]
                            
                            # Only include columns that exist
                            display_columns = [col for col in display_columns if col in display_df.columns]
                            
                            # Create a display copy with properly formatted numeric columns
                            display_for_table = display_df[display_columns].copy()
                            
                            # FIX: Convert all price columns from string to numeric for display
                            price_columns = ['Unit Price', 'Catalog_Retail_Price', 'Retail_Price_Diff', 
                                           'Unit Sale Price', 'Catalog_Sale_Price', 'Sale_Price_Diff',
                                           'Cost per Unit', 'Catalog_Max_Unit_Cost', 'Cost_Diff']
                            for col in price_columns:
                                if col in display_for_table.columns:
                                    display_for_table[col] = display_for_table[col].apply(clean_price_for_display)
                            
                            # Convert inventory to numeric
                            if 'Inventory Available' in display_for_table.columns:
                                display_for_table['Inventory Available'] = pd.to_numeric(display_for_table['Inventory Available'], errors='coerce')
                            
                            # Now apply formatting safely
                            format_dict = {
                                'Unit Price': '${:.2f}',
                                'Catalog_Retail_Price': '${:.2f}',
                                'Retail_Price_Diff': '${:+.2f}',
                                'Unit Sale Price': '${:.2f}',
                                'Catalog_Sale_Price': '${:.2f}',
                                'Sale_Price_Diff': '${:+.2f}',
                                'Cost per Unit': '${:.2f}',
                                'Catalog_Max_Unit_Cost': '${:.2f}',
                                'Cost_Diff': '${:+.2f}',
                                'Inventory Available': '{:.0f}'
                            }
                            
                            # Only apply formatting for columns that exist
                            format_dict = {k: v for k, v in format_dict.items() if k in display_for_table.columns}
                            
                            st.dataframe(
                                display_for_table.style.format(format_dict, na_rep='—'),
                                use_container_width=True
                            )
                            
                            # Export options
                            st.subheader("📥 Export Options")
                            
                            export_col1, export_col2 = st.columns(2)
                            
                            with export_col1:
                                # Standard CSV export
                                csv_buffer = io.StringIO()
                                display_df.to_csv(csv_buffer, index=False)
                                
                                st.download_button(
                                    label="📥 Download Filtered Results (CSV)",
                                    data=csv_buffer.getvalue(),
                                    file_name=f"price_differences_filtered_{len(display_df)}_products.csv",
                                    mime="text/csv"
                                )
                            
                            with export_col2:
                                # Blaze POS Export section
                                st.subheader("🔥 Blaze POS Bulk Update Export")
                                
                                # Exclusion text area
                                st.write("**Exclude Product IDs (one per line):**")
                                excluded_ids = st.text_area(
                                    "Product IDs to exclude:",
                                    height=100,
                                    help="Enter Product IDs to exclude from the Blaze POS export, one per line"
                                )
                                
                                # Process exclusions
                                excluded_list = [pid.strip() for pid in excluded_ids.split('\n') if pid.strip()]
                                
                                # Prepare Blaze export
                                blaze_df = display_df.copy()
                                
                                # Apply exclusions
                                if excluded_list:
                                    blaze_df = blaze_df[~blaze_df['Product ID'].isin(excluded_list)]
                                    st.info(f"Excluding {len(excluded_list)} Product IDs from export")
                                
                                # Filter to only products with valid catalog prices
                                blaze_df = blaze_df[
                                    (blaze_df['Catalog_Retail_Price'].notna()) | 
                                    (blaze_df['Catalog_Sale_Price'].notna())
                                ]
                                
                                if len(blaze_df) > 0:
                                    # Create Blaze format (3 columns)
                                    blaze_export = pd.DataFrame({
                                        'ProductId': blaze_df['Product ID'],
                                        'Retail Price': blaze_df['Catalog_Retail_Price'].fillna(0).round(2),
                                        'Sale Price': blaze_df.apply(
                                            lambda row: row['Catalog_Sale_Price'] if pd.notna(row['Catalog_Sale_Price']) 
                                            else row['Catalog_Retail_Price'], axis=1
                                        ).fillna(0).round(2)
                                    })
                                    
                                    # Export button
                                    csv_buffer = io.StringIO()
                                    blaze_export.to_csv(csv_buffer, index=False)
                                    
                                    st.download_button(
                                        label=f"🔥 Download Blaze POS Import ({len(blaze_export)} products)",
                                        data=csv_buffer.getvalue(),
                                        file_name=f"blaze_pos_import_{len(blaze_export)}_products.csv",
                                        mime="text/csv",
                                        type="primary"
                                    )
                                    
                                    st.success(f"✅ Ready to export {len(blaze_export)} products for Blaze POS bulk update")
                                else:
                                    st.warning("No valid products for Blaze POS export after applying filters and exclusions")
                        else:
                            st.warning("No products match the selected filters.")
                
                tab_index += 1
        
        # TROUBLESHOOTING TAB - FIXING THE EMPTY TAB ISSUE
        if 'troubleshooting_data' in st.session_state:
            with tabs[tab_index]:
                st.subheader("🔧 Troubleshooting")
                st.info("Detailed matching information for debugging and improvement")
                
                troubleshooting_df = st.session_state['troubleshooting_data']
                
                if len(troubleshooting_df) > 0:
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        match_statuses = troubleshooting_df['Match_Status'].value_counts()
                        st.metric("Total Products Processed", f"{len(troubleshooting_df):,}")
                    
                    with col2:
                        matched_count = len(troubleshooting_df[~troubleshooting_df['Catalog_Template'].isna()])
                        st.metric("Successfully Matched", f"{matched_count:,}")
                    
                    with col3:
                        unmatched_count = len(troubleshooting_df[troubleshooting_df['Catalog_Template'].isna()])
                        st.metric("Unmatched", f"{unmatched_count:,}")
                    
                    # Match status breakdown
                    st.subheader("📊 Match Status Breakdown")
                    status_counts = troubleshooting_df['Match_Status'].value_counts()
                    
                    # Create a more readable display
                    status_df = pd.DataFrame({
                        'Match Status': status_counts.index,
                        'Count': status_counts.values,
                        'Percentage': (status_counts.values / len(troubleshooting_df) * 100).round(2)
                    })
                    
                    st.dataframe(status_df, use_container_width=True, hide_index=True)
                    
                    # Filter options
                    st.subheader("🔍 Filter Troubleshooting Data")
                    
                    filter_col1, filter_col2 = st.columns(2)
                    
                    with filter_col1:
                        # Filter by match status
                        available_statuses = troubleshooting_df['Match_Status'].unique()
                        selected_status = st.selectbox(
                            "Filter by Match Status:",
                            options=['All'] + list(available_statuses),
                            help="Show only products with specific match status"
                        )
                    
                    with filter_col2:
                        # Filter by shop
                        if 'Shop' in troubleshooting_df.columns:
                            available_shops = troubleshooting_df['Shop'].dropna().unique()
                            selected_shop = st.selectbox(
                                "Filter by Shop:",
                                options=['All'] + list(available_shops),
                                help="Show only products from specific shop"
                            )
                        else:
                            selected_shop = 'All'
                    
                    # Apply filters
                    display_troubleshooting = troubleshooting_df.copy()
                    
                    if selected_status != 'All':
                        display_troubleshooting = display_troubleshooting[
                            display_troubleshooting['Match_Status'] == selected_status
                        ]
                    
                    if selected_shop != 'All' and 'Shop' in display_troubleshooting.columns:
                        display_troubleshooting = display_troubleshooting[
                            display_troubleshooting['Shop'] == selected_shop
                        ]
                    
                    st.info(f"Showing {len(display_troubleshooting):,} of {len(troubleshooting_df):,} products")
                    
                    # Display the data
                    st.dataframe(display_troubleshooting, use_container_width=True, height=600)
                    
                    # Export option
                    csv_buffer = io.StringIO()
                    display_troubleshooting.to_csv(csv_buffer, index=False)
                    
                    st.download_button(
                        label="📥 Download Troubleshooting Data",
                        data=csv_buffer.getvalue(),
                        file_name="troubleshooting_data.csv",
                        mime="text/csv"
                    )
                    
                    # Unmatched products analysis
                    unmatched_df = troubleshooting_df[troubleshooting_df['Catalog_Template'].isna()]
                    
                    if len(unmatched_df) > 0:
                        st.subheader("❌ Unmatched Products Analysis")
                        
                        # Group by brand
                        unmatched_by_brand = unmatched_df['Brand'].value_counts().head(10)
                        
                        st.write("**Top 10 Brands with Unmatched Products:**")
                        brand_analysis_df = pd.DataFrame({
                            'Brand': unmatched_by_brand.index,
                            'Unmatched Count': unmatched_by_brand.values
                        })
                        st.dataframe(brand_analysis_df, use_container_width=True, hide_index=True)
                        
                        # Sample of unmatched products
                        st.write("**Sample of Unmatched Products (first 20):**")
                        sample_unmatched = unmatched_df[['Brand', 'Item', 'Shop', 'Notes']].head(20)
                        st.dataframe(sample_unmatched, use_container_width=True, hide_index=True)
                else:
                    st.warning("No troubleshooting data available. Run matching first.")
            
            tab_index += 1
        
        # PRODUCT CATALOG TAB - FIXING THE EMPTY TAB ISSUE
        if 'df_catalog' in st.session_state:
            with tabs[tab_index]:
                st.subheader(f"📋 {st.session_state.get('df_catalog_name', 'Product Catalog')}")
                st.info("Reference catalog data used for matching and price comparison")
                
                df_catalog = st.session_state['df_catalog']
                
                # Show catalog statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Templates", f"{len(df_catalog):,}")
                
                with col2:
                    if 'Brand' in df_catalog.columns:
                        unique_brands = df_catalog['Brand'].nunique()
                        st.metric("Unique Brands", f"{unique_brands:,}")
                
                with col3:
                    if 'Category' in df_catalog.columns:
                        unique_categories = df_catalog['Category'].nunique()
                        st.metric("Categories", f"{unique_categories:,}")
                
                with col4:
                    if 'Status' in df_catalog.columns:
                        active_count = len(df_catalog[df_catalog['Status'] == 'Active'])
                        st.metric("Active Products", f"{active_count:,}")
                
                # Filter options
                st.subheader("🔍 Filter Catalog")
                
                filter_col1, filter_col2, filter_col3 = st.columns(3)
                
                with filter_col1:
                    # Brand filter
                    if 'Brand' in df_catalog.columns:
                        catalog_brands = ['All'] + sorted(df_catalog['Brand'].dropna().unique())
                        selected_catalog_brand = st.selectbox(
                            "Filter by Brand:",
                            options=catalog_brands[:100],  # Limit to 100 for performance
                            help="Filter catalog by brand"
                        )
                    else:
                        selected_catalog_brand = 'All'
                
                with filter_col2:
                    # Category filter
                    if 'Category' in df_catalog.columns:
                        catalog_categories = ['All'] + sorted(df_catalog['Category'].dropna().unique())
                        selected_catalog_category = st.selectbox(
                            "Filter by Category:",
                            options=catalog_categories,
                            help="Filter catalog by category"
                        )
                    else:
                        selected_catalog_category = 'All'
                
                with filter_col3:
                    # Status filter
                    if 'Status' in df_catalog.columns:
                        catalog_statuses = ['All'] + sorted(df_catalog['Status'].dropna().unique())
                        selected_catalog_status = st.selectbox(
                            "Filter by Status:",
                            options=catalog_statuses,
                            help="Filter catalog by status"
                        )
                    else:
                        selected_catalog_status = 'All'
                
                # Apply filters
                display_catalog = df_catalog.copy()
                
                if selected_catalog_brand != 'All' and 'Brand' in display_catalog.columns:
                    display_catalog = display_catalog[display_catalog['Brand'] == selected_catalog_brand]
                
                if selected_catalog_category != 'All' and 'Category' in display_catalog.columns:
                    display_catalog = display_catalog[display_catalog['Category'] == selected_catalog_category]
                
                if selected_catalog_status != 'All' and 'Status' in display_catalog.columns:
                    display_catalog = display_catalog[display_catalog['Status'] == selected_catalog_status]
                
                st.info(f"Showing {len(display_catalog):,} of {len(df_catalog):,} catalog entries")
                
                # Display the catalog
                st.dataframe(display_catalog, use_container_width=True, height=600)
                
                # Download option
                csv_buffer = io.StringIO()
                display_catalog.to_csv(csv_buffer, index=False)
                
                st.download_button(
                    label="📥 Download Filtered Catalog",
                    data=csv_buffer.getvalue(),
                    file_name=f"product_catalog_filtered_{len(display_catalog)}_items.csv",
                    mime="text/csv"
                )
    else:
        # Welcome screen
        st.info("👆 Upload your Products CSV in the sidebar to get started")
        
        st.subheader("📄 Data Processing Workflow")
        
        st.markdown(f"""
        **🎯 Product Price Checker v{VERSION} Features:**
        
        1. **🎯 Improved Filtering (v4.3.15)**
           - ✅ Multi-select filter for price differences
           - ✅ Choose any combination: Retail, Sale, Cost
           - ✅ Defaults to Retail + Sale for typical workflow
           - ✅ More flexible than single-select options
        
        2. **💰 Enhanced Cost Tracking (v4.3.14)**
           - ✅ Cost per Unit from Company Products
           - ✅ Max Unit Cost comparison from Product Catalog
           - ✅ Cost Issues metric showing products over budget
           - ✅ Fixed Price Inspector display issues
        
        3. **🏪 12 Store Support**
           - ✅ All Haven locations mapped
           - ✅ Including new Hawthorne store
           - ✅ Automatic shop detection
        
        4. **🧠 Smart Matching Engine**
           - ✅ 90%+ match accuracy
           - ✅ Weight/keyword extraction
           - ✅ Wildcard pattern matching (COLOR, STRAIN, FLAVOR)
           - ✅ Category-specific logic
        
        5. **💰 Advanced Price Comparison**
           - ✅ Retail and Sale price tracking
           - ✅ Cost vs Max Unit Cost comparison
           - ✅ Multi-status catalog support
           - ✅ Automatic fallback to .5 price fields
        
        6. **📊 Price Inspector Features**
           - ✅ Advanced filtering options
           - ✅ Include/Exclude modes
           - ✅ Inventory status filters
           - ✅ Price difference type selection
        
        7. **🔥 Blaze POS Export**
           - ✅ 3-column format ready for bulk upload
           - ✅ Auto-fills blank sale prices with retail
           - ✅ Product exclusion capability
           - ✅ Exact headers: ProductId,Retail Price,Sale Price
        """)

if __name__ == "__main__":
    main()