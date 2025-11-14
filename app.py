"""
Product Price Checker v4.2.5
Smart brand matching and price comparison tool for cannabis retail products
Now with automatic CSV type detection, shop filtering, and Blaze POS export

CHANGELOG:
v4.2.5 (2025-11-13)
- CRITICAL FIX: Changed Status filter from single to multi-select
- Can now select multiple statuses: Active, New Price, New Product, etc.
- Default includes Active + New Price + New Product (52 more products now get pricing!)
- Fixes issue where products with "New Product" status had no catalog prices

v4.2.4 (2025-11-13)
- Added Catalog Template filter to Price Inspector
- Added Category filter to Price Inspector
- Enhanced filtering capabilities for better price analysis

v4.2.3 (2025-11-13)
- CRITICAL FIX: Deduplicate catalog templates to fix Stiiizy and other multi-status brands
- Added wildcard matching for COLOR, STRAIN, FLAVOR patterns
- Wildcard matching runs after exact match but before auto-matching
- Fixes products like "Plug Play - Blue Steel Battery" to match "Plug Play - COLOR Steel Battery"

v4.2.2 (2025-11-13)
- Fixed DtypeWarning by adding low_memory=False to CSV reads
- Fixed pandas attribute warning by storing troubleshooting data in session state
- Fixed FutureWarning for fillna downcasting by using pd.to_numeric first
- Replaced deprecated use_container_width with width='stretch'
- Code cleanup for production readiness

v4.2.1 (2025-11-13)
- CRITICAL FIX: Catalog now loads ALL statuses for matching (includes DNO products)
- Status filter now only affects price comparison, not matching
- Products with DNO status can now be matched properly
- Added "Catalog_Status_Used" column to show which status was used for pricing
- Warning shown when matched products don't have selected status pricing

v4.2.0 (2025-11-13)
- Added Status filter to choose between "Active" and "New Price" catalog prices
- Added Blaze POS Export (Product ID, Retail Price, Sale Price format)
- Added Product ID to Price Inspector display
- Status-aware catalog loading with date-based pricing

v4.1.3 (2025-01-XX)
- Fixed shop name mapping for Corona (HAVEN - Corona vs Haven - Corona)

v4.1.2 (2025-01-XX)
- Fixed weight extraction to handle weights not at end of string (e.g., "3.75g 5pk")
- Fixed pack size extraction to handle both "5pk 3.75g" and "3.75g 5pk" formats
- Improved preroll matching logic to prioritize pack size over weight
- Pack size now correctly acts as strong distinguishing characteristic

v4.1.1 (2025-01-XX)
- Fixed CSV type detection display issues
- Added version number display in sidebar
- Added changelog documentation
- Improved shop selector visibility for single shop exports
- Enhanced detection messaging

v4.1.0 (2025-01-XX)
- Added automatic CSV type detection (Company vs Shop)
- Added shop filtering for company exports
- Added shop identification for single shop exports
- Updated catalog location mapping for both CSV types

v4.0.0 (2025-01-XX)
- Smart brand structure matching
- Enhanced matching for Flower, Preroll, Vape, Extract categories
- Weight and pack size extraction
- Category keyword extraction
- Price comparison functionality
- Troubleshooting tab
"""

import streamlit as st
import pandas as pd
import io
import re
from google.oauth2.service_account import Credentials
import gspread
from gspread_dataframe import get_as_dataframe

# Configure page
st.set_page_config(
    page_title="Product Price Checker v4.2.5",
    page_icon="🛒",
    layout="wide"
)

# Configuration
VERSION = "4.2.5"
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
    'HAVEN - Corona': 'Corona'
}

# Exact match brands that require product-level matching
EXACT_PRODUCT_MATCH_BRANDS = {
    'Blazy Susan', 'Camino', 'Crave', 'Daily Dose', "Dr. Norm's", 'Good Tide', 
    'Happy Fruit', 'High Gorgeous', 'Kiva', 'Lost Farm', 'Made From Dirt', 
    'Papa & Barkley', 'Sip Elixirs', 'St. Ides', "Uncle Arnie's", 'Vet CBD', 
    'Wyld', 'Yummi Karma', "Not Your Father's"
}

def detect_csv_type(df):
    """
    Detect if CSV is a Company export (multi-shop) or Shop export (single shop)
    Returns: 'company' if Shop column exists, 'shop' if not, None if invalid
    """
    if df is None or df.empty:
        return None
    
    # Check if first column is 'Shop'
    first_col = str(df.columns[0]).strip()
    
    if first_col == 'Shop':
        return 'company'
    elif first_col == 'SKU':
        return 'shop'
    else:
        return None

def get_unique_shops(df):
    """Extract unique shop names from company export"""
    if df is None or 'Shop' not in df.columns:
        return []
    
    shops = df['Shop'].dropna().unique()
    return sorted([str(shop).strip() for shop in shops if str(shop).strip() and str(shop) != 'nan'])

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

def extract_gid_from_url(sheet_url):
    """Extract the gid (worksheet ID) from a Google Sheets URL"""
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

def match_placeholder_pattern(product_name, template_name):
    """
    Check if a product name matches a template with placeholder patterns
    
    Handles placeholders like:
    - STRAIN (any strain name)
    - COLOR (any color)
    - FLAVOR (any flavor)
    - SIZE (any size)
    
    Args:
        product_name: Actual product name (e.g., "Plug Play - Blue Steel Battery")
        template_name: Template with placeholder (e.g., "Plug Play - COLOR Steel Battery")
    
    Returns:
        bool: True if product matches template pattern
    
    Examples:
        >>> match_placeholder_pattern("Plug Play - Blue Steel Battery", "Plug Play - COLOR Steel Battery")
        True
        >>> match_placeholder_pattern("Camino - Watermelon Lemonade Gummies", "Camino - FLAVOR Gummies")
        True
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
    
    # For each placeholder, try to match
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

@st.cache_data
def load_google_sheet_data(sheet_url, load_all_for_matching=False):
    """
    Load data from Google Sheets using service account authentication
    
    Args:
        sheet_url: Google Sheets URL
        load_all_for_matching: If True, load ALL statuses (for matching).
                              If False, load only Active + New Price (for price comparison)
    
    Returns:
        tuple: (DataFrame, worksheet_name)
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
                    df = None  # This indicates headers are wrong
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
            
            # NEW LOGIC: Different filtering based on purpose
            if 'Status' in df.columns:
                original_count = len(df)
                
                if load_all_for_matching:
                    # For matching: Load ALL statuses (including DNO)
                    # Don't filter at all
                    st.info(f"📋 Loaded {len(df)} products (ALL statuses) for matching")
                else:
                    # For price display: Only Active + New Price (exclude DNO, On-boarding, etc)
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
    """Load data from uploaded CSV file"""
    try:
        df = pd.read_csv(uploaded_file, skiprows=1, low_memory=False)
        return df, "Company Products"
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        return None, None

def filter_company_products(df, connect_catalog_df=None, selected_shop=None, csv_type='company'):
    """Filter company products data by Active field, shop (if company export), keep only specified columns, and filter by brands"""
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
    categories_to_exclude = [
        'Display', 'Clones', 'Apparel', 'Sample', 'Promo', 'Compassion', 
        'Donation', 'Boxes', 'Non-Cannabis', 'Gift Cards', 'xxxDONOTUSE-Buzzers'
    ]
    
    if 'Category' in active_df.columns:
        before_category_filter = len(active_df)
        active_df = active_df[~active_df['Category'].isin(categories_to_exclude)]
        after_category_filter = len(active_df)
        removed_count = before_category_filter - after_category_filter
        st.write(f"**After excluding unwanted categories**: {active_df.shape}")
        if removed_count > 0:
            st.info(f"🚫 Excluded {removed_count} products from categories: {', '.join(categories_to_exclude)}")
    
    # Keep only essential columns
    columns_to_keep = [
        'Shop', 'SKU', 'Item', 'Category', 'Cannabis', 'Measurement',
        'Cost per Unit', 'Unit Price', 'Unit Sale Price', 'Product ID',
        'Brand', 'Cannabis Type', 'Weight Per Unit', 'Custom Weight Measurement',
        'Custom Weight Type', 'Active', 'Available Online', 'Sell Type',
        'Master Product ID', 'Company Product ID', 'Inventory Available'
    ]
    
    existing_columns = [col for col in columns_to_keep if col in active_df.columns]
    missing_columns = [col for col in columns_to_keep if col not in active_df.columns]
    
    if missing_columns:
        st.warning(f"Missing columns: {missing_columns}")
    
    filtered_df = active_df[existing_columns].copy()
    st.write(f"**After column filtering**: {filtered_df.shape}")
    
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
    """Add a 'Catalog_Location' column to products"""
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
        # For shop exports, we need to use the selected shop name for mapping
        if selected_shop and selected_shop in SHOP_NAME_MAPPING:
            catalog_location = SHOP_NAME_MAPPING[selected_shop]
            df_copy['Catalog_Location'] = catalog_location
            st.info(f"✅ Mapped all {len(df_copy):,} products to catalog location: {catalog_location}")
        else:
            st.warning(f"⚠️ Cannot map shop '{selected_shop}' to catalog location")
    
    return df_copy

def normalize_categories(df):
    """Normalize category names to match Product Catalog categories"""
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

# [MATCHING FUNCTIONS CONTINUE - Same as original, keeping all the smart matching logic]
# I'll include the key functions here but they remain unchanged from v4.1.3


def match_wildcard_template(item_text, template, wildcards=['COLOR', 'STRAIN', 'FLAVOR']):
    """
    Match item against template with wildcards (COLOR, STRAIN, FLAVOR)
    Returns (match_found, extracted_values) tuple
    
    Example:
        item: "Plug Play - Blue Steel Battery"
        template: "Plug Play - COLOR Steel Battery"
        returns: (True, {'COLOR': 'Blue'})
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
    # Escape special regex characters except wildcards
    pattern = re.escape(template_str)
    
    # Replace escaped wildcards with capture groups
    # Match one or more words (can be multi-word like "Blue Dream")
    for wildcard in wildcards:
        escaped_wildcard = re.escape(wildcard)
        if escaped_wildcard in pattern:
            # Match word characters, spaces, and hyphens (for strain/flavor names)
            pattern = pattern.replace(escaped_wildcard, r'([\w\s\-]+?)', 1)
    
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
            # CRITICAL FIX: Only add template if not already in list (deduplicates across Active/New Price statuses)
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
        
        # 2. Try placeholder pattern match (NEW!)
        if not match_found and brand in brand_catalog_map:
            for template in brand_catalog_map[brand]:
                if match_placeholder_pattern(item, template):
                    matched_df.at[idx, 'Catalog_Match_Found'] = True
                    matched_df.at[idx, 'Catalog_Template'] = template
                    matched_df.at[idx, 'Match_Type'] = 'placeholder_pattern'
                    matched_df.at[idx, 'Match_Strategy'] = 'placeholder_pattern'
                    exact_matches += 1  # Count with exact matches since it's very reliable
                    match_found = True
                    troubleshooting_data.append({
                        'Brand': brand,
                        'Item': item,
                        'Shop': shop,
                        'Match_Status': 'Placeholder pattern match',
                        'Catalog_Template': template,
                        'Catalog_Options': f"{len(brand_catalog_map[brand])} options",
                        'Notes': 'Matched via placeholder (COLOR, STRAIN, FLAVOR, etc.)'
                    })
                    break
        
        # 2. Try wildcard match (NEW - FIX #2)
        if not match_found and brand in brand_catalog_map:
            for template in brand_catalog_map[brand]:
                is_wildcard_match, extracted_wildcards = match_wildcard_template(item, template)
                if is_wildcard_match:
                    matched_df.at[idx, 'Catalog_Match_Found'] = True
                    matched_df.at[idx, 'Catalog_Template'] = template
                    matched_df.at[idx, 'Match_Type'] = 'wildcard'
                    matched_df.at[idx, 'Match_Strategy'] = 'wildcard'
                    matched_df.at[idx, 'Match_Keywords'] = ', '.join([f"{k}={v}" for k, v in extracted_wildcards.items()])
                    wildcard_matches += 1
                    match_found = True
                    troubleshooting_data.append({
                        'Brand': brand,
                        'Item': item,
                        'Shop': shop,
                        'Match_Status': 'Wildcard match',
                        'Catalog_Template': template,
                        'Catalog_Options': f"{len(brand_catalog_map[brand])} options",
                        'Notes': f'Matched wildcards: {", ".join([f"{k}={v}" for k, v in extracted_wildcards.items()])}'
                    })
                    break
        
        # Skip auto-matching for exact product match brands
        skip_auto_matching = brand in EXACT_PRODUCT_MATCH_BRANDS
        
        # 3. Try single entry brand auto-match
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
        
        # 4. Try brand+category auto-match
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
        
        # 5. Try advanced weight/keyword matching for complex categories
        if not match_found and category in ['Flower', 'Preroll', 'Vape', 'Extract'] and brand in multiple_entry_brands:
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
    
    progress_bar.progress(1.0)
    
    # Show results
    total_matches = exact_matches + wildcard_matches + single_entry_matches + brand_category_matches + flower_weight_matches + preroll_matches + vape_extract_matches
    total_match_rate = (total_matches / len(matched_df)) * 100 if len(matched_df) > 0 else 0
    
    st.success(f"🎉 Enhanced Matching Results:")
    
    # Count placeholder pattern matches separately
    placeholder_pattern_count = matched_df[matched_df['Match_Type'] == 'placeholder_pattern'].shape[0]
    exact_only_count = exact_matches - placeholder_pattern_count
    
    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
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
        st.metric("📊 Total", f"{total_matches:,} ({total_match_rate:.1f}%)")
    
    # Store troubleshooting data in session state instead of as DataFrame attribute
    st.session_state['troubleshooting_data'] = pd.DataFrame(troubleshooting_data)
    
    return matched_df

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

def add_simple_price_comparison(company_df, catalog_df, selected_statuses=['Active']):
    """
    Simple price comparison - add basic price difference columns
    Only uses prices from catalog products with the selected statuses
    
    Args:
        company_df: Products dataframe
        catalog_df: Full catalog (all statuses)
        selected_statuses: List of statuses to use for pricing (default: ['Active'])
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
    
    def clean_price(price_str):
        """Clean and convert price string to float"""
        if pd.isna(price_str) or price_str == '':
            return None
        try:
            cleaned = str(price_str).replace('$', '').replace(',', '').strip()
            return float(cleaned)
        except:
            return None
    
    # Initialize price comparison columns
    company_df['Catalog_Retail_Price'] = None
    company_df['Catalog_Sale_Price'] = None
    company_df['Retail_Price_Diff'] = None
    company_df['Sale_Price_Diff'] = None
    company_df['Catalog_Status_Used'] = None
    
    # Build catalog lookup - filter by selected statuses (can be multiple)
    catalog_for_pricing = catalog_df[catalog_df['Status'].isin(selected_statuses)].copy() if 'Status' in catalog_df.columns else catalog_df
    
    st.info(f"🔍 {len(catalog_for_pricing)} catalog products have selected status(es) for pricing")
    
    catalog_lookup = {}
    for _, cat_row in catalog_for_pricing.iterrows():
        template = cat_row['Profile Template']
        if pd.notna(template):
            # Store both the row data and which status it has
            catalog_lookup[template] = {
                'data': cat_row,
                'status': cat_row.get('Status', 'Unknown')
            }
    
    pricing_issues = 0
    matched_but_no_pricing = 0
    
    # Compare prices for each matched product
    for idx, row in matched_products.iterrows():
        catalog_template = row['Catalog_Template']
        catalog_location = row['Catalog_Location']
        
        if pd.isna(catalog_template) or pd.isna(catalog_location):
            continue
        
        catalog_entry = catalog_lookup.get(catalog_template)
        
        if catalog_entry is None:
            # Product matched to a template, but that template doesn't have any of the selected statuses
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
        
        # Calculate differences
        if catalog_retail is not None and company_retail is not None:
            company_df.at[idx, 'Retail_Price_Diff'] = company_retail - catalog_retail
        
        if catalog_sale is not None and company_sale is not None:
            company_df.at[idx, 'Sale_Price_Diff'] = company_sale - catalog_sale
        
        # Count pricing issues (differences > $0.01)
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
    
    return company_df

def main():
    st.title(f"🛒 Product Price Checker v{VERSION}")
    st.markdown("Filter your products and connect to Product Catalog data with automatic CSV type detection")
    
    st.sidebar.header("📊 Data Sources")
    
    # Add Status Filter at the top of sidebar
    st.sidebar.subheader("📋 Catalog Price Sources")
    
    # Define available statuses (most common first)
    available_statuses = ['Active', 'New Price', 'New Product', 'DNO', 'REVIEW', 'On-boarding', 'Inactive']
    
    catalog_statuses = st.sidebar.multiselect(
        "Which catalog statuses to use for pricing:",
        options=available_statuses,
        default=['Active', 'New Price', 'New Product'],
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
        st.session_state['catalog_statuses'] = ['Active', 'New Price', 'New Product']
    
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
            st.sidebar.info(f"📍 Will process only: {selected_shop}")
    
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
        st.sidebar.success(f"📍 Data source: {selected_shop}")
    
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
        **v4.2.5** (Current - 2025-11-13)
        - 🔧 CRITICAL: Multi-status filter (was single radio)
        - ✅ Can select Active + New Price + New Product
        - 🎯 Fixes 52 products with "New Product" status
        
        **v4.2.4** (2025-11-13)
        - 🔍 NEW: Catalog Template filter in Price Inspector
        - 🔍 NEW: Category filter in Price Inspector
        - ✅ Enhanced filtering for price analysis
        
        **v4.2.3** (2025-11-13)
        - 🔧 CRITICAL FIX: Deduplicate catalog templates for Stiiizy
        - 🎨 NEW: Wildcard matching for COLOR/STRAIN/FLAVOR
        - ✅ Fixes "Plug Play - Blue Steel Battery" matching
        
        **v4.2.2** (2025-11-13)
        - Code cleanup: Fixed all warnings
        - DtypeWarning fixes
        - Pandas attribute warning fix
        - FutureWarning fixes
        - Deprecated API updates
        
        **v4.2.1** (2025-11-13)
        - CRITICAL: Catalog loads ALL statuses for matching
        - DNO products now match correctly
        - Status filter only affects pricing
        - Shows which status used for each price
        
        **v4.2.0** (2025-11-13)
        - Added Status filter (Active vs New Price)
        - Added Blaze POS Export format
        - Added Product ID to Price Inspector
        - Status-aware pricing
        
        **v4.1.3**
        - Fixed Corona shop name mapping
        
        **v4.1.2**
        - Fixed weight/pack extraction
        - Improved preroll matching
        
        **v4.1.1**
        - Fixed CSV type detection
        - Enhanced shop selector
        
        **v4.1.0**
        - Auto CSV type detection
        - Shop filtering
        
        **v4.0.0**
        - Smart brand matching
        - Price comparison
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
                
                # Main metrics
                col1, col2, col3, col4 = st.columns(4)
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
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
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
                
                # Data table
                st.dataframe(df_csv, width='stretch')
                
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
        
        # Price Inspector Tab - WITH NEW BLAZE POS EXPORT
        if 'df_csv' in st.session_state and 'Catalog_Match_Found' in st.session_state['df_csv'].columns:
            matched_data = st.session_state['df_csv'][st.session_state['df_csv']['Catalog_Match_Found'] == True]
            if len(matched_data) > 0:
                with tabs[tab_index]:
                    st.subheader("💰 Price Inspector")
                    
                    # Show which catalog status is being used
                    if 'df_catalog_statuses' in st.session_state:
                        status_list = st.session_state['df_catalog_statuses']
                        if len(status_list) == 1:
                            if status_list[0] == 'New Price':
                                st.warning("📅 Showing prices from **NEW PRICE** catalog (effective 11/14/2025)")
                            else:
                                st.info(f"✅ Showing prices from **{status_list[0]}** catalog")
                        else:
                            st.info(f"✅ Showing prices from **{len(status_list)} statuses**: {', '.join(status_list)}")
                    
                    st.info("Review matched products and identify pricing discrepancies")
                    
                    # Filters - Row 1
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        show_price_issues_only = st.checkbox(
                            "Show Only Price Issues (any difference)",
                            value=False,
                            help="Filter to only products with any retail or sale price differences"
                        )
                        show_in_stock_only = st.checkbox(
                            "Show Only In-stock Products",
                            value=False,
                            help="Filter to only products with Inventory Available > 0"
                        )
                    with col2:
                        selected_brands = st.multiselect(
                            "Filter by Brand:",
                            options=sorted(matched_data['Brand'].unique()),
                            default=None
                        )
                    with col3:
                        if 'Catalog_Location' in matched_data.columns:
                            selected_locations = st.multiselect(
                                "Filter by Location:",
                                options=sorted(matched_data['Catalog_Location'].dropna().unique()),
                                default=None
                            )
                        else:
                            selected_locations = None
                    
                    # Filters - Row 2 (NEW)
                    col4, col5 = st.columns(2)
                    with col4:
                        if 'Catalog_Template' in matched_data.columns:
                            selected_templates = st.multiselect(
                                "Filter by Catalog Template:",
                                options=sorted(matched_data['Catalog_Template'].dropna().unique()),
                                default=None,
                                help="Filter to specific catalog product templates"
                            )
                        else:
                            selected_templates = None
                    with col5:
                        if 'Category' in matched_data.columns:
                            selected_categories = st.multiselect(
                                "Filter by Category:",
                                options=sorted(matched_data['Category'].dropna().unique()),
                                default=None,
                                help="Filter to specific product categories"
                            )
                        else:
                            selected_categories = None
                    
                    # Apply filters
                    filtered_matches = matched_data.copy()
                    
                    if show_price_issues_only:
                        if 'Retail_Price_Diff' in filtered_matches.columns:
                            retail_diff_numeric = pd.to_numeric(filtered_matches['Retail_Price_Diff'], errors='coerce').fillna(0)
                            sale_diff_numeric = pd.to_numeric(filtered_matches['Sale_Price_Diff'], errors='coerce').fillna(0) if 'Sale_Price_Diff' in filtered_matches.columns else pd.Series([0] * len(filtered_matches))
                            
                            retail_issues = retail_diff_numeric != 0
                            sale_issues = sale_diff_numeric != 0
                            price_issue_mask = retail_issues | sale_issues
                            filtered_matches = filtered_matches[price_issue_mask]
                    
                    if show_in_stock_only:
                        if 'Inventory Available' in filtered_matches.columns:
                            inventory_numeric = pd.to_numeric(filtered_matches['Inventory Available'], errors='coerce').fillna(0)
                            in_stock_mask = inventory_numeric > 0
                            filtered_matches = filtered_matches[in_stock_mask]
                    
                    if selected_brands:
                        filtered_matches = filtered_matches[filtered_matches['Brand'].isin(selected_brands)]
                    if selected_locations and 'Catalog_Location' in filtered_matches.columns:
                        filtered_matches = filtered_matches[filtered_matches['Catalog_Location'].isin(selected_locations)]
                    if selected_templates and 'Catalog_Template' in filtered_matches.columns:
                        filtered_matches = filtered_matches[filtered_matches['Catalog_Template'].isin(selected_templates)]
                    if selected_categories and 'Category' in filtered_matches.columns:
                        filtered_matches = filtered_matches[filtered_matches['Category'].isin(selected_categories)]
                    
                    st.write(f"Showing {len(filtered_matches)} of {len(matched_data)} matched products")
                    
                    # Display filtered data
                    if len(filtered_matches) > 0:
                        display_columns = [
                            'Product ID', 'Brand', 'Item', 'Catalog_Template', 'Catalog_Location', 'Inventory Available',
                            'Unit Price', 'Catalog_Retail_Price', 'Retail_Price_Diff',
                            'Unit Sale Price', 'Catalog_Sale_Price', 'Sale_Price_Diff'
                        ]
                        
                        available_columns = [col for col in display_columns if col in filtered_matches.columns]
                        display_df = filtered_matches[available_columns].copy()
                        
                        # Format numeric columns
                        numeric_price_columns = ['Catalog_Retail_Price', 'Retail_Price_Diff', 'Catalog_Sale_Price', 'Sale_Price_Diff']
                        for col in numeric_price_columns:
                            if col in display_df.columns:
                                display_df[col] = pd.to_numeric(display_df[col], errors='coerce').round(2)
                        
                        st.dataframe(display_df, width='stretch')
                        
                        # Download buttons row
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Standard filtered data export
                            csv_buffer = io.StringIO()
                            display_df.to_csv(csv_buffer, index=False)
                            
                            filter_description = []
                            if show_price_issues_only:
                                filter_description.append("Price Issues")
                            if show_in_stock_only:
                                filter_description.append("In-Stock")
                            if selected_brands:
                                filter_description.append(f"{len(selected_brands)} Brand(s)")
                            if selected_locations:
                                filter_description.append(f"{len(selected_locations)} Location(s)")
                            if selected_templates:
                                filter_description.append(f"{len(selected_templates)} Template(s)")
                            if selected_categories:
                                filter_description.append(f"{len(selected_categories)} Category(s)")
                            
                            if filter_description:
                                download_label = f"📥 Download Filtered Data ({', '.join(filter_description)})"
                                filename = f"price_inspector_filtered_{len(filtered_matches)}_products.csv"
                            else:
                                download_label = "📥 Download All Matched Products"
                                filename = f"price_inspector_all_{len(filtered_matches)}_products.csv"
                            
                            st.download_button(
                                label=download_label,
                                data=csv_buffer.getvalue(),
                                file_name=filename,
                                mime="text/csv"
                            )
                        
                        with col2:
                            # NEW: Blaze POS Export
                            st.markdown("**🔄 Blaze POS Bulk Update**")
                            
                            # Create Blaze POS format: Product ID, Retail Price, Sale Price
                            blaze_export = filtered_matches[['Product ID']].copy()
                            blaze_export['Retail Price'] = filtered_matches['Catalog_Retail_Price']
                            blaze_export['Sale Price'] = filtered_matches['Catalog_Sale_Price']
                            
                            # Remove rows where Product ID is missing
                            blaze_export = blaze_export[blaze_export['Product ID'].notna()]
                            
                            # Format prices properly (remove $ and ensure numeric)
                            blaze_export['Retail Price'] = pd.to_numeric(blaze_export['Retail Price'], errors='coerce')
                            blaze_export['Sale Price'] = pd.to_numeric(blaze_export['Sale Price'], errors='coerce')
                            
                            blaze_csv = io.StringIO()
                            blaze_export.to_csv(blaze_csv, index=False)
                            
                            # Get status label for filename
                            catalog_statuses_list = st.session_state.get('df_catalog_statuses', ['Active'])
                            if len(catalog_statuses_list) == 1:
                                catalog_status_label = catalog_statuses_list[0]
                            else:
                                catalog_status_label = 'multi_status'
                            
                            blaze_filename = f"blaze_pos_update_{catalog_status_label.lower().replace(' ', '_')}_{len(blaze_export)}_products.csv"
                            
                            st.download_button(
                                label=f"📤 Download Blaze POS Update ({len(blaze_export)} products)",
                                data=blaze_csv.getvalue(),
                                file_name=blaze_filename,
                                mime="text/csv",
                                help=f"3-column format for Blaze POS bulk price updates using {catalog_status_label} prices"
                            )
                            
                            if len(blaze_export) < len(filtered_matches):
                                st.caption(f"⚠️ {len(filtered_matches) - len(blaze_export)} products excluded due to missing Product ID")
                    else:
                        st.info("No products match the selected filters.")
                
                tab_index += 1
        
        # Troubleshooting Tab (unchanged)
        if 'df_csv' in st.session_state and 'troubleshooting_data' in st.session_state:
            with tabs[tab_index]:
                st.subheader("🔧 Matching Troubleshooting")
                st.info("Debug unsuccessful matching results and identify improvement opportunities")
                
                troubleshooting_df = st.session_state['troubleshooting_data']
                
                unsuccessful_statuses = ['Missing brand or item']
                unsuccessful_matches = troubleshooting_df[troubleshooting_df['Match_Status'].isin(unsuccessful_statuses)]
                
                status_counts = unsuccessful_matches['Match_Status'].value_counts()
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    missing_data_count = status_counts.get('Missing brand or item', 0)
                    st.metric("📊 Missing Data", missing_data_count)
                with col2:
                    total_unsuccessful = len(unsuccessful_matches)
                    st.metric("🔧 Total Issues", total_unsuccessful)
                with col3:
                    total_records = len(troubleshooting_df)
                    success_rate = ((total_records - total_unsuccessful) / total_records * 100) if total_records > 0 else 0
                    st.metric("✅ Success Rate", f"{success_rate:.1f}%")
                
                # Filter options
                status_filter = st.selectbox(
                    "Filter by Issue Type:",
                    options=['All Issues'] + list(status_counts.index),
                    index=0
                )
                
                filtered_troubleshooting = unsuccessful_matches.copy()
                if status_filter != 'All Issues':
                    filtered_troubleshooting = filtered_troubleshooting[filtered_troubleshooting['Match_Status'] == status_filter]
                
                # Show problematic brands
                if len(filtered_troubleshooting) > 0:
                    st.write("**🔍 Brands with Most Unmatched Products:**")
                    brand_unmatched = filtered_troubleshooting['Brand'].value_counts().head(10)
                    st.bar_chart(brand_unmatched)
                    
                    st.write("**🎯 Top Problematic Brand Examples:**")
                    for brand_name in brand_unmatched.head(3).index:
                        brand_examples = filtered_troubleshooting[filtered_troubleshooting['Brand'] == brand_name].head(3)
                        st.write(f"**{brand_name}** ({brand_unmatched[brand_name]} unmatched):")
                        for _, example in brand_examples.iterrows():
                            st.write(f"  • \"{example['Item']}\" - {example['Notes']}")
                else:
                    st.success("🎉 No unsuccessful matches found!")
                
                # Detailed troubleshooting data
                st.write(f"**📋 Unsuccessful Match Details ({len(filtered_troubleshooting)} records):**")
                if len(filtered_troubleshooting) > 0:
                    st.dataframe(filtered_troubleshooting, width='stretch')
                    
                    csv_buffer = io.StringIO()
                    filtered_troubleshooting.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="📥 Download Unsuccessful Matches",
                        data=csv_buffer.getvalue(),
                        file_name="unsuccessful_matches_troubleshooting.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No unsuccessful matches to display!")
            tab_index += 1
        
        # Product Catalog Tab (unchanged)
        if 'df_catalog' in st.session_state:
            with tabs[tab_index]:
                st.subheader(f"📋 {st.session_state['df_catalog_name']}")
                st.info("Reference catalog data - Brand column used for brand extraction, Profile Template for matching")
                st.dataframe(st.session_state['df_catalog'], width='stretch')
                
                csv_buffer = io.StringIO()
                st.session_state['df_catalog'].to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Product Catalog",
                    data=csv_buffer.getvalue(),
                    file_name="connect_product_catalog_reference.csv",
                    mime="text/csv"
                )
    
    else:
        # Welcome screen (unchanged - truncated for length)
        st.info("👆 Upload your Products CSV in the sidebar to get started")
        
        st.subheader("📄 Data Processing Workflow")
        
        st.markdown(f"""
        **🎯 Product Price Checker v{VERSION} Features:**
        
        1. **📋 Status Filter (NEW!)**
           - ✅ Choose between "Active" (current) or "New Price" (11/14/2025) catalog prices
           - ✅ See which price set is being used throughout the app
           - ✅ Perfect for preparing price changes before they go live
        
        2. **📤 Blaze POS Export (NEW!)**
           - ✅ 3-column format: Product ID, Retail Price, Sale Price
           - ✅ Ready for direct bulk upload to Blaze POS
           - ✅ Uses catalog prices based on selected status
        
        3. **📄 Automatic CSV Type Detection**
           - ✅ Auto-detects Company Export (multi-shop) or Shop Export (single-shop)
           - ✅ Company Export: Select specific shop or process all shops
           - ✅ Shop Export: Identify which shop the data is from
        
        4. **🧠 Smart Matching & Processing**
           - ✅ Filters out inactive products
           - ✅ Excludes unwanted categories
           - ✅ Cross-references with Product Catalog brands
           - ✅ Advanced weight/keyword matching
           - ✅ Price comparison with catalog
        """)

if __name__ == "__main__":
    main()