import streamlit as st
import pandas as pd
import io
import json
import re
from google.oauth2.service_account import Credentials
import gspread
from gspread_dataframe import get_as_dataframe

# Configure page
st.set_page_config(
    page_title="Product Price Checker",
    page_icon="🛒",
    layout="wide"
)

# Default Google Sheets URLs
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
    'Haven - Corona': 'Corona'
}

def extract_weight_from_item(item_text):
    """Extract weight from item text (e.g., "Blue Dream 3.5g" → "3.5g")"""
    if pd.isna(item_text):
        return None
    
    item_str = str(item_text).strip()
    weight_patterns = [
        r'(\d+\.?\d*g)$',      # 3.5g, 7g, 28g
        r'(\d+\.\d+\s?oz?)$',  # 3.5 oz, 3.5oz
        r'(\d+\s?oz?)$',       # 1 oz, 1oz
        r'(1/8\s?oz?)$',       # 1/8 oz
        r'(1/4\s?oz?)$',       # 1/4 oz
        r'(1/2\s?oz?)$',       # 1/2 oz
    ]
    
    for pattern in weight_patterns:
        match = re.search(pattern, item_str, re.IGNORECASE)
        if match:
            return match.group(1).lower().replace(' ', '')
    
    return None

def extract_category_keywords(item_text, category):
    """Extract category-specific distinguishing keywords from item text"""
    if pd.isna(item_text) or pd.isna(category):
        return None
    
    item_str = str(item_text).lower()
    
    # Define distinguishing keywords by category
    vape_keywords = [
        'originals', 'ascnd', 'dna', 'exotics', 'disposable', 'live resin',
        'reload', 'rtu', 'curepen', 'curebar'
    ]
    
    if category.lower() == 'vape':
        keywords_to_check = vape_keywords
        # Find keywords present in the item text
        found_keywords = []
        for keyword in keywords_to_check:
            if keyword in item_str:
                found_keywords.append(keyword)
        return ', '.join(found_keywords) if found_keywords else None
        
    elif category.lower() == 'extract':
        # Targeted extraction based on actual catalog patterns
        found_keywords = []
        
        # Extract compound methods (order matters - check compounds first)
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
        
        # Extract tier levels (only for specific brands that use them)
        if any(brand in item_str for brand in ['bear labs', 'west coast cure']):
            tier_match = re.search(r'tier\s*([1-4])', item_str)
            if tier_match:
                found_keywords.append(f"tier {tier_match.group(1)}")
        
        # Extract modifiers
        modifiers = ['cold cure', 'fresh press', 'curated', 'hte blend', 'dino eggz']
        for modifier in modifiers:
            if modifier in item_str:
                found_keywords.append(modifier)
        
        # Extract consistencies
        consistencies = ['diamonds', 'budder', 'badder', 'sauce', 'sugar', 'jam']
        for consistency in consistencies:
            if consistency in item_str:
                found_keywords.append(consistency)
        
        # Extract product types
        product_types = ['rso', 'syringe']
        for product_type in product_types:
            if product_type in item_str:
                found_keywords.append(product_type)
        
        return ', '.join(found_keywords) if found_keywords else None
    
    else:
        return None

def extract_pack_size_from_item(item_text):
    """Extract pack size from item text (e.g., "OG Kush 3pk 1.5g" → "3pk")"""
    if pd.isna(item_text):
        return None
    
    item_str = str(item_text).strip()
    pack_patterns = [
        r'(\d+pk)\s+\d+\.?\d*g',      # 3pk 1.5g, 5pk 2.5g
        r'(\d+pk)\s+\d+\s?oz',        # 3pk 1 oz
        r'(\d+pk)\s+1/[248]\s?oz',    # 3pk 1/8 oz
    ]
    
    for pattern in pack_patterns:
        match = re.search(pattern, item_str, re.IGNORECASE)
        if match:
            return match.group(1).lower()
    
    return None

def extract_preroll_keywords(item_text):
    """Extract preroll-specific keywords from item text"""
    if pd.isna(item_text):
        return None
    
    item_str = str(item_text).lower()
    found_keywords = []
    
    # Product types
    preroll_types = ['blunts', 'preroll', 'prerolls', 'joints', 'mini']
    for preroll_type in preroll_types:
        if preroll_type in item_str:
            found_keywords.append(preroll_type)
    
    # Infused keyword
    if 'infused' in item_str:
        found_keywords.append('infused')
    
    return ', '.join(found_keywords) if found_keywords else None

def extract_gid_from_url(sheet_url):
    """
    Extract the gid (worksheet ID) from a Google Sheets URL
    """
    try:
        if 'gid=' in sheet_url:
            # Extract gid from URL like: ...?gid=95140308#gid=95140308
            gid_part = sheet_url.split('gid=')[1]
            # Handle both ?gid=123 and #gid=123 formats
            if '&' in gid_part:
                gid = gid_part.split('&')[0]
            elif '#' in gid_part:
                gid = gid_part.split('#')[0]
            else:
                gid = gid_part
            return int(gid)
        return None
    except:
        return None

@st.cache_data
def load_google_sheet_data(sheet_url, worksheet_name=None):
    """
    Load data from Google Sheets using service account authentication
    No filtering - pure reference data
    """
    try:
        # Get credentials from Streamlit secrets
        credentials_dict = st.secrets["google_sheets"]
        
        # Create credentials object
        creds = Credentials.from_service_account_info(
            credentials_dict,
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets.readonly",
                "https://www.googleapis.com/auth/drive.readonly"
            ]
        )
        
        # Initialize gspread client
        client = gspread.authorize(creds)
        
        # Open the spreadsheet
        sheet = client.open_by_url(sheet_url)
        
        # Try to get worksheet by gid from URL first
        gid = extract_gid_from_url(sheet_url)
        worksheet = None
        
        if gid:
            try:
                # Find worksheet by gid
                for ws in sheet.worksheets():
                    if ws.id == gid:
                        worksheet = ws
                        break
                if worksheet:
                    st.info(f"📋 Found worksheet by GID: {worksheet.title} (ID: {gid})")
                else:
                    st.warning(f"⚠️ Could not find worksheet with GID {gid}, using first worksheet")
                    worksheet = sheet.get_worksheet(0)
            except Exception as e:
                st.warning(f"⚠️ Error finding worksheet by GID: {str(e)}, using first worksheet")
                worksheet = sheet.get_worksheet(0)
        elif worksheet_name:
            worksheet = sheet.worksheet(worksheet_name)
        else:
            worksheet = sheet.get_worksheet(0)
        
        # Debug: Get raw data to inspect structure
        st.write(f"🔍 **Debug:** Inspecting {worksheet.title} structure...")
        
        # Get first few rows to understand structure
        try:
            raw_data = worksheet.get_all_values()
            if raw_data:
                st.write(f"🔍 **Debug:** Total rows in sheet: {len(raw_data)}")
                st.write(f"🔍 **Debug:** Row 0 (should be title): {raw_data[0][:5] if len(raw_data[0]) > 5 else raw_data[0]}")
                if len(raw_data) > 1:
                    st.write(f"🔍 **Debug:** Row 1 (should be headers): {raw_data[1][:5] if len(raw_data[1]) > 5 else raw_data[1]}")
                if len(raw_data) > 2:
                    st.write(f"🔍 **Debug:** Row 2 (should be data): {raw_data[2][:5] if len(raw_data[2]) > 5 else raw_data[2]}")
        except Exception as e:
            st.error(f"Error inspecting raw data: {str(e)}")
        
        # Try different approaches to get the data correctly
        df = None
        
        # Approach 1: Try header=0 (row 0 as headers) - This should be correct for Product Catalog
        try:
            df = get_as_dataframe(worksheet, parse_dates=True, header=0)
            st.info(f"✅ Loaded using header=0, shape: {df.shape}")
            if len(df.columns) > 0:
                # Check if this looks like proper headers (not data)
                first_col = str(df.columns[0]).lower()
                if first_col in ['status', 'active'] and 'retail price' in str(df.columns).lower():
                    st.success("✅ Headers look correct - found expected column patterns")
                elif first_col == 'active' and 'almora' in str(df.columns).lower():
                    st.warning("⚠️ This looks like data as headers - trying alternative approach")
                    df = None  # Try next approach
        except Exception as e:
            st.warning(f"⚠️ header=0 failed: {str(e)}")
        
        # Approach 2: If that fails, try header=1 (row 1 as headers)
        if df is None or df.empty:
            try:
                df = get_as_dataframe(worksheet, parse_dates=True, header=1)
                st.info(f"✅ Loaded using header=1, shape: {df.shape}")
            except Exception as e:
                st.warning(f"⚠️ header=1 failed: {str(e)}")
        
        # Approach 3: Manual approach - get all data and set headers manually
        if df is None or df.empty:
            try:
                all_values = worksheet.get_all_values()
                if len(all_values) > 1:
                    # Use row 0 as headers, data starts from row 1
                    headers = all_values[0]  # Row 0 as headers  
                    data_rows = all_values[1:]  # Rows 1+ as data
                    df = pd.DataFrame(data_rows, columns=headers)
                    st.info(f"✅ Loaded manually (row 0 as headers), shape: {df.shape}")
            except Exception as e:
                st.error(f"❌ Manual loading failed: {str(e)}")
        
        if df is not None:
            # Show column info for debugging
            st.write(f"🔍 **Debug:** Columns found: {list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}")
            
            # Clean up empty rows and columns
            original_shape = df.shape
            df = df.dropna(how='all').dropna(axis=1, how='all')
            st.write(f"🔍 **Debug:** After cleanup: {original_shape} → {df.shape}")
            
            return df, worksheet.title
        else:
            st.error("❌ Failed to load data with any method")
            return None, None
        
    except Exception as e:
        st.error(f"Error loading Google Sheet: {str(e)}")
        return None, None

def load_csv_data(uploaded_file):
    """
    Load data from uploaded CSV file
    """
    try:
        # Read the CSV, skipping the first row (row 0)
        df = pd.read_csv(uploaded_file, skiprows=1)
        return df, "Company Products"
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        return None, None

def normalize_brand_name(brand_name):
    """Normalize brand names to handle apostrophe and quote variations"""
    if pd.isna(brand_name):
        return brand_name
    
    # Convert to string and normalize different apostrophe types
    normalized = str(brand_name).strip()
    # Replace various apostrophe/quote characters with standard apostrophe
    apostrophe_chars = [''', ''', '`', '"', '"']
    for char in apostrophe_chars:
        normalized = normalized.replace(char, "'")
    
    return normalized

def filter_company_products(df, connect_catalog_df=None):
    """
    Filter company products data by Active field, keep only specified columns,
    and filter by brands that exist in Product Catalog
    """
    if df is None or df.empty:
        return None
    
    data_source_name = "Company Products"
    st.write(f"**{data_source_name}** - Original data shape: {df.shape}")
    
    # Step 1: Filter by Active field - keep only rows where Active is not "No" or "False"
    if 'Active' in df.columns:
        df['Active'] = df['Active'].astype(str).str.strip()
        active_df = df[~df['Active'].isin(['No', 'False', 'no', 'false', 'NO', 'FALSE', 'N', 'n'])]
        st.write(f"**{data_source_name}** - After filtering by Active field: {active_df.shape}")
    else:
        st.warning(f"No 'Active' column found in {data_source_name}. Using all data.")
        active_df = df.copy()
    
    # Step 1.5: Filter out unwanted categories
    categories_to_exclude = [
        'Display', 'Clones', 'Apparel', 'Sample', 'Promo', 'Compassion', 
        'Donation', 'Boxes', 'Non-Cannabis', 'Gift Cards', 'xxxDONOTUSE-Buzzers'
    ]
    
    if 'Category' in active_df.columns:
        before_category_filter = len(active_df)
        active_df = active_df[~active_df['Category'].isin(categories_to_exclude)]
        after_category_filter = len(active_df)
        removed_count = before_category_filter - after_category_filter
        st.write(f"**{data_source_name}** - After excluding unwanted categories: {active_df.shape}")
        if removed_count > 0:
            st.info(f"🚫 Excluded {removed_count} products from categories: {', '.join(categories_to_exclude)}")
    else:
        st.warning(f"No 'Category' column found in {data_source_name}. Category filtering skipped.")
    
    # Step 2: Define the columns to keep
    columns_to_keep = [
        'Shop', 'SKU', 'Item', 'Category', 'Cannabis', 'Measurement',
        'Cost per Unit', 'Unit Price', 'Unit Sale Price', 'Product ID',
        'Brand', 'Cannabis Type', 'Weight Per Unit', 'Custom Weight Measurement',
        'Custom Weight Type', 'Active', 'Available Online', 'Sell Type',
        'Master Product ID', 'Company Product ID', 'Inventory Available'
    ]
    
    # Keep only the specified columns (only if they exist in the dataframe)
    existing_columns = [col for col in columns_to_keep if col in active_df.columns]
    missing_columns = [col for col in columns_to_keep if col not in active_df.columns]
    
    if missing_columns:
        st.warning(f"**{data_source_name}** - Missing columns: {missing_columns}")
    
    filtered_df = active_df[existing_columns].copy()
    st.write(f"**{data_source_name}** - After column filtering: {filtered_df.shape}")
    
    # Step 3: Filter by brands from Product Catalog
    if connect_catalog_df is not None and not connect_catalog_df.empty and 'Brand' in filtered_df.columns:
        if 'Brand' in connect_catalog_df.columns:
            # Get unique brands from Product Catalog Brand column and normalize them
            valid_brands_raw = connect_catalog_df['Brand'].dropna().unique()
            valid_brands = [normalize_brand_name(brand) for brand in valid_brands_raw if str(brand).strip() and str(brand) != 'nan']
            valid_brands = [brand for brand in valid_brands if brand]  # Remove any empty results
            
            st.write(f"🔍 **Debug:** Found {len(valid_brands)} unique brands from Product Catalog: {valid_brands[:10]}{'...' if len(valid_brands) > 10 else ''}")
            
            # Normalize company product brand names for comparison
            filtered_df['Brand_Normalized'] = filtered_df['Brand'].apply(normalize_brand_name)
            
            # Filter company products to only include brands that exist in Product Catalog
            before_brand_filter = len(filtered_df)
            filtered_df = filtered_df[filtered_df['Brand_Normalized'].isin(valid_brands)]
            after_brand_filter = len(filtered_df)
            
            # Drop the temporary normalized column
            filtered_df = filtered_df.drop('Brand_Normalized', axis=1)
            
            st.write(f"**{data_source_name}** - After brand filtering: {filtered_df.shape}")
            st.info(f"🎯 Filtered to only include {len(valid_brands)} brands from Product Catalog. Removed {before_brand_filter - after_brand_filter} products.")
        else:
            st.warning("⚠️ No 'Brand' column found in Product Catalog. Brand filtering skipped.")
    elif connect_catalog_df is None:
        st.info("📋 Product Catalog not loaded. Brand filtering skipped.")
    elif connect_catalog_df.empty:
        st.warning("⚠️ Product Catalog is empty. Brand filtering skipped.")
    elif 'Brand' not in filtered_df.columns:
        st.warning("⚠️ No 'Brand' column found in Company Products. Brand filtering skipped.")
    
    # Add source identifier
    filtered_df['Data_Source'] = data_source_name
    
    # Step 4: Extract Weight, Pack Size, and Category Keywords for enhanced matching
    st.info("🔍 Extracting Weight, Pack Size, and Category Keywords for enhanced matching...")
    filtered_df['Extracted_Weight'] = filtered_df['Item'].apply(extract_weight_from_item)
    filtered_df['Extracted_Pack_Size'] = filtered_df['Item'].apply(extract_pack_size_from_item)
    filtered_df['Extracted_Category_Keywords'] = filtered_df.apply(
        lambda row: extract_category_keywords(row['Item'], row['Category']), axis=1
    )
    
    # Extract preroll-specific keywords for preroll products
    preroll_mask = filtered_df['Category'] == 'Preroll'
    filtered_df['Extracted_Preroll_Keywords'] = None
    if preroll_mask.any():
        filtered_df.loc[preroll_mask, 'Extracted_Preroll_Keywords'] = filtered_df.loc[preroll_mask, 'Item'].apply(extract_preroll_keywords)
    
    # Report extraction results
    weight_extracted_count = filtered_df['Extracted_Weight'].notna().sum()
    pack_extracted_count = filtered_df['Extracted_Pack_Size'].notna().sum()
    keywords_extracted_count = filtered_df['Extracted_Category_Keywords'].notna().sum()
    preroll_keywords_extracted_count = filtered_df['Extracted_Preroll_Keywords'].notna().sum() if preroll_mask.any() else 0
    st.info(f"🔍 Extracted weights from {weight_extracted_count:,} products")
    st.info(f"📦 Extracted pack sizes from {pack_extracted_count:,} products")
    st.info(f"🔤 Extracted category keywords from {keywords_extracted_count:,} products")
    if preroll_keywords_extracted_count > 0:
        st.info(f"🚬 Extracted preroll keywords from {preroll_keywords_extracted_count:,} preroll products")
    
    st.write(f"**{data_source_name}** - Final filtered data shape: {filtered_df.shape}")
    
    return filtered_df

def add_catalog_location_mapping(df):
    """
    Add a 'Catalog_Location' column to Company Products for easier price comparison
    """
    if df is None or df.empty or 'Shop' not in df.columns:
        return df
    
    df_copy = df.copy()
    df_copy['Catalog_Location'] = df_copy['Shop'].map(SHOP_NAME_MAPPING)
    
    # Report mapping results
    unmapped_shops = df_copy[df_copy['Catalog_Location'].isna()]['Shop'].unique()
    if len(unmapped_shops) > 0:
        st.warning(f"⚠️ Unmapped shops found: {list(unmapped_shops)}")
    
    mapped_count = df_copy['Catalog_Location'].notna().sum()
    total_count = len(df_copy)
    st.info(f"✅ Shop mapping: {mapped_count}/{total_count} products mapped to catalog locations")
    
    return df_copy

def normalize_categories(df):
    """
    Normalize category names to match Product Catalog categories
    """
    if df is None or df.empty or 'Category' not in df.columns:
        return df
    
    df_copy = df.copy()
    
    # Normalize flower categories
    flower_mapping = {
        'Flower (Indica)': 'Flower',
        'Flower (Sativa)': 'Flower', 
        'Flower (Hybrid)': 'Flower'
    }
    
    # Apply the mapping
    original_categories = df_copy['Category'].value_counts()
    df_copy['Category'] = df_copy['Category'].replace(flower_mapping)
    updated_categories = df_copy['Category'].value_counts()
    
    # Report changes
    normalized_count = 0
    for old_cat, new_cat in flower_mapping.items():
        if old_cat in original_categories:
            count = original_categories[old_cat]
            normalized_count += count
            st.info(f"📂 Normalized {count:,} products: '{old_cat}' → '{new_cat}'")
    
    if normalized_count > 0:
        st.success(f"✅ Category normalization: {normalized_count:,} products updated")
    
    return df_copy

def add_smart_brand_matching(company_df, catalog_df):
    """
    Smart brand-based matching using actual catalog structure:
    - Single catalog entry per brand → direct auto-match
    - Multiple catalog entries per brand → brand + category auto-matching first
    - Still multiple after brand+category (Flower) → weight-based matching
    - Exact product match brands (Camino, Blazy Susan, etc.) → skip auto-matching, exact only
    """
    if company_df is None or catalog_df is None:
        return company_df
    
    st.info("🧠 Starting smart brand structure matching...")
    
    # Create a copy to avoid modifying original data
    matched_df = company_df.copy()
    
    # Initialize matching columns
    matched_df['Catalog_Match_Found'] = False
    matched_df['Catalog_Template'] = None
    matched_df['Match_Type'] = None
    matched_df['Match_Strategy'] = None  # 'single_entry', 'keyword_match', 'exact'
    matched_df['Match_Keywords'] = None  # For debugging keyword matches
    
    # Step 1: Analyze catalog brand structure (including categories)
    st.write("📊 Analyzing Product Catalog brand and category structure...")
    
    brand_catalog_map = {}
    brand_category_catalog_map = {}
    
    # Debug: Track brands that are skipped
    skipped_brands = []
    debug_target = "Not Your Father's"
    found_target_rows = []
    
    for idx, cat_row in catalog_df.iterrows():
        brand = cat_row['Brand']
        template = cat_row['Profile Template']
        category = cat_row.get('Category', 'Unknown')
        
        # Debug: Track what happens to our target brand
        if pd.notna(brand) and debug_target.lower() in str(brand).lower():
            found_target_rows.append({
                'row': idx,
                'brand': brand,
                'template': template,
                'template_is_na': pd.isna(template),
                'template_stripped': str(template).strip() if pd.notna(template) else None,
                'category': category
            })
        
        if pd.notna(brand):
            if pd.notna(template) and str(template).strip():  # Also check for empty strings
                # Brand-only mapping (existing logic)
                if brand not in brand_catalog_map:
                    brand_catalog_map[brand] = []
                brand_catalog_map[brand].append(template)
                
                # Brand + Category mapping (new logic)
                brand_category_key = f"{brand}|{category}"
                if brand_category_key not in brand_category_catalog_map:
                    brand_category_catalog_map[brand_category_key] = []
                brand_category_catalog_map[brand_category_key].append(template)
            else:
                # Brand exists but no template - this is the likely issue
                if brand not in skipped_brands:
                    skipped_brands.append(brand)
    
    # Debug output for target brand
    if found_target_rows:
        st.write(f"🔍 **Debug: Found {len(found_target_rows)} rows for '{debug_target}':**")
        for row_data in found_target_rows:
            st.write(f"  Row {row_data['row']}: Brand='{row_data['brand']}', Template='{row_data['template']}', Category='{row_data['category']}'")
        
        # Check if it made it to the brand map
        if debug_target in brand_catalog_map:
            st.write(f"✅ **'{debug_target}' successfully added to brand_catalog_map with {len(brand_catalog_map[debug_target])} templates**")
        else:
            st.write(f"❌ **'{debug_target}' NOT found in brand_catalog_map**")
    else:
        st.write(f"🔍 **Debug: No rows found for '{debug_target}' in catalog analysis**")
    
    # Show skipped brands for debugging
    if skipped_brands:
        st.warning(f"⚠️ **Brands with missing Profile Templates:** {', '.join(skipped_brands)}")
        st.info("💡 These brands exist in the catalog but have no Profile Template, so they can't be used for matching.")
    
    # Separate single vs multiple entry brands
    single_entry_brands = {}
    multiple_entry_brands = {}
    
    for brand, templates in brand_catalog_map.items():
        if len(templates) == 1:
            single_entry_brands[brand] = templates[0]
        else:
            multiple_entry_brands[brand] = templates
    
    # Analyze brand + category combinations
    single_entry_brand_categories = {}
    multiple_entry_brand_categories = {}
    
    for brand_category_key, templates in brand_category_catalog_map.items():
        if len(templates) == 1:
            single_entry_brand_categories[brand_category_key] = templates[0]
        else:
            multiple_entry_brand_categories[brand_category_key] = templates

    # Define brands that should use exact product matching (same list as in matching logic)
    EXACT_PRODUCT_MATCH_BRANDS = {
        'Blazy Susan', 'Camino', 'Crave', 'Daily Dose', "Dr. Norm's", 'Good Tide', 
        'Happy Fruit', 'High Gorgeous', 'Kiva', 'Lost Farm', 'Made From Dirt', 
        'Papa & Barkley', 'Sip Elixirs', 'St. Ides', "Uncle Arnie's", 'Vet CBD', 
        'Wyld', 'Yummi Karma'
    }
    
    # Filter out exact product match brands from auto-match categories for accurate debug display
    filtered_single_entry_brands = {brand: template for brand, template in single_entry_brands.items() 
                                  if brand not in EXACT_PRODUCT_MATCH_BRANDS}
    filtered_single_entry_brand_categories = {key: template for key, template in single_entry_brand_categories.items() 
                                            if key.split('|')[0] not in EXACT_PRODUCT_MATCH_BRANDS}
    
    # Display analysis results
    total_brands = len(brand_catalog_map)
    single_count = len(filtered_single_entry_brands)
    multiple_count = len(multiple_entry_brands)
    
    single_brand_category_count = len(filtered_single_entry_brand_categories)
    multiple_brand_category_count = len(multiple_entry_brand_categories)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📋 Total Brands", total_brands)
    with col2:
        st.metric("1️⃣ Single Entry", single_count)
    with col3:
        st.metric("🔀 Multiple Entry", multiple_count)
    
    # Show brand + category analysis (filtered)
    st.write(f"**📂 Brand + Category Analysis:**")
    st.write(f"• **{len(filtered_single_entry_brand_categories)}** brand+category combinations have single catalog entries (auto-match)")
    st.write(f"• **{multiple_brand_category_count}** brand+category combinations need specific rules")
    st.write(f"• **{len(EXACT_PRODUCT_MATCH_BRANDS)}** exact-product-match brands (skip auto-matching)")
    
    # Show examples (filtered to exclude exact product match brands)
    if len(filtered_single_entry_brands) > 0:
        st.write("**🎯 Sample Single Entry Brands (auto-match):**")
        sample_single = list(filtered_single_entry_brands.items())[:3]
        for brand, template in sample_single:
            st.write(f"• **{brand}**: \"{template}\"")
    
    if len(filtered_single_entry_brand_categories) > 0:
        st.write("**📂 Sample Single Brand+Category Combinations (auto-match):**")
        sample_brand_categories = list(filtered_single_entry_brand_categories.items())[:3]
        for brand_category_key, template in sample_brand_categories:
            brand, category = brand_category_key.split('|')
            st.write(f"• **{brand} + {category}**: \"{template}\"")
    
    if len(EXACT_PRODUCT_MATCH_BRANDS) > 0:
        st.write("**🎯 Sample Exact Product Match Brands (exact matching only):**")
        exact_brands_in_catalog = [brand for brand in list(EXACT_PRODUCT_MATCH_BRANDS)[:5] if brand in brand_catalog_map]
        for brand in exact_brands_in_catalog:
            entry_count = len(brand_catalog_map[brand])
            st.write(f"• **{brand}**: {entry_count} catalog entries (exact match required)")
    
    if multiple_brand_category_count > 0:
        st.write("**🔧 Sample Multi-Entry Brand+Category Combinations (need rules):**")
        # Filter out exact product match brands from this display too
        filtered_multiple = {key: templates for key, templates in multiple_entry_brand_categories.items() 
                           if key.split('|')[0] not in EXACT_PRODUCT_MATCH_BRANDS}
        sample_multiple = list(filtered_multiple.items())[:3]
        for brand_category_key, templates in sample_multiple:
            brand, category = brand_category_key.split('|')
            st.write(f"• **{brand} + {category}**: {len(templates)} options")
            for template in templates[:2]:
                st.write(f"  - \"{template}\"")
            if len(templates) > 2:
                st.write(f"  - ... and {len(templates) - 2} more")
    
    # Step 3: Match company products
    st.write(f"\n🔍 Matching {len(matched_df):,} company products...")
    
    # Counters for different match types
    exact_matches = 0
    single_entry_matches = 0
    brand_category_matches = 0
    flower_weight_matches = 0
    preroll_matches = 0
    vape_extract_matches = 0
    no_matches = 0
    
    # Create a progress bar
    progress_bar = st.progress(0)
    
    # Troubleshooting data collection
    troubleshooting_data = []
    
    for counter, (idx, row) in enumerate(matched_df.iterrows()):
        # Update progress every 1000 items
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
        
        # Strategy 1: Try exact match first (fastest) - case insensitive
        # For exact product match brands, we ignore category completely
        if brand in brand_catalog_map:
            for template in brand_catalog_map[brand]:
                if item.lower() == template.lower():  # Case insensitive comparison
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
                        'Catalog_Options': f"{len(brand_catalog_map[brand])} options (all categories)",
                        'Notes': 'Perfect match (case insensitive, category ignored)'
                    })
                    break
        
        # Strategy 1.5: Exact product matching brands - skip auto-matching
        if not match_found and brand in EXACT_PRODUCT_MATCH_BRANDS:
            # These brands should only match exactly or through keyword matching
            # Skip single entry and brand+category auto-matching
            skip_auto_matching = True
        else:
            skip_auto_matching = False
        
        # Strategy 2: Single entry brand auto-match (if no exact match and not skipping)
        if not match_found and not skip_auto_matching and brand in single_entry_brands:
            template = single_entry_brands[brand]
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
        
        # Strategy 3: Brand + Category auto-match - skip for exact product match brands
        if not match_found and not skip_auto_matching and brand in multiple_entry_brands:
            brand_category_key = f"{brand}|{category}"
            if brand_category_key in single_entry_brand_categories:
                template = single_entry_brand_categories[brand_category_key]
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
        
        # Strategy 4: Category-specific matching for Flower, Preroll, Vape, and Extract
        if not match_found and category in ['Flower', 'Preroll', 'Vape', 'Extract'] and brand in multiple_entry_brands:
            brand_category_key = f"{brand}|{category}"
            if brand_category_key in multiple_entry_brand_categories:
                templates = multiple_entry_brand_categories[brand_category_key]
                
                if category == 'Flower':
                    # Flower: Weight matching only
                    company_weight = row.get('Extracted_Weight')
                    
                    if company_weight:
                        # Find catalog templates with matching weight
                        weight_matched_templates = []
                        for template in templates:
                            catalog_weight = extract_weight_from_item(template)
                            if catalog_weight == company_weight:
                                weight_matched_templates.append(template)
                        
                        # If exactly one template matches the weight, use it
                        if len(weight_matched_templates) == 1:
                            template = weight_matched_templates[0]
                            matched_df.at[idx, 'Catalog_Match_Found'] = True
                            matched_df.at[idx, 'Catalog_Template'] = template
                            matched_df.at[idx, 'Match_Type'] = 'flower_weight'
                            matched_df.at[idx, 'Match_Strategy'] = 'flower_weight'
                            matched_df.at[idx, 'Match_Keywords'] = f'weight: {company_weight}'
                            flower_weight_matches += 1
                            match_found = True
                            troubleshooting_data.append({
                                'Brand': brand,
                                'Item': item,
                                'Shop': shop,
                                'Match_Status': 'Flower weight match',
                                'Catalog_Template': template,
                                'Catalog_Options': f"{len(templates)} total, 1 with {company_weight}",
                                'Notes': f'Matched by weight: {company_weight}'
                            })
                
                elif category == 'Preroll':
                    # Preroll: Infused keyword → Weight → Pack Size → Product Type
                    company_has_infused = 'infused' in str(item).lower()
                    
                    # Step 1: Filter by infused keyword
                    infused_filtered_templates = []
                    for template in templates:
                        template_has_infused = 'infused' in str(template).lower()
                        if company_has_infused == template_has_infused:
                            infused_filtered_templates.append(template)
                    
                    current_templates = infused_filtered_templates if infused_filtered_templates else templates
                    match_steps = []
                    if infused_filtered_templates:
                        match_steps.append(f"infused: {'yes' if company_has_infused else 'no'}")
                    
                    # Step 2: Weight matching
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
                    
                    # Step 3: Pack size matching
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
                    
                    # Step 3.5: No pack size fallback matching
                    elif not company_pack and len(current_templates) > 1:
                        # Company has no pack size, prefer templates that also have no pack size
                        no_pack_templates = []
                        for template in current_templates:
                            catalog_pack = extract_pack_size_from_item(template)
                            if not catalog_pack:  # Template also has no pack size
                                no_pack_templates.append(template)
                        
                        if len(no_pack_templates) == 1:
                            current_templates = no_pack_templates
                            match_steps.append("no pack (fallback)")
                        elif len(no_pack_templates) > 0 and len(no_pack_templates) < len(current_templates):
                            # Some templates have no pack size, prefer those
                            current_templates = no_pack_templates
                            match_steps.append("no pack (partial fallback)")
                    
                    # Step 4: Product type matching (blunts vs prerolls)
                    company_preroll_keywords = row.get('Extracted_Preroll_Keywords')
                    if company_preroll_keywords and len(current_templates) > 1:
                        company_preroll_keyword_list = [kw.strip() for kw in str(company_preroll_keywords).split(',')]
                        # Remove 'infused' since that's handled in step 1
                        company_type_keywords = [kw for kw in company_preroll_keyword_list if kw != 'infused']
                        
                        if company_type_keywords:
                            # Score each template by preroll type keyword matches
                            template_scores = []
                            for template in current_templates:
                                catalog_preroll_keywords = extract_preroll_keywords(template)
                                if catalog_preroll_keywords:
                                    catalog_keyword_list = [kw.strip() for kw in catalog_preroll_keywords.split(',')]
                                    catalog_type_keywords = [kw for kw in catalog_keyword_list if kw != 'infused']
                                    # Count matching type keywords
                                    matches = sum(1 for ck in company_type_keywords if ck in catalog_type_keywords)
                                    template_scores.append((template, matches, len(catalog_type_keywords), catalog_type_keywords))
                                else:
                                    template_scores.append((template, 0, 0, []))
                            
                            # Find the maximum score for type keywords
                            max_score = max(score for _, score, _, _ in template_scores)
                            
                            if max_score > 0:
                                best_scored_templates = [(template, score, total_kw, kw_list) for template, score, total_kw, kw_list in template_scores if score == max_score]
                                
                                if len(best_scored_templates) == 1:
                                    current_templates = [best_scored_templates[0][0]]
                                    matched_keywords = [ck for ck in company_type_keywords if ck in best_scored_templates[0][3]]
                                    match_steps.append(f"type: {', '.join(matched_keywords)}")
                                else:
                                    # Tie-breaker: prefer template with fewer type keywords
                                    min_total_keywords = min(total_kw for _, _, total_kw, _ in best_scored_templates)
                                    final_candidates = [template for template, score, total_kw, kw_list in best_scored_templates if total_kw == min_total_keywords]
                                    
                                    if len(final_candidates) == 1:
                                        current_templates = final_candidates
                                        winner_keywords = [kw_list for template, score, total_kw, kw_list in best_scored_templates if template == final_candidates[0]][0]
                                        matched_keywords = [ck for ck in company_type_keywords if ck in winner_keywords]
                                        match_steps.append(f"type: {', '.join(matched_keywords)} (tiebreaker)")
                    
                    # If exactly one template remains, use it
                    if len(current_templates) == 1:
                        template = current_templates[0]
                        matched_df.at[idx, 'Catalog_Match_Found'] = True
                        matched_df.at[idx, 'Catalog_Template'] = template
                        matched_df.at[idx, 'Match_Type'] = 'preroll_multi_step'
                        matched_df.at[idx, 'Match_Strategy'] = 'preroll_multi_step'
                        matched_df.at[idx, 'Match_Keywords'] = ', '.join(match_steps)
                        preroll_matches += 1
                        match_found = True
                        troubleshooting_data.append({
                            'Brand': brand,
                            'Item': item,
                            'Shop': shop,
                            'Match_Status': 'Preroll multi-step match',
                            'Catalog_Template': template,
                            'Catalog_Options': f"{len(templates)} total, 1 after filtering",
                            'Notes': f'Matched by: {", ".join(match_steps)}'
                        })
                
                elif category in ['Vape', 'Extract']:
                    # Vape/Extract: Weight → Keywords
                    current_templates = templates
                    match_steps = []
                    
                    # Step 1: Weight matching
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
                    
                    # Step 2: Keyword matching
                    company_keywords = row.get('Extracted_Category_Keywords')
                    if company_keywords and len(current_templates) > 1:
                        company_keyword_list = [kw.strip() for kw in str(company_keywords).split(',')]
                        
                        # Score each template by number of keyword matches
                        template_scores = []
                        for template in current_templates:
                            catalog_keywords = extract_category_keywords(template, category)
                            if catalog_keywords:
                                catalog_keyword_list = [kw.strip() for kw in catalog_keywords.split(',')]
                                # Count matching keywords
                                matches = sum(1 for ck in company_keyword_list if ck in catalog_keyword_list)
                                template_scores.append((template, matches, len(catalog_keyword_list), catalog_keyword_list))
                            else:
                                template_scores.append((template, 0, 0, []))
                        
                        # Find the maximum score
                        max_score = max(score for _, score, _, _ in template_scores)
                        
                        # Get templates with the highest score
                        if max_score > 0:
                            best_scored_templates = [(template, score, total_kw, kw_list) for template, score, total_kw, kw_list in template_scores if score == max_score]
                            
                            if len(best_scored_templates) == 1:
                                # Single winner
                                current_templates = [best_scored_templates[0][0]]
                                matched_keywords = [ck for ck in company_keyword_list if ck in best_scored_templates[0][3]]
                                match_steps.append(f"keywords: {', '.join(matched_keywords)} ({max_score} matches)")
                            else:
                                # Tie-breaker: prefer template with fewer total keywords (more precise match)
                                min_total_keywords = min(total_kw for _, _, total_kw, _ in best_scored_templates)
                                final_candidates = [template for template, score, total_kw, kw_list in best_scored_templates if total_kw == min_total_keywords]
                                
                                if len(final_candidates) == 1:
                                    current_templates = final_candidates
                                    # Find the keyword list for the winner
                                    winner_keywords = [kw_list for template, score, total_kw, kw_list in best_scored_templates if template == final_candidates[0]][0]
                                    matched_keywords = [ck for ck in company_keyword_list if ck in winner_keywords]
                                    match_steps.append(f"keywords: {', '.join(matched_keywords)} ({max_score} matches, tiebreaker: fewer keywords)")
                    
                    # Step 3: No-keyword fallback matching
                    elif not company_keywords and len(current_templates) > 1:
                        # Company has no keywords, look for catalog templates that also have no keywords
                        no_keyword_templates = []
                        for template in current_templates:
                            catalog_keywords = extract_category_keywords(template, category)
                            if not catalog_keywords:  # Template also has no keywords
                                no_keyword_templates.append(template)
                        
                        if len(no_keyword_templates) == 1:
                            current_templates = no_keyword_templates
                            match_steps.append("no keywords (fallback)")
                    
                    # If exactly one template remains, use it
                    if len(current_templates) == 1:
                        template = current_templates[0]
                        matched_df.at[idx, 'Catalog_Match_Found'] = True
                        matched_df.at[idx, 'Catalog_Template'] = template
                        matched_df.at[idx, 'Match_Type'] = f'{category.lower()}_weight_keywords'
                        matched_df.at[idx, 'Match_Strategy'] = f'{category.lower()}_weight_keywords'
                        matched_df.at[idx, 'Match_Keywords'] = ', '.join(match_steps)
                        vape_extract_matches += 1
                        match_found = True
                        troubleshooting_data.append({
                            'Brand': brand,
                            'Item': item,
                            'Shop': shop,
                            'Match_Status': f'{category} weight+keywords match',
                            'Catalog_Template': template,
                            'Catalog_Options': f"{len(templates)} total, 1 after filtering",
                            'Notes': f'Matched by: {", ".join(match_steps)}'
                        })
        
        if not match_found:
            no_matches += 1
    
    progress_bar.progress(1.0)
    
    # Display results
    total_matches = exact_matches + single_entry_matches + brand_category_matches + flower_weight_matches + preroll_matches + vape_extract_matches
    total_match_rate = (total_matches / len(matched_df)) * 100 if len(matched_df) > 0 else 0
    
    st.success(f"🎉 Enhanced Matching Results (All Categories):")
    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
    with col1:
        st.metric("🎯 Exact", f"{exact_matches:,}")
    with col2:
        st.metric("1️⃣ Single Brand", f"{single_entry_matches:,}")
    with col3:
        st.metric("📂 Brand+Category", f"{brand_category_matches:,}")
    with col4:
        st.metric("🌸 Flower Weight", f"{flower_weight_matches:,}")
    with col5:
        st.metric("🚬 Preroll Multi", f"{preroll_matches:,}")
    with col6:
        st.metric("💨 Vape/Extract", f"{vape_extract_matches:,}")
    with col7:
        st.metric("📊 Total", f"{total_matches:,}")
    with col8:
        st.metric("📈 Rate", f"{total_match_rate:.1f}%")
    
    # Debug: Show brands that still need matching rules
    unmatched_products = matched_df[matched_df['Catalog_Match_Found'] == False]
    if len(unmatched_products) > 0:
        unmatched_brands = unmatched_products['Brand'].value_counts()
        st.write(f"**🔧 Brands Still Needing Rules ({len(unmatched_brands)} brands, {len(unmatched_products):,} products):**")
        
        # Show top unmatched brands
        top_unmatched = unmatched_brands.head(20)
        brands_list = []
        for brand, count in top_unmatched.items():
            brands_list.append(f"{brand} ({count:,})")
        
        st.write(f"**Top unmatched brands:** {', '.join(brands_list)}")
        
        # Also show unique list for easy copying
        unique_unmatched_brands = sorted(unmatched_brands.index.tolist())
        st.write(f"**Complete list of unmatched brands:** {', '.join(unique_unmatched_brands)}")
        
        # Show sample of brand+category combinations that need rules
        unmatched_brand_categories = unmatched_products.groupby(['Brand', 'Category']).size().reset_index(name='count')
        if len(unmatched_brand_categories) > 0:
            st.write(f"**Sample Brand+Category combinations needing rules:**")
            sample_combinations = unmatched_brand_categories.head(10)
            for _, row in sample_combinations.iterrows():
                st.write(f"• {row['Brand']} + {row['Category']}: {row['count']} products")
    else:
        st.success("🎉 All products matched successfully!")
    
    # Store troubleshooting data for later use
    troubleshooting_df = pd.DataFrame(troubleshooting_data)
    matched_df.troubleshooting_data = troubleshooting_df
    
    # Show some examples if we found matches
    if flower_weight_matches > 0:
        st.write("**🌸 Sample Flower Weight Matches:**")
        flower_examples = matched_df[matched_df['Match_Type'] == 'flower_weight'].head(5)
        for _, example in flower_examples.iterrows():
            weight_info = example.get('Match_Keywords', 'N/A')
            st.write(f"• **{example['Brand']}**: \"{example['Item']}\" → \"{example['Catalog_Template']}\" ({weight_info})")
    
    if preroll_matches > 0:
        st.write("**🚬 Sample Preroll Multi-Step Matches:**")
        preroll_examples = matched_df[matched_df['Match_Type'] == 'preroll_multi_step'].head(5)
        for _, example in preroll_examples.iterrows():
            match_info = example.get('Match_Keywords', 'N/A')
            st.write(f"• **{example['Brand']}**: \"{example['Item']}\" → \"{example['Catalog_Template']}\" ({match_info})")
    
    if vape_extract_matches > 0:
        st.write("**💨 Sample Vape/Extract Weight+Keywords Matches:**")
        vape_extract_examples = matched_df[matched_df['Match_Type'].isin(['vape_weight_keywords', 'extract_weight_keywords'])].head(5)
        for _, example in vape_extract_examples.iterrows():
            match_info = example.get('Match_Keywords', 'N/A')
            category = example.get('Category', 'Unknown')
            st.write(f"• **{example['Brand']}** ({category}): \"{example['Item']}\" → \"{example['Catalog_Template']}\" ({match_info})")
    
    if single_entry_matches > 0:
        st.write("**1️⃣ Sample Auto-Matches:**")
        auto_examples = matched_df[matched_df['Match_Type'] == 'brand_auto'].head(3)
        for _, example in auto_examples.iterrows():
            st.write(f"• **{example['Brand']}**: \"{example['Item']}\" → \"{example['Catalog_Template']}\"")
    
    return matched_df

def add_simple_price_comparison(company_df, catalog_df):
    """
    Simple price comparison - just add basic price difference columns
    """
    if company_df is None or catalog_df is None:
        return company_df
    
    # Only process matched products
    matched_products = company_df[company_df['Catalog_Match_Found'] == True].copy()
    
    if len(matched_products) == 0:
        st.warning("⚠️ No matched products found for price comparison")
        return company_df
    
    st.info(f"💰 Adding price comparison for {len(matched_products):,} matched products...")
    
    def clean_price(price_str):
        """Clean price string and convert to float"""
        if pd.isna(price_str) or price_str == '':
            return None
        try:
            # Remove $ signs, commas, spaces
            cleaned = str(price_str).replace('$', '').replace(',', '').strip()
            return float(cleaned)
        except:
            return None
    
    # Initialize new columns
    company_df['Catalog_Retail_Price'] = None
    company_df['Catalog_Sale_Price'] = None
    company_df['Retail_Price_Diff'] = None
    company_df['Sale_Price_Diff'] = None
    
    # Create catalog lookup
    catalog_lookup = {}
    for _, cat_row in catalog_df.iterrows():
        template = cat_row['Profile Template']
        if pd.notna(template):
            catalog_lookup[template] = cat_row
    
    pricing_issues = 0
    
    for idx, row in matched_products.iterrows():
        catalog_template = row['Catalog_Template']
        catalog_location = row['Catalog_Location']
        
        if pd.isna(catalog_template) or pd.isna(catalog_location):
            continue
        
        # Get catalog data
        catalog_data = catalog_lookup.get(catalog_template)
        if catalog_data is None:
            continue
        
        # Build price column names
        retail_price_col = f"{catalog_location} Retail Price"
        sale_price_col = f"{catalog_location} Sale Price"
        
        # Get prices
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
        
        # Count issues (differences > $0.01)
        retail_diff = company_df.at[idx, 'Retail_Price_Diff']
        sale_diff = company_df.at[idx, 'Sale_Price_Diff']
        
        has_retail_issue = (retail_diff is not None and not pd.isna(retail_diff) and abs(retail_diff) > 0.01)
        has_sale_issue = (sale_diff is not None and not pd.isna(sale_diff) and abs(sale_diff) > 0.01)
        
        if has_retail_issue or has_sale_issue:
            pricing_issues += 1
    
    st.success(f"💰 Price comparison complete! Found {pricing_issues:,} products with price differences > $0.01")
    
    return company_df

def main():
    st.title("🛒 Product Price Checker")
    st.markdown("Filter your Company Products CSV and connect to Product Catalog data")
    
    # Data source configuration
    st.sidebar.header("📊 Data Sources")
    
    # Company Products CSV Upload
    st.sidebar.subheader("📄 Upload Company Products")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Company Products CSV:",
        type=['csv'],
        help="Upload your company's product CSV file. Will be filtered by Active status and matched against Product Catalog brands."
    )
    
    # Google Sheets status (only show if credentials are configured)
    google_sheets_available = "google_sheets" in st.secrets
    
    if google_sheets_available:
        st.sidebar.subheader("🔗 Reference Data Sources")
        st.sidebar.success("✅ Connect Product Catalog - Configured")
        st.sidebar.info("Product catalog will be automatically loaded from configured Google Sheet")
    else:
        st.sidebar.warning("⚠️ Google Sheets API not configured")
        st.sidebar.info("Company Products filtering will work, but brand cross-referencing will be skipped")
    
    # Load and process data
    if st.sidebar.button("🚀 Load Data", type="primary"):
        with st.spinner("Loading data from all sources..."):
            
            # First, load Connect Product Catalog (we need this for extracting brands)
            connect_catalog_df = None
            if google_sheets_available:
                st.info("📊 Loading Connect Product Catalog reference data...")
                catalog_df, catalog_ws_name = load_google_sheet_data(CONNECT_CATALOG_URL)
                if catalog_df is not None:
                    connect_catalog_df = catalog_df  # No filtering - pure reference data
                    st.session_state['df_catalog'] = connect_catalog_df
                    st.session_state['df_catalog_name'] = f"Connect Product Catalog ({catalog_ws_name})"
                    st.success(f"✅ Loaded Product Catalog: {connect_catalog_df.shape[0]} records")
                else:
                    st.error("❌ Failed to load Connect Product Catalog")
            
            # Load Company Products CSV and filter using Product Catalog brands
            filtered_csv = None
            if uploaded_file is not None:
                st.info("📄 Processing Company Products CSV...")
                df_csv, csv_name = load_csv_data(uploaded_file)
                if df_csv is not None:
                    filtered_csv = filter_company_products(df_csv, connect_catalog_df)
                    if filtered_csv is not None:
                        # Add shop name mapping for price comparison
                        filtered_csv = add_catalog_location_mapping(filtered_csv)
                        
                        # Normalize categories to match Product Catalog
                        filtered_csv = normalize_categories(filtered_csv)
                        
                        # Add smart brand matching
                        if connect_catalog_df is not None:
                            filtered_csv = add_smart_brand_matching(filtered_csv, connect_catalog_df)
                            
                            # Add simple price comparison
                            filtered_csv = add_simple_price_comparison(filtered_csv, connect_catalog_df)
                        
                        st.session_state['df_csv'] = filtered_csv
                        st.session_state['df_csv_name'] = csv_name
                        st.success(f"✅ Processed Company Products: {filtered_csv.shape[0]} products after filtering, matching, and price comparison")
                    else:
                        st.error("❌ Failed to process Company Products")
                else:
                    st.error("❌ Failed to load Company Products CSV")
            
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
    
    # Display results
    if any(key in st.session_state for key in ['df_csv', 'df_catalog']):
        
        # Create tabs based on available data sources
        tab_names = ["📊 Overview"]
        if 'df_csv' in st.session_state:
            tab_names.append("📄 Company Products")
            # Add Price Inspector tab only if we have matches
            if 'Catalog_Match_Found' in st.session_state['df_csv'].columns and st.session_state['df_csv']['Catalog_Match_Found'].sum() > 0:
                tab_names.append("💰 Price Inspector")
            # Add Troubleshooting tab for debugging matches
            if hasattr(st.session_state['df_csv'], 'troubleshooting_data'):
                tab_names.append("🔧 Troubleshooting")
        if 'df_catalog' in st.session_state:
            tab_names.append("📋 Product Catalog")
        
        tabs = st.tabs(tab_names)
        tab_index = 0
        
        # Overview Tab
        with tabs[tab_index]:
            st.subheader("📊 Data Overview")
            
            # Pricing consistency overview
            if 'df_csv' in st.session_state:
                df_csv = st.session_state['df_csv']
                
                # Basic metrics
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
                        # Convert to numeric and handle safely
                        retail_diff_numeric = pd.to_numeric(df_csv['Retail_Price_Diff'], errors='coerce').fillna(0)
                        sale_diff_numeric = pd.to_numeric(df_csv['Sale_Price_Diff'], errors='coerce').fillna(0) if 'Sale_Price_Diff' in df_csv.columns else pd.Series([0] * len(df_csv))
                        
                        retail_issues = retail_diff_numeric.abs() > 0.50
                        sale_issues = sale_diff_numeric.abs() > 0.50
                        price_issues = (retail_issues | sale_issues).sum()
                        st.metric("Price Issues", f"{price_issues:,}")
                with col4:
                    if 'Retail_Price_Diff' in df_csv.columns and matched_count > 0:
                        consistency_rate = ((matched_count - price_issues) / matched_count * 100) if matched_count > 0 else 0
                        st.metric("Price Consistency", f"{consistency_rate:.1f}%")
                
                # Show enhanced match breakdown if available
                if 'Match_Type' in df_csv.columns:
                    st.write("**🎯 Enhanced Matching Breakdown:**")
                    match_type_counts = df_csv[df_csv['Catalog_Match_Found'] == True]['Match_Type'].value_counts()
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        exact_count = match_type_counts.get('exact', 0)
                        st.metric("🎯 Exact", f"{exact_count:,}")
                    with col2:
                        auto_count = match_type_counts.get('brand_auto', 0) + match_type_counts.get('brand_category_auto', 0)
                        st.metric("⚡ Auto", f"{auto_count:,}")
                    with col3:
                        flower_count = match_type_counts.get('flower_weight', 0)
                        preroll_count = match_type_counts.get('preroll_multi_step', 0)
                        st.metric("🌸 Weight+Multi", f"{flower_count + preroll_count:,}")
                    with col4:
                        vape_extract_count = match_type_counts.get('vape_weight_keywords', 0) + match_type_counts.get('extract_weight_keywords', 0)
                        st.metric("💨 Vape/Extract", f"{vape_extract_count:,}")
                    with col5:
                        total_matched = match_type_counts.sum()
                        st.metric("📊 Total Matched", f"{total_matched:,}")
                
                # Pricing summary
                if 'Retail_Price_Diff' in df_csv.columns:
                    st.write("**🎯 Pricing Analysis Summary:**")
                    matched_with_prices = df_csv[(df_csv['Catalog_Match_Found'] == True) & 
                                                (df_csv['Retail_Price_Diff'].notna() | df_csv['Sale_Price_Diff'].notna())]
                    
                    if len(matched_with_prices) > 0:
                        st.write(f"• **{len(matched_with_prices):,} products** have catalog price comparisons")
                        st.write(f"• **{price_issues:,} products** need price adjustments (>${0.50:.0f} difference)")
                        st.write(f"• **{len(matched_with_prices) - price_issues:,} products** have consistent pricing")
                        st.write(f"• Use the **Price Inspector** tab to review and export products needing fixes")
            else:
                st.info("Upload your Company Products CSV to see pricing analysis")
        
        tab_index += 1
        
        # Company Products Tab
        if 'df_csv' in st.session_state:
            with tabs[tab_index]:
                st.subheader(f"📄 {st.session_state['df_csv_name']}")
                st.info("Filtered and processed company product data with smart matching and price comparison")
                
                # Show processing summary
                df_csv = st.session_state['df_csv']
                
                # Clean metrics display
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📦 Total Products", f"{len(df_csv):,}")
                with col2:
                    if 'Catalog_Location' in df_csv.columns:
                        mapped_count = df_csv['Catalog_Location'].notna().sum()
                        total_count = len(df_csv)
                        st.metric("🔗 Location Mapped", f"{mapped_count:,}")
                with col3:
                    if 'Catalog_Match_Found' in df_csv.columns:
                        matched_count = df_csv['Catalog_Match_Found'].sum()
                        total_count = len(df_csv)
                        match_rate = (matched_count / total_count * 100) if total_count > 0 else 0
                        st.metric("🎯 Catalog Matched", f"{matched_count:,} ({match_rate:.1f}%)")
                with col4:
                    if 'Retail_Price_Diff' in df_csv.columns:
                        retail_issues = df_csv['Retail_Price_Diff'].fillna(0).abs() > 0.50
                        sale_issues = df_csv['Sale_Price_Diff'].fillna(0).abs() > 0.50 if 'Sale_Price_Diff' in df_csv.columns else False
                        price_issues = (retail_issues | sale_issues).sum()
                        st.metric("💰 Price Issues", f"{price_issues:,}")
                
                st.dataframe(df_csv, use_container_width=True)
                
                # Download button
                csv_buffer = io.StringIO()
                df_csv.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Processed Company Products",
                    data=csv_buffer.getvalue(),
                    file_name="company_products_with_price_comparison.csv",
                    mime="text/csv"
                )
            tab_index += 1
        
        # Price Inspector Tab (only if matches exist)
        if 'df_csv' in st.session_state and 'Catalog_Match_Found' in st.session_state['df_csv'].columns:
            matched_data = st.session_state['df_csv'][st.session_state['df_csv']['Catalog_Match_Found'] == True]
            if len(matched_data) > 0:
                with tabs[tab_index]:
                    st.subheader("💰 Price Inspector")
                    st.info("Review matched products and identify pricing discrepancies")
                    
                    # Filter options
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
                    
                    # Apply filters
                    filtered_matches = matched_data.copy()
                    
                    # Filter by price issues if checkbox is checked
                    if show_price_issues_only:
                        if 'Retail_Price_Diff' in filtered_matches.columns:
                            # Convert to numeric and handle NaN values safely
                            retail_diff_numeric = pd.to_numeric(filtered_matches['Retail_Price_Diff'], errors='coerce').fillna(0)
                            sale_diff_numeric = pd.to_numeric(filtered_matches['Sale_Price_Diff'], errors='coerce').fillna(0) if 'Sale_Price_Diff' in filtered_matches.columns else pd.Series([0] * len(filtered_matches))
                            
                            # Show ANY price difference (not just >$0.50)
                            retail_issues = retail_diff_numeric != 0
                            sale_issues = sale_diff_numeric != 0
                            price_issue_mask = retail_issues | sale_issues
                            filtered_matches = filtered_matches[price_issue_mask]
                    
                    # Filter by in-stock if checkbox is checked
                    if show_in_stock_only:
                        if 'Inventory Available' in filtered_matches.columns:
                            # Convert to numeric and filter for > 0
                            inventory_numeric = pd.to_numeric(filtered_matches['Inventory Available'], errors='coerce').fillna(0)
                            in_stock_mask = inventory_numeric > 0
                            filtered_matches = filtered_matches[in_stock_mask]
                    
                    # Apply other filters
                    if selected_brands:
                        filtered_matches = filtered_matches[filtered_matches['Brand'].isin(selected_brands)]
                    if selected_locations and 'Catalog_Location' in filtered_matches.columns:
                        filtered_matches = filtered_matches[filtered_matches['Catalog_Location'].isin(selected_locations)]
                    
                    st.write(f"Showing {len(filtered_matches)} of {len(matched_data)} matched products")
                    
                    # Display columns in specified order without custom renaming
                    if len(filtered_matches) > 0:
                        display_columns = [
                            'Brand', 'Item', 'Catalog_Template', 'Catalog_Location', 'Inventory Available',
                            'Unit Price', 'Catalog_Retail_Price', 'Retail_Price_Diff',
                            'Unit Sale Price', 'Catalog_Sale_Price', 'Sale_Price_Diff'
                        ]
                        
                        # Filter to only available columns
                        available_columns = [col for col in display_columns if col in filtered_matches.columns]
                        display_df = filtered_matches[available_columns].copy()
                        
                        # Format only the numeric price columns, leave string columns as-is
                        numeric_price_columns = ['Catalog_Retail_Price', 'Retail_Price_Diff', 'Catalog_Sale_Price', 'Sale_Price_Diff']
                        for col in numeric_price_columns:
                            if col in display_df.columns:
                                # Safely convert to numeric and round, handling any non-numeric values
                                display_df[col] = pd.to_numeric(display_df[col], errors='coerce').round(2)
                        
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Download filtered results with descriptive filename
                        csv_buffer = io.StringIO()
                        display_df.to_csv(csv_buffer, index=False)
                        
                        # Create descriptive download label based on active filters
                        filter_description = []
                        if show_price_issues_only:
                            filter_description.append("Price Issues")
                        if show_in_stock_only:
                            filter_description.append("In-Stock")
                        if selected_brands:
                            filter_description.append(f"{len(selected_brands)} Brand(s)")
                        if selected_locations:
                            filter_description.append(f"{len(selected_locations)} Location(s)")
                        
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
                    else:
                        st.info("No products match the selected filters.")
                
                tab_index += 1
        
        # Troubleshooting Tab
        if 'df_csv' in st.session_state and hasattr(st.session_state['df_csv'], 'troubleshooting_data'):
            with tabs[tab_index]:
                st.subheader("🔧 Matching Troubleshooting")
                st.info("Debug unsuccessful matching results and identify improvement opportunities")
                
                troubleshooting_df = st.session_state['df_csv'].troubleshooting_data
                
                # Filter to only show unsuccessful matches
                unsuccessful_statuses = ['Missing brand or item']
                unsuccessful_matches = troubleshooting_df[troubleshooting_df['Match_Status'].isin(unsuccessful_statuses)]
                
                # Troubleshooting summary for unsuccessful matches only
                status_counts = unsuccessful_matches['Match_Status'].value_counts()
                
                st.write("**🔍 Unsuccessful Match Breakdown:**")
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
                
                # Filter troubleshooting data
                status_filter = st.selectbox(
                    "Filter by Issue Type:",
                    options=['All Issues'] + list(status_counts.index),
                    index=0
                )
                
                filtered_troubleshooting = unsuccessful_matches.copy()
                if status_filter != 'All Issues':
                    filtered_troubleshooting = filtered_troubleshooting[filtered_troubleshooting['Match_Status'] == status_filter]
                
                # Show brands with most unmatched products
                st.write("**🔍 Brands with Most Unmatched Products:**")
                if len(filtered_troubleshooting) > 0:
                    brand_unmatched = filtered_troubleshooting['Brand'].value_counts().head(10)
                    st.bar_chart(brand_unmatched)
                    
                    # Show specific examples for top problematic brands
                    st.write("**🎯 Top Problematic Brand Examples:**")
                    for brand_name in brand_unmatched.head(3).index:
                        brand_examples = filtered_troubleshooting[filtered_troubleshooting['Brand'] == brand_name].head(3)
                        st.write(f"**{brand_name}** ({brand_unmatched[brand_name]} unmatched):")
                        for _, example in brand_examples.iterrows():
                            st.write(f"  • \"{example['Item']}\" - {example['Notes']}")
                else:
                    st.success("🎉 No unsuccessful matches found!")
                
                st.write(f"**📋 Unsuccessful Match Details ({len(filtered_troubleshooting)} records):**")
                if len(filtered_troubleshooting) > 0:
                    st.dataframe(filtered_troubleshooting, use_container_width=True)
                    
                    # Download troubleshooting data
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
        
        # Connect Product Catalog Tab
        if 'df_catalog' in st.session_state:
            with tabs[tab_index]:
                st.subheader(f"📋 {st.session_state['df_catalog_name']}")
                st.info("Reference catalog data - Brand column used for brand extraction, Profile Template for matching")
                st.dataframe(st.session_state['df_catalog'], use_container_width=True)
                
                # Download button
                csv_buffer = io.StringIO()
                st.session_state['df_catalog'].to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Product Catalog",
                    data=csv_buffer.getvalue(),
                    file_name="connect_product_catalog_reference.csv",
                    mime="text/csv"
                )
    
    else:
        st.info("👆 Upload your Company Products CSV in the sidebar and click 'Load Data' to get started")
        
        st.subheader("📄 Data Processing Workflow")
        
        st.markdown("""
        **🎯 Current Processing (Smart Matching + Simple Price Comparison):**
        
        1. **📄 Company Products (CSV Upload)**
           - ✅ Filters out inactive products (Active ≠ "No"/"False")
           - ✅ Excludes unwanted categories (Display, Clones, Apparel, etc.)
           - ✅ Keeps only essential columns for price checking
           - ✅ Cross-references with brands from Product Catalog
           - ✅ Maps shop names to catalog locations
           - ✅ Smart brand structure matching (93.4% success rate)
           - ✅ **NEW: Simple price comparison for matched products**
           - ✅ **NEW: Weight and pack size extraction for enhanced matching**
        
        2. **📋 Connect Product Catalog (Auto-loaded)**
           - ✅ Reference data for lookups and brand extraction
           - ✅ Brand column contains the master brand list
           - ✅ Profile Template column used for smart matching
           - ✅ **Price columns used for comparison (Retail Price, Sale Price by location)**
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🧠 Smart Matching Logic**
            - **Single Entry Brands**: Auto-match (e.g., Almora, Astronauts)
            - **Multiple Entry Brands**: Keyword matching (e.g., Kurvana "Originals" vs "ASCND")
            - **Special Cases**: Block Party (weight-based), Turn ("Live Resin")
            - **Enhanced with Weight/Pack extraction** for future flower/preroll matching
            """)
        
        with col2:
            if google_sheets_available:
                st.markdown("""
                **💰 Price Comparison Features**
                - Compares Unit Price vs Catalog Retail Price
                - Compares Sale Price vs Catalog Sale Price
                - Uses location mapping (Belmont → "Belmont Retail Price")
                - Flags differences > $0.50
                - Export products needing price fixes
                """)
            else:
                st.markdown("""
                **🔗 Reference Data**
                - ⚠️ Google Sheets API not configured
                - Smart matching will be skipped
                - CSV processing will still work
                - Configure API for full functionality
                """)

if __name__ == "__main__":
    main()