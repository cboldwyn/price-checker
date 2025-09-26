"""
Product Price Checker v4.0
Smart brand matching and price comparison tool for cannabis retail products
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
    page_title="Product Price Checker v4.0",
    page_icon="🛒",
    layout="wide"
)

# Configuration
VERSION = "4.0"
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

# Exact match brands that require product-level matching
EXACT_PRODUCT_MATCH_BRANDS = {
    'Blazy Susan', 'Camino', 'Crave', 'Daily Dose', "Dr. Norm's", 'Good Tide', 
    'Happy Fruit', 'High Gorgeous', 'Kiva', 'Lost Farm', 'Made From Dirt', 
    'Papa & Barkley', 'Sip Elixirs', 'St. Ides', "Uncle Arnie's", 'Vet CBD', 
    'Wyld', 'Yummi Karma', "Not Your Father's"
}

def extract_weight_from_item(item_text):
    """Extract weight from item text (e.g., "Blue Dream 3.5g" → "3.5g")"""
    if pd.isna(item_text):
        return None
    
    item_str = str(item_text).strip()
    weight_patterns = [
        r'(\d+\.?\d*g)$',
        r'(\d+\.\d+\s?oz?)$',
        r'(\d+\s?oz?)$',
        r'(1/8\s?oz?)$',
        r'(1/4\s?oz?)$',
        r'(1/2\s?oz?)$',
    ]
    
    for pattern in weight_patterns:
        match = re.search(pattern, item_str, re.IGNORECASE)
        if match:
            return match.group(1).lower().replace(' ', '')
    
    return None

def extract_pack_size_from_item(item_text):
    """Extract pack size from item text (e.g., "OG Kush 3pk 1.5g" → "3pk")"""
    if pd.isna(item_text):
        return None
    
    item_str = str(item_text).strip()
    pack_patterns = [
        r'(\d+pk)\s+\d+\.?\d*g',
        r'(\d+pk)\s+\d+\s?oz',
        r'(\d+pk)\s+1/[248]\s?oz',
    ]
    
    for pattern in pack_patterns:
        match = re.search(pattern, item_str, re.IGNORECASE)
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

@st.cache_data
def load_google_sheet_data(sheet_url):
    """Load data from Google Sheets using service account authentication"""
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
            return df, worksheet.title
        else:
            return None, None
        
    except Exception as e:
        st.error(f"Error loading Google Sheet: {str(e)}")
        return None, None

def load_csv_data(uploaded_file):
    """Load data from uploaded CSV file"""
    try:
        df = pd.read_csv(uploaded_file, skiprows=1)
        return df, "Company Products"
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        return None, None

def filter_company_products(df, connect_catalog_df=None):
    """Filter company products data by Active field, keep only specified columns, and filter by brands"""
    if df is None or df.empty:
        return None
    
    st.write(f"**Company Products** - Original data shape: {df.shape}")
    
    # Filter by Active status
    if 'Active' in df.columns:
        df['Active'] = df['Active'].astype(str).str.strip()
        active_df = df[~df['Active'].isin(['No', 'False', 'no', 'false', 'NO', 'FALSE', 'N', 'n'])]
        st.write(f"**Company Products** - After filtering by Active field: {active_df.shape}")
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
        st.write(f"**Company Products** - After excluding unwanted categories: {active_df.shape}")
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
    st.write(f"**Company Products** - After column filtering: {filtered_df.shape}")
    
    # Filter by valid brands from catalog
    if connect_catalog_df is not None and not connect_catalog_df.empty and 'Brand' in filtered_df.columns:
        if 'Brand' in connect_catalog_df.columns:
            valid_brands = connect_catalog_df['Brand'].dropna().unique()
            valid_brands = [str(brand).strip() for brand in valid_brands if str(brand).strip() and str(brand) != 'nan']
            
            before_brand_filter = len(filtered_df)
            filtered_df = filtered_df[filtered_df['Brand'].isin(valid_brands)]
            after_brand_filter = len(filtered_df)
            
            st.write(f"**Company Products** - After brand filtering: {filtered_df.shape}")
            st.info(f"🎯 Filtered to only include {len(valid_brands)} brands from Product Catalog. Removed {before_brand_filter - after_brand_filter} products.")
    
    # Add data source identifier
    filtered_df['Data_Source'] = "Company Products"
    
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
    
    st.write(f"**Company Products** - Final filtered data shape: {filtered_df.shape}")
    return filtered_df

def add_catalog_location_mapping(df):
    """Add a 'Catalog_Location' column to Company Products"""
    if df is None or df.empty or 'Shop' not in df.columns:
        return df
    
    df_copy = df.copy()
    df_copy['Catalog_Location'] = df_copy['Shop'].map(SHOP_NAME_MAPPING)
    
    unmapped_shops = df_copy[df_copy['Catalog_Location'].isna()]['Shop'].unique()
    if len(unmapped_shops) > 0:
        st.warning(f"⚠️ Unmapped shops found: {list(unmapped_shops)}")
    
    mapped_count = df_copy['Catalog_Location'].notna().sum()
    total_count = len(df_copy)
    st.info(f"✅ Shop mapping: {mapped_count}/{total_count} products mapped to catalog locations")
    
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
            if brand not in brand_catalog_map:
                brand_catalog_map[brand] = []
            brand_catalog_map[brand].append(template)
            
            brand_category_key = f"{brand}|{category}"
            if brand_category_key not in brand_category_catalog_map:
                brand_category_catalog_map[brand_category_key] = []
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
        
        # Skip auto-matching for exact product match brands
        skip_auto_matching = brand in EXACT_PRODUCT_MATCH_BRANDS
        
        # 2. Try single entry brand auto-match
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
        
        # 3. Try brand+category auto-match
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
        
        # 4. Try advanced weight/keyword matching for complex categories
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
    total_matches = exact_matches + single_entry_matches + brand_category_matches + flower_weight_matches + preroll_matches + vape_extract_matches
    total_match_rate = (total_matches / len(matched_df)) * 100 if len(matched_df) > 0 else 0
    
    st.success(f"🎉 Enhanced Matching Results:")
    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
    with col1:
        st.metric("🎯 Exact", f"{exact_matches:,}")
    with col2:
        st.metric("1️⃣ Single Brand", f"{single_entry_matches:,}")
    with col3:
        st.metric("📂 Brand+Category", f"{brand_category_matches:,}")
    with col4:
        st.metric("🌸 Flower", f"{flower_weight_matches:,}")
    with col5:
        st.metric("🚬 Preroll", f"{preroll_matches:,}")
    with col6:
        st.metric("💨 Vape/Extract", f"{vape_extract_matches:,}")
    with col7:
        st.metric("📊 Total", f"{total_matches:,}")
    with col8:
        st.metric("📈 Rate", f"{total_match_rate:.1f}%")
    
    # Store troubleshooting data
    troubleshooting_df = pd.DataFrame(troubleshooting_data)
    matched_df.troubleshooting_data = troubleshooting_df
    
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
    """Advanced matching for preroll products using infused status, weight, pack size, and keywords"""
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
    
    # Filter by pack size
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
    elif not company_pack and len(current_templates) > 1:
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

def add_simple_price_comparison(company_df, catalog_df):
    """Simple price comparison - add basic price difference columns"""
    if company_df is None or catalog_df is None:
        return company_df
    
    matched_products = company_df[company_df['Catalog_Match_Found'] == True].copy()
    
    if len(matched_products) == 0:
        st.warning("⚠️ No matched products found for price comparison")
        return company_df
    
    st.info(f"💰 Adding price comparison for {len(matched_products):,} matched products...")
    
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
    
    # Build catalog lookup
    catalog_lookup = {}
    for _, cat_row in catalog_df.iterrows():
        template = cat_row['Profile Template']
        if pd.notna(template):
            catalog_lookup[template] = cat_row
    
    pricing_issues = 0
    
    # Compare prices for each matched product
    for idx, row in matched_products.iterrows():
        catalog_template = row['Catalog_Template']
        catalog_location = row['Catalog_Location']
        
        if pd.isna(catalog_template) or pd.isna(catalog_location):
            continue
        
        catalog_data = catalog_lookup.get(catalog_template)
        if catalog_data is None:
            continue
        
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
    
    st.success(f"💰 Price comparison complete! Found {pricing_issues:,} products with price differences > $0.01")
    
    return company_df

def main():
    st.title(f"🛒 Product Price Checker v{VERSION}")
    st.markdown("Filter your Company Products CSV and connect to Product Catalog data")
    
    st.sidebar.header("📊 Data Sources")
    
    st.sidebar.subheader("📄 Upload Company Products")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Company Products CSV:",
        type=['csv'],
        help="Upload your company's product CSV file. Will be filtered by Active status and matched against Product Catalog brands."
    )
    
    google_sheets_available = "google_sheets" in st.secrets
    
    if google_sheets_available:
        st.sidebar.subheader("🔗 Reference Data Sources")
        st.sidebar.success("✅ Connect Product Catalog - Configured")
        st.sidebar.info("Product catalog will be automatically loaded from configured Google Sheet")
    else:
        st.sidebar.warning("⚠️ Google Sheets API not configured")
        st.sidebar.info("Company Products filtering will work, but brand cross-referencing will be skipped")
    
    if st.sidebar.button("🚀 Load Data", type="primary"):
        with st.spinner("Loading data from all sources..."):
            
            # Load Product Catalog
            connect_catalog_df = None
            if google_sheets_available:
                st.info("📊 Loading Connect Product Catalog reference data...")
                catalog_df, catalog_ws_name = load_google_sheet_data(CONNECT_CATALOG_URL)
                if catalog_df is not None:
                    connect_catalog_df = catalog_df
                    st.session_state['df_catalog'] = connect_catalog_df
                    st.session_state['df_catalog_name'] = f"Connect Product Catalog ({catalog_ws_name})"
                    st.success(f"✅ Loaded Product Catalog: {connect_catalog_df.shape[0]} records")
                else:
                    st.error("❌ Failed to load Connect Product Catalog")
            
            # Load and process Company Products
            filtered_csv = None
            if uploaded_file is not None:
                st.info("📄 Processing Company Products CSV...")
                df_csv, csv_name = load_csv_data(uploaded_file)
                if df_csv is not None:
                    filtered_csv = filter_company_products(df_csv, connect_catalog_df)
                    if filtered_csv is not None:
                        filtered_csv = add_catalog_location_mapping(filtered_csv)
                        filtered_csv = normalize_categories(filtered_csv)
                        
                        if connect_catalog_df is not None:
                            filtered_csv = add_smart_brand_matching(filtered_csv, connect_catalog_df)
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
    
    # Display data tabs if any data is loaded
    if any(key in st.session_state for key in ['df_csv', 'df_catalog']):
        
        # Build tab list dynamically
        tab_names = ["📊 Overview"]
        if 'df_csv' in st.session_state:
            tab_names.append("📄 Company Products")
            if 'Catalog_Match_Found' in st.session_state['df_csv'].columns and st.session_state['df_csv']['Catalog_Match_Found'].sum() > 0:
                tab_names.append("💰 Price Inspector")
            if hasattr(st.session_state['df_csv'], 'troubleshooting_data'):
                tab_names.append("🔧 Troubleshooting")
        if 'df_catalog' in st.session_state:
            tab_names.append("📋 Product Catalog")
        
        tabs = st.tabs(tab_names)
        tab_index = 0
        
        # Overview Tab
        with tabs[tab_index]:
            st.subheader("📊 Data Overview")
            
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
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        exact_count = match_type_counts.get('exact', 0)
                        st.metric("🎯 Exact", f"{exact_count:,}")
                    with col2:
                        auto_count = match_type_counts.get('brand_auto', 0) + match_type_counts.get('brand_category_auto', 0)
                        st.metric("⚡ Auto", f"{auto_count:,}")
                    with col3:
                        flower_count = match_type_counts.get('flower_weight_keywords', 0)
                        preroll_count = match_type_counts.get('preroll_weight_keywords', 0)
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
                        st.write(f"• **{price_issues:,} products** need price adjustments (>$0.01 difference)")
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
                        retail_issues = df_csv['Retail_Price_Diff'].fillna(0).abs() > 0.01
                        sale_issues = df_csv['Sale_Price_Diff'].fillna(0).abs() > 0.01 if 'Sale_Price_Diff' in df_csv.columns else False
                        price_issues = (retail_issues | sale_issues).sum()
                        st.metric("💰 Price Issues", f"{price_issues:,}")
                
                # Data table
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
        
        # Price Inspector Tab
        if 'df_csv' in st.session_state and 'Catalog_Match_Found' in st.session_state['df_csv'].columns:
            matched_data = st.session_state['df_csv'][st.session_state['df_csv']['Catalog_Match_Found'] == True]
            if len(matched_data) > 0:
                with tabs[tab_index]:
                    st.subheader("💰 Price Inspector")
                    st.info("Review matched products and identify pricing discrepancies")
                    
                    # Filters
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
                    
                    st.write(f"Showing {len(filtered_matches)} of {len(matched_data)} matched products")
                    
                    # Display filtered data
                    if len(filtered_matches) > 0:
                        display_columns = [
                            'Brand', 'Item', 'Catalog_Template', 'Catalog_Location', 'Inventory Available',
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
                        
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Download button
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
                    st.dataframe(filtered_troubleshooting, use_container_width=True)
                    
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
        
        # Product Catalog Tab
        if 'df_catalog' in st.session_state:
            with tabs[tab_index]:
                st.subheader(f"📋 {st.session_state['df_catalog_name']}")
                st.info("Reference catalog data - Brand column used for brand extraction, Profile Template for matching")
                st.dataframe(st.session_state['df_catalog'], use_container_width=True)
                
                csv_buffer = io.StringIO()
                st.session_state['df_catalog'].to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Product Catalog",
                    data=csv_buffer.getvalue(),
                    file_name="connect_product_catalog_reference.csv",
                    mime="text/csv"
                )
    
    else:
        # Welcome screen
        st.info("👆 Upload your Company Products CSV in the sidebar and click 'Load Data' to get started")
        
        st.subheader("📄 Data Processing Workflow")
        
        st.markdown(f"""
        **🎯 Product Price Checker v{VERSION} Features:**
        
        1. **📄 Company Products (CSV Upload)**
           - ✅ Filters out inactive products (Active ≠ "No"/"False")
           - ✅ Excludes unwanted categories (Display, Clones, Apparel, etc.)
           - ✅ Cross-references with brands from Product Catalog
           - ✅ Maps shop names to catalog locations
           - ✅ Smart brand structure matching with advanced algorithms
           - ✅ Weight and pack size extraction for enhanced matching
           - ✅ Price comparison for matched products
        
        2. **📋 Connect Product Catalog (Auto-loaded)**
           - ✅ Reference data for lookups and brand extraction
           - ✅ Brand column contains the master brand list
           - ✅ Profile Template column used for smart matching
           - ✅ Price columns used for comparison (Retail Price, Sale Price by location)
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🧠 Smart Matching Logic**
            - **Exact Match**: Perfect product name matches
            - **Single Entry Brands**: Auto-match when only one catalog option
            - **Multi-Entry Brands**: Advanced keyword and weight matching
            - **Category-Specific Rules**: Flower, Preroll, Vape, Extract matching
            """)
        
        with col2:
            if google_sheets_available:
                st.markdown("""
                **💰 Price Comparison Features**
                - Compares Unit Price vs Catalog Retail Price
                - Compares Sale Price vs Catalog Sale Price
                - Uses location mapping (Belmont → "Belmont Retail Price")
                - Flags differences > $0.01
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