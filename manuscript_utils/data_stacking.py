import pandas as pd
import re
from .config import *

# ================================================
#  Data Stacking Functions
# ================================================

def extract_wave_version_pairs(df):
    """Extract all available wave-version pairs from LQR columns"""
    wave_version_pairs = set()
    for col in df.columns:
        m = re.match(r'.*_w(\d+)_v(\d+)', col)
        if m:
            wave_version_pairs.add(m.groups())
    return wave_version_pairs


def reshape_country_data(df, country, base_cols):
    """Reshape single country data from wide to long format"""
    wave_version_pairs = extract_wave_version_pairs(df)
    country_rows = []
    
    for _, row in df.iterrows():
        for wave, version in wave_version_pairs:
            new_row = {col: row.get(col, pd.NA) for col in base_cols}
            new_row['country'] = all_country_codes[country]  # Use all_country_codes for stacking
            new_row['wave'] = int(wave)
            new_row['scale'] = int(version)
            
            # Add LQR indicators
            for indicator in lqr_indicators:
                col_name = f'{indicator}_w{wave}_v{version}'
                new_row[indicator] = row.get(col_name, pd.NA)
            
            country_rows.append(new_row)
    
    return country_rows


def format_stacked_dataframe(all_rows):
    """Format and clean the stacked DataFrame"""
    # Unify column order across countries
    base_cols_union = [
        'id', 'status','health_vs_mci_vs_dementia',
        'language', 'age', 'gender', 'raeducl',
        'intent_email', 'freq_3m_intent_email', 'r13email', 'r13drive'
    ]
    
    final_columns = base_cols_union + ['country', 'wave', 'scale'] + lqr_indicators
    stacked_df = pd.DataFrame(all_rows).reindex(columns=final_columns)
    
    # Sort and reformat ID field
    stacked_df = stacked_df.sort_values(by=['country', 'id', 'wave', 'scale']).reset_index(drop=True)
    stacked_df['id'] = stacked_df['country'].map(all_country_code_to_name) + "_" + stacked_df['id'].astype(str)
    
    # Rename AADL columns to indicate origin
    stacked_df.rename(columns={
        'intent_email': 'ELSA_intent_email',
        'freq_3m_intent_email': 'ELSA_freq_3m_intent_email',
        'r13email': 'HRS_r13email',
        'r13drive': 'HRS_r13drive'
    }, inplace=True)
    
    return stacked_df


def print_dataset_summary(stacked_df):
    """Print summary statistics for the stacked dataset"""
    row_counts = []
    for code in sorted(stacked_df['country'].unique()):
        name = all_country_code_to_name[code]
        count = len(stacked_df[stacked_df['country'] == code])
        row_counts.append(f"{name}: {count:,}")
    
    print(f"Total rows: {len(stacked_df):,}, Countries: {stacked_df['country'].nunique()}")
    print(" | ".join(row_counts))


def process_country_stacking():
    """Process and stack all countries' data from wide to long format"""
    print("🔄 Processing country data stacking...")
    
    # First pass: collect wide format statistics
    wide_format_summary = []
    for country, file_name in country_files.items():
        file_path = os.path.join(RAW_DATA_PATH, file_name)
        df = pd.read_csv(file_path)
        wide_format_summary.append(f"{country}: {len(df):,}")
    
    # Print wide format summary
    total_wide = sum(int(s.split(': ')[1].replace(',', '')) for s in wide_format_summary)
    print(f"\nWide format summary (Raw data):")
    print(f"Total participants: {total_wide:,}, Countries: 5")
    print(" | ".join(wide_format_summary))
    
    # Second pass: process and reshape data
    print(f"\nProcessing files:")
    all_rows = []
    
    for country, file_name in country_files.items():
        file_path = os.path.join(RAW_DATA_PATH, file_name)
        print(f"  → Processing: {country} ({file_name})")
        
        df = pd.read_csv(file_path)
        base_cols = country_base_columns[country]
        country_rows = reshape_country_data(df, country, base_cols)
        all_rows.extend(country_rows)
    
    # Format the stacked DataFrame
    stacked_df = format_stacked_dataframe(all_rows)
    
    # Save full stacked dataset
    stacked_csv_path = os.path.join(STACKED_DATA_PATH, "stacked_five_country.csv")
    stacked_df.to_csv(stacked_csv_path, index=False)
    
    print(f"\nLong-format data stacking completed:")
    print(f"📁 File saved: stacked_five_country.csv")
    print_dataset_summary(stacked_df)
    print("\n" + "🔹" * 40) 
    
    return stacked_df


def analyze_lqr_missingness(stacked_df):
    """Analyze and report LQR missing value patterns"""
    print("\n🔄 Analyzing LQR missingness patterns...")
    
    total_rows = len(stacked_df)
    total_cells = total_rows * len(lqr_indicators)
    total_missing = stacked_df[lqr_indicators].isna().sum().sum()
    overall_pct = (total_missing / total_cells) * 100
    asinmiss_positive_pct = (stacked_df['asinmiss'] > 0).mean() * 100
    
    print(f"\n📊 LQR Missingness Summary:")
    print(f"Total LQR missing: {total_missing:,} / {total_cells:,} = {overall_pct:.1f}%")
    print(f"asinmiss > 0: {asinmiss_positive_pct:.1f}% (item non-response)")
    
    print(f"\n📊 LQR Missingness by Country:")
    print(f"{'Country':<10} | {'Missing':>10} | {'% of Total':>10}")
    print("-" * 36)
    
    for code, group in stacked_df.groupby("country"):
        country_name = all_country_code_to_name[code]
        n_missing = group[lqr_indicators].isna().sum().sum()
        pct = (n_missing / total_cells) * 100
        print(f"{country_name:<10} | {n_missing:>10,} | {pct:10.1f}%")
    
    return generate_missingness_summary(stacked_df, total_missing, total_cells, overall_pct)


def generate_missingness_summary(stacked_df, total_missing, total_cells, overall_pct):
    """Generate detailed missingness summary for supplementary file"""
    summary_rows = []
    
    # Overall summary
    global_missing_pct = stacked_df[lqr_indicators].isna().mean() * 100
    asinmiss_positive_ratio_all = (stacked_df['asinmiss'] > 0).mean() * 100
    
    summary_rows.append({
        'Country': 'Overall',
        'Total_LQR_Missing': int(total_missing),
        'Total_LQR_Missing_Percent': round(overall_pct, 1),
        'asinmiss > 0 (%)': round(asinmiss_positive_ratio_all, 1),
        **{f"{col} (%)": round(global_missing_pct[col], 1) for col in lqr_indicators}
    })
    
    # Per country summary
    for code, group in stacked_df.groupby("country"):
        country_name = all_country_code_to_name[code]
        n_missing = group[lqr_indicators].isna().sum().sum()
        pct_total = (n_missing / total_cells) * 100
        
        col_missing = group[lqr_indicators].isna().sum()
        col_pct = (col_missing / total_cells) * 100
        asinmiss_positive_ratio = (group['asinmiss'] > 0).mean() * 100
        
        summary_rows.append({
            'Country': country_name,
            'Total_LQR_Missing': int(n_missing),
            'Total_LQR_Missing_Percent': round(pct_total, 3),
            'asinmiss > 0 (%)': round(asinmiss_positive_ratio, 3),
            **{f"{col} (%)": round(col_pct[col], 3) for col in lqr_indicators}
        })
    
    # Save summary
    supp_csv_path = os.path.join(SUPP_PATH, "0_lqr_missing_summary.csv")
    lqr_summary_df = pd.DataFrame(summary_rows)
    lqr_summary_df.to_csv(supp_csv_path, index=False)
    
    print(f"\nLQR missingness analysis completed:")
    print(f"📁 File saved: 0_lqr_missing_summary.csv")
    
    return lqr_summary_df


def split_datasets(stacked_df):
    """Split into CHARLS vs. four-country datasets"""
    print("\n🔄 Splitting datasets...")
    
    # CHARLS = country code 5
    charls_df = stacked_df[stacked_df['country'] == 5]
    four_country_df = stacked_df[stacked_df['country'] != 5]
    
    # Save both datasets
    charls_path = os.path.join(STACKED_DATA_PATH, "stacked_CHARLS.csv")
    four_country_path = os.path.join(STACKED_DATA_PATH, "stacked_four_country.csv")
    
    charls_df.to_csv(charls_path, index=False)
    four_country_df.to_csv(four_country_path, index=False)
    
    print("Dataset splitting completed:")
    print("📁 Files saved: stacked_CHARLS.csv, stacked_four_country.csv")
    print(f"  → External validation set (CHARLS): {len(charls_df):,} rows")
    print(f"  → Joint training set (HRS+ELSA+LASI+MHAS): {len(four_country_df):,} rows")
    
    return charls_df, four_country_df