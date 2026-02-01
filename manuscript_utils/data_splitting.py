import pandas as pd
from sklearn.model_selection import train_test_split
from .config import *

# ================================================
#  Data Splitting Functions
# ================================================

def print_dataset_check(df, dataset_name):
    """Print multi-country dataset sample size and prevalence statistics"""
    print(f"\n📊 [{dataset_name}] Sample Size and Cognitive Status Distribution")
    print("=" * 80)
    
    # Print header
    print(f"{'Dataset':<20} {'N':<8} {'Normal':<8} {'MCI':<8} {'Dementia':<10}")
    print("-" * 80)
    
    for country in sorted(df[country_col].unique()):
        sub_df_country = df[df[country_col] == country]
        cnt = len(sub_df_country)
        rate_normal = (sub_df_country[target] == 0).mean()
        rate_mci = (sub_df_country[target] == 1).mean()
        rate_dementia = (sub_df_country[target] == 2).mean()

        wave_info = country_wave_info[four_country_code_to_name[country]]  # Use four_country mapping
        print(f"{wave_info:<20} {cnt:<8,} {rate_normal:<8.3f} {rate_mci:<8.3f} {rate_dementia:<10.3f}")
    
    print("-" * 80)
    
    # Simplified scale consistency check
    print(f"\n🔍 [{dataset_name}] Scale Consistency Check:")
    all_consistent = True
    for country in sorted(df[country_col].unique()):
        sub_df = df[df[country_col] == country]
        country_name = four_country_code_to_name[country]
        wave_info = country_wave_info[country_name]
        scales = sorted(sub_df[scale_col].unique())
        scale_counts = [len(sub_df[sub_df[scale_col] == s]) for s in scales]
        
        counts_equal = all(scale_counts[0] == c for c in scale_counts)
        if not counts_equal:
            all_consistent = False
            print(f"❌ {wave_info}: Inconsistent counts {scale_counts}")
        else:
            print(f"✅ {wave_info}: {len(scales)} scales, {scale_counts[0]} samples each")
    
    if all_consistent:
        print("All datasets have consistent scale distributions ✅")


def print_charls_dataset_check(df, dataset_name, show_separator=True):
    """Print CHARLS dataset sample size and prevalence statistics"""
    print(f"\n📊 [{dataset_name}] CHARLS Sample Size and Cognitive Status Distribution")
    print("=" * 80)
    
    cnt = len(df)
    rate_normal = (df[target] == 0).mean()
    rate_mci = (df[target] == 1).mean()
    rate_dementia = (df[target] == 2).mean()

    # Print header
    print(f"{'Dataset':<20} {'N':<8} {'Normal':<8} {'MCI':<8} {'Dementia':<10}")
    print("-" * 80)
    print(f"{'CHARLS_wave_4':<20} {cnt:<8,} {rate_normal:<8.3f} {rate_mci:<8.3f} {rate_dementia:<10.3f}")
    print("-" * 80)

    # Simplified scale consistency check
    print(f"\n🔍 [{dataset_name}] CHARLS Scale Consistency Check:")
    scales = sorted(df[scale_col].unique())
    scale_counts = [len(df[df[scale_col] == s]) for s in scales]
    counts_equal = all(scale_counts[0] == c for c in scale_counts)
    
    if counts_equal:
        print(f"✅ CHARLS_wave_4: {len(scales)} scales, {scale_counts[0]} samples each")
    else:
        print(f"❌ CHARLS_wave_4: Inconsistent counts {scale_counts}")
    
    if show_separator:
        print("\n" + "🔹" * 40)


def split_by_scales(df, test_size, random_state, stratify_col):
    """Split dataframe by scales while maintaining stratification"""
    test_list, train_list = [], []
    for scale_val in sorted(df[scale_col].unique()):
        sub_df = df[df[scale_col] == scale_val]
        tr, te = train_test_split(
            sub_df, test_size=test_size, stratify=sub_df[stratify_col], random_state=random_state
        )
        train_list.append(tr)
        test_list.append(te)
    
    return pd.concat(train_list, ignore_index=True), pd.concat(test_list, ignore_index=True)


def split_country_data(country_df, country_name):
    """Split single country data into four subsets"""
    wave_info = country_wave_info.get(country_name, country_name)
    print(f"Processing {wave_info}...")
    
    # Step 1: Split into train/test (35% test)
    train_df, test_df = split_by_scales(country_df, test_size, country_seeds, target)
    
    # Step 2: Split train into validation (25% of total) and remaining train
    other_train_df, val_df = split_by_scales(train_df, 0.3846, country_val_seeds, target)
    
    # Step 3: Split remaining train into ICL train (10% of total) and fine-tune
    total = len(country_df)
    other_train_df_size = len(other_train_df)
    ratio = (total * 0.1) / other_train_df_size
    
    fine_tune_df, train_final_df = split_by_scales(other_train_df, ratio, country_train_seeds, target)
    
    # Verify total sample consistency
    total_check = len(test_df) + len(val_df) + len(train_final_df) + len(fine_tune_df)
    assert total == total_check, f"{wave_info} sample count mismatch!"
    
    return train_final_df, val_df, fine_tune_df, test_df


def process_four_country_data():
    """Process and split four-country dataset"""
    print("🔄 Loading four-country stacked dataset...")
    df = pd.read_csv(os.path.join(STACKED_DATA_PATH, STACKED_FILE))
    
    all_train, all_val, all_finetune, all_test = [], [], [], []
    
    for country_code, country_name in four_country_code_to_name.items():  # Use four_country mapping
        sub_df_country = df[df[country_col] == country_code]
        train_df, val_df, finetune_df, test_df = split_country_data(sub_df_country, country_name)
        
        all_train.append(train_df)
        all_val.append(val_df)
        all_finetune.append(finetune_df)
        all_test.append(test_df)
    
    # Combine all countries
    datasets = {
        'stacked_train.csv': pd.concat(all_train, ignore_index=True),
        'stacked_val.csv': pd.concat(all_val, ignore_index=True),
        'stacked_finetune.csv': pd.concat(all_finetune, ignore_index=True),
        'stacked_test.csv': pd.concat(all_test, ignore_index=True)
    }
    
    # Save datasets
    for filename, dataset in datasets.items():
        dataset.to_csv(os.path.join(SPLIT_OUTPUT_PATH, filename), index=False)
    
    file_list = ", ".join(datasets.keys())
    print(f"\nFour-country data splitting completed:")
    print(f"📁 Files saved: {file_list}")
    print()  
    
    return datasets


def process_charls_data():
    """Process and split CHARLS external validation dataset"""
    print("🔄 Processing CHARLS_wave_4 external validation dataset...")
    df_charls = pd.read_csv(os.path.join(STACKED_DATA_PATH, CHARLS_FILE))
    
    charls_test_size = 0.35
    
    # Step 1: Split CHARLS into train/test (35% test)
    train_charls_full, test_charls = split_by_scales(df_charls, charls_test_size, country_seeds, target)
    
    # Step 2: Subsample CHARLS ICL training set to 10% of total
    final_train_ratio = 0.1 / (1 - charls_test_size)  # 0.125 for 10% of total
    train_charls_final, unused_charls = split_by_scales(train_charls_full, 1 - final_train_ratio, country_train_seeds, target)
    
    # Save datasets
    train_charls_final.to_csv(os.path.join(SPLIT_OUTPUT_PATH, "charls_train.csv"), index=False)
    test_charls.to_csv(os.path.join(SPLIT_OUTPUT_PATH, "charls_test.csv"), index=False)
    
    print("CHARLS dataset splitting completed:")
    print("📁 Files saved: charls_train.csv, charls_test.csv")
    print("\n" + "🔹" * 40)
    
    return train_charls_final, test_charls, unused_charls, df_charls


def verify_and_report_results(four_country_datasets, charls_results):
    """Verify splits and print comprehensive reports"""
    # Load saved datasets for verification
    stacked_train = pd.read_csv(os.path.join(SPLIT_OUTPUT_PATH, "stacked_train.csv"))
    stacked_val = pd.read_csv(os.path.join(SPLIT_OUTPUT_PATH, "stacked_val.csv"))
    stacked_finetune = pd.read_csv(os.path.join(SPLIT_OUTPUT_PATH, "stacked_finetune.csv"))
    stacked_test = pd.read_csv(os.path.join(SPLIT_OUTPUT_PATH, "stacked_test.csv"))
    
    train_charls_final, test_charls, unused_charls, df_charls = charls_results
    
    # Print dataset statistics
    print_dataset_check(stacked_train, "ICL Training Set")
    print_dataset_check(stacked_val, "Validation Set")
    print_dataset_check(stacked_finetune, "Fine-tuning Set")
    print_dataset_check(stacked_test, "Test Set")
    
    print_charls_dataset_check(train_charls_final, "CHARLS ICL Training Set", show_separator=False) 
    print_charls_dataset_check(test_charls, "CHARLS Test Set", show_separator=True)  
    
    # Load data for statistics
    df = pd.read_csv(os.path.join(STACKED_DATA_PATH, STACKED_FILE))
    total_original = len(df)
    total_original_charls = len(df_charls)
    count_test = len(stacked_test)
    count_train = len(stacked_train)
    count_finetune = len(stacked_finetune)
    count_val = len(stacked_val)
    count_charls_test = len(test_charls)
    count_charls_train_final = len(train_charls_final)
    count_unused_charls = len(unused_charls)
    
    # Proportion statistics
    print("\n📊 Four-Country Dataset Proportion Statistics")
    print(f"Total samples (four countries): {total_original:,}")
    
    for name, count in zip(
        ["Test", "ICL Train", "Fine-tune", "Validation"],
        [count_test, count_train, count_finetune, count_val]
    ):
        proportion = count / total_original
        print(f"{name:<12} samples: {count:>6,} | Proportion: {proportion:.2%}")
    
    print("\n📊 CHARLS Dataset Proportion Statistics")
    print(f"Total samples (CHARLS): {total_original_charls:,}")
    for name, count in zip(
        ["Test", "ICL Train", "Unused"],
        [count_charls_test, count_charls_train_final, count_unused_charls]
    ):
        proportion = count / total_original_charls
        print(f"{name:<12} samples: {count:>6,} | Proportion: {proportion:.2%}")
    
    # Overall statistics across all five countries
    total_all_countries = total_original + total_original_charls
    
    print(f"\n📊 Overall Five-Country Dataset Statistics")
    print(f"Total samples (Four countries + CHARLS): {total_all_countries:,}")
    
    # Show scale counts by country for four countries
    for country_code, country_name in four_country_code_to_name.items():
        country_df = df[df[country_col] == country_code]
        scale_count = len(country_df[scale_col].unique())
        wave_info = country_wave_info[country_name]
        print(f"{wave_info:<15} scales: {scale_count:>2}")
    
    # Show scale counts for CHARLS
    charls_scale_count = len(df_charls[scale_col].unique())
    print(f"CHARLS_wave_4   scales: {charls_scale_count:>2}")