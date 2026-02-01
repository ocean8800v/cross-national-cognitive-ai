
# ========== Sensitivity Analysis - 1 Utilities ==========
import pandas as pd
import numpy as np
import random
import os
from decimal import Decimal, ROUND_HALF_UP
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

# ---------- Utility Functions ----------

def decimal_round(data, decimals=3):
    """Precise rounding for single values or iterable (mean) with ROUND_HALF_UP."""
    if hasattr(data, '__iter__') and not isinstance(data, str):
        decimal_values = [Decimal(str(val)) for val in data]
        result = sum(decimal_values) / Decimal(len(decimal_values))
    else:
        result = Decimal(str(data))

    quantize_format = '0.' + '0' * (decimals - 1) + '1'
    return float(result.quantize(Decimal(quantize_format), rounding=ROUND_HALF_UP))

def predict_proba_in_two_batches(clf, X):
    """
    Split input data X into two halves for batch prediction to prevent GPU memory errors.
    Returns concatenated prediction probabilities.
    """
    n = len(X)
    half = n // 2

    X1 = X.iloc[:half]
    X2 = X.iloc[half:]

    prob1 = clf.predict_proba(X1)
    prob2 = clf.predict_proba(X2)

    return np.vstack([prob1, prob2])

def multiclass_auc(y_true, y_prob, average='macro'):
    """Calculate multiclass macro-ROC AUC and individual class one-vs-rest ROC AUC"""
    n_class = 3  # Assuming labels 0,1,2 correspond to Normal/MCI/Dementia

    y_true_bin = label_binarize(y_true, classes=list(range(n_class)))
    aucs = []
    for i in range(n_class):
        if y_true_bin[:, i].sum() == 0 or y_true_bin[:, i].sum() == len(y_true_bin):
            auc = float('nan')
        else:
            auc = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
        aucs.append(auc)

    macro_auc = sum([a for a in aucs if not pd.isnull(a)]) / len([a for a in aucs if not pd.isnull(a)])
    return macro_auc, aucs

def load_baseline_results():
    """Load baseline AUC results from existing CSV files"""

    # Cross-country results (test subset only)
    cross_country_df = pd.read_csv('/content/drive/MyDrive/manuscript/4_results/1_Defualt/Defualt_Model_Performance/unified_model/all_groups_cross_country_4countries.csv')
    cross_country_test = cross_country_df[cross_country_df['subset'] == 'test']

    # CHARLS external validation results
    charls_df = pd.read_csv('/content/drive/MyDrive/manuscript/4_results/1_Defualt/Defualt_Model_Performance/unified_model/all_groups_CHARLS_within_country_test.csv')

    baseline_results = {}
    for group in ['Group1', 'Group2', 'Group3']:
        # Cross-country results
        cross_row = cross_country_test[cross_country_test['group'] == group].iloc[0]
        # CHARLS results
        charls_row = charls_df[charls_df['group'] == group].iloc[0]

        baseline_results[group] = {
            'overall_macro_auc': decimal_round(cross_row['ROC_AUC_Macro_mean']),
            'overall_normal_auc': decimal_round(cross_row['ROC_AUC_Normal_mean']),
            'overall_mci_auc': decimal_round(cross_row['ROC_AUC_MCI_mean']),
            'overall_dementia_auc': decimal_round(cross_row['ROC_AUC_Dementia_mean']),
            'charls_external_auc': decimal_round(charls_row['ROC_AUC_Macro_mean']),
            'charls_external_normal_auc': decimal_round(charls_row['ROC_AUC_Normal_mean']),
            'charls_external_mci_auc': decimal_round(charls_row['ROC_AUC_MCI_mean']),
            'charls_external_dementia_auc': decimal_round(charls_row['ROC_AUC_Dementia_mean'])
        }

    return baseline_results

def calculate_scale_level_auc(df_test, y_test, prob_test, country_col, scale_col):
    """
    Calculate overall AUC by averaging across all scales directly (scale-level averaging)
    instead of country-level then cross-country averaging
    """
    all_scale_aucs = []
    all_scale_normal_aucs = []
    all_scale_mci_aucs = []
    all_scale_dementia_aucs = []

    # Iterate through all countries and scales
    for country_code in sorted(df_test[country_col].unique()):
        country_test_df = df_test[df_test[country_col] == country_code]

        for scale_val in sorted(country_test_df[scale_col].unique()):
            scale_test_df = country_test_df[country_test_df[scale_col] == scale_val]

            if len(scale_test_df) == 0:
                continue

            scale_indices = scale_test_df.index
            scale_y = y_test.loc[scale_indices]
            scale_prob = prob_test[scale_indices]

            if len(scale_y.unique()) < 2:  # Skip scales with only one class
                continue

            scale_auc, scale_aucs_list = multiclass_auc(scale_y, scale_prob)
            if not pd.isnull(scale_auc):
                all_scale_aucs.append(scale_auc)

            # Individual class AUCs
            if len(scale_aucs_list) >= 3:
                if not pd.isnull(scale_aucs_list[0]):  # Normal
                    all_scale_normal_aucs.append(scale_aucs_list[0])
                if not pd.isnull(scale_aucs_list[1]):  # MCI
                    all_scale_mci_aucs.append(scale_aucs_list[1])
                if not pd.isnull(scale_aucs_list[2]):  # Dementia
                    all_scale_dementia_aucs.append(scale_aucs_list[2])

    # Calculate overall AUCs as direct average across all scales
    overall_macro_auc = decimal_round(np.mean(all_scale_aucs)) if all_scale_aucs else float('nan')
    overall_normal_auc = decimal_round(np.mean(all_scale_normal_aucs)) if all_scale_normal_aucs else float('nan')
    overall_mci_auc = decimal_round(np.mean(all_scale_mci_aucs)) if all_scale_mci_aucs else float('nan')
    overall_dementia_auc = decimal_round(np.mean(all_scale_dementia_aucs)) if all_scale_dementia_aucs else float('nan')

    return overall_macro_auc, overall_normal_auc, overall_mci_auc, overall_dementia_auc

def calculate_charls_scale_level_auc(test_charls, clf, predictors, scale_col, target):
    """Calculate CHARLS external validation AUC using scale-level averaging"""
    charls_scale_aucs = []
    charls_scale_normal_aucs = []
    charls_scale_mci_aucs = []
    charls_scale_dementia_aucs = []

    for scale_val in sorted(test_charls[scale_col].unique()):
        scale_test_df = test_charls[test_charls[scale_col] == scale_val]

        if len(scale_test_df) == 0:
            continue

        X_test_charls_scale = scale_test_df[predictors]
        y_test_charls_scale = scale_test_df[target].astype(int)

        if len(y_test_charls_scale.unique()) < 2:  # Skip scales with only one class
            continue

        # CHARLS external validation (4-country joint model predicting CHARLS current scale)
        charls_scale_prob = predict_proba_in_two_batches(clf, X_test_charls_scale)
        charls_scale_auc, charls_scale_aucs_list = multiclass_auc(y_test_charls_scale, charls_scale_prob)

        if not pd.isnull(charls_scale_auc):
            charls_scale_aucs.append(charls_scale_auc)

        # Individual class AUCs for CHARLS
        if len(charls_scale_aucs_list) >= 3:
            if not pd.isnull(charls_scale_aucs_list[0]):  # Normal
                charls_scale_normal_aucs.append(charls_scale_aucs_list[0])
            if not pd.isnull(charls_scale_aucs_list[1]):  # MCI
                charls_scale_mci_aucs.append(charls_scale_aucs_list[1])
            if not pd.isnull(charls_scale_aucs_list[2]):  # Dementia
                charls_scale_dementia_aucs.append(charls_scale_aucs_list[2])

    # CHARLS external validation AUC = average across scales
    charls_external_auc = decimal_round(np.mean(charls_scale_aucs)) if charls_scale_aucs else float('nan')
    charls_external_normal_auc = decimal_round(np.mean(charls_scale_normal_aucs)) if charls_scale_normal_aucs else float('nan')
    charls_external_mci_auc = decimal_round(np.mean(charls_scale_mci_aucs)) if charls_scale_mci_aucs else float('nan')
    charls_external_dementia_auc = decimal_round(np.mean(charls_scale_dementia_aucs)) if charls_scale_dementia_aucs else float('nan')

    return charls_external_auc, charls_external_normal_auc, charls_external_mci_auc, charls_external_dementia_auc

# ---------- Results Analysis Functions ----------

def analyze_robustness(z_scores):
    """Analyze robustness and generate conclusion"""
    if all(abs(z) < 1 for z in z_scores):
        return "All baseline metrics fall within ±1SD of random combinations → Results show good robustness"
    elif all(abs(z) < 2 for z in z_scores):
        return "All baseline metrics fall within ±2SD of random combinations → Results show acceptable robustness"
    else:
        return "Some metrics exceed ±2SD range → Results require further analysis"

def print_group_results(group_name, stats):
    """Print complete results for a single group"""
    print(f"\n" + "="*80)
    print(f"📊 {group_name} Results:")
    print("="*80)

    baseline = stats['baseline']
    random = stats['random']
    z_scores = stats['z_scores']

    print(f"Baseline combination results:")
    print(f"- Overall Macro AUC: {baseline['overall']}")
    print(f"- Overall Normal (Cognitive Impairment) AUC: {baseline['normal']}")
    print(f"- Overall MCI AUC: {baseline['mci']}")
    print(f"- Overall Dementia AUC: {baseline['dementia']}")
    print(f"- CHARLS External AUC: {baseline['charls']}")
    print(f"- CHARLS External Normal (Cognitive Impairment) AUC: {baseline['charls_normal']}")
    print(f"- CHARLS External MCI AUC: {baseline['charls_mci']}")
    print(f"- CHARLS External Dementia AUC: {baseline['charls_dementia']}")

    print(f"\nRandom 100 combinations:")
    print(f"- Overall Macro AUC: {random['overall_mean']} ± {random['overall_std']}")
    print(f"- Overall Normal (Cognitive Impairment) AUC: {random['normal_mean']} ± {random['normal_std']}")
    print(f"- Overall MCI AUC: {random['mci_mean']} ± {random['mci_std']}")
    print(f"- Overall Dementia AUC: {random['dementia_mean']} ± {random['dementia_std']}")
    print(f"- CHARLS External AUC: {random['charls_mean']} ± {random['charls_std']}")
    print(f"- CHARLS External Normal (Cognitive Impairment) AUC: {random['charls_normal_mean']} ± {random['charls_normal_std']}")
    print(f"- CHARLS External MCI AUC: {random['charls_mci_mean']} ± {random['charls_mci_std']}")
    print(f"- CHARLS External Dementia AUC: {random['charls_dementia_mean']} ± {random['charls_dementia_std']}")

    print(f"\nZ-score analysis:")
    print(f"- Overall Macro AUC Z-score: {decimal_round(z_scores[0], 2)}")
    print(f"- Overall Normal (Cognitive Impairment) AUC Z-score: {decimal_round(z_scores[1], 2)}")
    print(f"- Overall MCI AUC Z-score: {decimal_round(z_scores[2], 2)}")
    print(f"- Overall Dementia AUC Z-score: {decimal_round(z_scores[3], 2)}")
    print(f"- CHARLS External AUC Z-score: {decimal_round(z_scores[4], 2)}")
    print(f"- CHARLS External Normal (Cognitive Impairment) AUC Z-score: {decimal_round(z_scores[5], 2)}")
    print(f"- CHARLS External MCI AUC Z-score: {decimal_round(z_scores[6], 2)}")
    print(f"- CHARLS External Dementia AUC Z-score: {decimal_round(z_scores[7], 2)}")

    print(f"\nConclusion: {analyze_robustness(z_scores)}")

    # Additional detailed analysis if needed
    if not all(abs(z) < 2 for z in z_scores):
        print("Detailed analysis:")
        metrics_names = ['Overall Macro AUC', 'Overall Normal (Cognitive Impairment) AUC', 'Overall MCI AUC', 'Overall Dementia AUC',
                        'CHARLS External AUC', 'CHARLS External Normal (Cognitive Impairment) AUC', 'CHARLS External MCI AUC', 'CHARLS External Dementia AUC']
        for i, z in enumerate(z_scores):
            if abs(z) >= 2:
                print(f"  - {metrics_names[i]} exceeds ±2SD range (Z-score: {decimal_round(z, 2)})")

def calculate_overall_statistics(all_group_results):
    """Calculate overall statistics across all groups"""
    baseline_overall_aucs = []
    baseline_overall_normal_aucs = []
    baseline_overall_mci_aucs = []
    baseline_overall_dementia_aucs = []
    baseline_charls_aucs = []
    baseline_charls_normal_aucs = []
    baseline_charls_mci_aucs = []
    baseline_charls_dementia_aucs = []

    random_overall_aucs = []
    random_overall_normal_aucs = []
    random_overall_mci_aucs = []
    random_overall_dementia_aucs = []
    random_charls_aucs = []
    random_charls_normal_aucs = []
    random_charls_mci_aucs = []
    random_charls_dementia_aucs = []

    # Collect data for overall calculation
    for group_name, results_df in all_group_results.items():
        baseline_results_df = results_df[results_df['is_baseline']]
        random_results_df = results_df[~results_df['is_baseline']]

        # Baseline results for overall calculation
        baseline_overall_aucs.append(baseline_results_df['overall_macro_auc'].values[0])
        baseline_overall_normal_aucs.append(baseline_results_df['overall_normal_auc'].values[0])
        baseline_overall_mci_aucs.append(baseline_results_df['overall_mci_auc'].values[0])
        baseline_overall_dementia_aucs.append(baseline_results_df['overall_dementia_auc'].values[0])
        baseline_charls_aucs.append(baseline_results_df['charls_external_auc'].values[0])
        baseline_charls_normal_aucs.append(baseline_results_df['charls_external_normal_auc'].values[0])
        baseline_charls_mci_aucs.append(baseline_results_df['charls_external_mci_auc'].values[0])
        baseline_charls_dementia_aucs.append(baseline_results_df['charls_external_dementia_auc'].values[0])

        # Random results for overall calculation
        random_overall_aucs.extend(random_results_df['overall_macro_auc'].tolist())
        random_overall_normal_aucs.extend(random_results_df['overall_normal_auc'].tolist())
        random_overall_mci_aucs.extend(random_results_df['overall_mci_auc'].tolist())
        random_overall_dementia_aucs.extend(random_results_df['overall_dementia_auc'].tolist())
        random_charls_aucs.extend(random_results_df['charls_external_auc'].tolist())
        random_charls_normal_aucs.extend(random_results_df['charls_external_normal_auc'].tolist())
        random_charls_mci_aucs.extend(random_results_df['charls_external_mci_auc'].tolist())
        random_charls_dementia_aucs.extend(random_results_df['charls_external_dementia_auc'].tolist())

    # Calculate averages and standard deviations
    avg_baseline_overall = decimal_round(np.mean(baseline_overall_aucs))
    avg_baseline_overall_normal = decimal_round(np.mean(baseline_overall_normal_aucs))
    avg_baseline_overall_mci = decimal_round(np.mean(baseline_overall_mci_aucs))
    avg_baseline_overall_dementia = decimal_round(np.mean(baseline_overall_dementia_aucs))
    avg_baseline_charls = decimal_round(np.mean(baseline_charls_aucs))
    avg_baseline_charls_normal = decimal_round(np.mean(baseline_charls_normal_aucs))
    avg_baseline_charls_mci = decimal_round(np.mean(baseline_charls_mci_aucs))
    avg_baseline_charls_dementia = decimal_round(np.mean(baseline_charls_dementia_aucs))

    avg_random_overall_mean = decimal_round(np.mean(random_overall_aucs))
    avg_random_overall_std = decimal_round(np.std(random_overall_aucs))
    avg_random_overall_normal_mean = decimal_round(np.mean(random_overall_normal_aucs))
    avg_random_overall_normal_std = decimal_round(np.std(random_overall_normal_aucs))
    avg_random_overall_mci_mean = decimal_round(np.mean(random_overall_mci_aucs))
    avg_random_overall_mci_std = decimal_round(np.std(random_overall_mci_aucs))
    avg_random_overall_dementia_mean = decimal_round(np.mean(random_overall_dementia_aucs))
    avg_random_overall_dementia_std = decimal_round(np.std(random_overall_dementia_aucs))

    avg_random_charls_mean = decimal_round(np.mean(random_charls_aucs))
    avg_random_charls_std = decimal_round(np.std(random_charls_aucs))
    avg_random_charls_normal_mean = decimal_round(np.mean(random_charls_normal_aucs))
    avg_random_charls_normal_std = decimal_round(np.std(random_charls_normal_aucs))
    avg_random_charls_mci_mean = decimal_round(np.mean(random_charls_mci_aucs))
    avg_random_charls_mci_std = decimal_round(np.std(random_charls_mci_aucs))
    avg_random_charls_dementia_mean = decimal_round(np.mean(random_charls_dementia_aucs))
    avg_random_charls_dementia_std = decimal_round(np.std(random_charls_dementia_aucs))

    # Calculate Z-scores
    overall_z_score = (avg_baseline_overall - avg_random_overall_mean) / avg_random_overall_std
    overall_normal_z_score = (avg_baseline_overall_normal - avg_random_overall_normal_mean) / avg_random_overall_normal_std
    overall_mci_z_score = (avg_baseline_overall_mci - avg_random_overall_mci_mean) / avg_random_overall_mci_std
    overall_dementia_z_score = (avg_baseline_overall_dementia - avg_random_overall_dementia_mean) / avg_random_overall_dementia_std
    charls_z_score = (avg_baseline_charls - avg_random_charls_mean) / avg_random_charls_std
    charls_normal_z_score = (avg_baseline_charls_normal - avg_random_charls_normal_mean) / avg_random_charls_normal_std
    charls_mci_z_score = (avg_baseline_charls_mci - avg_random_charls_mci_mean) / avg_random_charls_mci_std
    charls_dementia_z_score = (avg_baseline_charls_dementia - avg_random_charls_dementia_mean) / avg_random_charls_dementia_std

    return {
        'baseline': {
            'overall': avg_baseline_overall,
            'normal': avg_baseline_overall_normal,
            'mci': avg_baseline_overall_mci,
            'dementia': avg_baseline_overall_dementia,
            'charls': avg_baseline_charls,
            'charls_normal': avg_baseline_charls_normal,
            'charls_mci': avg_baseline_charls_mci,
            'charls_dementia': avg_baseline_charls_dementia
        },
        'random': {
            'overall_mean': avg_random_overall_mean, 'overall_std': avg_random_overall_std,
            'normal_mean': avg_random_overall_normal_mean, 'normal_std': avg_random_overall_normal_std,
            'mci_mean': avg_random_overall_mci_mean, 'mci_std': avg_random_overall_mci_std,
            'dementia_mean': avg_random_overall_dementia_mean, 'dementia_std': avg_random_overall_dementia_std,
            'charls_mean': avg_random_charls_mean, 'charls_std': avg_random_charls_std,
            'charls_normal_mean': avg_random_charls_normal_mean, 'charls_normal_std': avg_random_charls_normal_std,
            'charls_mci_mean': avg_random_charls_mci_mean, 'charls_mci_std': avg_random_charls_mci_std,
            'charls_dementia_mean': avg_random_charls_dementia_mean, 'charls_dementia_std': avg_random_charls_dementia_std
        },
        'z_scores': [overall_z_score, overall_normal_z_score, overall_mci_z_score, overall_dementia_z_score,
                    charls_z_score, charls_normal_z_score, charls_mci_z_score, charls_dementia_z_score]
    }

# ---------- New Extracted Functions ----------

def generate_analysis_setup(n_combinations=100, seed=42):
    """Generate baseline results and random seed combinations for analysis"""
    # Load baseline combination results
    baseline_combination = (755, 87, 100)
    baseline_results = load_baseline_results()
    
    # Generate random combinations
    random.seed(seed)
    random_combinations = []
    while len(random_combinations) < n_combinations:
        train_seed = random.randint(1, 999)
        val_seed = random.randint(1, 999)
        test_seed = random.randint(1, 999)
        combo = (train_seed, val_seed, test_seed)
        if combo != baseline_combination and combo not in random_combinations:
            random_combinations.append(combo)
    
    return baseline_combination, baseline_results, random_combinations

def split_country_data(df, country_code, country_col, scale_col, target, 
                      train_seed, val_seed, test_seed, test_size=0.35):
    """Split data for a single country: train/test/val + 10% subsampling"""
    sub_df_country = df[df[country_col] == country_code]
    
    # Step 1: Split train/test by scale
    test_list, train_list = [], []
    for scale_val in sorted(sub_df_country[scale_col].unique()):
        sub_df = sub_df_country[sub_df_country[scale_col] == scale_val]
        tr, te = train_test_split(
            sub_df, test_size=test_size, stratify=sub_df[target], random_state=test_seed
        )
        train_list.append(tr)
        test_list.append(te)

    train_df = pd.concat(train_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)

    # Step 2: Split validation from training set
    val_list, other_train_list = [], []
    for scale_val in sorted(train_df[scale_col].unique()):
        sub_df = train_df[train_df[scale_col] == scale_val]
        other_train, val = train_test_split(
            sub_df, test_size=0.3846, stratify=sub_df[target], random_state=val_seed
        )
        other_train_list.append(other_train)
        val_list.append(val)

    other_train_df = pd.concat(other_train_list, ignore_index=True)
    val_df = pd.concat(val_list, ignore_index=True)

    # Step 3: Subsample training data (10%)
    train_subsample_list = []
    total = len(sub_df_country)
    other_train_df_size = len(other_train_df)
    ratio = (total * 0.10) / other_train_df_size

    for scale_val in sorted(other_train_df[scale_col].unique()):
        sub_df = other_train_df[other_train_df[scale_col] == scale_val]
        _, subsample = train_test_split(
            sub_df, test_size=ratio, stratify=sub_df[target], random_state=train_seed
        )
        train_subsample_list.append(subsample)

    train_final = pd.concat(train_subsample_list, ignore_index=True)
    
    return test_df, train_final, val_df

def split_charls_data(df_charls, scale_col, target, test_seed, test_size=0.35):
    """Split CHARLS data for test set"""
    charls_test_list = []
    for scale_val in sorted(df_charls[scale_col].unique()):
        sub_df = df_charls[df_charls[scale_col] == scale_val]
        _, te = train_test_split(
            sub_df, test_size=test_size, stratify=sub_df[target], random_state=test_seed
        )
        charls_test_list.append(te)
    return pd.concat(charls_test_list, ignore_index=True)

def generate_summary_csv(overall_stats, analysis_folder):
    """Generate final summary CSV file"""
    baseline = overall_stats['baseline']
    random = overall_stats['random']
    z_scores = overall_stats['z_scores']

    summary_data = [
        {
            'Metric': 'Overall Macro AUC',
            'Baseline_AUC': baseline['overall'],
            'Random_Mean': random['overall_mean'],
            'Random_SD': random['overall_std'],
            'Z_score': decimal_round(z_scores[0], 2)
        },
        {
            'Metric': 'Overall Normal AUC',
            'Baseline_AUC': baseline['normal'],
            'Random_Mean': random['normal_mean'],
            'Random_SD': random['normal_std'],
            'Z_score': decimal_round(z_scores[1], 2)
        },
        {
            'Metric': 'Overall MCI AUC',
            'Baseline_AUC': baseline['mci'],
            'Random_Mean': random['mci_mean'],
            'Random_SD': random['mci_std'],
            'Z_score': decimal_round(z_scores[2], 2)
        },
        {
            'Metric': 'Overall Dementia AUC',
            'Baseline_AUC': baseline['dementia'],
            'Random_Mean': random['dementia_mean'],
            'Random_SD': random['dementia_std'],
            'Z_score': decimal_round(z_scores[3], 2)
        },
        {
            'Metric': 'CHARLS External AUC',
            'Baseline_AUC': baseline['charls'],
            'Random_Mean': random['charls_mean'],
            'Random_SD': random['charls_std'],
            'Z_score': decimal_round(z_scores[4], 2)
        },
        {
            'Metric': 'CHARLS External Normal AUC',
            'Baseline_AUC': baseline['charls_normal'],
            'Random_Mean': random['charls_normal_mean'],
            'Random_SD': random['charls_normal_std'],
            'Z_score': decimal_round(z_scores[5], 2)
        },
        {
            'Metric': 'CHARLS External MCI AUC',
            'Baseline_AUC': baseline['charls_mci'],
            'Random_Mean': random['charls_mci_mean'],
            'Random_SD': random['charls_mci_std'],
            'Z_score': decimal_round(z_scores[6], 2)
        },
        {
            'Metric': 'CHARLS External Dementia AUC',
            'Baseline_AUC': baseline['charls_dementia'],
            'Random_Mean': random['charls_dementia_mean'],
            'Random_SD': random['charls_dementia_std'],
            'Z_score': decimal_round(z_scores[7], 2)
        }
    ]

    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(analysis_folder, 'robustness_analysis_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    return summary_file