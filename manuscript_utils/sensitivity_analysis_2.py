
# ========== Sensitivity Analysis - 2 Utilities ==========
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
from sklearn.model_selection import train_test_split
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
    """Predict class probabilities in two batches to prevent OOM on large datasets."""
    n = len(X)
    if n == 0:
        return np.empty((0, 3), dtype=float)
    half = n // 2
    X1 = X.iloc[:half]
    X2 = X.iloc[half:]
    prob1 = clf.predict_proba(X1)
    prob2 = clf.predict_proba(X2)
    return np.vstack([prob1, prob2])

def stratified_sampling_by_country_scale(df, target_percentage, target_col, country_col, scale_col, random_state=42):
    """Apply stratified sampling by country and scale"""
    if target_percentage == 100:
        return df.copy()

    sampled_list = []
    sampling_ratio = target_percentage / 100.0

    for country_code in sorted(df[country_col].unique()):
        country_data = df[df[country_col] == country_code]

        for scale in sorted(country_data[scale_col].unique()):
            scale_df = country_data[country_data[scale_col] == scale]

            # Check minimum class count
            class_counts = scale_df[target_col].value_counts()
            min_class_count = class_counts.min()

            # Keep all if too few samples
            if min_class_count < 5 or len(scale_df) < 10:
                sampled_list.append(scale_df)
            else:
                try:
                    selected_part, _ = train_test_split(
                        scale_df,
                        test_size=(1 - sampling_ratio),
                        stratify=scale_df[target_col],
                        random_state=random_state
                    )
                    sampled_list.append(selected_part)
                except ValueError:
                    # Use random sampling if stratified fails
                    selected_part = scale_df.sample(
                        n=max(1, int(len(scale_df) * sampling_ratio)),
                        random_state=random_state
                    )
                    sampled_list.append(selected_part)

    return pd.concat(sampled_list, ignore_index=True)

def calculate_auc_metrics(df_scale, target):
    """Compute ROC-AUC (macro + OvR for each class) for a scale subset."""
    try:
        scale_probs = np.column_stack([df_scale["pred_prob_normal"],
                                     df_scale["pred_prob_mci"],
                                     df_scale["pred_prob_dementia"]])
        auc_macro = roc_auc_score(df_scale[target], scale_probs, multi_class='ovr', average='macro')
        auc_normal = roc_auc_score((df_scale[target] == 0).astype(int), df_scale["pred_prob_normal"])
        auc_mci = roc_auc_score((df_scale[target] == 1).astype(int), df_scale["pred_prob_mci"])
        auc_dementia = roc_auc_score((df_scale[target] == 2).astype(int), df_scale["pred_prob_dementia"])
        return auc_macro, auc_normal, auc_mci, auc_dementia
    except ValueError:
        return None, None, None, None

def calculate_performance_statistics(dataset_scale_df, group_name, model_name, sampling_percentage):
    """Calculate within-country and cross-country performance statistics from scale-level results"""
    within_country_results = []

    # Within-Country Statistics
    for country in dataset_scale_df['country'].unique():
        country_data = dataset_scale_df[dataset_scale_df['country'] == country]

        within_country_result = {
            'sampling_percentage': sampling_percentage,
            'group': group_name,
            'model': model_name,
            'country': country,
            'ROC_AUC_Macro_mean': decimal_round(country_data['ROC_AUC_Macro'], 3),
            'ROC_AUC_Normal_mean': decimal_round(country_data['ROC_AUC_Normal'], 3),
            'ROC_AUC_MCI_mean': decimal_round(country_data['ROC_AUC_MCI'], 3),
            'ROC_AUC_Dementia_mean': decimal_round(country_data['ROC_AUC_Dementia'], 3),
            'ROC_AUC_Macro_std': decimal_round(country_data['ROC_AUC_Macro'].std(), 3),
            'ROC_AUC_Normal_std': decimal_round(country_data['ROC_AUC_Normal'].std(), 3),
            'ROC_AUC_MCI_std': decimal_round(country_data['ROC_AUC_MCI'].std(), 3),
            'ROC_AUC_Dementia_std': decimal_round(country_data['ROC_AUC_Dementia'].std(), 3),
            'sample_size': country_data['sample_size'].sum()
        }

        within_country_results.append(within_country_result)

    # Cross-Country Statistics - Scale-level averaging across all countries
    cross_country_result = {
        'sampling_percentage': sampling_percentage,
        'group': group_name,
        'model': model_name,
        'ROC_AUC_Macro_mean': decimal_round(dataset_scale_df['ROC_AUC_Macro'], 3),
        'ROC_AUC_Normal_mean': decimal_round(dataset_scale_df['ROC_AUC_Normal'], 3),
        'ROC_AUC_MCI_mean': decimal_round(dataset_scale_df['ROC_AUC_MCI'], 3),
        'ROC_AUC_Dementia_mean': decimal_round(dataset_scale_df['ROC_AUC_Dementia'], 3),
        'ROC_AUC_Macro_std': decimal_round(dataset_scale_df['ROC_AUC_Macro'].std(), 3),
        'ROC_AUC_Normal_std': decimal_round(dataset_scale_df['ROC_AUC_Normal'].std(), 3),
        'ROC_AUC_MCI_std': decimal_round(dataset_scale_df['ROC_AUC_MCI'].std(), 3),
        'ROC_AUC_Dementia_std': decimal_round(dataset_scale_df['ROC_AUC_Dementia'].std(), 3),
        'total_sample_size': dataset_scale_df['sample_size'].sum()
    }

    return within_country_results, cross_country_result

def calculate_charls_statistics(charls_scale_df, group_name, model_name, sampling_percentage):
  
    charls_country_result = {
        'sampling_percentage': sampling_percentage,
        'group': group_name,
        'model': model_name,
        'country': 'CHARLS',
        'ROC_AUC_Macro_mean': decimal_round(charls_scale_df['ROC_AUC_Macro'], 3),
        'ROC_AUC_Normal_mean': decimal_round(charls_scale_df['ROC_AUC_Normal'], 3),
        'ROC_AUC_MCI_mean': decimal_round(charls_scale_df['ROC_AUC_MCI'], 3),
        'ROC_AUC_Dementia_mean': decimal_round(charls_scale_df['ROC_AUC_Dementia'], 3),
        'ROC_AUC_Macro_std': decimal_round(charls_scale_df['ROC_AUC_Macro'].std(), 3),
        'ROC_AUC_Normal_std': decimal_round(charls_scale_df['ROC_AUC_Normal'].std(), 3),
        'ROC_AUC_MCI_std': decimal_round(charls_scale_df['ROC_AUC_MCI'].std(), 3),
        'ROC_AUC_Dementia_std': decimal_round(charls_scale_df['ROC_AUC_Dementia'].std(), 3),
        'sample_size': charls_scale_df['sample_size'].sum()
    }
    return charls_country_result