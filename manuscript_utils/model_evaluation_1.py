# ========== Model Evaluation - 1 Utilities ==========
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

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
    except ValueError as e:
        print(f"AUC calculation failed: {e}")
        return None, None, None, None

def calculate_performance_statistics(dataset_scale_df, group_name, dataset_name):
    """
    Calculate within-country and cross-country performance statistics from scale-level results

    Args:
        dataset_scale_df: DataFrame with scale-level results containing ROC_AUC metrics
        group_name: Current group name (e.g., 'Group1', 'Group2', 'Group3')
        dataset_name: Dataset subset name ('val' or 'test')

    Returns:
        within_country_results: List of dictionaries containing within-country statistics
        cross_country_result: Dictionary containing cross-country statistics (scale-level averaging)
    """
    within_country_results = []

    # Within-Country Statistics - Calculate statistics for each country separately
    for country in dataset_scale_df['country'].unique():
        country_data = dataset_scale_df[dataset_scale_df['country'] == country]

        within_country_result = {
            'group': group_name,
            'subset': dataset_name,
            'country': country,
            'ROC_AUC_Macro_mean': decimal_round(country_data['ROC_AUC_Macro'], 3),
            'ROC_AUC_Normal_mean': decimal_round(country_data['ROC_AUC_Normal'], 3),
            'ROC_AUC_MCI_mean': decimal_round(country_data['ROC_AUC_MCI'], 3),
            'ROC_AUC_Dementia_mean': decimal_round(country_data['ROC_AUC_Dementia'], 3),
            'ROC_AUC_Macro_std': decimal_round(country_data['ROC_AUC_Macro'].std(), 3),
            'ROC_AUC_Normal_std': decimal_round(country_data['ROC_AUC_Normal'].std(), 3),
            'ROC_AUC_MCI_std': decimal_round(country_data['ROC_AUC_MCI'].std(), 3),
            'ROC_AUC_Dementia_std': decimal_round(country_data['ROC_AUC_Dementia'].std(), 3),
            'ROC_AUC_Macro_max': decimal_round(country_data['ROC_AUC_Macro'].max(), 3),
            'ROC_AUC_Normal_max': decimal_round(country_data['ROC_AUC_Normal'].max(), 3),
            'ROC_AUC_MCI_max': decimal_round(country_data['ROC_AUC_MCI'].max(), 3),
            'ROC_AUC_Dementia_max': decimal_round(country_data['ROC_AUC_Dementia'].max(), 3),
            'ROC_AUC_Macro_min': decimal_round(country_data['ROC_AUC_Macro'].min(), 3),
            'ROC_AUC_Normal_min': decimal_round(country_data['ROC_AUC_Normal'].min(), 3),
            'ROC_AUC_MCI_min': decimal_round(country_data['ROC_AUC_MCI'].min(), 3),
            'ROC_AUC_Dementia_min': decimal_round(country_data['ROC_AUC_Dementia'].min(), 3),
            'ROC_AUC_Macro_median': decimal_round(country_data['ROC_AUC_Macro'].median(), 3),
            'ROC_AUC_Normal_median': decimal_round(country_data['ROC_AUC_Normal'].median(), 3),
            'ROC_AUC_MCI_median': decimal_round(country_data['ROC_AUC_MCI'].median(), 3),
            'ROC_AUC_Dementia_median': decimal_round(country_data['ROC_AUC_Dementia'].median(), 3),
            'sample_size': country_data['sample_size'].sum()
        }

        within_country_results.append(within_country_result)

    # Cross-Country Statistics - Scale-level averaging across all countries
    cross_country_result = {
        'group': group_name,
        'subset': dataset_name,
        'ROC_AUC_Macro_mean': decimal_round(dataset_scale_df['ROC_AUC_Macro'], 3),
        'ROC_AUC_Normal_mean': decimal_round(dataset_scale_df['ROC_AUC_Normal'], 3),
        'ROC_AUC_MCI_mean': decimal_round(dataset_scale_df['ROC_AUC_MCI'], 3),
        'ROC_AUC_Dementia_mean': decimal_round(dataset_scale_df['ROC_AUC_Dementia'], 3),
        'ROC_AUC_Macro_std': decimal_round(dataset_scale_df['ROC_AUC_Macro'].std(), 3),
        'ROC_AUC_Normal_std': decimal_round(dataset_scale_df['ROC_AUC_Normal'].std(), 3),
        'ROC_AUC_MCI_std': decimal_round(dataset_scale_df['ROC_AUC_MCI'].std(), 3),
        'ROC_AUC_Dementia_std': decimal_round(dataset_scale_df['ROC_AUC_Dementia'].std(), 3),
        'ROC_AUC_Macro_max': decimal_round(dataset_scale_df['ROC_AUC_Macro'].max(), 3),
        'ROC_AUC_Normal_max': decimal_round(dataset_scale_df['ROC_AUC_Normal'].max(), 3),
        'ROC_AUC_MCI_max': decimal_round(dataset_scale_df['ROC_AUC_MCI'].max(), 3),
        'ROC_AUC_Dementia_max': decimal_round(dataset_scale_df['ROC_AUC_Dementia'].max(), 3),
        'ROC_AUC_Macro_min': decimal_round(dataset_scale_df['ROC_AUC_Macro'].min(), 3),
        'ROC_AUC_Normal_min': decimal_round(dataset_scale_df['ROC_AUC_Normal'].min(), 3),
        'ROC_AUC_MCI_min': decimal_round(dataset_scale_df['ROC_AUC_MCI'].min(), 3),
        'ROC_AUC_Dementia_min': decimal_round(dataset_scale_df['ROC_AUC_Dementia'].min(), 3),
        'ROC_AUC_Macro_median': decimal_round(dataset_scale_df['ROC_AUC_Macro'].median(), 3),
        'ROC_AUC_Normal_median': decimal_round(dataset_scale_df['ROC_AUC_Normal'].median(), 3),
        'ROC_AUC_MCI_median': decimal_round(dataset_scale_df['ROC_AUC_MCI'].median(), 3),
        'ROC_AUC_Dementia_median': decimal_round(dataset_scale_df['ROC_AUC_Dementia'].median(), 3),
        'total_sample_size': dataset_scale_df['sample_size'].sum()
    }

    return within_country_results, cross_country_result

def calculate_charls_statistics(charls_results_df, group_name):
    """
    Calculate CHARLS performance statistics from scale-level results

    Args:
        charls_results_df: DataFrame with CHARLS scale-level results containing ROC_AUC metrics
        group_name: Current group name (e.g., 'Group1', 'Group2', 'Group3')

    Returns:
        dict: Dictionary containing CHARLS statistics
    """
    return {
        'group': group_name,
        'country': 'CHARLS',
        'ROC_AUC_Macro_mean': decimal_round(charls_results_df['ROC_AUC_Macro'], 3),
        'ROC_AUC_Normal_mean': decimal_round(charls_results_df['ROC_AUC_Normal'], 3),
        'ROC_AUC_MCI_mean': decimal_round(charls_results_df['ROC_AUC_MCI'], 3),
        'ROC_AUC_Dementia_mean': decimal_round(charls_results_df['ROC_AUC_Dementia'], 3),
        'ROC_AUC_Macro_std': decimal_round(charls_results_df['ROC_AUC_Macro'].std(), 3),
        'ROC_AUC_Normal_std': decimal_round(charls_results_df['ROC_AUC_Normal'].std(), 3),
        'ROC_AUC_MCI_std': decimal_round(charls_results_df['ROC_AUC_MCI'].std(), 3),
        'ROC_AUC_Dementia_std': decimal_round(charls_results_df['ROC_AUC_Dementia'].std(), 3),
        'ROC_AUC_Macro_max': decimal_round(charls_results_df['ROC_AUC_Macro'].max(), 3),
        'ROC_AUC_Normal_max': decimal_round(charls_results_df['ROC_AUC_Normal'].max(), 3),
        'ROC_AUC_MCI_max': decimal_round(charls_results_df['ROC_AUC_MCI'].max(), 3),
        'ROC_AUC_Dementia_max': decimal_round(charls_results_df['ROC_AUC_Dementia'].max(), 3),
        'ROC_AUC_Macro_min': decimal_round(charls_results_df['ROC_AUC_Macro'].min(), 3),
        'ROC_AUC_Normal_min': decimal_round(charls_results_df['ROC_AUC_Normal'].min(), 3),
        'ROC_AUC_MCI_min': decimal_round(charls_results_df['ROC_AUC_MCI'].min(), 3),
        'ROC_AUC_Dementia_min': decimal_round(charls_results_df['ROC_AUC_Dementia'].min(), 3),
        'ROC_AUC_Macro_median': decimal_round(charls_results_df['ROC_AUC_Macro'].median(), 3),
        'ROC_AUC_Normal_median': decimal_round(charls_results_df['ROC_AUC_Normal'].median(), 3),
        'ROC_AUC_MCI_median': decimal_round(charls_results_df['ROC_AUC_MCI'].median(), 3),
        'ROC_AUC_Dementia_median': decimal_round(charls_results_df['ROC_AUC_Dementia'].median(), 3),
        'sample_size': charls_results_df['sample_size'].sum()
    }
