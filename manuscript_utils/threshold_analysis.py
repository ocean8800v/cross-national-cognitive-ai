# ========== Threshold Analysis ==========
import os
import numpy as np
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
from sklearn.metrics import roc_auc_score

def decimal_round(data, decimals=3):
    if hasattr(data, '__iter__') and not isinstance(data, str):
        decimal_values = [Decimal(str(val)) for val in data]
        result = sum(decimal_values) / Decimal(len(decimal_values))
    else:
        result = Decimal(str(data))
    quantize_format = '0.' + '0' * (decimals - 1) + '1'
    return float(result.quantize(Decimal(quantize_format), rounding=ROUND_HALF_UP))

def bootstrap_metrics_ci(y_true, y_prob, threshold, n_bootstrap=1000, random_state=42):
    """
    Calculate confidence intervals for sensitivity and specificity at a given threshold using bootstrap
    """
    np.random.seed(random_state)
    n_samples = len(y_true)

    sens_values = []
    spec_values = []

    for i in range(n_bootstrap):

        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_prob_boot = y_prob[indices]

        if len(np.unique(y_true_boot)) < 2:
            continue

        pred_boot = (y_prob_boot >= threshold).astype(int)

        TP = np.sum((pred_boot == 1) & (y_true_boot == 1))
        FP = np.sum((pred_boot == 1) & (y_true_boot == 0))
        FN = np.sum((pred_boot == 0) & (y_true_boot == 1))
        TN = np.sum((pred_boot == 0) & (y_true_boot == 0))

        if (TP + FN) > 0:
            sens = TP / (TP + FN)
            sens_values.append(sens)

        if (TN + FP) > 0:
            spec = TN / (TN + FP)
            spec_values.append(spec)

    sens_ci_lower = np.percentile(sens_values, 2.5) if len(sens_values) > 0 else np.nan
    sens_ci_upper = np.percentile(sens_values, 97.5) if len(sens_values) > 0 else np.nan
    spec_ci_lower = np.percentile(spec_values, 2.5) if len(spec_values) > 0 else np.nan
    spec_ci_upper = np.percentile(spec_values, 97.5) if len(spec_values) > 0 else np.nan

    return sens_ci_lower, sens_ci_upper, spec_ci_lower, spec_ci_upper

def bootstrap_auc_ci(y_true, y_prob, n_bootstrap=1000, random_state=42):
    """
    Calculate AUC confidence interval using bootstrap resampling
    """
    original_auc = roc_auc_score(y_true, y_prob)

    np.random.seed(random_state)
    bootstrap_aucs = []
    n_samples = len(y_true)

    for i in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_prob_boot = y_prob[indices]

        if len(np.unique(y_true_boot)) < 2:
            continue

        auc = roc_auc_score(y_true_boot, y_prob_boot)
        bootstrap_aucs.append(auc)

    ci_lower = np.percentile(bootstrap_aucs, 2.5)
    ci_upper = np.percentile(bootstrap_aucs, 97.5)

    return original_auc, ci_lower, ci_upper

def bootstrap_prevalence_ci(y_true, n_bootstrap=1000, random_state=42):
    """
    Calculate prevalence confidence interval using bootstrap resampling
    """
    np.random.seed(random_state)
    bootstrap_prevalences = []
    n_samples = len(y_true)
    
    for i in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        prevalence_boot = np.mean(y_true_boot)
        bootstrap_prevalences.append(prevalence_boot)
    
    ci_lower = np.percentile(bootstrap_prevalences, 2.5)
    ci_upper = np.percentile(bootstrap_prevalences, 97.5)
    
    return ci_lower, ci_upper

def calculate_metrics_at_threshold(y_true, y_prob, threshold, auc_ci_str, prevalence_ci_str):
    pred = (y_prob >= threshold).astype(int)

    TP = int(np.sum((pred == 1) & (y_true == 1)))
    FP = int(np.sum((pred == 1) & (y_true == 0)))
    FN = int(np.sum((pred == 0) & (y_true == 1)))
    TN = int(np.sum((pred == 0) & (y_true == 0)))

    sensitivity = decimal_round(TP / (TP + FN)) if (TP + FN) > 0 else np.nan
    specificity = decimal_round(TN / (TN + FP)) if (TN + FP) > 0 else np.nan
    nnr = decimal_round((TP + FP) / TP) if TP > 0 else np.nan
    referral_rate = decimal_round(np.mean(pred))
    youden_index = decimal_round(sensitivity + specificity - 1) if not (np.isnan(sensitivity) or np.isnan(specificity)) else np.nan

    sens_ci_lower, sens_ci_upper, spec_ci_lower, spec_ci_upper = bootstrap_metrics_ci(y_true, y_prob, threshold)

    sens_ci_str = f"{sensitivity} [{decimal_round(sens_ci_lower)}-{decimal_round(sens_ci_upper)}]" if not np.isnan(sens_ci_lower) else f"{sensitivity} [NaN-NaN]"
    spec_ci_str = f"{specificity} [{decimal_round(spec_ci_lower)}-{decimal_round(spec_ci_upper)}]" if not np.isnan(spec_ci_lower) else f"{specificity} [NaN-NaN]"

    return {
        'Threshold': decimal_round(threshold),
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'NNR': nnr,
        'ReferralRate': referral_rate,
        'YoudenIndex': youden_index,
        'Sensitivity_CI': sens_ci_str,
        'Specificity_CI': spec_ci_str,
        'AUC_CI': auc_ci_str,
        'Prevalence_CI': prevalence_ci_str
    }

def analyze_subset(df, subset_name):
    if len(df) < 10:
        return None

    y_true = df['true_cog_impairment'].values
    y_prob = df['prob_cog_impairment'].values

    sample_size = len(df)
    normal_n = int(np.sum(y_true == 0))
    impaired_n = int(np.sum(y_true == 1))
    prevalence = decimal_round(np.mean(y_true))
    auc_val = decimal_round(roc_auc_score(y_true, y_prob))

    if subset_name == "Overall (4-country)":
        prev_ci_lower, prev_ci_upper = bootstrap_prevalence_ci(y_true)
        prevalence_ci_str = f"[{decimal_round(prev_ci_lower)}-{decimal_round(prev_ci_upper)}]"
    else:
        prevalence_ci_str = ""

    original_auc, auc_ci_lower, auc_ci_upper = bootstrap_auc_ci(y_true, y_prob)
    auc_ci_str = f"{decimal_round(original_auc)} [{decimal_round(auc_ci_lower)}-{decimal_round(auc_ci_upper)}]"

    thresholds = np.arange(0.01, 0.801, 0.001)
    results = []

    print(f"Processing {subset_name} with {len(thresholds)} thresholds...")

    for i, thresh in enumerate(thresholds):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(thresholds)} thresholds")

        metrics = calculate_metrics_at_threshold(y_true, y_prob, thresh, auc_ci_str, prevalence_ci_str)
        metrics.update({
            'Scale': subset_name,
            'SampleSize': sample_size,
            'AUC': auc_val,
            'Prevalence': prevalence
        })
        results.append(metrics)

    return pd.DataFrame(results)
