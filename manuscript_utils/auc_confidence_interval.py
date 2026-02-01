# ========== AUC Confidence Intervals ==========

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
from decimal import Decimal, ROUND_HALF_UP

def decimal_round(data, decimals=3):
    """Precise rounding for single values or iterable (mean) with ROUND_HALF_UP."""
    if hasattr(data, '__iter__') and not isinstance(data, str):
        decimal_values = [Decimal(str(val)) for val in data]
        result = sum(decimal_values) / Decimal(len(decimal_values))
    else:
        result = Decimal(str(data))
    quantize_format = '0.' + '0' * (decimals - 1) + '1'
    return float(result.quantize(Decimal(quantize_format), rounding=ROUND_HALF_UP))

def bootstrap_auc_ci(y_true, y_pred_proba, n_bootstrap=1000, random_state=42):
    """
    Calculate AUC confidence interval using bootstrap resampling

    Returns:
        original_auc, ci_lower, ci_upper
    """
    # Calculate original AUC first
    original_auc = roc_auc_score(y_true, y_pred_proba)
    
    np.random.seed(random_state)
    bootstrap_aucs = []
    n_samples = len(y_true)
    
    for i in range(n_bootstrap):
        # Bootstrap resampling
        indices = resample(range(n_samples), n_samples=n_samples, replace=True, random_state=i)
        
        # Get bootstrap sample
        y_true_boot = y_true.iloc[indices]
        y_pred_boot = y_pred_proba.iloc[indices]
        
        # Skip if no variation in labels
        if len(np.unique(y_true_boot)) < 2:
            continue
            
        # Calculate AUC for this bootstrap sample
        auc = roc_auc_score(y_true_boot, y_pred_boot)
        bootstrap_aucs.append(auc)
    
    # Calculate confidence interval
    ci_lower = np.percentile(bootstrap_aucs, 2.5)
    ci_upper = np.percentile(bootstrap_aucs, 97.5)
    
    return original_auc, ci_lower, ci_upper

def process_prediction_file(file_path, group_name):
    """
    Process a single prediction file and calculate AUC CI for each country-scale combination
    
    Args:
        file_path: Path to the prediction CSV file
        group_name: Group name (Group1, Group2, or Group3)
    
    Returns:
        DataFrame with AUC confidence intervals
    """
    
    # Load prediction file
    df = pd.read_csv(file_path)
    
    # Required columns
    required_cols = ['country', 'scale', 'health_vs_mci_vs_dementia', f'{group_name}_prob_Normal']
    
    # Check if all required columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return pd.DataFrame()
    
    # Create binary labels: 1=Normal, 0=Cognitive Impairment (MCI+Dementia)
    df['normal_label'] = (df['health_vs_mci_vs_dementia'] == 0).astype(int)
    
    # Results storage
    results = []
    
    # Process each country
    countries = sorted(df['country'].unique())
    
    for country in countries:
        df_country = df[df['country'] == country]
        
        # Process each scale within the country
        scales = sorted(df_country['scale'].unique())
        
        for scale in scales:
            df_scale = df_country[df_country['scale'] == scale].copy()
            
            # Skip if insufficient data or no variation in labels
            if len(df_scale) < 50 or len(df_scale['normal_label'].unique()) < 2:
                continue
            
            # Get labels and predictions
            y_true = df_scale['normal_label']
            y_pred_proba = df_scale[f'{group_name}_prob_Normal']
            
            # Calculate AUC with confidence interval
            try:
                original_auc, ci_lower, ci_upper = bootstrap_auc_ci(y_true, y_pred_proba)
                
                # Use precise rounding
                auc_rounded = decimal_round(original_auc, 3)
                ci_lower_rounded = decimal_round(ci_lower, 3)
                ci_upper_rounded = decimal_round(ci_upper, 3)
                
                # Format result
                auc_ci_str = f"{auc_rounded} [{ci_lower_rounded}, {ci_upper_rounded}]"
                
                results.append({
                    'group': group_name,
                    'country': country,
                    'scale': scale,
                    'Cog_Imp_Risk_AUC_CI': auc_ci_str,
                    'sample_size': len(df_scale),
                    'normal_count': sum(df_scale['normal_label'] == 1),
                    'impaired_count': sum(df_scale['normal_label'] == 0)
                })
                
            except Exception as e:
                continue
    
    return pd.DataFrame(results)