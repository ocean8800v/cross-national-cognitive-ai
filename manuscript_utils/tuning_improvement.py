# ========== Fine-tuning improvement calculation ==========
import pandas as pd
import numpy as np
import os
import re
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

def extract_auc_value(auc_ci_string):
    """Extract AUC value from string format '0.866 [0.838, 0.893]'"""
    # Extract the number before the first '['
    match = re.match(r'^([\d.]+)', str(auc_ci_string).strip())
    if match:
        return float(match.group(1))
    else:
        raise ValueError(f"Cannot extract AUC value from: {auc_ci_string}")

def load_and_process_data(file_path):
    """Load CSV file and extract AUC values from CI format"""
    df = pd.read_csv(file_path)
    # Extract AUC values from the CI column
    df['auc_value'] = df['Cog_Imp_Risk_AUC_CI'].apply(extract_auc_value)
    return df[['group', 'country', 'scale', 'auc_value']]

def calculate_improvements():
    """Calculate performance improvements between default and fine-tuned models"""
    
    # Define paths
    default_base_path = "/content/drive/MyDrive/manuscript/4_results/1_Defualt/Defualt_Model_Performance/unified_model"
    finetuned_base_path = "/content/drive/MyDrive/manuscript/4_results/2_Fine-tuned/Fine-tuned_Model_Performance"
    
    # Initialize results list
    results = []
    
    # Process each group
    for group_num in [1, 2, 3]:
        group_name = f"Group{group_num}"
        
        # Load default model data
        default_file = os.path.join(default_base_path, f"Group{group_num}_CogImpRisk_AUC_CI_5countries.csv")
        default_df = load_and_process_data(default_file)
        
        # Load fine-tuned model data
        finetuned_file = os.path.join(finetuned_base_path, f"Group{group_num}_CogImpRisk_AUC_CI_5countries_Fine-tuned.csv")
        finetuned_df = load_and_process_data(finetuned_file)
        
        # Merge datasets on group, country, and scale
        merged_df = pd.merge(default_df, finetuned_df, on=['group', 'country', 'scale'], 
                           suffixes=('_default', '_finetuned'))
        
        # Calculate absolute improvement
        merged_df['absolute_improvement'] = merged_df['auc_value_finetuned'] - merged_df['auc_value_default']
        
        # Calculate relative improvement (percentage)
        merged_df['relative_improvement'] = (merged_df['absolute_improvement'] / merged_df['auc_value_default']) * 100
        
        # Process each country within the group
        countries = merged_df['country'].unique()
        
        for country in countries:
            country_data = merged_df[merged_df['country'] == country]
            
            # Calculate metrics for this country
            n_scales = len(country_data)
            mean_absolute_improvement = decimal_round(country_data['absolute_improvement'])
            mean_relative_improvement = decimal_round(country_data['relative_improvement'], decimals=2)  # 2 decimals for percentage
            
            # Calculate consistency (percentage of scales that improved)
            improved_scales = (country_data['absolute_improvement'] > 0).sum()
            consistency_pct = decimal_round((improved_scales / n_scales) * 100, decimals=2)  # 2 decimals for percentage
            
            results.append({
                'group': group_name,
                'country': country,
                'scales': n_scales,
                'Mean_AUROC_Cognitive_Impairment': mean_absolute_improvement,
                'Relative_AUROC_Cognitive_Impairment': mean_relative_improvement,
                'Consistency_AUROC_Cognitive_Impairment': consistency_pct
            })
        
        # Calculate overall metrics for the group (4 training countries only, excluding CHARLS)
        training_countries = ['HRS', 'ELSA', 'LASI', 'MHAS']
        overall_data = merged_df[merged_df['country'].isin(training_countries)]
        
        overall_scales = len(overall_data)
        overall_mean_absolute = decimal_round(overall_data['absolute_improvement'])
        overall_mean_relative = decimal_round(overall_data['relative_improvement'], decimals=2)
        overall_improved = (overall_data['absolute_improvement'] > 0).sum()
        overall_consistency = decimal_round((overall_improved / overall_scales) * 100, decimals=2)
        
        results.append({
            'group': group_name,
            'country': 'overall',
            'scales': overall_scales,
            'Mean_AUROC_Cognitive_Impairment': overall_mean_absolute,
            'Relative_AUROC_Cognitive_Impairment': overall_mean_relative,
            'Consistency_AUROC_Cognitive_Impairment': overall_consistency
        })
    
    # Create final DataFrame
    results_df = pd.DataFrame(results)
    
    # Define the desired country order
    country_order = ['HRS', 'ELSA', 'LASI', 'MHAS', 'CHARLS', 'overall']
    results_df['country'] = pd.Categorical(results_df['country'], categories=country_order, ordered=True)
    
    # Sort by group and country order
    results_df = results_df.sort_values(['group', 'country']).reset_index(drop=True)
    
    # Rename columns for better readability
    results_df = results_df.rename(columns={
        'Mean_AUROC_Cognitive_Impairment': 'Mean_Improve_Cog_Imp',
        'Relative_AUROC_Cognitive_Impairment': 'Relative_Improv_Cog_Imp_Pct', 
        'Consistency_AUROC_Cognitive_Impairment': 'Consistency_Improv_Cog_Imp_Pct'
    })
    
    # Update country names for clarity (convert to string first to avoid categorical warning)
    results_df['country'] = results_df['country'].astype(str)
    results_df['country'] = results_df['country'].replace({
        'overall': 'Overall (4-country)',
        'CHARLS': 'External CHARLS'
    })
    
    return results_df