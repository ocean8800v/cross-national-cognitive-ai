# ========== Model Performance and Generalisation Analysis Functions ==========
"""
Model Comparison Utilities for Cross-national vs Single-country Analysis

This module provides functions to:
1. Load and compare cross-national vs single-country model results
2. Calculate performance improvement metrics
3. Calculate generalisation gap metrics
4. Pivot and sort results for analysis
"""

import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
import os

# ================================================
#  Helper Functions
# ================================================

def decimal_round(data, decimals=3):
    """Precise rounding for single values or iterable with ROUND_HALF_UP."""
    if hasattr(data, '__iter__') and not isinstance(data, str):
        decimal_values = [Decimal(str(val)) for val in data]
        result = sum(decimal_values) / Decimal(len(decimal_values))
    else:
        result = Decimal(str(data))
    
    quantize_format = '0.' + '0' * (decimals - 1) + '1'
    return float(result.quantize(Decimal(quantize_format), rounding=ROUND_HALF_UP))

# ================================================
#  Data Loading Functions
# ================================================

def load_cross_national_data(cross_national_path, groups):
    """Load cross-national model results for 4 countries"""
    data_frames = []
    for group in groups:
        file_path = os.path.join(cross_national_path, f'{group}_scale_4countries.csv')
        df = pd.read_csv(file_path)
        data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True)

def load_single_country_data(single_country_path, groups):
    """Load single-country model results for 4 countries"""
    data_frames = []
    for group in groups:
        file_path = os.path.join(single_country_path, f'{group}_scale_country_specific.csv')
        df = pd.read_csv(file_path)
        data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True)

def load_charls_data(cross_national_path, single_country_path):
    """Load CHARLS external validation data"""
    cross_national_file = os.path.join(cross_national_path, 'all_groups_CHARLS_scale_test.csv')
    single_country_file = os.path.join(single_country_path, 'all_groups_CHARLS_scale_country_specific.csv')
    
    cross_national_df = pd.read_csv(cross_national_file)
    single_country_df = pd.read_csv(single_country_file)
    
    return cross_national_df, single_country_df

def load_charls_average_data(single_country_path):
    """Load CHARLS averaged country-specific models data"""
    file_path = os.path.join(single_country_path, 'all_groups_CHARLS_scale_country_specific_average.csv')
    df = pd.read_csv(file_path)
    
    df = df[['country', 'scale', 'ROC_AUC_Macro', 'ROC_AUC_Normal', 
             'ROC_AUC_MCI', 'ROC_AUC_Dementia', 'sample_size', 'group']]
    
    df['country'] = 'CHARLS'
    
    return df
# ================================================
#  Performance Analysis Functions
# ================================================

def calculate_performance_metrics(cross_national_df, single_country_df, auc_columns, groups, subset_filter='test'):
    """Calculate performance improvement metrics"""
    results = []
    
    # Filter data by subset (test for performance comparison)
    if 'subset' in cross_national_df.columns:
        cross_national_test = cross_national_df[cross_national_df['subset'] == subset_filter]
    else:
        cross_national_test = cross_national_df.copy()
    
    if 'subset' in single_country_df.columns:
        single_country_test = single_country_df[single_country_df['subset'] == subset_filter]
    else:
        single_country_test = single_country_df.copy()
    
    # Calculate for each group, country, and metric
    for group in groups:
        for country in cross_national_test['country'].unique():
            cross_data = cross_national_test[(cross_national_test['group'] == group) & 
                                           (cross_national_test['country'] == country)]
            single_data = single_country_test[(single_country_test['group'] == group) & 
                                            (single_country_test['country'] == country)]
            
            if len(cross_data) == 0 or len(single_data) == 0:
                continue
            
            # Scale-level analysis
            for metric_name, column_name in auc_columns.items():
                absolute_improvements = []
                relative_improvements = []
                consistency_count = 0
                total_scales = 0
                
                for scale in cross_data['scale'].unique():
                    cross_scale = cross_data[cross_data['scale'] == scale]
                    single_scale = single_data[single_data['scale'] == scale]
                    
                    if len(cross_scale) == 0 or len(single_scale) == 0:
                        continue
                    
                    cross_auc = cross_scale[column_name].iloc[0]
                    single_auc = single_scale[column_name].iloc[0]
                    
                    absolute_imp = cross_auc - single_auc
                    relative_imp = (absolute_imp / single_auc) * 100
                    
                    absolute_improvements.append(absolute_imp)
                    relative_improvements.append(relative_imp)
                    
                    if absolute_imp > 0:
                        consistency_count += 1
                    total_scales += 1
                    
                    # Add scale-level result
                    results.append({
                        'Group': group,
                        'Country': country,
                        'Analysis_Level': 'Scale-level',
                        'Metric': metric_name,
                        'Absolute_Improvement': decimal_round(absolute_imp),
                        'Relative_Improvement_Pct': round(relative_imp, 1),  
                        'Improvement_Consistency_Pct': round((absolute_imp > 0) * 100, 1)
                    })
                
                # Within-country analysis
                if absolute_improvements:
                    avg_absolute = decimal_round(absolute_improvements)
                    avg_relative = round(sum(relative_improvements) / len(relative_improvements), 1) 
                    consistency_pct = round((consistency_count / total_scales) * 100, 1) 
                    
                    results.append({
                        'Group': group,
                        'Country': country,
                        'Analysis_Level': 'Within-country',
                        'Metric': metric_name,
                        'Absolute_Improvement': avg_absolute,
                        'Relative_Improvement_Pct': avg_relative,
                        'Improvement_Consistency_Pct': consistency_pct
                    })
    
    # Cross-country analysis - Scale-level averaging across all countries
    df_results = pd.DataFrame(results)
    scale_level_results = df_results[df_results['Analysis_Level'] == 'Scale-level']
    
    for group in groups:
        for metric_name in auc_columns.keys():
            subset_data = scale_level_results[(scale_level_results['Group'] == group) & 
                                            (scale_level_results['Metric'] == metric_name)]
            
            if len(subset_data) > 0:
                avg_absolute = decimal_round(subset_data['Absolute_Improvement'].tolist())
                
                relative_values = [float(x) for x in subset_data['Relative_Improvement_Pct']]      
                consistency_values = [float(x) for x in subset_data['Improvement_Consistency_Pct']] 
                
                avg_relative = round(sum(relative_values) / len(relative_values), 1)     
                avg_consistency = round(sum(consistency_values) / len(consistency_values), 1)  
                
                results.append({
                    'Group': group,
                    'Country': 'Overall',
                    'Analysis_Level': 'Cross-country',
                    'Metric': metric_name,
                    'Absolute_Improvement': avg_absolute,
                    'Relative_Improvement_Pct': avg_relative,
                    'Improvement_Consistency_Pct': avg_consistency
                })
    
    return pd.DataFrame(results)

def calculate_generalisation_metrics(data_df, model_name, auc_columns, groups):
    """Calculate generalisation gap metrics (validation vs test)"""
    results = []
    
    # Ensure we have both validation and test data
    if 'subset' not in data_df.columns:
        return pd.DataFrame(results)
    
    val_data = data_df[data_df['subset'] == 'val']
    test_data = data_df[data_df['subset'] == 'test']
    
    if len(val_data) == 0 or len(test_data) == 0:
        return pd.DataFrame(results)
    
    # Calculate for each group, country, and metric
    for group in groups:
        for country in data_df['country'].unique():
            val_country = val_data[(val_data['group'] == group) & (val_data['country'] == country)]
            test_country = test_data[(test_data['group'] == group) & (test_data['country'] == country)]
            
            if len(val_country) == 0 or len(test_country) == 0:
                continue
            
            # Scale-level analysis
            for metric_name, column_name in auc_columns.items():
                gaps = []
                
                for scale in val_country['scale'].unique():
                    val_scale = val_country[val_country['scale'] == scale]
                    test_scale = test_country[test_country['scale'] == scale]
                    
                    if len(val_scale) == 0 or len(test_scale) == 0:
                        continue
                    
                    val_auc = val_scale[column_name].iloc[0]
                    test_auc = test_scale[column_name].iloc[0]
                    gap = val_auc - test_auc
                    gaps.append(gap)
                    
                    # Add scale-level result
                    results.append({
                        'Model': model_name,
                        'Group': group,
                        'Country': country,
                        'Analysis_Level': 'Scale-level',
                        'Metric': metric_name,
                        'Generalisation_Gap': decimal_round(gap)
                    })
                
                # Within-country analysis
                if gaps:
                    avg_gap = decimal_round(gaps)
                    sd_gap = decimal_round(np.std(gaps, ddof=1))
                    
                    results.append({
                        'Model': model_name,
                        'Group': group,
                        'Country': country,
                        'Analysis_Level': 'Within-country',
                        'Metric': metric_name,
                        'Generalisation_Gap_Mean': avg_gap, 
                        'Generalisation_Gap_SD': sd_gap
                    })
    
    # Cross-country analysis - Scale-level averaging across all countries
    df_results = pd.DataFrame(results)
    scale_level_results = df_results[df_results['Analysis_Level'] == 'Scale-level']
    
    for group in groups:
        for metric_name in auc_columns.keys():
            subset_data = scale_level_results[(scale_level_results['Group'] == group) & 
                                            (scale_level_results['Metric'] == metric_name)]
            
            if len(subset_data) > 0:
                gaps = subset_data['Generalisation_Gap'].tolist()
                avg_gap = decimal_round(gaps)
                sd_gap = decimal_round(np.std(gaps, ddof=1))
                
                results.append({
                    'Model': model_name,
                    'Group': group,
                    'Country': 'Overall',
                    'Analysis_Level': 'Cross-country',
                    'Metric': metric_name,
                    'Generalisation_Gap_Mean': avg_gap, 
                    'Generalisation_Gap_SD': sd_gap 
                })
    
    return pd.DataFrame(results)

# ================================================
#  Data Processing Functions
# ================================================

def pivot_dataframe(df, index_cols, metric_col, value_cols):
    """
    Pivot dataframe from long to wide format.
    
    Parameters:
        df (pd.DataFrame): Original dataframe.
        index_cols (list): Columns to keep as index.
        metric_col (str): Column with metric names to pivot.
        value_cols (list): Columns containing values to pivot.
        
    Returns:
        pd.DataFrame: Pivoted wide-format dataframe.
    """
    df_pivot = df.pivot_table(index=index_cols, columns=metric_col, values=value_cols, aggfunc='first')
    df_pivot.columns = ['{}_{}'.format(metric, stat) for stat, metric in df_pivot.columns]
    df_pivot.reset_index(inplace=True)
    return df_pivot

def sort_dataframe(df):
    """Sort dataframe with custom order"""
    country_order = ['HRS', 'ELSA', 'LASI', 'MHAS', 'CHARLS', 'Overall']
    group_order = ['Group1', 'Group2', 'Group3']
    analysis_order = ['Within-country', 'Cross-country']
    
    # Create categorical columns for proper sorting
    df['Country_cat'] = pd.Categorical(df['Country'], categories=country_order, ordered=True)
    df['Group_cat'] = pd.Categorical(df['Group'], categories=group_order, ordered=True)
    df['Analysis_Level_cat'] = pd.Categorical(df['Analysis_Level'], categories=analysis_order, ordered=True)
    
    # Check if Model column exists (for Generalisation CSV)
    if 'Model' in df.columns:
        model_order = ['Cross-national', 'Single-country']
        df['Model_cat'] = pd.Categorical(df['Model'], categories=model_order, ordered=True)
        # Sort by Model, Country, Group, Analysis_Level
        df_sorted = df.sort_values(['Model_cat', 'Country_cat', 'Group_cat', 'Analysis_Level_cat'])
        df_sorted = df_sorted.drop(['Model_cat', 'Country_cat', 'Group_cat', 'Analysis_Level_cat'], axis=1)
    else:
        # Sort by Country, Group, Analysis_Level (for Performance CSV)
        df_sorted = df.sort_values(['Country_cat', 'Group_cat', 'Analysis_Level_cat'])
        df_sorted = df_sorted.drop(['Country_cat', 'Group_cat', 'Analysis_Level_cat'], axis=1)
    
    return df_sorted.reset_index(drop=True)

# ================================================
#  Convenience Functions
# ================================================

def process_model_comparison(data_base_path, output_path, auc_columns, groups):
    """
    Complete pipeline for model comparison analysis
    
    Parameters:
        data_base_path (str): Base path to model performance data
        output_path (str): Path to save results
        auc_columns (dict): Mapping of metric names to column names
        groups (list): List of group names
        
    Returns:
        tuple: (performance_4countries, performance_charls, generalisation_all)
    """
    # Setup paths
    cross_national_path = os.path.join(data_base_path, 'unified_model')
    single_country_path = os.path.join(data_base_path, 'country_specific')
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load data
    cross_national_4countries = load_cross_national_data(cross_national_path, groups)
    single_country_4countries = load_single_country_data(single_country_path, groups)
    charls_cross, charls_single = load_charls_data(cross_national_path, single_country_path)
    
    # Calculate performance metrics
    performance_4countries = calculate_performance_metrics(
        cross_national_4countries, single_country_4countries, auc_columns, groups
    )
    performance_charls = calculate_performance_metrics(charls_cross, charls_single, auc_columns, groups)
    
    # Calculate generalisation metrics
    gen_cross_4countries = calculate_generalisation_metrics(
        cross_national_4countries, 'Cross-national', auc_columns, groups
    )
    gen_cross_charls = calculate_generalisation_metrics(charls_cross, 'Cross-national', auc_columns, groups)
    gen_single_4countries = calculate_generalisation_metrics(
        single_country_4countries, 'Single-country', auc_columns, groups
    )
    gen_single_charls = calculate_generalisation_metrics(charls_single, 'Single-country', auc_columns, groups)
    
    # Combine generalisation results
    generalisation_all = pd.concat([
        gen_cross_4countries, gen_cross_charls,
        gen_single_4countries, gen_single_charls
    ], ignore_index=True)
    
    return performance_4countries, performance_charls, generalisation_all