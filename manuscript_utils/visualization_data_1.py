# ========== Visualization Data Preparation Utilities ==========

import pandas as pd
import os

def format_auc_result(row, metric):
    """Format AUC result as mean (std) [min, max] format"""
    mean = row[f'ROC_AUC_{metric}_mean']
    std = row[f'ROC_AUC_{metric}_std']
    min_val = row[f'ROC_AUC_{metric}_min']
    max_val = row[f'ROC_AUC_{metric}_max']
    return f"{mean:.3f} ({std:.3f}) [{min_val:.3f}, {max_val:.3f}]"

def format_gap_result(row, metric):
    """Format generalisation gap result as mean (std) format"""
    # Note: Generalisation analysis files use 'Cognitive_impairment' instead of 'Normal'
    metric_name = 'Cognitive_impairment' if metric == 'Normal' else metric
    mean = row[f'{metric_name}_Generalisation_Gap_Mean']
    std = row[f'{metric_name}_Generalisation_Gap_SD']
    return f"{mean:.3f} ({std:.3f})"

def format_improvement_result(row, metric, improvement_type):
    """Format improvement result for individual metric"""
    # Same mapping issue for improvement files
    metric_name = 'Cognitive_impairment' if metric == 'Normal' else metric
    value = row[f'{metric_name}_{improvement_type}']
    return f"{value:.3f}"

def format_combined_improvement(row, metric):
    """Format combined improvement as abs (rel%, con%) format"""
    metric_name = 'Cognitive_impairment' if metric == 'Normal' else metric
    abs_val = row[f'{metric_name}_Absolute_Improvement']
    rel_val = row[f'{metric_name}_Relative_Improvement_Pct']
    con_val = row[f'{metric_name}_Improvement_Consistency_Pct']
    return f"{abs_val:.3f} ({rel_val:.1f}%, {con_val:.1f}%)"

def create_auc_performance_csv(cross_country, within_country, charls_within_country):
    """Create AUC Performance CSV with transposed format"""
    
    groups = ['Group1', 'Group2', 'Group3']
    countries = ['HRS', 'ELSA', 'LASI', 'MHAS']
    metrics = ['Macro', 'Normal', 'MCI', 'Dementia']
    
    # Create column names
    columns = ['Group_Country'] + [f'{metric.replace("Normal", "Cognitive impairment")} Mean (SD) [Min, Max]' for metric in metrics]
    
    rows = []
    
    for group in groups:
        # Overall (4-country) row - use cross-country test data
        cross_test = cross_country[(cross_country['group'] == group) & 
                                 (cross_country['subset'] == 'test')]
        
        if not cross_test.empty:
            row_data = [f'{group}_Overall (4-country)']
            for metric in metrics:
                row_data.append(format_auc_result(cross_test.iloc[0], metric))
            rows.append(row_data)
        
        # Country rows - use within-country test data
        within_test = within_country[(within_country['group'] == group) & 
                                   (within_country['subset'] == 'test')]
        
        for country in countries:
            country_data = within_test[within_test['country'] == country]
            if not country_data.empty:
                row_data = [f'{group}_{country}']
                for metric in metrics:
                    row_data.append(format_auc_result(country_data.iloc[0], metric))
                rows.append(row_data)
        
        # CHARLS (external) row
        charls_group = charls_within_country[charls_within_country['group'] == group]
        if not charls_group.empty:
            row_data = [f'{group}_CHARLS (external)']
            for metric in metrics:
                row_data.append(format_auc_result(charls_group.iloc[0], metric))
            rows.append(row_data)
    
    return pd.DataFrame(rows, columns=columns)

def create_generalisation_gap_csv(gen_gap):
    """Create Generalisation Gap CSV with transposed format"""
    
    groups = ['Group1', 'Group2', 'Group3']
    countries = ['HRS', 'ELSA', 'LASI', 'MHAS']
    metrics = ['Macro', 'Normal', 'MCI', 'Dementia']
    
    # Create column names
    columns = ['Group_Country'] + [f'{metric.replace("Normal", "Cognitive impairment")} Mean (SD)' for metric in metrics]
    
    rows = []
    
    for group in groups:
        # Overall (4-country) row - use cross-country data
        cross_gap = gen_gap[(gen_gap['Model'] == 'Cross-national') &
                          (gen_gap['Group'] == group) &
                          (gen_gap['Analysis_Level'] == 'Cross-country') &
                          (gen_gap['Country'] == 'Overall')]
        
        if not cross_gap.empty:
            row_data = [f'{group}_Overall (4-country)']
            for metric in metrics:
                row_data.append(format_gap_result(cross_gap.iloc[0], metric))
            rows.append(row_data)
        
        # Country rows - use within-country data
        within_gap = gen_gap[(gen_gap['Model'] == 'Cross-national') &
                           (gen_gap['Group'] == group) &
                           (gen_gap['Analysis_Level'] == 'Within-country')]
        
        for country in countries:
            country_gap = within_gap[within_gap['Country'] == country]
            if not country_gap.empty:
                row_data = [f'{group}_{country}']
                for metric in metrics:
                    row_data.append(format_gap_result(country_gap.iloc[0], metric))
                rows.append(row_data)
        
        # CHARLS (external) row - likely empty, but include for consistency
        row_data = [f'{group}_CHARLS (external)'] + [''] * len(metrics)
        rows.append(row_data)
    
    return pd.DataFrame(rows, columns=columns)

def create_auc_improvement_csv(performance, charls_performance, charls_avg_performance):  # ✅
    """Create AUC Improvement CSV with transposed format and combined metrics"""
    
    groups = ['Group1', 'Group2', 'Group3']
    countries = ['HRS', 'ELSA', 'LASI', 'MHAS']
    metrics = ['Macro', 'Normal', 'MCI', 'Dementia']
    
    # Create column names with combined format
    columns = ['Group_Country'] + [f'{metric.replace("Normal", "Cognitive impairment")} Absolute Improvement (Relative %, Consistency %)' for metric in metrics]
    
    rows = []
    
    for group in groups:
        # Overall (4-country) row - use cross-country data
        cross_improve = performance[(performance['Group'] == group) & 
                                  (performance['Country'] == 'Overall') &
                                  (performance['Analysis_Level'] == 'Cross-country')]
        
        if not cross_improve.empty:
            row_data = [f'{group}_Overall (4-country)']
            for metric in metrics:
                row_data.append(format_combined_improvement(cross_improve.iloc[0], metric))
            rows.append(row_data)
        
        # Country rows - use within-country data
        within_improve = performance[(performance['Group'] == group) & 
                                   (performance['Analysis_Level'] == 'Within-country')]
        
        for country in countries:
            country_improve = within_improve[within_improve['Country'] == country]
            if not country_improve.empty:
                row_data = [f'{group}_{country}']
                for metric in metrics:
                    row_data.append(format_combined_improvement(country_improve.iloc[0], metric))
                rows.append(row_data)
        
        # CHARLS (external) row - use CHARLS performance data
        charls_improve = charls_performance[(charls_performance['Group'] == group) & 
                                  (charls_performance['Country'] == 'CHARLS') &
                                  (charls_performance['Analysis_Level'] == 'Within-country')]

        if not charls_improve.empty:
            row_data = [f'{group}_CHARLS (external)']
            for metric in metrics:
                row_data.append(format_combined_improvement(charls_improve.iloc[0], metric))
            rows.append(row_data)
        else:
            row_data = [f'{group}_CHARLS (external)'] + [''] * len(metrics)
            rows.append(row_data)

        charls_avg_improve = charls_avg_performance[(charls_avg_performance['Group'] == group) & 
                                                    (charls_avg_performance['Country'] == 'CHARLS') &
                                                    (charls_avg_performance['Analysis_Level'] == 'Within-country')]

        if not charls_avg_improve.empty:
            row_data = [f'{group}_CHARLS_Average (external)']
            for metric in metrics:
                row_data.append(format_combined_improvement(charls_avg_improve.iloc[0], metric))
            rows.append(row_data)
        else:
            row_data = [f'{group}_CHARLS_Average (external)'] + [''] * len(metrics)  # ✅
            rows.append(row_data)
    
    return pd.DataFrame(rows, columns=columns)

def visualization_prep_1(base_path='/content/drive/MyDrive/manuscript/4_results/1_Defualt'):
    """
    Prepare visualization data by creating three separate CSV files:
    1. AUC_Performance.csv - Transposed AUC performance data
    2. Generalisation_Gap.csv - Transposed generalisation gap data  
    3. AUC_Improvement.csv - Transposed improvement data with combined metrics
    
    Args:
        base_path (str): Base path to the data files
        
    Returns:
        dict: Dictionary containing the three DataFrames, or None if failed
    """
    
    # Data file paths
    paths = {
        'cross_country': f'{base_path}/Defualt_Model_Performance/unified_model/all_groups_cross_country_4countries.csv',
        'within_country': f'{base_path}/Defualt_Model_Performance/unified_model/all_groups_within_country_4countries.csv',
        'charls_within_country': f'{base_path}/Defualt_Model_Performance/unified_model/all_groups_CHARLS_within_country_test.csv',
        'generalisation_gap': f'{base_path}/Model_Comparison/AUC_Generalisation_Analysis_Wide.csv',
        'performance_4countries': f'{base_path}/Model_Comparison/AUC_Performance_Comparison_4Countries_Wide.csv',
        'performance_charls': f'{base_path}/Model_Comparison/AUC_Performance_Comparison_CHARLS_Wide.csv',
        'performance_charls_avg': f'{base_path}/Model_Comparison/AUC_Performance_Comparison_CHARLS_Average_Wide.csv'  # 新增
    }
    
    # Load data files
    try:
        cross_country = pd.read_csv(paths['cross_country'])
        within_country = pd.read_csv(paths['within_country'])
        charls_within_country = pd.read_csv(paths['charls_within_country'])
        gen_gap = pd.read_csv(paths['generalisation_gap'])
        performance = pd.read_csv(paths['performance_4countries'])
        charls_performance = pd.read_csv(paths['performance_charls'])
        charls_avg_performance = pd.read_csv(paths['performance_charls_avg'])
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return None
    
    # Create output folder
    output_folder = os.path.join(base_path, 'Visualization_Data')
    os.makedirs(output_folder, exist_ok=True)
    
    # Create three CSV files
    try:
        # 1. AUC Performance
        auc_performance_df = create_auc_performance_csv(cross_country, within_country, charls_within_country)
        auc_file = os.path.join(output_folder, 'AUC_Performance.csv')
        auc_performance_df.to_csv(auc_file, index=False)
        
        # 2. Generalisation Gap
        gap_df = create_generalisation_gap_csv(gen_gap)
        gap_file = os.path.join(output_folder, 'Generalisation_Gap.csv')
        gap_df.to_csv(gap_file, index=False)
        
        # 3. AUC Improvement
        improvement_df = create_auc_improvement_csv(performance, charls_performance, charls_avg_performance)
        improvement_file = os.path.join(output_folder, 'AUC_Improvement.csv')
        improvement_df.to_csv(improvement_file, index=False)
        
        print("📁 AUC_Performance.csv")
        print("📁 Generalisation_Gap.csv") 
        print("📁 AUC_Improvement.csv")
        
        return {
            'auc_performance': auc_performance_df,
            'generalisation_gap': gap_df,
            'auc_improvement': improvement_df
        }
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return None