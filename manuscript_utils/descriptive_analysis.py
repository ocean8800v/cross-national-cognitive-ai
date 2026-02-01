# ========== Descriptive Analysis Functions ==========
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================================================
#  All Helper Functions for Descriptive Analysis
# ================================================

def load_and_prepare_data(data_path, files, country_id_map):
    """Load and prepare data from multiple countries"""
    dataframes = {}
    for country, file in files.items():
        df = pd.read_csv(
            os.path.join(data_path, file),
            usecols=['id', 'age', 'gender', 'raeducl', 'health_vs_mci_vs_dementia']
        ).copy()
        df['country'] = country
        df['country_id'] = country_id_map[country]
        dataframes[country] = df
    
    # Combine all datasets (common columns only)
    overall_df = pd.concat(
        [df[['id', 'age', 'gender', 'raeducl', 'health_vs_mci_vs_dementia', 'country', 'country_id']]
         for df in dataframes.values()],
        ignore_index=True
    )
    
    return dataframes, overall_df

def create_age_groups(df):
    """Create age groups for analysis"""
    df = df.copy()
    df['age_group'] = pd.cut(
        df['age'],
        bins=[65, 70, 75, 80, 85, 90, float('inf')],
        labels=['65-69', '70-74', '75-79', '80-84', '85-89', '90+'],
        right=False
    )
    return df

def calculate_status_percentages(dataframes, overall_df):
    """Calculate cognitive status percentages for all datasets"""
    # Overall status percentages
    overall_status_pct = (
        overall_df['health_vs_mci_vs_dementia']
        .value_counts(normalize=True)
        .sort_index() * 100
    )
    
    # Status percentages for each country
    status_summary = {}
    for country, df in dataframes.items():
        s = df['health_vs_mci_vs_dementia'].value_counts(normalize=True).sort_index() * 100
        status_summary[country] = s
    
    return overall_status_pct, status_summary

def print_summary_table(dataframes, overall_df, overall_status_pct, status_summary):
    """Print the summary statistics table"""
    print('Sample Size and Descriptive Statistics')
    print('=' * 125)

    # Updated header with CI risk
    print(f"{'Dataset':<15} {'N (respondents)':>15} {'Age (SD)':>12} {'Female (%)':>11} "
          f"{'<Upper Sec (%)':>14} {'Normal':>8} {'MCI risk':>10} {'Dementia risk':>15} {'CI risk':>10}")
    print('-' * 125)

    # Sample sizes
    sample_sizes = {country: len(df) for country, df in dataframes.items()}

    # Overall row calculation
    overall_n = len(overall_df)
    overall_age = f"{overall_df['age'].mean():.1f} ({overall_df['age'].std():.1f})"
    overall_female = f"{(overall_df['gender'] == 2).mean() * 100:.1f}"
    overall_edu = f"{(overall_df['raeducl'] == 1).mean() * 100:.1f}"
    overall_normal = f"{overall_status_pct[0]:.1f}"
    overall_mci = f"{overall_status_pct[1]:.1f}"
    overall_dementia = f"{overall_status_pct[2]:.1f}"
    overall_ci_risk = overall_status_pct[1] + overall_status_pct[2]

    # Print Overall row
    print(f"{'Overall':<15} {overall_n:>15,} {overall_age:>12} {overall_female:>11} "
          f"{overall_edu:>14} {overall_normal:>8} {overall_mci:>10} {overall_dementia:>15} {overall_ci_risk:>10.1f}")

    # Country-specific rows
    for country, df in dataframes.items():
        n = sample_sizes[country]
        age = f"{df['age'].mean():.1f} ({df['age'].std():.1f})"
        female = f"{(df['gender'] == 2).mean() * 100:.1f}"
        edu = f"{(df['raeducl'] == 1).mean() * 100:.1f}"

        # Calculate status percentages for each country
        status_data = status_summary[country]
        normal = f"{status_data[0]:.1f}"
        mci = f"{status_data[1]:.1f}"
        dementia = f"{status_data[2]:.1f}"
        ci_risk = status_data[1] + status_data[2]

        # Print each country's data
        print(f"{country:<15} {n:>15,} {age:>12} {female:>11} "
              f"{edu:>14} {normal:>8} {mci:>10} {dementia:>15} {ci_risk:>10.1f}")

    print('-' * 125)

def create_status_plot(overall_status_pct, status_summary, output_dir):
    """Create cognitive status distribution plot"""
    # Visualization Data Preparation
    status_labels = {0: 'Normal', 1: 'MCI risk', 2: 'Dementia risk'}
    status_plot_df = pd.DataFrame(status_summary).T
    status_plot_df.loc['Overall'] = overall_status_pct
    status_plot_df = status_plot_df.rename(columns=status_labels)[['Normal', 'MCI risk', 'Dementia risk']]

    # Reorder: Overall first
    status_plot_df = status_plot_df.loc[['Overall'] + [i for i in status_plot_df.index if i != 'Overall']]

    # Reorder columns for stacking order: bottom -> top
    status_plot_df = status_plot_df[['Dementia risk', 'MCI risk', 'Normal']]

    # Plot Settings
    plt.style.use('default')
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    # Colors
    colors = sns.color_palette("Set2", 3)

    # Plot stacked bar chart
    status_plot_df.plot(
        kind='bar', stacked=True, color=colors, ax=ax,
        width=0.6, alpha=0.9, edgecolor='white', linewidth=1.5
    )
    country_labels = ['Overall', 'HRS\n(US)', 'ELSA\n(England)', 'LASI\n(India)', 'MHAS\n(Mexico)', 'CHARLS\n(China)']
    ax.set_xticklabels(country_labels, rotation=0, fontsize=12)
    ax.tick_params(axis='x', pad=8)

    # Axis labels and title
    ax.tick_params(axis='y', labelsize=11)
    ax.set_xlabel('Country/Region', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('Prevalence (%)', fontsize=14, fontweight='bold', labelpad=10)
    #ax.set_title('Prevalence of Cognitive Status by Country', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 100)

    # Grid lines
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()

    # Legend outside plot
    ax.legend(
        handles[::-1], labels[::-1],
        title='Cognitive Status',
        fontsize=11,
        title_fontsize=12,
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        borderaxespad=0,
        frameon=False
    )

    # Percentage labels inside bars (hide if <5%)
    for container in ax.containers:
        ax.bar_label(
            container,
            labels = [
                "0%" if v == 0
                else "" if v < 5
                else f"{int(round(v))}%" if abs(v - round(v)) < 1e-6
                else f"{v:.1f}%"
                for v in container.datavalues
            ],
            label_type='center',
            fontsize=13,
            fontweight='bold',
            color='white'
        )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)

    # Layout adjustment
    fig.tight_layout()

    # Save Figure
    output_path = os.path.join(output_dir, 'cognitive_status_distribution.png')
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor='none',
        format='png',
        pad_inches=0.2
    )

    output_path_pdf = os.path.join(output_dir, 'cognitive_status_distribution.pdf')
    plt.savefig(
        output_path_pdf,
        bbox_inches="tight",
        facecolor="white",
        edgecolor='none',
        format='pdf',
        pad_inches=0.1 
    )
    plt.show()

def prepare_age_group_data(data):
    """Prepare age group data for plotting"""
    age_status_data = []
    age_groups = ['65-69', '70-74', '75-79', '80-84', '85-89', '90+']

    for age_group in age_groups:
        age_data = data[data['age_group'] == age_group]
        if len(age_data) > 0:
            status_pct = (age_data['health_vs_mci_vs_dementia']
                         .value_counts(normalize=True)
                         .sort_index() * 100)
            # Ensure all categories are present
            for status in [0, 1, 2]:
                if status not in status_pct:
                    status_pct[status] = 0
            age_status_data.append(status_pct.sort_index())
        else:
            # If no data for this age group, append zeros
            age_status_data.append(pd.Series([0, 0, 0], index=[0, 1, 2]))

    # Create DataFrame for plotting
    age_plot_df = pd.DataFrame(age_status_data, index=age_groups)
    age_plot_df = age_plot_df.rename(columns={0: 'Normal', 1: 'MCI risk', 2: 'Dementia risk'})

    # Reorder columns for stacking order: bottom -> top
    age_plot_df = age_plot_df[['Dementia risk', 'MCI risk', 'Normal']]

    return age_plot_df

def create_age_group_plots(overall_df, dataframes, output_dir):
    """Create age group analysis plots"""
    # Plot settings
    plt.style.use('default')
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=100)
    fig.suptitle('Prevalence of Cognitive Status by Age Group and Country',
                 fontsize=20, fontweight='bold', y=0.98)

    # Colors (same as original)
    colors = sns.color_palette("Set2", 3)

    # Datasets and titles
    datasets = [
        (overall_df, 'Overall'),
        (dataframes['HRS'], 'HRS'),
        (dataframes['ELSA'], 'ELSA'),
        (dataframes['LASI'], 'LASI'),
        (dataframes['MHAS'], 'MHAS'),
        (dataframes['CHARLS'], 'CHARLS')
    ]

    # Plot each subplot
    for idx, (data, title) in enumerate(datasets):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        # Prepare data
        age_plot_df = prepare_age_group_data(data)

        # Plot stacked bar chart
        age_plot_df.plot(
            kind='bar', stacked=True, color=colors, ax=ax,
            width=0.7, alpha=0.9, edgecolor='white', linewidth=1.2
        )

        # Subplot settings
        ax.tick_params(axis='x', labelsize=10, rotation=45, pad=5)
        ax.tick_params(axis='y', labelsize=10)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_ylim(0, 100)

        # Grid lines
        ax.grid(True, linestyle='--', alpha=0.3, axis='y')
        ax.set_axisbelow(True)

        # Remove individual legends
        ax.legend().set_visible(False)

        # Spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)

        # X-axis labels
        if row == 1:  # Bottom row
            if col == 1:  # Center column
                ax.set_xlabel('Age Group', fontsize=12, fontweight='bold', labelpad=8)
            else:
                ax.set_xlabel('')
        else:
            ax.set_xlabel('')

        # Y-axis labels
        if col == 0 and row == 0:  # Top-left only for centered positioning
            pass
        else:
            ax.set_ylabel('')

    # Add single legend for all subplots
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles[::-1], labels[::-1],
        title='Cognitive Status',
        fontsize=12,
        title_fontsize=13,
        bbox_to_anchor=(0.80, 0.5),
        loc='center left',
        borderaxespad=0,
        frameon=False
    )

    # Add centered Y-axis label
    fig.text(0.04, 0.5, 'Prevalence (%)', va='center', rotation='vertical',
             fontsize=14, fontweight='bold')

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.80, top=0.90, left=0.08)

    # Save combined figure
    output_path = os.path.join(output_dir, 'cognitive_status_age_group_combined.png')
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor='none',
        format='png',
        pad_inches=0.3
    )

    output_path_pdf = os.path.join(output_dir, 'cognitive_status_age_group_combined.pdf')
    plt.savefig(
        output_path_pdf,
        bbox_inches="tight",
        facecolor="white",
        edgecolor='none',
        format='pdf',
        pad_inches=0.2
    )
    plt.close()

def format_n_percent(n, total, decimals=1):
    """Format number and percentage as 'n (%)' with comma separator and space"""
    if total == 0:
        return "0 (0.0)"
    percent = (n / total) * 100
    return f"{n:,} ({percent:.{decimals}f})"

def calculate_group_stats(df, group_column, group_values=None):
    """Calculate statistics for a grouping variable"""
    if group_values is None:
        group_values = df[group_column].unique()
    
    total = len(df)
    stats = {}
    
    for value in group_values:
        count = len(df[df[group_column] == value])
        stats[value] = format_n_percent(count, total)
    
    return stats

def create_summary_table(dataframes, overall_df):
    """Create comprehensive summary table for export"""
    
    # Initialize the summary data
    summary_data = []
    
    # Define datasets
    datasets = [('Overall', overall_df)] + [(country, df) for country, df in dataframes.items()]
    
    # Define wave information
    wave_info = {
        'Overall': '',
        'HRS': 'Wave 13',
        'ELSA': 'Wave 7 or 8', 
        'LASI': 'Wave 1',
        'MHAS': 'Wave 4',
        'CHARLS': 'Wave 4'
    }
    
    # Define scale counts
    scale_counts = {
        'Overall': 29,
        'HRS': 14,
        'ELSA': 7,
        'LASI': 2,
        'MHAS': 4,
        'CHARLS': 2
    }
    
    # Create first row with country names only
    columns = ['Characteristics'] + [f'{name}, n (%)' for name, _ in datasets]
    
    # Create second row with wave information
    wave_row = [''] + [wave_info[name] if name != 'Overall' else '' for name, _ in datasets]
    summary_data.append(wave_row)
    
    # Sample Size
    sample_row = ['Total Respondents (Scales)']
    for name, df in datasets:
        sample_row.append(f"{len(df):,} ({scale_counts[name]})")
    summary_data.append(sample_row)
    
    # Age Groups header
    summary_data.append(['Age Groups'] + ['' for _ in datasets])  # Section header
    
    # Age Groups data
    age_groups = ['65-69', '70-74', '75-79', '80-84', '85-89', '90+']
    for age_group in age_groups:
        age_row = [age_group]
        for name, df in datasets:
            count = len(df[df['age_group'] == age_group])
            total = len(df)
            age_row.append(format_n_percent(count, total))
        summary_data.append(age_row)
    
    # Sex header
    summary_data.append(['Sex'] + ['' for _ in datasets])  # Section header
    
    # Gender data
    gender_labels = {1: 'Male', 2: 'Female'}
    for gender_code, gender_label in gender_labels.items():
        gender_row = [gender_label]
        for name, df in datasets:
            count = len(df[df['gender'] == gender_code])
            total = len(df)
            gender_row.append(format_n_percent(count, total))
        summary_data.append(gender_row)
    
    # Education header
    summary_data.append(['Harmonised Education Level'] + ['' for _ in datasets])  # Section header
    
    # Education Level data
    edu_labels = {1: 'Less than upper secondary', 2: 'Upper secondary and vocational', 3: 'Tertiary'}
    for edu_code, edu_label in edu_labels.items():
        edu_row = [edu_label]
        for name, df in datasets:
            count = len(df[df['raeducl'] == edu_code])
            total = len(df)
            edu_row.append(format_n_percent(count, total))
        summary_data.append(edu_row)
    
    # Cognitive Status header
    summary_data.append(['Cognitive Status'] + ['' for _ in datasets])  # Section header
    
    # Cognitive Status data - Normal first
    normal_row = ['Normal']
    for name, df in datasets:
        count = len(df[df['health_vs_mci_vs_dementia'] == 0])
        total = len(df)
        normal_row.append(format_n_percent(count, total))
    summary_data.append(normal_row)
    
    # Add Cognitive Impairment Risk (MCI + Dementia combined)
    ci_risk_row = ['Cognitive Impairment Risk (MCI + Dementia)']
    for name, df in datasets:
        count = len(df[df['health_vs_mci_vs_dementia'].isin([1, 2])])
        total = len(df)
        ci_risk_row.append(format_n_percent(count, total))
    summary_data.append(ci_risk_row)
    
    # Add individual MCI and Dementia rows
    status_labels = {1: 'MCI risk', 2: 'Dementia risk'}
    for status_code, status_label in status_labels.items():
        status_row = [status_label]
        for name, df in datasets:
            count = len(df[df['health_vs_mci_vs_dementia'] == status_code])
            total = len(df)
            status_row.append(format_n_percent(count, total))
        summary_data.append(status_row)
    
    # Create DataFrame
    summary_df = pd.DataFrame(summary_data, columns=columns)
    
    return summary_df

def export_summary_table(summary_df, output_dir):
    """Export summary table to CSV"""
    output_path = os.path.join(output_dir, 'descriptive_statistics_table.csv')
    summary_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"📁 Descriptive statistics table exported to CSV")
    print("\nTable preview:")
    print(summary_df.to_string(index=False))
