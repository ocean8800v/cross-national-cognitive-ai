import os

# ================================================
#  Shared Configuration for Manuscript Processing
# ================================================

# Base paths
BASE_PATH = "/content/drive/MyDrive/manuscript"
RAW_DATA_PATH = os.path.join(BASE_PATH, "1_raw_data")
STACKED_DATA_PATH = os.path.join(BASE_PATH, "2_stacked_data")
SPLIT_OUTPUT_PATH = os.path.join(BASE_PATH, "3_split_data")
SUPP_PATH = os.path.join(BASE_PATH, "0_supplementary_file")

# Create output directories
os.makedirs(STACKED_DATA_PATH, exist_ok=True)
os.makedirs(SPLIT_OUTPUT_PATH, exist_ok=True)
os.makedirs(SUPP_PATH, exist_ok=True)

# Country files for stacking
country_files = {
    'CHARLS': "CHARLS_wave_4.csv",
    'ELSA': "ELSA_wave_7_or_8.csv",
    'HRS': "HRS_wave_13.csv",
    'LASI': "LASI_wave_1.csv",
    'MHAS': "MHAS_wave_4.csv"
}

# All country codes (for stacking - 5 countries)
all_country_codes = {'HRS': 1, 'ELSA': 2, 'LASI': 3, 'MHAS': 4, 'CHARLS': 5}
all_country_code_to_name = {v: k for k, v in all_country_codes.items()}

# Four country codes (for splitting - exclude CHARLS)
four_country_codes = {'HRS': 1, 'ELSA': 2, 'LASI': 3, 'MHAS': 4}
four_country_code_to_name = {v: k for k, v in four_country_codes.items()}

# Wave information
country_wave_info = {
    'HRS': 'HRS_wave_13',
    'ELSA': 'ELSA_wave_7_or_8', 
    'LASI': 'LASI_wave_1',
    'MHAS': 'MHAS_wave_4'
}

# Seven LQR indicators based on questionnaire response behaviors
lqr_indicators = ['asinmiss', 'extreme', 'gnorm', 'logresvar', 'lz', 'mdistance', 'u3']

# Define country-specific base columns
country_base_columns = {
    'CHARLS': ['id', 'status', 'language', 'age', 'gender', 'raeducl','health_vs_mci_vs_dementia'],
    'ELSA': ['id', 'status', 'intent_email', 'freq_3m_intent_email', 'language', 'age', 'gender', 'raeducl', 'health_vs_mci_vs_dementia'],
    'HRS': ['id', 'status', 'r13email', 'r13drive', 'language', 'age', 'gender', 'raeducl', 'health_vs_mci_vs_dementia'],
    'LASI': ['id', 'status', 'language', 'age', 'gender', 'raeducl', 'health_vs_mci_vs_dementia'],
    'MHAS': ['id', 'status', 'language', 'age', 'gender', 'raeducl', 'health_vs_mci_vs_dementia']
}

# Data splitting configuration
STACKED_FILE = "stacked_four_country.csv"
CHARLS_FILE = "stacked_CHARLS.csv"
target = "health_vs_mci_vs_dementia"
scale_col = 'scale'
country_col = 'country'

# Random seeds for reproducible splitting
country_seeds = 100
country_val_seeds = 87
country_train_seeds = 755
test_size = 0.35