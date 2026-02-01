# ========== Model Evaluation - 1 Utilities ==========
import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from decimal import Decimal, ROUND_HALF_UP
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

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

def draw_calibration_curve(y_true_binary, p_uncal, p_cal, title, save_path, n_bins=10):
    """Draw 'uncalibrated vs calibrated' reliability plot and save."""
    yb = np.asarray(y_true_binary).astype(int)
    p_uncal = np.clip(np.asarray(p_uncal), 1e-7, 1 - 1e-7)
    p_cal = np.clip(np.asarray(p_cal), 1e-7, 1 - 1e-7)

    frac_pos_uncal, mean_pred_uncal = calibration_curve(yb, p_uncal, n_bins=n_bins)
    frac_pos_cal,   mean_pred_cal   = calibration_curve(yb, p_cal,   n_bins=n_bins)

    plt.figure(figsize=(7, 6))
    plt.plot(mean_pred_uncal, frac_pos_uncal, marker='o', label='Uncalibrated')
    plt.plot(mean_pred_cal,   frac_pos_cal,   marker='o', label='Calibrated')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def calculate_performance_statistics(dataset_scale_df, group_name, dataset_name):
    """
    Calculate within-country and cross-country performance statistics from scale-level results
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

# ---------- Calibration helpers ----------
def platt_fit(p_pos, y_binary):
   platt = LogisticRegression()
   platt.fit(np.asarray(p_pos).reshape(-1, 1), np.asarray(y_binary).astype(int))
   return platt

def platt_predict(calibrator, p_pos):
   return calibrator.predict_proba(np.asarray(p_pos).reshape(-1, 1))[:, 1]

def run_binary_calibration_and_plots_platt(
   base_model,                   # fitted 3-class model (TabPFN)
   df_val, df_test, charls_test, # dataframes
   feature_cols,
   target_col,                   # 0=Normal,1=MCI,2=Dementia
   country_map4,                 # {1:'HRS',2:'ELSA',3:'LASI',4:'MHAS'}
   out_root_models,              # Calibrated_Fine-tuned_Saved_Models
   out_root_eval,                # Calibrated_Fine-tuned_Model_Evaluated_Set
   out_root_curves,              # Calibrated_Fine-tuned_Calibration_Curves
   group_name
):
   """
   Build CI=MCI+Dementia, fit calibrator on 4-country val, evaluate on test+CHARLS,
   save calibrator pkl, calibrated CSVs (2-class only), and 3 calibration curves.
   """
   # --- directories ---
   model_dir = out_root_models
   eval_dir  = out_root_eval
   curve_dir = out_root_curves
   for d in [model_dir, eval_dir, curve_dir]:
       ensure_dir(d)

   # --- Fit calibrator on 4-country val ---
   val_probs = predict_proba_in_two_batches(base_model, df_val[feature_cols])
   val_ci_prob = np.clip(val_probs[:, 1] + val_probs[:, 2], 1e-7, 1 - 1e-7)
   val_y_bin = (df_val[target_col].values > 0).astype(int)

   calibrator = platt_fit(val_ci_prob, val_y_bin)

   # Save calibrator
   cal_name = f"Calibrated_finetuned_binary_CI_{group_name}.pkl"
   with open(os.path.join(model_dir, cal_name), 'wb') as f:
       pickle.dump(calibrator, f)

   # Helper: get CI probs for a dataframe
   def ci_probs(df):
       probs = predict_proba_in_two_batches(base_model, df[feature_cols])
       p_ci  = np.clip(probs[:, 1] + probs[:, 2], 1e-7, 1 - 1e-7)
       p_ci_cal = platt_predict(calibrator, p_ci)
       y_bin = (df[target_col].values > 0).astype(int)
       return y_bin, p_ci, p_ci_cal

   # --- test ---
   y_test_bin, p_test_uncal, p_test_cal = ci_probs(df_test)
   test_out = df_test.copy()
   test_out["prob_CI_uncalibrated"] = p_test_uncal
   test_out["prob_CI_calibrated"]   = p_test_cal
   test_csv_name = f"{group_name}_calibrated_4country_test_predictions.csv"
   test_out.to_csv(os.path.join(eval_dir, test_csv_name), index=False)

   # --- CHARLS ---
   y_charls_bin, p_charls_uncal, p_charls_cal = ci_probs(charls_test)
   charls_out = charls_test.copy()
   charls_out["prob_CI_uncalibrated"] = p_charls_uncal
   charls_out["prob_CI_calibrated"]   = p_charls_cal
   charls_csv_name = f"{group_name}_calibrated_CHARLS_test_predictions.csv"
   charls_out.to_csv(os.path.join(eval_dir, charls_csv_name), index=False)

   # --- Calibration curves ---
   # overall 5
   y_all = np.concatenate([y_test_bin, y_charls_bin])
   p_all_uncal = np.concatenate([p_test_uncal, p_charls_uncal])
   p_all_cal   = np.concatenate([p_test_cal,   p_charls_cal])
   draw_calibration_curve(
       y_all, p_all_uncal, p_all_cal,
       title=f'Calibration Curve - Overall (5 incl. CHARLS) - {group_name}',
       save_path=os.path.join(curve_dir, f"calibration_curve_overall_5countries_{group_name}.png")
   )
   print(f"✅ Calibration saved for {group_name} (model + evaluated sets + curves)")
