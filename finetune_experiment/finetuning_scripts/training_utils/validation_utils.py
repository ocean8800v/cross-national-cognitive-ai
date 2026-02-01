from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING
from typing import Literal
import torch
from finetuning_scripts.constant_utils import SupportedDevice, TaskType
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from finetuning_scripts.metric_utils.ag_metrics import Scorer
    from tabpfn.model.transformer import PerFeatureTransformer
    from tabpfn import TabPFNClassifier, TabPFNRegressor


def print_dataset_check(df, dataset_name, country_col, scale_col, target):
    print("\n=== 🚩 Final Consistency Check for Fine-tuning Dataset 🚩 ===")
    print(f"\n📊 [{dataset_name}] Sample Counts and Prevalence by Country (3-class):")
    for country in sorted(df[country_col].unique()):
        sub_df_country = df[df[country_col] == country]
        cnt = len(sub_df_country)
        rate_overall = (sub_df_country[target] > 0).mean()
        rate_normal = (sub_df_country[target] == 0).mean()
        rate_mci = (sub_df_country[target] == 1).mean()
        rate_dementia = (sub_df_country[target] == 2).mean()

        print(
            f"Country {country} N={cnt}, prevalence(MCI+Dementia)={rate_overall:.3f}, "
            f"Normal={rate_normal:.3f}, MCI={rate_mci:.3f}, Dementia={rate_dementia:.3f}"
        )

    print(f"\n🔍 [{dataset_name}] ID/Count Consistency Checks Across Scales:")
    for country in sorted(df[country_col].unique()):
        sub_df = df[df[country_col] == country]
        scales = sorted(sub_df[scale_col].unique())
        id_sets = [frozenset(sub_df[sub_df[scale_col] == s]['id'].astype(str)) for s in scales]
        scale_counts = [len(sub_df[sub_df[scale_col] == s]) for s in scales]

        ids_equal = all(id_sets[0] == s for s in id_sets)
        counts_equal = all(scale_counts[0] == c for c in scale_counts)

        id_check = "✅" if ids_equal else "❌"
        count_check = "✅" if counts_equal else "❌"

        print(f"Country {country}: ID consistency across scales: {id_check} | "
              f"Count consistency: {count_check} | Counts per scale: {scale_counts}")
        
def create_val_data(
    *,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    rng: np.random.RandomState,
    n_samples: int,
    is_classification: bool,
    scale_col: str = 'scale',
    country_col: str = 'country',
    test_size: float = 0.2,
    target: str = 'health_vs_mci_vs_dementia',
    task_type: Literal["binary", "multiclass"] = "multiclass"
):
    train_list, val_list = [], []

    df = X_train.copy()
    df[target] = y_train.values if isinstance(y_train, pd.Series) else y_train

    # 🔥 首先统一排序
    df = df.sort_values(by=[country_col, 'id', scale_col]).reset_index(drop=True)

    for country in sorted(df[country_col].unique()):
        sub_df_country = df[df[country_col] == country]

        unique_ids = sub_df_country['id'].unique()

        # 🔥 新增：每个ID对应的类别（用于stratify）
        id_labels = sub_df_country.groupby('id')[target].first()

        # 🔥 增加stratify=id_labels
        train_ids, val_ids = train_test_split(
            unique_ids,
            test_size=test_size,
            random_state=rng,
            stratify=id_labels  # ← 关键修改点
        )

        for scale_val in sorted(sub_df_country[scale_col].unique()):
            sub_df_scale = sub_df_country[sub_df_country[scale_col] == scale_val]

            train_list.append(sub_df_scale[sub_df_scale['id'].isin(train_ids)])
            val_list.append(sub_df_scale[sub_df_scale['id'].isin(val_ids)])

    train_df = pd.concat(train_list, ignore_index=True)
    val_df = pd.concat(val_list, ignore_index=True)

    print_dataset_check(train_df, "Fine-tuning Train Set", country_col, scale_col, target)
    print_dataset_check(val_df, "Validation Set", country_col, scale_col, target)

    X_train_out = train_df.drop(columns=[target, country_col, scale_col, "id"]).astype(np.float32)
    y_train_out = train_df[target].astype(np.int64)

    X_val_out = val_df.drop(columns=[target, country_col, scale_col, "id"]).astype(np.float32)
    y_val_out = val_df[target].astype(np.int64)

    if task_type == "binary":
        y_train_out = (y_train_out > 0).astype(np.int64)
        y_val_out = (y_val_out > 0).astype(np.int64)

        print("\n✅ [Binary classification] Data splitting successful!")
        print("Training set y-value counts:")
        print(y_train_out.value_counts())
        print("Training set y-value proportions:")
        print(y_train_out.value_counts(normalize=True))

        print("Validation set y-value counts:")
        print(y_val_out.value_counts())
        print("Validation set y-value proportions:")
        print(y_val_out.value_counts(normalize=True))

    print(f"\n✅ Final predictors for fine-tuning: {X_train_out.columns.tolist()}, total columns: {X_train_out.shape[1]}")
    print(f"Validation set proportion (test_size): {test_size:.2%}")

    return X_train_out, X_val_out, y_train_out, y_val_out
def validate_tabpfn(
    *,
    X_train: torch.Tensor,  # (n_samples, batch_size, n_features)
    y_train: torch.Tensor,  # (n_samples, batch_size, 1)
    X_val: torch.Tensor,    # (n_samples, batch_size, n_features)
    y_val: torch.Tensor,    # (n_samples, batch_size, 1)
    validation_metric: Scorer,
    model: PerFeatureTransformer,
    model_forward_fn: Callable,
    task_type: TaskType,
    device: SupportedDevice,
    use_sklearn_interface_for_validation: bool = False,
    model_for_validation: TabPFNClassifier | TabPFNRegressor = None,
) -> float:
    """Original validate_tabpfn (未修改)"""
    if use_sklearn_interface_for_validation:
        if model_for_validation is None:
            raise ValueError("Model for validation is required when using full TabPFN preprocessing.")
        if model_for_validation.fit_mode != 'fit_preprocessors':
            raise ValueError("fit_mode must be 'fit_preprocessors' for model_for_validation.")
        if model_for_validation.memory_saving_mode:
            raise ValueError("memory_saving_mode must be False for model_for_validation.")

        estimator_type = model_for_validation.__sklearn_tags__().estimator_type
        if task_type == TaskType.REGRESSION and estimator_type != 'regressor':
            raise ValueError(f"model_for_validation must be TabPFNRegressor, got {type(model_for_validation)}")
        if task_type in {TaskType.MULTICLASS_CLASSIFICATION, TaskType.BINARY_CLASSIFICATION} and estimator_type != 'classifier':
            raise ValueError(f"model_for_validation must be TabPFNClassifier, got {type(model_for_validation)}")

        X_val = X_val.cpu().detach().numpy()[:, 0, :]
        y_true = y_val.flatten().cpu().detach().numpy()

        if not hasattr(model_for_validation, 'executor_'):
            X_train_np = X_train.cpu().detach().numpy()[:, 0, :]
            y_train_np = y_train.flatten().cpu().detach().numpy()
            model_for_validation.fit(X_train_np, y_train_np)

        model_for_validation.model_ = model
        model_for_validation.executor_.model = model

        if task_type == TaskType.REGRESSION:
            y_pred = model_for_validation.predict(X_val, output_type="mean")
        else:
            y_pred = model_for_validation.predict_proba(X_val)

        model.to(device)
    else:
        X_train, y_train, X_val, y_val = X_train.to(device), y_train.to(device), X_val.to(device), y_val.to(device)
        pred_logits = model_forward_fn(model=model, X_train=X_train, y_train=y_train, X_test=X_val, forward_for_validation=True)

        match task_type:
            case TaskType.REGRESSION:
                y_pred = pred_logits.float().flatten().cpu().detach().numpy()
                y_true = y_val.float().flatten().cpu().detach().numpy()
            case TaskType.BINARY_CLASSIFICATION:
                y_pred = torch.sigmoid(pred_logits[:, 0, 1]).cpu().detach().numpy()
                y_true = y_val.long().flatten().cpu().detach().numpy()
            case TaskType.MULTICLASS_CLASSIFICATION:
                y_pred = torch.softmax(pred_logits[:, 0, :], dim=-1).cpu().detach().numpy()
                y_true = y_val.long().flatten().cpu().detach().numpy()
            case _:
                raise ValueError(f"Task type {task_type} not supported.")

    X_train.cpu()
    y_train.cpu()
    X_val.cpu()
    y_val.cpu()

    score = validation_metric(y_true=y_true, y_pred=y_pred)
    return validation_metric.convert_score_to_error(score=score)