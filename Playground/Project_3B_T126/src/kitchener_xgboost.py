"""
Kitchener XGBoost Modelling - Project 3B T126

Trains and evaluates an XGBoost classifier using the processed Kitchener
model-ready dataset.

This version uses the best hyperparameters identified during earlier tuning.
Optuna is not required for the final demo notebook.

Main use in notebook:
- run_xgboost_workflow()
- display_xgboost_summary()
- plot_confusion_matrix()
- plot_feature_importance()
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


BEST_XGB_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.04110359181702955,
    "subsample": 0.7514526781869276,
    "colsample_bytree": 0.6976106662148625,
    "min_child_weight": 7,
    "gamma": 1.9395005517710073,
}


def load_data(data_path, target_col="has_break"):
    """Load model-ready dataset and separate features and target."""
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return df, X, y


def split_data(X, y, random_state=42):
    """Split data into 70% train, 15% validation and 15% test."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=random_state,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def convert_categorical_columns(*datasets):
    """Convert object/string columns to category dtype for native XGBoost."""
    converted = []

    for df in datasets:
        df_copy = df.copy()
        cat_cols = df_copy.select_dtypes(include=["object", "string"]).columns.tolist()

        for col in cat_cols:
            df_copy[col] = df_copy[col].astype("category")

        converted.append(df_copy)

    return converted


def calculate_scale_pos_weight(y):
    """Calculate class imbalance weight for XGBoost."""
    return (y == 0).sum() / (y == 1).sum()


def build_xgboost_model(scale_pos_weight, random_state=42):
    """Build XGBoost classifier using fixed tuned parameters."""
    return xgb.XGBClassifier(
        **BEST_XGB_PARAMS,
        n_estimators=800,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        enable_categorical=True,
        eval_metric="aucpr",
        n_jobs=-1,
        verbosity=0,
    )


def evaluate_model(model, X, y):
    """Evaluate model using imbalanced classification metrics."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    return {
        "roc_auc": round(roc_auc_score(y, y_prob), 4),
        "pr_auc": round(average_precision_score(y, y_prob), 4),
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y, y_pred, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y, y_pred),
    }


def get_feature_importance(model, importance_type="gain"):
    """Return XGBoost feature importance as a DataFrame."""
    booster = model.get_booster()
    importance = booster.get_score(importance_type=importance_type)

    importance_df = pd.DataFrame(
        {
            "feature": list(importance.keys()),
            "importance": list(importance.values()),
        }
    )

    return importance_df.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)


def run_xgboost_workflow(
    data_path,
    target_col="has_break",
    random_state=42,
):
    """Run full XGBoost workflow."""
    np.random.seed(random_state)

    df, X, y = load_data(data_path, target_col=target_col)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X,
        y,
        random_state=random_state,
    )

    X_train, X_val, X_test = convert_categorical_columns(
        X_train,
        X_val,
        X_test,
    )

    scale_pos_weight = calculate_scale_pos_weight(y_train)

    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)

    # Re-convert after concat because categorical columns can revert to object
    X_train_val = convert_categorical_columns(X_train_val)[0]
    X_test = convert_categorical_columns(X_test)[0]

    model = build_xgboost_model(
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
    )

    model.fit(X_train_val, y_train_val, verbose=False)

    results = {
        "data": df,
        "model": model,
        "best_params": BEST_XGB_PARAMS,
        "scale_pos_weight": round(scale_pos_weight, 2),
        "split_summary": {
            "train_shape": X_train.shape,
            "validation_shape": X_val.shape,
            "test_shape": X_test.shape,
            "train_val_shape": X_train_val.shape,
            "train_target_distribution": y_train.value_counts(),
            "validation_target_distribution": y_val.value_counts(),
            "test_target_distribution": y_test.value_counts(),
        },
        "validation_metrics": evaluate_model(model, X_val, y_val),
        "test_metrics": evaluate_model(model, X_test, y_test),
        "feature_importance": get_feature_importance(model),
    }

    return results


def display_xgboost_summary(results):
    """Display key XGBoost outputs for notebook demonstration."""
    print("Data split")
    print("Train shape:", results["split_summary"]["train_shape"])
    print("Validation shape:", results["split_summary"]["validation_shape"])
    print("Test shape:", results["split_summary"]["test_shape"])

    print("\nModel settings")
    print("Scale pos weight:", results["scale_pos_weight"])
    print("Fixed tuned parameters:", results["best_params"])

    print("\nTest metrics")
    for metric, value in results["test_metrics"].items():
        if metric != "confusion_matrix":
            print(f"{metric}: {value}")

    print("\nTest confusion matrix")
    print(results["test_metrics"]["confusion_matrix"])

    print("\nTop 10 feature importance")
    display(results["feature_importance"].head(10))


def plot_confusion_matrix(results, save_path=None):
    """Plot XGBoost confusion matrix."""
    cm = results["test_metrics"]["confusion_matrix"]

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No observed break", "Observed break"],
    ).plot()

    plt.title("XGBoost Test Confusion Matrix")
    plt.grid(False)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_feature_importance(results, top_n=15, save_path=None):
    """Plot XGBoost feature importance."""

    importance_df = (
        results["feature_importance"]
        .head(top_n)
        .sort_values("importance", ascending=True)
    )

    plt.figure(figsize=(8, 6))

    plt.barh(
        importance_df["feature"],
        importance_df["importance"],
    )

    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} XGBoost Features")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()