"""
Kitchener Random Forest Modelling - Project 3B T126

Trains and evaluates a Random Forest classifier using the processed Kitchener
model-ready dataset.

Main use in notebook:
- run_random_forest_workflow()
- display_random_forest_summary()
- plot_feature_importance()
"""

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def load_data(data_path, target_col="has_break"):
    """Load model-ready dataset and split into X and y."""
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


def build_model(X_train, random_state=42):
    """Build preprocessing + Random Forest pipeline."""
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X_train.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", rf),
        ]
    )

    return model


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


def get_feature_importance(model):
    """Return Random Forest feature importance as a DataFrame."""
    preprocessor = model.named_steps["preprocessor"]
    rf = model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()

    return (
        pd.DataFrame({
            "feature": feature_names,
            "importance": rf.feature_importances_,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def run_random_forest_workflow(
    data_path,
    target_col="has_break",
    random_state=42,
):
    """Run full Random Forest workflow."""
    df, X, y = load_data(data_path, target_col=target_col)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X,
        y,
        random_state=random_state,
    )

    model = build_model(X_train, random_state=random_state)
    model.fit(X_train, y_train)

    results = {
        "data": df,
        "model": model,
        "split_summary": {
            "train_shape": X_train.shape,
            "validation_shape": X_val.shape,
            "test_shape": X_test.shape,
            "train_target_distribution": y_train.value_counts(),
            "validation_target_distribution": y_val.value_counts(),
            "test_target_distribution": y_test.value_counts(),
        },
        "train_metrics": evaluate_model(model, X_train, y_train),
        "validation_metrics": evaluate_model(model, X_val, y_val),
        "test_metrics": evaluate_model(model, X_test, y_test),
        "feature_importance": get_feature_importance(model),
    }

    return results


def display_random_forest_summary(results):
    """Display key Random Forest outputs for notebook demonstration."""
    print("Data split")
    print("Train shape:", results["split_summary"]["train_shape"])
    print("Validation shape:", results["split_summary"]["validation_shape"])
    print("Test shape:", results["split_summary"]["test_shape"])

    print("\nValidation metrics")
    for metric, value in results["validation_metrics"].items():
        if metric != "confusion_matrix":
            print(f"{metric}: {value}")

    print("\nTest metrics")
    for metric, value in results["test_metrics"].items():
        if metric != "confusion_matrix":
            print(f"{metric}: {value}")

    print("\nTest confusion matrix")
    print(results["test_metrics"]["confusion_matrix"])

    print("\nTop 10 feature importance")
    display(results["feature_importance"].head(10))


def plot_feature_importance(results, top_n=15, save_path=None):
    """Plot Random Forest feature importance."""
    
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
    plt.title(f"Top {top_n} Random Forest Features")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()
    

def plot_confusion_matrix(results, save_path=None):
    """Plot Random Forest confusion matrix."""
    cm = results["test_metrics"]["confusion_matrix"]

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No observed break", "Observed break"],
    ).plot()

    plt.title("Random Forest Test Confusion Matrix")
    plt.grid(False)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()