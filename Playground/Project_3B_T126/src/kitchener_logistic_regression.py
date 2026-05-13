"""
Kitchener Logistic Regression Modelling - Project 3B T126

Trains and evaluates a Logistic Regression classifier using the processed
Kitchener model-ready dataset.

Main use in notebook:
- run_logistic_regression_workflow()
- display_logistic_regression_summary()
- plot_confusion_matrix()
- plot_feature_importance()
"""

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LogisticRegression
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
from sklearn.preprocessing import StandardScaler


def load_data(data_path, target_col="has_break"):
    """Load model-ready dataset and separate features and target."""
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return df, X, y


def clean_missing_values(X):
    """Fill missing categorical and numeric values."""
    X = X.copy()

    categorical_cols = X.select_dtypes(include=["object", "string"]).columns
    numeric_cols = X.select_dtypes(include=["number"]).columns

    X[categorical_cols] = X[categorical_cols].fillna("UNKNOWN")
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

    return X


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


def encode_splits(X_train, X_val, X_test):
    """One-hot encode categorical columns after splitting and align columns."""
    cat_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()

    X_train_enc = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
    X_val_enc = pd.get_dummies(X_val, columns=cat_cols, drop_first=True)
    X_test_enc = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)

    X_val_enc = X_val_enc.reindex(columns=X_train_enc.columns, fill_value=0)
    X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

    return X_train_enc, X_val_enc, X_test_enc


def scale_splits(X_train, X_val, X_test):
    """Scale train, validation and test data using training statistics only."""
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def build_model():
    """Build Logistic Regression model."""
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
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


def get_feature_importance(model, feature_names):
    """Return Logistic Regression coefficients as feature importance."""
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": model.coef_[0],
        "absolute_coefficient": abs(model.coef_[0]),
    })

    return importance_df.sort_values(
        "absolute_coefficient",
        ascending=False,
    ).reset_index(drop=True)


def run_logistic_regression_workflow(
    data_path,
    target_col="has_break",
    random_state=42,
):
    """Run full Logistic Regression workflow."""
    df, X, y = load_data(data_path, target_col=target_col)

    X = clean_missing_values(X)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X,
        y,
        random_state=random_state,
    )

    X_train_enc, X_val_enc, X_test_enc = encode_splits(
        X_train,
        X_val,
        X_test,
    )

    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_splits(
        X_train_enc,
        X_val_enc,
        X_test_enc,
    )

    model = build_model()
    model.fit(X_train_scaled, y_train)

    results = {
        "data": df,
        "model": model,
        "scaler": scaler,
        "feature_names": X_train_enc.columns,
        "split_summary": {
            "train_shape": X_train.shape,
            "validation_shape": X_val.shape,
            "test_shape": X_test.shape,
            "encoded_train_shape": X_train_enc.shape,
            "encoded_validation_shape": X_val_enc.shape,
            "encoded_test_shape": X_test_enc.shape,
            "train_target_distribution": y_train.value_counts(),
            "validation_target_distribution": y_val.value_counts(),
            "test_target_distribution": y_test.value_counts(),
        },
        "validation_metrics": evaluate_model(model, X_val_scaled, y_val),
        "test_metrics": evaluate_model(model, X_test_scaled, y_test),
        "feature_importance": get_feature_importance(
            model,
            X_train_enc.columns,
        ),
    }

    return results


def display_logistic_regression_summary(results):
    """Display key Logistic Regression outputs for notebook demonstration."""
    print("Data split")
    print("Train shape:", results["split_summary"]["train_shape"])
    print("Validation shape:", results["split_summary"]["validation_shape"])
    print("Test shape:", results["split_summary"]["test_shape"])

    print("\nEncoded feature space")
    print("Encoded train shape:", results["split_summary"]["encoded_train_shape"])
    print("Encoded validation shape:", results["split_summary"]["encoded_validation_shape"])
    print("Encoded test shape:", results["split_summary"]["encoded_test_shape"])

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

    print("\nTop 10 coefficients by absolute value")
    display(results["feature_importance"].head(10))

def plot_confusion_matrix(results, save_path=None):
    """Plot Logistic Regression confusion matrix."""
    cm = results["test_metrics"]["confusion_matrix"]

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No observed break", "Observed break"],
    ).plot()

    plt.title("Logistic Regression Test Confusion Matrix")
    plt.grid(False)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()

def plot_feature_importance(results, top_n=15, save_path=None):
    """Plot top Logistic Regression coefficients."""
    
    importance_df = (
        results["feature_importance"]
        .head(top_n)
        .sort_values("absolute_coefficient", ascending=True)
    )

    plt.figure(figsize=(8, 6))

    plt.barh(
        importance_df["feature"],
        importance_df["coefficient"],
    )

    plt.xlabel("Coefficient value")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Logistic Regression Coefficients")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()