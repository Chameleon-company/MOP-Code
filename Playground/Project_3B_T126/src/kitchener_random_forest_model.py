"""
Random Forest Model - Kitchener Water Pipe Failure Prediction

This module contains the reusable Random Forest workflow converted from the
original Random Forest notebook.

Main workflow:
1. Load model-ready Kitchener dataset
2. Split into train / validation / test sets
3. One-hot encode categorical features
4. Train baseline Random Forest
5. Tune Random Forest using validation PR-AUC
6. Retrain final model using train + validation data
7. Evaluate final model on the unseen test set
8. Extract feature importance and grouped business-friendly drivers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)


SEED = 42


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

def load_random_forest_data(data_path):
    """Load the model-ready Kitchener dataset."""
    return pd.read_csv(data_path)


def split_features_target(df, target_col="has_break"):
    """Split dataframe into features X and target y."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


# --------------------------------------------------
# 2. Train / validation / test split
# --------------------------------------------------

def create_train_val_test_split(X, y, seed=SEED):
    """
    Split data into 70% train, 15% validation, and 15% test.

    Stratified splitting is used to preserve the has_break class distribution.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=seed,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=seed,
        stratify=y_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# --------------------------------------------------
# 3. Encode categorical features
# --------------------------------------------------

def get_column_groups(X_train):
    """Identify categorical and numerical columns from the training set."""
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X_train.select_dtypes(exclude=["object"]).columns.tolist()
    return cat_cols, num_cols


def encode_splits(X_train, X_val, X_test, cat_cols):
    """
    One-hot encode categorical columns and align validation/test sets
    to the training feature space.
    """
    X_train_enc = pd.get_dummies(X_train, columns=cat_cols, drop_first=False)
    X_val_enc = pd.get_dummies(X_val, columns=cat_cols, drop_first=False)
    X_test_enc = pd.get_dummies(X_test, columns=cat_cols, drop_first=False)

    X_val_enc = X_val_enc.reindex(columns=X_train_enc.columns, fill_value=0)
    X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

    return X_train_enc, X_val_enc, X_test_enc


# --------------------------------------------------
# 4. Train baseline Random Forest
# --------------------------------------------------

def train_baseline_random_forest(X_train_enc, y_train, seed=SEED):
    """Train the baseline Random Forest model from the original notebook."""
    rf_baseline = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )

    rf_baseline.fit(X_train_enc, y_train)
    return rf_baseline


# --------------------------------------------------
# 5. Evaluation helpers
# --------------------------------------------------

def evaluate_classifier(model, X, y):
    """Return core classification metrics, predictions, probabilities, and confusion matrix."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    metrics = {
        "ROC-AUC": roc_auc_score(y, y_prob),
        "PR-AUC": average_precision_score(y, y_prob),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1": f1_score(y, y_pred),
    }

    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred)

    return {
        "metrics": metrics,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "confusion_matrix": cm,
        "classification_report": report,
    }


# --------------------------------------------------
# 6. Tune Random Forest
# --------------------------------------------------

def tune_random_forest(X_train_enc, y_train, X_val_enc, y_val, seed=SEED):
    """
    Tune Random Forest hyperparameters using validation PR-AUC.

    This matches the parameter grid used in the original notebook.
    """
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 10],
        "min_samples_leaf": [1, 4],
        "max_features": ["sqrt", "log2"],
        "class_weight": ["balanced"],
    }

    best_score = -1
    best_params = None
    best_model = None

    for params in ParameterGrid(param_grid):
        rf = RandomForestClassifier(
            **params,
            random_state=seed,
            n_jobs=-1,
        )

        rf.fit(X_train_enc, y_train)

        y_val_prob = rf.predict_proba(X_val_enc)[:, 1]
        score = average_precision_score(y_val, y_val_prob)

        if score > best_score:
            best_score = score
            best_params = params
            best_model = rf

    return best_model, best_params, best_score


# --------------------------------------------------
# 7. Retrain final model
# --------------------------------------------------

def retrain_final_random_forest(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    best_params,
    cat_cols,
    seed=SEED,
):
    """
    Retrain the final Random Forest model using train + validation data.

    The final test set remains untouched until final evaluation.
    """
    X_trainval = pd.concat([X_train, X_val], axis=0)
    y_trainval = pd.concat([y_train, y_val], axis=0)

    X_trainval_enc = pd.get_dummies(X_trainval, columns=cat_cols, drop_first=False)
    X_test_final_enc = pd.get_dummies(X_test, columns=cat_cols, drop_first=False)

    X_test_final_enc = X_test_final_enc.reindex(
        columns=X_trainval_enc.columns,
        fill_value=0,
    )

    final_rf = RandomForestClassifier(
        **best_params,
        random_state=seed,
        n_jobs=-1,
    )

    final_rf.fit(X_trainval_enc, y_trainval)

    return final_rf, X_trainval_enc, X_test_final_enc, y_trainval


# --------------------------------------------------
# 8. Feature importance
# --------------------------------------------------

def get_feature_importance(model, feature_names):
    """Extract feature importance from the final Random Forest model."""
    feature_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    })

    feature_importance = feature_importance.sort_values(
        by="importance",
        ascending=False,
    ).reset_index(drop=True)

    return feature_importance


def group_feature_importance(feature_importance):
    """
    Group one-hot encoded features into business-friendly driver groups.

    This matches the grouping logic used in the original notebook.
    """
    grouped_importance = {
        "Condition Score": 0,
        "Pipe Length": 0,
        "Pipe Age": 0,
        "Pipe Size": 0,
        "Material": 0,
        "Pressure Zone": 0,
        "Acquisition": 0,
        "Ownership": 0,
        "Cleaning Area / Subarea": 0,
        "Criticality": 0,
        "Maintenance / Cleaning Indicators": 0,
        "Physical Risk Flags": 0,
        "Category": 0,
        "Lining": 0,
        "Other": 0,
    }

    for _, row in feature_importance.iterrows():
        feature = row["feature"]
        importance = row["importance"]

        if feature == "condition_score":
            grouped_importance["Condition Score"] += importance
        elif feature == "shape__length":
            grouped_importance["Pipe Length"] += importance
        elif feature == "pipe_age":
            grouped_importance["Pipe Age"] += importance
        elif feature == "pipe_size":
            grouped_importance["Pipe Size"] += importance
        elif feature.startswith("material_"):
            grouped_importance["Material"] += importance
        elif feature.startswith("pressure_zone_"):
            grouped_importance["Pressure Zone"] += importance
        elif feature.startswith("acquisition_"):
            grouped_importance["Acquisition"] += importance
        elif feature.startswith("ownership_"):
            grouped_importance["Ownership"] += importance
        elif feature == "rel_cleaning_area" or feature.startswith("rel_cleaning_subarea_"):
            grouped_importance["Cleaning Area / Subarea"] += importance
        elif feature == "criticality":
            grouped_importance["Criticality"] += importance
        elif feature == "cleaned":
            grouped_importance["Maintenance / Cleaning Indicators"] += importance
        elif feature in ["bridge_main", "undersized", "shallow_main", "oversized"]:
            grouped_importance["Physical Risk Flags"] += importance
        elif feature.startswith("category_"):
            grouped_importance["Category"] += importance
        elif feature == "lined" or feature.startswith("lined_material_"):
            grouped_importance["Lining"] += importance
        else:
            grouped_importance["Other"] += importance

    grouped_importance_df = pd.DataFrame({
        "driver_group": grouped_importance.keys(),
        "importance": grouped_importance.values(),
    })

    grouped_importance_df = grouped_importance_df.sort_values(
        by="importance",
        ascending=False,
    ).reset_index(drop=True)

    return grouped_importance_df


# --------------------------------------------------
# 9. Plot helpers
# --------------------------------------------------

def plot_confusion_matrix(cm, title="Random Forest Confusion Matrix"):
    """Plot confusion matrix."""
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.title(title)
    plt.show()


def plot_precision_recall_curve(y_true, y_prob, title="Random Forest Precision-Recall Curve"):
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_grouped_feature_importance(grouped_importance_df):
    """Plot grouped feature importance."""
    grouped_plot = grouped_importance_df.sort_values(by="importance", ascending=True)

    plt.figure(figsize=(9, 6))
    plt.barh(grouped_plot["driver_group"], grouped_plot["importance"])
    plt.xlabel("Importance")
    plt.title("Grouped Random Forest Feature Importance")
    plt.show()


# --------------------------------------------------
# 10. Full workflow
# --------------------------------------------------

def run_random_forest_workflow(data_path, seed=SEED, verbose=True):
    """
    Run the full Random Forest workflow.

    Returns a dictionary containing:
    - dataset summary
    - baseline validation results
    - best parameters
    - final test results
    - feature importance
    - grouped feature importance
    - final model
    - encoded test data and labels for plotting
    """
    np.random.seed(seed)

    # Load data
    df = load_random_forest_data(data_path)

    if verbose:
        print("Dataset shape:", df.shape)
        print("\nTarget distribution:")
        print(df["has_break"].value_counts())
        print("\nTarget distribution (%):")
        print(df["has_break"].value_counts(normalize=True))

    # Split features and target
    X, y = split_features_target(df)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_split(
        X,
        y,
        seed=seed,
    )

    # Column groups
    cat_cols, num_cols = get_column_groups(X_train)

    if verbose:
        print("\nCategorical columns:", cat_cols)
        print("Numerical columns:", num_cols)

    # Encode data
    X_train_enc, X_val_enc, X_test_enc = encode_splits(
        X_train,
        X_val,
        X_test,
        cat_cols,
    )

    # Baseline model
    rf_baseline = train_baseline_random_forest(X_train_enc, y_train, seed=seed)
    baseline_val_results = evaluate_classifier(rf_baseline, X_val_enc, y_val)

    if verbose:
        print("\nBaseline validation metrics:")
        print(baseline_val_results["metrics"])

    # Tune model
    best_model, best_params, best_val_pr_auc = tune_random_forest(
        X_train_enc,
        y_train,
        X_val_enc,
        y_val,
        seed=seed,
    )

    if verbose:
        print("\nBest validation PR-AUC:", best_val_pr_auc)
        print("Best parameters:", best_params)

    # Final retraining
    final_rf, X_trainval_enc, X_test_final_enc, y_trainval = retrain_final_random_forest(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        best_params,
        cat_cols,
        seed=seed,
    )

    # Final evaluation
    test_results = evaluate_classifier(final_rf, X_test_final_enc, y_test)

    if verbose:
        print("\nFinal test metrics:")
        print(test_results["metrics"])
        print("\nConfusion matrix:")
        print(test_results["confusion_matrix"])

    # Feature importance
    feature_importance = get_feature_importance(
        final_rf,
        X_trainval_enc.columns,
    )

    grouped_importance = group_feature_importance(feature_importance)

    return {
        "data_shape": df.shape,
        "target_distribution": df["has_break"].value_counts(),
        "target_distribution_pct": df["has_break"].value_counts(normalize=True),
        "categorical_columns": cat_cols,
        "numerical_columns": num_cols,
        "train_shape": X_train.shape,
        "validation_shape": X_val.shape,
        "test_shape": X_test.shape,
        "encoded_train_shape": X_train_enc.shape,
        "encoded_validation_shape": X_val_enc.shape,
        "encoded_test_shape": X_test_enc.shape,
        "baseline_model": rf_baseline,
        "baseline_validation_results": baseline_val_results,
        "best_validation_model": best_model,
        "best_params": best_params,
        "best_validation_pr_auc": best_val_pr_auc,
        "final_model": final_rf,
        "test_results": test_results,
        "feature_importance": feature_importance,
        "grouped_feature_importance": grouped_importance,
        "X_test_final_enc": X_test_final_enc,
        "y_test": y_test,
    }