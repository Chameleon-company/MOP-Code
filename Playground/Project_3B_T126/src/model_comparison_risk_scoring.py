"""
Model Comparison and Melbourne Risk Scoring - Project 3B T126

Compares Logistic Regression, Random Forest and XGBoost results, consolidates
transferable risk drivers, and creates a Melbourne risk scoring output for
dashboard and LLM recommendation stages.

Main outputs:
- model_results.csv
- model_ranking.csv
- risk_drivers.csv
- melbourne_risk_llm_ready.csv
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


MODEL_METRICS = [
    "ROC-AUC",
    "PR-AUC",
    "Precision",
    "Recall",
    "F1-score",
]


DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_PROCESSED_DIR = "data/processed"


def build_model_results():
    """Create final model comparison table using latest test results."""
    return pd.DataFrame([
        {
            "Model": "Logistic Regression",
            "ROC-AUC": 0.9840,
            "PR-AUC": 0.9404,
            "Precision": 0.7964,
            "Recall": 0.9514,
            "F1-score": 0.8670,
            "True Negatives": 2195,
            "False Positives": 45,
            "False Negatives": 9,
            "True Positives": 176,
        },
        {
            "Model": "Random Forest",
            "ROC-AUC": 0.9906,
            "PR-AUC": 0.9596,
            "Precision": 0.9508,
            "Recall": 0.9405,
            "F1-score": 0.9457,
            "True Negatives": 2231,
            "False Positives": 9,
            "False Negatives": 11,
            "True Positives": 174,
        },
        {
            "Model": "XGBoost",
            "ROC-AUC": 0.9899,
            "PR-AUC": 0.9614,
            "Precision": 0.9133,
            "Recall": 0.9676,
            "F1-score": 0.9396,
            "True Negatives": 2223,
            "False Positives": 17,
            "False Negatives": 6,
            "True Positives": 179,
        },
    ])


def rank_models(model_results):
    """Rank models across key evaluation metrics."""
    ranking = model_results[["Model"] + MODEL_METRICS].copy()

    for metric in MODEL_METRICS:
        ranking[f"{metric} Rank"] = (
            ranking[metric]
            .rank(ascending=False, method="min")
            .astype(int)
        )

    return ranking


def build_risk_driver_table():
    """Create consolidated transferable risk driver table."""
    return pd.DataFrame([
        {
            "Risk Driver": "Pipe Age",
            "Evidence from Kitchener Models": "Important in tree-based models and linked to deterioration over time",
            "Melbourne Field": "PIPE_AGE",
            "Transferability": "High",
            "Use in Melbourne Risk System": "Older pipes receive higher risk weight",
        },
        {
            "Risk Driver": "Pipe Length",
            "Evidence from Kitchener Models": "Strong driver in Random Forest and Logistic Regression",
            "Melbourne Field": "PIPE_LENGTH",
            "Transferability": "High",
            "Use in Melbourne Risk System": "Longer pipe segments receive higher risk weight",
        },
        {
            "Risk Driver": "Material",
            "Evidence from Kitchener Models": "Material-related features contributed to prediction",
            "Melbourne Field": "MATERIAL",
            "Transferability": "High",
            "Use in Melbourne Risk System": "Materials associated with higher failure risk receive higher weight",
        },
        {
            "Risk Driver": "Pipe Size / Width",
            "Evidence from Kitchener Models": "Moderate contribution in Random Forest and XGBoost",
            "Melbourne Field": "PIPE_WIDTH",
            "Transferability": "Medium",
            "Use in Melbourne Risk System": "Used as a supporting physical risk factor",
        },
        {
            "Risk Driver": "Condition / Asset Health",
            "Evidence from Kitchener Models": "Strongest driver through condition_score",
            "Melbourne Field": "Not directly available",
            "Transferability": "Limited",
            "Use in Melbourne Risk System": "Use proxy features such as age and relining status",
        },
        {
            "Risk Driver": "Relining / Maintenance History",
            "Evidence from Kitchener Models": "Lining-related features had lower importance but remain relevant",
            "Melbourne Field": "HAS_RELINED, YEARS_SINCE_RELINED",
            "Transferability": "Medium",
            "Use in Melbourne Risk System": "No relining or older relining may increase risk",
        },
        {
            "Risk Driver": "Network / Operational Context",
            "Evidence from Kitchener Models": "Pressure zone and cleaning area contributed to prediction",
            "Melbourne Field": "MAIN_LINE_TYPE, MAIN_CLASS, FIELD_TEAM",
            "Transferability": "Medium",
            "Use in Melbourne Risk System": "Used as context for explanation and grouping",
        },
    ])


def generate_risk_reasons(row):
    """Generate plain-English reasons for assigned risk score."""
    reasons = []

    if row["PIPE_AGE"] >= 80:
        reasons.append("Very old pipe")
    elif row["PIPE_AGE"] >= 50:
        reasons.append("Aging pipe")

    if row["PIPE_LENGTH"] > 50:
        reasons.append("Long pipe segment")

    if row["MATERIAL"] in ["CI", "CICL", "AC", "RC"]:
        reasons.append("Higher-risk material")

    if row["HAS_RELINED"] == 0:
        reasons.append("No relining history")

    if row["YEARS_SINCE_RELINED"] >= 50:
        reasons.append("Relined long time ago")

    return ", ".join(reasons)


def generate_action(row):
    """Generate recommended action based on risk level."""
    if row["RISK_LEVEL"] == "HIGH":
        return "Inspect and prioritise maintenance"
    if row["RISK_LEVEL"] == "MEDIUM":
        return "Monitor condition and schedule inspection"
    return "Routine monitoring"


def score_melbourne_risk(input_path, output_path=None):
    """Create Melbourne risk score, risk level, reasons and actions."""
    df = pd.read_csv(input_path)

    df["RISK_SCORE"] = 0

    # Age risk
    df.loc[df["PIPE_AGE"] >= 80, "RISK_SCORE"] += 3
    df.loc[(df["PIPE_AGE"] >= 50) & (df["PIPE_AGE"] < 80), "RISK_SCORE"] += 2
    df.loc[(df["PIPE_AGE"] >= 30) & (df["PIPE_AGE"] < 50), "RISK_SCORE"] += 1

    # Missing construction date uncertainty
    df.loc[df["MISSING_CONSTRUCTION_DATE"] == 1, "RISK_SCORE"] += 1

    # Pipe length risk
    length_75 = df["PIPE_LENGTH"].quantile(0.75)
    length_90 = df["PIPE_LENGTH"].quantile(0.90)

    df.loc[df["PIPE_LENGTH"] >= length_90, "RISK_SCORE"] += 2
    df.loc[
        (df["PIPE_LENGTH"] >= length_75) & (df["PIPE_LENGTH"] < length_90),
        "RISK_SCORE",
    ] += 1

    # Material risk
    higher_risk_materials = ["CI", "CICL", "AC", "RC"]
    df.loc[df["MATERIAL"].isin(higher_risk_materials), "RISK_SCORE"] += 2

    # Relining risk
    df.loc[df["HAS_RELINED"] == 0, "RISK_SCORE"] += 1
    df.loc[df["YEARS_SINCE_RELINED"] >= 50, "RISK_SCORE"] += 1

    # Data quality risk
    df.loc[df["INVALID_PIPE_LENGTH"] == 1, "RISK_SCORE"] += 1
    df.loc[df["INVALID_PIPE_WIDTH"] == 1, "RISK_SCORE"] += 1
    df.loc[df["FUTURE_RELINED_DATE"] == 1, "RISK_SCORE"] += 1

    df["RISK_LEVEL"] = pd.cut(
        df["RISK_SCORE"],
        bins=[-1, 2, 5, 100],
        labels=["LOW", "MEDIUM", "HIGH"],
    )

    df["RISK_REASONS"] = df.apply(generate_risk_reasons, axis=1)
    df["RISK_RANK"] = df["RISK_SCORE"].rank(ascending=False, method="min").astype(int)
    df["RECOMMENDED_ACTION"] = df.apply(generate_action, axis=1)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


def get_high_risk_pipes(df, top_n=20):
    """Return top high-risk Melbourne pipes."""
    return (
        df[df["RISK_LEVEL"] == "HIGH"]
        .sort_values(by="RISK_SCORE", ascending=False)
        .head(top_n)
    )


def save_model_outputs(
    model_results,
    ranking,
    risk_drivers,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    """Save model comparison and risk driver outputs."""
    os.makedirs(output_dir, exist_ok=True)

    paths = {
        "model_results": os.path.join(output_dir, "model_results.csv"),
        "model_ranking": os.path.join(output_dir, "model_ranking.csv"),
        "risk_drivers": os.path.join(output_dir, "risk_drivers.csv"),
    }

    model_results.to_csv(paths["model_results"], index=False)
    ranking.to_csv(paths["model_ranking"], index=False)
    risk_drivers.to_csv(paths["risk_drivers"], index=False)

    return paths


def run_model_comparison_and_risk_scoring(
    melbourne_input_path="data/processed/melbourne_cleaned_for_adaptation.csv",
    melbourne_output_path="data/processed/melbourne_risk_llm_ready.csv",
    output_dir=DEFAULT_OUTPUT_DIR,
):
    """Run model comparison, risk driver consolidation and Melbourne scoring."""
    model_results = build_model_results()
    ranking = rank_models(model_results)
    risk_drivers = build_risk_driver_table()

    saved_paths = save_model_outputs(
        model_results=model_results,
        ranking=ranking,
        risk_drivers=risk_drivers,
        output_dir=output_dir,
    )

    melbourne_scored = score_melbourne_risk(
        input_path=melbourne_input_path,
        output_path=melbourne_output_path,
    )

    return {
        "model_results": model_results,
        "ranking": ranking,
        "risk_drivers": risk_drivers,
        "melbourne_scored": melbourne_scored,
        "high_risk_pipes": get_high_risk_pipes(melbourne_scored),
        "saved_paths": saved_paths,
        "melbourne_output_path": melbourne_output_path,
    }


def plot_model_performance(model_results, save_path=None):
    """Plot model performance comparison."""
    plot_df = model_results.set_index("Model")[MODEL_METRICS]

    plot_df.plot(kind="bar", figsize=(10, 5))

    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.ylim(0.75, 1.00)
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_error_comparison(model_results, save_path=None):
    """Plot false positive and false negative comparison."""
    error_df = model_results.set_index("Model")[["False Positives", "False Negatives"]]

    error_df.plot(kind="bar", figsize=(8, 5))

    plt.title("False Positive and False Negative Comparison")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def display_model_comparison_summary(model_results, ranking):
    """Display model comparison outputs."""
    print("Model results")
    display(model_results)

    print("\nModel ranking")
    display(ranking)


def display_risk_scoring_summary(scored_df):
    """Display Melbourne risk scoring outputs."""
    print("Risk level distribution")
    display(scored_df["RISK_LEVEL"].value_counts())

    print("\nOutput shape:", scored_df.shape)

    preview_cols = [
        "ASSET_ID",
        "COMPKEY",
        "MATERIAL",
        "PIPE_LENGTH",
        "PIPE_WIDTH",
        "PIPE_AGE",
        "HAS_RELINED",
        "YEARS_SINCE_RELINED",
        "RISK_SCORE",
        "RISK_LEVEL",
        "RISK_REASONS",
        "RECOMMENDED_ACTION",
    ]

    display(scored_df[preview_cols].head())