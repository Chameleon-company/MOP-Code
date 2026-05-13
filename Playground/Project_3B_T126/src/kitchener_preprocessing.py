"""
Kitchener Preprocessing - Project 3B T126

Converts raw Kitchener water mains and water main break datasets into clean
pipe-level datasets for modelling.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BREAKS_DROP_COLS = [
    "Type of Planned Maintenance",
    "List Valves Opened",
    "List Valves Closed",
    "Related Asset Depth (m)",
    "CW Service Request Number",
    "List Hydrants Called Back In",
    "List Hydrants Called Out",
    "Depth of Frost (m)",
    "Date operations was returned to normal service",
    "Repair Type",
    "Estimated Number of Units Impacted",
    "Does the sidewalk need to be closed?",
    "Estimated Hours for Repair",
    "Does the road need to be closed?",
    "CW Workorder #",
    "OBJECTID",
    "GLOBALID",
    "UPDATE_BY",
    "UPDATE_DATE",
]

MAINS_DROP_COLS = [
    "BRIDGE_DETAILS",
    "LINED_DATE",
    "CONSULTANT",
    "OBJECTID",
    "MAP_LABEL",
]

BINARY_MAPS = {
    "lined": {"YES": 1, "NO": 0},
    "bridge_main": {"Y": 1, "N": 0},
    "undersized": {"Y": 1, "N": 0},
    "shallow_main": {"Y": 1, "N": 0},
    "oversized": {"Y": 1, "N": 0},
    "cleaned": {"Y": 1, "N": 0},
}

DROP_FOR_MODEL = [
    "watmainid",
    "roadsegmentid",
    "installation_date",
    "install_year",
    "break_count",
    "first_break_date",
    "last_break_date",
    "status",
]


def load_kitchener_data(mains_path, breaks_path):
    """Load raw Kitchener mains and break datasets."""
    mains = pd.read_csv(mains_path)
    breaks = pd.read_csv(breaks_path)
    return mains, breaks


def missing_summary(df):
    """Return missing percentage by column."""
    return (df.isna().mean() * 100).round(2).sort_values(ascending=False)


def clean_source_tables(mains, breaks):
    """Drop weak columns and convert date fields."""
    breaks_clean = breaks.drop(
        columns=[col for col in BREAKS_DROP_COLS if col in breaks.columns]
    ).copy()

    mains_clean = mains.drop(
        columns=[col for col in MAINS_DROP_COLS if col in mains.columns]
    ).copy()

    for col in ["Incident date", "Status last updated date"]:
        if col in breaks_clean.columns:
            breaks_clean[col] = pd.to_datetime(
                breaks_clean[col],
                errors="coerce",
                format="mixed",
            )

    if "INSTALLATION_DATE" in mains_clean.columns:
        mains_clean["INSTALLATION_DATE"] = pd.to_datetime(
            mains_clean["INSTALLATION_DATE"],
            errors="coerce",
            format="mixed",
        )

    return mains_clean, breaks_clean


def duplicate_summary(mains_clean, breaks_clean):
    """Return duplicate checks after basic cleaning."""
    return {
        "duplicate_break_ids": int(
            breaks_clean["Wat Break Incident ID"].duplicated().sum()
        ),
        "duplicate_main_ids": int(mains_clean["WATMAINID"].duplicated().sum()),
        "duplicate_roadsegment_ids_in_mains": int(
            mains_clean["ROADSEGMENTID"].duplicated().sum()
        ),
        "duplicate_related_asset_ids_in_breaks": int(
            breaks_clean["Related Asset ID"].duplicated().sum()
        ),
    }


def asset_match_summary(mains_clean, breaks_clean):
    """Check how many break rows match a water main asset ID."""
    matched_mask = breaks_clean["Related Asset ID"].isin(mains_clean["WATMAINID"])

    return {
        "asset_match_rate_pct": round(matched_mask.mean() * 100, 2),
        "matched_break_rows": int(matched_mask.sum()),
        "unmatched_break_rows": int((~matched_mask).sum()),
    }


def filter_matched_main_breaks(mains_clean, breaks_clean):
    """Keep MAIN break records that match a WATMAINID in the mains table."""
    breaks_main = breaks_clean[breaks_clean["Type of Asset Broken"] == "MAIN"].copy()

    breaks_main_matched = breaks_main[
        breaks_main["Related Asset ID"].isin(mains_clean["WATMAINID"])
    ].copy()

    return breaks_main, breaks_main_matched


def build_break_summary(breaks_main_matched):
    """Aggregate event-level break records into pipe-level break history."""
    break_summary = (
        breaks_main_matched
        .groupby("Related Asset ID")
        .agg(
            break_count=("Wat Break Incident ID", "count"),
            first_break_date=("Incident date", "min"),
            last_break_date=("Incident date", "max"),
        )
        .reset_index()
        .rename(columns={"Related Asset ID": "WATMAINID"})
    )

    break_summary["has_break"] = 1
    return break_summary


def build_pipe_level_dataset(mains_clean, break_summary):
    """Merge mains asset data with pipe-level break history."""
    pipe_df = mains_clean.merge(
        break_summary,
        on="WATMAINID",
        how="left",
    )

    pipe_df["break_count"] = pipe_df["break_count"].fillna(0).astype(int)
    pipe_df["has_break"] = pipe_df["has_break"].fillna(0).astype(int)

    return pipe_df


def standardise_column_names(df):
    """Standardise column names for easier modelling workflow."""
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return df


def finalise_model_datasets(pipe_df, breaks_main_matched):
    """
    Create final master and model-ready datasets.

    Median imputation here creates a baseline model-ready file.
    For strict modelling experiments, imputation should ideally be fitted
    on the training split only.
    """
    model_df = pipe_df.copy()

    reference_date = breaks_main_matched["Incident date"].max()

    model_df["install_year"] = model_df["INSTALLATION_DATE"].dt.year
    model_df["pipe_age"] = (
        (reference_date - model_df["INSTALLATION_DATE"]).dt.days / 365.25
    ).round(1)

    model_df["Condition Score"] = model_df["Condition Score"].replace(-1, np.nan)
    model_df["CRITICALITY"] = model_df["CRITICALITY"].replace(-1, np.nan)

    model_df = standardise_column_names(model_df)

    for col, mapping in BINARY_MAPS.items():
        if col in model_df.columns:
            model_df[col] = model_df[col].map(mapping)

    pipe_master_df = model_df.copy()

    model_ready_df = model_df.drop(
        columns=[col for col in DROP_FOR_MODEL if col in model_df.columns]
    ).copy()

    for col in ["criticality", "condition_score"]:
        if col in model_ready_df.columns:
            model_ready_df[col] = model_ready_df[col].fillna(
                model_ready_df[col].median()
            )

    return pipe_master_df, model_ready_df, reference_date


def save_processed_outputs(
    output_dir,
    breaks_clean,
    mains_clean,
    pipe_df,
    pipe_master_df,
    model_ready_df,
):
    """Save processed Kitchener datasets and return output paths."""
    os.makedirs(output_dir, exist_ok=True)

    output_paths = {
        "breaks_clean": os.path.join(output_dir, "kitchener_breaks_clean.csv"),
        "mains_clean": os.path.join(output_dir, "kitchener_mains_clean.csv"),
        "pipe_level": os.path.join(output_dir, "kitchener_pipe_level.csv"),
        "pipe_master": os.path.join(output_dir, "kitchener_pipe_master.csv"),
        "model_ready": os.path.join(output_dir, "kitchener_model_ready.csv"),
    }

    breaks_clean.to_csv(output_paths["breaks_clean"], index=False)
    mains_clean.to_csv(output_paths["mains_clean"], index=False)
    pipe_df.to_csv(output_paths["pipe_level"], index=False)
    pipe_master_df.to_csv(output_paths["pipe_master"], index=False)
    model_ready_df.to_csv(output_paths["model_ready"], index=False)

    return output_paths


def plot_target_distribution(pipe_df):
    """Plot target class distribution."""
    class_counts = pipe_df["has_break"].value_counts().sort_index()

    plt.figure()
    class_counts.plot(kind="bar")
    plt.xticks([0, 1], ["No observed break", "Observed break"], rotation=0)
    plt.ylabel("Number of pipes")
    plt.xlabel("Target class")
    plt.title("Class Distribution of Pipe Failure Target")
    plt.show()


def display_kitchener_preprocessing_summary(results):
    """Display key preprocessing outputs for notebook demonstration."""
    raw_summary = results["raw_summary"]
    final_summary = results["final_summary"]
    match_summary = results["asset_match_summary"]

    print("Raw data")
    print("Raw mains shape:", raw_summary["raw_mains_shape"])
    print("Raw breaks shape:", raw_summary["raw_breaks_shape"])

    print("\nAfter cleaning")
    print("Cleaned mains shape:", final_summary["cleaned_mains_shape"])
    print("Cleaned breaks shape:", final_summary["cleaned_breaks_shape"])

    print("\nAsset matching")
    print(match_summary)

    print("\nMAIN break filtering")
    print("MAIN breaks shape:", final_summary["main_breaks_shape"])
    print("Matched MAIN breaks shape:", final_summary["matched_main_breaks_shape"])

    print("\nFinal datasets")
    print("Pipe-level dataset shape:", final_summary["pipe_level_shape"])
    print("Pipe master dataset shape:", final_summary["pipe_master_shape"])
    print("Model-ready dataset shape:", final_summary["model_ready_shape"])

    print("\nTarget distribution")
    print(final_summary["target_distribution"])

    print("\nTarget distribution (%)")
    print(final_summary["target_distribution_pct"])


def run_kitchener_preprocessing(
    mains_path,
    breaks_path,
    output_dir="data/processed",
    save_outputs=True,
    verbose=False,
):
    """Run full Kitchener preprocessing pipeline."""
    mains, breaks = load_kitchener_data(mains_path, breaks_path)

    raw_summary = {
        "raw_mains_shape": mains.shape,
        "raw_breaks_shape": breaks.shape,
        "mains_missing_pct": missing_summary(mains),
        "breaks_missing_pct": missing_summary(breaks),
    }

    mains_clean, breaks_clean = clean_source_tables(mains, breaks)

    dup_summary = duplicate_summary(mains_clean, breaks_clean)
    match_summary = asset_match_summary(mains_clean, breaks_clean)

    breaks_main, breaks_main_matched = filter_matched_main_breaks(
        mains_clean,
        breaks_clean,
    )

    break_summary = build_break_summary(breaks_main_matched)
    pipe_df = build_pipe_level_dataset(mains_clean, break_summary)

    pipe_master_df, model_ready_df, reference_date = finalise_model_datasets(
        pipe_df,
        breaks_main_matched,
    )

    output_paths = {}
    if save_outputs:
        output_paths = save_processed_outputs(
            output_dir=output_dir,
            breaks_clean=breaks_clean,
            mains_clean=mains_clean,
            pipe_df=pipe_df,
            pipe_master_df=pipe_master_df,
            model_ready_df=model_ready_df,
        )

    final_summary = {
        "cleaned_mains_shape": mains_clean.shape,
        "cleaned_breaks_shape": breaks_clean.shape,
        "main_breaks_shape": breaks_main.shape,
        "matched_main_breaks_shape": breaks_main_matched.shape,
        "pipe_level_shape": pipe_df.shape,
        "pipe_master_shape": pipe_master_df.shape,
        "model_ready_shape": model_ready_df.shape,
        "unique_pipes_with_observed_break": break_summary["WATMAINID"].nunique(),
        "total_pipes_in_mains": mains_clean["WATMAINID"].nunique(),
        "target_distribution": model_ready_df["has_break"].value_counts(),
        "target_distribution_pct": (
            model_ready_df["has_break"].value_counts(normalize=True) * 100
        ).round(2),
        "model_ready_missing_pct": missing_summary(model_ready_df),
        "reference_date": reference_date,
    }

    results = {
        "raw_mains": mains,
        "raw_breaks": breaks,
        "mains_clean": mains_clean,
        "breaks_clean": breaks_clean,
        "breaks_main": breaks_main,
        "breaks_main_matched": breaks_main_matched,
        "break_summary": break_summary,
        "pipe_df": pipe_df,
        "pipe_master_df": pipe_master_df,
        "model_ready_df": model_ready_df,
        "raw_summary": raw_summary,
        "duplicate_summary": dup_summary,
        "asset_match_summary": match_summary,
        "final_summary": final_summary,
        "output_paths": output_paths,
    }

    if verbose:
        display_kitchener_preprocessing_summary(results)

    return results