"""
Dataset Review - Project 3B T126

Loads and summarises the shortlisted datasets reviewed during the dataset
selection stage.

The purpose is to compare dataset size, structure, missingness, duplicate rows,
join keys, and suitability before selecting the final modelling dataset.

Main use in notebook:
- run_dataset_review()
- display_dataset_summary()
"""

import numpy as np
import pandas as pd


def summarise_dataframe(df, name):
    """Return a compact summary of one dataset."""
    return {
        "dataset": name,
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "column_names": list(df.columns),
    }


def load_melbourne_mains(path):
    """Load Melbourne mains dataset and remove unnamed export columns."""
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    return df


def load_netherlands_data(mains_path, breaks_path):
    """Load Netherlands mains data and reshape 3D break history array."""
    mains = pd.read_pickle(mains_path)

    if "unit_ID" in mains.columns:
        mains = mains.drop(columns="unit_ID")

    breaks_array = np.load(breaks_path)
    breaks_2d = breaks_array.reshape(breaks_array.shape[0], -1)
    breaks_df = pd.DataFrame(breaks_2d)

    return mains, breaks_df, breaks_array.shape


def check_kitchener_join_keys(kitchener_mains, kitchener_breaks):
    """Check if expected Kitchener join keys are available."""
    return {
        "mains_key": "WATMAINID" in kitchener_mains.columns,
        "breaks_key": "Related Asset ID" in kitchener_breaks.columns,
    }


def run_dataset_review(
    melbourne_mains_path,
    melbourne_soil_path,
    netherlands_mains_path,
    netherlands_breaks_path,
    kitchener_mains_path,
    kitchener_breaks_path,
    bozeman_path,
):
    """Load shortlisted datasets and return basic review summaries."""

    melbourne_mains = load_melbourne_mains(melbourne_mains_path)
    melbourne_soil = pd.read_csv(melbourne_soil_path)

    netherlands_mains, netherlands_breaks_df, netherlands_breaks_shape = (
        load_netherlands_data(
            mains_path=netherlands_mains_path,
            breaks_path=netherlands_breaks_path,
        )
    )

    kitchener_mains = pd.read_csv(kitchener_mains_path)
    kitchener_breaks = pd.read_csv(kitchener_breaks_path)

    bozeman = pd.read_csv(bozeman_path)

    summaries = [
        summarise_dataframe(melbourne_mains, "Melbourne Water Mains"),
        summarise_dataframe(melbourne_soil, "Melbourne Soil Readings"),
        summarise_dataframe(netherlands_mains, "Netherlands Water Mains"),
        summarise_dataframe(netherlands_breaks_df, "Netherlands Break History"),
        summarise_dataframe(kitchener_mains, "Kitchener Water Mains"),
        summarise_dataframe(kitchener_breaks, "Kitchener Water Main Breaks"),
        summarise_dataframe(bozeman, "Bozeman Water Main Breaks"),
    ]

    summary_table = pd.DataFrame(summaries)

    return {
        "summary_table": summary_table,
        "netherlands_breaks_shape_original": netherlands_breaks_shape,
        "kitchener_has_join_keys": check_kitchener_join_keys(
            kitchener_mains=kitchener_mains,
            kitchener_breaks=kitchener_breaks,
        ),
    }


def display_dataset_summary(review_results):
    """Display clean summary outputs for the demo notebook."""
    display_cols = [
        "dataset",
        "rows",
        "columns",
        "missing_cells",
        "duplicate_rows",
    ]

    display(review_results["summary_table"][display_cols])

    print("Original Netherlands break history shape:")
    print(review_results["netherlands_breaks_shape_original"])

    print("\nKitchener join key check:")
    print(review_results["kitchener_has_join_keys"])