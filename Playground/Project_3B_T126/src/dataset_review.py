import pandas as pd
import numpy as np


def summarise_dataframe(df, name):
    """Return a simple summary of one dataset."""
    return {
        "dataset": name,
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "column_names": list(df.columns),
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

    melbourne_mains = pd.read_csv(melbourne_mains_path)
    melbourne_mains = melbourne_mains.loc[
        :, ~melbourne_mains.columns.str.contains("^Unnamed")
    ]

    melbourne_soil = pd.read_csv(melbourne_soil_path)

    netherlands_mains = pd.read_pickle(netherlands_mains_path)
    if "unit_ID" in netherlands_mains.columns:
        netherlands_mains = netherlands_mains.drop(columns="unit_ID")

    netherlands_breaks = np.load(netherlands_breaks_path)
    netherlands_breaks_2d = netherlands_breaks.reshape(netherlands_breaks.shape[0], -1)
    netherlands_breaks_df = pd.DataFrame(netherlands_breaks_2d)

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

    return {
        "summary_table": pd.DataFrame(summaries),
        "netherlands_breaks_shape_original": netherlands_breaks.shape,
        "kitchener_has_join_keys": {
            "mains_key": "WATMAINID" in kitchener_mains.columns,
            "breaks_key": "Related Asset ID" in kitchener_breaks.columns,
        },
    }


def display_dataset_summary(review_results):
    """Display clean summary outputs for the demo notebook."""
    display(review_results["summary_table"][["dataset", "rows", "columns", "missing_cells", "duplicate_rows"]])

    print("Original Netherlands break history shape:")
    print(review_results["netherlands_breaks_shape_original"])

    print("\nKitchener join key check:")
    print(review_results["kitchener_has_join_keys"])