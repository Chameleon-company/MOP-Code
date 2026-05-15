import pandas as pd
import numpy as np


RAW_PATH = "data/raw/Melbourne/Water_Supply_Main_Pipelines.csv"
OUTPUT_PATH = "data/processed/melbourne_cleaned_for_adaptation.csv"


# Columns selected from the raw Melbourne dataset
RAW_COLUMNS = [
    "ASSET_ID",
    "COMPKEY",
    "UNITID",
    "UNITID2",
    "MAIN_LINE_TYPE",
    "MAIN_CLASS",
    "MAIN_NAME",
    "MATERIAL",
    "PIPE_LENGTH",
    "PIPE_WIDTH",
    "DATE_RELINED",
    "DATE_OF_CONSTRUCTION",
    "FIELD_TEAM",
    "SERVICE_STATUS",
    "COMMENTS",
    "DATE_LAST_UPDATED",
]


# Final columns kept for Melbourne finding adaptation / risk scoring
TARGET_COLUMNS = [
    "ASSET_ID",
    "COMPKEY",
    "UNITID",
    "UNITID2",
    "MAIN_NAME",
    "MAIN_LINE_TYPE",
    "MAIN_CLASS",
    "MATERIAL",
    "PIPE_LENGTH",
    "PIPE_WIDTH",
    "DATE_OF_CONSTRUCTION",
    "PIPE_AGE",
    "MISSING_CONSTRUCTION_DATE",
    "DATE_RELINED",
    "HAS_RELINED",
    "YEARS_SINCE_RELINED",
    "MISSING_RELINED_DATE",
    "INVALID_PIPE_LENGTH",
    "INVALID_PIPE_WIDTH",
    "FUTURE_RELINED_DATE",
    "FIELD_TEAM",
    "SERVICE_STATUS",
    "COMMENTS",
    "DATE_LAST_UPDATED",
]


def load_raw_melbourne_data(raw_path: str = RAW_PATH) -> pd.DataFrame:
    """Load raw Melbourne water main data."""
    return pd.read_csv(raw_path)


def preprocess_melbourne_data(
    df: pd.DataFrame,
    current_year: int = 2026,
) -> pd.DataFrame:
    """
    Prepare Melbourne water main data for finding adaptation.

    This does not create a supervised learning target.
    It creates a cleaned dataset with transferable risk features from the
    Kitchener model findings, ready for Melbourne risk scoring.
    """
    # Keep only useful raw columns that are available
    available_raw_cols = [col for col in RAW_COLUMNS if col in df.columns]
    melbourne = df[available_raw_cols].copy()

    # Standardise text fields
    text_cols = melbourne.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        melbourne[col] = (
            melbourne[col]
            .astype("string")
            .str.strip()
            .str.upper()
        )

    # Convert date fields
    for col in ["DATE_OF_CONSTRUCTION", "DATE_RELINED", "DATE_LAST_UPDATED"]:
        if col in melbourne.columns:
            melbourne[col] = pd.to_datetime(melbourne[col], errors="coerce")

    # Engineer pipe age from construction date
    melbourne["PIPE_AGE"] = current_year - melbourne["DATE_OF_CONSTRUCTION"].dt.year

    # Engineer relining features
    melbourne["HAS_RELINED"] = melbourne["DATE_RELINED"].notna().astype(int)
    melbourne["YEARS_SINCE_RELINED"] = current_year - melbourne["DATE_RELINED"].dt.year

    # Missing-value flags
    melbourne["MISSING_CONSTRUCTION_DATE"] = (
        melbourne["DATE_OF_CONSTRUCTION"].isna().astype(int)
    )
    melbourne["MISSING_RELINED_DATE"] = (
        melbourne["DATE_RELINED"].isna().astype(int)
    )

    # Invalid-value flags
    melbourne["INVALID_PIPE_LENGTH"] = (melbourne["PIPE_LENGTH"] <= 0).astype(int)
    melbourne["INVALID_PIPE_WIDTH"] = (melbourne["PIPE_WIDTH"] <= 0).astype(int)
    melbourne["FUTURE_RELINED_DATE"] = (
        melbourne["YEARS_SINCE_RELINED"] < 0
    ).astype(int)

    # Replace invalid values with missing before filling
    melbourne.loc[melbourne["PIPE_LENGTH"] <= 0, "PIPE_LENGTH"] = np.nan
    melbourne.loc[melbourne["PIPE_WIDTH"] <= 0, "PIPE_WIDTH"] = np.nan
    melbourne.loc[
        melbourne["YEARS_SINCE_RELINED"] < 0,
        "YEARS_SINCE_RELINED",
    ] = np.nan

    # Fill categorical values needed for later scoring / dashboard / LLM context
    categorical_cols = [
        "ASSET_ID",
        "UNITID",
        "UNITID2",
        "MAIN_LINE_TYPE",
        "MAIN_CLASS",
        "MAIN_NAME",
        "MATERIAL",
        "FIELD_TEAM",
        "SERVICE_STATUS",
    ]

    for col in categorical_cols:
        if col in melbourne.columns:
            melbourne[col] = melbourne[col].fillna("UNKNOWN")

    # Fill physical numeric values after invalid values have been flagged
    melbourne["PIPE_LENGTH"] = melbourne["PIPE_LENGTH"].fillna(
        melbourne["PIPE_LENGTH"].median()
    )
    melbourne["PIPE_WIDTH"] = melbourne["PIPE_WIDTH"].fillna(
        melbourne["PIPE_WIDTH"].median()
    )

    # Do not impute PIPE_AGE.
    # Missing age is preserved and identified using MISSING_CONSTRUCTION_DATE.

    # Missing relining date means no confirmed relining record.
    # Use 0 for scoring support and keep HAS_RELINED / MISSING_RELINED_DATE as context.
    melbourne["YEARS_SINCE_RELINED"] = melbourne["YEARS_SINCE_RELINED"].fillna(0)

    # Comments are context only
    if "COMMENTS" in melbourne.columns:
        melbourne["COMMENTS"] = melbourne["COMMENTS"].fillna("NO COMMENT")

    # Keep target output columns
    available_target_cols = [col for col in TARGET_COLUMNS if col in melbourne.columns]

    melbourne_cleaned = (
        melbourne[available_target_cols]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return melbourne_cleaned


def save_melbourne_cleaned_data(
    df: pd.DataFrame,
    output_path: str = OUTPUT_PATH,
) -> None:
    """Save cleaned Melbourne dataset for finding adaptation."""
    df.to_csv(output_path, index=False)


def build_melbourne_cleaned_dataset(
    raw_path: str = RAW_PATH,
    output_path: str = OUTPUT_PATH,
    current_year: int = 2026,
) -> pd.DataFrame:
    raw_df = load_raw_melbourne_data(raw_path)
    cleaned_df = preprocess_melbourne_data(raw_df, current_year=current_year)
    save_melbourne_cleaned_data(cleaned_df, output_path)
    return cleaned_df