# data_cleaning.py
# Cleans the extracted CSV and saves population_forecasts_clean.csv

import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
IN_CSV = os.path.join(HERE, "population_forecasts.csv")
OUT_CSV = os.path.join(HERE, "population_forecasts_clean.csv")

def main():
    if not os.path.exists(IN_CSV):
        raise FileNotFoundError(
            f"Missing input: {IN_CSV}\nRun data_extraction.py first to create it."
        )

    df = pd.read_csv(IN_CSV)

    # Normalise column names 
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    # Expected from your run: geography | year | gender | age | value

    # Basic tidy 
    df = df.dropna(how="all").copy()            # drop fully empty rows
    df = df.drop_duplicates().copy()            # remove exact dupes

    # Coerce types 
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Clean strings 
    for col in ["geography", "gender", "age"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Drop rows with missing criticals (optional but sensible) 
    keep_cols = [c for c in ["geography", "year", "value"] if c in df.columns]
    if keep_cols:
        before = len(df)
        df = df.dropna(subset=[c for c in keep_cols if c in df.columns])
        print(f"Dropped {before - len(df)} rows missing {keep_cols}")

    print("Cleaned shape:", df.shape)
    print(df.head(5))

    # Save
    df.to_csv(OUT_CSV, index=False)
    print("Saved ->", OUT_CSV)

if __name__ == "__main__":
    main()
