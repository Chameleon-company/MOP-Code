# eda.py
# Basic exploratory analysis and plots of population forecasts

import os
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
IN_CSV = os.path.join(HERE, "population_forecasts_clean.csv")

def main():
    if not os.path.exists(IN_CSV):
        raise FileNotFoundError(
            f"Missing input: {IN_CSV}\nRun data_cleaning.py first to create it."
        )

    df = pd.read_csv(IN_CSV)

    print("EDA dataset shape:", df.shape)
    print(df.head(5))

    # --- Summary statistics ---
    print("\n--- Summary statistics ---")
    print(df.describe(include="all"))

    # --- Population trend by year ---
    if "year" in df.columns and "value" in df.columns:
        yearly = df.groupby("year")["value"].sum().reset_index()
        print("\n--- Total population by year ---")
        print(yearly.head(10))

        plt.figure(figsize=(10, 6))
        plt.plot(yearly["year"], yearly["value"], marker="o")
        plt.title("City of Melbourne: Forecasted Population by Year")
        plt.xlabel("Year")
        plt.ylabel("Population (sum)")
        plt.grid(True)
        out_path = os.path.join(HERE, "population_trend.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print("Saved yearly trend plot ->", out_path)

    # --- Population distribution by age (latest year) ---
    if {"year", "age", "value"}.issubset(df.columns):
        latest_year = df["year"].max()
        subset = df[df["year"] == latest_year].groupby("age")["value"].sum().reset_index()

        plt.figure(figsize=(12, 6))
        plt.bar(subset["age"], subset["value"])
        plt.title(f"Population Distribution by Age (Year {latest_year})")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Population")
        out_path = os.path.join(HERE, f"population_age_{latest_year}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print("Saved age distribution plot ->", out_path)

if __name__ == "__main__":
    main()
