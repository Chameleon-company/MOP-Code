# analysis.py
# Deeper insights on the City of Melbourne population forecasts:
# - Year-over-year (YoY) growth
# - Top geographies by total population (value)
# - Simple 5-year projection using a linear trend (no extra deps)
#
# Saves summary CSVs and a projection plot.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
IN_CSV = os.path.join(HERE, "population_forecasts_clean.csv")

OUT_DIR = HERE  # keep outputs next to scripts for now
YOY_CSV = os.path.join(OUT_DIR, "yoy_growth_by_year.csv")
TOP_GEO_CSV = os.path.join(OUT_DIR, "top_geographies_total.csv")
PROJ_CSV = os.path.join(OUT_DIR, "projection_totals_next5.csv")
PROJ_PNG = os.path.join(OUT_DIR, "projection_totals_next5.png")

def safe_to_int(x):
    try:
        return int(x)
    except Exception:
        return np.nan

def main():
    if not os.path.exists(IN_CSV):
        raise FileNotFoundError(
            f"Missing input: {IN_CSV}\nRun data_cleaning.py first to create it."
        )

    df = pd.read_csv(IN_CSV)
    # Expect: geography | year | gender | age | value
    # Ensure numeric
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Drop rows that can't be used
    df = df.dropna(subset=["year", "value"]).copy()

    # ---------- 1) Year-over-year growth (total) ----------
    yearly = df.groupby("year")["value"].sum().sort_index()
    yoy = yearly.diff().dropna()  # change from previous year
    yoy_df = yoy.reset_index(name="yoy_growth")
    yoy_df.to_csv(YOY_CSV, index=False)
    print(f"Saved YoY growth -> {YOY_CSV}")
    print(yoy_df.tail(10))

    # ---------- 2) Top geographies by total population ----------
    if "geography" in df.columns:
        top_geo = (
            df.groupby("geography")["value"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        top_geo.to_csv(TOP_GEO_CSV, index=False)
        print(f"Saved top geographies -> {TOP_GEO_CSV}")
        print(top_geo.head(10))
    else:
        print("Column 'geography' not found; skipping top geographies.")

    # ---------- 3) Simple 5-year projection (linear trend on totals) ----------
    # Use numpy.polyfit to fit a line: total_value = a*year + b
    years = yearly.index.values.astype(float)
    totals = yearly.values.astype(float)

    if len(years) >= 2:
        a, b = np.polyfit(years, totals, 1)  # slope, intercept
        last_year = int(np.nanmax(years))
        future_years = np.arange(last_year + 1, last_year + 6)  # next 5 years
        future_values = a * future_years + b

        proj_df = pd.DataFrame({"year": future_years, "projected_total_value": future_values})
        proj_df.to_csv(PROJ_CSV, index=False)
        print(f"Saved projection (next 5 years) -> {PROJ_CSV}")
        print(proj_df)

        # Plot historical vs. projection
        plt.figure(figsize=(10, 6))
        plt.plot(years, totals, label="Historical total (sum of 'value')", marker="o")
        plt.plot(future_years, future_values, label="Linear projection (next 5 years)", linestyle="--", marker="o")
        plt.title("City of Melbourne: Total Forecasts (Historical + 5-year Projection)")
        plt.xlabel("Year")
        plt.ylabel("Population (sum of 'value')")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(PROJ_PNG, dpi=150)
        plt.close()
        print(f"Saved projection plot -> {PROJ_PNG}")
    else:
        print("Not enough yearly data points for projection.")

if __name__ == "__main__":
    main()
