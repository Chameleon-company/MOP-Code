# data_extraction.py
# Downloads the population forecasts dataset from City of Melbourne.
# Writes population_forecasts.csv 

import os
import requests
import pandas as pd
from io import StringIO

DATASET_ID = "city-of-melbourne-population-forecasts-by-small-area-2020-2040"
BASE_URL = "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets"

def _read_csv_any_delim(text: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(StringIO(text), delimiter=';')
        if df.shape[1] == 1:  
            return pd.read_csv(StringIO(text))
        return df
    except Exception:
        return pd.read_csv(StringIO(text))

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_csv = os.path.join(here, "population_forecasts.csv")

    url = f"{BASE_URL}/{DATASET_ID}/exports/csv"
    params = {"select": "*", "limit": -1, "lang": "en", "timezone": "UTC"}

    print("Fetching:", url)
    r = requests.get(url, params=params, timeout=120)
    print("HTTP", r.status_code)
    r.raise_for_status()

    text = r.content.decode("utf-8", errors="replace")
    df = _read_csv_any_delim(text)

    print("Rows:", len(df), "| Cols:", len(df.columns))
    print(df.head(3))

    df.to_csv(out_csv, index=False)
    print("Saved ->", out_csv)

if __name__ == "__main__":
    main()
