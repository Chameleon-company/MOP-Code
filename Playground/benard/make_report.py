# make_report.py
# Bundles your outputs (CSV summaries + PNG charts) into a single HTML report.
# Input files expected next to this script:
#   - population_forecasts_clean.csv
#   - yoy_growth_by_year.csv
#   - top_geographies_total.csv
#   - projection_totals_next5.csv
#   - population_trend.png (from eda.py)
#   - projection_totals_next5.png (from analysis.py)
# Output:
#   - report_population_forecasts.html

import os
import base64
import pandas as pd
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(HERE, "population_forecasts_clean.csv")
YOY   = os.path.join(HERE, "yoy_growth_by_year.csv")
TOPG  = os.path.join(HERE, "top_geographies_total.csv")
PROJ  = os.path.join(HERE, "projection_totals_next5.csv")

TREND_IMG = os.path.join(HERE, "population_trend.png")
PROJ_IMG  = os.path.join(HERE, "projection_totals_next5.png")

OUT_HTML = os.path.join(HERE, "report_population_forecasts.html")

def img_to_base64(p):
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def df_to_html_table(df, max_rows=20):
    # clamp rows to keep report light
    if df is None:
        return "<p><em>Not available</em></p>"
    if len(df) > max_rows:
        df = df.head(max_rows)
    return df.to_html(index=False, border=0)

def safe_read_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None

def main():
    # Load data (gracefully handle missing files)
    df_clean = safe_read_csv(CLEAN)
    df_yoy   = safe_read_csv(YOY)
    df_topg  = safe_read_csv(TOPG)
    df_proj  = safe_read_csv(PROJ)

    trend_b64 = img_to_base64(TREND_IMG)
    proj_b64  = img_to_base64(PROJ_IMG)

    # Light stats
    totals_by_year_html = ""
    if df_clean is not None and {"year", "value"}.issubset(df_clean.columns):
        totals_by_year = (
            df_clean.groupby("year")["value"].sum().reset_index().sort_values("year")
        )
        totals_by_year_html = df_to_html_table(totals_by_year, max_rows=50)

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Population Forecasts – Summary Report</title>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; margin: 24px; color: #222; }}
    h1, h2, h3 {{ margin: 0.4em 0; }}
    .block {{ margin: 20px 0; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #eee; padding: 6px 8px; text-align: left; }}
    .muted {{ color: #666; }}
    .imgwrap {{ margin: 10px 0; text-align: center; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
    code {{ background: #f5f5f5; padding: 2px 4px; border-radius: 3px; }}
  </style>
</head>
<body>
  <h1>City of Melbourne – Population Forecasts</h1>
  <p class="muted">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>

  <div class="block">
    <h2>Overview</h2>
    <p>This report summarizes the cleaned population forecast dataset and highlights
       year-over-year growth, top geographies, and a lightweight 5-year projection.</p>
  </div>

  <div class="block">
    <h2>Totals by Year (sum of <code>value</code>)</h2>
    {totals_by_year_html or "<p><em>Not available</em></p>"}
  </div>

  <div class="block">
    <h2>Year-over-Year Growth</h2>
    {df_to_html_table(df_yoy)}
  </div>

  <div class="block">
    <h2>Top Geographies (by total <code>value</code>)</h2>
    {df_to_html_table(df_topg)}
  </div>

  <div class="block">
    <h2>Projection (Next 5 Years)</h2>
    {df_to_html_table(df_proj)}
  </div>

  <div class="block">
    <h2>Charts</h2>
    <div class="imgwrap">
      <h3>Historical Total by Year</h3>
      {("<img src='data:image/png;base64," + trend_b64 + "' />") if trend_b64 else "<p class='muted'>No chart found.</p>"}
    </div>
    <div class="imgwrap">
      <h3>5-Year Linear Projection</h3>
      {("<img src='data:image/png;base64," + proj_b64 + "' />") if proj_b64 else "<p class='muted'>No projection chart found.</p>"}
    </div>
  </div>

  <div class="block">
    <h2>Files</h2>
    <ul class="muted">
      <li>{os.path.basename(CLEAN)}</li>
      <li>{os.path.basename(YOY)}</li>
      <li>{os.path.basename(TOPG)}</li>
      <li>{os.path.basename(PROJ)}</li>
      <li>{os.path.basename(TREND_IMG)}</li>
      <li>{os.path.basename(PROJ_IMG)}</li>
    </ul>
  </div>
</body>
</html>
    """.strip()

    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print("Saved report ->", OUT_HTML)

if __name__ == "__main__":
    main()
