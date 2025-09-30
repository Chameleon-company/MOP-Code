# Smart Luggage Management for Tourist Mobility — Melbourne
Optimising where and when to provide luggage lockers so visitors can move easily through the city.

## Overview
This project analyses visitor footfall and locker proximity to identify coverage gaps and prioritise new locker placements across Melbourne. The workflow combines data wrangling, visualisation, clustering and a lightweight predictive model — all delivered via a single Jupyter notebook.

**Key outcomes**
- When visitor pressure is highest across the day (time‑window visualisations).
- Which locations drive the most demand (Top‑10 hotspots).
- Where lockers are missing relative to footfall (coverage & proximity analysis).
- Which sites are *far* from lockers **and** busy (clustering by effort vs demand).
- Clear, executive‑grade charts for reports and presentations.

## Repository structure
```
.
├─ UC00205_Smart_Luggage_Management_for_Tourist_Mobility.ipynb  # main analysis notebook
├─ data/                                                        # your input files (not tracked here)
└─ exports/                                                     # optional: figures/HTML exported from the notebook
```
> If you’re working only in the notebook, you don’t need to create the folders above — they’re suggestions for organisation.

## Getting started
### 1) Requirements
- Python 3.9+
- JupyterLab or Jupyter Notebook
- Packages: `pandas`, `numpy`, `plotly`, `scikit-learn`, `folium` (optional for maps), `kaleido` (optional for PNG export).

**Quick install**
```bash
pip install pandas numpy plotly scikit-learn folium kaleido
```

### 2) Data
The notebook expects one row per **location** with the following columns (rename to match if yours differ):

| Column | Type | Description |
|---|---|---|
| `Location` | string | Name of the site / hotspot. |
| `Estimated_Footfall_10_12` | number | Footfall estimate for 10–12. |
| `Estimated_Footfall_12_2`  | number | Footfall estimate for 12–2. |
| `Estimated_Footfall_2_4`   | number | Footfall estimate for 2–4. |
| `Estimated_Footfall_4_6`   | number | Footfall estimate for 4–6. |
| `Footfall_Total` | number | *(Derived)* Sum of the four time windows. |
| `Footfall_Peak`  | number | *(Derived)* Max of the four time windows. |
| `Nearby_Locker`  | bool/flag | `Yes/No` or `1/0` indicating whether a locker is nearby. |
| `Travel_Distance_km` | number | Distance to nearest locker (km). |
| `Type` | category | Site type (e.g., Transport Hub, Museum). |
| `Latitude`, `Longitude` | number | Optional, for maps. |
| `Locker_Capacity` | number | Optional, installed capacity. |
| `Predicted_Demand` | number | Optional, modelled demand. |

> **Important:** there’s a common typo in some drafts: `Estimated_FootfalSl_4_6`. Make sure it is spelt **`Estimated_Footfall_4_6`**.

### 3) Run the notebook
1. Open `UC00205_Smart_Luggage_Management_for_Tourist_Mobility.ipynb`.
2. Set your data path in the **Load Data** cell.
3. Run top‑to‑bottom. Each section contains a short tutorial paragraph explaining: *what the code does, why it’s needed, and how to interpret the output.*

### 4) Exporting figures (optional)
Add after any Plotly figure:
```python
# PNG (requires kaleido)
fig.write_image("exports/fig_name.png", scale=2)
# Interactive HTML
fig.write_html("exports/fig_name.html", include_plotlyjs="cdn")
```

## What the notebook contains
- **Footfall by Time Window** — totals and share (%) for 10–12, 12–2, 2–4, 4–6.  
  *Insight:* plan staffing/turnover just before the peak windows.
- **Top Hotspots by Total Footfall** — ranked bar chart with value labels and Top‑3 highlight.  
  *Insight:* the locations that matter most for capacity decisions.
- **Locker Proximity vs Footfall** — two‑panel view: absolute counts and % share with/without nearby lockers.  
  *Insight:* quantifies the coverage gap.
- **Peak vs Total Footfall** — collision‑free scatter; labels only the key hotspots; optional “priority zone”.  
  *Insight:* spots congestion‑prone sites (spiky demand).
- **Clustering — Demand vs Effort (K‑Means)** — groups sites by `Footfall_Total` and `Travel_Distance_km`; centroids reported in **original units**.  
  *Insight:* *far + busy* locations are first candidates for new lockers.
- **Bipartite Layout (Locations → Lockers)** — clean, non‑overlapping network linking each location to its nearest locker, with hover distances and longest‑edge callouts.  
  *Insight:* makes travel burden and locker hubs visually obvious.

## Reproducibility & configuration
- Random seeds are fixed where relevant (e.g., `random_state=7` for clustering).  
- K‑Means is configured with `n_init=25` to avoid warnings and improve stability.  
- All charts use a consistent, accessible palette (blue = coverage, orange/red = gap; green = lockers).  
- If your column names differ, edit the small mapping dicts near the top of each section.

## Interpreting results (cheat‑sheet)
- **Midday dominance** → schedule locker turnover/cleaning **before** lunch peaks.  
- **Top‑10 hotspots** → primary rollout list; cross‑check with proximity and deficit.  
- **Coverage gap large** → deploy lockers where footfall is high and lockers are not nearby.  
- **Peak vs total** → high ratio indicates short‑term pressure; consider temporary capacity or pricing.  
- **Clusters** → *high footfall + long travel* = immediate install; *high footfall + short travel* = manage capacity; *low footfall* = monitor.  
- **Bipartite edges** → long edges flag underserved sites; lockers with many incoming edges are hubs (watch queuing).

## Troubleshooting
- **Charts look empty or labels overlap:** ensure columns are present and numeric; the notebook wraps/shortens labels and positions text to avoid collisions.  
- **Totals/peaks look wrong:** double‑check the 4 time‑window column names and fix the `4_6` typo.  
- **K‑Means warning:** make sure `n_init` is set; try a different `K` if clusters look unbalanced.  
- **Distances missing:** compute `Travel_Distance_km` (nearest locker) before running the bipartite section.

## Data & ethics
- City of Melbourne open data is used under its respective licence. Always credit original sources.  
- Do not expose personally identifiable information; this analysis works on **aggregated** footfall only.

## Licence
Code in this repository is released under the **MIT Licence**. Data remains subject to its original terms.

## Acknowledgements
- City of Melbourne Open Data.  
- Team contributors for dataset preparation, validation and visual design.
