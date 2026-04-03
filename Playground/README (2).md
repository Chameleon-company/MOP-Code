# 📊 Nearby Locker Coverage Visualisation

## 📖 Overview
This script analyses **locker proximity coverage** for tourist locations and produces static visualisations.  
It takes a dataset with information about whether locations have a **locker nearby** or not, then generates:

- A **donut chart** showing counts and percentages of locations by coverage.  
- A **100% stacked bar chart** showing the share of coverage.  

Both charts are saved as a **static PNG image** (`nearby_locker_coverage.png`), making them suitable for reports, GitHub repositories, or PDFs without relying on interactive Plotly outputs.

---

## 🚀 Features
- Normalises inconsistent locker labels (`1/0`, `Yes/No`, `True/False`, etc.) into stable categories:
  - **Locker Nearby**
  - **No Locker Nearby**
  - **Unknown**
- Produces two clear, publication-ready charts:
  - Donut (counts + percentages)
  - Stacked bar (share of locations)
- Saves results as `nearby_locker_coverage.png`.

---

## 🛠️ Requirements
Make sure you have the following installed:

```bash
pip install pandas numpy matplotlib
```

---

## ▶️ Usage
1. Place your dataset in a pandas DataFrame called `luggage_data` with a column named `Nearby_Locker` or `Nearby_Locker_bin`.  
2. Run the script:
   ```bash
   python nearby_locker_coverage.py
   ```
3. The file `nearby_locker_coverage.png` will be created in the same directory.

---

## 📂 Output
- **nearby_locker_coverage.png**  
  Contains two panels:
  - **Left:** Donut chart with total counts and shares.  
  - **Right:** 100% stacked bar showing percentage share of locker coverage.  

---

## 📌 Notes
- The script is **static-only**: visualisations are saved as PNGs (no Plotly or kaleido required).  
- If running in Jupyter, the image will also display inline.  
- Can be integrated with other analysis notebooks or scripts by reusing the plotting section.  
