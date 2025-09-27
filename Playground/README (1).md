# UC00205 — Smart Luggage Management for Tourist Mobility

## 📌 Overview
This use case explores how data-driven analysis can provide practical insights into **tourist mobility challenges in Melbourne**, with a focus on **luggage management**.  
Tourists often face difficulties carrying luggage after checking out of their accommodation, which limits their ability to explore attractions or spend money before their departure.  

By combining **Reddit-derived observations** with **simulated urban datasets**, this notebook analyses footfall patterns, locker accessibility, and transport proximity to identify where luggage storage would have the greatest impact.  

---

## 🎯 Objectives
- Analyse **footfall patterns** across tourist hotspots and transport hubs.  
- Identify **coverage gaps** where no nearby locker facilities exist.  
- Use **clustering** (KMeans) to group sites by demand and accessibility.  
- Build a **bipartite graph** showing connections between tourist locations and lockers.  
- Provide **actionable insights** to improve tourist experience and reduce congestion.  

---

## 🗂️ Dataset Used
- **Reddit discussions** (exported into Excel/CSV format; no API/authentication required).  
- **Simulated footfall dataset**: includes tourist locations, estimated footfall across time windows, and nearest locker distances.  
- Key columns:  
  - `Location` — tourist hotspot or transport hub.  
  - `Estimated_Footfall_*` — footfall counts for time windows (10–12, 12–2, 2–4, 4–6).  
  - `Footfall_Total`, `Footfall_Peak` — derived metrics.  
  - `Nearby_Locker` — Yes/No indicator.  
  - `Travel_Distance_km` — distance to nearest locker.  

---

## ⚙️ How to Run
1. Clone or download this repository.  
2. Open the notebook `UC00205_Smart_Luggage_Management_for_Tourist_Mobility.ipynb` in **Jupyter Notebook** or **JupyterLab**.  
3. Install required libraries (if not already available):  
   ```bash
   pip install pandas plotly folium scikit-learn networkx
   ```  
4. Run each cell in sequence.  
   - Ensure the dataset path is updated to your local machine.  
   - Visualisations require `plotly` and will render in the notebook/HTML export.  

---

## 📊 Features & Visualisations
- **Time-of-day footfall analysis** → identifies peak congestion times.  
- **Top hotspot ranking** → highlights busiest attractions by total footfall.  
- **Locker coverage visualisation** → compares Yes/No locker availability.  
- **Scatter plot (Peak vs Total)** → short labels used for clarity.  
- **Clustering (KMeans)** → groups sites by demand and travel distance.  
- **Bipartite graph (NetworkX)** → shows travel burden between locations and lockers.  

---

## 🔑 Insights & Findings
- Afternoon hours show the **highest traveller congestion**.  
- **Southern Cross Station, Flinders Street Station, Federation Square, and Queen Victoria Market** are priority hotspots.  
- Several high-demand sites **lack nearby lockers**, revealing service gaps.  
- Clustering highlighted distinct groups of sites needing **different locker strategies**.  
- The bipartite graph clearly showed **long-distance burdens** for some attractions.  

---

## ✅ Conclusion
This project demonstrates how even simple datasets, when combined with clear **EDA, ML models, and visualisations**, can lead to meaningful urban planning recommendations.  
By strategically placing lockers near Melbourne’s busiest hotspots and transport hubs, the city can improve mobility, enhance tourist satisfaction, and reduce congestion — benefiting both visitors and local businesses.  
