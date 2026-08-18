# Melbourne Solar Opportunity Mapping - Collaborative Project

## Team Members
- **Jawaharsriraam Sathya Rajendran** (Teammate 1)
- **Sujay Narayana** (Teammate 2)

## Project Overview
This project was completed collaboratively in 4 distinct phases, with each teammate contributing specialized analysis and unique insights. The work was then merged into a comprehensive final notebook.

---

## Collaborative Workflow

### 📁 Part 1: Data Loading & Exploratory Analysis
**Owner:** Jawaharsriraam Sathya Rajendran  
**File:** `Part1_Data_Loading_EDA.ipynb`

**Responsibilities:**
- Load Melbourne building footprints and solar irradiance data
- Perform initial data quality checks
- Calculate accurate roof areas using projected coordinate systems
- Join spatial and solar data

**Unique Contribution:**
- **Shape Complexity Analysis**: Analyzed building perimeter-to-area ratios to understand roof complexity and potential installation challenges
- Created visualizations showing relationship between roof area and shape complexity

**Key Outputs:**
- `buildings` GeoDataFrame with roof areas and solar data
- Shape complexity metrics for installation planning

---

### 📁 Part 2: Solar Potential & System Sizing
**Owner:** Sujay Narayana  
**File:** `Part2_Solar_Potential_SystemSizing.ipynb`

**Responsibilities:**
- Calculate usable roof area (applying reduction factors)
- Classify buildings by type (Residential/Commercial/Industrial)
- Determine optimal solar system sizes
- Estimate annual energy generation

**Unique Contribution:**
- **Panel Configuration Optimization**: Detailed analysis of optimal panel layouts using standard panel dimensions (1.7m × 1.0m)
- Calculated equipment requirements (inverters, wiring length)
- Analyzed generation efficiency per panel across different building types

**Key Outputs:**
- Usable roof area calculations
- System sizing (kW) for each building
- Annual generation estimates (kWh)
- Panel and equipment requirements

---

### 📁 Part 3: Economic & Environmental Analysis
**Owner:** Jawaharsriraam Sathya Rajendran  
**File:** `Part3_Economic_Environmental_Analysis.ipynb`

**Responsibilities:**
- Calculate installation costs with economies of scale
- Project annual savings from electricity generation
- Estimate CO2 emissions reduction
- Analyze financial viability

**Unique Contribution:**
- **Payback Period & NPV Analysis**: Comprehensive financial modeling including:
  - Payback period calculations by building type
  - Net Present Value (NPV) over 25-year system lifetime
  - Return on Investment (ROI) analysis
  - Time-series cumulative cash flow projections

**Key Outputs:**
- Installation cost estimates
- Annual savings projections
- CO2 reduction calculations
- Financial viability metrics (payback, NPV, ROI)

---

### 📁 Part 4: Opportunity Scoring & Final Analysis
**Owner:** Sujay Narayana  
**File:** `Part4_Opportunity_Scoring_Final.ipynb`

**Responsibilities:**
- Develop multi-factor opportunity scoring algorithm
- Categorize buildings (High/Medium/Low opportunity)
- Create correlation analysis
- Generate final visualizations and recommendations

**Unique Contribution:**
- **Priority Area Identification**: Geographic clustering analysis to identify:
  - Spatial clusters of high-opportunity buildings
  - Priority zones for targeted solar programs
  - Cluster-level aggregate metrics (total savings, CO2 reduction)
  - Implementation roadmap with phased rollout strategy

**Key Outputs:**
- Opportunity scores (0-100)
- Building categorization
- Priority cluster identification
- Final implementation recommendations


### What's Included in the Final Notebook
The complete notebook (`Melbourne_Solar_Project_Complete.ipynb`) contains:

1. **All Core Analysis** from Parts 1-4:
   - Data loading and EDA
   - Solar potential calculations
   - Economic and environmental analysis
   - Opportunity scoring

2. **Selected Unique Contributions**:
   - Shape complexity analysis (Part 1)
   - Panel configuration optimization insights (Part 2)
   - Financial modeling methods (Part 3)
   - Geographic clustering approach (Part 4)

3. **Streamlined Workflow**:
   - Removed duplicate data loading steps
   - Optimized visualization placement
   - Unified narrative flow

### Differences Between Individual and Final Versions

**Individual Part Files:**
- Standalone analysis focusing on specific aspects
- Include detailed exploratory work by each teammate
- Show individual problem-solving approaches
- Contain some extra/experimental analyses

**Complete Final File:**
- Integrated end-to-end analysis
- Streamlined for presentation
- Focuses on key findings and results
- Removes intermediate/exploratory steps
- Maintains the best insights from each part

---

## Key Metrics - Project Summary

### Dataset
- **Buildings Analyzed**: ~7,000+ Melbourne buildings
- **Area Coverage**: Central Melbourne region
- **Data Sources**: Building footprints + Solar irradiance data

### Solar Potential
- **Total Roof Area**: ~X million m²
- **Usable Roof Area**: ~Y million m² (after reduction factors)
- **Total System Capacity**: ~Z MW
- **Annual Generation**: ~A GWh/year

### Economic Impact
- **Total Installation Cost**: ~$X million
- **Annual Savings**: ~$Y million/year
- **Average Payback**: ~Z years
- **25-Year NPV**: ~$A million

### Environmental Impact
- **Annual CO2 Reduction**: ~X tons/year
- **Equivalent**: Y cars removed from roads

### Opportunity Distribution
- **High Opportunity**: X% of buildings
- **Medium Opportunity**: Y% of buildings
- **Low Opportunity**: Z% of buildings

---

## Files Included

1. `Part1_Data_Loading_EDA.ipynb` - Data loading and exploration
2. `Part2_Solar_Potential_SystemSizing.ipynb` - Solar calculations
3. `Part3_Economic_Environmental_Analysis.ipynb` - Financial analysis
4. `Part4_Opportunity_Scoring_Final.ipynb` - Scoring and recommendations
5. `Melbourne_Solar_Project_Complete.ipynb` - **FINAL INTEGRATED VERSION**
---

## Running the Notebooks

### Option 1: Run Individual Parts (Shows Collaboration)
```bash
# Run in order:
jupyter notebook Part1_Data_Loading_EDA.ipynb
jupyter notebook Part2_Solar_Potential_SystemSizing.ipynb
jupyter notebook Part3_Economic_Environmental_Analysis.ipynb
jupyter notebook Part4_Opportunity_Scoring_Final.ipynb
```

### Option 2: Run Final Complete Version
```bash
jupyter notebook Melbourne_Solar_Project_Complete.ipynb
```
