# Street Lighting Design Methodology

Source: Compiled from AS/NZS 1158 design guides and Australian lighting engineering practice.

## The Lighting Design Process

### Step 1: Site Classification
The designer must classify the road or area according to AS/NZS 1158:
- Determine the road hierarchy (freeway, arterial, collector, local, pedestrian)
- Assess pedestrian activity level (volume, time of day, safety risk)
- Identify special requirements (crossings, school zones, heritage areas, ecological zones)

### Step 2: Category Selection
Based on the classification:
- V-category (V1-V5) for roads where vehicular traffic is the primary concern
- P-category (P1-P12) for areas where pedestrian safety and amenity are primary
- Select the appropriate sub-category based on activity level and risk assessment

### Step 3: Determine Lighting Requirements
From the selected category, extract:
- Average maintained illuminance (lux for P-category, cd/m2 for V-category)
- Minimum illuminance
- Uniformity ratio (Emin/Eavg)
- Vertical illuminance requirements (if applicable for facial recognition)
- Colour rendering requirements

### Step 4: Luminaire Selection
Choose the luminaire based on:
- Required lumen output to achieve target illuminance at the design mounting height
- Beam distribution (Type II, III, or IV depending on road width and arrangement)
- Colour temperature: 3000K for ecologically sensitive areas, up to 4000K for high-activity areas
- IP rating: minimum IP65 for outdoor luminaires
- IK rating: minimum IK08 for vandal resistance in public areas

### Step 5: Pole Placement Design
Determine the lighting layout:
- Mounting height: typically 4-6m for pedestrian areas, 8-12m for roads
- Spacing: 3-5 times mounting height (varies by category and luminaire distribution)
- Arrangement: single-side, staggered, opposite, or central median
- Set-back from road edge: typically 0.5-1.0m behind kerb

### Step 6: Design Verification
Verify the design meets requirements:
- Calculate point-by-point illuminance across the area
- Check that average, minimum, and uniformity values meet the selected category
- Verify no dark spots or excessive glare (threshold increment)
- Apply maintenance factor (0.85-0.90 for LED)

## Luminaire Light Distribution Types

### Type II Distribution
- Suitable for: narrow pathways (up to 1.5x mounting height wide)
- Beam spread: primarily along the road, moderate across
- Typical use: park paths with poles on one side

### Type III Distribution
- Suitable for: medium roads (1.5-2.5x mounting height wide)
- Beam spread: wider lateral distribution
- Typical use: residential streets, shared paths

### Type IV Distribution
- Suitable for: wide areas (>2.5x mounting height wide)
- Beam spread: very wide, forward-throw
- Typical use: car parks, plazas, wide intersections

## Pole Arrangement Options

### Single-Side Arrangement
- Poles on one side of the pathway or road only
- Suitable for: paths up to 1.0-1.5x mounting height wide
- Most common for park pathways (P-category)
- Lower cost than staggered or opposite

### Staggered Arrangement
- Poles alternating on each side of the road
- Suitable for: roads 1.5-2.5x mounting height wide
- Better uniformity than single-side for wider roads
- Common for residential streets (P4-P6)

### Opposite Arrangement
- Poles directly opposite each other on both sides
- Suitable for: roads >2.5x mounting height wide
- Best uniformity for wide roads
- Highest cost (double the number of poles)
- Common for arterial roads (V1-V3)

### Central Median
- Poles mounted on a central median or overhead
- Suitable for: divided roads with median
- Efficient use of pole locations
- Common for dual carriageway arterials

## Maintenance Factors

The maintenance factor (MF) accounts for:
- Lamp lumen depreciation over time
- Luminaire dirt accumulation
- Surface reflectance degradation

| Technology | Initial MF | End-of-life MF |
|-----------|-----------|----------------|
| LED | 0.90 | 0.85 |
| HPS | 0.80 | 0.70 |
| Metal Halide | 0.75 | 0.65 |

Design illuminance = maintained illuminance / maintenance factor

This means LED installations can be designed closer to the target (less over-design needed), saving energy from day one.

## Glare Control

### Threshold Increment (TI)
Threshold increment measures the disability glare from luminaires:
- V1-V2: TI maximum 10%
- V3-V4: TI maximum 15%
- V5: TI maximum 20%
- P-categories: not specifically limited but luminaires should be shielded

### Upward Light Output Ratio (ULOR)
For environmental compliance and dark-sky guidelines:
- ULOR = 0%: full cut-off luminaire (no light above horizontal)
- ULOR < 5%: recommended for urban areas
- ULOR = 0%: required for ecologically sensitive areas in Melbourne

## Common Design Calculations

### Number of Lights
For a straight pathway of length L with spacing S:
- Number of lights = floor(L / S) + 1
- This places a light at position 0, S, 2S, ..., up to the end

### Illuminance Estimate
Approximate average illuminance from a single luminaire:
- E_avg = (luminaire lumens x utilization factor x maintenance factor) / (spacing x road width)
- Utilization factor depends on luminaire type and mounting height

### Energy Consumption
- Annual energy (kWh) = total watts x operating hours / 1000
- Melbourne operating hours: approximately 4,200 hours per year (dusk to dawn average)
- With dimming: multiply by average dimming ratio across the night
