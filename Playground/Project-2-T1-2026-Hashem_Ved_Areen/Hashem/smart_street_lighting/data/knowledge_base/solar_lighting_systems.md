# Solar Street Lighting Systems — Design and Sizing

Source: Compiled from BOM data, solar industry guidelines, and Australian municipal deployment reports.

## Solar Street Light Components

A standalone solar street light system consists of:
1. **Solar panel:** Converts sunlight to electricity (typically 100-400W monocrystalline)
2. **Battery:** Stores energy for nighttime operation (typically 100-400Ah lithium iron phosphate)
3. **Charge controller:** Manages battery charging and protects from overcharge/deep discharge
4. **LED luminaire:** The light source (typically 20-80W for solar systems)
5. **Pole and mounting:** Supports all components at correct height and orientation

## Solar Sizing Methodology

### Step 1: Determine Daily Energy Requirement
- LED wattage x operating hours per night = daily energy (Wh)
- Example: 40W LED x 12 hours = 480 Wh/night
- Add 20% for system losses (controller, wiring, battery efficiency): 480 x 1.2 = 576 Wh/night

### Step 2: Determine Solar Panel Size
- Daily energy requirement / peak sun hours / panel efficiency = panel wattage
- Melbourne worst case (June): 1.61 peak sun hours, 85% panel efficiency
- Example: 576 / 1.61 / 0.85 = 421W panel required
- This is impractical for a single pole-mounted panel (typically max 300W)

### Step 3: Determine Battery Capacity
- Daily energy x days of autonomy / depth of discharge / battery voltage
- Melbourne: 3-5 days autonomy recommended (cloudy winter periods)
- Example: 576 Wh x 4 days / 0.80 DoD / 12V = 240 Ah battery
- Lithium iron phosphate (LiFePO4) recommended for longevity (2000+ cycles)

### Step 4: Check Winter Viability
- Compare daily solar generation to daily consumption for the worst month
- If solar generation < consumption, the system will have a winter deficit
- Options: oversized panel, reduced output in winter, hybrid solar/grid connection

## Melbourne Solar Viability by P-Category

| P-Category | LED Wattage | Winter Night (hrs) | Daily Need (Wh) | Panel Needed (W) | Viable? |
|-----------|------------|-------------------|-----------------|-----------------|---------|
| P10 (low) | 15-20W | 14 | 336 | 245W | Marginal |
| P9 (moderate) | 25-30W | 14 | 504 | 368W | Not viable standalone |
| P5 (low ped) | 30-40W | 14 | 672 | 491W | Not viable standalone |
| P3 (moderate ped) | 50-60W | 14 | 1,008 | 736W | Not viable standalone |

### Key Finding
Only P10 (low-use park paths with 15-20W LEDs) is marginally viable for standalone solar in Melbourne. All higher categories require grid connection or hybrid systems.

## Hybrid Solar/Grid Systems

For Melbourne, hybrid systems are the practical approach:
- Grid power as primary source
- Solar panel supplements during daylight (reduces grid draw)
- Battery provides buffer for peak demand and short outages
- Net energy saving: 30-50% of grid consumption offset by solar
- Additional cost over grid-only: $1,500-3,000 per light
- Payback on solar component: 10-15 years

## Solar Irradiance Across Melbourne Suburbs

All suburbs in metropolitan Melbourne receive similar solar irradiance (within 5% variation). The BOM station at Melbourne Olympic Park (086071) is representative:

### Seasonal Profile
- Summer (Dec-Feb): 5.6-6.3 peak sun hours — excellent solar generation
- Autumn (Mar-May): 2.1-4.4 peak sun hours — adequate
- Winter (Jun-Aug): 1.6-2.4 peak sun hours — deficit period
- Spring (Sep-Nov): 3.4-5.3 peak sun hours — adequate to good

### Tilt Angle Optimisation
- Melbourne latitude: 37.8 degrees south
- Optimal fixed tilt angle: 35-40 degrees from horizontal (facing north)
- Winter-optimised tilt: 55 degrees (increases winter generation by 15-20%)
- Pole-mounted panels are typically fixed at 30-45 degrees

## When Solar Makes Sense in Melbourne

### Good Use Cases
- Remote parkland paths far from grid (trenching cost > $200/m saved)
- Temporary or seasonal installations (avoid trenching entirely)
- Sustainability showcase projects (visible commitment to renewables)
- Low-wattage bollard lighting (5-15W) for garden paths

### Poor Use Cases
- Any pathway requiring P3 or higher category (too much power needed)
- Locations with significant shading from trees or buildings
- Areas requiring guaranteed all-night full-output operation
- High-vandalism areas (solar panels are attractive targets)

## Cost Comparison: Solar vs Grid-Connected LED

| Item | Grid-Connected LED | Standalone Solar LED | Hybrid Solar/Grid |
|------|-------------------|---------------------|-------------------|
| Luminaire | $800-1,800 | $800-1,800 | $800-1,800 |
| Pole + installation | $2,000-4,000 | $2,500-5,000 | $2,500-5,000 |
| Solar panel + controller | N/A | $800-2,000 | $600-1,500 |
| Battery | N/A | $500-1,500 | $400-1,000 |
| Grid connection + trenching | $1,500-5,000 | N/A | $1,500-5,000 |
| **Total installed** | **$4,300-10,800** | **$4,600-10,300** | **$5,800-13,300** |
| Annual energy cost | $50-150 | $0 | $25-75 |
| Battery replacement (every 8-10 yrs) | N/A | $500-1,500 | $400-1,000 |

**Conclusion:** Standalone solar is only cost-competitive when grid trenching costs exceed $3,000 (i.e., for remote locations > 15m from existing power).
