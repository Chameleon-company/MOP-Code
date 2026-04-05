# Energy Efficiency Context for Melbourne Street Lighting

## Melbourne Electricity Rates (2024-25)

- Residential average: $0.22-0.28/kWh (varies by retailer)
- Commercial/government average: $0.18-0.22/kWh
- Street lighting tariff (council): ~$0.20/kWh (used in calculations)
- Feed-in tariff (solar): $0.05-0.08/kWh
- Peak demand surcharge: additional $0.05-0.10/kWh during 3pm-9pm summer

## LED vs HPS vs Solar LED Comparison

### High Pressure Sodium (HPS) — Legacy Technology
| Parameter | 70W HPS | 150W HPS | 250W HPS |
|---|---|---|---|
| Luminous efficacy | 80-90 lm/W | 95-110 lm/W | 100-115 lm/W |
| Colour temperature | 2000-2200K | 2000-2200K | 2000-2200K |
| CRI | 20-25 | 20-25 | 20-25 |
| Lifespan | 12,000-16,000h | 16,000-20,000h | 20,000-24,000h |
| Maintenance cost | $60-80/year | $70-90/year | $80-100/year |
| Warm-up time | 5-10 minutes | 5-10 minutes | 5-10 minutes |
| Dimming capability | Limited (40-100%) | Limited (40-100%) | Limited (40-100%) |

### LED — Current Standard
| Parameter | 30W LED | 60W LED | 100W LED | 150W LED |
|---|---|---|---|---|
| Luminous efficacy | 150 lm/W | 150 lm/W | 150 lm/W | 150 lm/W |
| Output (lumens) | 4,500 | 9,000 | 15,000 | 22,500 |
| Colour temperature | 3000K | 3000K | 3000K | 3000K |
| CRI | 70+ | 70+ | 70+ | 70+ |
| Lifespan | 50,000-100,000h | 50,000-100,000h | 50,000-100,000h | 50,000-100,000h |
| Maintenance cost | $10-15/year | $10-15/year | $15-20/year | $15-20/year |
| Warm-up time | Instant | Instant | Instant | Instant |
| Dimming capability | Full (0-100%) | Full (0-100%) | Full (0-100%) | Full (0-100%) |

### Solar LED — Off-Grid Option
| Parameter | 20W Solar | 40W Solar | 60W Solar |
|---|---|---|---|
| Panel size | 80W | 160W | 240W |
| Battery capacity | 120Wh | 240Wh | 360Wh |
| Autonomy (cloudy days) | 3-4 | 3-4 | 3-4 |
| Capital cost | $4,000-5,000 | $6,000-8,000 | $8,000-12,000 |
| Running cost | $0 electricity | $0 electricity | $0 electricity |
| Maintenance cost | $50-80/year | $50-80/year | $60-100/year |
| Battery replacement | Every 5-7 years | Every 5-7 years | Every 5-7 years |
| Best for | Remote paths, P10 | Park paths, P9 | Shared paths, P5 |

## CO2 Emission Factors (Victoria)

Source: National Greenhouse Accounts Factors (2024), Department of Climate Change

| Factor | Value | Unit |
|---|---|---|
| Scope 2 (electricity generation) | 0.96 | kg CO2-e/kWh |
| Scope 3 (transmission losses) | 0.12 | kg CO2-e/kWh |
| Total (Scope 2 + 3) | 1.08 | kg CO2-e/kWh |

Victoria's grid is more carbon-intensive than the national average due to brown coal generation. This makes energy efficiency gains more impactful for CO2 reduction.

## Melbourne Solar Resource

- Average daily solar insolation: 4.2 kWh/m² (annual average)
- Winter minimum: 2.0 kWh/m² (June)
- Summer maximum: 6.5 kWh/m² (January)
- Peak sun hours for panel sizing: 3.5h (conservative design)
- Bureau of Meteorology station: Melbourne Regional Office (086071)

## Adaptive Dimming Energy Savings

Typical energy savings from dimming schedules based on pedestrian traffic:

| Period | Traffic Level | Dimming Level | Energy Factor |
|---|---|---|---|
| 5pm-9pm | Peak | 100% | 1.00 |
| 9pm-11pm | Moderate | 80% | 0.80 |
| 11pm-1am | Low | 60% | 0.60 |
| 1am-5am | Very low | 40% | 0.40 |
| 5am-7am | Rising | 80% | 0.80 |

Typical annual saving: 25-35% of baseline energy consumption when dimming is applied.

## Luminaire Specifications — Common Melbourne Models

### Philips Luma Gen2 (widely used by City of Melbourne)
- Available in 30W, 50W, 80W, 120W configurations
- IP66 rated, IK08 impact resistance
- 3000K or 4000K CCT options
- DALI dimmable, Zhaga socket compatible
- Expected lifespan: 100,000 hours at L80

### Sylvania Urban LED
- 30W to 150W range
- Asymmetric light distribution for pathway applications
- Tool-less maintenance access
- Compatible with smart city control systems

## References
- Victorian Default Offer (Essential Services Commission, 2024)
- National Greenhouse Accounts Factors (Department of Climate Change, 2024)
- Bureau of Meteorology — Melbourne climate statistics
- IES TM-15-20 — Luminaire Classification System for Outdoor Luminaires
