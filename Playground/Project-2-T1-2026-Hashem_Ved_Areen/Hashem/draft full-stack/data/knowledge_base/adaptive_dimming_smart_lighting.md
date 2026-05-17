# Adaptive Dimming and Smart Lighting Systems

Source: Compiled from industry best practice, AS/NZS 1158 provisions, and Melbourne smart city trials.

## AS/NZS 1158 Dimming Provisions

AS/NZS 1158.3.1 acknowledges that lighting may be reduced during periods of low pedestrian activity, provided the reduced level still meets the applicable lower P-category. This is the standards basis for adaptive dimming.

### Permitted Dimming Examples
- P2 (10 lux) during peak → P3 (7 lux) during off-peak
- P3 (7 lux) during peak → P5 (3.5 lux) after midnight
- P9 (2.0 lux) during peak → P10 (1.0 lux) during very low activity
- Any category may NOT be dimmed below the minimum illuminance of the lowest applicable P-category for that location

### Dimming Schedule Design
A typical adaptive dimming schedule for Melbourne park pathways:
- Sunset to 11:00 PM: Full design illuminance (100% output)
- 11:00 PM to midnight: 80% output (reduced one P-category step)
- Midnight to 5:00 AM: 60% output (reduced two P-category steps, minimum P10)
- 5:00 AM to sunrise: 80% output (anticipating early morning users)

### Energy Savings from Dimming
- Dimming to 80% saves approximately 20% energy (not linear due to LED driver efficiency)
- Dimming to 60% saves approximately 35-40% energy
- Combined with LED baseline: total energy saving of 70-80% compared to undimmed HPS
- Annual energy cost reduction per light from dimming: approximately $15-25 AUD on top of LED savings

## Smart Lighting Technologies

### Sensor-Based Control
- PIR (Passive Infrared) motion sensors detect pedestrian presence
- Lights dim to minimum level (e.g., P10) when no pedestrians detected
- Lights brighten to full design level when motion detected
- Response time: typically 0.5-2 seconds from detection to full brightness
- Sensor range: 10-15 metres typical for pathway-mounted sensors

### Networked Smart Lighting
- Central Management System (CMS) controls all lights remotely
- Individual light addressing for group or zone control
- Real-time monitoring: energy consumption, lamp status, fault detection
- Integration with pedestrian counting data for demand-responsive dimming
- Communication protocols: DALI (Digital Addressable Lighting Interface), Zigbee, LoRaWAN, NB-IoT

### City of Melbourne Smart Lighting Trials
- Melbourne has trialled adaptive dimming with IoT sensors in several parks
- Pilot results: additional 20-30% energy saving beyond LED baseline
- Integration with existing pedestrian counting infrastructure is feasible
- 3000K warm white maintained in all smart lighting deployments for ecological compliance

## Cost Considerations for Smart Controls

### Additional Capital Cost per Light
- Basic timer-based dimming: $50-100 per light
- PIR motion sensor + controller: $150-300 per light
- Full CMS-connected smart controller: $300-500 per light
- Installation and commissioning: add $100-200 per light

### Payback for Smart Controls (on top of LED upgrade)
- Timer-based dimming: 2-3 year payback
- Motion sensor dimming: 4-6 year payback
- Full CMS: 6-10 year payback (but provides remote monitoring, fault detection, and data)

## Dark Sky and Ecological Compliance

### Melbourne Ecological Lighting Guidelines
- Maximum colour temperature: 3000K (warm white) to reduce blue light impact on nocturnal wildlife
- Shielded luminaires required: no upward light spill (full cut-off or ULOR = 0)
- Avoid illuminating tree canopy, waterways, and known wildlife corridors
- Royal Park and other ecologically sensitive areas may require additional restrictions
- Australian Dark Sky guidelines recommend amber (2200K) for areas near wildlife habitats
