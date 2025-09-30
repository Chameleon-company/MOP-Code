# Melbourne Public Transport (MPT) Chatbot

*A Rasa-powered assistant with real-time routing, disruptions, parking, and interactive maps for Melbourne commuters.*

---

## 1) Overview

The **MPT Chatbot** helps users compare travel modes, plan journeys, and get live service information for Melbourne. It integrates **GTFS** data, **PTV GTFS-Realtime**, **TomTom** traffic, **Google Maps** transit routing, and **City of Melbourne (Socrata)** parking feeds.  
The front end is a simple website with a floating chat widget and quick-access tiles (Train/Tram/Bus). Custom Rasa actions power the backend.

### Core Use Cases

**Smart Commute Advisor**  
Shows next departures (train/tram), nearby stops, real-time disruptions, live transit routing (encoded polyline), and **PT vs driving time** comparison.

**Park & Ride / Nearby Parking / Direct Tram**  
Finds parking near a station, suggests park-and-ride plans (parking → walk → train departures), and checks if a **direct tram** exists between two stops.

---

## 2) Features (high level)

- **Station name recognition:** Robust extraction & normalisation (keeps multi-word names; preserves **from → to** order).
- **Real-time disruptions:** PTV GTFS-Realtime alerts and basic incident surfacing.
- **Next departures:** `action_find_next_train`, `action_find_next_tram` with defensive fallbacks.
- **Nearby stops:** Geocode → Haversine distance → nearest stops within ~20 km.
- **Live transit routing + map:** Google Directions (Transit) + encoded polyline output.
- **Drive ETA:** TomTom route time with clean ETA/distance formatting.
- **Amenities:** Quick answers for lifts/ramps/toilets; targeted follow-ups when data is missing.
- **Park & Ride:** Live parking (Socrata) + walking time + train options in one reply.
- **Direct tram check:** Detects shared tram routes; lists route numbers or says none.
- **UI:** Gradient hero, quick-access tiles, responsive chat/map pane, command badges.

---

## 3) Architecture

```
client (index.html, static/js, Leaflet)
   └── chat widget  ─────────────┐
                                 │  REST webhook (Rasa)
server (Rasa core + NLU)
   └── custom actions (Python) ──┼─► GTFS (static)
                                 ├─► PTV GTFS-Realtime (alerts, vehicles)
                                 ├─► Google Maps Directions (transit polyline)
                                 ├─► TomTom (geocoding, routing, ETA)
                                 └─► City of Melbourne (Socrata parking)
```

**Key modules**
- `actions/actions.py` – action classes and orchestration  
- `actions/gtfs_utils.py` – station resolution, trips/stop_times lookup, canonical IDs  
- `actions/tomtom_utils.py` – `tt_geocode`, `tt_route`, `fmt_time`, `fmt_km`  
- `data/` – Rasa NLU examples, stories, and rules  
- `domain.yml` – intents, entities, slots, responses, action registrations  
- `static/js/` – UI interactivity (tiles, map toggles), chat helpers

---

## 4) Repository layout (key paths)

```
artificial-intelligence/chatbot/code/mpt_bot/
├── actions/
│   ├── actions.py
│   ├── gtfs_utils.py
│   └── tomtom_utils.py
├── data/
│   ├── nlu.yml
│   ├── rules.yml
│   └── stories.yml
├── domain.yml
├── index.html
├── static/
│   ├── js/
│   │   ├── components/chat.js
│   │   └── script.js
│   └── css/
│       └── style.css
├── API_SETUP_GUIDE.md
└── key.env.example   ← template only (real keys via env vars)
```

> **Do not commit real keys.** Keep `key.env` local and git-ignored.

---

## 5) Prerequisites

- **Python** 3.9+ (Rasa-compatible)  
- **Node.js** (optional; for static hosting/dev tools)  
- **Rasa** (OSS)  
- API keys (see next section)

---

## 6) Configuration & secrets

Create a local `key.env` (not checked in) with:

```
GOOGLE_API_KEY=your_google_maps_api_key
TOMTOM_API_KEY=your_tomtom_api_key
PTV_DEV_ID=your_ptv_dev_id
PTV_API_KEY=your_ptv_api_key
SOC_DATASET_APP_TOKEN=optional_if_required
```

**Loading**
- Local dev: `python-dotenv` loads `key.env` in the action server.  
- CI/Prod: use GitHub **Actions Secrets**, map to workflow `env`.

**Security**
- Restrict browser-exposed keys (referrer/IP).  
- Prefer server calls for sensitive APIs.

---

## 7) Setup & run (local)

### 7.1 Create and activate env
```bash
python -m venv .venv
# Windows: .venv\Scriptsctivate
source .venv/bin/activate
pip install -r requirements.txt  # or: pip install rasa[full] requests python-dotenv rapidfuzz
```

### 7.2 Train Rasa (if NLU/stories changed)
```bash
rasa train
```

### 7.3 Start services
```bash
# 1) Action server (loads key.env)
rasa run actions --debug

# 2) Rasa server
rasa run --enable-api --cors "*" --debug
```

### 7.4 Serve the front end
Open `index.html` with a static server (e.g., VSCode Live Server) or:
```bash
python -m http.server
```
Visit: `http://localhost:5500/index.html` (or your port).

### 7.5 Enter API keys
Add keys in `mpt_bot/key.env` **or** export as env vars (`GOOGLE_API_KEY`, `TOMTOM_API_KEY`).  
Make sure **Google Routes API** is enabled.

---

## 8) Rasa intents & custom actions (selected)

**Intents**  
`find_next_train`, `find_next_tram`, `find_nearby_stop`, `compare_two_mode`, `driving_time_by_car`, `check_amenity`, `route_map`, `parking_near_station`, `park_and_ride`, `direct_tram_check`  
Conversational: `affirm`, `goodbye`, `mood_unhappy`

**Actions**  
`action_find_next_train`, `action_find_next_tram`  
`action_generate_train_map`, `action_generate_tram_map`, `action_generate_bus_map`  
`action_compare_two_mode`, `action_driving_time_by_car`  
`action_check_feature` (amenities)  
`action_nearby_stops`  
`action_disruptions_ptv`  
`action_parking_near_station`, `action_suggest_park_and_ride_enhanced`  
`action_direct_tram_check`

---

## 9) Data & integrations

- **GTFS (static):** stops, stop_times, trips, routes → station resolution & next departures  
- **PTV GTFS-Realtime:** alerts/service updates/vehicle positions → disruptions  
- **Google Maps Directions (Transit):** live routing, encoded polyline for map  
- **TomTom:** geocoding, routing, ETA → car-time comparison & driving-time action  
- **City of Melbourne (Socrata):** live off-street parking → Park & Ride

---

## 10) Front-end

- `index.html`: hero header, quick-access tiles (Live Status, Quick Tools, Service Alerts, Weather, Accessibility), command badges, floating chat widget  
- **Leaflet map:** default markers; toggles for Bus/Train/Tram layers  
- **Chat widget:** tidy bubbles, input placeholder, basic error toasts  
- **Note:** chat→map payload standardisation is pending (current defaults work)

---

## 11) Example queries

- “Next train from South Yarra to Melbourne Central.”  
- “Any tram delays right now?”  
- “Stops near 200 Spencer Street.”  
- “Compare PT vs car from Docklands to Chadstone at 5pm.”  
- “Driving time by car from Richmond to Dandenong.”  
- “Is there parking near Flinders Street?”  
- “Park and ride from Southbank to Melbourne Central at 9am.”  
- “Is there a direct tram from Docklands to Brunswick?”
