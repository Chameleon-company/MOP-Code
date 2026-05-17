# Smart Street Lighting — React Dashboard

React + Tailwind CSS frontend for the Smart Street Lighting Design System.

## Setup

```bash
npm install
```

## Development

Requires the FastAPI backend running on port 8000:

```bash
# Terminal 1 (from project root):
uvicorn api:app --port 8000

# Terminal 2 (from this directory):
npm run dev
# Open http://localhost:3000
```

Vite proxies `/api` requests to the backend automatically.

## Build

```bash
npm run build
```

## Panels

| Panel | Description |
|-------|-------------|
| **Design Assistant** | Chat interface — describe a location, get a lighting design report |
| **Traffic Analysis** | Hourly pedestrian traffic chart with adaptive dimming schedule |
| **Efficiency** | Pie chart of city-wide lighting efficiency, top sensors table |
| **Data Explorer** | AS/NZS 1158 reference table, system architecture, key findings |

## Tech

- React 19 + Vite
- Tailwind CSS v4
- Recharts (interactive charts)
- ReactMarkdown (LLM report rendering)
- Lucide React (icons)
