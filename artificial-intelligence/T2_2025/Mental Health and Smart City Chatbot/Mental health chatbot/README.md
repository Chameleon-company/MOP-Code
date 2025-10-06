# Gemini Health Chatbot

Empathetic mental-health support chatbot with multilingual replies, private journaling per session, a crisis-safety template, and one-click chat download. Built with Python and Flask using Google’s Gemini models. Deployable to Heroku or runnable locally.

> This app is for support and education only. It is not a substitute for professional care.

## Features

- Multilingual replies  
  Auto-detects the user’s language and responds in the same language (English, Arabic, Hindi, and more).

- Journaling mode  
  Stores user and bot messages for the current session in a local SQLite database; exposes a read-only journal feed.

- Crisis safety template  
  Detects crisis phrases and returns a supportive, non-harm message with region hints.

- Download chat  
  Backend trigger to export the current session conversation as a file.

- Simple web UI  
  `templates/index.html` + `static/app.js` served by Flask.

## Tech stack

- Backend: Python 3, Flask, SQLite
- LLM: Google Gemini
- Deployment: Heroku (Procfile + runtime.txt)
- Frontend: vanilla HTML, CSS, JavaScript

## Project layout

templates/
index.html
static/
app.js
app.py
requirements.txt
runtime.txt
Procfile


## Quick start

### 1) Prerequisites

- Python 3.10 or newer
- A Google Gemini API key

Set the environment variable:

macOS or Linux

export GEMINI_API_KEY="your_key_here"
Windows PowerShell

powershell
Copy code
$env:GEMINI_API_KEY="your_key_here"

2) Run locally

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
Open the printed localhost link in your browser.

3) Environment variables
Required

GEMINI_API_KEY

Optional

PORT (Heroku provides this automatically)

FLASK_ENV (set to development for reload and debug)

How it works
The frontend posts user text to a Flask endpoint.

app.py calls chat_once, which wraps safety checks, multilingual routing, and optional retrieval examples.

When journaling is enabled, messages and a simple mood tag are saved into SQLite.

A download endpoint returns the session transcript as a file.

A crisis detection helper short-circuits to a safety template when needed.

API endpoints
GET /
Serves the web app.

POST /api/chat
Request body:

{
  "message": "text",
  "concise": true,
  "journaling": true,
  "crisis": false
}
Response:


{
  "reply": "text",
  "mood": "mixed|sad|ok",
  "used_examples": []
}
GET /api/journal
Returns recent rows for the current session.

GET /api/download
Returns the current session conversation as a downloadable file.

Deploy to Heroku
One time


heroku login
heroku create gemini-chatbot-sarthak    # or any unique name
heroku config:set GEMINI_API_KEY="your_key_here"
git push heroku main
heroku open
Subsequent updates


git push heroku main
The repository includes Procfile and runtime.txt so Heroku’s Python buildpack will run the app on the provided port.

Privacy and data handling
Conversations are stored only in a local SQLite database when journaling is enabled.

Exported downloads are generated per session.

No analytics or third-party storage is used by default.

For shared hosting, consider a managed database (e.g., Postgres) and enforce HTTPS.

Limitations
Not medical advice or therapy.

Language detection can misclassify very short messages.

SQLite is not designed for multi-instance deployments.

Roadmap
Switch journaling to Postgres on Heroku for persistence.

Improve crisis region routing using country detection.

Richer journaling search and CSV export.

UI polish and mobile layout improvements.

Add unit tests and CI.

Development notes
Run linters or tests if added later:


pytest
ruff check .
ruff format .
License
MIT. See LICENSE.

Acknowledgements
Thanks to the open-source community and model providers for tooling and documentation.
