🧠 Mental Health Chatbot App
This is a full-stack mental health chatbot application designed to provide emotional support through a friendly chat interface. It uses React for the frontend and Flask for the backend, with integration to the Google Gemini Pro API for generating AI responses.

📁 Project Structure

mental-health-chatbot/
├── backend/ # Flask backend
│ ├── app.py # Main API server
│ ├── .env # Environment variables (ignored)
│ └── venv/ # Python virtual environment (ignored)
│
├── frontend/ # React frontend
│ ├── src/ # React components
│ ├── node_modules/ # Dependencies (ignored)
│ └── public/ # Static files
│
├── .gitignore # Files/directories to ignore in Git
└── README.md # Project documentation
🚀 Features
🤖 AI-powered mental health chatbot using Gemini Pro API

💬 Chat interface with message history and avatars

🔁 Reset conversation feature

🔒 Environment variables stored securely in .env

🌐 Cross-origin support for frontend-backend communication

🛠️ Setup Instructions

1. Backend (Flask)

cd backend
python3 -m venv venv
source venv/bin/activate # or venv\Scripts\activate on Windows
pip install -r requirements.txt
Create a .env file in the backend/ directory:

env

GOOGLE_API_KEY=your_google_api_key_here
Run the Flask server:

python app.py 2. Frontend (React)

cd frontend
npm install
npm start
Make sure the backend is running at http://localhost:5001.

🧪 Testing the App
Open your browser and go to http://localhost:3000

Start chatting with the AI assistant

Use the reset button to start a new conversation

📦 .gitignore Highlights
backend/.env: Hides API keys and secrets

backend/venv/: Avoids uploading Python virtual environment

frontend/node_modules/: Skips bulky Node.js dependencies

\*.log, .DS_Store: Ignores logs and system files
