#!/bin/bash

echo "🔄 Starting Flask server..."
cd "$(dirname "$0")"

# Start Flask app in background
python app.py &

# Save PID to kill later if needed
FLASK_PID=$!

# Wait a few seconds to ensure Flask starts
sleep 4

echo "🌐 Launching ngrok tunnel..."
ngrok http 5005
echo "📌 Press CTRL+C to stop the tunnel and Flask server."

# Once ngrok stops, also kill Flask
kill $FLASK_PID
