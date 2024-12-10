from flask import Flask, render_template


# Create the Flask app
app = Flask(__name__)

#-------------------------
#Routes for the main pages

#Home page
@app.route('/')
def home():
    return render_template('home.html')

#Chatbot
@app.route('/chatbot_main')
def chatbot_main():
    return render_template('chatbot_main.html')

#Vehicle
@app.route('/vehicle_main')
def vehicle_main():
    return render_template('vehicle_main.html')

#Traffic
@app.route('/traffic_main')
def traffic_main():
    return render_template('traffic_main.html')

#Health
@app.route('/health_main')
def health_main():
    return render_template('health_main.html')

#-------------------------
#Routes for the project pages

# Vehicle Classification -> Vehicle detection
@app.route('/vehicle_detect')
def vehicle_detect():
    return render_template('vehicle_detect.html')

#-------------------------

# Running the app
if __name__ == '__main__':
    app.run(debug=True,port=5000)