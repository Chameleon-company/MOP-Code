from flask import Flask, render_template, request, jsonify
import health_safety 


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
#Routes for the Vehicle Classification project pages

#Vehicle detection
@app.route('/vehicle_detect')
def vehicle_detect():
    return render_template('vehicle_detect.html')

#-------------------------
#Routes for the Traffic Analysis project pages


#-------------------------
#Routes for the Health behaviour project pages
#Health Safety
@app.route('/health_safety', methods=['GET', 'POST'])
def health_safety_app():
    return health_safety.index()

@app.route('/health_safety/predict', methods=['POST'])
def health_data_view():
    category = request.form.get('category')
    gender = request.form.get('gender')
    age_range = request.form.get('age_range')
    suburb = request.form.get('suburb')

    # Pass the data to the health_safety.predict function to handle the logic and get the response
    health_data = health_safety.predict(category, gender, age_range, suburb)

    # Return the prediction data as JSON
    return jsonify(health_data)


#-------------------------
# Running the app
if __name__ == '__main__':
    app.run(debug=True,port=5000)