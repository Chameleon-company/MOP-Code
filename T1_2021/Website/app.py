from config import Config
from flask import Flask, render_template, request, json, Response, redirect, flash,jsonify
from forms import Pedestrian_prediction_Form 
from datetime import datetime, timedelta
import numpy as np
import pickle
 
scaler = pickle.load(open('newscaler.pkl', 'rb'))
model = pickle.load(open('pedestriant_ml_prediction_model_rf.pkl', 'rb'))

app = Flask(__name__)
 
app.config.from_object(Config)


 
@app.route("/")
@app.route("/index")
@app.route("/home")
def index():
    return render_template("index.html", index = True)


@app.route("/forecasting")
def forecasting():
    return render_template("index.html", index = True)

	
@app.route("/prediction")
def prediction():
    return render_template("index.html", index = True)
	
@app.route("/data_analysis")
def data_analysis():
    return render_template("index.html", index = True)
	
@app.route("/Energy_forecast")
def Energy_forecast():
    return render_template("Energy_forecast.html", Energy_forecast = True)

 
@app.route("/Pedestrian_forecast")
def Pedestrian_forecast():
    return render_template("Pedestrian_forecast.html", Pedestrian_forecast = True)
 
@app.route("/Pedestrian_prediction", methods=['GET','POST'])
def Pedestrian_prediction():
    form = Pedestrian_prediction_Form()
    # 
    if form.is_submitted():
 
        independent_variables = request.form
        date_time_obj = datetime.strptime(independent_variables['date'], '%Y-%m-%d')
        X_test = [date_time_obj.timetuple().tm_wday+1,date_time_obj.month,  date_time_obj.year, date_time_obj.timetuple().tm_yday,  int(independent_variables['restriction']),  int(independent_variables['public_holiday']), float(independent_variables['rainfall']), float(independent_variables['minimum_temperature']), float(independent_variables['maximum_temperature']), float(independent_variables['solar_exposure'])]
        new_X_test = np.array(X_test)
        new_X_test_ = new_X_test.reshape(1,-1)
        X_test_scaled = scaler.transform(new_X_test_)
        prediction = model.predict(X_test_scaled)
        output = round(prediction[0])
 
        return render_template("user.html", independent_variables = independent_variables, independent_variables_year = date_time_obj.year , independent_variables_day_of_year = date_time_obj.timetuple().tm_yday, independent_variables_month = date_time_obj.month, independent_variables_week_index = date_time_obj.timetuple().tm_wday+1, prediction_text = 'The Total Expected Pedestrian for {} is {}'.format(independent_variables['date'], output) )
                    
    return render_template( "Pedestrian_prediction.html",title = "Pedestrian prediction",  form = form,  Pedestrian_prediction = True)
	
@app.route("/about_us")
def about_us():
    return render_template("about_us.html", about_us = True)
    
if __name__ == '__main__':
    app.run(host = '0.0.0.0',port = 8080)

