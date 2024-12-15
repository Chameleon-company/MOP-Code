import numpy as np
from flask import Flask, request, jsonify, render_template, json, Response, redirect, flash
import pickle
from config import Config
from forms import Pedestrian_prediction_Form 
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)
app.config.from_object(Config)
#model = pickle.load(open('model.pkl', 'rb'))

@app.route("/")
@app.route("/index")
@app.route("/home")
def index():
    return render_template("index.html", index = True)

@app.route("/Pedestrian_forecast")
def Pedestrian_forecast():
    return render_template("Pedestrian_forecast.html", Pedestrian_forecast = True)

@app.route("/Pedestrian_prediction", methods=['GET','POST'])
def Pedestrian_prediction():
    form = Pedestrian_prediction_Form()

    #using standard scaler

    #scaler = StandardScaler()

    if form.is_submitted():

        return render_template("user.html", result = request.form)
            
        date = form.date.data
        year = date.dt.year
        day_of_year1 = date.dt.dayofyear
        monthly_index = date.dt.month
        day_of_week = date.dt.dayofweek + 1
        rainfall = form.rainfall.data
        solar_exposure = form.solar_exposure.data
        restriction = form.restriction.data
        public_holiday = form.public_holiday.data
        minimum_temperature = form.minimum_temperature.data
        maximum_temperature = form.maximum_temperature.data

        
    return render_template( "Pedestrian_prediction.html",title = "Pedestrian prediction",  form = form,  Pedestrian_prediction = True)


@app.route("/contributions")
def contributions():

    return render_template("contributions.html", contributions = True)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port = 8080)