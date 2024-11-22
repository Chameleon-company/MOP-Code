from config import Config
from flask import Flask, render_template, request, json, Response, redirect, flash,jsonify
from forms import Pedestrian_prediction_Form 

from datetime import datetime, timedelta
import numpy as np
import plotly.express as px
import pandas as pd
from pandas import DataFrame as df, Series as se
import pickle
import csv
 
scaler = pickle.load(open('newscaler.pkl', 'rb'))
model = pickle.load(open('pedestriant_ml_prediction_model_rf.pkl', 'rb'))


app = Flask(__name__)
 
app.config.from_object(Config)



 
@app.route("/")

@app.route("/index")
@app.route("/home")
def index():
    return render_template("index.html", index = True)

@app.route("/Predictor_Variable")
def Predictor_Variable():
    return render_template("index.html", index = True)
	
@app.route("/prediction")
def prediction():
    return render_template("index.html", index = True)
	
@app.route("/Pedestrian_Trend")
def Pedestrian_Trend():
    return render_template("index.html", index = True)



	
@app.route("/Energy_forecast")
def Energy_forecast():
    return render_template("Energy_forecast.html", Energy_forecast = True)

	
@app.route("/Bourke_Street_Mall_South")
def Bourke_Street_Mall_South(chartID = 'chart_ID', chart_type = 'scatter', chart_height = 800):

	with open('Bourke_Street_Mall_South.csv') as csv_file:
		data = csv.reader(csv_file, delimiter=',')
		first_line = True
		
		places = []
		
		for row in data:
		
			if not first_line:
			
				places.append({"date": row[1], "daily_count": int(row[4])})
				
			else:
				first_line = False
				
	chart = {"renderTo": chartID, "type": chart_type, "height": chart_height}
	series = [{"name": 'Pedestrian Count', "color" : '#4572A7', "data": [d['daily_count'] for d in places]}]
	title = {"text": 'Bourke Street Mall South Daily Pedestrian Count from January 2015 to Febuary 2021'}
	xAxis = {"categories":  [d['date'] for d in places], "tickInterval": 90}
	yAxis = {"title": {"text": 'Daily Pedestrian Count'}}
	
	return render_template("Bourke_Street_Mall_South.html", Bourke_Street_Mall_South = True, chartID=chartID, chart=chart, series=series, title=title, xAxis=xAxis, yAxis=yAxis)
	
@app.route("/Victoria_Point")
def Victoria_Point(chartID = 'chart_ID', chart_type = 'scatter', chart_height = 800):

	
	with open('Victoria_Point.csv') as csv_file:
		data = csv.reader(csv_file, delimiter=',')
		first_line = True
		places = []
		for row in data:
			if not first_line:
				places.append({"date": row[1], "daily_count": int(row[4])})
			else:
				first_line = False
		
	chart = {"renderTo": chartID, "type": chart_type, "height": chart_height,}
	series = [{"name": 'Pedestrian Count', "data": [d['daily_count'] for d in places]}]
	title = {"text": 'Victoria Point Daily Pedestrian Count from January 2015 to Febuary 2021'}
	xAxis = {"categories":  [d['date'] for d in places], "tickInterval": 90}
	yAxis = {"title": {"text": 'Daily Pedestrian Count'}}
	
	return render_template('Victoria_Point.html', Victoria_Point = True, chartID=chartID, chart=chart, series=series, title=title, xAxis=xAxis, yAxis=yAxis)

	
@app.route("/Collins_Place_North")
def Collins_Place_North(chartID = 'chart_ID', chart_type = 'scatter', chart_height = 800):

	with open('Collins_Place_North.csv') as csv_file:
		data = csv.reader(csv_file, delimiter=',')
		first_line = True
		places = []
		for row in data:
			if not first_line:
				places.append({"date": row[1], "daily_count": int(row[4])})
			else:
				first_line = False
		
	chart = {"renderTo": chartID, "type": chart_type, "height": chart_height,}
	series = [{"name": 'Pedestrian Count', "data": [d['daily_count'] for d in places]}]
	title = {"text": 'Collins Place North Daily Pedestrian Count from January 2015 to Febuary 2021'}
	xAxis = {"categories":  [d['date'] for d in places], "tickInterval": 90}
	yAxis = {"title": {"text": 'Daily Pedestrian Count'}}
	
	return render_template("Collins_Place_North.html", Collins_Place_North = True, chartID=chartID, chart=chart, series=series, title=title, xAxis=xAxis, yAxis=yAxis)
	
@app.route("/Flinders_St_Spark_La")
def Flinders_St_Spark_La(chartID = 'chart_ID', chart_type = 'scatter', chart_height = 800):

	with open('Flinders_St_Spark_La.csv') as csv_file:
		data = csv.reader(csv_file, delimiter=',')
		first_line = True
		places = []
		for row in data:
			if not first_line:
				places.append({"date": row[1], "daily_count": int(row[4])})
			else:
				first_line = False
		
	chart = {"renderTo": chartID, "type": chart_type, "height": chart_height,}
	series = [{"name": 'Pedestrian Count', "data": [d['daily_count'] for d in places]}]
	title = {"text": 'Flinders_St_Spark_La Daily Pedestrian Count from January 2015 to Febuary 2021'}
	xAxis = {"categories":  [d['date'] for d in places], "tickInterval": 90}
	yAxis = {"title": {"text": 'Daily Pedestrian Count'}}
	
	return render_template("Flinders_St_Spark_La.html", Flinders_St_Spark_La = True, chartID=chartID, chart=chart, series=series, title=title, xAxis=xAxis, yAxis=yAxis)

	
@app.route("/Southern_Cross_Station")
def Southern_Cross_Station(chartID = 'chart_ID', chart_type = 'scatter', chart_height = 800):

	with open('Southern_Cross_Station.csv') as csv_file:
		data = csv.reader(csv_file, delimiter=',')
		first_line = True
		places = []
		for row in data:
			if not first_line:
				places.append({"date": row[1], "daily_count": int(row[4])})
			else:
				first_line = False
		
	chart = {"renderTo": chartID, "type": chart_type, "height": chart_height,}
	series = [{"name": 'Pedestrian Count', "data": [d['daily_count'] for d in places]}]
	title = {"text": 'Southern Cross Station Daily Pedestrian Count from January 2015 to Febuary 2021'}
	xAxis = {"categories":  [d['date'] for d in places], "tickInterval": 90}
	yAxis = {"title": {"text": 'Daily Pedestrian Count'}}
	
	return render_template("Southern_Cross_Station.html", Southern_Cross_Station = True, chartID=chartID, chart=chart, series=series, title=title, xAxis=xAxis, yAxis=yAxis )

@app.route("/Mini_Temperature")
def Mini_Temperature():

#
#Below are Jason's Inputs to be tested on the main website.
#
#
#

#	with open('Min_Max_Temp.csv') as csv_file:
#		data = csv.reader(csv_file, delimiter=',')
#		first_line = True
#		places = []
#		for row in data:
#			if not first_line:
#				places.append({"Date": row[0], "Min": float(row[8])})
#			else:
#				first_line = False

#	chart = {"renderTo": chartID, "type": chart_type, "height": chart_height,}
#	series = [{"name": 'Daily Minimum Temperature °C', "data": [d['Max'] for d in places]}]
#	title = {"text": 'Daily Minimum Temperature from January 2015 to 4 April 2021'}
#	xAxis = { "categories":  [d['Date'] for d in places]}
#	yAxis = {"title": {"text": 'Daily Minimum Temperature °C'}}				
#	
#	
#	
	return render_template("Mini_Temperature.html", Maximum_Temperature = True, chartID=chartID, chart=chart, series=series, title=title, xAxis=xAxis, yAxis=yAxis)

    return render_template("Mini_Temperature.html", us = True)
	
@app.route("/Maximum_Temperature")
def Maximum_Temperature():
#
#Below are Jason's Inputs to be tested on the main website.
#
#
#

#	with open('Min_Max_Temp.csv') as csv_file:
#		data = csv.reader(csv_file, delimiter=',')
#		first_line = True
#		places = []
#		for row in data:
#			if not first_line:
#				places.append({"Date": row[0], "Max": float(row[9])})
#			else:
#				first_line = False

#	chart = {"renderTo": chartID, "type": chart_type, "height": chart_height,}
#	series = [{"name": 'Daily Maximum Temperature °C', "data": [d['Max'] for d in places]}]
#	title = {"text": 'Daily Maximum Temperature from January 2015 to 4 April 2021'}
#	xAxis = { "categories":  [d['Date'] for d in places]}
#	yAxis = {"title": {"text": 'Daily Maximum Temperature °C'}}				
#	
#	

	return render_template("Maximum_Temperature.html", Maximum_Temperature = True, chartID=chartID, chart=chart, series=series, title=title, xAxis=xAxis, yAxis=yAxis)
	
@app.route("/Rainfall")
def Rainfall(chartID = 'chart_ID', chart_type = 'scatter', chart_height = 500):

	with open('rainfall_dataset.csv') as csv_file:
		data = csv.reader(csv_file, delimiter=',')
		first_line = True
		places = []
		for row in data:
			if not first_line:
				places.append({"Date": row[0], "Rainfall_amount_(millimetres)": float(row[4])})
			else:
				first_line = False

	chart = {"renderTo": chartID, "type": chart_type, "height": chart_height,}
	series = [{"name": 'Daily rainfall', "data": [d['Rainfall_amount_(millimetres)'] for d in places]}]
	title = {"text": 'Daily rainfall from January 2015 to Febuary 2021'}
	xAxis = { "categories":  [d['Date'] for d in places]}
	yAxis = {"title": {"text": 'Daily rainfall'}}				
	
	
	return render_template("Rainfall.html", Rainfall = True, chartID=chartID, chart=chart, series=series, title=title, xAxis=xAxis, yAxis=yAxis)
	
@app.route("/Solar_Exposure")
def Solar_Exposure(chartID = 'chart_ID', chart_type = 'scatter', chart_height = 500):

	with open('solar_exposure.csv') as csv_file:
		data = csv.reader(csv_file, delimiter=',')
		first_line = True
		places = []
		for row in data:
			if not first_line:
				places.append({"Date": row[0], "Daily_global_solar_exposure": float(row[6])})
			else:
				first_line = False
		
	chart = {"renderTo": chartID, "type": chart_type, "height": chart_height,}
	series = [{"name": 'Daily global solar exposure', "data": [d['Daily_global_solar_exposure'] for d in places]}]
	title = {"text": 'Daily solar_exposure from January 2015 to Febuary 2021'}
	xAxis = { "categories":  [d['Date'] for d in places]}
	
	yAxis = {"title": {"text": 'Daily solar_exposure'}}
	return render_template("Solar_Exposure.html", Solar_Exposure = True,chartID=chartID, chart=chart, series=series, title=title, xAxis=xAxis, yAxis=yAxis)

	


	
	
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

