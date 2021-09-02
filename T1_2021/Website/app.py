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
import os

scaler = pickle.load(open('newscaler.pkl', 'rb'))
model = pickle.load(open('pedestriant_ml_prediction_model_rf.pkl', 'rb'))

#scaler_ = pickle.load(open('scaler_ST_A.pkl', 'rb'))
#model_ = pickle.load(open('RF_model_ST_A.pkl', 'rb'))

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



	


	
@app.route("/Bourke_Street_Mall_South")
def Bourke_Street_Mall_South(chartID = 'chart_ID', chart_type = 'scatter', chart_height = 600):

	with open('Bourke_Street_Mall_South.csv') as csv_file:
		data = csv.reader(csv_file, delimiter=',')
		first_line = True
		
		places = []
		
		for row in data:
		
			if not first_line:
			
				places.append({"date": row[1], "daily_count": int(row[4])})
				
			else:
				first_line = False
				

	
	with open('BourkeStPredictionOnly.csv') as csv_file:
		data1 = csv.reader(csv_file, delimiter=',')
		first_line1 = True
		places1 = []
		for row1 in data1:
			if not first_line1:
				places1.append({"date": row1[1], "daily_count": round(float(row1[10]))})
			else:
				first_line1 = False
	
	
	
	data1 = [d['daily_count'] for d in places]
	data1_2 = [d['daily_count'] for d in places1]	
	chart = {"renderTo": chartID, "type": chart_type, "height": chart_height,}
	title = {"text": 'Bourke Street Mall South Daily Pedestrian Count from January 2015 to August 2021'}
	xAxis = { "title": {"text": 'Date'} , "type": 'datetime', "dateTimeLabelFormats": {"day": '%e %b' }}
	yAxis = {"title": {"text": 'Daily Pedestrian Count'}}		
	return render_template("Bourke_Street_Mall_South.html", Bourke_Street_Mall_South = True, chartID=chartID, chart=chart, data1=data1,data1_2=data1_2, title=title, xAxis=xAxis, yAxis=yAxis)
	
@app.route("/Victoria_Point")
def Victoria_Point(chartID = 'chart_ID', chart_type = 'scatter', chart_height = 600):

	
	with open('Victoria_Point.csv') as csv_file:
		data = csv.reader(csv_file, delimiter=',')
		first_line = True
		places = []
		for row in data:
			if not first_line:
				places.append({"date": row[1], "daily_count": int(row[4])})
			else:
				first_line = False
		
	with open('SouthernCrossCount_VictoriaPointCount_prediction.csv') as csv_file1:
		data1 = csv.reader(csv_file1, delimiter=',')
		first_line1 = True
		places1 = []
		for row1 in data1:
			if not first_line1:
				places1.append({"date": row1[1], "daily_count": round(float(row1[9]))})
			else:
				first_line1 = False
	
	
	
	data1 = [d['daily_count'] for d in places]
	data1_2 = [d['daily_count'] for d in places1]	
	chart = {"renderTo": chartID, "type": chart_type, "height": chart_height,}
	title = {"text": 'Victoria Point Daily Pedestrian Count from January 2015 to August 2021'} 
	xAxis = { "title": {"text": 'Date'} , "type": 'datetime', "dateTimeLabelFormats": {"day": '%e %b' }}
	yAxis = {"title": {"text": 'Daily Pedestrian Count'}}		
	return render_template('Victoria_Point.html', Victoria_Point = True, chartID=chartID, chart=chart, title=title, xAxis=xAxis, yAxis=yAxis,data1=data1,data1_2=data1_2)

	
@app.route("/Collins_Place_North")
def Collins_Place_North(chartID = 'chart_ID', chart_type = 'scatter', chart_height = 600):

	with open('Collins_Place_North.csv') as csv_file:
		data = csv.reader(csv_file, delimiter=',')
		first_line = True
		places = []
		for row in data:
			if not first_line:
				places.append({"date": row[1], "daily_count": int(row[4])})
			else:
				first_line = False
	
	with open('flinder_and_collin_prediction.csv') as csv_file:
		data1 = csv.reader(csv_file, delimiter=',')
		first_line1 = True
		places1 = []
		for row1 in data1:
			if not first_line1:
				places1.append({"date": row1[1], "daily_count": round(float(row1[12]))})
			else:
				first_line1 = False
				
	
	data1 = [d['daily_count'] for d in places]
	data1_2 = [d['daily_count'] for d in places1]	
	
	chart = {"renderTo": chartID, "type": chart_type, "height": chart_height,}
	title = {"text": 'Collins Place North Daily Pedestrian Count from January 2015 to August 2021'}
	xAxis = { "title": {"text": 'Date'} , "type": 'datetime', "dateTimeLabelFormats": {"day": '%e %b' }}
	yAxis = {"title": {"text": 'Daily Pedestrian Count'}}
	



	

	
	return render_template("Collins_Place_North.html", Collins_Place_North = True, chartID=chartID, chart=chart, title=title, xAxis=xAxis, yAxis=yAxis,data1=data1,data1_2=data1_2)
	
@app.route("/Flinders_St_Spark_La")
def Flinders_St_Spark_La(chartID = 'chart_ID', chart_type = 'scatter', chart_height = 600):

	with open('Flinders_St_Spark_La.csv') as csv_file:
		data = csv.reader(csv_file, delimiter=',')
		first_line = True
		places = []
		for row in data:
			if not first_line:
				places.append({"date": row[1], "daily_count": int(row[4])})
			else:
				first_line = False
		
	with open('flinder_and_collin_prediction.csv') as csv_file:
		data1 = csv.reader(csv_file, delimiter=',')
		first_line1 = True
		places1 = []
		for row1 in data1:
			if not first_line1:
				places1.append({"date": row1[1], "daily_count": round(float(row1[11]))})
			else:
				first_line1 = False
	
	
	
	data1 = [d['daily_count'] for d in places]
	data1_2 = [d['daily_count'] for d in places1]	
	chart = {"renderTo": chartID, "type": chart_type, "height": chart_height,}
	title = {"text": 'Flinders St Spark Lane Daily Pedestrian Count from January 2015 to August 2021'}
	xAxis = { "title": {"text": 'Date'} , "type": 'datetime', "dateTimeLabelFormats": {"day": '%e %b' }}
	yAxis = {"title": {"text": 'Daily Pedestrian Count'}}
	
	
	return render_template("Flinders_St_Spark_La.html", Flinders_St_Spark_La = True, chartID=chartID, chart=chart, title=title, xAxis=xAxis, yAxis=yAxis, data1=data1,data1_2=data1_2)

	
@app.route("/Southern_Cross_Station")
def Southern_Cross_Station(chartID = 'chart_ID', chart_type = 'scatter', chart_height = 600):

	with open('Southern_Cross_Station.csv') as csv_file:
		data = csv.reader(csv_file, delimiter=',')
		first_line = True
		places = []
		for row in data:
			if not first_line:
				places.append({"date": row[1], "daily_count": int(row[4])})
			else:
				first_line = False
		

	
	with open('SouthernCrossCount_VictoriaPointCount_prediction.csv') as csv_file:
		data1 = csv.reader(csv_file, delimiter=',')
		first_line1 = True
		places1 = []
		for row1 in data1:
			if not first_line1:
				places1.append({"date": row1[1], "daily_count": round(float(row1[10]))})
			else:
				first_line1 = False
	
	
	
	data1 = [d['daily_count'] for d in places]
	data1_2 = [d['daily_count'] for d in places1]	
	chart = {"renderTo": chartID, "type": chart_type, "height": chart_height,}
	title = {"text": 'Southern Cross Station Daily Pedestrian Count from January 2015 to August 2021'}
	xAxis = { "title": {"text": 'Date'} , "type": 'datetime', "dateTimeLabelFormats": {"day": '%e %b' }}
	yAxis = {"title": {"text": 'Daily Pedestrian Count'}}	
	return render_template("Southern_Cross_Station.html", Southern_Cross_Station = True, chartID=chartID, chart=chart, title=title, xAxis=xAxis, yAxis=yAxis,data1=data1,data1_2=data1_2 )

@app.route("/Mini_Temperature")
def Mini_Temperature(chartID = 'chart_ID', chart_type = 'scatter', chart_height = 600,chartID_2 = 'chartID_2', chart_type_2 = 'boxplot', chart_height_2 = 600, chartID_5 = 'chart_ID_5', chart_type_5 = 'histogram', chart_height_5 = 600,chartID_7 = 'chartID_7', chart_type_7 = 'scatter', chart_height_7 = 600,):

	with open('Min_Max_Temp_Jan_2015_to_Feb_2021.csv') as csv_file:
		data = csv.reader(csv_file, delimiter=',')
		first_line = True
		places = []
		for row in data:
			if not first_line:
				places.append({"Date": row[0], "Min": float(row[8])})
			else:
				first_line = False

	data = [d['Min'] for d in places]
	chart = {"renderTo": chartID, "type": chart_type, "height": chart_height,}
	title = {"text": 'Daily Minimum Temperature from January 2015 to February 2021'}
	xAxis = { "title": {"text": 'Date'} , "type": 'datetime', "dateTimeLabelFormats": {"day": '%e %b' }}
	yAxis = {"title": {"text": 'Daily Minimum Temperature °C'}}			

	chart5 = {"renderTo": chartID_5, "type": chart_type_5, "height": chart_height_5}
	series5 = [{"name": 'Daily Minimum Temperature', "data": [35, 214, 498 , 571, 480 ,317,107,22,5, 2]}]
	title5 = {"text": 'Daily Minimum Temperature Distribution (January 2015 to February 2021)'}
	xAxis5 = {"title": {"text": 'Daily Minimum Temperature range (mm)'}, "categories":  [ "0.6-3.6", "3.6-6.6" , "6.6-9.6" , "9.6-12.6" , "12.6-15.6" ,  "15.6-18.6", "18.6-21.6" ,"21.6-24.6","24.6-27.6","27.6-30.6"]}
	yAxis5 = {"title": {"text": 'Frequency'}}

	chart2 = {"renderTo": chartID_2, "type": chart_type_2, "height": chart_height_2}
	title2 = {"text": 'Box Plots of Minimum Temperature by Month'}
	xAxis2 = {"title": {"text": 'Month'}, "categories":  ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']}
	yAxis2 = {"title": {"text": 'Daily Minimum Temperature °C'}}

	with open('Max_Min_Temp_predicted_values.csv') as csv_file2:
		data2 = csv.reader(csv_file2, delimiter=',')
		first_line2 = True
		places2 = []
		for row2 in data2:
			if not first_line2:
				places2.append({"min_temp": np.round(float(row2[5]),2)})

			else:
				first_line2 = False

	data7_2 = [d['min_temp'] for d in places2]

	chart7 = {"renderTo": chartID_7, "type": chart_type_7, "height": chart_height_7,}
	title7 = {"text": 'Daily minimum temperature amount from January 2015 to August 2021'}
	xAxis7 = { "title": {"text": 'Date'} , "type": 'datetime', "dateTimeLabelFormats": {"day": '%e %b' }}
	yAxis7 = {"title": {"text": 'Daily minimum temperature amount'}}
	
	return render_template("Mini_Temperature.html", Mini_Temperature = True, chartID=chartID, data = data, chart=chart, title=title, xAxis=xAxis, yAxis=yAxis,chartID_2=chartID_2, chart2=chart2, title2=title2, xAxis2=xAxis2, yAxis2=yAxis2, chartID_5 = chartID_5, chart5=chart5, series5=series5, title5=title5, xAxis5=xAxis5, yAxis5=yAxis5,chartID_7 = chartID_7, chart7=chart7, data7_2=data7_2, title7=title7, xAxis7=xAxis7, yAxis7=yAxis7,)


	
@app.route("/Maximum_Temperature")
def Maximum_Temperature(chartID = 'chart_ID', chart_type = 'scatter', chart_height = 600,chartID_2 = 'chartID_2', chart_type_2 = 'boxplot', chart_height_2 = 600, chartID_5 = 'chart_ID_5', chart_type_5 = 'histogram', chart_height_5 = 600,chartID_7 = 'chartID_7', chart_type_7 = 'scatter', chart_height_7 = 600,):

 

	with open('Min_Max_Temp_Jan_2015_to_Feb_2021.csv') as csv_file:
		data = csv.reader(csv_file, delimiter=',')
		first_line = True
		places = []
		for row in data:
			if not first_line:
				places.append({"Date": row[0], "Max": float(row[9])})
			else:
				first_line = False

	data = [d['Max'] for d in places]
	chart = {"renderTo": chartID, "type": chart_type, "height": chart_height,}
	title = {"text": 'Daily Maximum Temperature from January 2015 to February 2021'}
	xAxis = { "title": {"text": 'Date'} , "type": 'datetime', "dateTimeLabelFormats": {"day": '%e %b' }}
	yAxis = {"title": {"text": 'Daily Maximum Temperature °C'}}
	
	chart5 = {"renderTo": chartID_5, "type": chart_type_5, "height": chart_height_5}
	series5 = [{"name": 'Daily Maximum Temperature', "data": [258, 814, 597, 320, 174 ,71, 17]}]
	title5 = {"text": 'Daily Maximum Temperature Distribution (January 2015 to February 2021)'}
	xAxis5 = {"title": {"text": 'Daily Maximum Temperature range (°C)'}, "categories":  [ "9-14", "14-19" , "19-24" , "24-29" , "29-34" ,  "34-39",  "39-44"]}
	yAxis5 = {"title": {"text": 'Frequency'}}

	chart2 = {"renderTo": chartID_2, "type": chart_type_2, "height": chart_height_2}
	title2 = {"text": 'Box Plots of Maximum Temperature by Month'}
	xAxis2 = {"title": {"text": 'Month'}, "categories":  ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']}
	yAxis2 = {"title": {"text": 'Daily Maximum Temperature °C'}}	

	with open('Max_Min_Temp_predicted_values.csv') as csv_file2:
		data2 = csv.reader(csv_file2, delimiter=',')
		first_line2 = True
		places2 = []
		for row2 in data2:
			if not first_line2:
				places2.append({"max_temp": np.round(float(row2[4]),2)})

			else:
				first_line2 = False

	data7_2 = [d['max_temp'] for d in places2]

	chart7 = {"renderTo": chartID_7, "type": chart_type_7, "height": chart_height_7,}
	title7 = {"text": 'Daily maximum temperature from January 2015 to August 2021'}
	xAxis7 = { "title": {"text": 'Date'} , "type": 'datetime', "dateTimeLabelFormats": {"day": '%e %b' }}
	yAxis7 = {"title": {"text": 'Daily maximum_temperature'}}
	
	return render_template("Maximum_Temperature.html", Maximum_Temperature = True, chartID=chartID, data = data, chart=chart, title=title, xAxis=xAxis, yAxis=yAxis,chartID_2=chartID_2, chart2=chart2, title2=title2, xAxis2=xAxis2, yAxis2=yAxis2,  chartID_5 = chartID_5, chart5=chart5, series5=series5, title5=title5, xAxis5=xAxis5, yAxis5=yAxis5,chartID_7 = chartID_7, chart7=chart7, data7_2=data7_2, title7=title7, xAxis7=xAxis7, yAxis7=yAxis7,)
	
@app.route("/Rainfall")
def Rainfall(chartID = 'chart_ID', chart_type = 'scatter', chart_height = 600, chartID_2 = 'chartID_2', chart_type_2 = 'column', chart_height_2 = 600, chartID_3 = 'chart_ID_3', chart_type_3 = 'column', chart_height_3 = 600, chartID_4 = 'chart_ID_4', chart_type_4 = 'column', chart_height_4 = 600, chartID_5 = 'chart_ID_5', chart_type_5 = 'histogram', chart_height_5 = 600, chartID_7 = 'chartID_7', chart_type_7 = 'line', chart_height_7 = 600,):

	with open('rainfall_dataset.csv') as csv_file:
		data = csv.reader(csv_file, delimiter=',')
		first_line = True
		places = []
		for row in data:
			if not first_line:
				places.append({"Date": row[0], "Rainfall_amount_(millimetres)": float(row[4])})
			else:
				first_line = False

	chart = {"renderTo": chartID, "type": chart_type, "height": chart_height}
	series = [{"name": 'Daily rainfall', "data": [d['Rainfall_amount_(millimetres)'] for d in places]}]
	title = {"text": 'Daily rainfall from January 2015 to February 2021'}
	xAxis = { "title": {"text": 'Date'}, "categories":  [d['Date'] for d in places], "tickInterval": 90}
	yAxis = {"title": {"text": 'Daily rainfall'}}
	
	chart2 = {"renderTo": chartID_2, "type": chart_type_2, "height": chart_height_2}
	series2 = [{"name": 'Rainfall Amount', "data": [30.6, 22.6, 43.8, 41.2, 15.4, 54.6, 43.2] }]
	title2 = {"text": 'Maximum rainfall Amount per year from 2015 to 2021'}
	xAxis2 = {"title": {"text": 'Year'}, "categories":  [2015, 2016, 2017, 2018, 2019, 2020, 2021]}
	yAxis2 = {"title": {"text": 'Rainfall (millimetres)'}}

	chart3 = {"renderTo": chartID_3, "type": chart_type_3, "height": chart_height_3}
	series3 = [{"name": 'Daily rainfall', "data": [43.8, 32.4, 44, 38, 54.6, 25.2, 43.2]}]
	title3 = {"text": 'Maximum rainfall Amount per day of the week for 2015 to 2021'}
	xAxis3 = { "title": {"text": 'Day of the week'}, "categories":  ["Sunday" ,"Monday", "Tuesday", "Wednesday", "Thursday", "Friday" ,  "Saturday"  ]}
	yAxis3 = {"title": {"text": 'Daily rainfall'}}	

	chart4 = {"renderTo": chartID_4, "type": chart_type_4, "height": chart_height_4}
	series4 = [{"name": 'Maximum rainfall', "data": [44, 38, 54.6, 35.2, 18.6, 15.6, 28.4, 25.6, 21.4, 32.2, 35.8, 43.8 ]}]
	title4 = {"text": 'Maximum rainfall per month for 2015 to 2021'}
	xAxis4 = {"title": {"text": 'Month'}, "categories":  ["Jan", "Feb" , "Mar" , "Apr" , "May" ,  "Jun" ,  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec" ]}
	yAxis4 = {"title": {"text": 'Daily rainfall'}}
	
	chart5 = {"renderTo": chartID_5, "type": chart_type_5, "height": chart_height_5}
	series5 = [{"name": 'Daily rainfall', "data": [2164, 55, 19 , 8, 4 ,1 ]}]
	title5 = {"text": 'Daily rainfall Distribution (January 2015 to February 2021)'}
	xAxis5 = {"title": {"text": 'Daily rainfall range (mm)'}, "categories":  [ "0-10", "10-20" , "20-30" , "30-40" , "40-50" ,  "50-60"]}
	yAxis5 = {"title": {"text": 'Frequency'}}

	with open('rainfall_predicting.csv') as csv_file2:
		data2 = csv.reader(csv_file2, delimiter=',')
		first_line2 = True
		places2 = []
		for row2 in data2:
			if not first_line2:
				places2.append({"predict_rainfall": np.round(float(row2[5]),2)})

			else:
				first_line2 = False
	data7 = [d['Rainfall_amount_(millimetres)'] for d in places]
	data7_2 = [d['predict_rainfall'] for d in places2]

	chart7 = {"renderTo": chartID_7, "type": chart_type_7, "height": chart_height_7,}
	title7 = {"text": 'Daily rainfall amount from January 2015 to August 2021'}
	xAxis7 = { "title": {"text": 'Date'} , "type": 'datetime', "dateTimeLabelFormats": {"day": '%e %b' }}
	yAxis7 = {"title": {"text": 'Daily rainfall amount'}}
	
	return render_template("Rainfall.html", Rainfall = True, chartID=chartID, chart=chart, series=series, title=title, xAxis=xAxis, yAxis=yAxis, chartID_2=chartID_2, chart2=chart2, series2=series2, title2=title2, xAxis2=xAxis2, yAxis2=yAxis2, chartID_3=chartID_3, chart3=chart3, series3=series3, title3=title3, xAxis3=xAxis3, yAxis3=yAxis3, chartID_4 = chartID_4, chart4=chart4, series4=series4, title4=title4, xAxis4=xAxis4, yAxis4=yAxis4, chartID_5 = chartID_5, chart5=chart5, series5=series5, title5=title5, xAxis5=xAxis5, yAxis5=yAxis5,chartID_7=chartID_7, chart7=chart7, title7=title7, xAxis7=xAxis7,data7_2=data7_2, data7=data7, yAxis7=yAxis7)
	
@app.route("/Solar_Exposure")
def Solar_Exposure(chartID = 'chart_ID', chart_type = 'scatter', chart_height = 600, chartID_2 = 'chartID_2', chart_type_2 = 'column', chart_height_2 = 600, chartID_3 = 'chart_ID_3', chart_type_3 = 'column', chart_height_3 = 600, chartID_4 = 'chart_ID_4', chart_type_4 = 'column', chart_height_4 = 600, chartID_5 = 'chart_ID_5', chart_type_5 = 'histogram', chart_height_5 = 600, chartID_7 = 'chartID_7', chart_type_7 = 'scatter', chart_height_7 = 600,):

	with open('solar_exposure.csv') as csv_file:
		data = csv.reader(csv_file, delimiter=',')
		first_line = True
		places = []
		for row in data:
			if not first_line:
				places.append({"Date": row[0], "Daily_global_solar_exposure": float(row[6])})
			else:
				first_line = False


	data = [d['Daily_global_solar_exposure'] for d in places]
	chart = {"renderTo": chartID, "type": chart_type, "height": chart_height,}
	title = {"text": 'Daily solar_exposure from January 2015 to February 2021'}
	xAxis = { "title": {"text": 'Date'} , "type": 'datetime', "dateTimeLabelFormats": {"day": '%e %b' }}
	yAxis = {"title": {"text": 'Daily solar_exposure'}}
    
	chart2 = {"renderTo": chartID_2, "type": chart_type_2, "height": chart_height_2}
	series2 = [{"name": 'Solar exposure', "data": [15.26, 14.70, 15.00, 15.02, 15.38, 14.54, 20.61] }]
	title2 = {"text": 'Mean solar exposure per year from 2015 to 2021'}
	xAxis2 = {"title": {"text": 'Year'}, "categories":  [2015, 2016, 2017, 2018, 2019, 2020, 2021]}
	yAxis2 = {"title": {"text": 'Daily solar exposure'}}
    
	chart3 = {"renderTo": chartID_3, "type": chart_type_3, "height": chart_height_3}
	series3 = [{"name": 'Solar exposure', "data": [15.10, 15.09, 15.39, 15.50, 15.23, 14.71, 14.88]}]
	title3 = {"text": 'Mean solar exposure  per day of the week for 2015 to 2021'}
	xAxis3 = { "title": {"text": 'Day of the week'}, "categories":  ["Sunday" ,"Monday", "Tuesday", "Wednesday", "Thursday", "Friday" ,  "Saturday"  ]}
	yAxis3 = {"title": {"text": 'Daily solar exposure'}}	
    
	chart4 = {"renderTo": chartID_4, "type": chart_type_4, "height": chart_height_4}
	series4 = [{"name": 'Solar exposure', "data": [23.19, 20.19, 15.81, 10.94, 7.86, 7.06, 7.60, 9.99, 13.90, 18.43, 20.78, 23.88 ]}]
	title4 = {"text": 'Mean solar exposure per month for 2015 to 2021'}
	xAxis4 = {"title": {"text": 'Month'}, "categories":  ["Jan", "Feb" , "Mar" , "Apr" , "May" ,  "Jun" ,  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec" ]}
	yAxis4 = {"title": {"text": 'Daily solar exposure'}}
    
	chart5 = {"renderTo": chartID_5, "type": chart_type_5, "height": chart_height_5}
	series5 = [{"name": 'Daily Solar exposure', "data": [74, 416, 528 , 316, 267 ,236, 195, 195, 24]}]
	title5 = {"text": 'Daily Solar exposure Distribution (January 2015 to February 2021)'}
	xAxis5 = {"title": {"text": 'Daily Solar exposure'}, "categories":  [ "0-4", "4-8" , "8-12" , "12-16" , "16-20" ,  "20-24", "24-28", "28-32", "32-36"]}
	yAxis5 = {"title": {"text": 'Frequency'}}   

	with open('predicted_solar_exposure.csv') as csv_file2:
    
		data2 = csv.reader(csv_file2, delimiter=',')
		first_line2 = True
		places2 = []
		for row2 in data2:
			if not first_line2:
				places2.append({"Solar_exp": np.round(float(row2[7]),2)})
			else:
				first_line2 = False

	data7_2 = [d['Solar_exp'] for d in places2]

	chart7 = {"renderTo": chartID_7, "type": chart_type_7, "height": chart_height_7,}
	title7 = {"text": 'Daily Solar exposure amount from January 2015 to August 2021'}
	xAxis7 = { "title": {"text": 'Date'} , "type": 'datetime', "dateTimeLabelFormats": {"day": '%e %b' }}
	yAxis7 = {"title": {"text": 'Daily rainfall amount'}}
    


	return render_template("Solar_Exposure.html", Solar_Exposure = True, chartID=chartID, chart=chart, title=title, data= data, xAxis=xAxis, yAxis=yAxis, chartID_2=chartID_2, chart2=chart2, series2=series2, title2=title2, xAxis2=xAxis2, yAxis2=yAxis2, chartID_3=chartID_3, chart3=chart3, series3=series3, title3=title3, xAxis3=xAxis3, yAxis3=yAxis3, chartID_4 = chartID_4, chart4=chart4, series4=series4, title4=title4, xAxis4=xAxis4, yAxis4=yAxis4, chartID_5 = chartID_5, chart5=chart5, series5=series5, title5=title5, xAxis5=xAxis5, yAxis5=yAxis5,chartID_7=chartID_7, chart7=chart7, title7=title7, xAxis7=xAxis7,data7_2=data7_2, yAxis7=yAxis7)

@app.route("/RRP")
def RRP(chartID = 'chart_ID', chart_type = 'line', chart_height = 600,chartID_2 = 'chartID_2', chart_type_2 = 'column', chart_height_2 = 600, chartID_3 = 'chart_ID_3', chart_type_3 = 'column', chart_height_3 = 600,  chartID_5 = 'chart_ID_5', chart_type_5 = 'histogram', chart_height_5 = 600,chartID_6 = 'chartID_6', chart_type_6 = 'scatter', chart_height_6 = 600, chartID_4 = 'chartID_4', chart_type_4 = 'scatter', chart_height_4 = 600, chartID_7 = 'chartID_7', chart_type_7 = 'line', chart_height_7 = 600, ):

	with open('electricity_demand.csv') as csv_file:
		data = csv.reader(csv_file, delimiter=',')
		first_line = True
		places = []
		for row in data:
			if not first_line:
				places.append({"Date": row[0],"RRP": np.round(float(row[2]),2), "Demand": np.round(float(row[1]),2)})

			else:
				first_line = False
	lst = []
 
	for k in places:
		lst.append([k['RRP'],k['Demand']])
		 
	

		
	chart = {"renderTo": chartID, "type": chart_type, "height": chart_height,}
	series = [{"name": 'Daily Recommended Retail Price', "data": [d['RRP'] for d in places]}]
	title = {"text": 'Daily Recommended Retail Price from January 2015 to February 2021'}
	xAxis = { "title": {"text": 'Date'}, "categories":  [d['Date'] for d in places], "tickInterval": 90}
	yAxis = {"title": {"text": 'Daily Recommended Retail Price'}}

	chart2 = {"renderTo": chartID_2, "type": chart_type_2, "height": chart_height_2}
	series2 = [{"name": 'Average RRP per month', "data": [112.95, 63.59, 72.45 ,66.74, 69.27,81.25 ,75.48,72.73,68.58,65.99,64.20,59.15]}]
	title2 = {"text": 'Average RRP per month from January 2015 to Febuary 2021'}
	xAxis2 = {"title": {"text": 'Month'} ,"categories":  ["Jan", "Feb" , "Mar" , "Apr" , "May" ,  "Jun" ,  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec" ]}
	yAxis2 = {"title": {"text": 'Average RRP Amount'}}		

	chart3 = {"renderTo": chartID_3, "type": chart_type_3, "height": chart_height_3}
	series3 = [{"name": 'Average RRP per day of week', "data": [56.62,69.92,72.31,76.43,91.99,85.69, 59.67]}]
	title3 = {"text": 'Average RRP per day of week from January 2015 to February 2021'}
	xAxis3 = { "title": {"text": 'Day of week'},"categories":  ["Sunday", "Monday", "Tuesday" , "Wednesday" , "Thursday" , "Friday" ,  "Saturday"]}
	yAxis3 = {"title": {"text": 'Average RRP Amount'}}	

	
	chart5 = {"renderTo": chartID_5, "type": chart_type_5, "height": chart_height_5}
	series5 = [{"name": 'Frequency', "data": [1228, 986, 25 , 3, 1 ,2 ,1,1,1,1,1,1]}]
	title5 = {"text": 'Daily RRP Distribution (January 2015 to February 2021)'}
	xAxis5 = {"title": {"text": 'Daily RRP range (AUD$)'}, "categories":  [ "-31.15-68.85", "68.85-168.85" , "168.85-268.85" , "268.85-368.85" , "468.85-568.85" ,  "568.85-668.85", "868.85-968.85","968.85-1068.85" , "1168.85-1268.85", "1268.85-1368.85", "2768.85-2868.85", "14468.85-4568.85"]}
	yAxis5 = {"title": {"text": 'Frequency'}}
	
	chart6 = {"renderTo": chartID_6, "type": chart_type_6, "height": chart_height_6}
	series6 = [{"name": 'Demand (megawatt per hour)', "data": lst }]
	title6 = {"text": 'Scatter plot of RRP against Demand'}
	xAxis6 = {"title": {"text": 'Daily RRP in (AUD$)'}}
	yAxis6 = {"title": {"text": 'Demand (megawatt per hour)'}}	


	with open('Dataset_for_scatterplot_RRP.csv') as csv_file1:
		data1 = csv.reader(csv_file1, delimiter=',')
		first_line1 = True
		places1 = []
		for row1 in data1:
			if not first_line1:
				places1.append({"RRP": np.round(float(row1[2]),2), "Demand": np.round(float(row1[1]),2)})

			else:
				first_line1 = False
	lst1 = []
	for k1 in places1:
		lst1.append([k1['RRP'],k1['Demand']])	

	chart4 = {"renderTo": chartID_4, "type": chart_type_4, "height": chart_height_4}
	series4 = [{"name": 'Demand (megawatt per hour)', "data": lst1 }]
	title4 = {"text": 'Scatter plot of RRP against Demand without outliers'}
	xAxis4 = {"title": {"text": 'Daily RRP in (AUD$)'}}
	yAxis4 = {"title": {"text": 'Demand (megawatt per hour)'}}		
	
	with open('dataset_for_prediction.csv') as csv_file2:
		data2 = csv.reader(csv_file2, delimiter=',')
		first_line2 = True
		places2 = []
		for row2 in data2:
			if not first_line2:
				places2.append({"Date": row2[0],"RRP": np.round(float(row2[2]),2)})

			else:
				first_line2 = False
			
				
	data7 = [d['RRP'] for d in places]
	data7_2 = [d['RRP'] for d in places2]
	chart7 = {"renderTo": chartID_7, "type": chart_type_7, "height": chart_height_7,}
	title7 = {"text": 'Daily Recommended Retail Price from January 2015 to August 2021'}
	xAxis7 = { "title": {"text": 'Date'} , "type": 'datetime', "dateTimeLabelFormats": {"day": '%e %b' }}
	yAxis7 = {"title": {"text": 'Daily Recommended Retail Price'}}

	return render_template("RRP.html", RRP = True,chartID=chartID, chart=chart, series=series, title=title, xAxis=xAxis, yAxis=yAxis,chartID_2=chartID_2, chart2=chart2, series2=series2, title2=title2, xAxis2=xAxis2, yAxis2=yAxis2, chartID_3=chartID_3, chart3=chart3, series3=series3, title3=title3, xAxis3=xAxis3, yAxis3=yAxis3,  chartID_5 = chartID_5, chart5=chart5, series5=series5, title5=title5, xAxis5=xAxis5, yAxis5=yAxis5,chartID_6 = chartID_6, chart6=chart6, series6=series6, title6=title6, xAxis6=xAxis6, yAxis6=yAxis6, chartID_4 = chartID_4, chart4=chart4, series4=series4, title4=title4, xAxis4=xAxis4, yAxis4=yAxis4, chartID_7=chartID_7, chart7=chart7, title7=title7, xAxis7=xAxis7,data7_2=data7_2, data7=data7, yAxis7=yAxis7)
	

@app.route("/Energy_forecast")
def Energy_forecast(chartID = 'chart_ID', chart_type = 'scatter', chart_height = 600):
	with open('electricity_demand.csv') as csv_file:
		data = csv.reader(csv_file, delimiter=',')
		first_line = True
		places = []
		for row in data:
			if not first_line:
				places.append({"Date": row[0], "Demand": np.round(float(row[1]),2)})
			else:
				first_line = False
	with open('dataset_for_prediction.csv') as csv_file:
		data = csv.reader(csv_file, delimiter=',')
		first_line = True
		forecast = []
		for row in data:
			if not first_line:
				forecast.append({"Date": row[0], "Demand": np.round(float(row[1]),2)})
			else:
				first_line = False		
                
	data1 = [d['Demand'] for d in places]
	data1_2 = [d['Demand'] for d in forecast]
	chart = {"renderTo": chartID, "type": chart_type, "height": chart_height}
	title = {"text": 'Energy consumption forecast from January 2015 to August 2021'}
	xAxis = { "title": {"text": 'Date'} , "type": 'datetime', "dateTimeLabelFormats": {"day": '%e %b' }}
	yAxis = {"title": {"text": 'Energy Consumption'}}
	return render_template("Energy_forecast.html", Energy_forecast= True,chartID=chartID, chart=chart, title=title, xAxis=xAxis, yAxis=yAxis, data1=data1, data1_2=data1_2)

	
	
@app.route("/Pedestrian_forecast")
def Pedestrian_forecast(chartID = 'chart_ID', chart_type = 'scatter', chart_height = 600):
	file_path = os.path.join('time_series','FinalPedestrianPrediction.tsv')
	df = pd.read_csv(file_path, sep='\t')
	places = df.to_dict(orient='list')

	def get_datetime_object():
		return datetime.now()

	current_date_obj = get_datetime_object()
	last_pred_month = df.iloc[-1]['Date'].split('-')[1]
	chart = {"renderTo": chartID, "type": chart_type, "height": chart_height}
	# series = [{"name": 'Pedestrian Count', "color" : '#4572A7', "data": df['Total_Pedestrian_Count_per_day'].tolist()}]


	daily_pedestrian_list = df['Total_Pedestrian_Count_per_day'].tolist()

	daily_pedestrian_hist_list = daily_pedestrian_list[:-21]
	daily_pedestrian_forecasted_list = [0]*len(daily_pedestrian_list[:-21])+daily_pedestrian_list[-21:]			# first apart is to align with the dates

	series = [
			  {"name": 'Pedestrian Count', "color" : '#4572A7', "data": daily_pedestrian_hist_list},
			  {"name": 'Pedestrian Count Forecasted', "color" : '#FFA500', "data": daily_pedestrian_forecasted_list}
			  ]

	title = {"text": 'Melbourne City Daily Pedestrian Count from January 2015 to {}'.format(current_date_obj.strftime('%B %Y'))}
	xAxis = {"title": {"text": 'Date'}, "categories":  df['Date'].tolist(), "tickInterval": 90}
	yAxis = {"title": {"text": 'Daily Pedestrian Count'}}
	return render_template("Pedestrian_forecast.html", Bourke_Street_Mall_South = True, chartID=chartID, chart=chart, series=series, title=title, xAxis=xAxis, yAxis=yAxis)


@app.route("/Pedestrian_prediction", methods=['GET','POST'])
def Pedestrian_prediction():
    form = Pedestrian_prediction_Form()
    # 
    if form.is_submitted():
        independent_variables = request.form
        date_time_obj = datetime.strptime(independent_variables['date'], '%Y-%m-%d')
        X_test = [date_time_obj.timetuple().tm_wday+1,date_time_obj.month,  date_time_obj.year, date_time_obj.timetuple().tm_yday,  int(independent_variables['restriction']),  int(independent_variables['public_holiday']), float(independent_variables['rainfall']), float(independent_variables['minimum_temperature']), float(independent_variables['maximum_temperature']), float(independent_variables['solar_exposure'])]
        #X_test = [float(independent_variables['solar_exposure']), int(independent_variables['restriction']), float(independent_variables['rainfall']), int(independent_variables['public_holiday']), float(independent_variables['maximum_temperature']) , float(independent_variables['minimum_temperature'])]        
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

