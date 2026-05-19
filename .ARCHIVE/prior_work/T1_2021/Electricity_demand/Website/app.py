@app.route("/Energy_forecast")
def Energy_forecast(chartID = 'chart_ID', chart_type = 'line', chart_height = 800):

	with open('electricity_demand.csv') as csv_file:
		data = csv.reader(csv_file, delimiter=',')
		first_line = True
		demand_values = []
		for row in data:
			if not first_line:
				demand_values.append({"Date": row[0], "Demand": np.round(float(row[1]),2)})
				
			else:
				first_line = False
				
	with open('dataset_for_prediction.csv') as csv_file:
		data = csv.reader(csv_file, delimiter=',')
		first_line = True
		demand_forecast = []
		for row in data:
			if not first_line:
				demand_forecast.append({"Date": row[0], "Demand": np.round(float(row[1]),2)})
			else:
				first_line = False		
                
                
	data1 = [d['Demand'] for d in demand_values]
	data1_2 = [d['Demand'] for d in demand_forecast]
	chart = {"renderTo": chartID, "type": chart_type, "height": chart_height}
	title = {"text": 'Energy consumption prediction from January 2015 to August 2021'}
	xAxis = { "title": {"text": 'Date'} , "type": 'datetime', "dateTimeLabelFormats": {"day": '%e %b' }}
	yAxis = {"title": {"text": 'Energy Consumption'}}
	return render_template("Energy_forecast.html", Energy_forecast= True,chartID=chartID, chart=chart, title=title, xAxis=xAxis, yAxis=yAxis, data1=data1, data1_2=data1_2)