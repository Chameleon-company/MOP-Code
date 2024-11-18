#Required packages
import pandas as pd
import numpy as np
import folium
from folium import plugins
import datetime
from flask import Flask, render_template, request
from datetime import datetime, timedelta
import os
from pathlib import Path
import waitress

#Reading data/ needs to be replaced by SQL server
sensor_data_df = pd.read_csv("TidyPedL.csv")
sensor_data_df["date_time"] =  pd.to_datetime(sensor_data_df["date_time"])
sensor_data_df = sensor_data_df.set_index("date_time")

app = Flask(__name__)

@app.route('/')
def form():
    for p in Path("templates.").glob("2020*.html"):
        p.unlink()
    return render_template('index.html')



@app.route('/output', methods=['POST', 'GET'])
def data():
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/form' to submit form"
    if request.method == 'POST':
        data = []
        timeString = str(request.form.get("Year"))+str(' ')+str(request.form.get("Month"))+str(' ')+str(request.form.get("Day"))+str(' ')+str(request.form.get("Hour"))
        timeStriped = datetime.strptime(timeString, '%Y %m %d %H')
        timevalue = str(timeStriped.date())+ str(" ")+ str(timeStriped.time())
        for index, row in sensor_data_df.loc[timevalue[:13]].iterrows():
            #folium.Marker(location=[row['latitude'], row['longitude']], popup=row['sensor_name']).add_to(melb_city)
            data.append([row['latitude'], row['longitude'], row['hourly_counts']])

        melb_city = folium.Map(location=[-37.80841814,144.95906317], zoom_start=14, width=1000, height=600, control_scale=True)
        plugins.HeatMap(data, min_opacity = 0.01, max_val = 50).add_to(melb_city)

        title_html = '''
                    <h3 align="left" style="font-size:22px"><b>{}</b></h3>
                    '''.format('Year: ' + str(timevalue))   
        melb_city.get_root().html.add_child(folium.Element(title_html))
        melb_city.save(os.path.join('./templates', str(timevalue[:13])+'.html'))
        link = str(timevalue[:13])+'.html'
        return render_template("output.html", link=link)
        #return render_template(str(timevalue[:13])+'.html')

@app.route('/test', methods=['POST'])
def test():
    timeString = str(request.form.get("Year"))+str(' ')+str(request.form.get("Month"))+str(' ')+str(request.form.get("Day"))+str(' ')+str(request.form.get("Hour"))
    timeStriped = datetime.strptime(timeString, '%Y %m %d %H')
    timevalue = str(timeStriped.date())+ str(" ")+ str(timeStriped.time())
    return render_template('test.html', timevalue=timevalue)

if __name__ == '__main__':
    app.run()