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
import imgkit
import imageio


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
        x = 0
        timeString = str(request.form.get("Year"))+str(' ')+str(request.form.get("Month"))+str(' ')+str(request.form.get("Day"))
        timeStriped = datetime.strptime(timeString, '%Y %m %d')
        timevalue = str(timeStriped.date())
        time = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"]
        date = timevalue
        for h in time:
            x += 1
            data = []
            string = date + " " + h
            for index, row in sensor_data_df.loc[string].iterrows():
                data.append([row['latitude'], row['longitude'], row['hourly_counts']])
            melb_city = folium.Map(location=[-37.81351814,144.96206317], zoom_start=14.4, width=800, height=600, control_scale=True)
            plugins.HeatMap(data, min_opacity = 0.01, max_val = 50).add_to(melb_city)
            title_html = '''
                        <h3 align="left" style="font-size:22px"><b>{}</b></h3>
                        '''.format('Date: ' + date + ' Time(Hour): ' + str(h))   
            melb_city.get_root().html.add_child(folium.Element(title_html))
            melb_city.save(str(x)+'.html')
        
        #config = imgkit.config(wkhtmltoimage='./wkhtmltox_0.12.6-1.focal_amd64.deb')
        for x in range(1,25):
            imgkit.from_file(str(x)+'.html', str(x)+'.jpg')

        images = []
        for n in range(1,25):
            images.append(str(n)+".jpg")
            
        image_list = []
        for file_name in images:
            image_list.append(imageio.imread(file_name))
        
        imageio.mimwrite('static/' + date + '.gif', image_list, fps=1)
        link = 'static/' + date + '.gif'
        return render_template("output.html", link=link)
                #return render_template(str(timevalue[:13])+'.html')


@app.route('/test')
def test():
    return render_template("output.html")

if __name__ == '__main__':
    app.run()