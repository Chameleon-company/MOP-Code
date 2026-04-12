from flask import Flask
from flask import request
from flask import render_template
import pandas as pd 
import os
import auth
import csv
import idealista as ide
import json
import folium
from folium.plugins import HeatMap

app = Flask(__name__)

@app.route('/drawMap')
def draw_map():
    map_data = pd.read_csv("C:\Users\kazim\Documents\flask_app\heatmap_test.csv")
    melb_city = folium.Map(location=[-37.80841814,144.95906317], zoom_start=14, width=1000, height=600, control_scale=True)
    plugins.HeatMap(map_data).add_to(melb_city)

    # Adds the heatmap element to the map
    melb_city.add_child(hm_wide)
    # Saves the map to heatmap.hmtl
    hmap.save(os.path.join('./templates', 'heatmap.html'))
    #Render the heatmap
    return render_template('heatmap.html')