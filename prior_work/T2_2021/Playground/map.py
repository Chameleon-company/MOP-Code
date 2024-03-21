import folium
import pandas as pd
import os 
import webbrowser

def mark_map(data):
    """
    A map with markings
    :param data:
    :return:
    """
    # Map production
    myMap = folium.Map(location=[-37.82, 144.95], zoom_start=14)

    for i in range(len(data)):
        # Customize popup content
        test = folium.Html(
            '<b>Bay_id:{}</b></br> <b>Statues:{}</b></br> <b>Description:{}</b></br> <b>Duration:{}</b></br> <b>Disability{}</b></br> '.format(data.iloc[i]['bay_id'], data.iloc[i]['status'], data.iloc[i]['description1'],data.iloc[i]['duration1'],data.iloc[i]['disabilityext1']),
            script=True)
        
        popup = folium.Popup(test)
        
        if data.iloc[i]['status']=='Present':
            icon=folium.Icon(color='red', icon='info-sign')
        else:
            icon=folium.Icon(color='blue', icon='ok-sign')
        
        folium.Marker([data.iloc[i]['lat'], data.iloc[i]['lon']], popup=popup,icon=icon).add_to(myMap)
        
    return myMap
        



#if __name__ == '__main__':
#mark_map(df)



from geopy.distance import geodesic
import numpy as np


def geofilter(data,pin,n):
    filter=pd.DataFrame(columns=data.columns.values)
    j=0
    for i in np.arange(0,data.shape[0]):
        d = geodesic(pin[0],data.iloc[i]['location'].values()).km
        if d<=n:
            filter.loc[j]= data.iloc[i]
            j=j+1
        else:
            continue
    filter.to_csv('Geofilter_Pin[{}]_Distance[{}].csv'.
                 format(pin, n))
    return mark_map(filter)