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

    # Save the map
    geo_name = 'Geo_Map.html'
    #to save it in a file
    myMap.save('images/'+geo_name)
    #webbrowser.open('images/'+geo_name)

    image_path = "file://"+os.getcwd()+ '/'+ 'images/'+geo_name
    webbrowser.open(image_path)

    
    #myMap.save('testMap.html')

#if __name__ == '__main__':
#mark_map(df)