map.on('load', () => {

    fetch($SCRIPT_ROOT + "/playground/traffic_lights")
        .then(response => response.json())
        .then((data) => {
            // Add an image to use as a custom marker
            map.loadImage(
                'https://docs.mapbox.com/mapbox-gl-js/assets/custom_marker.png',
                (error, image) => {
                    if (error) throw error;
                    map.addImage('custom-marker', image);
                    let features = data.map(lnglat => {
                        let [lng, lat] = lnglat
                        return {
                            // feature for Mapbox DC
                            'type': 'Feature',
                            'geometry': {
                                'type': 'Point',
                                'coordinates': [
                                    lng, lat
                                ]
                            },
                            'properties': {
                                'title': 'Traffic Light'
                            }
                        }

                    })

                    // Add a GeoJSON source with 2 points
                    map.addSource('points', {
                        'type': 'geojson',
                        'data': {
                            'type': 'FeatureCollection',
                            'features': features
                        }
                    });

                    // Add a symbol layer
                    map.addLayer({
                        'id': 'points',
                        'type': 'symbol',
                        'source': 'points',
                        'layout': {
                            'icon-image': 'custom-marker',
                            // get the title name from the source's "title" property
                            'text-field': ['get', 'title'],
                            'text-font': [
                                'Open Sans Semibold',
                                'Arial Unicode MS Bold'
                            ],
                            'text-offset': [0, 1.25],
                            'text-anchor': 'top'
                        }
                    });
                }
            );

        })

});


map.on('click', function(e) {
    fetch($SCRIPT_ROOT + `/playground/query_location?lng=${e.lngLat.lng}&lat=${e.lngLat.lat}`)
    .then(request => request.json())
    .then(data => console.dir(data))
})

