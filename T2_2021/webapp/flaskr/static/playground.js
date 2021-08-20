map.on('load', () => {
    showParkingSensorsOnMap()
});


map.on('click', function(e) {
    fetch($SCRIPT_ROOT + `/playground/query_location?lng=${e.lngLat.lng}&lat=${e.lngLat.lat}`)
    .then(request => request.json())
    .then(data => console.dir(data))
})

function showParkingSensorsOnMap() {
    let occupiedImage = loadMapImage('/static/occupied_parking_sensor_25px.png')
        .then((image) => map.addImage('occupied_marker', image));
    let unoccupiedImage = loadMapImage('/static/unoccupied_parking_sensor_25px.png')
    .then((image) => map.addImage('unoccupied_marker', image));
    let unknownImage = loadMapImage('/static/unknown_parking_sensor_25px.png')
    .then((image) => map.addImage('unknown_marker', image));
    let imagesPromise = Promise.all([occupiedImage, unoccupiedImage, unknownImage])

    fetch($SCRIPT_ROOT + "/playground/parking-sensors/now")
    .then(result => result.json())
    .then((data) => {
        // after all the images have been added
        // create the markers for the map
        // make sure each status uses a different image
        return imagesPromise.then(() => {
            return data
            .reduce((features, parkingSensor) => {
                let {lat, lng, status} = parkingSensor
                let feature = {
                    // feature for Mapbox DC
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Point',
                        'coordinates': [
                            lng, lat
                        ]
                    },
                    'properties': {
                        'title': status
                    }
                }

                features[status].push(feature)
                return features
            }, {'Present': [], 'Unoccupied': [], 'Unknown': []})
        })
        .then(features => {
            let {Present, Unoccupied, Unknown} = features
            
            addLayer(Unknown, 'unknown_marker')
            addLayer(Present, 'occupied_marker')
            addLayer(Unoccupied, 'unoccupied_marker')
        })
    })
}

function loadMapImage(image) {
    return new Promise((resolve, reject) => map.loadImage($SCRIPT_ROOT + image, 
    (error, image) => {
        if(error)
            reject(error)
        else
            resolve(image)
    }))
}

function addLayer(features, markerName) {
    map.addSource(markerName, {
        'type': 'geojson',
        'data': {
            'type': 'FeatureCollection',
            'features': features
        }
    });
    map.addLayer({
        'id': markerName,
        'type': 'symbol',
        'source': markerName,
        'layout': {
            'icon-image': markerName,
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

function showTrafficLightsOnMap() {
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
}
