mapboxgl.accessToken = 'pk.eyJ1IjoianNhbnNvbXNoZXJ3aWxsIiwiYSI6ImNrc2Y4Z255MjE4NXAydXBpbGs5bHlha3IifQ.dgv4c4Gl2v_K0TN_RpYfKg';

function setupParkingSensorStep() {
    return new Promise((resolve, _) => {
        const map = new mapboxgl.Map({
            container: 'map', // container ID
            style: 'mapbox://styles/mapbox/light-v10', // style URL
            center: [144.95460780722914, -37.81422463241198], // starting position [lng, lat]
            zoom: 13 // starting zoom
        });

        map.on('load', async () => {
            await showParkingSensorsOnMap(map)
            resolve() // after this step, we consider the mapbox setup, completed
        });
    })
}

var unknownMarkerName = 'unknown_marker'
var presentMarkerName = 'occupied_marker'
var unoccupiedMarkerName = 'unoccupied_marker'

async function showParkingSensorsOnMap(map) {
    // add the markers for different statuses to our available images
    let imagesPromise = loadMarkers(map)
    let latestSensors = fetch($SCRIPT_ROOT + "/playground/parking-sensors/latest.json")
        .then(result => result.json())

    let [_, data] = await Promise.all([imagesPromise, latestSensors])
    // convert sensor information into geojson
    let features = data
        .reduce((features, parkingSensor) => {
            let { lat, lon, status } = parkingSensor
            let lng = lon
            let feature = {

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
            // add next parking sensor to the geojson features
            // with respect to status
            features[status].push(feature)
            return features
        }, { 'Present': [], 'Unoccupied': [], 'Unknown': [] })

    // add each status type to the map
    // as new marker mapbox layer
    let { Present, Unoccupied, Unknown } = features

    addLayer(map, Unknown, unknownMarkerName)
    addLayer(map, Present, presentMarkerName)
    addLayer(map, Unoccupied, unoccupiedMarkerName)
}

function loadMapImage(map, image) {
    return new Promise((resolve, reject) => map.loadImage($SCRIPT_ROOT + image,
        (error, image) => {
            if (error)
                reject(error)
            else
                resolve(image)
        }))
}

function loadMarkers(map) {
    return Promise.all([
        loadMapImage(map, '/static/occupied_parking_sensor_25px.png'), // url of custom png markers to represent status
        loadMapImage(map, '/static/unoccupied_parking_sensor_25px.png'),
        loadMapImage(map, '/static/unknown_parking_sensor_25px.png')
    ]).then(([occupied, unoccupied, unknown]) => {
        map.addImage(presentMarkerName, occupied)
        map.addImage(unoccupiedMarkerName, unoccupied)
        map.addImage(unknownMarkerName, unknown)
    })
}

function addLayer(map, features, markerName) {
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
            // 'text-field': ['get', 'title'],
            // 'text-font': [
            //     'Open Sans Semibold',
            //     'Arial Unicode MS Bold'
            // ],
            // 'text-offset': [0, 1.25],
            // 'text-anchor': 'top'
        }
    });
}

function delete_placeholder(id) {
    document.getElementById(id).remove()
}

window.addEventListener('load', async () => {
    // Load the maps in the order in which they are rendered on the playground screen
    await setupParkingAvailabilityMap('solutionMap', '.solution_demo')
    await setupParkingSensorStep()
    await setupParkingAvailabilityMap('toolMap', '.final_step')
})