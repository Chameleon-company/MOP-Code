function initMap(id) {
    return new mapboxgl.Map({
        container: id, // container ID
        style: 'mapbox://styles/mapbox/light-v10', // style URL
        center: [144.95460780722914, -37.81422463241198], // starting position [lng, lat]
        zoom: 13 // starting zoom
    });
}

function addSearchRadiusLayer(map) {
    map.addLayer({
        id: 'search-radius',
        source: {
            type: 'geojson',
            data: { "type": "FeatureCollection", "features": [] },
        },
        type: 'fill',
        paint: {
            'fill-color': '#F1CF65',
            'fill-opacity': 0.4
        }
    })
}

function makeRadius([lat, lng], radiusInMeters) {
    let point = turf.point([lng, lat])
    let buffered = turf.buffer(point, radiusInMeters, { units: 'meters' })
    return buffered
}

function spatialJoin(geojson, filterFeature) {
    return geojson.features.filter(function(feature) {
        return turf.booleanPointInPolygon(feature, filterFeature);
    });
}

var unknownMarkers = undefined
var presentMarkers = undefined
var unoccupiedMarkers = undefined

function filterSensors(map, filterFeature) {
    return new Promise((resolve) => {
        unknownMarkers = unknownMarkers || map.getSource(unknownMarkerName)._data
        presentMarkers = presentMarkers || map.getSource(presentMarkerName)._data
        unoccupiedMarkers = unoccupiedMarkers || map.getSource(unoccupiedMarkerName)._data

        let markerNames = [unknownMarkerName, presentMarkerName, unoccupiedMarkerName]
        let allMarkers = [unknownMarkers, presentMarkers, unoccupiedMarkers]
        let index = 0

        let featureBuffers = []
        for (const markers of allMarkers) {
            let featureBuffer = spatialJoin(markers, filterFeature)

            // update map to only show data within search radius
            map.getSource(markerNames[index]).setData(turf.featureCollection(featureBuffer))
            featureBuffers.push(featureBuffer)
            index += 1
        }

        resolve(featureBuffers)
    })
}


function renderGraphContent(map, eventLatLng, radius, data) {
    let graphTemplate = document.querySelector('#parking_tool_graph').content.cloneNode(true)
    let graphClose = graphTemplate.querySelector('.parking_tool_close')
    let graphTitle = graphTemplate.querySelector('.graph_title')
    let graphTotalParks = graphTemplate.querySelector('.graph_total_parks')
    let graphAvailableParks = graphTemplate.querySelector('.graph_available_parks')
    let graphUnavailableParks = graphTemplate.querySelector('.graph_unavailable_parks')

    graphClose.addEventListener('click', () => {
        removeSearch(map)
    })

    // set the visualization title
    graphTitle.textContent = eventLatLng.toString()
    graphTotalParks.textContent = data.reduceRight((current, next) => current + next.length, 0)

    graphUnavailableParks.textContent = data[1].length
    graphAvailableParks.textContent = data[2].length

    let graphImages = graphTemplate.querySelectorAll('.graph_img')

    let filter = `radius=${encodeURIComponent(radius)}&latlng=${encodeURIComponent(JSON.stringify({
        lat: eventLatLng[0],
        lng: eventLatLng[1]
    }))}`

    let index = 0
    const base_graph = ['daily', 'hourly']
    for (const graphImage of graphImages) {
        graphImage.src = `${$SCRIPT_ROOT}/playground/parking-sensors/${base_graph[index]}_filtered.png?${filter}`
        index += 1
    }

    let graphContainer = document.querySelector('.parking_tool_graph')

    // remove old items from node
    graphContainer.innerHTML = ""

    graphContainer.appendChild(graphTemplate)

    // initially the graphContainer is hidden
    graphContainer.style.display = "block"
}

var latestMapClickEvent = undefined
var radius = 300

function onMapSelected(map, e) {
    latestMapClickEvent = e
    let eventLatLng = [e.lngLat.lat, e.lngLat.lng]
    let searchRadius = makeRadius(eventLatLng, radius)

    map.getSource('search-radius').setData(searchRadius)


    filterSensors(map, searchRadius)
        .then((data) => {
            renderGraphContent(map, eventLatLng, radius, data)
        })
}

function removeSearch(map) {
    unknownMarkers = unknownMarkers || map.getSource(unknownMarkerName)._data
    presentMarkers = presentMarkers || map.getSource(presentMarkerName)._data
    unoccupiedMarkers = unoccupiedMarkers || map.getSource(unoccupiedMarkerName)._data

    let markerNames = [unknownMarkerName, presentMarkerName, unoccupiedMarkerName]
    let allMarkers = [unknownMarkers, presentMarkers, unoccupiedMarkers]
    let index = 0
    for (const markers of allMarkers) {
        // update map to only show data within search radius
        map.getSource(markerNames[index]).setData(markers)
        index += 1
    }

    // remove the search radius overlay
    map.getSource('search-radius').setData({ "type": "FeatureCollection", "features": [] })

    let graphContainer = document.querySelector('.parking_tool_graph')
    graphContainer.style.display = "none"
}

// source: https://stackoverflow.com/questions/18544890/onchange-event-on-input-type-range-is-not-triggering-in-firefox-while-dragging
function onRangeChange(r, f) {
    var n, c, m;
    r.addEventListener("input", function(e) {
        n = 1;
        c = e.target.value;
        if (c != m) f(e);
        m = c;
    });
    r.addEventListener("change", function(e) { if (!n) f(e); });
}


var rangeUpdatedDelay = undefined

window.addEventListener("load", () => {
    const map = initMap("tool_map")

    // a layer onto which the search radius is eventually drawn
    map.on('style.load', () => {
        addSearchRadiusLayer(map)
        showParkingSensorsOnMap(map)
    })

    map.on('click', (e) => onMapSelected(map, e))

    let radiusSlider = document.querySelector('.final_step .parking_tool_radius_slider > .slider')
    let radiusValue = document.querySelector('.final_step .parking_tool_radius_slider > .radius_value')


    onRangeChange(radiusSlider, (e) => {
        radius = e.target.value * 10
        radiusValue.innerHTML = `${radius}m`

        if (rangeUpdatedDelay)
            clearTimeout(rangeUpdatedDelay)

        // redo the map selection event
        // with new radius setting
        if (latestMapClickEvent) {
            rangeUpdatedDelay = setTimeout(() => onMapSelected(map, latestMapClickEvent), 200)
        }
    })
})

window.addEventListener("load", () => {
    const map = initMap("solution_map")

    // a layer onto which the search radius is eventually drawn
    map.on('style.load', () => {
        addSearchRadiusLayer(map)
        showParkingSensorsOnMap(map)
    })

    map.on('click', (e) => onMapSelected(map, e))

    let radiusSlider = document.querySelector('.solution_demo .parking_tool_radius_slider > .slider')
    let radiusValue = document.querySelector('.solution_demo .parking_tool_radius_slider > .radius_value')


    onRangeChange(radiusSlider, (e) => {
        radius = e.target.value * 10
        radiusValue.innerHTML = `${radius}m`

        if (rangeUpdatedDelay)
            clearTimeout(rangeUpdatedDelay)

        // redo the map selection event
        // with new radius setting
        if (latestMapClickEvent) {
            rangeUpdatedDelay = setTimeout(() => onMapSelected(map, latestMapClickEvent), 200)
        }
    })
})