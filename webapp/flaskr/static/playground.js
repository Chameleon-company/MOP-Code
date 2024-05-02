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

function delete_placeholder(id) {
    document.getElementById(id).remove()
}

window.addEventListener('load', async () => {
    // Load the maps in the order in which they are rendered on the playground screen
    await setupParkingAvailabilityMap('solution_demo')
    await setupParkingSensorStep()
    await setupParkingAvailabilityMap('final_step')
})