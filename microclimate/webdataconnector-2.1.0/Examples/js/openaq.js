(function() {
    // Create the connector object
    var myConnector = tableau.makeConnector();

    // Define the schema
    myConnector.getSchema = function(schemaCallback) {
        var cols = [{
            id: "city",
            dataType: tableau.dataTypeEnum.string,
            alias:"City"
        },{
            id: "location",
            dataType: tableau.dataTypeEnum.string
        },{
            id: "parameter",
            dataType: tableau.dataTypeEnum.string
        },{
            id: "value",
            dataType: tableau.dataTypeEnum.float
        },{
            id: "unit",
            dataType: tableau.dataTypeEnum.string
        },{
            id: "date",
            dataType: tableau.dataTypeEnum.string,
            alias:"Date"
        },{
            id: "latitude",
            dataType: tableau.dataTypeEnum.float
        },{
            id: "longitude",
            dataType: tableau.dataTypeEnum.float
        }];

        var tableSchema = {
            id: "openaqmel",
            alias: "Melbourne city air quality data",
            columns: cols
        };

        schemaCallback([tableSchema]);
    };

    // Download the data
    myConnector.getData = function(table, doneCallback) {
        $.getJSON("https://api.openaq.org/v1/measurements?parameter=pm25&date_from=2020-01-01&coordinates=-37.808,144.97&radius=10000", function(resp) {
            var feat = resp.results,
                tableData = [];

            // Iterate over the JSON object
            for (var i = 0, len = feat.length; i < len; i++) {
                tableData.push({
                    "city": feat[i].city,
                    "location": feat[i].location,
                    "parameter": feat[i].parameter,
                    "value": feat[i].value,
                    "unit": feat[i].unit,
                    "date": feat[i].date.utc,
                    "longitude": feat[i].coordinates.longitude,
                    "latitude": feat[i].coordinates.latitude
                });
            }

            table.appendRows(tableData);
            doneCallback();
        });
    };

    tableau.registerConnector(myConnector);

    // Create event listeners for when the user submits the form
    $(document).ready(function() {
        $("#submitButton").click(function() {
            tableau.connectionName = "Melbourne City Air Quality data"; // This will be the data source name in Tableau
            tableau.submit(); // This sends the connector object to Tableau
        });
    });
})();
