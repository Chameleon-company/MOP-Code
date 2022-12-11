var rowTemplate = "<tr><td>{{name}}</td><td>{{difficulty}}</td></tr>";

function createNewRow(name, difficulty) {
  return rowTemplate.replace("{{name}}", name)
                    .replace("{{difficulty}}", difficulty);
}

function addUseCases() {
    let useCaseTable = document.getElementById("use-case-table");
    fetch(`${$SCRIPT_ROOT}/static/search.json`)
        .then((response) => response.json())
        .then ((data) => {
            for (item in data) {
                useCaseTable.innerHTML += createNewRow(data[item].title,data[item].difficulty);
            }
        });
}

function addDatasets() {
    let datasetTable = document.getElementById("dataset-table");
    fetch(`${$SCRIPT_ROOT}/search/datasets?query`)
        .then((response) => response.json())
        .then((data) => {
            for (item in data) {
                let datasetName = data[item].Name
                let datasetDownloads = data[item].Downloads
                let datasetURL = data[item].Permalink
                if (datasetName.indexOf("(") > -1) {
                    datasetTable.innerHTML += createNewRow(datasetName.substring(0,datasetName.indexOf("(")), datasetDownloads)
                } else {
                    datasetTable.innerHTML += createNewRow(datasetName, datasetDownloads)
                }
            }
        })
}

addUseCases();
addDatasets();