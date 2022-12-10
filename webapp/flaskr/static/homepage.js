// First, define the template for the new rows as a string
var rowTemplate = "<tr><td>{{name}}</td><td>{{difficulty}}</td></tr>";

// Next, define a function for generating a new row based on the template
// and some provided data
function createNewRow(name, difficulty) {
  // Replace the placeholders in the template with the actual data
  return rowTemplate.replace("{{name}}", name)
                    .replace("{{difficulty}}", difficulty);
}

function addUseCases() {
    var table = document.getElementById("use-case-table");
    fetch(`${$SCRIPT_ROOT}/static/search.json`)
        .then((response) => response.json())
        .then ((data) => {
            for (item in data) {
                table.innerHTML += createNewRow(data[item].title,data[item].difficulty);
            }
        });
}

function addDatasets() {
    var table = document.getElementById("dataset-table");
    fetch(`${$SCRIPT_ROOT}/search/datasets?query=${searchTerms.join(' ')}`)
        .then(result => result.json())
        .then ((datasets) => {
            for (item in datasets) {
                console.log(datasets)
            }
        })
}

addUseCases()