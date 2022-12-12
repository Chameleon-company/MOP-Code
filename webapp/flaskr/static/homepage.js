// First, define the template for the new rows as a string
var rowTemplateUseCase = "<tr{{border}}><td>{{name}}</td><td class='level-col'>{{difficulty}}</td><td>{{link}}</td></tr>";
var rowTemplateDataset = "<tr{{border}}><td>{{name}}</td><td>{{difficulty}}</td></tr>";

// Define values for the use-case table
var tableRowsInitial = 4;
var tableRowsTotal;
var tableRowsTarget = tableRowsInitial;
var usecaseRows = 0;
var useCaseTable;
var globalDataUseCases;


// Define values for the dataset table
var tableRowsInitialDataset = 4;
var tableRowsTotalDataset;
var datasetRows = 0;
var globalDataDataset;
var datasetTable;


// Function for generating a new row based on the dataset table row template
function createNewRowDataset(name, downloads, url) {
    datasetRows++;
    let row = rowTemplateDataset.replace("{{name}}", name)
                      .replace("{{difficulty}}", "<div class='advanced bubble'><a href='#'>" + downloads + "</a></div>");

    if (datasetRows == tableRowsTotalDataset) {
        return row.replace("{{border}}"," id='row-final-dataset'")
    }
    else if (datasetRows == tableRowsInitialDataset) {
        return row.replace("{{border}}"," id='row-initial-final-dataset'");
    }
    else {
        return row.replace("{{border}}"," class='row-bottom-border'");
    }
  }


// Next, define a function for generating a new row based on the template
// and some provided data
function createNewRowUsecase(name, difficulty, link) {
    usecaseRows++;
  // Replace the placeholders in the template with the actual data
  let row = rowTemplateUseCase.replace("{{name}}", name)
                        .replace("{{difficulty}}", "<div class='" + difficulty.toLowerCase() + " bubble'>" + difficulty + "</div>")
                        .replace("{{link}}","<div class='link-col'><a href='" + link + "'>➤</a></div>");
    
    if (usecaseRows == tableRowsTotal) {
        return row.replace("{{border}}"," id='row-final'");
    }
    else if (usecaseRows == tableRowsInitial) {
       return row.replace("{{border}}"," id='row-initial-final'")
    }
    else {
        return row.replace("{{border}}"," class='row-bottom-border'")
    }
}

// Removes the final (bottom-most) row from the specified table and reduces the count of rows
function deleteFinalRow(table) {
    table.deleteRow(-1);
    usecaseRows--;
}

// The function that executes when "Show more" is pressed
// This will first continue creating the table using the search.json data,
// then it will replace the "Show more" link with a "Show less" one.
function showmoreUseCases() {
    // Create a border for the initial final row
    let rowInitialFinal = document.getElementById("row-initial-final");
    rowInitialFinal.style.borderBottom = "thin solid";
    rowInitialFinal.style.borderColor = "#00cc70";

    // Add the remaining use cases to the table
    for (let i = usecaseRows; i < globalDataUseCases.length; i++) {
        useCaseTable.innerHTML += createNewRowUsecase(globalDataUseCases[i].title,globalDataUseCases[i].difficulty,globalDataUseCases[i].name);
    }

    // Replace the "Show more" link with a "Show less" one
    var showLessBtnHTML = '<div id="show-more-use-cases"><a href="javascript:showlessUseCases()">Show less ↑</a></div>'
    var Obj = document.getElementById("show-more-use-cases");
    if (Obj.outerHTML) { 
        // Before doing this, check that the browser supports OuterHTML as this makes it a lot easier
        Obj.outerHTML = showLessBtnHTML;
    }
    else {
        // Otherwise, use this alternative method for browser support
        var tmpObj = document.createElement('div');
        tmpObj.innerHTML = '<!--To be replaced-->';
        ObjParent=Obj.parentNode;
        ObjParent.replaceChild(tmpObj, Obj);
        ObjParent.innerHTML=ObjParent.innerHTML.replace('<div><!--To be replaced--></div>',showLessBtnHTML);
    }
}

// This function executes when "Show less" is pressed.
// It will reduce the table down to the initial size, before "Show more" was
// pressed. Then, it will replace the "Show less" link with a "Show more" one.
function showlessUseCases() {
    tableRowsTarget = tableRowsInitial;
    let rowsToRemove = tableRowsTotal - tableRowsInitial;
    for (let i = 0; i < rowsToRemove; i++) {
        deleteFinalRow(useCaseTable);
    }

    // Remove the border from below the final use case on the table
    document.getElementById("row-initial-final").style.borderBottom = "none";

        // Replace the "Show less" link with a "Show more" one
        let showMoreBtnHTML = '<div id="show-more-use-cases"><a href="javascript:showmoreUseCases()">Show more ↓</a></div>'
        let Obj = document.getElementById("show-more-use-cases");
        if (Obj.outerHTML) { 
            // Before doing this, check that the browser supports OuterHTML as this makes it a lot easier
            Obj.outerHTML = showMoreBtnHTML;
        }
        else {
            // Otherwise, use this alternative method for browser support
            let tmpObj = document.createElement('div');
            tmpObj.innerHTML = '<!--To be replaced-->';
            ObjParent=Obj.parentNode;
            ObjParent.replaceChild(tmpObj, Obj);
            ObjParent.innerHTML=ObjParent.innerHTML.replace('<div><!--To be replaced--></div>',showMoreBtnHTML);
        }
}

// Create an initial smaller table of use cases and store the information read from the json file as a global variable
function initialUseCases() {
    useCaseTable = document.getElementById("use-case-table");
    fetch(`${$SCRIPT_ROOT}/static/search.json`)
        .then((response) => response.json())
        .then ((data) => {
            globalDataUseCases = data;
            for (let i = 0; i < tableRowsInitial; i++) {
                useCaseTable.innerHTML += createNewRowUsecase(globalDataUseCases[i].title,globalDataUseCases[i].difficulty,globalDataUseCases[i].name);
            }
            tableRowsTotal = globalDataUseCases.length;
        });
}

// *************************************************************************
// **********************  DATASET TABLE CODE ******************************
// *************************************************************************

function addDatasets() {
    datasetTable = document.getElementById("dataset-table");
    fetch(`${$SCRIPT_ROOT}/search/datasets?query`)
        .then((response) => response.json())
        .then((data) => {
            globalDataDataset = data;
            tableRowsTotalDataset = globalDataDataset.length;
            for (let i = 0; i < tableRowsInitialDataset; i++) {
                let datasetName = globalDataDataset[i].Name
                let datasetDownloads = globalDataDataset[i].Downloads
                let datasetURL = globalDataDataset[i].Permalink
                if (datasetName.indexOf("(") > -1) {
                    datasetTable.innerHTML += createNewRowDataset(datasetName.substring(0,datasetName.indexOf("(")),datasetDownloads)
                } else {
                    datasetTable.innerHTML += createNewRowDataset(datasetName, datasetDownloads)
                }
            }
        })
}

function showmoreDatasets() {
    // Create a border for the initial final row
    let rowInitialFinal = document.getElementById("row-initial-final-dataset");
    rowInitialFinal.style.borderBottom = "thin solid";
    rowInitialFinal.style.borderColor = "#00cc70";

    for (let i = datasetRows; i < tableRowsTotalDataset; i++) {
        let datasetName = globalDataDataset[i].Name
        let datasetDownloads = globalDataDataset[i].Downloads
        let datasetURL = globalDataDataset[i].Permalink
        if (datasetName.indexOf("(") > -1) {
            datasetTable.innerHTML += createNewRowDataset(datasetName.substring(0,datasetName.indexOf("(")),datasetDownloads, datasetURL)
        } else {
            datasetTable.innerHTML += createNewRowDataset(datasetName, datasetDownloads, datasetURL)
        }
    }

        // Replace the "Show more" link with a "Show less" one
        let showLessBtnHTML = '<div id="show-more-datasets"><a href="javascript:showlessDatasets()">Show less ↑</a></div>'
        let Obj = document.getElementById("show-more-datasets");
        if (Obj.outerHTML) { 
            // Before doing this, check that the browser supports OuterHTML as this makes it a lot easier
            Obj.outerHTML = showLessBtnHTML;
        }
        else {
            // Otherwise, use this alternative method for browser support
            let tmpObj = document.createElement('div');
            tmpObj.innerHTML = '<!--To be replaced-->';
            ObjParent=Obj.parentNode;
            ObjParent.replaceChild(tmpObj, Obj);
            ObjParent.innerHTML=ObjParent.innerHTML.replace('<div><!--To be replaced--></div>',showLessBtnHTML);
        }
}

function showlessDatasets() {
    while (datasetRows > tableRowsInitialDataset) {
        datasetTable.deleteRow(-1);
        datasetRows--;
    }

    // Remove the border from below the final use case on the table
    document.getElementById("row-initial-final-dataset").style.borderBottom = "none";

    // Replace the "Show less" link with a "Show more" one
    let showMoreBtnHTML = '<div id="show-more-datasets"><a href="javascript:showmoreDatasets()">Show more ↓</a></div>'
    let Obj = document.getElementById("show-more-datasets");
    if (Obj.outerHTML) { 
        // Before doing this, check that the browser supports OuterHTML as this makes it a lot easier
        Obj.outerHTML = showMoreBtnHTML;
    }
    else {
        // Otherwise, use this alternative method for browser support
        var tmpObj = document.createElement('div');
        tmpObj.innerHTML = '<!--To be replaced-->';
        ObjParent=Obj.parentNode;
        ObjParent.replaceChild(tmpObj, Obj);
        ObjParent.innerHTML=ObjParent.innerHTML.replace('<div><!--To be replaced--></div>',showMoreBtnHTML);
    }
}

initialUseCases()
addDatasets()

