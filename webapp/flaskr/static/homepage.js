// Templates for each row of the tables
var rowTemplateUseCase = "<tr class='row-bottom-border-usecase' style='color: var(--default-text-color);'><td>{{name}}</td><td class='level-col'>{{difficulty}}</td><td>{{link}}</td></tr>";
var rowTemplateDataset = "<tr class='row-bottom-border-dataset' style='color: var(--default-text-color);'><td>{{name}}</td><td>{{difficulty}}</td></tr>";

// Values for the use-case table
var tableRowsInitial = 4;
var usecaseRows = 0;
var useCaseTable;
var globalDataUseCases;
const smUsecases = "show-more-use-cases";
var smUsecasesCount = 0;
const usecaseRowClass = "row-bottom-border-usecase";
var currentFilter = "None";
var smEnable = true;

// Values for the dataset table
var tableRowsInitialDataset = 4;
var datasetRows = 0;
var datasetTable;
var globalDataDataset;
const smDatasets = "show-more-datasets";
var smDatasetsCount = 0;
const datasetRowClass = "row-bottom-border-dataset";

/**
 * Generates the HTML code for a new row for the use-case table, based off of the template
 * 
 * @param {*} name          The "title" field of the use-case
 * @param {*} difficulty    The "difficulty" field of the use-case
 * @param {*} link          The "name" field of the use-case
 * @returns                 The HTML code for a new row for the use-case table
 */
function createNewRowUsecase(name, difficulty, link) {
    usecaseRows++;
    // Replace the placeholders in the template with the actual data
    if (window.innerWidth <= 768){
        return rowTemplateUseCase.replace("{{name}}", "<div class='usecase-col no-underline-link'><a href='/use-cases/" + link + "'>" + name + "</a></div>")
                            .replace("{{difficulty}}", "<div class='dot " + difficulty.toLowerCase() + "'></div>")
                            .replace("{{link}}","<div class='link-col no-underline-link'><a href='/use-cases/" + link + "'>➤</a></div>");
    }
    else{
        return rowTemplateUseCase.replace("{{name}}", "<div class='usecase-col no-underline-link'><a href='/use-cases/" + link + "'>" + name + "</a></div>")
                            .replace("{{difficulty}}", "<div class='" + difficulty.toLowerCase() + " bubble'>" + difficulty + "</div>")
                            .replace("{{link}}","<div class='link-col no-underline-link'><a href='/use-cases/" + link + "'>➤</a></div>");
    }
}
window.addEventListener("resize", function() {
    var rows = document.getElementsByClassName("usecase-row");
  
    for (var i = 0; i < rows.length; i++) {
      var name = rows[i].getAttribute("data-usecase-name");
      var difficulty = rows[i].getAttribute("data-usecase-difficulty");
      var link = rows[i].getAttribute("data-usecase-link");
      rows[i].innerHTML = createNewRowUsecase(name, difficulty, link);
    }
  });



/**
 * Expands the use-case table to show all of them. Then, replaces the "Show more" button with a "Show less" one
 */
function showmoreUseCases() {

    if (smEnable) {
        // Create a border for the initial final row
        updateBottomBorder(usecaseRowClass, 1);

        // Add the remaining use cases to the table
        for (let i = usecaseRows; i < globalDataUseCases.length; i++) {
            useCaseTable.innerHTML += createNewRowUsecase(globalDataUseCases[i].title,globalDataUseCases[i].difficulty,globalDataUseCases[i].name);
        }

        // Remove the border from the final row
        updateBottomBorder(usecaseRowClass, 0);

        // Replace the "Show more" link with a "Show less" one
        toggleShowButton(smUsecases);
    }
}

/**
 * Reduces the use-case table back to the initial size of the table and then replaces the "Show less" button
 * with a "Show more" one.
 */
function showlessUseCases() {

    if (smEnable) {
        while (usecaseRows > tableRowsInitial) {
            useCaseTable.deleteRow(-1);
            usecaseRows--;
        }
    
        // Remove the border from below the new final row on the table
        updateBottomBorder(usecaseRowClass, 0);
    
        // Replace the "Show less" link with a "Show more" one
        toggleShowButton(smUsecases);
    }
}

/**
 * Retrieves the relevant information needed for the use-case table from the json file. Creates the initial table after
 * doing so. 
 */
function initialUseCases() {
    useCaseTable = document.getElementById("use-case-table");

    // Read in the specified json file and create the rows of the table
    fetch(`${$SCRIPT_ROOT}/static/search.json`)
        .then((response) => response.json())
        .then ((data) => {
            globalDataUseCases = data;
            addUseCases();
        });
}

/**
 * Creates the use-case table until it reaches the specified initial size.
 */
function addUseCases() {
    let tablesize = tableRowsInitial;
    if (smUsecasesCount % 2 == 1) {
        tablesize = globalDataUseCases.length;
    }

    for (let i = 0; i < tablesize; i++) {
        useCaseTable.innerHTML += createNewRowUsecase(globalDataUseCases[i].title,globalDataUseCases[i].difficulty,globalDataUseCases[i].name);
    }

    // Remove the bottom border from the final row
    updateBottomBorder(usecaseRowClass, 0);
}

/**
 * Updates the final row in one of the tables on the home page, specified by the CSS class used for the rows in
 * each table.
 * 
 * @param {string} cssClass The CSS class which we want to update the final row for
 * @param {number} option   "1" to add a border to the final row, otherwise the final row's border will be removed
 */
function updateBottomBorder(cssClass, option) {
    let children = document.getElementsByClassName(cssClass);
    let lastChild = children[children.length-1];
    if (option == 1) {
        lastChild.style.borderBottom = "thin solid";
        lastChild.style.borderColor = "#00cc70";
    } 
    else {
        lastChild.style.borderBottom = "none";
    }
}

/**
 * Changes a specified "Show more" or "Show less" button into the other
 * @param {*} id    The css ID of the button to be changed
 */
function toggleShowButton(id) {
    let Obj = document.getElementById(id);
    let newBtnHTML;
    if (Obj != null) {
        switch (id) {
            case smUsecases:
                // If the button is "Show more", prepare to replace it with "Show less"
                if (smUsecasesCount % 2 == 0) {
                    newBtnHTML = '<div id="show-more-use-cases"><a href="javascript:showlessUseCases()">Show less ↑</a></div>';
                }
                // Otherwise, prepare to replace the "Show less" button with "Show more"
                else {
                    newBtnHTML = '<div id="show-more-use-cases"><a href="javascript:showmoreUseCases()">Show more ↓</a></div>';
                }
                smUsecasesCount++;
                break;
            case smDatasets:
                // Same as above, but for the Datasets "Show more" button
                if (smDatasetsCount % 2 == 0) {
                    newBtnHTML = '<div id="show-more-datasets"><a href="javascript:showlessDatasets()">Show less ↑</a></div>';
                }
                else {
                    newBtnHTML = '<div id="show-more-datasets"><a href="javascript:showmoreDatasets()">Show more ↓</a></div>';
                }
                smDatasetsCount++;
                break;
            default:
                break;
        }
        if (newBtnHTML != null) {
            if (Obj.outerHTML) { 
                // Before doing this, check that the browser supports OuterHTML as this makes it a lot easier
                Obj.outerHTML = newBtnHTML;
            }
            else {
                // Otherwise, use this alternative method for browser support
                let tmpObj = document.createElement('div');
                tmpObj.innerHTML = '<!--To be replaced-->';
                ObjParent=Obj.parentNode;
                ObjParent.replaceChild(tmpObj, Obj);
                ObjParent.innerHTML=ObjParent.innerHTML.replace('<div><!--To be replaced--></div>',newBtnHTML);
            }
        }
    }
}

/**
 * Recreates the use-case table to only show use-cases of the specified difficulty
 * 
 * @param {string} difficulty   Use cases with this difficulty will be shown
 */
function filterDifficulty(difficulty) {
    // Remove all rows
    while(usecaseRows > 0) {
        useCaseTable.deleteRow(-1);
        usecaseRows--;
    }

    // If we are disabling the difficulty filter, just recreate the initial table
    if (currentFilter == difficulty) {
        addUseCases();
        currentFilter = "None";
        smEnable = true;
    }
    else {
        // Add in all rows that match the specified difficulty
        for (item in globalDataUseCases) {
            if (globalDataUseCases[item].difficulty.includes(difficulty)){
            useCaseTable.innerHTML += createNewRowUsecase(globalDataUseCases[item].title,globalDataUseCases[item].difficulty,globalDataUseCases[item].name);
            }
        }

        // Update what the filter is currently set to
        currentFilter = difficulty;

        // Disable the "Show more" and "Show less" buttons
        smEnable = false;

        // Remove the border from the bottom row
        updateBottomBorder(usecaseRowClass, 0);
    }

}

// *************************************************************************
// ***********************  DATASET TABLE FUNCTIONS ************************
// *************************************************************************

/**
 * Generates the HTML code for a new row for the dataset table, based off of the template
 * 
 * @param {*} New       The "Name" property of the dataset entry
 * @param {*} links     The "Downloads" property of the dataset entry
 * @param {*} url           The "Permalink" property of the dataset entry
 * @returns                 The HTML code for a new row for the dataset table
 */
function createNewRowDataset(dataset) {
    
    datasetRows++;
    
    return rowTemplateDataset.replace("{{name}}" , dataset)
                             .replace("{{difficulty}}", "<div class='advanced bubble'><a href='https://data.melbourne.vic.gov.au/explore/dataset/" + dataset + "/export/'' target='blanks' ><img src='/static/download-button.png' style='width: 15px; height: 15px;'></a></div>");
                             
    }


/**
 * Creates an initial table of datasets and their download links with an amount of rows specified by the
 * "tableRowsInitialDataset" global variable. Additionally, stores the dataset information in a global variable.
 */
function addDatasets() {
    datasetTable = document.getElementById("dataset-table");
    fetch(`${$SCRIPT_ROOT}/static/search.json`)
        .then((response) => response.json())
        .then((data) => {
            globalDataDataset = data;
            
                for (let j = 0; j < tableRowsInitialDataset; j++){
                let datasetName = globalDataDataset[0].dataset[j]

                if (datasetName.indexOf("(") > -1) {
                    datasetTable.innerHTML += createNewRowDataset(datasetName.substring(0,datasetName.indexOf("(")))
                } else {
                    datasetTable.innerHTML += createNewRowDataset(datasetName)
                }
            
            }
        })
}

/**
 * Expands the dataset table to show all the dataset entries. Then, replaces the "Show more" button with a "Show less" one
 */
function showmoreDatasets() {
    
    datasetTable = document.getElementById("dataset-table");
    fetch(`${$SCRIPT_ROOT}/static/search.json`)
        .then((response) => response.json())
        .then((data) => {
            globalDataDataset = data;
    
        for (let i = 1; i < data.length; i++) {
            for (let j = 0; j < data.length ; j++){
                let datasetName = globalDataDataset[i].dataset[j]
                // let datasetDownloads = globalDataDataset[i].datasetlinks[j]
                
                // let datasetURL = globalDataDataset[i].Permalink
                
                    
                
                if (datasetName != undefined) {
                    datasetTable.innerHTML += createNewRowDataset(datasetName)
                } 
            }
            }
        // Remove the border the new final row
        })
        // Replace the "Show more" link with a "Show less" one
        toggleShowButton(smDatasets);
}


// // /**
// //  * Reduces the dataset table back to its initial size. Then, replaces the "Show less" button with a "Show more" one
// //  */
function showlessDatasets() {
    while (datasetRows > tableRowsInitialDataset) {
        datasetTable.deleteRow(-1);
        datasetRows--;
    }

    // Remove the border from below the new final row on the table
    updateBottomBorder(datasetRowClass, 0);

    // Replace the "Show less" link with a "Show more" one
    toggleShowButton(smDatasets);
}

initialUseCases()

addDatasets()