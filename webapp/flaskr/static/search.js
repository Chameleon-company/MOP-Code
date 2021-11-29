let technologyIcon = {
    javascript: 'fa-js-square',
    python: 'fa-python'
}

let input = document.querySelector('.search')
input.addEventListener('input', insertSearchResults)

// convert article to string so that its super easy to
// use in a search
function toString(article) {
    let combined = article.title + " " +
        article.description + " " +
        article.tags.join(" ") +
        article.technology
        .map(technology => technology.name + " " + technology.code)
        .join(" ")

    return combined.toLowerCase() // to lower for case-insensitive search results
}

var timeout = undefined

function insertSearchResults(event) {

    if (timeout) { // debounce the updates for better performance
        clearTimeout(timeout)
    }

    timeout = setTimeout(() => {

        timeout = undefined

        let searchValue = event && event.target && event.target.value || ''
        let searchTerms = searchValue.toLowerCase().split(' ')

        let searchControl = document.querySelector('.search-control')
        searchControl.classList += " is-loading"

        //remove old results
        let articleTileParents = document.querySelectorAll(".article-tile-parent");
        for (let index in articleTileParents) {
            if (index > 0)
                articleTileParents[index].remove()
        }
        let articleTileParent = articleTileParents[0]
        articleTileParent.innerHTML = ""
            // get new results
        let articlePromise = searchArticles(searchTerms)
            .then((results) => {
                let tileParent = articleTileParent
                for (let index in results.slice(0, 42)) {
                    // should only have 3 nodes per row
                    if (index > 0 && index % 3 == 0) {
                        let newTileParent = tileParent.cloneNode(false)
                        tileParent.parentElement.appendChild(newTileParent)
                        tileParent = newTileParent
                    }

                    let article = results[index]
                    let articleNode = createArticle(article)
                        // articleNode.querySelector('.box').style.backgroundColor = getColor(Math.floor(Math.random()*10) % 4)
                    tileParent.appendChild(articleNode)
                }
            })

        // remove old results
        let datasetTileParents = document.querySelectorAll(".dataset-tile-parent");

        for (let index in datasetTileParents) {
            if (index > 0)
                datasetTileParents[index].remove()
        }
        let datasetTileParent = datasetTileParents[0]
        datasetTileParent.innerHTML = ""
            // lookup datasets
        let datasetPromise = searchDatasets(searchTerms)
            .then((results) => {
                let tileParent = datasetTileParent
                for (let index in results.slice(0, 42)) {
                    // should only have 6 nodes per row
                    if (index > 0 && index % 4 == 0) {
                        let newTileParent = tileParent.cloneNode(false)
                        tileParent.parentElement.appendChild(newTileParent)
                        tileParent = newTileParent
                    }

                    let dataset = results[index]
                    let datasetNode = createDataset(dataset)
                        // datasetNode.querySelector('.box').style.backgroundColor = getColor(Math.floor(Math.random() * 10) % 4)
                    tileParent.appendChild(datasetNode)
                }
            })

        // Finished fetching results
        Promise.all([datasetPromise, articlePromise])
            .finally(() => {
                // remove is loading class
                searchControl.classList = searchControl.classList.toString().replace(' is-loading', '')
            })

    }, 500)
}

function createArticle(article) {

    let articleTemplate = document.querySelector('#article_tile_template');
    let technologyRowTemplate = document.querySelector('#technology_row_template')
    let iconTemplate = document.querySelector('#icon_template')

    // create new article tile
    let articleTile = articleTemplate.content.cloneNode(true);

    // set the title & description
    let title = articleTile.querySelectorAll('.title')[0]
    title.textContent = article.title
    title.href = $SCRIPT_ROOT + "/use-cases/" + article.name
    let description = articleTile.querySelectorAll('.description')[0]
    description.textContent = article.description

    // add technolgy information to the article tile
    let technologyTable = articleTile.querySelectorAll('.table')[0]
    for (let techIndex in article.technology) {
        let technology = article.technology[techIndex]
            // set the technology name
        let technologyRow = technologyRowTemplate.content.cloneNode(true)
        let td = technologyRow.querySelectorAll("td");
        td[0].textContent = technology.name;

        // create icon
        let icon = iconTemplate.content.cloneNode(true)
        let italic = document.createElement("I")
        italic.classList = "fab " + technologyIcon[technology.code]
        icon.querySelector('.icon').appendChild(italic)

        td[1].appendChild(icon)

        // add row to table
        technologyTable.appendChild(technologyRow)
    }

    //return the article
    return articleTile;
}

function createDataset(dataset) {

    let datasetTemplate = document.querySelector('#dataset_tile_template');
    // create new article tile
    let datasetTile = datasetTemplate.content.cloneNode(true);

    // set permalink
    let link = datasetTile.querySelectorAll('.perma')[0]
    link.href = dataset.Permalink

    // set the title & description
    let title = datasetTile.querySelectorAll('.subtitle')[0]
    title.textContent = dataset.Name

    let downloadCount = datasetTile.querySelectorAll('.description')[0]
    downloadCount.textContent = numeral(dataset.Downloads).format('0,0')

    return datasetTile;
}

// Search through json file for results with respect to search terms
function searchArticles(searchTerms) {
    return fetch(`${$SCRIPT_ROOT}/static/search.json`)
        .then(response => response.json())
        .then((searchResults) => {
            let results = searchResults.filter(article => {
                let articleSearch = toString(article)
                    // if we have a search term, then we search based on that term
                if (searchTerms.some(x => x.length))
                    return searchTerms.some(term => articleSearch.includes(term))
                else
                // if we have no search term, then we show all
                    return true
            })
            return results
        })
}

// use API to search for available dataset that match
// search terms
function searchDatasets(searchTerms) {
    if (!searchTerms.length)
        return;
    return fetch(`${$SCRIPT_ROOT}/search/datasets?query=${searchTerms.join(' ')}`)
        .then(result => result.json())
}

insertSearchResults('')