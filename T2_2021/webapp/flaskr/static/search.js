let technologyIcon = {
    javascript: 'fa-js-square',
    python: 'fa-python'
}

let input = document.querySelector('.search')
input.addEventListener('input', showResults)

// convert article to string so that its super easy to
// use in a search
function toString(article) {
    let combined = article.title + " " +
    article.description + " " + 
    article.tags.join(" ") + 
    article.technology
    .map(technology=> technology.name + " " + technology.code)
    .join(" ")

    return combined.toLowerCase()// to lower for case-insensitive search results
}

function showResults(event) {

    let searchValue = event && event.target && event.target.value || ''
    let searchTerms = searchValue.toLowerCase().split(' ')
    
    let results = searchResults.filter(article => {
        let articleSearch = toString(article)
        // if we have a search term, then we search based on that term
        if(searchTerms.some(x => x.length))
            return searchTerms.some(term => articleSearch.includes(term))
        else
            // if we have no search term, then we show all
            return true
    })

    // Instantiate the table with the existing HTML tbody
    let tileAncestor = document.querySelector(".is-ancestor");
    tileAncestor.innerHTML = ""
    for(let index in results){
        let article = results[index]
        let articleNode = createArticle(article)
        tileAncestor.appendChild(articleNode)
    }
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
    let description = articleTile.querySelectorAll('.description')[0]
    description.textContent = article.description
    
    // add technolgy information to the article tile
    let technologyTable = articleTile.querySelectorAll('.table')[0]
    for(let techIndex in article.technology) {
        let technology = article.technology[techIndex]
        // set the technology name
        let technologyRow = technologyRowTemplate.content.cloneNode(true)
        let td = technologyRow.querySelectorAll("td");
        td[0].textContent = technology.name;

        // create icon
        let icon = iconTemplate.content.cloneNode(true)
        let italic = document.createElement("I")
        italic.classList = "fab " +technologyIcon[technology.code]
        icon.querySelector('.icon').appendChild(italic)

        td[1].appendChild(icon)

        // add row to table
        technologyTable.appendChild(technologyRow)
    }

    //return the article
    return articleTile;
}

var searchResults = []
fetch(searchUrl)
  .then(response => response.json())
  .then(result => searchResults = result)
  .then(() => showResults(''));
