## Saving to this folder

Only save your notebook to this folder if you have downloaded a copy of another **published** notebook that you have reworked.

This could be:

* Changing the API.
* Redefining the use case.
* Code library corrections.
* Anything that is a valid reason to download and modify an *existing notebook*.

# WHAT TO INCLUDE

* The updated notebook.
* The updated notebook in HTML form.
* The accompyaning JSON file that goes with a use case.
* A folder with the same name as your use case (if you have external files)
* * Inside that folder have a folder named "images" or "src" which includes any external files needed for your notebook to work.  

# Meta JSON File
Create a json file for the use case wuth the following information:

* title
* file name 
* description
* tags
* difficulty
* technology (This  is the python packages you have used)
* datasets

### Expected output

` 
    
    { 
        "title": "This is the title of your use case",
        "name": "usecase_version_1",
        "description": "add a brief description. e.g. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Praesent eu tempor nulla, sit amet ultricies justo. Fusce felis erat, pretium in magna a, elementum gravida odio. Duis sit amet tortor.",
        "tags": ["geojson", "folium", "bicucles", "liveability", "safety"],
        "difficultly": "intermediate",
        "technology": [{
            "name": "numpy",
            "code": "python"
        },{
            "name": "pandas",
            "code": "python"
        },{
            "name": "geopandas",
            "code": "python"
        },{
            "name": "folium",
            "code": "python"
        }],
        "datasets":["coworking-spaces", "trees-with-species-and-dimensions-urban-forest", "public-barbecues"] 
    }

`

# Web Dev Team!

Copy all 3 files (and the folder if it exists) into the **PUBLISHED** folder when you have taken a copy of the project for publishing.

Follow the instructions provided on how to publish a notebook, and what changes need to be made.

<br>
<hr>

*Written by Angie Hollingworth 30 April 2023*