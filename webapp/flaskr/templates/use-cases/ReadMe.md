# Translate Jupyter Notebooks to Use Cases

1. First execute ```jupyter nbconvert `file.ipynb` --to html```
1. Copy the converted HTML file into a new flask template file
1. Place the `{% extends 'base.html' %}` preprocessor at the top of the file
1. Remove `Doctype` & `html` tags
1. Find and replace `{{` with `{` and vice versa for `}}` (CAREFUL: only do this in code beginning with **var**, e.g. **var gd** or **var x**. If you accidentally replace *all* `{{` and `}}`, it will break other parts of the code). You can also enclose these **var** code blocks containing '{{ }}' in `{% raw %} ... {% endraw %}` blocks.
1. Place the styles and script tags in the header content block `{% block head %}...{% endblock %}`. Be sure to remove any `<head>`, `<meta>`, and `<title>` tags from the content.
1. Place the playground libraries into the head content block
    ```<!-- Load in the highlight js -->
    <link href="{{ url_for('static', filename='highlight/styles/default.min.css') }}" rel="stylesheet" />
    <script src="{{ url_for('static', filename='highlight/highlight.min.js') }}"></script>
    ```
1. Place the content in the content block `{% block content %}...{% endblock %}`
1. Add the highlight init js to the bottom of content block:
    ```
    <script>
        hljs.highlightAll();
    </script>
    ```
1. Change the `<body>` tags into a `<div>` and add the `container` class to that div.
1. Remove `body { ... }` styles from the header css.
1. Remove `a{ ... }` & `a:hover{ ... }` styles from header css.
1. Remove `.jp-Notebook {...}` styles from the header css.
1. Remove all Reset CSS under sections with comments ~ `Override Blueprint's _reset.scss styles`
1. Update css variable variable to `--jp-content-font-size1: 16px`
1. Improve the code snippet styles:
    1. Add the following header below the div with the ```jp-CodeMirrorEditor``` class:
        ```
            <div class="snippet-language">
                <span class="icon"><i class="fab fa-python"></i></span>
                <span class=>Python</span>
            </div>
        ```
    1. Remove the wrapping div with class `CodeMirror` but leave the content.
    1. Replace `<div` with class `highlight` with a figure with only a `highlight` class
    1. Add a parent node `<code class="language-python" data-lang="python">...</code>` around the content in the `<pre>...</pre>` tag.  
    Result should be `<pre><code ...>...</code></pre>` block.
    1. Repeat for all code sections
    1. LAST STEP - Add the search information / tile information to the `flaskr/static/search.json` file so that the new use cases shows up in the search results in the home page.