# Urban Parking Demand Forecasting – Handover / Update Needed

## Current Status

This project is functionally complete as a Visual Studio Code prototype.

The current implementation successfully:

* Processes and cleans parking occupancy data.
* Constructs a graph network of parking bays using spatial relationships.
* Generates node features for machine learning.
* Trains a Graph Convolutional Network (GCN).
* Saves and loads trained models (`model.pth`).
* Produces parking occupancy predictions based on area, hour, and day selections.
* Displays predictions through an interactive graph visualisation.

The project has been tested and executed successfully within Visual Studio Code using the provided Python modules and project structure.

No known issues currently prevent the prototype from running in its intended Visual Studio Code environment.

---

## Why This Project Was Moved to UPDATE NEEDED

The project was originally developed as a Python-based application using multiple modular `.py` files and an interactive graph visualisation workflow.

The required final deliverable for website publication was a self-contained Jupyter Notebook (`.ipynb`) solution capable of being converted into website-ready formats such as HTML and JSON.

While an initial notebook conversion was attempted, the project's interactive visualisation architecture relies heavily on Python scripts and desktop-based graph interaction methods. This created compatibility challenges when attempting to migrate the complete user experience into a notebook environment.

As a result, the project remains a fully functional prototype but requires additional work before it can be published to the website.

---

## Completed Components

The following components are complete and operational:

### Data Processing

* Dataset loading
* Data cleaning
* Area mapping
* Feature generation

### Graph Construction

* Parking bays represented as graph nodes
* Spatial relationships represented as graph edges
* Graph generation and validation

### Machine Learning Pipeline

* Feature tensor generation
* Label preparation
* Model training
* Model evaluation
* Model persistence

### Prediction System

* Area selection
* Hour selection
* Day selection
* Occupancy prediction generation

### Visualisation

* Graph generation
* Occupancy visualisation
* Prediction display
* Interactive node selection within the Visual Studio Code implementation

---

## Remaining Work Required

### Priority 1 – Notebook Conversion

Convert the Visual Studio Code workflow into a fully notebook-based workflow.

The final notebook should:

* Execute the entire pipeline from start to finish.
* Include explanatory markdown cells.
* Allow users to run the workflow without manually executing multiple Python scripts.
* Produce outputs directly inside the notebook.

---

### Priority 2 – Web-Compatible Visualisation

The original implementation uses desktop-based matplotlib interaction.

Future work should investigate replacing or adapting the visualisation layer using notebook-compatible or web-compatible technologies such as:

* Plotly
* Bokeh
* PyVis
* Other browser-based graph visualisation frameworks

The goal is to replicate the existing interactive graph experience within a notebook and website environment.

---

### Priority 3 – Publication Assets

Create the required publication outputs:

* Final `.ipynb` notebook
* Exported HTML version
* JSON output (if required by publishing workflow)

---

## Recommended Starting Point for Future Team

Future contributors should begin by reviewing:

```text
src/predict_by_area.py
src/train.py
src/graph.py
src/features.py
```

These files contain the core functionality of the project.

Once understood, focus should shift toward migrating the existing workflow into a notebook-based implementation rather than rebuilding the machine learning pipeline.

---

## Summary

The machine learning model, graph construction process, prediction workflow, and Visual Studio Code implementation are complete and operational.

The primary outstanding task is adapting the existing prototype into a notebook-based, website-ready format while preserving the interactive visualisation functionality.
