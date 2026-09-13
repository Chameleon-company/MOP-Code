# Crack Detection — Dashboard (Team Member 2)

## Overview

The dashboard servers as the user interface for the project. A user can open the dashboard and visually interact with the database through the table and easily upload images and have crack reports generated and added to the database.

## Files

- **src:**
  - **App.css:** Styling for dashboard
  - **App.tsx:** The route component for the dashboard.
  - **index.css:** Global styles
  - **main.tsx**: Just the javascript entry point
  - **Components:**
    - **drawer.tsx:** Contains all logic for draw (Page navigation in upper left corner)
    - **multipleUpload.tsx:** Handles logic for upload page
    - **noReportTable.tsx:** Handles logic and rendering of table in the Generate Reports page. (Names as such because it only renders entries with no report)
    - **tableSelect.tsx:** Handles logic and rendering for home page table
  - **Pages:**
    - **batchUpload.tsx**: The upload image page
    - **generateReports.tsx:** The Generate Reports page
    - **Home.tsx:** The dashboards home page

## How to use:

### Prerequisite:

The API in the main project directory must have all dependencies installed.

### Step 1 - Clone and setup project

Clone project, then navigate to the Dashboard directory and run:

```cmd
npm install
```

### Step 2 - Run API

Run the API that is in the main directory (api.py) and wait for it to start.

### Step 3 - Run Dashboard

```cmd
npm run dev
```

### Step 4 - Use Dashboard

The dashboard can now be used normally and all features should work!
