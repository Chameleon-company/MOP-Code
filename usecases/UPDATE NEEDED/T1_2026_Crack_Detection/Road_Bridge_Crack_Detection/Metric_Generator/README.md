# Crack Detection — Crack Metric Generation Module (Team Member 2)

## Overview

This module take the binary mask generated from module 1 (Crack Detection) and generates a crack metric report about the cracks in the image. It does this by running BFS and Skeleonization algorithms to generate metrics about the crack, these metrics and other figures are obtained about the crack are then used to generate a report.

## Pipeline

Binary Mask is uploaded → generateMetricReport() → Report is returned

---

## Files

| File               | Purpose                                                                                                                                             |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `crackAnalyser.py` | Contains all logic and methods for the pipeline                                                                                                     |
| `severity.py`      | Methods used for generating severity metric. crackAnalyser.py implements these methods through an interface so that they can be easily changed out. |
| `massUploader.py`  | Used for testing. Allows an entire folder worth of binary masks to be uploaded to the associated                                                    |

---

## Setup

### Step 1 - Install dependencies

```cmd
pip install pillow scikit-image numpy requests
```

Thats it!

## How to use

### Step 1

Create a folder in the same directory as the crackAnalyser.py and name it "masks". Please as many binary crack masks in the folder as you wish and then run the crackAnalyser.py program. It will then process and generate reports for every mask in the folder.
