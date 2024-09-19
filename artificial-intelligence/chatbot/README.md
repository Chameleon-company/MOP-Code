# Task 10: Transport Planner Setup Guide

## Introduction

This document outlines the key steps to set up and run the transport planner, including installation instructions and commands to test features.

## Installations

It is recommended to create a virtual environment before installing the required packages to ensure isolation of dependencies. You can follow the commands below:

```bash
pip install rasa-sdk
pip install spacy
python -m spacy download en_core_web_md
python -m spacy link en_core_web_md en
pip install tensorflow
```
## Setting Up Rasa

After getting the rasa files setup, you first need to train the model. For all of these rasa commands, please run them from the main directory `mpt_bot` via the CLI.
To train the rasa model:
```bash
rasa train
```
Every time you update the rasa files its recommended to do this.

To test the model in the CLI, you can run:
```bash
rasa shell
```

## Activating the model
After training is completed, you can run these commands to activate rasa.

1) To activate rasa actions:
```bash
rasa run actions
```

2) Once this is completed, to start the model in a separate CLI in the same virtual environment, run:
```bash
rasa run --enable-api --cors "*"
```

3) To access the UI on a simple HTTP server run:
```bash
python -m http.server
```

Then, open a web browser and navigate to http://localhost:8000 (or whichever URL your server is set to).

## Commands for testing the Transport Planner

These are some example commmands to then run in rasa's shell in the CLI, or in the hosted UI.

### Route planning
What’s the best way to go from Box Hill to Glenferrie? <br>
Which train should I take to get from Camberwell to Canterbury? <br>
 
### Schedule information
What time does the next train to Ringwood leave from Flinders Street? <br>
When is the next train from Ringwood to Parliament station? <br>
 
### Transfers and connection
How do I get from Flinders Street to Melbourne Central? <br>
How many transfers are there between North Melbourne to Hawthorns? <br>
 
### Route optimisation
Which route has the fewest stops between Ringwood and Parliament station? <br>
What is the route with the least number of stops from Dandenong station to Parliament station? <br>

### Directions from user location to specified destination with map
**Ask the chatbot for guidance** <br>
Prompt: Guide me to a location

**When prompted provide your location e.g:** <br>
Prompt: Southbank <br>
Prompt: St Kilda <br>
**And your destination e.g:** <br>
Prompt: Mcg <br>
Prompt: Docklands <br>