# Task 10: Transport Planner Setup Guide

## Introduction

This document outlines the key steps to set up and run the transport planner, including installation instructions and commands to test features.

## Installations

It is recommended to create a virtual environment before installing the required packages to ensure isolation of dependencies. Use the `environment.yml` file in the repo to create an environment. Details on setting up a conda environment can be found here:

[Conda User Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

Also, to familiarize yourself with Rasa, it is worthwhile to watch this video:
[Rasa Introduction Video](https://www.youtube.com/watch?v=Co7QtrJBkpY)


If you prefer to make your own environment without the yml file, you can use these commands:

```bash
pip install rasa
pip install rasa-sdk
pip install spacy
python -m spacy download en_core_web_md
python -m spacy link en_core_web_md en
pip install tensorflow


```
## Setting Up Rasa

For these commands, some are run from the Anaconda Prompt, whereas others are run via the standard Command Prompt (on Windows). 

After getting the rasa files setup, you first need to train the model. For all of these rasa commands, please run them from the main directory `mpt_bot` via the Anaconda Prompt.
To train the rasa model:
```bash
rasa train
```
Every time you update the rasa files its recommended to do this.

To test the model in the Anaconda Prompt, you can run:
```bash
rasa shell
```

## Activating the model
After training is completed, you can run these commands to activate rasa (within Anaconda prompt).

1) To activate rasa actions:
```bash
rasa run actions
```

2) Once this is completed, to start the model in a separate CLI in the same virtual environment, run:
```bash
rasa run --enable-api --cors "*"
```

3) To access the UI on a simple HTTP server run the following command, within the standard Command Prompt:
```bash
python -m http.server 8080
```

Then, open a web browser and navigate to http://localhost:8080 (or whichever URL your server is set to).

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
