from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from typing import Any, Text, Dict, List
import pandas as pd
import subprocess
import os
import logging


logger = logging.getLogger(__name__)

class ActionRunMappingScript(Action):

    def name(self) -> Text:
        return "action_run_mapping_script"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        try:
            #Full path to the Python executable in conda env
            python_executable = r"C:\Users\logan\anaconda3\envs\rasa_env\python.exe"
            
            #Full path to the script
            script_path = r"C:\Users\logan\Desktop\Uni\Team proj\base model with map\actions\userlocationsmaps_executable.py"
            
            #Set up the environment
            env = os.environ.copy()
            env["PATH"] = r"C:\Users\logan\anaconda3\envs\rasa_env;" + env["PATH"]
            
            #Use subprocess to run the external Python script
            result = subprocess.run([python_executable, script_path], capture_output=True, text=True, env=env)

            if result.returncode == 0:
                dispatcher.utter_message(text="The mapping script has been executed successfully.")
                logger.info(f"Script output:\n{result.stdout}")
            else:
                dispatcher.utter_message(text=f"Script execution failed with error: {result.stderr}")
        except Exception as e:
            dispatcher.utter_message(text=f"An error occurred: {str(e)}")

        return []

# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []
