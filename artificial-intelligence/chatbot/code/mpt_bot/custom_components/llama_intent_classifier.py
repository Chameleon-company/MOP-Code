from rasa.nlu.components import Component
from rasa.nlu.model import Metadata
from typing import Any, Dict, List, Text
import requests  # Use an API call to your Llama server


class LlamaIntentClassifier(Component):
    """A custom intent classifier using Llama"""

    name = "llama_intent_classifier"
    provides = ["intent"]
    requires = ["text"]
    defaults = {}
    language_list = ["en"]

    def __init__(self, component_config: Dict[Text, Any] = None):
        super().__init__(component_config)

    def train(self, training_data, config, **kwargs):
        """No training needed as Llama is pre-trained"""
        pass

    def process(self, message, **kwargs):
        """Classify intent using Llama"""
        user_message = message.data["text"]

        # Call your Llama API for intent classification
        response = requests.post("http://localhost:8000/classify", json={"text": user_message})
        response_data = response.json()

        if "intent" in response_data:
            intent_name = response_data["intent"]
            confidence = response_data.get("confidence", 1.0)  # Default to 100% confidence

            message.set("intent", {"name": intent_name, "confidence": confidence}, add_to_output=True)
        else:
            message.set("intent", {"name": "nlu_fallback", "confidence": 0.0}, add_to_output=True)

    def persist(self, file_name, model_dir):
        """No need to persist anything"""
        return {"model_file": None}
