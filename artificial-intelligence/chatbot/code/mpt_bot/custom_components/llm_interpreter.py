import re
import string
from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from llama_client import LlamaClient
from rapidfuzz import process, fuzz
import json
import random

# Define your training data as a dictionary.
TRAINING_DATA = {
    "request_train_map": [
       "Show me a map of Melbourne train stations.",
       "I need a map of the train stations in Melbourne.",
       "give me train map of Melbourne",
       "gimme Melb train map pleaseee"
    ],
    "check_disruptions_bus": [
       "Are there any disruptions on bus route 863?",
       "Please tell me if there are disruptions on the 903 bus line.",
       "Is the bus service for 626 disrupted today?",
       "there any bus disruption on 732?",
       "tell me disruptions in bus 624"
    ],
    "check_disruptions_tram": [
       "What's happening with trams between Port Melbourne and Box Hill?",
       "There any disruptions with tram 75?",
       "Any issues in the tram line from Deakin University to Flinders Street?",
       "Is the tram service for 70 disrupted today?",
       "Are there any delays or issues with tram 5?",
       "tell me disruptions in tram 6"
    ],
    "find_nearest_tram_stop": [
       "tell me tram stops near '17 Doepel Way Docklands'",
       "trams stops from '143 Russel Street melbourne'",
       "what trams can I take from '56 little collins street melbourne central'",
       "are there any tram stops near '39 Kooringa Way  Port Melbourne",
       "Which tram stop is closest to 567 Collins Street?",
       "Is there a tram near 21 Swanston Street?",
       "Nearest tram from Queen Victoria Market?",
       "Any tram stops close to '10 Bourke Street?'",
       "Where's the closest tram stop to 'Crown Casino?'",
       "How do I get a tram from 'RMIT'?",
       "Can I catch a tram near 'Melbourne Central?'",
       "Are trams available near '101 Flinders Lane'?",
       "Where can I board a tram near '500 La Trobe St'?"
    ],
    "find_nearest_station": [
       "tell me train stations near '95 Wurundjeri Way  Docklands;'",
       "train stations from '143 Russel Street melbourne'",
       "what trains can I take from '56 little collins street melbourne central'",
       "are there any train stations near '56 little collins street melbourne central?'",
       "Where is the nearest train station from '700 Collins St?'",
       "Find me a train stop close to 'Emporium Melbourne'",
       "Closest train station from '61 Cook Street  Port Melbourne'",
       "Any nearby train from 123 Spring Street?"
    ],
    "find_nearest_bus_stop": [
       "tell me bus services near '17 Doepel Way Docklands'",
       "bus stops from '57 Charles Street  East Melbourne",
       "what buses can I take from '4 Southbank Boulevard  Southbank",
       "are there any bus stops near '400 Epsom Road  Flemington",
       "Which bus stop is near 5 Jeffcott Street  West Melbourne",
       "Is there a bus near '236 Bourke Street  Melbourne'",
       "Find bus stops around 'Royal Botanic Gardens'",
       "Any buses close to '139 Gatehouse Street  Parkville'",
       "Where’s the nearest bus stop to '61 Cook Street  Port Melbourne'",
       "Can I catch a bus near '45 Gracie Street  North Melbourne'",
       "What buses leave from near '252 Swanston Street  Melbourne'",
       "Bus stops close to '85 Rathdowne Street?'",
       "Is there a bus around '111 Spencer Street?'",
    ],
    "nlu_fallback": [
       "I'm feeling hungry",
       "The weather is good today",
       "Suggest me a song",
       "tell me a public transport joke"
    ]
}

def build_prompt(user_message, training_data, num_examples_per_intent=1):
    """
    Construct a prompt that includes a high-level instruction together with a few examples
    for each intent. This way, the examples are maintained outside the code logic.
    """
    base_instruction = (
        "You are a chatbot classifier for a transit assistant. "
        "Map the user's message to one of the following intent labels:\n"
        + ", ".join(training_data.keys()) + ".\n"
        "If the message does not clearly correspond to any, reply with 'nlu_fallback'.\n\n"
    )

    examples = ""
    for intent, examples_list in training_data.items():
        # Select a few representative examples (you can adjust how many)
        selected = random.sample(examples_list, min(num_examples_per_intent, len(examples_list)))
        for ex in selected:
            examples += f"User: \"{ex}\"\nIntent: {intent}\n\n"

    prompt = base_instruction + "Examples:\n" + examples
    prompt += f"Now classify the following message:\nUser: \"{user_message}\"\nIntent:"
    return prompt

@DefaultV1Recipe.register("llama_intent_classifier", is_trainable=False)
class LlamaIntentClassifier(GraphComponent):
    @classmethod
    def required_components(cls):
        return []

    @classmethod
    def create(cls, config: dict, model: object, context: ExecutionContext) -> "LlamaIntentClassifier":
        llama_client = model if isinstance(model, LlamaClient) else LlamaClient(model_path=config.get("model_path"))
        return cls(llama_client, config)

    def __init__(self, model: LlamaClient, config: dict) -> None:
        self.model = model
        self.component_config = config

    def parse(self, text: str, time: float = None) -> dict:
        # Build prompt as before
        prompt = build_prompt(text, TRAINING_DATA, num_examples_per_intent=1)
        print("[DEBUG] Prompt to LLM:\n", prompt, flush=True)

        # Optionally pass a generation parameter like max_tokens if your client supports it:
        # raw_response = self.model.generate(prompt, max_tokens=5)
        raw_response = self.model.generate(prompt)
        print("[DEBUG] Raw LLM output:", raw_response, flush=True)

        # Extract only the first non-empty line from the response.
        first_line = next((line.strip() for line in raw_response.splitlines() if line.strip()), "")
        print("[DEBUG] First line extracted:", repr(first_line), flush=True)

        cleaned_output = first_line.lower().strip(string.punctuation).strip()
        print("[DEBUG] Cleaned LLM output:", repr(cleaned_output), flush=True)

        valid_intents = ["request_train_map", "check_disruptions_bus", "check_disruptions_tram", "find_nearest_tram_stop", "find_nearest_bus_stop","find_nearest_station"]

        # If the cleaned output is a valid intent, use it; otherwise fallback.
        predicted_intent = cleaned_output if cleaned_output in valid_intents else "nlu_fallback"
        print("[DEBUG] Predicted Intent from LLM output:", predicted_intent, flush=True)

        # Fuzzy matching backup: if predicted_intent is nlu_fallback
        if predicted_intent == "nlu_fallback":
            best_intent = None
            best_score = 0
            for intent in valid_intents:
                for ex in TRAINING_DATA[intent]:
                    score = fuzz.token_sort_ratio(text, ex)
                    if score > best_score:
                        best_score = score
                        best_intent = intent
            print(f"[DEBUG] Fuzzy matching best score: {best_score} and best intent: {best_intent}", flush=True)
            if best_score >= 60:  # adjust threshold as needed
                predicted_intent = best_intent

        print("[DEBUG] Final Predicted Intent:", predicted_intent, flush=True)
        confidence = 0.95 if predicted_intent != "nlu_fallback" else 0.6

        return {
            "text": text,
            "intent": {"name": predicted_intent, "confidence": confidence},
            "entities": [],
            "intent_ranking": [{"name": predicted_intent, "confidence": confidence}],
            "response_selector": {}
        }

    def persist(self, file_name: str, model_dir: str) -> dict:
        return {}
