#-----------------------------------------------------
# By Jubal
#-----------------------------------------------------

import json
from llama_cpp import Llama

class LlamaClient:
    def __init__(self, model_path: str):
        """
        Initializes the LLaMA client.

        Parameters:
        - model_path (str): The file path to your LLaMA model.
        """
        # Create a Llama instance from the llama_cpp library.
        self.model = Llama(model_path=model_path, temperature=0.3)

    def generate(self, prompt: str, max_tokens: int = 50, temperature: float = 0.7) -> dict:
        """
        Calls the LLaMA model to generate a text completion for the given prompt.

        Parameters:
        - prompt (str): The input prompt for the LLM.
        - max_tokens (int): Maximum tokens to generate.
        - temperature (float): Sampling temperature for generation.

        Returns:
        - dict: The parsed JSON response from the LLaMA model.
        """
        try:
            # Generate the completion using the model.
            output = self.model(prompt)

            print("[DEBUG] Raw output from Llama model:", output)

            # Try to parse the output as JSON if it's a string
            if isinstance(output, str):
                output = json.loads(output)  # Parse the string to a dictionary

            # Extract text from dictionary output
            text = output.get("choices", [{}])[0].get("text", "").strip()

            # Log the extracted text
            print("[DEBUG] Extracted text:", text)

            return text

        except Exception as e:
            # Handle unexpected errors gracefully
            print(f"[ERROR] Error in LlamaClient generate method: {e}")
            return ""  # Return empty string on error or handle as needed
