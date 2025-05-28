# main.py

from llama_client import LlamaClient
from custom_components.llm_interpreter import LlamaIntentClassifier
from rasa.core.agent import Agent
import asyncio


def main():
    # Instantiate the LLaMA client with the proper model path.
    print("start main")
    llama_client = LlamaClient(model_path="C:/Users/jubal/Documents/SIT374/mistral-7b-instruct-v0.1.Q4_K_M.gguf")

    # Create the custom interpreter with the LLaMA client.
    interpreter = LlamaIntentClassifier(llama_client, config={})

    # Load your Rasa agent using this custom interpreter.
    agent = Agent.load("models/current")

    agent.interpreter = interpreter

    user_message = "When is the next train from Ringwood to Parliament station?"

    # Use asyncio.run to call the agent's asynchronous handle_text() method.
    responses = asyncio.run(agent.handle_text(user_message))

    # Print out each response.
    for response in responses:
        if "text" in response:
            print(response["text"])

if __name__ == "__main__":
    main()
