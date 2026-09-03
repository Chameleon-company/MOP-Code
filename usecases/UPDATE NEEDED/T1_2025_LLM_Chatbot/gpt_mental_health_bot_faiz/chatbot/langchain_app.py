import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# Load API key from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise Exception(" OpenAI API key not found. Add it to your .env file!")

# Setting up the OpenAI GPT-3.5 chatbot
chat = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.3,
    top_p=1.0
)

# System prompt
system_prompt = """
You are a compassionate and helpful mental health assistant.
You provide emotional support and guidance for users experiencing anxiety and depression.
Be empathetic, calm, and non-judgmental in your responses.
"""

# Welcome message
print("=" * 50)
print("|{:^48}|".format("Mental Health Chatbot"))
print("=" * 50)
print("Type 'exit' or 'quit' to end the conversation.\n")

# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit","bye"]:
        print("\n" + "-" * 50)
        print("👋 Take care. You’re not alone.")
        print("-" * 50 + "\n")
        break

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]

    try:
        response = chat.invoke(messages)

        print("\n" + "-" * 50)
        print("Bot:")
        print(f"{response.content}")
        print("-" * 50 + "\n")
    except Exception as e:
        print(f" Error: {e}")
