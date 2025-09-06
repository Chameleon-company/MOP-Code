from langchain_core.runnables.config import RunnableConfig
import json

import uuid
from graph import chatbot_graph as app

from dotenv import load_dotenv
load_dotenv()

def stream_graph_updates(user_input: str, config: RunnableConfig):
    events = app.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values"
    )
    
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()

config_id = uuid.uuid4()
config = {"configurable": {"thread_id": config_id}}

current_message_id = ""
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        try:
            last_message_id = app.get_state(config=config)[0]["messages"][-1].id
        except:
            last_message_id = "_"
            
        # Check if there is a new message:
        if current_message_id != last_message_id:
            stream_graph_updates(user_input, config=config)
            current_message_id = last_message_id
        
    except Exception as e:
        print(f"Error: {e}")
        break

app_state = app.get_state(config=config)
parsed_json = app_state[0]["parsed"]
data = json.loads(parsed_json)
print("Parsed JSON:")
print(data)