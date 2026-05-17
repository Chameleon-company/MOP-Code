from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import HumanMessage
from langgraph.types import Command

import json
import uuid
from chatbot.graph import create_graph

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
app = create_graph()

while True:
    try:
        print("HELLO!")
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        for stream_mode, chunk in app.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config,
            stream_mode=["values", "updates"]
        ):  
            if "messages" in chunk:
                chunk["messages"][-1].pretty_print()
                
            if "__interrupt__" in chunk:
                human_input = input(chunk["__interrupt__"][0].value)
                for stream_mode, chunk in app.stream(
                    Command(resume=human_input),
                    config,
                    stream_mode=["values", "updates"]
                ):
                    if "messages" in chunk:
                        chunk["messages"][-1].pretty_print()
        
    except Exception as e:
        print(f"Error: {e}")
        break

app_state = app.get_state(config=config)
parsed_json = app_state[0]["parsed"]
data = json.loads(parsed_json)
print("Parsed JSON:")
print(data)