from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import Command, interrupt

from configs.prompts import PROMPT
from configs.json_schema import PropertyDetails
from utils.utils import HousingData

import pandas as pd
import json

from dotenv import load_dotenv
load_dotenv()


NUM_MAX_LOOP = 5
HOUSING_DATA_PTH = "data/housing_data_preprocessed.csv"
DATA_PTH = "data/temp_data.csv"

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    parsed: str
    loop_counter: int
    data_pth: str
    
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
)
structured_llm = llm.with_structured_output(PropertyDetails)

def parse_query(state: AgentState) -> AgentState:
    """Use LLM to parse user query into structured JSON."""
    
    if not state["messages"]:
        state["messages"] = ""

    system_prompt = PROMPT

    all_messages = [system_prompt] + list(state["messages"])
    response = structured_llm.invoke(all_messages)

    state["parsed"] = response.model_dump_json()
    return state

def check_completion(state: AgentState) -> Command:
    """Decide whether we have enough info."""
    
    try:
        loop_counter = state["loop_counter"]
    except:
        loop_counter = 0
    
    if loop_counter < NUM_MAX_LOOP:
        try:
            data = json.loads(state["parsed"])
            if data.get("area") and data.get("max_rental_fee_per_week"):
                return Command(goto="filter_data")
            else:
                return Command(goto="ask_more")
        except Exception:
            return Command(goto="ask_more")
    else:
        return Command(goto="end_query")
    
def ask_more(state: AgentState) -> AgentState:
    """Ask for more information."""
    
    data = json.loads(state["parsed"])
    
    missing = []
    if not data.get("area"):
        missing.append("area")
    if not data.get("max_rental_fee_per_week"):
        missing.append("budget")

    message = f"Could you please provide your {' and '.join(missing)}?"
    state["messages"].append(AIMessage(content=message))
    
    try:
        loop_counter = state["loop_counter"]
    except:
        loop_counter = 0
    state["loop_counter"] = loop_counter + 1
    
    return state

def human_input(state: AgentState):
    """Ask for human input"""
    
    additional_info = interrupt(state["messages"][-1].content)
 
    return {"messages": HumanMessage(content=additional_info)}

def end_query(state: AgentState) -> AgentState:
    """Show ending message from AI when it can't perform the action."""
    
    message = "Sorry, I can't perform the task right now. Please try again later."
    state["messages"].append(AIMessage(content=message))
    
    return state

def filter_data(state: AgentState) -> AgentState:
    """Filter the listing based on the extracted entites."""
    
    # Load the housing listing
    data = pd.read_csv(HOUSING_DATA_PTH)
    json_data = json.loads(state["parsed"])
    
    data = HousingData.preprocess_address(data)
    data = HousingData.get_coordination(data)
    data = HousingData.filter_basic_data(data, json_data)
    data = HousingData.get_distance(data, json_data)
    data = HousingData.filter_distance(data, json_data)
    data.to_csv(DATA_PTH, index=False)
    state["data_pth"] = DATA_PTH
    
    return state

def rank_properties(state: AgentState) -> AgentState:
    """Calculate the score and rank the properties."""
    
    data = pd.read_csv(state["data_pth"])
    data = HousingData.rank_properties(data)
    
    data = data.sort_values(by="score", ascending=False)
    data.to_csv(DATA_PTH, index=False)
    state["data_pth"] = DATA_PTH
    
    return state

def show_json_output(state: AgentState) -> AgentState:
    """Show the parsed JSON."""
    
    message = f"The JSON output is: {state['parsed']}"
    state["messages"].append(AIMessage(content=message))
    
    return state

def show_data_output(state: AgentState) -> AgentState:
    """Show the filtered properties."""
    
    data = pd.read_csv(state["data_pth"])
    num_properties = len(data)
    
    if num_properties == 0:
        message = f"Unfortunately, there is no available property that matches your search. Please try again with another query."
    else:
        message = f"We have found {num_properties} properties that matches your query.\n"
        for i in range(num_properties):
            message += f"{i+1}. Address: {data.iloc[i, :]['Address']}. Type: {data.iloc[i, :]['Type of property']} with {data.iloc[i, :]['Number of bedrooms']} bedrooms and {data.iloc[i, :]['Number of bathrooms']} bathrooms. Price per week: ${data.iloc[i, :]['Price pw']}.\n"
    
    state["messages"].append(AIMessage(content=message))
    
    return state

def create_graph() -> CompiledStateGraph:
    """Create the chatbot graph."""
    
    graph = StateGraph(AgentState)

    graph.add_node("parse_query", parse_query)
    graph.add_node("check_completion", check_completion)
    graph.add_node("ask_more", ask_more)
    graph.add_node("human_input", human_input)
    graph.add_node("end_query", end_query)
    graph.add_node("filter_data", filter_data)
    graph.add_node("rank_properties", rank_properties)
    graph.add_node("show_data_output", show_data_output)

    graph.add_edge(START, "parse_query")
    graph.add_edge("parse_query", "check_completion")
    graph.add_edge("ask_more", "human_input")
    graph.add_edge("human_input", "parse_query")
    graph.add_edge("filter_data", "rank_properties")
    graph.add_edge("rank_properties", "show_data_output")
    graph.add_edge("end_query", END)
    graph.add_edge("show_data_output", END)

    memory = MemorySaver()
    chatbot_graph = graph.compile(checkpointer=memory)
    return chatbot_graph