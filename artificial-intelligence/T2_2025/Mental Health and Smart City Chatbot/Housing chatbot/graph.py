from typing import Dict, Any, Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage

from langchain_core.tools import tool
from langchain_core.runnables.config import RunnableConfig

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import Command, interrupt

from prompts import PROMPT
from json_schema import CloseEntity, PropertyDetails

import uuid
from dotenv import load_dotenv
load_dotenv()


NUM_MAX_LOOP = 5

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    parsed: str
    loop_counter: int
    
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
    # print(f"Parsed JSON: {state['parsed']}")
    return state

def check_completion(state: AgentState) -> str:
    """Decide whether we have enough info."""
    import json
    try:
        loop_counter = state["loop_counter"]
    except:
        loop_counter = 0
    
    if loop_counter < NUM_MAX_LOOP:
        try:
            data = json.loads(state["parsed"])
            if data.get("area") and data.get("max_rental_fee_per_week"):
                return "enough_info"
            else:
                return "ask_more"
        except Exception:
            return "ask_more"
    else:
        return "end_query"
    
def ask_more(state: AgentState) -> AgentState:
    """Ask for more information."""
    
    import json
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

def human_input(state: AgentState) -> Command:
    additional_info = interrupt(state["messages"][-1])
    return Command(
        update={"messages": state["messages"] + [HumanMessage(content=additional_info)]}, 
        goto="parse_query"
    )

def end_query(state: AgentState) -> AgentState:
    """Show ending message from AI."""
    
    message = "Sorry, I can't perform the task right now. Please try again later."
    state["messages"].append(AIMessage(content=message))
    
    return state


graph = StateGraph(AgentState)

graph.add_node("parse_query", parse_query)
graph.add_node("ask_more", ask_more)
graph.add_node("human_input", human_input)
graph.add_node("end_query", end_query)

graph.add_edge(START, "parse_query")
graph.add_conditional_edges(
    "parse_query",
    check_completion,
    {
        "enough_info": END,
        "ask_more": "ask_more",
        "end_query": "end_query"
    }
)
graph.add_edge("ask_more", "human_input")
graph.add_edge("end_query", END)

memory = MemorySaver()
chatbot_graph = graph.compile(checkpointer=memory)

