from typing import cast
import chainlit as cl

from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from chatbot.graph import create_graph


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("messages", [])
    agent = await get_chat_agent()
    cl.user_session.set("agent", agent)

@cl.on_message
async def on_message(message: cl.Message):
    agent = cast(CompiledStateGraph, cl.user_session.get("agent"))
    config = RunnableConfig(
        configurable={"thread_id": cl.context.session.id}, 
        callbacks=[cl.LangchainCallbackHandler()],
    )
    
    messages = cl.user_session.get("messages")
    messages.append(HumanMessage(content=message.content))
    cl.user_session.set("messages", messages)

    interrupt = None
    response = cl.Message(content="")

    stream = agent.astream(
        {"messages": messages},
        config=config,
        stream_mode=['messages', 'updates'],
    )
    
    while stream:
        async for stream_mode, pack in stream:
            if stream_mode == 'messages':
                msg, metadata = pack
                if (
                    msg.content
                    and not isinstance(msg, HumanMessage)
                    and metadata["langgraph_node"] in ["show_data_output", "end_query"]
                ):
                    await response.stream_token(msg.content)
                stream = None

            else:
                if '__interrupt__' in pack:
                    interrupt = pack['__interrupt__'][0]
                    res = await cl.AskUserMessage(content=str(interrupt.value)).send()
                    
                    cmd = Command(resume=res["output"])

                    stream = agent.astream(
                        cmd,
                        config=config,
                        stream_mode=['messages', 'updates'],
                    )
                else:
                    stream = None

    messages.append(AIMessage(content=response.content))
    cl.user_session.set("messages", messages)

    await response.send()

async def get_chat_agent() -> CompiledStateGraph:
    graph = create_graph()
    return graph