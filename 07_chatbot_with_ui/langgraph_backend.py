from langgraph.graph import StateGraph, START, END
from langchain_mistralai import ChatMistralAI
from typing import TypedDict, Literal, Annotated
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatMistralAI(model="mistral-small-latest")


# State of workflow
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# node functions
def chat_node(state: ChatState) -> ChatState:
    # take user query from the state
    messages = state["messages"]

    # Send it to LLM
    response = llm.invoke(messages)
    return {"messages": [response]}


# Defining Graph with State
graph = StateGraph(ChatState)

# Defining Nodes
graph.add_node("chat_node", chat_node)

# Defining Edges

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# Defining Checkpointer Memory
checkpointer = MemorySaver()

# Graph Compilation
chatbot = graph.compile(checkpointer=checkpointer)
