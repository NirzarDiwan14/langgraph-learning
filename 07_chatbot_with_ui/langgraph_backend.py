from langgraph.graph import StateGraph, START, END
from langchain_mistralai import ChatMistralAI
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import os 


import sqlite3

load_dotenv()
os.environ['LANGCHAIN_PROJECT'] = "LangGraph-Chatbot"
llm = ChatMistralAI(model="mistral-small-latest")


# State of workflow
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    title: str


# node functions
def chat_node(state: ChatState) -> ChatState:
    # take user query from the state
    messages = state["messages"]
    title = state.get("title")
    if not title:
        first_user_msg = messages[0].content
        title_prompt = f"""
Generate a concise and descriptive title (3–5 words) that captures the main topic of this conversation.

Text:
{first_user_msg}

Rules:
- Use only plain text
- No quotes, punctuation, or extra symbols
- No explanations or additional output
- Return exactly one title

Title:
"""
        title_response = llm.invoke(title_prompt)
        title = title_response.content.strip()

    # Send it to LLM
    response = llm.invoke(messages)
    return {"messages": [response], "title": title}


# Defining Graph with State
graph = StateGraph(ChatState)

# Defining Nodes
graph.add_node("chat_node", chat_node)

# Defining Edges

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# Database
CONFIG = {"configurable": {"thread_id": "11"}}

conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# # Defining Checkpointer Memory
# checkpointer = MemorySaver()

# Graph Compilation
chatbot = graph.compile(checkpointer=checkpointer)


# Utility functions
def retrieve_all_threads():

    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)
