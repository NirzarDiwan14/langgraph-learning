from langgraph.graph import StateGraph, START, END
from langchain_mistralai import ChatMistralAI
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import os
import requests


import sqlite3

load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "LangGraph-Chatbot"
llm = ChatMistralAI(model="mistral-small-latest")

# Tools
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()


tools = [search_tool, get_stock_price, calculator]
llm_with_tools = llm.bind_tools(tools)


# State of workflow
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    title: str


# node functions
def chat_node(state: ChatState) -> ChatState:
    """LLM node that may answer or request a tool call."""
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
    response = llm_with_tools.invoke(messages)
    return {"messages": [response], "title": title}


# Defining Graph with State
graph = StateGraph(ChatState)
tool_node = ToolNode(tools)

# Defining Nodes
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

# Defining Edges

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

# Database
# CONFIG = {"configurable": {"thread_id": "11"}}

conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
# # Defining Checkpointer Memory
checkpointer = SqliteSaver(conn=conn)


# Graph Compilation
chatbot = graph.compile(checkpointer=checkpointer)


# Utility functions
def retrieve_all_threads():

    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)
