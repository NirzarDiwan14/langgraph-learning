import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage
import uuid
from uuid import UUID

#utility functions
def generate_thread_id()-> UUID:
    thread_id = uuid.uuid4()
    return thread_id

#Session Setup
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

#Sidebar UI
st.sidebar.title("LangGraph Chatbot")
st.sidebar.button("New Chat")
st.sidebar.header("My Conversations")
st.sidebar.text(st.session_state["thread_idf"])
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type here:")
CONFIG = {"configurable": {"thread_id": st.session_state['thread_id']}}

if user_input:
    # first add the user message to history
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)
    # first add the assitant message to history
    initial_state = {"messages": [HumanMessage(content=user_input)]}
    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            message_chunk.content
            for message_chunk, metadata in chatbot.stream(
                initial_state, config=CONFIG, stream_mode="messages"
            )
        )
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
