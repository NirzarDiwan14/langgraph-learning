import streamlit as st
from langgraph_backend import chatbot, retrieve_all_threads
from langchain_core.messages import AIMessage, HumanMessage,ToolMessage
import uuid
from uuid import UUID


# utility functions
def generate_thread_id() -> UUID:
    thread_id = uuid.uuid4()
    return thread_id


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    return chatbot.get_state(config={"configurable": {"thread_id": thread_id}}).values[
        "messages"
    ]


def get_thread_title(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("title", "New Chat")


# Session Setup
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

add_thread(st.session_state["thread_id"])

# Sidebar UI
st.sidebar.title("LangGraph Chatbot")
if st.sidebar.button("New Chat"):
    reset_chat()
    add_thread(st.session_state["thread_id"])
st.sidebar.header("My Conversations")

# st.sidebar.text(st.session_state["thread_id"])
for thread_id in reversed(st.session_state["chat_threads"]):
    title = get_thread_title(thread_id)
    if st.sidebar.button(title, key=str(thread_id)):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            else:
                role = "assistant"
            temp_messages.append({"role": role, "content": msg.content})
        st.session_state["message_history"] = temp_messages


for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type here:")
CONFIG = {
    "configurable": {"thread_id": st.session_state["thread_id"]},
    "metadata": {"thread_id": st.session_state["thread_id"]},
    "run_name": "chat_turn",
}

if user_input:
    # first add the user message to history
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)
    # first add the assitant message to history
    initial_state = {"messages": [HumanMessage(content=user_input)]}
    # first add the message to message_history
    with st.chat_message("assistant"):
       status_holder = {"box": None}

    def stream_wrapper():
        for message_chunk, metadata in chatbot.stream(
            initial_state,
            config=CONFIG,
            stream_mode="messages"
        ):
            if isinstance(message_chunk, ToolMessage):
                tool_name = getattr(message_chunk, "name", "tool")

                if status_holder["box"] is None:
                    status_holder["box"] = st.status(
                        f"🔧 Using `{tool_name}` ...", expanded=True
                    )
                else:
                    status_holder["box"].update(
                        label=f"🔧 Using `{tool_name}` ...",
                        state="running",
                        expanded=True,
                    )
                status_holder["box"].write(message_chunk.content)
            if isinstance(message_chunk, AIMessage):
                yield message_chunk.content


    ai_message = st.write_stream(stream_wrapper())

    # ✅ finalize status
    if status_holder["box"] is not None:
        status_holder["box"].update(
            label="✅ Tool finished",
            state="complete",
            expanded=False
        )
    #Save assistant state
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
