import streamlit as st
from langgraph_database_backend import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage
import uuid

# ── Utility Functions ──────────────────────────────────────────────────────────────

# Each call produces a unique thread ID for a new conversation
def generate_thread_id():
    return uuid.uuid4()

# Wipes current chat and registers a fresh thread
def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []

# Guards against duplicate entries in the sidebar thread list
def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

# Pulls persisted LangGraph state from SQLite; returns [] if thread is empty
def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


# ── Session State Bootstrap ────────────────────────────────────────────────────────

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

# Hydrate thread list from SQLite on first load; persists across app restarts
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

# Register the active thread in case it was just created and not yet in the list
add_thread(st.session_state["thread_id"])


# ── Sidebar UI ─────────────────────────────────────────────────────────────────────

st.sidebar.title("LangGraph Chatbot")

# Starts a brand-new thread and clears the chat window
if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")

# Reverse so the most recently created thread appears at the top
for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(str(thread_id)):
        # Switch active thread and restore its messages from the checkpointer
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)

        # Convert LangChain message objects → flat dicts expected by the chat UI
        temp_messages = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": msg.content})

        st.session_state["message_history"] = temp_messages


# ── Main Chat UI ───────────────────────────────────────────────────────────────────

# Rebuild the full chat history on every Streamlit rerun
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    # Persist immediately so the message survives the next rerun
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    # run_name tags this turn in LangSmith traces; metadata mirrors thread_id for filtering
    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        # Streams tokens live; write_stream returns the full reply string when done
        ai_message = st.write_stream(
            message_chunk.content
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",  # yields (chunk, metadata) pairs
            )
        )

    # Stored only after streaming finishes so the full reply is captured
    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})