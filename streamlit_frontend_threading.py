import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# ── Utility Functions ──────────────────────────────────────────────────────────────

# Generates a unique ID for each new conversation thread
def generate_thread_id():
    return uuid.uuid4()

# Resets the chat by creating a fresh thread and clearing message history
def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []

# Registers a thread ID only if it isn't already tracked
def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

# Fetches persisted LangGraph state for a given thread; returns [] if no messages yet
def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


# ── Session State Bootstrap ────────────────────────────────────────────────────────

# All three keys must exist before any UI code runs
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []

# Ensure the active thread is always listed in the sidebar
add_thread(st.session_state["thread_id"])


# ── Sidebar UI ─────────────────────────────────────────────────────────────────────

st.sidebar.title("LangGraph Chatbot")

# Clicking this wipes the current session and starts a brand-new thread
if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")

# Reverse order so the most recent thread appears at the top
for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(str(thread_id)):
        # Switch active thread and reload its messages from the checkpointer
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)

        # Convert LangChain message objects to the flat dict format used by the UI
        temp_messages = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": msg.content})

        st.session_state["message_history"] = temp_messages


# ── Main Chat UI ───────────────────────────────────────────────────────────────────

# Rebuild the full chat UI from history on every Streamlit rerun
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    # Persist user message immediately so it survives reruns
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    # Config is built per-turn to always use the currently active thread
    CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}

    with st.chat_message("assistant"):
        # Inner generator filters out non-AI chunks (e.g. tool call metadata)
        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content  # yield only assistant text tokens

        # Streams tokens live and returns the full reply string when done
        ai_message = st.write_stream(ai_only_stream())

    # Append only after streaming completes so the full reply is stored
    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})