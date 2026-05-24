import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage

# Thread ID ties all turns to the same conversation in the checkpointer
CONFIG = {"configurable": {"thread_id": "thread-1"}}

# Initialize message history in session state on first run
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# Re-render all previous messages on every Streamlit rerun
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

# Blocks until the user submits input; returns None if empty
user_input = st.chat_input("Type here")

if user_input:
    # Persist user message before invoking the model
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    # Send only the new message; checkpointer handles full history internally
    response = chatbot.invoke({"messages": [HumanMessage(content=user_input)]}, config=CONFIG)

    # Extract the last message, which is always the latest AI reply
    ai_message = response["messages"][-1].content

    # Persist AI reply and render it in the chat UI
    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})
    with st.chat_message("assistant"):
        st.text(ai_message)