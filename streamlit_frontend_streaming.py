import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage

# Thread ID links this session to the correct checkpointer memory
CONFIG = {"configurable": {"thread_id": "thread-1"}}

# Initialize message history once; persists across Streamlit reruns
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# Replay all past messages to rebuild UI on every rerun
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

# Returns None until the user submits; triggers a rerun on submit
user_input = st.chat_input("Type here")

if user_input:
    # Persist user message before rendering to keep history in sync
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    with st.chat_message("assistant"):
        # Stream tokens as they arrive; write_stream returns the full joined string
        ai_message = st.write_stream(
            message_chunk.content                                    # extract text token
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",                              # yields (chunk, metadata) pairs
            )
        )

    # Persist the complete AI reply only after streaming finishes
    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})