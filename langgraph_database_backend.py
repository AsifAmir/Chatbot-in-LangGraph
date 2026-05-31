from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3

# Load API keys and env vars before any client is initialized
load_dotenv()

# Initialize Gemini model with partial randomness in responses
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.7)

# Shared state schema; add_messages merges instead of replacing on each turn
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Node: passes full message history to LLM and returns its reply
def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

# check_same_thread=False required since Streamlit runs on multiple threads
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)

# SqliteSaver persists conversation state to disk across app restarts
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")  # entry point → chat node
graph.add_edge("chat_node", END)    # chat node → exit

# Compile with checkpointer so every thread's state is saved to SQLite
chatbot = graph.compile(checkpointer=checkpointer)

# Scans all saved checkpoints and returns unique thread IDs
def retrieve_all_threads():
    all_threads = set()                              # set deduplicates thread IDs
    for checkpoint in checkpointer.list(None):       # None = fetch across all threads
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)