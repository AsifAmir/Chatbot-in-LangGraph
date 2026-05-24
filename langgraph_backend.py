# Load env vars from .env file
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

# Initialize Gemini model with partial randomness in responses
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.7)

# Define the graph state; messages auto-merge via add_messages reducer
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Node: sends current messages to LLM and returns its reply
def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

# In-memory checkpointer to persist conversation state across turns
checkpointer = InMemorySaver()

# Build the graph with ChatState as the shared state schema
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)   # Register the chat node
graph.add_edge(START, "chat_node")       # Entry point -> chat node
graph.add_edge("chat_node", END)         # Chat node -> exit

# Compile into a runnable chatbot with memory checkpointing enabled
chatbot = graph.compile(checkpointer=checkpointer)