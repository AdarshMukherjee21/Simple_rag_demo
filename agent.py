import os
import json
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from openai import OpenAI

from langchain_core.tools import tool
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()
api_key = os.environ.get("NVIDIA_API_KEY", "")

# Initialize Nemotron Client via OpenAI-compatible API
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

class State(TypedDict):
    messages: Annotated[list, add_messages]

# 1. Define the Vector Search Tool
@tool
def search_knowledge_base(query: str) -> str:
    """Searches the personal knowledge base for relevant document excerpts regarding the student's resume, skills, CGPA, projects, etc."""
    print(f"\n[Tool Execution] Searching vector DB for query: '{query}'")
    embeddings = OllamaEmbeddings(model="qwen3-embedding:4b")
    all_docs = []
    
    if os.path.exists("data"):
        for root, dirs, files in os.walk("data"):
            if "faiss_folder" in root:
                try:
                    db = FAISS.load_local(root, embeddings, allow_dangerous_deserialization=True)
                    docs = db.similarity_search(query, k=4)
                    all_docs.extend(docs)
                except Exception as e:
                    print(f"Error loading vector index at {root}: {e}")

    if not all_docs:
        return "No relevant information found in the knowledge base."
    print(all_docs)
    return "\n\n---\n\n".join([d.page_content for d in all_docs[:5]])


# Tool metadata for OpenAI API schema
openai_tools = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Searches the personal knowledge base for relevant document excerpts regarding the student's background, resume, skills, CGPA, projects, experience, or certifications.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to retrieve information from the vector database."
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# 2. Agent Decision Node
def call_model(state: State):
    messages = state["messages"]
    
    system_prompt = """You are an intelligent assistant for a student's personal knowledge base.
- If the user asks a question about the student (resume, background, skills, CGPA, projects, experience), you MUST use the `search_knowledge_base` tool to query the knowledge base first.
- If the query is conversational (e.g. "Hello", "How are you?"), respond directly without calling any tools.
- When answering from tool output, rely ONLY on the provided context. If information is missing, state that it is unavailable."""

    # Convert LangChain roles to OpenAI roles ("human" -> "user")
    formatted_msgs = []
    for m in messages:
        role = m.type
        if role == "human":
            role = "user"
        elif role not in ["system", "assistant", "user", "tool"]:
            role = "user"
        formatted_msgs.append({"role": role, "content": str(m.content)})

    formatted_msgs.insert(0, {"role": "system", "content": system_prompt})

    # First LLM Call: Let Nemotron decide whether to call a tool or reply directly
    response = client.chat.completions.create(
        model="nvidia/nemotron-3-super-120b-a12b",
        messages=formatted_msgs,
        tools=openai_tools,
        tool_choice="auto",
        temperature=0.1,
        max_tokens=1024
    )

    response_message = response.choices[0].message

    # Check if the model decided to call the search tool
    if response_message.tool_calls:
        tool_call = response_message.tool_calls[0]
        tool_args = json.loads(tool_call.function.arguments)
        query = tool_args.get("query", "")
        
        # Execute tool conditionally
        tool_result = search_knowledge_base.invoke(query)

        # Append tool call and result to message context for final completion
        formatted_msgs.append(response_message)
        formatted_msgs.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_result
        })

        # Second LLM Call: Generate final answer based on retrieved context
        final_response = client.chat.completions.create(
            model="nvidia/nemotron-3-super-120b-a12b",
            messages=formatted_msgs,
            temperature=0.1,
            max_tokens=1024
        )
        return {"messages": [("ai", final_response.choices[0].message.content)]}

    # Model chose NOT to use a tool (direct answer)
    return {"messages": [("ai", response_message.content)]}


# Build LangGraph State Machine
workflow = StateGraph(State)
workflow.add_node("agent", call_model)
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

app = workflow.compile()