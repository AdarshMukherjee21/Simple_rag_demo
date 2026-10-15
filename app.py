import os
import streamlit as st
from ingest import process_document
from agent import app as agent_app

st.set_page_config(layout="wide", page_title="Personal RAG Assistant")

# Initialize session state for tracking processed files and chat messages
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Define the 3-column layout
col_left, col_center, col_right = st.columns([1, 2, 1])

# ==========================================
# LEFT PANE: Document Upload & Ingestion
# ==========================================
with col_left:
    st.header(" Knowledge Base")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        for file in uploaded_files:
            file_name = os.path.splitext(file.name)[0]
            doc_dir = os.path.join("data", file_name)
            faiss_path = os.path.join(doc_dir, "faiss_folder")
            
            # Check both session_state and local disk to avoid redundant re-ingestion
            if file.name in st.session_state.processed_files or os.path.exists(faiss_path):
                st.session_state.processed_files.add(file.name)
                continue
                
            os.makedirs(doc_dir, exist_ok=True)
            save_path = os.path.join(doc_dir, file.name)
            
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())
                
            with st.spinner(f"Ingesting {file.name}..."):
                process_document(save_path)
                st.session_state.processed_files.add(file.name)
                
        st.success("Documents Ready!")

# ==========================================
# CENTER PANE: Chat Interface
# ==========================================
with col_center:
    st.header("Assistant")

    # Render full chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Process user query
    if prompt := st.chat_input("Ask about your documents (e.g., 'What is my CGPA?'):"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Invoke LangGraph Agent Loop
        with st.spinner("Processing query..."):
            result = agent_app.invoke({"messages": [("user", prompt)]})
            final_response = result["messages"][-1].content

        st.session_state.messages.append({"role": "assistant", "content": final_response})
        st.chat_message("assistant").write(final_response)

# ==========================================
# RIGHT PANE: Agent Traces & Metadata
# ==========================================
with col_right:
    st.header("Agent Traces")
    if st.session_state.messages:
        latest_user_msg = [m for m in st.session_state.messages if m["role"] == "user"]
        if latest_user_msg:
            st.write("**Latest Query:**", latest_user_msg[-1]["content"])
            st.write("**LLM Orchestration:** LangGraph")
            st.write("**LLM Endpoint:** `nvidia/nemotron-3-super-120b-a12b`")
            st.write("**Embedding Model:** `qwen3-embedding:4b`")
            st.write("**Vector Store:** FAISS")