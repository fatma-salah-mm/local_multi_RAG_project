import streamlit as st
import requests
import os

API_URL = "http://localhost:8000/api/v1"

st.set_page_config(page_title="Enterprise RAG Chat", page_icon="🤖", layout="centered")

st.title("🤖 Enterprise RAG Chatbot")
st.markdown("Upload multiple documents and chat with your knowledge base seamlessly.")

# --- INITIALIZE CHAT HISTORY ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR: System Status & Multi-File Upload ---
with st.sidebar:
    st.header("System Status")
    try:
        response = requests.get(f"{API_URL}/status")
        if response.status_code == 200:
            st.success("🟢 Backend Connected")
        else:
            st.error("🔴 Backend Error")
    except requests.exceptions.ConnectionError:
        st.error("🔴 Backend Disconnected")

    st.divider()
    
    st.header("📂 Document Ingestion")
    # Enabled multi-file selection here
    uploaded_files = st.file_uploader(
        "Upload documents (.txt, .pdf, .md)", 
        type=["txt", "pdf", "md"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        
        for uploaded_file in uploaded_files:
            file_path = os.path.join(data_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
        st.success(f"Successfully saved {len(uploaded_files)} file(s)!")
        
        # Automatically trigger ingestion for all files in data/
        with st.spinner("Indexing documents into ChromaDB..."):
            try:
                res = requests.post(f"{API_URL}/ingest")
                if res.status_code == 200:
                    st.success("All files indexed successfully!")
                else:
                    st.warning("Files saved, but indexing endpoint failed.")
            except Exception as e:
                st.error(f"Ingestion trigger failed: {e}")

# --- DISPLAY CHAT HISTORY ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- CHAT INPUT & BACKEND REQUEST ---
if prompt := st.chat_input("Ask a question about your documents..."):
    # Append user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                res = requests.post(
                    f"{API_URL}/query", 
                    json={"query": prompt}
                )
                
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("answer", "No answer returned.")
                    st.markdown(answer)
                    # Append assistant response to history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    error_msg = f"Backend error: {res.status_code}"
                    st.error(error_msg)
            except Exception as e:
                error_msg = f"Failed to connect to backend: {e}"
                st.error(error_msg)