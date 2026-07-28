# Local Multi-RAG System

A local, privacy-focused Retrieval-Augmented Generation (RAG) system built with Python, LangChain, ChromaDB, and Ollama (Llama 3.1). This project allows users to upload documents, maintain chat history with query rewriting, and query data locally without relying on external paid APIs.

## Features
- **Local LLM Execution**: Powered completely offline by Ollama (Llama 3.1).
- **Vector Database**: Utilizes ChromaDB for high-performance document embedding storage.
- **Conversational Memory & Query Rewriting**: Handles follow-up questions contextually.
- **Hybrid Search Capability**: Falls back gracefully to general knowledge if the documents don't contain the answer.

## Tech Stack
- **Python**
- **LangChain & LangChain-Core**
- **ChromaDB**
- **Ollama**

## Project Structure
```text
MULTI_RAG/
│
├── backend/
│   ├── routes/             # API Endpoints
│   ├── services/           # Business logic (RAG service, document loaders)
│   ├── vectorstore/        # ChromaDB setup and repository
│   └── utils/              # Logger and helper functions
│
├── frontend/               # User interface (Streamlit / UI)
├── main.py                 # Core application entry point
├── run.py                  # Script to run the app
└── README.md
