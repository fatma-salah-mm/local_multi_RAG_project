import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.vectorstore.chroma_store import VectorStoreRepository
from backend.utils.logger import app_logger

def ingest_local_files():
    """
    Loads all files from the data/ directory, chunks them, 
    and indexes them into the Chroma vector store.
    """
    data_dir = "data"
    if not os.path.exists(data_dir):
        return
        
    try:
        vector_repo = VectorStoreRepository()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)
            
            if os.path.isfile(file_path):
                docs = []
                try:
                    if filename.endswith(".txt"):
                        loader = TextLoader(file_path, encoding="utf-8")
                        docs = loader.load()
                    elif filename.endswith(".pdf"):
                        loader = PyPDFLoader(file_path)
                        docs = loader.load()
                    elif filename.endswith(".md"):
                        loader = UnstructuredMarkdownLoader(file_path)
                        docs = loader.load()
                        
                    if docs:
                        for doc in docs:
                            doc.metadata["filename"] = filename
                            
                        chunks = text_splitter.split_documents(docs)
                        vector_repo.vector_store.add_documents(chunks)
                        app_logger.info(f"Successfully indexed multi-file: {filename}")
                except Exception as file_err:
                    app_logger.error(f"Could not parse file {filename}: {file_err}")
                    
    except Exception as e:
        app_logger.error(f"Error during bulk file ingestion: {e}")