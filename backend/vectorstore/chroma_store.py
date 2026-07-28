from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import settings
from backend.utils.logger import app_logger

class VectorStoreRepository:
    """
    Manages local vector storage and similarity search using ChromaDB 
    and HuggingFace local embeddings.
    """

    def __init__(self):
        try:
            # Initialize local embedding model
            self.embeddings = HuggingFaceEmbeddings(
                model_name=getattr(settings, "EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            )
            
            # Initialize persistent Chroma client/store
            self.vector_store = Chroma(
                collection_name=getattr(settings, "COLLECTION_NAME", "enterprise_rag"),
                embedding_function=self.embeddings,
                persist_directory=getattr(settings, "PERSIST_DIRECTORY", "./db")
            )
            app_logger.info("Successfully initialized VectorStoreRepository with ChromaDB.")
            
        except Exception as e:
            app_logger.error(f"Failed to initialize VectorStoreRepository: {str(e)}")
            raise e

    def search(self, query: str, k: int = 4, score_threshold: float = 0.0):
        """
        Performs a similarity search on the vector store.
        Returns a list of tuples: (Document, similarity_score)
        """
        try:
            # Using similarity search with relevance scores if supported
            results = self.vector_store.similarity_search_with_score(query, k=k)
            app_logger.info(f"Retrieved {len(results)} chunks for query: '{query}'")
            return results
        except Exception as e:
            app_logger.error(f"Error executing vector search: {str(e)}")
            return []