from langchain_huggingface import HuggingFaceEmbeddings
from config import settings
from backend.utils.logger import app_logger

class EmbeddingManager:
    """
    Manages the local embedding model initialization using HuggingFace.
    """
    def __init__(self):
        try:
            model_name = getattr(settings, "EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
            app_logger.info(f"Initialized EmbeddingManager with model: {model_name}")
        except Exception as e:
            app_logger.error(f"Failed to initialize EmbeddingManager: {str(e)}")
            raise e

    def get_embedding_function(self):
        """Returns the embeddings instance for vector storage."""
        return self.embeddings