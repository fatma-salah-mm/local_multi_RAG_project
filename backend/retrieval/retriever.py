from typing import List, Tuple
from langchain_core.documents import Document
from backend.vectorstore.chroma_store import VectorStoreRepository
from config import settings
from backend.utils.logger import app_logger

class ContextRetriever:
    """
    Service layer responsible for fetching relevant documents from the vector store
    and formatting them into a structured context window for the LLM.
    """

    def __init__(self, vector_store: VectorStoreRepository):
        """
        Injects the database repository instance.
        """
        self.vector_store = vector_store

    def get_context(self, query: str) -> str:
        """
        Executes the semantic search and formats the results.
        
        Args:
            query: The user's question.
            
        Returns:
            A formatted string containing the retrieved text and its metadata.
        """
        results = self.vector_store.search(
            query=query, 
            k=settings.TOP_K, 
            score_threshold=settings.SCORE_THRESHOLD
        )
        
        if not results:
            app_logger.info("No relevant context found for the query.")
            return ""

        formatted_context = self._format_documents(results)
        return formatted_context

    def _format_documents(self, search_results: List[Tuple[Document, float]]) -> str:
        """
        Extracts the page content and metadata from the results, joining them 
        into a clear format that prevents LLM hallucination and enables citations.
        """
        context_parts = []
        
        for i, (doc, score) in enumerate(search_results, start=1):
            filename = doc.metadata.get("filename", "Unknown Source")
            page = doc.metadata.get("page", "N/A")
            content = doc.page_content.strip()
            
            # The structure [Source: ...] explicitly tells the LLM where the data originated
            formatted_chunk = f"--- Document {i} [Source: {filename}, Page/Row: {page}] ---\n{content}\n"
            context_parts.append(formatted_chunk)
            
        app_logger.debug(f"Formatted {len(context_parts)} documents for LLM context.")
        return "\n".join(context_parts)