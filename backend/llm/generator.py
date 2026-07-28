from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from config import settings
from backend.utils.logger import app_logger

class LLMGenerator:
    """
    Handles local LLM text generation using Ollama and custom RAG prompts.
    """

    def __init__(self):
        """Initializes the Ollama LLM client with configured parameters."""
        try:
            self.llm = Ollama(
                model=settings.OLLAMA_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=settings.TEMPERATURE,
            )
            app_logger.info(f"Initialized Ollama LLM using model: {settings.OLLAMA_MODEL}")
        except Exception as e:
            app_logger.error(f"Failed to initialize Ollama LLM: {str(e)}")
            raise e

        # Strict RAG prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an Enterprise AI Knowledge Assistant.
Your task is to answer user questions accurately based ONLY on the provided context below.

Guidelines:
1. Ground your answer strictly in the provided Context.
2. If the context does not contain enough information to answer the question, clearly state: "I cannot find sufficient information in the provided documents to answer your question."
3. Include inline source citations whenever referencing specific information (e.g., [Source: report.pdf, Page: 4]).
4. Keep your tone professional, clear, and objective.

Context:
{context}"""),
            ("user", "{question}")
        ])

    def generate_answer(self, query: str, context: str) -> str:
        """
        Combines query and retrieved context into the prompt template and streams or invokes the LLM.

        Args:
            query: The user's question.
            context: Formatted context string retrieved from the vector store.

        Returns:
            The generated response string from Llama 3.1.
        """
        if not context.strip():
            return "I could not find any relevant information in the uploaded documents to answer your question."

        try:
            chain = self.prompt_template | self.llm
            response = chain.invoke({"context": context, "question": query})
            app_logger.info(f"Successfully generated answer for query: '{query}'")
            return response
        except Exception as e:
            app_logger.error(f"Error during LLM generation: {str(e)}")
            return f"An error occurred while generating the response from Ollama: {str(e)}"