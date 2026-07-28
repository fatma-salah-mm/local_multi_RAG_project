from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from backend.vectorstore.chroma_store import VectorStoreRepository
from backend.utils.logger import app_logger

class IntelligentRAGService:
    def __init__(self):
        # Initialize the local Ollama model
        self.llm = Ollama(model="llama3.1", temperature=0.1)
        self.vector_repo = VectorStoreRepository()
        self.retriever = self.vector_repo.vector_store.as_retriever(
            search_kwargs={"k": 4}
        )
        self.chat_history = []

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def answer_query(self, user_query: str) -> str:
        try:
            # 1. QUERY REWRITING: Make follow-up questions standalone
            effective_query = user_query
            if self.chat_history:
                rewrite_prompt = ChatPromptTemplate.from_messages([
                    ("system", "Given the chat history and the user's latest question, rewrite the question so that it is a completely standalone query containing all necessary context (e.g., resolving pronouns). Output ONLY the rewritten query."),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}")
                ])
                rewriter = rewrite_prompt | self.llm | StrOutputParser()
                effective_query = rewriter.invoke({
                    "chat_history": self.chat_history,
                    "input": user_query
                }).strip()
                app_logger.info(f"Original Query: '{user_query}' -> Rewritten Query: '{effective_query}'")

            # 2. RETRIEVE DOCUMENTS using the smart query
            retrieved_docs = self.retriever.invoke(effective_query)
            context_text = self.format_docs(retrieved_docs)

            # 3. GENERATION PROMPT (Hybrid: uses documents if available, or falls back gracefully)
            system_prompt = (
                "You are an expert AI assistant. Answer the user's question accurately. "
                "If the question relates to the uploaded documents, use the provided context below. "
                "If the context does not contain the answer, use your own general knowledge.\n\n"
                "Context:\n{context}"
            )
            
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])

            # 4. EXECUTE CHAIN cleanly without legacy dependencies
            chain = qa_prompt | self.llm | StrOutputParser()
            
            answer = chain.invoke({
                "context": context_text,
                "chat_history": self.chat_history,
                "input": user_query
            })

            # 5. UPDATE MEMORY
            self.chat_history.append(HumanMessage(content=user_query))
            self.chat_history.append(AIMessage(content=answer))

            return answer

        except Exception as e:
            app_logger.error(f"Error in Intelligent RAG Service: {e}")
            return f"An error occurred while generating the response: {e}"

# Global instance
rag_service = IntelligentRAGService()