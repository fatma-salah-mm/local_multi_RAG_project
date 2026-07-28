from fastapi import APIRouter
from backend.services.rag_service import rag_service
from backend.services.ingestion_service import ingest_local_files
from backend.models.schemas import QueryRequest, QueryResponse

router = APIRouter()

@router.get("/status")
def get_status():
    return {"status": "healthy", "service": "Enterprise RAG Backend"}

@router.post("/ingest")
def trigger_ingestion():
    """Triggers the local file indexing service."""
    ingest_local_files()
    return {"message": "Files ingested and indexed successfully."}

@router.post("/query", response_model=QueryResponse)
def query_rag(payload: QueryRequest):
    answer = rag_service.answer_query(payload.query)
    return {"answer": answer}