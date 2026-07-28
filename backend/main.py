from fastapi import FastAPI
from backend.api.routes import router

app = FastAPI(
    title="Enterprise RAG Backend",
    version="1.0.0"
)

# Connect the routes
app.include_router(router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Multi-RAG Backend API!"}