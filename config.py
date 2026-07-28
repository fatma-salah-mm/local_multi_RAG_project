import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Ensure base directories exist
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "db")
LOG_DIR = os.path.join(BASE_DIR, "logs")

for directory in [DATA_DIR, DB_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

class Settings(BaseSettings):
    """
    Centralized configuration class utilizing Pydantic for type safety 
    and environment variable parsing.
    """
    
    # App Settings
    APP_NAME: str = Field(default="Enterprise RAG Assistant")
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    
    # LLM Settings (Ollama)
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434")
    OLLAMA_MODEL: str = Field(default="llama3.1")
    TEMPERATURE: float = 0.1
    # Embedding Settings
    EMBEDDING_MODEL: str = Field(default="BAAI/bge-small-en-v1.5")
    
    # Vector Store Settings
    CHROMA_PERSIST_DIR: str = Field(default=DB_DIR)
    CHROMA_COLLECTION_NAME: str = Field(default="enterprise_docs")
    
    # Chunking Settings
    CHUNK_SIZE: int = Field(default=800)
    CHUNK_OVERLAP: int = Field(default=150)
    
    # Retrieval Settings
    TOP_K: int = Field(default=4)
    SCORE_THRESHOLD: float = Field(default=0.3)
    
    # Directory Settings
    UPLOAD_DIR: str = Field(default=DATA_DIR)
    LOG_FILE_PATH: str = Field(default=os.path.join(LOG_DIR, "app.log"))

    # Pydantic v2 configuration for loading .env files
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

# Instantiate a global settings object to be injected throughout the app
settings = Settings()