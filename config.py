from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    groq_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str = "rag-index"
    pinecone_environment: str = "us-east-1"

    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 5
    embedding_model: str = "all-MiniLM-L6-v2"   # free local model
    llm_model: str = "llama-3.1-8b-instant"            # free on Groq

    class Config:
        env_file = ".env"
        extra = "ignore"

@lru_cache()
def get_settings() -> Settings:
    return Settings()