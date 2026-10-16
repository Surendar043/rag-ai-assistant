from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Models
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    LLM_MODEL: str = "gpt-3.5-turbo"
    LLM_TEMPERATURE: float = 0.0

    # Chunking
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # Retrieval
    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.3

    # Paths
    DATA_DIR: str = "data"
    VECTOR_STORE_PATH: str = "data/processed/index.faiss"
    EVAL_DATASET: str = "eval/dataset.json"
    EVAL_EDGE_CASES: str = "eval/edge_cases.json"

    class Config:
        env_file = ".env"


settings = Settings()
