from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg2://idp:idp@db:5432/idp"
    ocr_langs: str = "spa+eng"
    storage_dir: str = "/data/uploads"
    # Umbral mínimo de similitud (0..1) para auto-asignar una plantilla a un documento
    match_threshold: float = 0.55

    # RAG con LLM local (Ollama)
    ollama_url: str = "http://ollama:11434"
    ollama_model: str = "llama3.2"
    ollama_vision: bool = False  # si el modelo admite imágenes, enviar el documento
    ollama_embed_model: str = "nomic-embed-text"
    rag_top_k: int = 3  # ejemplos confirmados similares usados como few-shot

    class Config:
        env_file = ".env"


settings = Settings()
