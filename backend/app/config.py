from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg2://idp:idp@db:5432/idp"
    ocr_langs: str = "spa+eng"
    storage_dir: str = "/data/uploads"
    # Umbral mínimo de similitud (0..1) para auto-asignar una plantilla a un documento
    match_threshold: float = 0.55

    # Preprocesado de imagen
    auto_deskew: bool = False       # enderezado automático (OSD + Hough fino).
                                    # False por defecto: el Hough puede malinterpretar
                                    # líneas de tablas/bordes y poner el documento en oblicuo.
                                    # Actívalo solo si los documentos llegan torcidos.
    multi_filter_ocr: bool = False  # probar múltiples filtros y elegir el mejor OCR
    multi_angle_matching: bool = True  # probar matching en 0°/90°/180°/270°

    # Matching visual + geométrico
    visual_match_weight: float = 0.60   # peso del matching visual (ORB) vs geométrico
    visual_match_threshold: float = 0.55  # umbral mínimo de similitud combinada (0..1)
    zone_detection_method: str = "orb"  # "orb", "template", o "none"
    zone_match_threshold: float = 0.3   # umbral mínimo para template matching visual

    # Hitos / anclas de plantilla (puntos fijos de referencia)
    anchor_match_weight: float = 0.5    # peso de las anclas en el score combinado (si existen)
    anchor_text_threshold: float = 0.7  # umbral SequenceMatcher para dar texto por casado
    anchor_image_threshold: float = 0.3  # umbral de detect_best_zone para el patch

    # RAG con LLM local (Ollama)
    ollama_url: str = "http://ollama:11434"
    ollama_model: str = "llama3.2"
    ollama_vision: bool = False  # si el modelo admite imágenes, enviar el documento
    ollama_embed_model: str = "nomic-embed-text"
    rag_top_k: int = 3  # ejemplos confirmados similares usados como few-shot

    # Backend de GENERACIÓN separado (opcional).
    # Permite mandar la generación de texto a otro Ollama (p.ej. un server remoto
    # más potente) mientras los embeddings siguen en el Ollama local.
    # Si ollama_gen_url está vacío, se usa ollama_url (todo en local).
    ollama_gen_url: str = ""          # ej. "https://bot.dealerbest.com/ollama"
    ollama_gen_api_key: str = ""      # Bearer token del server de generación (si lo pide)

    class Config:
        env_file = ".env"


settings = Settings()
