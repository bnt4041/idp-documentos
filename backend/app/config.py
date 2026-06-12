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
    # Error de reproyección relativo máximo (0..1) al ajustar la similitud por anclas.
    # Si se supera (3+ anclas que no encajan = formulario distinto), no se aplica la
    # transformación para no distorsionar las coordenadas de los campos.
    anchor_fit_max_error: float = 0.15
    # Límites de "cordura" de la afín por anclas: si los supera, la forma es
    # demasiado distinta (otro formulario) y se descarta la afín.
    anchor_max_anisotropy: float = 6.0  # ratio máx escala_x/escala_y (permite recortes muy distintos del mismo form)
    anchor_max_shear: float = 0.2       # cizalla máxima admitida (alta = otro formulario)
    # Filtros (en cascada) para localizar anclas de imagen difíciles (B/N, bajo contraste)
    anchor_filters: list[str] = ["grayscale", "binary", "clahe"]

    # Re-OCR por región (recorte+ampliación, PSM 6) para campos flojos. Mejora los
    # valores que el OCR de página completa lee mal. Se aplica tanto en el procesado
    # interactivo como en el job de segundo plano.
    region_ocr_refine: bool = True
    region_ocr_min_conf: float = 80.0   # se re-OCR si el campo está vacío o por debajo de esto

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

    # -----------------------------------------------------------------------
    # Backend DeepSeek (API cloud, compatible con OpenAI)
    # -----------------------------------------------------------------------
    # Si deepseek_enabled=True, la generación usa la API de DeepSeek en lugar
    # de Ollama (local o remoto). Los embeddings siguen en Ollama local.
    deepseek_enabled: bool = False
    deepseek_api_key: str = ""        # Clave API de DeepSeek (https://platform.deepseek.com)
    deepseek_model: str = "deepseek-chat"  # deepseek-chat (V3) o deepseek-reasoner (R1)
    deepseek_base_url: str = "https://api.deepseek.com"

    class Config:
        env_file = ".env"


settings = Settings()
