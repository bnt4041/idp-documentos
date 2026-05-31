from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import Base, engine, SessionLocal
from .routers import ai, documents, records, templates

# Crea las tablas si no existen (MVP; en producción usar Alembic)
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="IDP Visual API",
    description="Procesamiento inteligente de documentos con plantillas visuales y OCR local.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(templates.router)
app.include_router(records.router)
app.include_router(ai.router)


@app.on_event("startup")
def _warmup_ollama():
    """Pre-calienta Ollama para que la primera sugerencia no tarde 30s."""
    import threading
    def _warm():
        try:
            from . import ollama_client
            if ollama_client.available():
                ollama_client.generate_json("Di 'ok'")
        except Exception:  # noqa: BLE001
            pass
    threading.Thread(target=_warm, daemon=True).start()


@app.get("/api/health", tags=["health"])
def health():
    return {"status": "ok"}


@app.post("/api/reset", tags=["admin"])
def reset_all_data():
    """Elimina TODOS los datos (documentos, plantillas, registros, ejemplos).
    Útil para reiniciar el MVP a cero."""
    from . import models

    db = SessionLocal()
    try:
        db.query(models.Record).delete()
        db.query(models.LearningExample).delete()
        db.query(models.TemplateField).delete()
        db.query(models.TemplateAnchor).delete()
        db.query(models.Template).delete()
        db.query(models.Document).delete()
        db.commit()
        return {
            "status": "ok",
            "message": "Todos los datos han sido eliminados. La aplicación está como nueva.",
        }
    except Exception as exc:
        db.rollback()
        return {"status": "error", "message": str(exc)}
    finally:
        db.close()
