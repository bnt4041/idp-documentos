from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import Base, engine
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


@app.get("/api/health", tags=["health"])
def health():
    return {"status": "ok"}
