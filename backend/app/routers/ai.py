"""Endpoints de IA: estado de Ollama y extracción con RAG (LLM local)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from .. import models, ollama_client, rag
from ..config import settings
from ..database import get_db

router = APIRouter(prefix="/api/ai", tags=["ai"])


def _installed(name: str, models: list[str]) -> bool:
    # Coincide "llama3.2" con "llama3.2:latest", "llama3.2:3b", etc.
    return any(m == name or m.split(":")[0] == name for m in models)


@router.get("/status")
def ai_status():
    # Generación (puede ser un backend remoto) y embeddings (backend local) se
    # comprueban por separado: cada modelo contra el backend donde vive.
    gen_available = ollama_client.available()
    gen_models = ollama_client.list_models() if gen_available else []
    model_ok = _installed(settings.ollama_model, gen_models)

    embed_available = ollama_client.embed_available()
    embed_models = ollama_client.list_embed_models() if embed_available else []
    embed_ok = _installed(settings.ollama_embed_model, embed_models)

    return {
        "available": gen_available,
        "ready": gen_available and model_ok and embed_available and embed_ok,
        "model": settings.ollama_model,
        "model_installed": model_ok,
        "embed_model": settings.ollama_embed_model,
        "embed_installed": embed_ok,
        "vision": settings.ollama_vision,
        "models_installed": gen_models,
        "embed_models_installed": embed_models,
    }


@router.post("/extract/{document_id}")
def ai_extract(
    document_id: int,
    template_id: int = Query(..., description="Plantilla cuyos campos se extraen"),
    db: Session = Depends(get_db),
):
    """Extrae los campos de la plantilla con el LLM local + ejemplos confirmados."""
    doc = db.get(models.Document, document_id)
    if not doc:
        raise HTTPException(404, "Documento no encontrado")
    tpl = db.get(models.Template, template_id)
    if not tpl:
        raise HTTPException(404, "Plantilla no encontrada")
    result = rag.extract(db, doc, tpl)
    if not result.get("available"):
        raise HTTPException(
            503,
            f"Ollama no disponible. Arráncalo y descarga el modelo '{settings.ollama_model}'.",
        )
    return result


@router.get("/examples")
def list_examples(template_id: int | None = None, db: Session = Depends(get_db)):
    """Dataset de ejemplos confirmados (base de conocimiento del RAG)."""
    q = db.query(models.LearningExample)
    if template_id is not None:
        q = q.filter_by(template_id=template_id)
    rows = q.order_by(models.LearningExample.created_at.desc()).all()
    return [
        {
            "id": r.id,
            "template_id": r.template_id,
            "document_id": r.document_id,
            "fields": r.fields,
            "has_embedding": bool(r.embedding),
            "created_at": r.created_at,
        }
        for r in rows
    ]
