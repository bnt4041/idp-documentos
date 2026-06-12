"""Endpoints de configuración: gestión del backend de IA (Ollama / DeepSeek)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import ollama_client
from ..config import settings

router = APIRouter(prefix="/api/config", tags=["config"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AIConfigOut(BaseModel):
    """Configuración actual de IA (la API key nunca se devuelve completa)."""
    backend: str                     # "ollama" | "deepseek"
    deepseek_enabled: bool
    deepseek_model: str
    deepseek_base_url: str
    deepseek_api_key_set: bool       # True si hay clave configurada (no la mostramos)
    ollama_model: str
    ollama_url: str
    ollama_gen_url: str
    ollama_gen_api_key_set: bool
    ollama_embed_model: str
    ollama_vision: bool
    # Estado de conectividad
    gen_available: bool
    embed_available: bool
    gen_models: list[str]
    embed_models: list[str]


class AIConfigUpdate(BaseModel):
    """Campos que el usuario puede modificar desde el panel de configuración."""
    deepseek_enabled: bool | None = None
    deepseek_api_key: str | None = None   # "" para borrar
    deepseek_model: str | None = None
    deepseek_base_url: str | None = None
    ollama_model: str | None = None
    ollama_url: str | None = None
    ollama_gen_url: str | None = None
    ollama_gen_api_key: str | None = None
    ollama_vision: bool | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/ai", response_model=AIConfigOut)
def get_ai_config():
    """Devuelve la configuración actual del backend de IA."""
    gen_available = ollama_client.available()
    gen_models = ollama_client.list_models() if gen_available else []
    embed_available = ollama_client.embed_available()
    embed_models = ollama_client.list_embed_models() if embed_available else []

    backend = "deepseek" if (settings.deepseek_enabled and settings.deepseek_api_key) else "ollama"

    return AIConfigOut(
        backend=backend,
        deepseek_enabled=settings.deepseek_enabled,
        deepseek_model=settings.deepseek_model,
        deepseek_base_url=settings.deepseek_base_url,
        deepseek_api_key_set=bool(settings.deepseek_api_key),
        ollama_model=settings.ollama_model,
        ollama_url=settings.ollama_url,
        ollama_gen_url=settings.ollama_gen_url,
        ollama_gen_api_key_set=bool(settings.ollama_gen_api_key),
        ollama_embed_model=settings.ollama_embed_model,
        ollama_vision=settings.ollama_vision,
        gen_available=gen_available,
        embed_available=embed_available,
        gen_models=gen_models,
        embed_models=embed_models,
    )


@router.put("/ai", response_model=AIConfigOut)
def update_ai_config(payload: AIConfigUpdate):
    """Actualiza la configuración de IA en runtime (sin reiniciar)."""
    if payload.deepseek_enabled is not None:
        settings.deepseek_enabled = payload.deepseek_enabled
    if payload.deepseek_api_key is not None:
        settings.deepseek_api_key = payload.deepseek_api_key
    if payload.deepseek_model is not None:
        settings.deepseek_model = payload.deepseek_model
    if payload.deepseek_base_url is not None:
        settings.deepseek_base_url = payload.deepseek_base_url
    if payload.ollama_model is not None:
        settings.ollama_model = payload.ollama_model
    if payload.ollama_url is not None:
        settings.ollama_url = payload.ollama_url
    if payload.ollama_gen_url is not None:
        settings.ollama_gen_url = payload.ollama_gen_url
    if payload.ollama_gen_api_key is not None:
        settings.ollama_gen_api_key = payload.ollama_gen_api_key
    if payload.ollama_vision is not None:
        settings.ollama_vision = payload.ollama_vision

    # Devolver la config actualizada (mismo formato que GET)
    return get_ai_config()


# ---------------------------------------------------------------------------
# Estadísticas de consumo de tokens
# ---------------------------------------------------------------------------

@router.get("/ai/tokens")
def get_token_stats():
    """Estadísticas acumuladas de consumo de tokens (en memoria, desde el arranque)."""
    return ollama_client.get_token_stats()


@router.post("/ai/tokens/reset")
def reset_token_stats():
    """Resetea el contador de tokens a cero."""
    import ollama_client as oc
    oc._token_stats.update({"docs": 0, "tokens_in": 0, "tokens_out": 0, "last_model": ""})
    return {"status": "ok"}
