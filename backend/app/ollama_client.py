"""Cliente ligero para un LLM local servido por Ollama (embeddings + generación)."""
from __future__ import annotations

import httpx

from .config import settings


def available() -> bool:
    """True si el servidor Ollama responde."""
    try:
        r = httpx.get(f"{settings.ollama_url}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


def list_models() -> list[str]:
    try:
        r = httpx.get(f"{settings.ollama_url}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:  # noqa: BLE001
        return []


def embed(text: str) -> list[float]:
    """Vector de embedding del texto (para el retrieval del RAG)."""
    r = httpx.post(
        f"{settings.ollama_url}/api/embeddings",
        json={"model": settings.ollama_embed_model, "prompt": text},
        timeout=60,
    )
    r.raise_for_status()
    return r.json().get("embedding", [])


def generate_json(prompt: str, images: list[str] | None = None) -> str:
    """Genera una respuesta forzando formato JSON. `images` = lista base64 (visión)."""
    payload: dict = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }
    if images and settings.ollama_vision:
        payload["images"] = images
    r = httpx.post(f"{settings.ollama_url}/api/generate", json=payload, timeout=300)
    r.raise_for_status()
    return r.json().get("response", "")
