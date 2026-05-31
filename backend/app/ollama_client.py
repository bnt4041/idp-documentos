"""Cliente ligero para Ollama.

Soporta dos backends:
  - Embeddings  -> siempre el Ollama local (`ollama_url`).
  - Generación  -> `ollama_gen_url` si está definido (p.ej. un server remoto más
    potente, con token Bearer), o `ollama_url` en su defecto.
"""
from __future__ import annotations

import httpx

from .config import settings


# ---------------------------------------------------------------------------
# Selección de backend (generación vs embeddings)
# ---------------------------------------------------------------------------

def _gen_url() -> str:
    """URL base del backend de generación (remoto si está configurado)."""
    return (settings.ollama_gen_url or settings.ollama_url).rstrip("/")


def _gen_headers() -> dict[str, str]:
    """Cabeceras para el backend de generación (Bearer si hay token)."""
    if settings.ollama_gen_api_key:
        return {"Authorization": f"Bearer {settings.ollama_gen_api_key}"}
    return {}


# ---------------------------------------------------------------------------
# Estado / listado (sobre el backend de generación)
# ---------------------------------------------------------------------------

def available() -> bool:
    """True si el backend de generación responde."""
    try:
        r = httpx.get(f"{_gen_url()}/api/tags", headers=_gen_headers(), timeout=5)
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


def list_models() -> list[str]:
    try:
        r = httpx.get(f"{_gen_url()}/api/tags", headers=_gen_headers(), timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:  # noqa: BLE001
        return []


# ---------------------------------------------------------------------------
# Embeddings (siempre Ollama local)
# ---------------------------------------------------------------------------

def embed_available() -> bool:
    """True si el backend local de embeddings responde."""
    try:
        r = httpx.get(f"{settings.ollama_url.rstrip('/')}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


def list_embed_models() -> list[str]:
    """Modelos instalados en el backend LOCAL de embeddings."""
    try:
        r = httpx.get(f"{settings.ollama_url.rstrip('/')}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:  # noqa: BLE001
        return []


def embed(text: str) -> list[float]:
    """Vector de embedding del texto (para el retrieval del RAG). Backend local."""
    r = httpx.post(
        f"{settings.ollama_url.rstrip('/')}/api/embeddings",
        json={"model": settings.ollama_embed_model, "prompt": text},
        timeout=60,
    )
    r.raise_for_status()
    return r.json().get("embedding", [])


# ---------------------------------------------------------------------------
# Generación (backend de generación, remoto o local)
# ---------------------------------------------------------------------------

def generate_json(
    prompt: str,
    images: list[str] | None = None,
    *,
    force_json: bool = False,
    num_predict: int = 256,
) -> str:
    """Genera una respuesta. `images` = lista base64 (visión).

    - force_json: usa el modo `format=json` de Ollama para forzar JSON válido.
    - num_predict: máximo de tokens de salida (sube si esperas muchos campos).
    """
    payload: dict = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": num_predict},
    }
    if force_json:
        payload["format"] = "json"
    if images and settings.ollama_vision:
        payload["images"] = images
    # Timeout largo: la primera llamada carga el modelo en RAM (~10-30s)
    r = httpx.post(
        f"{_gen_url()}/api/generate",
        json=payload,
        headers=_gen_headers(),
        timeout=300,
    )
    r.raise_for_status()
    return r.json().get("response", "")
