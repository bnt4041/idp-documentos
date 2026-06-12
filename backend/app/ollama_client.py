"""Cliente ligero para Ollama (y opcionalmente DeepSeek API).

Soporta tres backends de generación (en orden de prioridad):
  1. DeepSeek API  -> si `deepseek_enabled=True` (API cloud OpenAI-compatible).
  2. Ollama remoto -> `ollama_gen_url` si está definido (server más potente, con token).
  3. Ollama local  -> `ollama_url` (por defecto).

Embeddings -> siempre el Ollama local (`ollama_url`).
"""
from __future__ import annotations

import httpx

from .config import settings

# ---------------------------------------------------------------------------
# Contador de tokens acumulado (en memoria, se resetea al reiniciar)
# ---------------------------------------------------------------------------

_token_stats = {
    "docs": 0,
    "tokens_in": 0,
    "tokens_out": 0,
    "last_model": "",
}


def _record_tokens(tokens_in: int, tokens_out: int, model: str = "") -> None:
    """Acumula tokens de una llamada de generación."""
    _token_stats["docs"] += 1
    _token_stats["tokens_in"] += tokens_in
    _token_stats["tokens_out"] += tokens_out
    if model:
        _token_stats["last_model"] = model


def get_token_stats() -> dict:
    """Estadísticas acumuladas de consumo de tokens."""
    s = dict(_token_stats)
    if s["docs"] > 0:
        s["avg_in"] = round(s["tokens_in"] / s["docs"])
        s["avg_out"] = round(s["tokens_out"] / s["docs"])
    else:
        s["avg_in"] = 0
        s["avg_out"] = 0
    s["cost_estimate"] = _estimate_cost(s)
    return s


def _estimate_cost(stats: dict) -> str:
    """Estimación orientativa del coste con DeepSeek (V3)."""
    if stats["docs"] == 0:
        return "—"
    # Precios deepseek-chat: 0.27 $/M input, 1.10 $/M output
    cost = (stats["tokens_in"] / 1_000_000) * 0.27 + (stats["tokens_out"] / 1_000_000) * 1.10
    if cost < 0.01:
        return f"~${cost:.5f}  (~{cost * 0.93:.5f} €)"
    return f"~${cost:.4f}  (~{cost * 0.93:.4f} €)"


# ---------------------------------------------------------------------------
# Selección de backend de generación
# ---------------------------------------------------------------------------

def _use_deepseek() -> bool:
    """True si la generación debe ir por DeepSeek API."""
    return settings.deepseek_enabled and bool(settings.deepseek_api_key)


def _gen_url() -> str:
    """URL base del backend de generación."""
    if _use_deepseek():
        return settings.deepseek_base_url.rstrip("/")
    return (settings.ollama_gen_url or settings.ollama_url).rstrip("/")


def _gen_headers() -> dict[str, str]:
    """Cabeceras para el backend de generación."""
    if _use_deepseek():
        return {
            "Authorization": f"Bearer {settings.deepseek_api_key}",
            "Content-Type": "application/json",
        }
    if settings.ollama_gen_api_key:
        return {"Authorization": f"Bearer {settings.ollama_gen_api_key}"}
    return {}


# ---------------------------------------------------------------------------
# Estado / listado (sobre el backend de generación)
# ---------------------------------------------------------------------------

def available() -> bool:
    """True si el backend de generación responde."""
    if _use_deepseek():
        # DeepSeek: intentamos listar modelos (o un ping ligero)
        try:
            r = httpx.get(
                f"{_gen_url()}/v1/models",
                headers=_gen_headers(),
                timeout=10,
            )
            return r.status_code == 200
        except Exception:  # noqa: BLE001
            return False
    try:
        r = httpx.get(f"{_gen_url()}/api/tags", headers=_gen_headers(), timeout=5)
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


def list_models() -> list[str]:
    if _use_deepseek():
        try:
            r = httpx.get(
                f"{_gen_url()}/v1/models",
                headers=_gen_headers(),
                timeout=10,
            )
            r.raise_for_status()
            return [m["id"] for m in r.json().get("data", [])]
        except Exception:  # noqa: BLE001
            return [settings.deepseek_model]
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

    - force_json: fuerza salida JSON (modo nativo en Ollama, response_format en DeepSeek).
    - num_predict: máximo de tokens de salida (sube si esperas muchos campos).
    """
    if _use_deepseek():
        return _generate_deepseek(prompt, images, force_json=force_json, num_predict=num_predict)

    # --- Ollama (local o remoto) ---
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
    body = r.json()
    # Extraer tokens usados (Ollama)
    tokens_in = body.get("prompt_eval_count", 0)
    tokens_out = body.get("eval_count", 0)
    _record_tokens(tokens_in, tokens_out, settings.ollama_model)
    return body.get("response", "")


def _generate_deepseek(
    prompt: str,
    images: list[str] | None = None,
    *,
    force_json: bool = False,
    num_predict: int = 256,
) -> str:
    """Genera respuesta usando la API OpenAI-compatible de DeepSeek."""
    messages: list[dict] = [{"role": "user", "content": prompt}]

    # Si hay imágenes y el modelo las soporta, las añadimos como image_url
    if images:
        content_parts: list[dict] = []
        for img_b64 in images:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
            })
        content_parts.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content_parts}]

    payload: dict = {
        "model": settings.deepseek_model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": num_predict,
        "stream": False,
    }
    if force_json:
        payload["response_format"] = {"type": "json_object"}

    r = httpx.post(
        f"{_gen_url()}/v1/chat/completions",
        json=payload,
        headers=_gen_headers(),
        timeout=120,
    )
    r.raise_for_status()
    body = r.json()
    # Extraer tokens usados (DeepSeek / OpenAI-compatible)
    usage = body.get("usage", {})
    tokens_in = usage.get("prompt_tokens", 0)
    tokens_out = usage.get("completion_tokens", 0)
    _record_tokens(tokens_in, tokens_out, settings.deepseek_model)
    return body["choices"][0]["message"]["content"]
