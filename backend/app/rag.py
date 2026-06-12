"""RAG con LLM local (Ollama): retrieval de ejemplos confirmados + extracción.

El 'aprendizaje' se basa en los documentos que el usuario confirma:
  - se guardan como ejemplos (dataset) con su texto OCR y embedding,
  - se recuperan los más parecidos al procesar uno nuevo (few-shot),
  - sus posiciones de campos actualizan la plantilla.
"""
from __future__ import annotations

import base64
import json
import os
import unicodedata
from typing import Any

import numpy as np

from . import models, ollama_client
from .config import settings


def ocr_text(words: list[dict[str, Any]]) -> str:
    """Texto plano del documento, en orden de lectura aproximado."""
    ordered = sorted(words, key=lambda w: (round(w["box"]["y"], 2), w["box"]["x"]))
    return " ".join(w["text"] for w in ordered)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    va, vb = np.asarray(a), np.asarray(b)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def retrieve(
    db, template_id: int | None, query_embedding: list[float], k: int
) -> list[models.LearningExample]:
    """Ejemplos confirmados más parecidos (por embedding del texto OCR)."""
    q = db.query(models.LearningExample)
    if template_id is not None:
        q = q.filter_by(template_id=template_id)
    scored = []
    for ex in q.all():
        if not ex.embedding:
            continue
        scored.append((_cosine(query_embedding, ex.embedding), ex))
    scored.sort(key=lambda s: s[0], reverse=True)
    return [ex for _, ex in scored[:k]]


def _norm(s: str) -> str:
    """Normaliza para comparar: minúsculas, sin acentos ni signos."""
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return "".join(c for c in s if c.isalnum())


def _region_from_value(words: list[dict], value: str) -> dict | None:
    """Localiza el valor en las palabras OCR y devuelve su bounding box (img 0..1).

    El LLM da el valor limpio; aquí lo casamos con las palabras detectadas para
    obtener la región (proporcionalidad para el OCR), de forma determinista.
    """
    targets = [_norm(t) for t in value.split() if _norm(t)]
    if not targets:
        return None
    boxes = []
    for w in words:
        wt = _norm(w["text"])
        if not wt:
            continue
        for t in targets:
            if wt == t or (len(t) >= 3 and (t in wt or wt in t)):
                boxes.append(w["box"])
                break
    if not boxes:
        return None
    x0 = min(b["x"] for b in boxes)
    y0 = min(b["y"] for b in boxes)
    x1 = max(b["x"] + b["w"] for b in boxes)
    y1 = max(b["y"] + b["h"] for b in boxes)
    return {"x": round(x0, 5), "y": round(y0, 5), "w": round(x1 - x0, 5), "h": round(y1 - y0, 5)}


def _build_prompt(template, examples, text: str) -> str:
    field_lines = "\n".join(f'  - "{f.key}": {f.name}' for f in template.fields)

    shots = []
    for ex in examples:
        target = {k: v.get("value", "") for k, v in (ex.fields or {}).items()}
        shots.append(
            f"TEXTO:\n{ex.ocr_text[:800]}\nJSON:\n{json.dumps(target, ensure_ascii=False)}"
        )
    shots_block = "\n\n".join(shots) if shots else "(sin ejemplos todavía)"

    hints: dict[str, list[str]] = {}
    for ex in examples:
        for k, v in (ex.fields or {}).items():
            val = v.get("value")
            if val:
                hints.setdefault(k, [])
                if val not in hints[k]:
                    hints[k].append(val)
    hints_block = (
        "\n".join(f'  - "{k}": ej. {", ".join(vs[:5])}' for k, vs in hints.items())
        or "(ninguna)"
    )

    return (
        "Eres un extractor de datos de documentos. A partir del TEXTO OCR del "
        "documento, devuelve ÚNICAMENTE un objeto JSON con EXACTAMENTE estas claves. "
        "Para cada clave, el valor LIMPIO (sin la etiqueta). Si un dato no aparece, "
        "cadena vacía. No inventes.\n\n"
        f"CLAVES Y DESCRIPCIÓN:\n{field_lines}\n\n"
        f"PISTAS (valores ya vistos):\n{hints_block}\n\n"
        f"EJEMPLOS CONFIRMADOS:\n{shots_block}\n\n"
        f"DOCUMENTO A PROCESAR:\nTEXTO:\n{text[:2000]}\n\n"
        "Responde solo con el JSON."
    )


def _image_b64(document) -> list[str]:
    if not document or not os.path.exists(document.stored_path):
        return []
    with open(document.stored_path, "rb") as fh:
        return [base64.b64encode(fh.read()).decode()]


def extract(db, document, template) -> dict[str, Any]:
    """Extrae los campos con el LLM local + RAG. Degrada si Ollama no está."""
    if not ollama_client.available():
        return {"available": False, "fields": {}, "used_examples": 0, "model": settings.ollama_model}

    words = document.ocr_words or []
    text = ocr_text(words)
    try:
        emb = ollama_client.embed(text)
    except Exception:  # noqa: BLE001
        emb = []
    examples = retrieve(db, template.id, emb, settings.rag_top_k) if emb else []

    prompt = _build_prompt(template, examples, text)
    images = _image_b64(document) if settings.ollama_vision else None
    try:
        raw = ollama_client.generate_json(prompt, images=images, force_json=True)
        data = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        return {
            "available": True,
            "error": str(exc),
            "fields": {},
            "used_examples": len(examples),
            "model": settings.ollama_model,
        }

    fields = {}
    for f in template.fields:
        entry = data.get(f.key)
        if isinstance(entry, dict):
            entry = entry.get("value", "")
        value = str(entry or "")
        region = _region_from_value(words, value)
        fields[f.key] = {"name": f.name, "value": value, "region": region}
    return {
        "available": True,
        "fields": fields,
        "used_examples": len(examples),
        "model": settings.ollama_model,
    }


# ---------------------------------------------------------------------------
# Extracción libre (sin plantilla): el LLM decide qué extraer
# ---------------------------------------------------------------------------

_FREEFORM_PROMPT = (
    "Eres un extractor inteligente de documentos. Te voy a dar el TEXTO OCR de un "
    "documento (factura, DNI, contrato, formulario, permiso de circulación, nómina, "
    "etc.).\n\n"
    "Tu tarea:\n"
    "1. Identifica QUÉ TIPO de documento es.\n"
    "2. Extrae TODOS los datos relevantes que encuentres.\n"
    "3. Devuelve ÚNICAMENTE un objeto JSON con este formato:\n\n"
    '{{"tipo_documento": "factura", "datos": {{"campo1": "valor1", "campo2": "valor2", ...}}}}\n\n'
    "REGLAS:\n"
    "- Usa claves descriptivas en snake_case (ej. 'nombre_cliente', 'fecha_emision', "
    "'importe_total', 'dni', 'matricula').\n"
    "- Los valores deben ser el texto LIMPIO, sin la etiqueta (ej. si dice "
    "'Nombre: Juan Pérez' -> 'Juan Pérez').\n"
    "- Si un dato está compuesto por varias partes (ej. dirección en 2 líneas), "
    "únelas en un solo valor.\n"
    "- Si no estás seguro de algo, no inventes: omite ese campo.\n"
    "- No incluyas comentarios, solo el JSON.\n\n"
    "TEXTO OCR DEL DOCUMENTO:\n{text}\n\n"
    "Responde solo con el JSON."
)


def extract_freeform(db, document) -> dict[str, Any]:
    """Extrae datos con el LLM sin plantilla: el modelo decide qué campos extraer.

    Devuelve {available, tipo_documento, fields: {key: {value, region}}, model}.
    Si el LLM no está disponible, devuelve available=False para que el caller
    use un fallback determinista (field_suggestions).
    """
    if not ollama_client.available():
        import sys
        print("[rag] extract_freeform: LLM no disponible", file=sys.stderr)
        return {
            "available": False,
            "tipo_documento": None,
            "fields": {},
            "model": settings.ollama_model,
        }

    words = document.ocr_words or []
    text = ocr_text(words)
    import sys
    print(f"[rag] extract_freeform: {len(words)} palabras OCR, texto={text[:120]}...", file=sys.stderr)

    prompt = _FREEFORM_PROMPT.format(text=text[:3000])
    images = _image_b64(document) if settings.ollama_vision else None

    try:
        raw = ollama_client.generate_json(
            prompt, images=images, force_json=True, num_predict=512,
        )
        print(f"[rag] extract_freeform: respuesta raw={raw[:200]}...", file=sys.stderr)
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"[rag] extract_freeform: JSON inválido: {exc}", file=sys.stderr)
        return {
            "available": True,
            "error": f"JSON inválido: {exc}",
            "tipo_documento": None,
            "fields": {},
            "model": settings.ollama_model,
        }
    except Exception as exc:  # noqa: BLE001
        print(f"[rag] extract_freeform: error: {exc}", file=sys.stderr)
        return {
            "available": True,
            "error": str(exc),
            "tipo_documento": None,
            "fields": {},
            "model": settings.ollama_model,
        }

    tipo = data.get("tipo_documento", "desconocido") if isinstance(data, dict) else "desconocido"
    datos = data.get("datos", {}) if isinstance(data, dict) else {}

    fields = {}
    for key, value in datos.items():
        value_str = str(value or "")
        region = _region_from_value(words, value_str)
        fields[key] = {"value": value_str, "region": region}

    return {
        "available": True,
        "tipo_documento": tipo,
        "fields": fields,
        "model": settings.ollama_model,
    }


def learn_from_record(db, template, document, regions: dict[str, dict], values: dict) -> None:
    """Guarda el ejemplo confirmado y actualiza posiciones/diccionario de la plantilla.

    `regions` = {key: {x,y,w,h}} en coords de imagen (0..1). Se convierten a
    coords relativas al borde del documento para la plantilla.
    """
    border = document.border or {"x": 0, "y": 0, "w": 1, "h": 1}
    bx, by = border["x"], border["y"]
    bw, bh = max(border["w"], 1e-6), max(border["h"], 1e-6)

    def to_rel(r):
        return {
            "x": round((r["x"] - bx) / bw, 5),
            "y": round((r["y"] - by) / bh, 5),
            "w": round(r["w"] / bw, 5),
            "h": round(r["h"] / bh, 5),
        }

    # 1) Guarda el ejemplo y actualiza el diccionario (sample_text). IMPORTANTE:
    #    NO se sobrescribe la geometría (x/y/w/h) de los campos ya existentes en la
    #    plantilla: esas regiones llegan en coords del DOCUMENTO (alineadas por
    #    anclas a su geometría concreta) y reescribirlas haría DERIVAR la plantilla
    #    en cada confirmación. La geometría de la plantilla solo se edita a mano en
    #    el editor de plantillas. Los campos NUEVOS sí se crean (no había geometría).
    existing = {f.key: f for f in template.fields}
    example_fields: dict[str, Any] = {}
    for key, region in (regions or {}).items():
        rel = to_rel(region)
        if key in existing:
            f = existing[key]
            if values.get(key):
                f.sample_text = str(values[key])
        else:
            template.fields.append(
                models.TemplateField(
                    name=key,
                    key=key,
                    data_type="text",
                    x=rel["x"],
                    y=rel["y"],
                    w=rel["w"],
                    h=rel["h"],
                    sample_text=str(values.get(key, "")),
                )
            )
        example_fields[key] = {"value": values.get(key, ""), "region": rel}

    # 2) Guarda el ejemplo (dataset + embedding) para el retrieval futuro
    text = ocr_text(document.ocr_words or [])
    embedding: list[float] = []
    try:
        if ollama_client.available():
            embedding = ollama_client.embed(text)
    except Exception:  # noqa: BLE001
        embedding = []

    db.add(
        models.LearningExample(
            template_id=template.id,
            document_id=document.id,
            fields=example_fields,
            ocr_text=text,
            embedding=embedding,
        )
    )
