"""Emparejado de un documento con la plantilla más parecida y extracción de campos.

La similitud combina tres señales de la firma geométrica:
  - aspect_ratio (proporción de la forma)
  - bg_color (color de fondo)
  - density (organización geométrica del texto en rejilla)
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def _aspect_score(a: float, b: float) -> float:
    if not a or not b:
        return 0.0
    ratio = min(a, b) / max(a, b)
    return ratio  # 1.0 = idéntico


def _color_score(a: list[int], b: list[int]) -> float:
    if not a or not b:
        return 0.0
    dist = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    max_dist = math.sqrt(3 * 255**2)
    return 1.0 - dist / max_dist


def _density_score(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    va, vb = np.asarray(a), np.asarray(b)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))  # similitud coseno


def similarity(sig_a: dict[str, Any], sig_b: dict[str, Any]) -> float:
    """Puntuación 0..1 entre dos firmas. Ponderada hacia la organización del texto."""
    aspect = _aspect_score(sig_a.get("aspect_ratio", 0), sig_b.get("aspect_ratio", 0))
    color = _color_score(sig_a.get("bg_color", []), sig_b.get("bg_color", []))
    density = _density_score(sig_a.get("density", []), sig_b.get("density", []))
    return round(0.25 * aspect + 0.20 * color + 0.55 * density, 4)


def best_template(
    doc_signature: dict[str, Any], templates: list
) -> tuple[Any | None, float]:
    """Devuelve (plantilla, score) de la plantilla más parecida, o (None, 0)."""
    best, best_score = None, 0.0
    for tpl in templates:
        score = similarity(doc_signature, tpl.signature or {})
        if score > best_score:
            best, best_score = tpl, score
    return best, best_score


def _box_center(box: dict[str, float]) -> tuple[float, float]:
    return box["x"] + box["w"] / 2, box["y"] + box["h"] / 2


def extract_field(
    field: dict[str, Any], words: list[dict[str, Any]]
) -> dict[str, Any]:
    """Extrae el valor de un campo recogiendo las palabras cuyo centro cae en la región.

    `field` debe traer x, y, w, h (normalizados). Las palabras se ordenan por
    posición (línea, luego columna) y se concatenan.
    """
    fx, fy, fw, fh = field["x"], field["y"], field["w"], field["h"]
    # Pequeño margen para tolerar variaciones de alineación entre documentos
    pad_x = fw * 0.10
    pad_y = fh * 0.20
    x0, y0 = fx - pad_x, fy - pad_y
    x1, y1 = fx + fw + pad_x, fy + fh + pad_y

    hits = []
    for word in words:
        cx, cy = _box_center(word["box"])
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            hits.append(word)

    # Orden de lectura: por filas (y) y dentro de la fila por x
    hits.sort(key=lambda w: (round(w["box"]["y"], 2), w["box"]["x"]))
    value = " ".join(h["text"] for h in hits).strip()
    confidence = (
        round(sum(h["conf"] for h in hits) / len(hits), 1) if hits else 0.0
    )
    return {"value": value, "confidence": confidence, "n_words": len(hits)}


FULL_BORDER = {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}


def field_to_image(field: dict[str, float], border: dict[str, float]) -> dict[str, float]:
    """Convierte coords relativas al borde -> coords del escaneo completo (0..1)."""
    b = border or FULL_BORDER
    return {
        "x": round(b["x"] + field["x"] * b["w"], 5),
        "y": round(b["y"] + field["y"] * b["h"], 5),
        "w": round(field["w"] * b["w"], 5),
        "h": round(field["h"] * b["h"], 5),
    }


def extract_all(
    template, words: list[dict[str, Any]], border: dict[str, float] | None = None
) -> dict[str, Any]:
    """Extrae todos los campos. Los campos son relativos al borde del documento;
    se mapean al borde detectado en este documento antes de extraer.

    Devuelve {key: {value, confidence, region (en coords de imagen), ...}}.
    """
    result: dict[str, Any] = {}
    for f in template.fields:
        rel = {"x": f.x, "y": f.y, "w": f.w, "h": f.h}
        region = field_to_image(rel, border or FULL_BORDER)
        extracted = extract_field(region, words)
        result[f.key] = {
            "name": f.name,
            "data_type": f.data_type,
            "region": region,
            **extracted,
        }
    return result
