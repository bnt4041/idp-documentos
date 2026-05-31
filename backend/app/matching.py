"""Emparejado de un documento con la plantilla más parecida y extracción de campos.

La similitud combina tres señales de la firma geométrica:
  - aspect_ratio (proporción de la forma)
  - bg_color (color de fondo)
  - density (organización geométrica del texto en rejilla)

Además, soporta detección por zonas (template matching visual y ORB keypoints)
y emparejado multi-ángulo (prueba 0°, 90°, 180°, 270°).
"""
from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
from PIL import Image

from . import preprocessing


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
    """Puntuación 0..1 entre dos firmas geométricas. Ponderada hacia la organización del texto."""
    aspect = _aspect_score(sig_a.get("aspect_ratio", 0), sig_b.get("aspect_ratio", 0))
    color = _color_score(sig_a.get("bg_color", []), sig_b.get("bg_color", []))
    density = _density_score(sig_a.get("density", []), sig_b.get("density", []))
    return round(0.25 * aspect + 0.20 * color + 0.55 * density, 4)


def combined_similarity(
    geom_score: float,
    visual_score: float,
    visual_weight: float = 0.6,
) -> float:
    """Mezcla el score geométrico (firma) con el visual (ORB sobre imagen).

    Args:
        geom_score: similitud de firma geométrica 0..1
        visual_score: similitud visual (ORB keypoints) 0..1
        visual_weight: peso de la parte visual (0..1). Default 0.6 = 60% visual.

    Returns:
        score combinado 0..1
    """
    if visual_score <= 0:
        return geom_score  # sin similitud visual, usar solo geométrica
    return round(
        (1 - visual_weight) * geom_score + visual_weight * visual_score, 4
    )


def best_template(
    doc_signature: dict[str, Any],
    templates: list,
    doc_image_path: str | None = None,
    visual_weight: float = 0.6,
) -> tuple[Any | None, float, float, float]:
    """Devuelve (plantilla, score_combinado, score_geométrico, score_visual).

    Si hay doc_image_path y la plantilla tiene imagen de muestra, combina
    similitud geométrica + visual (ORB multi-ángulo y multi-filtro).
    """
    best, best_score, best_geom, best_vis = None, 0.0, 0.0, 0.0
    for tpl in templates:
        geom = similarity(doc_signature, tpl.signature or {})
        vis = 0.0
        if doc_image_path and tpl.sample_document_id:
            try:
                # Intentar obtener la ruta de la imagen de muestra
                from . import models as _m
                from .database import SessionLocal
                db = SessionLocal()
                try:
                    sample = db.query(_m.Document).get(tpl.sample_document_id)
                    if sample and os.path.exists(sample.stored_path):
                        vis = preprocessing.visual_similarity_score(
                            doc_image_path, sample.stored_path,
                        )
                finally:
                    db.close()
            except Exception:  # noqa: BLE001
                vis = 0.0

        score = combined_similarity(geom, vis, visual_weight)
        if score > best_score:
            best, best_score = tpl, score
            best_geom, best_vis = geom, vis

    return best, best_score, best_geom, best_vis


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


# ---------------------------------------------------------------------------
# Emparejado multi-ángulo
# ---------------------------------------------------------------------------

def best_template_multi_angle(
    doc_signature: dict[str, Any],
    templates: list,
    image_path: str | None = None,
    angles: list[int] | None = None,
) -> tuple[Any | None, float, int, dict[str, Any]]:
    """Como best_template pero prueba la firma en varios ángulos rotando la
    rejilla de densidad (aproximación rápida sin re-OCR).

    Si hay image_path, también usa similitud visual (ORB) y la combina.

    Returns:
        (plantilla, score_combinado, mejor_ángulo, {geom_score, vis_score, visual_weight})
    """
    if angles is None:
        angles = [0, 90, 180, 270]

    # Obtener la ruta de muestra de cada plantilla
    tpl_samples: dict[int, str] = {}
    if image_path:
        try:
            from . import models as _m2
            from .database import SessionLocal as _SL
            db = _SL()
            try:
                for tpl in templates:
                    if tpl.sample_document_id:
                        sample = db.query(_m2.Document).get(tpl.sample_document_id)
                        if sample and os.path.exists(sample.stored_path):
                            tpl_samples[tpl.id] = sample.stored_path
            finally:
                db.close()
        except Exception:  # noqa: BLE001
            pass

    best_tpl = None
    best_score = 0.0
    best_angle = 0
    best_info: dict[str, Any] = {}

    for angle in angles:
        # Rotar la rejilla de densidad según el ángulo
        rotated_sig = _rotate_signature(doc_signature, angle)
        tpl, score, geom, vis = best_template(
            rotated_sig, templates, image_path,
        )

        # Bonus si también coincide el aspect ratio invertido (útil para 90°/270°)
        if angle in (90, 270) and tpl is not None:
            sig_inv = dict(rotated_sig)
            if sig_inv.get("aspect_ratio"):
                sig_inv["aspect_ratio"] = 1.0 / max(sig_inv["aspect_ratio"], 1e-6)
            tpl2, s2, g2, v2 = best_template(sig_inv, templates, image_path)
            if s2 > score:
                tpl, score, geom, vis = tpl2, s2, g2, v2

        if score > best_score:
            best_score = score
            best_tpl = tpl
            best_angle = angle
            best_info = {"geom_score": geom, "vis_score": vis}

    return best_tpl, best_score, best_angle, best_info


def _rotate_signature(sig: dict[str, Any], angle: int) -> dict[str, Any]:
    """Rota la rejilla de densidad de la firma según el ángulo (0, 90, 180, 270)."""
    if angle == 0 or "density" not in sig:
        return sig

    density = np.asarray(sig["density"])
    grid_size = int(math.sqrt(len(density)))
    if grid_size * grid_size != len(density):
        return sig

    grid = density.reshape(grid_size, grid_size)
    k = (angle // 90) % 4
    rotated = np.rot90(grid, k=k)

    result = dict(sig)
    result["density"] = rotated.flatten().round(4).tolist()
    return result


# ---------------------------------------------------------------------------
# Emparejado por zonas visuales (template matching + ORB)
# ---------------------------------------------------------------------------

def match_by_zones(
    document_path: str,
    template,
    db=None,
) -> dict[str, Any]:
    """Detecta si la zona de la plantilla está presente en el documento usando
    template matching visual y ORB keypoints.

    Requiere que la plantilla tenga un documento de muestra (sample_document_id).

    Returns:
        {
            "found": bool,
            "method": "orb" | "template" | "none",
            "score": float,
            "region": {x,y,w,h} | None,   # zona detectada en el documento
            "offset": {x,y} | None,
        }
    """
    if not os.path.exists(document_path):
        return {"found": False, "method": "none", "score": 0.0, "region": None, "offset": None}

    # Cargar el documento de muestra de la plantilla
    sample_path = None
    if template.sample_document_id and db:
        sample_doc = db.get(db.get_tables() if hasattr(db, 'get_tables') else type(template), 0)
        # Intentar obtener el documento de muestra de la BD
        try:
            from . import models
            sample_doc = db.query(models.Document).get(template.sample_document_id)
            if sample_doc and os.path.exists(sample_doc.stored_path):
                sample_path = sample_doc.stored_path
        except Exception:  # noqa: BLE001
            pass

    if not sample_path:
        return {"found": False, "method": "none", "score": 0.0, "region": None, "offset": None}

    doc_img = Image.open(document_path).convert("RGB")
    tpl_img = Image.open(sample_path).convert("RGB")

    # 1) Intentar ORB (más robusto)
    orb_result = preprocessing.detect_zones_orb(doc_img, tpl_img)
    if orb_result["found"]:
        return {
            "found": True,
            "method": "orb",
            "score": round(min(1.0, orb_result["inliers"] / max(1, orb_result["matches"])), 4),
            "region": orb_result["region"],
            "offset": {"x": orb_result["region"]["x"], "y": orb_result["region"]["y"]},
            "corners": orb_result.get("corners"),
        }

    # 2) Fallback: template matching
    tm_result = preprocessing.detect_template_zones(doc_img, tpl_img)
    return {
        "found": tm_result["found"],
        "method": "template" if tm_result["found"] else "none",
        "score": tm_result["score"],
        "region": tm_result["region"] if tm_result["found"] else None,
        "offset": {"x": tm_result["offset_x"], "y": tm_result["offset_y"]} if tm_result["found"] else None,
    }
