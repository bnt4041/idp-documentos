"""Hitos / anclas de plantilla.

Una *ancla* es una zona de referencia de la plantilla (coords relativas al borde,
igual que un campo) con texto fijo esperado y/o un trozo de imagen recortado de la
muestra. Las anclas sirven para:

  - elegir la plantilla correcta (cuántas anclas se encuentran en el documento),
  - orientar el documento (en qué giro 0/90/180/270 casan mejor las anclas de texto),
  - alinear las regiones de los campos (transformación src->dst entre la posición
    esperada del ancla y la encontrada en el documento),
  - enderezar (rotación fina deducida de 2+ anclas).

Reutiliza utilidades ya existentes: matching.field_to_image, preprocessing.detect_best_zone
y ocr.run_ocr.
"""
from __future__ import annotations

import math
import os
import unicodedata
from difflib import SequenceMatcher
from typing import Any

import cv2
import numpy as np
from PIL import Image

from . import matching, ocr, preprocessing
from .config import settings


# ---------------------------------------------------------------------------
# Utilidades de texto y geometría
# ---------------------------------------------------------------------------

def normalize(s: str) -> str:
    """Minúsculas, sin acentos y espacios colapsados para comparar texto OCR."""
    s = unicodedata.normalize("NFD", s or "")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return " ".join(s.lower().split())


def _center(box: dict[str, float]) -> tuple[float, float]:
    return (box["x"] + box["w"] / 2, box["y"] + box["h"] / 2)


def _bbox(boxes: list[dict[str, float]]) -> dict[str, float]:
    x0 = min(b["x"] for b in boxes)
    y0 = min(b["y"] for b in boxes)
    x1 = max(b["x"] + b["w"] for b in boxes)
    y1 = max(b["y"] + b["h"] for b in boxes)
    return {"x": round(x0, 5), "y": round(y0, 5), "w": round(x1 - x0, 5), "h": round(y1 - y0, 5)}


# ---------------------------------------------------------------------------
# Localización por texto OCR (en TODO el documento, no solo la zona esperada)
# ---------------------------------------------------------------------------

def find_text_anchor(
    anchor_text: str, words: list[dict[str, Any]], threshold: float
) -> dict[str, Any]:
    """Busca la ventana de palabras OCR que mejor casa con `anchor_text`.

    Devuelve {found, score, region} (region = bbox de las palabras casadas).
    """
    target = normalize(anchor_text)
    if not target or not words:
        return {"found": False, "score": 0.0, "region": None}

    n_target = len(target.split())
    ordered = sorted(words, key=lambda w: (round(w["box"]["y"], 2), w["box"]["x"]))
    norm_words = [(normalize(w["text"]), w) for w in ordered]

    # Probar ventanas de tamaño n_target-1..n_target+1 para tolerar cortes/uniones del OCR
    sizes = {s for s in (n_target - 1, n_target, n_target + 1) if s >= 1}

    best_score = 0.0
    best_window: list[dict[str, Any]] | None = None
    for size in sizes:
        for i in range(0, len(norm_words) - size + 1):
            window = norm_words[i : i + size]
            text = " ".join(t for t, _ in window if t)
            if not text:
                continue
            score = SequenceMatcher(None, target, text).ratio()
            if score > best_score:
                best_score = score
                best_window = [w for _, w in window]

    found = best_score >= threshold and best_window is not None
    region = _bbox([w["box"] for w in best_window]) if best_window else None
    return {"found": found, "score": round(best_score, 4), "region": region}


# ---------------------------------------------------------------------------
# Localización por trozo de imagen (patch recortado de la muestra)
# ---------------------------------------------------------------------------

def find_image_anchor(
    doc_img: Image.Image,
    sample_img: Image.Image,
    anchor_rel: dict[str, float],
    tpl_border: dict[str, float] | None,
    threshold: float,
) -> dict[str, Any]:
    """Recorta el patch del ancla de la muestra y lo localiza en el documento.

    Reutiliza preprocessing.detect_best_zone (ORB multiescala + template matching
    multiángulo), tolerante a escala, giro y filtros B/N.
    """
    img_region = matching.field_to_image(anchor_rel, tpl_border or matching.FULL_BORDER)
    sw, sh = sample_img.size
    left = int(img_region["x"] * sw)
    top = int(img_region["y"] * sh)
    right = int((img_region["x"] + img_region["w"]) * sw)
    bottom = int((img_region["y"] + img_region["h"]) * sh)
    left = max(0, min(left, sw - 1))
    top = max(0, min(top, sh - 1))
    right = max(left + 1, min(right, sw))
    bottom = max(top + 1, min(bottom, sh))

    patch = sample_img.crop((left, top, right, bottom))
    if patch.width < 12 or patch.height < 12:
        return {"found": False, "score": 0.0, "region": None}

    try:
        res = preprocessing.detect_best_zone(doc_img, patch)
    except Exception:  # noqa: BLE001
        return {"found": False, "score": 0.0, "region": None}

    score = float(res.get("score", 0.0) or 0.0)
    found = bool(res.get("found")) and score >= threshold and res.get("region")
    return {
        "found": bool(found),
        "score": round(score, 4),
        "region": res.get("region") if found else None,
    }


# ---------------------------------------------------------------------------
# Localización combinada de todas las anclas de una plantilla
# ---------------------------------------------------------------------------

def locate_anchors(
    db,
    doc,
    tpl,
    doc_img: Image.Image | None = None,
) -> dict[str, Any]:
    """Localiza las anclas de `tpl` en `doc`.

    Devuelve {score, n_found, located:[{name, found, text_score, image_score,
    region, src_center, dst_center, weight}]}. `src_center` está en coords
    relativas al borde; `dst_center` en coords de imagen del documento (0..1).
    """
    anchors = list(getattr(tpl, "anchors", []) or [])
    if not anchors:
        return {"score": 0.0, "n_found": 0, "located": []}

    need_image = any(a.use_image for a in anchors)
    sample_img = None
    if need_image and getattr(tpl, "sample_document_id", None):
        from . import models

        sample = db.get(models.Document, tpl.sample_document_id)
        if sample and os.path.exists(sample.stored_path):
            sample_img = Image.open(sample.stored_path).convert("RGB")
    if need_image and doc_img is None and doc and os.path.exists(doc.stored_path):
        doc_img = Image.open(doc.stored_path).convert("RGB")

    words = doc.ocr_words or []
    located: list[dict[str, Any]] = []
    total_w = 0.0
    found_w = 0.0

    for a in anchors:
        rel = {"x": a.x, "y": a.y, "w": a.w, "h": a.h}
        src_center = _center(rel)
        text_score = 0.0
        image_score = 0.0
        dst_center: tuple[float, float] | None = None
        region = None

        if a.use_text and a.anchor_text:
            tr = find_text_anchor(a.anchor_text, words, settings.anchor_text_threshold)
            text_score = tr["score"]
            if tr["found"]:
                region = tr["region"]
                dst_center = _center(region)

        if a.use_image and sample_img is not None and doc_img is not None:
            ir = find_image_anchor(
                doc_img, sample_img, rel, tpl.border, settings.anchor_image_threshold
            )
            image_score = ir["score"]
            if ir["found"] and dst_center is None:
                region = ir["region"]
                dst_center = _center(region)

        found = dst_center is not None
        w = max(0.0, float(a.weight or 1.0))
        total_w += w
        if found:
            found_w += w

        located.append(
            {
                "name": a.name,
                "found": found,
                "text_score": round(text_score, 4),
                "image_score": round(image_score, 4),
                "region": region,
                "src_center": src_center,
                "dst_center": dst_center,
                "weight": w,
            }
        )

    score = round(found_w / total_w, 4) if total_w else 0.0
    n_found = int(sum(1 for l in located if l["found"]))
    return {"score": score, "n_found": n_found, "located": located}


def public_anchors(located: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Versión serializable (sin tuplas internas) para el MatchResult/UI."""
    return [
        {
            "name": l["name"],
            "found": l["found"],
            "text_score": l["text_score"],
            "image_score": l["image_score"],
            "region": l["region"],
        }
        for l in located
    ]


# ---------------------------------------------------------------------------
# Transformación de alineación (src relativo-al-borde -> dst imagen del doc)
# ---------------------------------------------------------------------------

def estimate_transform(located: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Estima la transformación que mapea coords relativas al borde -> imagen del doc.

    - 0 anclas encontradas -> None (usar el mapeo de borde habitual).
    - 1 -> solo traslación.
    - >=2 -> similitud (escala + rotación + traslación) con estimateAffinePartial2D.
    """
    pairs = [
        (l["src_center"], l["dst_center"])
        for l in located
        if l["found"] and l["dst_center"]
    ]
    if not pairs:
        return None

    if len(pairs) == 1:
        (sx, sy), (dx, dy) = pairs[0]
        return {"matrix": None, "scale": 1.0, "angle": 0.0, "dx": dx - sx, "dy": dy - sy}

    src = np.float32([p[0] for p in pairs])
    dst = np.float32([p[1] for p in pairs])
    M, _inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)
    if M is None:
        d = dst.mean(axis=0) - src.mean(axis=0)
        return {"matrix": None, "scale": 1.0, "angle": 0.0, "dx": float(d[0]), "dy": float(d[1])}

    scale = float(math.hypot(M[0, 0], M[1, 0]))
    angle = float(math.degrees(math.atan2(M[1, 0], M[0, 0])))
    return {
        "matrix": M.tolist(),
        "scale": round(scale, 5),
        "angle": round(angle, 3),
        "dx": float(M[0, 2]),
        "dy": float(M[1, 2]),
    }


def apply_transform_to_region(
    region_rel: dict[str, float],
    transform: dict[str, Any] | None,
    fallback_border: dict[str, float] | None,
) -> dict[str, float]:
    """Mapea una región de campo (relativa al borde) a coords de imagen del doc.

    Si no hay transform de anclas, cae al mapeo por borde (matching.field_to_image).
    """
    if not transform:
        return matching.field_to_image(region_rel, fallback_border or matching.FULL_BORDER)

    M = transform.get("matrix")

    def map_pt(x: float, y: float) -> tuple[float, float]:
        if M is not None:
            nx = M[0][0] * x + M[0][1] * y + M[0][2]
            ny = M[1][0] * x + M[1][1] * y + M[1][2]
            return nx, ny
        return x + transform["dx"], y + transform["dy"]

    x0, y0 = region_rel["x"], region_rel["y"]
    x1, y1 = x0 + region_rel["w"], y0 + region_rel["h"]
    pts = [map_pt(x0, y0), map_pt(x1, y0), map_pt(x1, y1), map_pt(x0, y1)]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    nx0, ny0, nx1, ny1 = min(xs), min(ys), max(xs), max(ys)
    return {
        "x": round(nx0, 5),
        "y": round(ny0, 5),
        "w": round(nx1 - nx0, 5),
        "h": round(ny1 - ny0, 5),
    }


# ---------------------------------------------------------------------------
# Orientación (0/90/180/270) según anclas de texto
# ---------------------------------------------------------------------------

def estimate_orientation(
    image: Image.Image, tpl, angles: list[int] | None = None
) -> tuple[int, float]:
    """Devuelve (mejor_ángulo, score) probando las anclas de texto en cada giro.

    Solo aplica si la plantilla tiene anclas de texto; re-OCR por ángulo, así que
    conviene llamarlo solo cuando la orientación es dudosa.
    """
    text_anchors = [
        a for a in (getattr(tpl, "anchors", []) or []) if a.use_text and a.anchor_text
    ]
    if not text_anchors:
        return 0, 0.0
    if angles is None:
        angles = [0, 90, 180, 270]

    best_angle, best_score = 0, -1.0
    for angle in angles:
        rot = image if angle == 0 else image.rotate(-angle, expand=True)
        try:
            words = ocr.run_ocr(rot)
        except Exception:  # noqa: BLE001
            continue
        total = 0.0
        got = 0.0
        for a in text_anchors:
            w = max(0.0, float(a.weight or 1.0))
            total += w
            if find_text_anchor(a.anchor_text, words, settings.anchor_text_threshold)["found"]:
                got += w
        s = got / total if total else 0.0
        if s > best_score:
            best_score, best_angle = s, angle

    return best_angle, round(best_score, 4)
