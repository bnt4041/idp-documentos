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

    `src_center` se expresa en PÍXELES de la imagen de muestra de la plantilla y
    `dst_center` en PÍXELES del documento: así la transformación que los une es la
    similitud/afín física entre muestra y documento, independiente del borde
    detectado del documento (que suele ser poco fiable).

    Devuelve {score, n_found, tpl_w, tpl_h, located:[...]}.
    """
    anchors = list(getattr(tpl, "anchors", []) or [])
    if not anchors:
        return {"score": 0.0, "n_found": 0, "tpl_w": None, "tpl_h": None, "located": []}

    need_image = any(a.use_image for a in anchors)
    sample_img = None
    tpl_w = tpl_h = None
    if getattr(tpl, "sample_document_id", None):
        from . import models

        sample = db.get(models.Document, tpl.sample_document_id)
        if sample:
            tpl_w = float(sample.width or 0) or None
            tpl_h = float(sample.height or 0) or None
            if need_image and os.path.exists(sample.stored_path):
                sample_img = Image.open(sample.stored_path).convert("RGB")
                if not tpl_w or not tpl_h:
                    tpl_w, tpl_h = float(sample_img.width), float(sample_img.height)
    if need_image and doc_img is None and doc and os.path.exists(doc.stored_path):
        doc_img = Image.open(doc.stored_path).convert("RGB")

    words = doc.ocr_words or []
    # Centros en PÍXELES: src en la imagen de muestra de la plantilla, dst en el
    # documento. La transformación src->dst es entonces la relación física real
    # entre ambas (escala+giro+traslación), sin depender del borde del documento.
    W = float(doc.width or 0) or 1.0
    H = float(doc.height or 0) or 1.0
    border = getattr(doc, "border", None) or matching.FULL_BORDER
    tpl_border = getattr(tpl, "border", None) or matching.FULL_BORDER
    has_tpl_px = bool(tpl_w and tpl_h)

    located: list[dict[str, Any]] = []
    total_w = 0.0
    found_w = 0.0

    for a in anchors:
        rel = {"x": a.x, "y": a.y, "w": a.w, "h": a.h}
        # src: posición del ancla en PÍXELES de la muestra de la plantilla
        src_center = None
        if has_tpl_px:
            tpl_region = matching.field_to_image(rel, tpl_border)
            src_center = (
                (tpl_region["x"] + tpl_region["w"] / 2) * tpl_w,
                (tpl_region["y"] + tpl_region["h"] / 2) * tpl_h,
            )
        # expected: solo para la UI (dónde caería según el borde del documento)
        expected = matching.field_to_image(rel, border)
        text_score = 0.0
        image_score = 0.0
        dst_center: tuple[float, float] | None = None
        region = None

        if a.use_text and a.anchor_text:
            tr = find_text_anchor(a.anchor_text, words, settings.anchor_text_threshold)
            text_score = tr["score"]
            if tr["found"]:
                region = tr["region"]
                cx, cy = _center(region)
                dst_center = (cx * W, cy * H)

        if a.use_image and sample_img is not None and doc_img is not None:
            ir = find_image_anchor(
                doc_img, sample_img, rel, tpl.border, settings.anchor_image_threshold
            )
            image_score = ir["score"]
            if ir["found"] and dst_center is None:
                region = ir["region"]
                cx, cy = _center(region)
                dst_center = (cx * W, cy * H)

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
                "region": region,            # encontrada (normalizada, para la UI)
                "expected_region": expected,  # esperada (normalizada, para la UI)
                "src_center": src_center,     # píxeles
                "dst_center": dst_center,     # píxeles
                "weight": w,
            }
        )

    score = round(found_w / total_w, 4) if total_w else 0.0
    n_found = int(sum(1 for l in located if l["found"]))
    return {
        "score": score,
        "n_found": n_found,
        "tpl_w": tpl_w,
        "tpl_h": tpl_h,
        "located": located,
    }


def _clean_region(region: dict | None) -> dict | None:
    """Convierte los valores de una región a float de Python (ORB devuelve numpy
    float32, que Pydantic no sabe serializar)."""
    if not region:
        return None
    return {k: float(v) for k, v in region.items()}


def public_anchors(located: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Versión serializable (sin tuplas internas ni numpy) para el MatchResult/UI."""
    return [
        {
            "name": l["name"],
            "found": bool(l["found"]),
            "text_score": float(l["text_score"]),
            "image_score": float(l["image_score"]),
            "region": _clean_region(l["region"]),
            "expected_region": _clean_region(l["expected_region"]),
        }
        for l in located
    ]


# ---------------------------------------------------------------------------
# Transformación de alineación (src relativo-al-borde -> dst imagen del doc)
# ---------------------------------------------------------------------------

def _reproj_error(M, src, dst) -> float:
    """Error de reproyección RMS relativo a la extensión de los puntos dst."""
    proj = (M @ np.hstack([src, np.ones((len(src), 1), np.float32)]).T).T
    rms = float(np.sqrt(np.mean(np.sum((proj - dst) ** 2, axis=1))))
    extent = float(np.linalg.norm(dst.max(axis=0) - dst.min(axis=0))) or 1.0
    return rms / extent


def _affine_params(M) -> dict[str, float] | None:
    """Descompone una afín 2x3 en escala X/Y, cizalla y giro."""
    a, b = float(M[0, 0]), float(M[0, 1])
    c, d = float(M[1, 0]), float(M[1, 1])
    sx = math.hypot(a, c)
    if sx < 1e-9:
        return None
    det = a * d - b * c
    sy = det / sx
    shear = (a * b + c * d) / (sx * sx)
    angle = math.degrees(math.atan2(c, a))
    return {"sx": sx, "sy": sy, "shear": shear, "angle": angle}


def estimate_transform(located: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Estima la transformación PÍXELES muestra-plantilla -> PÍXELES documento.

    Estrategia para distinguir "mismo formulario" de "formulario distinto":
      - 1 ancla  -> solo traslación.
      - >=3      -> afín completa (admite recortes/proporciones distintas del MISMO
                    formulario), pero solo se acepta si es geométricamente sensata
                    (escala/anisotropía/cizalla acotadas). Si la afín está muy
                    distorsionada = formulario distinto -> no se usa.
      - si la afín no vale -> similitud rígida (escala uniforme + giro), validada
        por error de reproyección. Si tampoco encaja -> None (no se distorsiona).
    """
    pairs = [
        (l["src_center"], l["dst_center"])
        for l in located
        if l["found"] and l["dst_center"] and l["src_center"]
    ]
    if not pairs:
        return None

    if len(pairs) == 1:
        (sx, sy), (dx, dy) = pairs[0]
        return {"matrix": None, "scale": 1.0, "angle": 0.0, "dx": dx - sx, "dy": dy - sy,
                "residual": 0.0, "n": 1, "kind": "translation"}

    src = np.float32([p[0] for p in pairs])
    dst = np.float32([p[1] for p in pairs])
    n = len(pairs)
    cv2.setRNGSeed(0)  # RANSAC determinista: mismos puntos -> misma transformación

    # 1) Afín completa (>=3 anclas): tolera diferencias de recorte/aspecto del mismo
    #    formulario, pero la rechazamos si está muy distorsionada (cizalla = otro form).
    if n >= 3:
        Ma, _ = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC)
        if Ma is not None:
            p = _affine_params(Ma)
            if p:
                asx, asy = abs(p["sx"]), abs(p["sy"])
                aniso = max(asx, asy) / max(1e-9, min(asx, asy))
                resid = _reproj_error(Ma, src, dst)
                # Se permite mucha anisotropía (mismo formulario recortado distinto),
                # pero NO reflexión (sy<0) ni cizalla alta (eso = otro formulario).
                sane = (
                    p["sy"] > 0
                    and 0.1 <= asx <= 6.0
                    and 0.1 <= asy <= 6.0
                    and aniso <= settings.anchor_max_anisotropy
                    and abs(p["shear"]) <= settings.anchor_max_shear
                    and (n < 4 or resid <= settings.anchor_fit_max_error)
                )
                if sane:
                    return {
                        "matrix": Ma.tolist(),
                        "scale": round((asx + asy) / 2, 5),
                        "angle": round(p["angle"], 3),
                        "dx": float(Ma[0, 2]),
                        "dy": float(Ma[1, 2]),
                        "residual": round(resid, 4),
                        "n": n,
                        "kind": "affine",
                    }

    # 2) Similitud rígida (2+ anclas), validada por error y escala
    Ms, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)
    if Ms is None:
        return None
    resid = _reproj_error(Ms, src, dst)
    scale = float(math.hypot(Ms[0, 0], Ms[1, 0]))
    angle = float(math.degrees(math.atan2(Ms[1, 0], Ms[0, 0])))
    if not (0.25 <= scale <= 4.0):
        return None
    if n >= 3 and resid > settings.anchor_fit_max_error:
        return None
    return {
        "matrix": Ms.tolist(),
        "scale": round(scale, 5),
        "angle": round(angle, 3),
        "dx": float(Ms[0, 2]),
        "dy": float(Ms[1, 2]),
        "residual": round(resid, 4),
        "n": n,
        "kind": "similarity",
    }


def apply_transform_to_region(
    region_rel: dict[str, float],
    transform: dict[str, Any] | None,
    tpl_border: dict[str, float] | None,
    tpl_w: float | None,
    tpl_h: float | None,
    doc_w: float | None,
    doc_h: float | None,
) -> dict[str, float]:
    """Mapea una región de campo (relativa al borde de la plantilla) al documento.

    Convierte el campo a píxeles de la muestra de la plantilla, aplica la
    transformación de anclas (muestra->documento) y normaliza a coords del doc.
    """
    tpl_region = matching.field_to_image(region_rel, tpl_border or matching.FULL_BORDER)
    if not transform:
        # Sin anclas no hay corrección posible aquí; usar el mapeo normalizado tal cual
        return tpl_region

    TW = float(tpl_w or 0) or 1.0
    TH = float(tpl_h or 0) or 1.0
    DW = float(doc_w or 0) or 1.0
    DH = float(doc_h or 0) or 1.0
    M = transform.get("matrix")

    def map_pt(xn: float, yn: float) -> tuple[float, float]:
        # normalizado(plantilla) -> px(plantilla) -> px(doc) -> normalizado(doc)
        xp, yp = xn * TW, yn * TH
        if M is not None:
            nx = M[0][0] * xp + M[0][1] * yp + M[0][2]
            ny = M[1][0] * xp + M[1][1] * yp + M[1][2]
        else:
            nx = xp + transform["dx"]
            ny = yp + transform["dy"]
        return nx / DW, ny / DH

    x0, y0 = tpl_region["x"], tpl_region["y"]
    x1, y1 = x0 + tpl_region["w"], y0 + tpl_region["h"]
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
