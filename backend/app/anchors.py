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

    Usa ORB directo (invariante a rotación y escala): la región devuelta queda
    SIEMPRE en el marco ORIGINAL del documento, de modo que la disposición de las
    anclas codifica correctamente el giro del documento (para el auto-enderezado).
    Si ORB no encuentra suficientes puntos, recurre a template matching multiescala
    (sin rotación). NO se usa detect_best_zone porque rota el documento por dentro y
    devolvería coordenadas en un marco rotado.
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

    # Tamaño esperado del parche en proporción del documento (referencia para
    # validar que la región localizada es coherente y no una homografía degenerada).
    exp_w = img_region["w"]
    exp_h = img_region["h"]

    def _accept(reg: dict | None) -> dict | None:
        """Valida la región ORB/TM y la normaliza al TAMAÑO ESPERADO del parche,
        centrada en la posición encontrada.

        El bounding box que devuelve ORB es inestable entre documentos distintos
        (homografías que se estiran), pero su CENTRO sí es fiable y es lo único que
        usa la transformación de alineación. Reconstruir la caja con el tamaño
        esperado evita zonas deformadas en la UI sin afectar al centro.
        """
        if not reg:
            return None
        rw, rh = reg.get("w", 0), reg.get("h", 0)
        if rw <= 0 or rh <= 0:
            return None
        # Descartar homografías claramente degeneradas (caja gigantesca)
        if rw * rh > 0.12:
            return None
        if exp_w > 0 and rw > exp_w * 4.0 + 0.05:
            return None
        if exp_h > 0 and rh > exp_h * 4.0 + 0.05:
            return None
        cx = reg["x"] + rw / 2
        cy = reg["y"] + rh / 2
        w = exp_w or rw
        h = exp_h or rh
        return {
            "x": round(max(0.0, cx - w / 2), 5),
            "y": round(max(0.0, cy - h / 2), 5),
            "w": round(w, 5),
            "h": round(h, 5),
        }

    # 1) ORB en el marco original (invariante a rotación/escala)
    try:
        orb = preprocessing.detect_zones_orb(doc_img, patch, min_matches=8)
    except Exception:  # noqa: BLE001
        orb = {"found": False}
    if orb.get("found"):
        norm = _accept(orb.get("region"))
        if norm:
            inliers = orb.get("inliers", 0)
            matches = orb.get("matches", 1)
            score = min(1.0, (inliers / max(1, matches)) / 0.5)
            if score >= threshold:
                return {"found": True, "score": round(float(score), 4), "region": norm}

    # 2) Fallback: template matching multiescala (sin rotación)
    try:
        tm = preprocessing.detect_template_zones_multiscale(doc_img, patch, threshold=threshold)
    except Exception:  # noqa: BLE001
        tm = {"found": False}
    if tm.get("found"):
        norm = _accept(tm.get("region"))
        if norm:
            return {"found": True, "score": round(float(tm.get("score", 0.0)), 4), "region": norm}

    return {"found": False, "score": 0.0, "region": None}


def _sample_image(db, tpl) -> Image.Image | None:
    """Imagen de muestra de la plantilla (PIL RGB) o None si no existe."""
    if not getattr(tpl, "sample_document_id", None):
        return None
    from . import models

    sample = db.get(models.Document, tpl.sample_document_id)
    if sample and os.path.exists(sample.stored_path):
        return Image.open(sample.stored_path).convert("RGB")
    return None


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
# Localización robusta (multi-ángulo × multi-filtro) + rectificación a plantilla
# ---------------------------------------------------------------------------

def _filtered(image: Image.Image, name: str) -> Image.Image:
    """Aplica un filtro por nombre reutilizando preprocessing. Devuelve PIL."""
    fn = preprocessing.FILTERS.get(name)
    if not fn:
        return image
    try:
        out = fn(image)
        return out if out.mode == "RGB" else out.convert("RGB")
    except Exception:  # noqa: BLE001
        return image


def _locate_in_image(
    image: Image.Image,
    words: list[dict[str, Any]],
    anchors_list: list,
    sample_img: Image.Image | None,
    tpl,
    tpl_w: float,
    tpl_h: float,
) -> dict[str, Any]:
    """Localiza todas las anclas en una imagen YA orientada.

    Texto: sobre `words` (OCR de esta imagen). Imagen: ORB del patch, probando en
    cascada los filtros de settings.anchor_filters hasta encontrarla.
    Devuelve {located:[{name, found, src_center, dst_center(px de ESTA imagen), ...}],
    score, n_found}. src_center en píxeles de la MUESTRA.
    """
    W, H = float(image.width), float(image.height)
    tpl_border = getattr(tpl, "border", None) or matching.FULL_BORDER
    # Variantes filtradas de la imagen del documento (perezosas)
    doc_variants: dict[str, Image.Image] = {}

    def doc_variant(name: str) -> Image.Image:
        if name not in doc_variants:
            doc_variants[name] = _filtered(image, name) if name != "grayscale" else image
        return doc_variants[name]

    located: list[dict[str, Any]] = []
    total_w = found_w = 0.0
    for a in anchors_list:
        rel = {"x": a.x, "y": a.y, "w": a.w, "h": a.h}
        src_center = None
        if tpl_w and tpl_h:
            tr = matching.field_to_image(rel, tpl_border)
            src_center = (
                (tr["x"] + tr["w"] / 2) * tpl_w,
                (tr["y"] + tr["h"] / 2) * tpl_h,
            )
        text_score = image_score = 0.0
        dst_center = None
        region = None

        if a.use_text and a.anchor_text:
            res = find_text_anchor(a.anchor_text, words, settings.anchor_text_threshold)
            text_score = res["score"]
            if res["found"]:
                region = res["region"]
                cx, cy = _center(region)
                dst_center = (cx * W, cy * H)

        if dst_center is None and a.use_image and sample_img is not None:
            for flt in settings.anchor_filters:
                dv = doc_variant(flt)
                sv = sample_img if flt == "grayscale" else _filtered(sample_img, flt)
                res = find_image_anchor(
                    dv, sv, rel, tpl.border, settings.anchor_image_threshold
                )
                if res["score"] > image_score:
                    image_score = res["score"]
                if res["found"]:
                    region = res["region"]
                    cx, cy = _center(region)
                    dst_center = (cx * W, cy * H)
                    break

        found = dst_center is not None
        w = max(0.0, float(a.weight or 1.0))
        total_w += w
        if found:
            found_w += w
        located.append(
            {
                "name": a.name,
                "required": bool(getattr(a, "required", False)),
                "found": found,
                "text_score": round(text_score, 4),
                "image_score": round(image_score, 4),
                "region": region,
                "expected_region": matching.field_to_image(rel, tpl_border),
                "src_center": src_center,
                "dst_center": dst_center,
                "weight": w,
            }
        )

    return {
        "located": located,
        "score": round(found_w / total_w, 4) if total_w else 0.0,
        "n_found": int(sum(1 for l in located if l["found"])),
    }


def locate_anchors_robust(db, doc, tpl, image: Image.Image) -> dict[str, Any]:
    """Localiza las anclas probando las 4 orientaciones (0/90/180/270) y multi-filtro.

    Devuelve {angle, image (PIL en el mejor ángulo), located, score, n_found, tpl_w,
    tpl_h, sample_size}. El `angle` es el giro que se aplicó a la imagen original para
    llegar a la orientación ganadora (para poder persistir esa rotación).
    """
    anchors_list = list(getattr(tpl, "anchors", []) or [])
    sample_img = _sample_image(db, tpl)
    tpl_w = tpl_h = None
    if getattr(tpl, "sample_document_id", None):
        from . import models

        s = db.get(models.Document, tpl.sample_document_id)
        if s:
            tpl_w = float(s.width or 0) or (float(sample_img.width) if sample_img else None)
            tpl_h = float(s.height or 0) or (float(sample_img.height) if sample_img else None)

    need_text = any(a.use_text and a.anchor_text for a in anchors_list)

    # Fijar la orientación con la señal robusta de página-completa (aspect ratio +
    # ORB global + OSD), no por parches pequeños (que casan a cualquier giro).
    angles = [0, 90, 180, 270]
    if sample_img is not None:
        try:
            pref, _inl = preprocessing.best_rotation_by_orb(image, sample_img)
            pref %= 360
            # Probar primero el ángulo preferido y su opuesto (desambiguación 180)
            angles = [pref, (pref + 180) % 360, (pref + 90) % 360, (pref + 270) % 360]
        except Exception:  # noqa: BLE001
            pass

    sample_size = (int(tpl_w), int(tpl_h)) if (tpl_w and tpl_h) else None
    best = None
    best_rank = (-1, -1.0, -1.0)  # (n_found, rectificable, score)
    for angle in angles:
        rot = image if angle == 0 else image.rotate(-angle, expand=True)
        words = []
        if need_text:
            try:
                words = ocr.run_ocr(rot)
            except Exception:  # noqa: BLE001
                words = []
        res = _locate_in_image(rot, words, anchors_list, sample_img, tpl, tpl_w or 0, tpl_h or 0)
        res["angle"] = angle
        res["image"] = rot
        res["words"] = words
        _r, rect_ok = rectify_to_template(res["located"], rot, sample_size, sample_img)
        rank = (res["n_found"], 1.0 if rect_ok else 0.0, res["score"])
        if rank > best_rank:
            best_rank = rank
            best = res
        if rect_ok and res["n_found"] == len(anchors_list) and res["n_found"] > 0:
            break

    if best is None:
        best = {"angle": 0, "image": image, "words": [], "located": [], "score": 0.0, "n_found": 0}

    best["tpl_w"] = tpl_w
    best["tpl_h"] = tpl_h
    best["sample_size"] = (int(tpl_w), int(tpl_h)) if (tpl_w and tpl_h) else None
    best["sample_img"] = sample_img
    return best


def required_ok(located: list[dict[str, Any]]) -> bool:
    """True si todas las anclas marcadas como obligatorias se han localizado."""
    req = [l for l in located if l.get("required")]
    return all(l["found"] for l in req) if req else True


def rectify_with_homography(
    image: Image.Image,
    sample_img: Image.Image,
    sample_size: tuple[int, int] | None,
    info: dict | None = None,
) -> tuple[Image.Image | None, bool]:
    """Rectifica la PÁGINA COMPLETA del documento al marco de la muestra mediante una
    transformación RÍGIDA (rotación + escala uniforme + traslación) estimada a partir
    de los keypoints ORB de toda la página.

    Se usa una SIMILITUD (no homografía completa): un permiso es una hoja plana, así
    que no hay perspectiva real. La homografía de 8 dof introduce cizalla/perspectiva
    falsa entre documentos distintos (líneas de tabla en diagonal). La similitud no
    puede deformar: solo gira, escala y traslada.

    Si se pasa `info` (dict), se rellena con la traza: angle, scale, inliers, ecc.
    """
    if not sample_size:
        return None, False

    doc_arr = preprocessing.pil_to_cv2(image)
    tpl_arr = preprocessing.pil_to_cv2(sample_img)
    doc_gray = cv2.cvtColor(doc_arr, cv2.COLOR_BGR2GRAY)
    tpl_gray = cv2.cvtColor(tpl_arr, cv2.COLOR_BGR2GRAY)

    # ORB casa mucho mejor cuando documento y muestra tienen tamaño similar. Si
    # difieren mucho (p.ej. muestra DNI 532px vs escaneo 2339px), pre-escalamos el
    # documento al tamaño de la muestra y luego componemos la escala en la matriz.
    sw, sh = sample_img.size
    dw, dh = image.size
    pre = 1.0
    long_doc = max(dw, dh)
    long_tpl = max(sw, sh)
    if long_tpl and abs(long_doc / long_tpl - 1.0) > 0.5:
        pre = long_tpl / long_doc
        doc_small = cv2.resize(doc_gray, (max(1, int(dw * pre)), max(1, int(dh * pre))))
    else:
        doc_small = doc_gray

    try:
        orb = cv2.ORB_create(nfeatures=3000)
        kp1, des1 = orb.detectAndCompute(doc_small, None)  # documento (pre-escalado)
        kp2, des2 = orb.detectAndCompute(tpl_gray, None)   # muestra
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return None, False
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(bf.match(des1, des2), key=lambda m: m.distance)
        good = matches[:200]
        if len(good) < 12:
            return None, False
        # Puntos del documento en coords ORIGINALES (deshacer el pre-escalado)
        src = np.float32([[kp1[m.queryIdx].pt[0] / pre, kp1[m.queryIdx].pt[1] / pre] for m in good])
        dst = np.float32([kp2[m.trainIdx].pt for m in good])  # muestra
        # Similitud documento->muestra (rotación+escala uniforme+traslación)
        M, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    except Exception:  # noqa: BLE001
        return None, False
    if M is None:
        return None, False

    # Cordura de la escala (evita transformaciones degeneradas)
    scale = float(math.hypot(M[0, 0], M[1, 0]))
    if not (0.2 <= scale <= 5.0):
        return None, False
    inliers = int(mask.sum()) if mask is not None else 0
    if inliers < 10:
        return None, False

    # Traza del enderezamiento (ángulo en sentido horario del documento original)
    rot_angle = float(math.degrees(math.atan2(M[1, 0], M[0, 0])))
    if info is not None:
        info["scale"] = round(scale, 4)
        info["rotation"] = round(rot_angle, 2)
        info["inliers"] = inliers

    warped = cv2.warpAffine(
        doc_arr, M, sample_size,
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )

    # Refinamiento por ECC (Enhanced Correlation Coefficient): alinea por intensidad
    # el documento ya warpeado con la muestra, corrigiendo el pequeño residual de
    # rotación+traslación+escala que deja la similitud ORB global. Dos permisos del
    # mismo modelo comparten la rejilla de la tabla, así que ECC converge bien.
    try:
        wg = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        sg = tpl_gray.astype(np.float32) / 255.0
        if wg.shape == sg.shape:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5,
            )
            # MOTION_EUCLIDEAN = rotación + traslación (sin cizalla/escala -> no deforma)
            cc, warp_matrix = cv2.findTransformECC(
                sg, wg, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria, None, 5,
            )
            # Solo aplicar si la corrección es pequeña (refina, no reubica)
            dx, dy = float(warp_matrix[0, 2]), float(warp_matrix[1, 2])
            ang = abs(math.degrees(math.atan2(warp_matrix[1, 0], warp_matrix[0, 0])))
            max_shift = 0.12 * max(wg.shape)
            if cc > 0.3 and abs(dx) <= max_shift and abs(dy) <= max_shift and ang <= 5.0:
                warped = cv2.warpAffine(
                    warped, warp_matrix, sample_size,
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
                )
    except cv2.error:
        pass  # ECC no convergió: nos quedamos con la similitud ORB
    except Exception:  # noqa: BLE001
        pass

    return preprocessing.cv2_to_pil(warped), True


def _refine_translation_by_text_anchors(
    warped: Image.Image, tpl, sample_size: tuple[int, int]
) -> Image.Image:
    """Corrige el desplazamiento residual usando las anclas de TEXTO.

    El texto de cabecera (p.ej. "OBSERVACIONES:") es idéntico en todos los
    documentos del mismo tipo, así que su posición es una referencia fiable —al
    contrario que ORB/ECC, que entre modelos distintos dejan ~60px de error. Se
    mide el desfase medio (esperado vs encontrado) y se traslada la imagen.
    """
    text_anchors = [
        a for a in (getattr(tpl, "anchors", []) or []) if a.use_text and a.anchor_text
    ]
    if not text_anchors:
        return warped
    tpl_border = getattr(tpl, "border", None) or matching.FULL_BORDER
    W, H = sample_size
    try:
        words = ocr.run_ocr(warped)
    except Exception:  # noqa: BLE001
        return warped
    dxs, dys = [], []
    for a in text_anchors:
        res = find_text_anchor(a.anchor_text, words, settings.anchor_text_threshold)
        if not res["found"]:
            continue
        exp = matching.field_to_image({"x": a.x, "y": a.y, "w": a.w, "h": a.h}, tpl_border)
        ex_cx = (exp["x"] + exp["w"] / 2) * W
        ex_cy = (exp["y"] + exp["h"] / 2) * H
        fc = res["region"]
        fo_cx = (fc["x"] + fc["w"] / 2) * W
        fo_cy = (fc["y"] + fc["h"] / 2) * H
        dxs.append(ex_cx - fo_cx)
        dys.append(ex_cy - fo_cy)
    if not dxs:
        return warped
    dx = float(np.median(dxs))
    dy = float(np.median(dys))
    max_shift = 0.15 * max(W, H)
    if abs(dx) > max_shift or abs(dy) > max_shift:
        return warped  # desfase improbable: no arriesgar
    if abs(dx) < 1 and abs(dy) < 1:
        return warped
    arr = preprocessing.pil_to_cv2(warped)
    T = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(
        arr, T, sample_size,
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )
    return preprocessing.cv2_to_pil(shifted)


def rectify_to_template(
    located: list[dict[str, Any]],
    image: Image.Image,
    sample_size: tuple[int, int] | None,
    sample_img: Image.Image | None = None,
    tpl=None,
    info: dict | None = None,
) -> tuple[Image.Image | None, bool]:
    """Rectifica `image` (doc->muestra). Prioriza la similitud ORB de página
    completa (estable); si no, cae a la afín por centros de anclas. Finalmente
    refina la traslación con las anclas de texto (referencia fiable entre modelos).

    Devuelve (imagen_rectificada, ok). ok=False si la geometría no es plausible.
    `info` (dict) se rellena con la traza: prerotate, rotation, scale, inliers, method.
    """
    if not sample_size:
        return None, False

    # 1) Similitud rígida ORB de página completa (preferida). ORB empareja el
    #    contenido, así que la transformación ya incluye la rotación necesaria
    #    (incluido 180°): no hace falta pre-orientar la imagen. Probamos la imagen
    #    tal cual y, si el aspecto no casa con la muestra, su versión a 90°.
    if sample_img is not None:
        candidates = [(0, image)]
        sw, sh = sample_size
        tpl_ratio = (sw / sh) if sh else 1.0
        img_ratio = (image.width / image.height) if image.height else 1.0
        # Si la proporción difiere mucho, el documento está girado 90/270: probar rotado
        if abs(img_ratio - tpl_ratio) > 0.3 * tpl_ratio:
            candidates = [(90, image.rotate(-90, expand=True)),
                          (270, image.rotate(-270, expand=True)),
                          (0, image)]
        for prerot, cand in candidates:
            warped, ok = rectify_with_homography(cand, sample_img, sample_size, info)
            if ok:
                if info is not None:
                    info["prerotate"] = prerot
                    info["method"] = "orb-similarity"
                if tpl is not None:
                    warped = _refine_translation_by_text_anchors(warped, tpl, sample_size)
                return warped, True

    # 2) Fallback: afín por centros de anclas (>=3) / similitud (2)
    pairs = [
        (l["src_center"], l["dst_center"])
        for l in located
        if l["found"] and l["src_center"] and l["dst_center"]
    ]
    if len(pairs) < 2:
        return None, False
    src = np.float32([p[0] for p in pairs])  # muestra
    dst = np.float32([p[1] for p in pairs])  # documento
    if len(pairs) >= 3:
        M, _ = cv2.estimateAffine2D(dst, src, method=cv2.RANSAC)
    else:
        M, _ = cv2.estimateAffinePartial2D(dst, src, method=cv2.RANSAC)
    if M is None:
        return None, False
    p = _affine_params(M)
    if not p:
        return None, False
    asx, asy = abs(p["sx"]), abs(p["sy"])
    aniso = max(asx, asy) / max(1e-9, min(asx, asy))
    if not (0.2 <= asx <= 5.0 and 0.2 <= asy <= 5.0):
        return None, False
    if p["sy"] <= 0 or aniso > settings.anchor_max_anisotropy or abs(p["shear"]) > settings.anchor_max_shear:
        return None, False
    arr = preprocessing.pil_to_cv2(image)
    warped = cv2.warpAffine(
        arr, M, sample_size,
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )
    return preprocessing.cv2_to_pil(warped), True


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


def border_from_transform(
    transform: dict[str, Any] | None,
    tpl_border: dict[str, float] | None,
    tpl_w: float | None,
    tpl_h: float | None,
    doc_w: float | None,
    doc_h: float | None,
) -> dict[str, float] | None:
    """Borde del documento derivado de la transformación de anclas: mapea el borde
    de la plantilla (toda su área de campos) al documento. Coherente con las anclas,
    a diferencia de detect_border que en escaneos con tablas detecta una columna.
    Devuelve None si no hay transformación.
    """
    if not transform:
        return None
    full = tpl_border or matching.FULL_BORDER
    # El borde de la plantilla, como región relativa-al-borde, es (0,0,1,1)
    reg = apply_transform_to_region(
        {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
        transform, full, tpl_w, tpl_h, doc_w, doc_h,
    )
    # Acotar a [0,1]
    x = max(0.0, min(1.0, reg["x"]))
    y = max(0.0, min(1.0, reg["y"]))
    w = max(0.05, min(1.0 - x, reg["w"]))
    h = max(0.05, min(1.0 - y, reg["h"]))
    return {"x": round(x, 5), "y": round(y, 5), "w": round(w, 5), "h": round(h, 5)}


# ---------------------------------------------------------------------------
# Giro cardinal (90/180/270) deducido de las anclas de IMAGEN
# ---------------------------------------------------------------------------

def snap_orientation(transform: dict[str, Any] | None) -> int:
    """Giro cardinal del documento respecto a la plantilla (0/90/180/270) según el
    ángulo de la transformación de anclas. Devuelve 0 si no es un giro cardinal claro.
    """
    if not transform:
        return 0
    angle = transform.get("angle", 0.0) or 0.0
    a = ((angle + 180) % 360) - 180   # normaliza a [-180, 180]
    nearest = round(a / 90.0) * 90
    if nearest % 360 == 0:
        return 0
    if abs(a - nearest) <= 20:        # razonablemente cerca de un múltiplo de 90
        return int(nearest % 360)
    return 0


def _osd_rotate(image: Image.Image) -> tuple[int, float]:
    """Orientación absoluta del texto vía OSD de Tesseract.

    Devuelve (grados_a_rotar_horario, confianza). El OSD detecta si el texto está
    en pie / 90 / 180 / 270, cosa que ORB (invariante a rotación) no puede.
    """
    import pytesseract

    try:
        osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
        return int(osd.get("rotate", 0)) % 360, float(osd.get("orientation_conf", 0) or 0)
    except Exception:  # noqa: BLE001
        return 0, 0.0


def cardinal_rotation(db, doc, tpl, image: Image.Image) -> int:
    """Giro cardinal (0/90/180/270) que hay que aplicar a `image` para enderezarla.

    Combina dos señales independientes:
      - PROPORCIÓN frente a la muestra (best_rotation_by_orb): fija el eje
        (0/180 vs 90/270). ORB es invariante a la rotación, así que NO distingue
        un giro de 180° por sí solo.
      - OSD de Tesseract: da la orientación ABSOLUTA del texto (en pie / invertido),
        que es lo que desambigua 90 vs 270 y 0 vs 180.

    Si ambas coinciden en el eje, se usa el ángulo de OSD (preciso para el sentido).
    Si OSD no tiene confianza, se cae al de la proporción.
    """
    sample_img = _sample_image(db, tpl)
    if sample_img is None:
        return 0
    try:
        ratio_angle, _inliers = preprocessing.best_rotation_by_orb(image, sample_img)
        ratio_angle %= 360
        osd_angle, osd_conf = _osd_rotate(image)

        # Mismo eje (par/impar de 90°): proporción y OSD concuerdan -> usar OSD
        if (ratio_angle % 180) == (osd_angle % 180):
            return osd_angle if osd_conf >= 1.0 else ratio_angle
        # Difieren de eje: si OSD es fiable, mandar OSD (orientación del texto);
        # si no, quedarnos con la proporción (cambio de aspecto es señal fuerte).
        return osd_angle if osd_conf >= 2.0 else ratio_angle
    except Exception:  # noqa: BLE001
        return 0


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
