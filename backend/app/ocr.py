"""OCR con Tesseract + extracción de firma geométrica del documento.

Todas las coordenadas que salen de este módulo están NORMALIZADAS (0..1)
respecto al ancho/alto de la imagen, de modo que el matching y la extracción
sean independientes de la resolución/escala del documento.
"""
from __future__ import annotations

import io
from typing import Any

import cv2
import numpy as np
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes

from .config import settings

# Tamaño de la rejilla de densidad de texto usada en la firma (GRID x GRID celdas)
GRID = 8


def load_first_page(content: bytes, content_type: str) -> Image.Image:
    """Convierte el fichero subido (imagen o PDF) en una imagen PIL (primera página)."""
    if content_type == "application/pdf" or content[:4] == b"%PDF":
        pages = convert_from_bytes(content, dpi=200, first_page=1, last_page=1)
        if not pages:
            raise ValueError("No se pudo convertir el PDF")
        return pages[0].convert("RGB")
    return Image.open(io.BytesIO(content)).convert("RGB")


def correct_orientation(image: Image.Image) -> tuple[Image.Image, int]:
    """Detecta la orientación con OSD de Tesseract y endereza la imagen.

    Devuelve (imagen_corregida, grados_rotados). Si OSD falla (poco texto, etc.)
    no rota y devuelve 0.
    """
    try:
        osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
        rotate = int(osd.get("rotate", 0)) % 360
    except Exception:  # noqa: BLE001
        rotate = 0
    if rotate:
        # OSD da el ángulo en el que está girada la página; rotamos en sentido
        # contrario para enderezarla (expand para no recortar).
        image = image.rotate(-rotate, expand=True)
    return image, rotate


def run_ocr(image: Image.Image) -> list[dict[str, Any]]:
    """Devuelve la lista de palabras detectadas con su caja normalizada."""
    width, height = image.size
    data = pytesseract.image_to_data(
        image, lang=settings.ocr_langs, output_type=pytesseract.Output.DICT
    )
    words: list[dict[str, Any]] = []
    n = len(data["text"])
    for i in range(n):
        text = (data["text"][i] or "").strip()
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1.0
        if not text or conf < 0:
            continue
        left, top = data["left"][i], data["top"][i]
        w, h = data["width"][i], data["height"][i]
        words.append(
            {
                "text": text,
                "conf": round(conf, 1),
                "box": {
                    "x": round(left / width, 5),
                    "y": round(top / height, 5),
                    "w": round(w / width, 5),
                    "h": round(h / height, 5),
                },
            }
        )
    return words


def ocr_region(image: Image.Image, box: dict[str, float]) -> dict[str, Any]:
    """Re-ejecuta OCR sobre un recorte concreto (coords normalizadas 0..1).

    Más preciso que reutilizar el OCR de la página completa: recorta la región,
    la amplía si es pequeña y usa PSM 6 (bloque uniforme de texto).
    """
    width, height = image.size
    left = int(box["x"] * width)
    top = int(box["y"] * height)
    right = int((box["x"] + box["w"]) * width)
    bottom = int((box["y"] + box["h"]) * height)

    # Clamp dentro de la imagen y asegura tamaño mínimo
    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(left + 1, min(right, width))
    bottom = max(top + 1, min(bottom, height))

    crop = image.crop((left, top, right, bottom))

    # Amplía recortes pequeños para mejorar el reconocimiento
    if crop.width < 600:
        scale = min(4, max(1, round(600 / max(1, crop.width))))
        if scale > 1:
            crop = crop.resize(
                (crop.width * scale, crop.height * scale), Image.LANCZOS
            )

    data = pytesseract.image_to_data(
        crop,
        lang=settings.ocr_langs,
        config="--psm 6",
        output_type=pytesseract.Output.DICT,
    )
    parts: list[str] = []
    confs: list[float] = []
    for i in range(len(data["text"])):
        text = (data["text"][i] or "").strip()
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1.0
        if text and conf >= 0:
            parts.append(text)
            confs.append(conf)

    return {
        "text": " ".join(parts).strip(),
        "confidence": round(sum(confs) / len(confs), 1) if confs else 0.0,
    }


def _background_color(arr: np.ndarray) -> list[int]:
    """Color de fondo dominante: mediana de una franja de los cuatro bordes."""
    height, width = arr.shape[:2]
    band = max(2, min(width, height) // 50)
    edges = np.concatenate(
        [
            arr[:band, :, :].reshape(-1, 3),
            arr[-band:, :, :].reshape(-1, 3),
            arr[:, :band, :].reshape(-1, 3),
            arr[:, -band:, :].reshape(-1, 3),
        ]
    )
    return [int(c) for c in np.median(edges, axis=0)]


FULL_BORDER = {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}


def detect_border(image: Image.Image) -> dict[str, float]:
    """Detecta el rectángulo del documento separándolo del fondo del escaneo.

    Estrategia: máscara de píxeles que difieren del color de fondo, se cierra
    morfológicamente y se toma el bounding box del mayor contorno. Si no se
    encuentra algo razonable, devuelve el documento completo.
    """
    arr = np.asarray(image)
    height, width = arr.shape[:2]
    bg = np.array(_background_color(arr), dtype=int)

    diff = np.abs(arr.astype(int) - bg).sum(axis=2).astype(np.uint8)
    _, mask = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return dict(FULL_BORDER)

    x, y, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
    area_frac = (bw * bh) / (width * height)
    # Descarta resultados degenerados (ruido pequeño o casi toda la imagen)
    if area_frac < 0.15 or area_frac > 0.995:
        return dict(FULL_BORDER)

    return {
        "x": round(x / width, 5),
        "y": round(y / height, 5),
        "w": round(bw / width, 5),
        "h": round(bh / height, 5),
    }


def rectify(image: Image.Image, quad: dict[str, dict]) -> Image.Image:
    """Endereza el documento mediante una transformación de perspectiva (4 puntos).

    `quad` = {tl,tr,br,bl} con cada esquina {x,y} normalizada (0..1). El documento
    delimitado por el cuadrilátero se 'aplana' a un rectángulo recto.
    """
    arr = np.asarray(image)
    h, w = arr.shape[:2]

    def pt(key: str) -> list[float]:
        return [quad[key]["x"] * w, quad[key]["y"] * h]

    src = np.float32([pt("tl"), pt("tr"), pt("br"), pt("bl")])
    tl, tr, br, bl = src

    def dist(a, b) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    out_w = max(10, int(max(dist(br, bl), dist(tr, tl))))
    out_h = max(10, int(max(dist(tr, br), dist(tl, bl))))
    dst = np.float32(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]]
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(arr, matrix, (out_w, out_h))
    return Image.fromarray(warped)


def compute_signature(
    image: Image.Image,
    words: list[dict[str, Any]],
    border: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Firma geométrica del documento (relativa al borde) para emparejar plantillas.

    Componentes:
      - aspect_ratio: ancho/alto del documento (dentro del borde)
      - bg_color: color de fondo dominante RGB 0..255
      - density: rejilla GRIDxGRID con densidad de texto, en coords del borde
    """
    width, height = image.size
    arr = np.asarray(image)
    b = border or FULL_BORDER
    bx, by, bw, bh = b["x"], b["y"], max(b["w"], 1e-6), max(b["h"], 1e-6)

    bg_color = _background_color(arr)

    # Densidad de texto remapeada a coordenadas relativas al borde
    density = np.zeros((GRID, GRID), dtype=float)
    for word in words:
        box = word["box"]
        cx = (box["x"] + box["w"] / 2 - bx) / bw
        cy = (box["y"] + box["h"] / 2 - by) / bh
        if not (0 <= cx <= 1 and 0 <= cy <= 1):
            continue
        gx = min(GRID - 1, int(cx * GRID))
        gy = min(GRID - 1, int(cy * GRID))
        density[gy, gx] += (box["w"] / bw) * (box["h"] / bh)
    if density.max() > 0:
        density = density / density.max()

    return {
        "aspect_ratio": round((width * bw) / (height * bh), 4),
        "bg_color": bg_color,
        "density": density.flatten().round(4).tolist(),
    }


def process_upload(content: bytes, content_type: str) -> dict[str, Any]:
    """Pipeline completo: imagen -> orientación -> OCR -> borde -> firma."""
    image = load_first_page(content, content_type)
    image, rotation = correct_orientation(image)
    words = run_ocr(image)
    border = detect_border(image)
    signature = compute_signature(image, words, border)
    return {
        "image": image,
        "width": image.size[0],
        "height": image.size[1],
        "ocr_words": words,
        "signature": signature,
        "border": border,
        "rotation": rotation,
    }
