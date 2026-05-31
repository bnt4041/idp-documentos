"""
Preprocesado avanzado de imagen para IDP: enderezado fino, filtros múltiples,
detección multi-ángulo y localización de zonas por características visuales.

Flujo recomendado:
  1. deskew_fine()        -> corrige inclinaciones de 0.5°–45°
  2. preprocess_pipeline() -> aplica filtros (gris, binarización, CLAHE, etc.)
  3. try_orientations()   -> prueba 0°/90°/180°/270° y elige la mejor coincidencia
  4. detect_zones()       -> localiza zonas de la plantilla con template matching visual
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Conversión PIL <-> OpenCV
# ---------------------------------------------------------------------------

def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """PIL RGB -> OpenCV BGR."""
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(arr: np.ndarray) -> Image.Image:
    """OpenCV BGR -> PIL RGB."""
    if arr.ndim == 2:
        return Image.fromarray(arr)
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


# ---------------------------------------------------------------------------
# Filtros de imagen
# ---------------------------------------------------------------------------

def apply_grayscale(image: Image.Image) -> Image.Image:
    """Convierte a escala de grises (devuelve PIL modo 'L')."""
    return image.convert("L")


def apply_binary_otsu(image: Image.Image) -> Image.Image:
    """Binarización global con umbral de Otsu (blanco y negro puro)."""
    arr = pil_to_cv2(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if arr.ndim == 3 else arr
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2_to_pil(binary)


def apply_adaptive_threshold(image: Image.Image, block_size: int = 31, c: int = 10) -> Image.Image:
    """Binarización adaptativa: útil con iluminación irregular o sombras."""
    arr = pil_to_cv2(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if arr.ndim == 3 else arr
    # block_size debe ser impar
    if block_size % 2 == 0:
        block_size += 1
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, c,
    )
    return cv2_to_pil(binary)


def apply_clahe(image: Image.Image, clip_limit: float = 2.0, grid_size: int = 8) -> Image.Image:
    """Mejora de contraste local (CLAHE). Excelente para resaltar texto débil."""
    arr = pil_to_cv2(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if arr.ndim == 3 else arr
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    enhanced = clahe.apply(gray)
    return cv2_to_pil(enhanced)


def apply_sharpen(image: Image.Image, strength: float = 1.0) -> Image.Image:
    """Filtro de nitidez (unsharp masking) para resaltar bordes del texto."""
    arr = pil_to_cv2(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if arr.ndim == 3 else arr
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    sharpened = cv2.addWeighted(gray, 1.0 + strength, blurred, -strength, 0)
    return cv2_to_pil(sharpened)


def apply_denoise(image: Image.Image, strength: float = 10) -> Image.Image:
    """Reducción de ruido preservando bordes (Non-Local Means)."""
    arr = pil_to_cv2(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if arr.ndim == 3 else arr
    denoised = cv2.fastNlMeansDenoising(gray, None, strength, 7, 21)
    return cv2_to_pil(denoised)


def apply_morphology_clean(image: Image.Image) -> Image.Image:
    """Limpieza morfológica: elimina puntos sueltos y cierra huecos en letras."""
    arr = pil_to_cv2(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if arr.ndim == 3 else arr
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2_to_pil(cleaned)


# ---------------------------------------------------------------------------
# Pipeline de preprocesado
# ---------------------------------------------------------------------------

FILTERS = {
    "grayscale": apply_grayscale,
    "binary_otsu": apply_binary_otsu,
    "adaptive_threshold": apply_adaptive_threshold,
    "clahe": apply_clahe,
    "sharpen": apply_sharpen,
    "denoise": apply_denoise,
    "morphology_clean": apply_morphology_clean,
}


def preprocess_pipeline(
    image: Image.Image,
    steps: list[str] | None = None,
) -> tuple[Image.Image, list[str]]:
    """Aplica una secuencia de filtros. Si no se especifica, usa defaults para OCR.

    Devuelve (imagen_procesada, pasos_aplicados).
    Las imágenes intermedias se convierten a 'L' (gris) tras el primer filtro.
    """
    if steps is None:
        # Defaults óptimos para OCR general
        steps = ["grayscale", "clahe", "sharpen"]
    applied: list[str] = []
    out = image
    for step in steps:
        fn = FILTERS.get(step)
        if fn:
            try:
                out = fn(out)
                applied.append(step)
            except Exception:  # noqa: BLE001
                pass  # si un filtro falla, continuamos con el siguiente
    return out, applied


def generate_filtered_variants(image: Image.Image) -> dict[str, Image.Image]:
    """Genera múltiples variantes de la imagen con distintos filtros combinados.

    Devuelve un diccionario {nombre: imagen} para usar en OCR ensemble.
    """
    variants: dict[str, Image.Image] = {}

    # Original en gris
    gray = apply_grayscale(image)
    variants["grayscale"] = gray

    # Binarización Otsu
    variants["binary"] = apply_binary_otsu(image)

    # CLAHE (bueno para texto tenue)
    variants["clahe"] = apply_clahe(image)

    # Denoise + CLAHE (para documentos con ruido)
    variants["denoise_clahe"] = apply_clahe(apply_denoise(image))

    # Adaptativo (bueno para iluminación irregular)
    variants["adaptive"] = apply_adaptive_threshold(image)

    # Sharpen + Otsu
    variants["sharpen_binary"] = apply_binary_otsu(apply_sharpen(image))

    # Morfología limpia
    variants["morph_clean"] = apply_morphology_clean(image)

    return variants


# ---------------------------------------------------------------------------
# Enderezado fino (deskew)
# ---------------------------------------------------------------------------

def deskew_fine(image: Image.Image, max_angle: float = 45.0) -> tuple[Image.Image, float]:
    """Enderezado fino usando la transformada de Hough sobre líneas de texto.

    Corrige inclinaciones pequeñas (0.5°–30°) que Tesseract OSD no detecta bien.
    Devuelve (imagen_enderezada, ángulo_corregido).

    Solo actúa si hay al menos 5 líneas con inclinación clara (entre 0.5° y 30°).
    Ignora líneas horizontales (<0.5°) y casi-verticales (>30°) para no confundir
    bordes de tabla, subrayados o columnas con texto inclinado.
    """
    arr = pil_to_cv2(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if arr.ndim == 3 else arr

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    lines = cv2.HoughLinesP(
        binary, 1, np.pi / 180 / 2,
        threshold=100, minLineLength=80, maxLineGap=10,
    )

    if lines is None or len(lines) < 5:
        return image, 0.0

    # Solo líneas con inclinación apreciable (0.5° a 30°)
    angles: list[float] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angle = angle % 180
        if angle > 90:
            angle -= 180
        if 0.5 < abs(angle) <= min(max_angle, 30.0):
            angles.append(angle)

    if len(angles) < 5:
        return image, 0.0

    median_angle = float(np.median(angles))

    if abs(median_angle) < 0.5:
        return image, 0.0

    # Rotar
    h, w = gray.shape[:2]
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    cos = abs(rot_matrix[0, 0])
    sin = abs(rot_matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    rot_matrix[0, 2] += new_w / 2 - center[0]
    rot_matrix[1, 2] += new_h / 2 - center[1]

    if arr.ndim == 3:
        rotated = cv2.warpAffine(arr, rot_matrix, (new_w, new_h),
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))
    else:
        rotated = cv2.warpAffine(arr, rot_matrix, (new_w, new_h),
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=255)

    return cv2_to_pil(rotated), round(median_angle, 2)


# ---------------------------------------------------------------------------
# Detección de ángulo de rotación
# ---------------------------------------------------------------------------

def detect_rotation_angle(image: Image.Image) -> float:
    """Estima el ángulo de rotación del documento combinando Hough + minAreaRect.

    Devuelve el ángulo en grados (positivo = horario).
    """
    arr = pil_to_cv2(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if arr.ndim == 3 else arr
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Método 1: HoughLinesP (texto)
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180 / 2, threshold=100,
                            minLineLength=60, maxLineGap=15)
    angles: list[float] = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            a = math.degrees(math.atan2(y2 - y1, x2 - x1))
            a = a % 180
            if a > 90:
                a -= 180
            if abs(a) <= 45:
                angles.append(a)

    # Método 2: minAreaRect del contorno del documento
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)
        rect_angle = rect[2]
        # Normalizar: minAreaRect devuelve ángulo del lado más cercano a horizontal
        if rect_angle < -45:
            rect_angle += 90
        if abs(rect_angle) <= 45:
            angles.append(rect_angle)

    if not angles:
        return 0.0

    return round(float(np.median(angles)), 2)


# ---------------------------------------------------------------------------
# Detección multi-ángulo de plantillas
# ---------------------------------------------------------------------------

def try_orientations(
    image: Image.Image,
    match_fn,
    candidates: list[Any],
    angles: list[int] | None = None,
) -> tuple[Any | None, float, int, Image.Image]:
    """Prueba la imagen en varios ángulos y devuelve la mejor coincidencia.

    Args:
        image: imagen a procesar
        match_fn: función (img, candidates) -> (best_candidate, score)
        candidates: lista de candidatos (plantillas)
        angles: ángulos a probar (default: [0, 90, 180, 270])

    Returns:
        (mejor_candidato, mejor_score, mejor_ángulo, imagen_rotada)
    """
    if angles is None:
        angles = [0, 90, 180, 270]

    best_candidate = None
    best_score = 0.0
    best_angle = 0
    best_image = image

    for angle in angles:
        if angle == 0:
            rotated = image
        else:
            rotated = image.rotate(-angle, expand=True)

        try:
            candidate, score = match_fn(rotated, candidates)
            if score > best_score:
                best_score = score
                best_candidate = candidate
                best_angle = angle
                best_image = rotated
        except Exception:  # noqa: BLE001
            continue

    return best_candidate, best_score, best_angle, best_image


# ---------------------------------------------------------------------------
# Detección de zonas de plantilla mediante template matching visual
# ---------------------------------------------------------------------------

def detect_template_zones(
    document: Image.Image,
    template_sample: Image.Image,
    method: int = cv2.TM_CCOEFF_NORMED,
    threshold: float = 0.3,
) -> dict[str, Any]:
    """Detecta la posición de la plantilla dentro del documento mediante
    template matching visual (correlación cruzada normalizada).

    Útil cuando la plantilla está en una zona concreta del escaneo (ej. una
    etiqueta en una esquina de la hoja).

    Devuelve:
        {
            "found": bool,
            "score": float,           # 0..1
            "region": {x, y, w, h},  # normalizada 0..1
            "offset_x": float,        # desplazamiento x normalizado
            "offset_y": float,        # desplazamiento y normalizado
        }
    """
    doc_arr = pil_to_cv2(document)
    tpl_arr = pil_to_cv2(template_sample)

    # Convertir a gris
    doc_gray = cv2.cvtColor(doc_arr, cv2.COLOR_BGR2GRAY) if doc_arr.ndim == 3 else doc_arr
    tpl_gray = cv2.cvtColor(tpl_arr, cv2.COLOR_BGR2GRAY) if tpl_arr.ndim == 3 else tpl_arr

    dh, dw = doc_gray.shape[:2]
    th, tw = tpl_gray.shape[:2]

    if th > dh or tw > dw:
        # La plantilla es más grande que el documento -> escalamos la plantilla
        scale = min(dh / th, dw / tw) * 0.9
        new_th, new_tw = int(th * scale), int(tw * scale)
        tpl_gray = cv2.resize(tpl_gray, (new_tw, new_th))
        th, tw = new_th, new_tw

    result = cv2.matchTemplate(doc_gray, tpl_gray, method)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    found = max_val >= threshold
    x, y = max_loc

    return {
        "found": found,
        "score": round(float(max_val), 4),
        "region": {
            "x": round(x / dw, 5),
            "y": round(y / dh, 5),
            "w": round(tw / dw, 5),
            "h": round(th / dh, 5),
        },
        "offset_x": round(x / dw, 5),
        "offset_y": round(y / dh, 5),
    }


def detect_template_zones_multiscale(
    document: Image.Image,
    template_sample: Image.Image,
    scales: list[float] | None = None,
    threshold: float = 0.3,
) -> dict[str, Any]:
    """Template matching a múltiples escalas para encontrar la plantilla
    aunque el documento esté a distinta resolución.

    Args:
        scales: factores de escala a probar (default: [0.5, 0.75, 1.0, 1.25, 1.5])
    """
    if scales is None:
        scales = [0.5, 0.75, 1.0, 1.25, 1.5]

    doc_arr = pil_to_cv2(document)
    tpl_arr = pil_to_cv2(template_sample)
    doc_gray = cv2.cvtColor(doc_arr, cv2.COLOR_BGR2GRAY) if doc_arr.ndim == 3 else doc_arr
    tpl_gray = cv2.cvtColor(tpl_arr, cv2.COLOR_BGR2GRAY) if tpl_arr.ndim == 3 else tpl_arr

    dh, dw = doc_gray.shape[:2]
    th_orig, tw_orig = tpl_gray.shape[:2]

    best_score = 0.0
    best_region: dict[str, float] | None = None

    for scale in scales:
        new_tw = int(tw_orig * scale)
        new_th = int(th_orig * scale)
        if new_tw < 20 or new_th < 20:
            continue
        if new_tw > dw or new_th > dh:
            continue

        tpl_scaled = cv2.resize(tpl_gray, (new_tw, new_th))
        result = cv2.matchTemplate(doc_gray, tpl_scaled, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_score:
            best_score = max_val
            best_region = {
                "x": round(max_loc[0] / dw, 5),
                "y": round(max_loc[1] / dh, 5),
                "w": round(new_tw / dw, 5),
                "h": round(new_th / dh, 5),
            }

    return {
        "found": best_score >= threshold and best_region is not None,
        "score": round(float(best_score), 4),
        "region": best_region,
        "offset_x": best_region["x"] if best_region else 0.0,
        "offset_y": best_region["y"] if best_region else 0.0,
    }


# ---------------------------------------------------------------------------
# Detección combinada de zona: ORB + template matching multiescala
# ---------------------------------------------------------------------------

def detect_best_zone(
    document: Image.Image,
    template_sample: Image.Image,
    angles: list[int] | None = None,
) -> dict[str, Any]:
    """Busca la zona de la plantilla dentro del documento combinando ORB y
    template matching a múltiples escalas y ángulos.

    Devuelve la mejor zona encontrada con su método y score.
    """
    if angles is None:
        angles = [0, 90, 180, 270]

    best: dict[str, Any] = {"found": False, "method": "none", "score": 0.0, "region": None}

    # 1) ORB (más robusto si hay suficientes keypoints)
    for angle in angles:
        doc_rot = document if angle == 0 else document.rotate(-angle, expand=True)
        try:
            orb_result = detect_zones_orb(doc_rot, template_sample, min_matches=8)
            if orb_result["found"]:
                inliers = orb_result.get("inliers", 0)
                matches = orb_result.get("matches", 1)
                score = min(1.0, (inliers / max(1, matches)) / 0.5)
                if score > best["score"]:
                    # Las coordenadas de ORB ya están en el espacio del doc rotado
                    best = {
                        "found": True,
                        "method": "orb",
                        "score": round(score, 4),
                        "region": orb_result["region"],
                        "angle": angle,
                        "corners": orb_result.get("corners"),
                    }
        except Exception:  # noqa: BLE001
            continue

    # 2) Template matching multiescala si ORB no encontró nada bueno
    if best["score"] < 0.3:
        tm_result = detect_template_zones_multiscale(document, template_sample, threshold=0.25)
        if tm_result["found"] and tm_result["score"] > best["score"]:
            best = {
                "found": True,
                "method": "template_multiscale",
                "score": tm_result["score"],
                "region": tm_result["region"],
                "angle": 0,
            }

    return best


# ---------------------------------------------------------------------------
# Detección de zonas por keypoints (ORB)
# ---------------------------------------------------------------------------

def detect_zones_orb(
    document: Image.Image,
    template_sample: Image.Image,
    min_matches: int = 8,
) -> dict[str, Any]:
    """Detecta la zona de la plantilla usando ORB (keypoints + homografía).

    Más robusto que template matching para cambios de escala, rotación
    moderada y perspectiva. Devuelve la región y la matriz de homografía.
    """
    doc_arr = pil_to_cv2(document)
    tpl_arr = pil_to_cv2(template_sample)

    doc_gray = cv2.cvtColor(doc_arr, cv2.COLOR_BGR2GRAY) if doc_arr.ndim == 3 else doc_arr
    tpl_gray = cv2.cvtColor(tpl_arr, cv2.COLOR_BGR2GRAY) if tpl_arr.ndim == 3 else tpl_arr

    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(tpl_gray, None)
    kp2, des2 = orb.detectAndCompute(doc_gray, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return {"found": False, "matches": 0, "homography": None, "region": None}

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)

    good = matches[:50]  # top 50 matches

    if len(good) < min_matches:
        return {"found": False, "matches": len(good), "homography": None, "region": None}

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return {"found": False, "matches": len(good), "homography": None, "region": None}

    inliers = int(mask.sum()) if mask is not None else 0

    # Proyectar las esquinas de la plantilla al documento
    th, tw = tpl_gray.shape[:2]
    dh, dw = doc_gray.shape[:2]
    corners = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

    x0 = max(0, projected[:, 0].min())
    y0 = max(0, projected[:, 1].min())
    x1 = min(dw, projected[:, 0].max())
    y1 = min(dh, projected[:, 1].max())

    return {
        "found": True,
        "matches": len(good),
        "inliers": inliers,
        "homography": H.tolist(),
        "region": {
            "x": round(x0 / dw, 5),
            "y": round(y0 / dh, 5),
            "w": round((x1 - x0) / dw, 5),
            "h": round((y1 - y0) / dh, 5),
        },
        "corners": [
            {"x": round(p[0] / dw, 5), "y": round(p[1] / dh, 5)}
            for p in projected
        ],
    }


# ---------------------------------------------------------------------------
# Utilidad: evaluar calidad de imagen para OCR
# ---------------------------------------------------------------------------

def evaluate_image_quality(image: Image.Image) -> dict[str, float]:
    """Estima la calidad de la imagen para OCR.

    Devuelve métricas: contraste, nitidez (Laplacian), nivel de ruido, etc.
    """
    arr = pil_to_cv2(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if arr.ndim == 3 else arr

    # Nitidez: varianza del Laplaciano
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(laplacian.var())

    # Contraste: desviación estándar
    contrast = float(gray.std())

    # Ruido estimado: std del high-pass (diferencia con la media local)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = float(np.std(gray.astype(float) - blurred.astype(float)))

    # Brillo medio
    brightness = float(gray.mean())

    return {
        "sharpness": round(sharpness, 2),
        "contrast": round(contrast, 2),
        "noise": round(noise, 2),
        "brightness": round(brightness, 2),
    }


# ---------------------------------------------------------------------------
# Similitud visual multi-ángulo + multi-filtro
# ---------------------------------------------------------------------------

def visual_similarity_score(
    doc_path: str,
    template_sample_path: str,
    angles: list[int] | None = None,
    filters: list[str] | None = None,
) -> float:
    """Calcula el % de similitud visual entre dos imágenes de documentos.

    Prueba ORB en varios ángulos (0°, 90°, 180°, 270°) y con varios filtros
    (gris, binario, CLAHE) para encontrar la mejor coincidencia. Devuelve
    un score 0..1 basado en el ratio de inliers sobre matches totales.

    Args:
        doc_path: ruta al documento subido
        template_sample_path: ruta a la imagen de muestra de la plantilla
        angles: ángulos a probar (default [0, 90, 180, 270])
        filters: filtros a probar (default ["grayscale", "binary", "clahe"])

    Returns:
        score 0..1 (0 = nada parecido, 1 = idéntico)
    """
    if angles is None:
        angles = [0, 90, 180, 270]
    if filters is None:
        filters = ["grayscale", "binary", "clahe"]

    doc_img = Image.open(doc_path).convert("RGB")
    tpl_img = Image.open(template_sample_path).convert("RGB")

    best_score = 0.0

    for angle in angles:
        doc_rot = doc_img if angle == 0 else doc_img.rotate(-angle, expand=True)

        for flt in filters:
            try:
                doc_filtered, _ = preprocess_pipeline(doc_rot, [flt])
                tpl_filtered, _ = preprocess_pipeline(tpl_img, [flt])

                # Asegurar que ambas están en modo 'L' para ORB
                if doc_filtered.mode != "L":
                    doc_filtered = doc_filtered.convert("L")
                if tpl_filtered.mode != "L":
                    tpl_filtered = tpl_filtered.convert("L")

                orb_result = detect_zones_orb(doc_filtered, tpl_filtered, min_matches=8)

                if orb_result["found"]:
                    # Score basado en ratio inliers / matches (calidad del match)
                    inliers = orb_result.get("inliers", 0)
                    matches = orb_result.get("matches", 1)
                    ratio = inliers / max(1, matches)
                    # Normalizar: un buen match suele tener ratio > 0.4
                    score = min(1.0, ratio / 0.5)
                    if score > best_score:
                        best_score = score
            except Exception:  # noqa: BLE001
                continue

    return round(best_score, 4)


def visual_similarity_from_images(
    doc_img: Image.Image,
    tpl_img: Image.Image,
    angles: list[int] | None = None,
) -> float:
    """Como visual_similarity_score pero recibiendo imágenes PIL en memoria.

    Útil cuando el documento aún no se ha guardado a disco.
    """
    if angles is None:
        angles = [0, 90, 180, 270]

    best_score = 0.0

    for angle in angles:
        doc_rot = doc_img if angle == 0 else doc_img.rotate(-angle, expand=True)

        try:
            doc_gray = doc_rot.convert("L")
            tpl_gray = tpl_img.convert("L")

            orb_result = detect_zones_orb(doc_gray, tpl_gray, min_matches=8)

            if orb_result["found"]:
                inliers = orb_result.get("inliers", 0)
                matches = orb_result.get("matches", 1)
                ratio = inliers / max(1, matches)
                score = min(1.0, ratio / 0.5)
                if score > best_score:
                    best_score = score
        except Exception:  # noqa: BLE001
            continue

    return round(best_score, 4)
