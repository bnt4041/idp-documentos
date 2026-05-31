"""
Análisis inteligente del OCR para proponer campos de formulario automáticamente.

Cuando se sube un documento de muestra para crear una plantilla, este módulo
analiza las palabras detectadas y sugiere campos basándose en:

  1. Patrones sintácticos: "Etiqueta:" seguido de valor
  2. Patrones semánticos: palabras clave como "nombre", "fecha", "dni", etc.
  3. Tipado automático: detecta si el valor es fecha, número, texto, DNI…
  4. Posición espacial: agrupa palabras cercanas en filas para formar pares clave-valor
"""
from __future__ import annotations

import json
import re
from typing import Any


# ---------------------------------------------------------------------------
# Diccionarios de palabras clave por idioma (español principalmente)
# ---------------------------------------------------------------------------

LABEL_PATTERNS: dict[str, list[str]] = {
    "nombre": [
        "nombre", "apellidos", "apellido", "razón social", "razon social",
        "titular", "cliente", "paciente", "alumno", "empleado",
    ],
    "dni": [
        "dni", "nif", "nie", "cif", "documento", "identificación",
        "identificacion", "id", "pasaporte", "nº", "n.º", "núm", "num",
    ],
    "fecha": [
        "fecha", "fecha de nacimiento", "fecha nacimiento", "fecha emisión",
        "fecha emision", "fecha expedición", "fecha expedicion", "fecha validez",
        "fecha inicio", "fecha fin", "nacido", "f. nacimiento",
    ],
    "direccion": [
        "dirección", "direccion", "domicilio", "calle", "población",
        "poblacion", "provincia", "código postal", "codigo postal", "cp",
        "localidad", "municipio", "residencia",
    ],
    "telefono": [
        "teléfono", "telefono", "tlf", "tfno", "móvil", "movil",
        "contacto", "tel", "fax",
    ],
    "email": [
        "email", "correo", "e-mail", "mail", "correo electrónico",
        "correo electronico",
    ],
    "importe": [
        "importe", "total", "subtotal", "base imponible", "iva",
        "irpf", "neto", "bruto", "cantidad", "precio", "coste", "costo",
        "saldo", "cuota", "cuantía", "cuantia", "importe total",
        "total factura", "a pagar", "a ingresar", "import",
    ],
    "banco": [
        "iban", "bic", "swift", "cuenta", "entidad", "oficina",
        "dc", "ccc", "cuenta corriente", "número de cuenta", "numero de cuenta",
        "titular cuenta", "banco", "caja",
    ],
    "referencia": [
        "referencia", "ref", "nº expediente", "expediente", "nº factura",
        "factura nº", "factura no", "albarán", "albaran", "pedido", "código",
        "codigo", "matrícula", "matricula", "registro",
    ],
    "firma": [
        "firma", "firmado", "fdo", "sello", "conforme", "recibí", "recibi",
    ],
}

# Mapa inverso: palabra clave -> categoría
_KEYWORD_TO_FIELD: dict[str, str] = {}
for _field_type, _keywords in LABEL_PATTERNS.items():
    for _kw in _keywords:
        _KEYWORD_TO_FIELD[_kw] = _field_type


# ---------------------------------------------------------------------------
# Detectores de tipo de dato por regex
# ---------------------------------------------------------------------------

DATE_REGEX = re.compile(
    r"^\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}$"
)
CURRENCY_REGEX = re.compile(
    r"^[\d.,]+\s*[€$]$|^[€$]\s*[\d.,]+$"  # exige símbolo: si no, es 'number'
)
DNI_REGEX = re.compile(
    r"^\d{7,8}[A-Za-z]$"
)
PHONE_REGEX = re.compile(
    r"^\+?\d[\d\s\-.]{6,14}\d$"
)
EMAIL_REGEX = re.compile(
    r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
)
NUMBER_REGEX = re.compile(r"^[\d.,]+$")


def guess_data_type(value: str) -> str:
    """Adivina el tipo de dato mirando el formato del valor."""
    v = value.strip()
    if not v:
        return "text"
    if EMAIL_REGEX.match(v):
        return "email"
    if DATE_REGEX.match(v):
        return "date"
    if DNI_REGEX.match(v):
        return "dni"
    if PHONE_REGEX.match(v):
        return "phone"
    if CURRENCY_REGEX.match(v):
        return "currency"
    if NUMBER_REGEX.match(v):
        return "number"
    return "text"


def _norm(s: str) -> str:
    """Normaliza para comparar."""
    import unicodedata
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return "".join(c for c in s if c.isalnum() or c in ".:")


# ---------------------------------------------------------------------------
# Agrupación de palabras en líneas
# ---------------------------------------------------------------------------

def _group_by_lines(words: list[dict], tolerance: float = 0.01) -> list[list[dict]]:
    """Agrupa palabras por posición Y (misma línea) con tolerancia normalizada."""
    if not words:
        return []
    sorted_words = sorted(words, key=lambda w: (w["box"]["y"], w["box"]["x"]))
    lines: list[list[dict]] = []
    current_line: list[dict] = [sorted_words[0]]
    current_y = sorted_words[0]["box"]["y"]

    for w in sorted_words[1:]:
        if abs(w["box"]["y"] - current_y) < tolerance:
            current_line.append(w)
        else:
            lines.append(current_line)
            current_line = [w]
            current_y = w["box"]["y"]
    lines.append(current_line)
    return lines


def _line_text(words: list[dict]) -> str:
    return " ".join(w["text"] for w in words)


# ---------------------------------------------------------------------------
# Análisis principal
# ---------------------------------------------------------------------------

def analyze_words(
    ocr_words: list[dict[str, Any]],
    existing_fields: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Analiza las palabras OCR del documento y devuelve una lista de campos sugeridos.

    Cada sugerencia incluye:
      - key: clave única (snake_case)
      - name: nombre legible del campo
      - data_type: "text", "date", "number", "currency", "dni", "email", "phone"
      - x, y, w, h: región normalizada 0..1 del VALOR (no de la etiqueta)
      - sample_text: texto OCR del valor
      - confidence: confianza media del OCR en esa región
      - label_text: texto de la etiqueta detectada
      - label_region: región de la etiqueta (si se detectó)
    """
    if not ocr_words:
        return []

    existing_keys = set()
    if existing_fields:
        existing_keys = {f.get("key", "") for f in existing_fields}

    lines = _group_by_lines(ocr_words)
    suggestions: list[dict[str, Any]] = []

    for i, line_words in enumerate(lines):
        if len(line_words) < 2:
            continue

        line_str = _line_text(line_words)
        norm_line = _norm(line_str)

        # Buscar patrón "etiqueta: valor" o "etiqueta  valor"
        # Primero: ¿hay una palabra clave conocida al inicio de la línea?
        for j, word in enumerate(line_words):
            word_norm = _norm(word["text"])
            # Quitar posibles ":" al final
            word_clean = word_norm.rstrip(":")

            field_type = _KEYWORD_TO_FIELD.get(word_clean)
            # También buscar si la palabra es parte de una frase clave
            if not field_type and j == 0 and len(line_words) >= 3:
                phrase2 = _norm(line_words[0]["text"] + " " + line_words[1]["text"])
                phrase3 = _norm(
                    line_words[0]["text"] + " " + line_words[1]["text"] + " " + line_words[2]["text"]
                )
                field_type = (
                    _KEYWORD_TO_FIELD.get(phrase2)
                    or _KEYWORD_TO_FIELD.get(phrase3)
                )

            if field_type and j + 1 < len(line_words):
                # Las palabras restantes forman el valor
                value_words = line_words[j + 1:]
                value_text = " ".join(w["text"] for w in value_words).strip()
                if not value_text or len(value_text) < 1:
                    continue

                # Calcular región del valor
                vx = min(w["box"]["x"] for w in value_words)
                vy = min(w["box"]["y"] for w in value_words)
                vx2 = max(w["box"]["x"] + w["box"]["w"] for w in value_words)
                vy2 = max(w["box"]["y"] + w["box"]["h"] for w in value_words)
                value_region = {
                    "x": round(vx, 5),
                    "y": round(vy, 5),
                    "w": round(vx2 - vx, 5),
                    "h": round(vy2 - vy, 5),
                }

                # Región de la etiqueta
                label_words_list = line_words[: j + 1]
                lx = min(w["box"]["x"] for w in label_words_list)
                ly = min(w["box"]["y"] for w in label_words_list)
                lx2 = max(w["box"]["x"] + w["box"]["w"] for w in label_words_list)
                ly2 = max(w["box"]["y"] + w["box"]["h"] for w in label_words_list)

                avg_conf = (
                    sum(w["conf"] for w in value_words) / len(value_words)
                    if value_words
                    else 0.0
                )
                data_type = guess_data_type(value_text)

                # Generar key única
                base_key = field_type
                key = base_key
                counter = 1
                while key in existing_keys:
                    counter += 1
                    key = f"{base_key}_{counter}"

                suggestions.append({
                    "key": key,
                    "name": _label_to_name(word_norm.rstrip(":")),
                    "data_type": data_type,
                    "x": value_region["x"],
                    "y": value_region["y"],
                    "w": value_region["w"],
                    "h": value_region["h"],
                    "sample_text": value_text,
                    "confidence": round(avg_conf, 1),
                    "label_text": _line_text(label_words_list).rstrip(":").strip(),
                    "label_region": {
                        "x": round(lx, 5),
                        "y": round(ly, 5),
                        "w": round(lx2 - lx, 5),
                        "h": round(ly2 - ly, 5),
                    },
                })
                existing_keys.add(key)
                break  # solo un campo por línea

        # También detectar patrones "clave: valor" con ":"
        if not any(s.get("label_text") and _line_text(line_words).startswith(
            s.get("label_text", "")
        ) for s in suggestions[-1:] if suggestions):
            # Buscar el primer ":" en palabras
            for j, word in enumerate(line_words):
                if ":" in word["text"] and j + 1 < len(line_words):
                    label_text = _line_text(line_words[: j + 1]).rstrip(":").strip()
                    value_words = line_words[j + 1:]
                    value_text = " ".join(w["text"] for w in value_words).strip()
                    if not value_text or len(label_text) < 2:
                        continue

                    vx = min(w["box"]["x"] for w in value_words)
                    vy = min(w["box"]["y"] for w in value_words)
                    vx2 = max(w["box"]["x"] + w["box"]["w"] for w in value_words)
                    vy2 = max(w["box"]["y"] + w["box"]["h"] for w in value_words)

                    lx = min(w["box"]["x"] for w in line_words[: j + 1])
                    ly = min(w["box"]["y"] for w in line_words[: j + 1])
                    lx2 = max(w["box"]["x"] + w["box"]["w"] for w in line_words[: j + 1])
                    ly2 = max(w["box"]["y"] + w["box"]["h"] for w in line_words[: j + 1])

                    avg_conf = (
                        sum(w["conf"] for w in value_words) / len(value_words)
                        if value_words
                        else 0.0
                    )
                    data_type = guess_data_type(value_text)

                    base_key = _name_to_key(label_text)
                    key = base_key
                    counter = 1
                    while key in existing_keys:
                        counter += 1
                        key = f"{base_key}_{counter}"

                    suggestions.append({
                        "key": key,
                        "name": label_text[:50],
                        "data_type": data_type,
                        "x": round(vx, 5),
                        "y": round(vy, 5),
                        "w": round(vx2 - vx, 5),
                        "h": round(vy2 - vy, 5),
                        "sample_text": value_text,
                        "confidence": round(avg_conf, 1),
                        "label_text": label_text,
                        "label_region": {
                            "x": round(lx, 5),
                            "y": round(ly, 5),
                            "w": round(lx2 - lx, 5),
                            "h": round(ly2 - ly, 5),
                        },
                    })
                    existing_keys.add(key)
                    break

    return suggestions


def _label_to_name(label: str) -> str:
    """Convierte una etiqueta detectada a un nombre bonito."""
    # Capitalizar primera letra de cada palabra
    return " ".join(
        w[0].upper() + w[1:] if w else w
        for w in label.replace("_", " ").split()
    )[:80]


def _name_to_key(name: str) -> str:
    """Convierte un nombre a snake_case."""
    import unicodedata
    s = unicodedata.normalize("NFD", name.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s[:30] if s else "campo"


# ---------------------------------------------------------------------------
# Extractor determinista de tablas CÓDIGO → VALOR (fichas técnicas, ITV…)
# ---------------------------------------------------------------------------

# Un "código" de tabla: 1–2 letras opcionalmente seguidas de grupos ".n" o ".x"
# (ej. E, G, CL, D.1, F.2.1, P.1.1, L.2). Se permite minúscula por errores de OCR.
CODE_REGEX = re.compile(r"^[A-Za-z]{1,3}(\.[A-Za-z0-9]{1,3}){0,3}$")

# Token decorativo: solo separadores (guiones, barras, puntos suspensivos…).
_DECORATIVE_REGEX = re.compile(r"^[\s\-/_.|·•…¥*]+$")

# Palabras vacías españolas que NO son códigos aunque parezcan (1-3 letras).
# Evita que el texto en prosa/legal del documento se interprete como tabla.
_STOPWORDS = {
    "EL", "LA", "LOS", "LAS", "UN", "UNA", "UNOS", "UNAS", "DE", "DEL",
    "Y", "E", "O", "U", "EN", "A", "AL", "SE", "ES", "SU", "SUS", "LE",
    "LO", "NA", "POR", "CON", "SIN", "QUE", "NO", "SI", "ETA", "SB", "LN",
    "N", "RO", "AC",
}


def _is_code_token(text: str) -> bool:
    """True si el token parece un código de tabla (E, D.1, F.2.1…)."""
    t = text.strip()
    if not t or len(t) > 6:
        return False
    if not CODE_REGEX.match(t):
        return False
    # Debe contener al menos una letra (descarta '1.2', '3.3' que son valores).
    if not any(c.isalpha() for c in t):
        return False
    # Descarta palabras vacías (prosa/texto legal).
    return t.upper() not in _STOPWORDS


def _looks_like_prose(value: str) -> bool:
    """Heurística: un valor con 2+ palabras en minúscula parece una frase, no un dato."""
    lower_words = [
        w for w in value.split()
        if len(w) >= 3 and w[:1].isalpha() and w == w.lower()
    ]
    return len(lower_words) >= 2


def analyze_table_pairs(
    ocr_words: list[dict[str, Any]],
    existing_fields: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Extrae pares código→valor de tablas usando las cajas reales del OCR.

    Recorre cada fila de izquierda a derecha: al encontrar un token-código,
    los tokens siguientes (hasta el próximo código) forman su valor. Determinista
    e instantáneo; ideal para tablas densas donde el LLM falla.
    """
    if not ocr_words:
        return []

    existing_keys = set()
    if existing_fields:
        existing_keys = {f.get("key", "") for f in existing_fields}

    suggestions: list[dict[str, Any]] = []
    lines = _group_by_lines(ocr_words, tolerance=0.012)

    for line_words in lines:
        line_words = sorted(line_words, key=lambda w: w["box"]["x"])
        i = 0
        n = len(line_words)
        while i < n:
            if not _is_code_token(line_words[i]["text"]):
                i += 1
                continue
            code_word = line_words[i]
            code = code_word["text"].strip().upper()
            # Acumular el valor hasta el próximo código (o fin de línea).
            # Se corta en el primer token decorativo (---, /, |…) y se topa a 8
            # palabras para no absorber la columna siguiente ni prosa.
            j = i + 1
            value_words = []
            while j < n and not _is_code_token(line_words[j]["text"]):
                if _DECORATIVE_REGEX.match(line_words[j]["text"]):
                    j += 1
                    break
                value_words.append(line_words[j])
                if len(value_words) >= 8:
                    j += 1
                    break
                j += 1
            # Saltar el resto de tokens hasta el próximo código real
            while j < n and not _is_code_token(line_words[j]["text"]):
                j += 1

            value_text = " ".join(w["text"] for w in value_words).strip()
            # Descartar valores vacíos, solo decorativos o que parezcan prosa
            if (
                value_words
                and value_text
                and any(c.isalnum() for c in value_text)
                and not _looks_like_prose(value_text)
            ):
                vx = min(w["box"]["x"] for w in value_words)
                vy = min(w["box"]["y"] for w in value_words)
                vx2 = max(w["box"]["x"] + w["box"]["w"] for w in value_words)
                vy2 = max(w["box"]["y"] + w["box"]["h"] for w in value_words)
                avg_conf = sum(w["conf"] for w in value_words) / len(value_words)

                base_key = _name_to_key(code)
                key = base_key
                counter = 1
                while key in existing_keys:
                    counter += 1
                    key = f"{base_key}_{counter}"
                existing_keys.add(key)

                cb = code_word["box"]
                suggestions.append({
                    "key": key,
                    "name": code,
                    "data_type": guess_data_type(value_text),
                    "x": round(vx, 5),
                    "y": round(vy, 5),
                    "w": round(vx2 - vx, 5),
                    "h": round(vy2 - vy, 5),
                    "sample_text": value_text[:120],
                    "confidence": round(avg_conf, 1),
                    "label_text": code,
                    "label_region": {
                        "x": round(cb["x"], 5),
                        "y": round(cb["y"], 5),
                        "w": round(cb["w"], 5),
                        "h": round(cb["h"], 5),
                    },
                    "source": "table",
                })
            i = j  # continuar tras el valor consumido

    return suggestions


def _regions_overlap(a: dict, b: dict, iou_thresh: float = 0.5) -> bool:
    """True si dos regiones normalizadas se solapan por encima del umbral IoU."""
    ax2, ay2 = a["x"] + a["w"], a["y"] + a["h"]
    bx2, by2 = b["x"] + b["w"], b["y"] + b["h"]
    ix = max(0.0, min(ax2, bx2) - max(a["x"], b["x"]))
    iy = max(0.0, min(ay2, by2) - max(a["y"], b["y"]))
    inter = ix * iy
    if inter <= 0:
        return False
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return union > 0 and (inter / union) >= iou_thresh


def merge_suggestions(
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fusiona dos listas de sugerencias evitando duplicados por región/clave.

    `primary` tiene prioridad (sus campos se conservan tal cual); de `secondary`
    solo se añaden los que no solapan en región ni chocan en clave.
    """
    merged = list(primary)
    used_keys = {s["key"] for s in merged}
    for s in secondary:
        region = {"x": s["x"], "y": s["y"], "w": s["w"], "h": s["h"]}
        dup = any(
            _regions_overlap(region, {"x": m["x"], "y": m["y"], "w": m["w"], "h": m["h"]})
            for m in merged
        )
        if dup:
            continue
        key = s["key"]
        counter = 1
        while key in used_keys:
            counter += 1
            key = f"{s['key']}_{counter}"
        s = {**s, "key": key}
        used_keys.add(key)
        merged.append(s)
    return merged


# ---------------------------------------------------------------------------
# Análisis con LLM local (Ollama) — mucho más preciso que las heurísticas
# ---------------------------------------------------------------------------

def _ocr_text_ordered(words: list[dict]) -> str:
    """Texto OCR en orden de lectura aproximado (por filas, luego columnas)."""
    ordered = sorted(words, key=lambda w: (round(w["box"]["y"], 2), w["box"]["x"]))
    return " ".join(w["text"] for w in ordered)


def _ocr_text_lines(words: list[dict]) -> str:
    """Texto OCR estructurado por filas (un renglón por línea).

    Mantener la estructura de filas ayuda mucho al LLM a entender tablas:
    cada línea es una fila real del documento en vez de un chorro de palabras.
    """
    lines = _group_by_lines(words, tolerance=0.012)
    out: list[str] = []
    for line_words in lines:
        line_words = sorted(line_words, key=lambda w: w["box"]["x"])
        out.append(" ".join(w["text"] for w in line_words))
    return "\n".join(out)


def _match_value_to_words(
    value: str, ocr_words: list[dict]
) -> dict | None:
    """Busca un valor en las palabras OCR y devuelve su región normalizada.

    En lugar de coger todas las palabras dispersas que coincidan (lo que en
    documentos tipo tabla produce cajas enormes), busca el **tramo contiguo de
    palabras en una misma línea** cuya concatenación se parezca más al valor.
    Así la caja queda ajustada al valor real, no a media página.
    """
    import unicodedata
    from difflib import SequenceMatcher

    def _n(s: str) -> str:
        s = unicodedata.normalize("NFD", s.lower())
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
        return "".join(c for c in s if c.isalnum())

    target = _n(value)
    if not target or not ocr_words:
        return None

    # Palabras en orden de lectura (por fila, luego por columna)
    ordered = sorted(ocr_words, key=lambda w: (round(w["box"]["y"], 2), w["box"]["x"]))
    norm = [_n(w["text"]) for w in ordered]
    n_tokens = max(1, len([t for t in value.split() if _n(t)]))

    # Tolerancia vertical: las palabras del tramo deben estar en la misma línea
    heights = [w["box"]["h"] for w in ordered if w["box"]["h"] > 0]
    line_tol = (sorted(heights)[len(heights) // 2] * 1.2) if heights else 0.02

    best_window: list[dict] | None = None
    best_score = 0.0
    max_size = min(len(ordered), n_tokens + 2)

    for size in range(1, max_size + 1):
        for i in range(len(ordered) - size + 1):
            window = ordered[i:i + size]
            ys = [w["box"]["y"] for w in window]
            if max(ys) - min(ys) > line_tol:
                continue  # el tramo cruza varias filas: descartar
            concat = "".join(norm[i:i + size])
            if not concat:
                continue
            score = SequenceMatcher(None, concat, target).ratio()
            # Penaliza tramos mucho más largos que el valor buscado
            if len(concat) > len(target) * 2:
                score *= 0.7
            if score > best_score:
                best_score = score
                best_window = window

    if best_window is None or best_score < 0.6:
        return None

    boxes = [w["box"] for w in best_window]
    x0 = min(b["x"] for b in boxes)
    y0 = min(b["y"] for b in boxes)
    x1 = max(b["x"] + b["w"] for b in boxes)
    y1 = max(b["y"] + b["h"] for b in boxes)
    return {
        "x": round(x0, 5),
        "y": round(y0, 5),
        "w": round(x1 - x0, 5),
        "h": round(y1 - y0, 5),
    }


def analyze_words_with_ollama(
    ocr_words: list[dict[str, Any]],
    existing_fields: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Usa Ollama (LLM local) para analizar el texto OCR y sugerir campos."""
    import sys
    from . import ollama_client

    if not ollama_client.available():
        print("[field_suggestions] Ollama no disponible", file=sys.stderr)
        return []

    # Texto estructurado por filas: ayuda al modelo a leer tablas.
    text = _ocr_text_lines(ocr_words)[:3000]
    if not text.strip():
        return []

    existing_keys = set()
    if existing_fields:
        existing_keys = {f.get("key", "") for f in existing_fields}

    prompt = (
        "Eres un asistente que extrae campos de un documento español a partir de "
        "su texto OCR (una fila por línea).\n"
        "Devuelve SOLO un objeto JSON con esta forma exacta:\n"
        '{\"fields\": [{\"label\": \"...\", \"value\": \"...\", \"data_type\": \"...\"}]}\n\n'
        "Hay DOS tipos de campos que debes extraer:\n"
        "1) Pares ETIQUETA: VALOR normales (ej. 'Nº de Serie: e011278447').\n"
        "2) TABLAS de CÓDIGO + VALOR. Muchos documentos (fichas técnicas, ITV, "
        "facturas) tienen tablas donde cada fila lleva uno o VARIOS pares de "
        "código y valor. Cada código (letras/números como D.1, E, P.1, F.2.1, "
        "L.2, CL) es una etiqueta válida: extrae su valor.\n"
        "   Ejemplo: la línea 'F.1 1927 M.4 655' contiene DOS campos: "
        "F.1=1927 y M.4=655.\n\n"
        "REGLAS IMPORTANTES:\n"
        "- El \"value\" debe copiarse EXACTAMENTE como aparece en el texto, sin "
        "reescribirlo ni inventarlo (se usa para localizarlo en la imagen).\n"
        "- El \"label\" para una tabla es el propio código (ej. 'E', 'D.1', 'P.1').\n"
        "- IGNORA celdas vacías o cuyo valor sea solo guiones, barras o puntos "
        "(ej. '---', '/', '----- / -----', '...').\n"
        "- Ignora códigos de barras, sellos, firmas y párrafos largos de texto legal.\n"
        "- No inventes teléfonos, emails ni importes si no aparecen claramente.\n"
        "- data_type debe ser uno de: text, date, number, currency, dni, email, phone.\n"
        "- Máximo 25 campos, prioriza los que tengan un valor alfanumérico real.\n\n"
        "Ejemplo de salida:\n"
        '{\"fields\": [{\"label\": \"Nº de Serie\", \"value\": \"e011278447\", \"data_type\": \"text\"}, '
        '{\"label\": \"E\", \"value\": \"3MVDM6WE60E211471\", \"data_type\": \"text\"}, '
        '{\"label\": \"D.1\", \"value\": \"MAZDA\", \"data_type\": \"text\"}, '
        '{\"label\": \"P.1\", \"value\": \"1998\", \"data_type\": \"number\"}, '
        '{\"label\": \"Fecha de emisión\", \"value\": \"27/05/2021\", \"data_type\": \"date\"}]}\n\n'
        "TEXTO DEL DOCUMENTO:\n"
        f"{text}"
    )

    try:
        print(f"[field_suggestions] Llamando a Ollama con {len(text)} chars...", file=sys.stderr)
        raw = ollama_client.generate_json(prompt, force_json=True, num_predict=1536)
        print(f"[field_suggestions] Ollama respondió {len(raw)} chars", file=sys.stderr)
        # Limpiar respuesta
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines)
        parsed = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        print(f"[field_suggestions] ERROR: {exc}", file=sys.stderr)
        return []

    # El modelo puede devolver {"fields": [...]} o directamente una lista.
    if isinstance(parsed, dict):
        data = parsed.get("fields") or parsed.get("campos") or []
    elif isinstance(parsed, list):
        data = parsed
    else:
        data = []
    if not isinstance(data, list):
        return []
    print(f"[field_suggestions] JSON parseado: {len(data)} campos", file=sys.stderr)

    suggestions: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        value = str(item.get("value", "")).strip()
        data_type = str(item.get("data_type", "text")).strip()
        if not label or not value:
            continue

        # Buscar la posición del valor en las palabras OCR.
        # Si no localizamos el valor en la página, descartamos la sugerencia:
        # una caja inventada solo sería ruido que el usuario tendría que borrar.
        region = _match_value_to_words(value, ocr_words)
        if region is None:
            continue

        # Generar clave única
        base_key = _name_to_key(label)
        key = base_key
        counter = 1
        while key in existing_keys:
            counter += 1
            key = f"{base_key}_{counter}"
        existing_keys.add(key)

        suggestions.append({
            "key": key,
            "name": label[:80],
            "data_type": data_type if data_type in (
                "text", "date", "number", "currency", "dni", "email", "phone"
            ) else "text",
            "x": region["x"],
            "y": region["y"],
            "w": region["w"],
            "h": region["h"],
            "sample_text": value,
            "confidence": 85.0,  # estimación alta para LLM
            "label_text": label,
            "label_region": None,
            "source": "ollama",
        })

    return suggestions
