"""Subida de documentos: ejecuta OCR, calcula firma y guarda la imagen normalizada."""
from __future__ import annotations

import os
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from sqlalchemy.orm import Session

from .. import anchors, field_suggestions, matching, models, ocr, preprocessing, schemas
from ..config import settings
from ..database import get_db

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("", response_model=schemas.DocumentOut)
async def upload_document(
    file: UploadFile = File(...),
    deskew: bool | None = None,
    multi_filter: bool | None = None,
    db: Session = Depends(get_db),
):
    content = await file.read()
    if not content:
        raise HTTPException(400, "Fichero vacío")

    # Usar valores de configuración si no se especifican
    do_deskew = deskew if deskew is not None else settings.auto_deskew
    do_multi = multi_filter if multi_filter is not None else settings.multi_filter_ocr

    try:
        result = ocr.process_upload(
            content, file.content_type or "",
            deskew=do_deskew,
            multi_filter=do_multi,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, f"No se pudo procesar el documento: {exc}")

    os.makedirs(settings.storage_dir, exist_ok=True)
    stored_name = f"{uuid.uuid4().hex}.png"
    stored_path = os.path.join(settings.storage_dir, stored_name)
    # Guardamos siempre como PNG normalizado (primera página) para el visor
    result["image"].save(stored_path, "PNG")

    doc = models.Document(
        filename=file.filename or stored_name,
        stored_path=stored_path,
        width=result["width"],
        height=result["height"],
        ocr_words=result["ocr_words"],
        signature=result["signature"],
        border=result["border"],
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


@router.get("/{doc_id}", response_model=schemas.DocumentOut)
def get_document(doc_id: int, db: Session = Depends(get_db)):
    doc = db.get(models.Document, doc_id)
    if not doc:
        raise HTTPException(404, "Documento no encontrado")
    return doc


@router.get("/{doc_id}/image")
def get_document_image(doc_id: int, db: Session = Depends(get_db)):
    doc = db.get(models.Document, doc_id)
    if not doc or not os.path.exists(doc.stored_path):
        raise HTTPException(404, "Imagen no encontrada")
    return FileResponse(doc.stored_path, media_type="image/png")


@router.post("/{doc_id}/ocr-region")
def ocr_region(doc_id: int, box: schemas.Box, db: Session = Depends(get_db)):
    """Re-ejecuta OCR sobre la región seleccionada (coords normalizadas 0..1)."""
    doc = db.get(models.Document, doc_id)
    if not doc or not os.path.exists(doc.stored_path):
        raise HTTPException(404, "Documento no encontrado")
    image = Image.open(doc.stored_path).convert("RGB")
    return ocr.ocr_region(image, box.model_dump())


@router.post("/{doc_id}/rotate", response_model=schemas.DocumentOut)
def rotate_document(
    doc_id: int,
    degrees: int = Query(90, description="Grados en sentido horario: 90, -90 o 180"),
    db: Session = Depends(get_db),
):
    """Rota manualmente el documento y regenera OCR, borde y firma."""
    doc = db.get(models.Document, doc_id)
    if not doc or not os.path.exists(doc.stored_path):
        raise HTTPException(404, "Documento no encontrado")

    image = Image.open(doc.stored_path).convert("RGB")
    # PIL rota en sentido antihorario; negamos para que 'degrees' sea horario
    image = image.rotate(-degrees, expand=True)
    image.save(doc.stored_path, "PNG")

    words = ocr.run_ocr(image)
    border = ocr.detect_border(image)
    doc.width, doc.height = image.size
    doc.ocr_words = words
    doc.border = border
    doc.signature = ocr.compute_signature(image, words, border)
    db.commit()
    db.refresh(doc)
    return doc


@router.post("/{doc_id}/rectify", response_model=schemas.DocumentOut)
def rectify_document(doc_id: int, quad: schemas.Quad, db: Session = Depends(get_db)):
    """Endereza el documento (warp de perspectiva con 4 puntos) y regenera OCR/firma."""
    doc = db.get(models.Document, doc_id)
    if not doc or not os.path.exists(doc.stored_path):
        raise HTTPException(404, "Documento no encontrado")

    image = Image.open(doc.stored_path).convert("RGB")
    straight = ocr.rectify(image, quad.model_dump())
    straight.save(doc.stored_path, "PNG")

    words = ocr.run_ocr(straight)
    border = dict(ocr.FULL_BORDER)  # el documento ya ocupa todo el encuadre
    doc.width, doc.height = straight.size
    doc.ocr_words = words
    doc.border = border
    doc.signature = ocr.compute_signature(straight, words, border)
    db.commit()
    db.refresh(doc)
    return doc


@router.post("/{doc_id}/detect-border", response_model=schemas.Box)
def detect_border(doc_id: int, db: Session = Depends(get_db)):
    """Re-detecta automáticamente el borde del documento."""
    doc = db.get(models.Document, doc_id)
    if not doc or not os.path.exists(doc.stored_path):
        raise HTTPException(404, "Documento no encontrado")
    image = Image.open(doc.stored_path).convert("RGB")
    border = ocr.detect_border(image)
    doc.border = border
    db.commit()
    return border


@router.post("/{doc_id}/deskew", response_model=schemas.DocumentOut)
def deskew_document(
    doc_id: int,
    fine: bool = Query(True, description="Aplicar deskew fino (Hough) además de OSD"),
    db: Session = Depends(get_db),
):
    """Endereza automáticamente el documento (OSD + Hough) y regenera OCR."""
    doc = db.get(models.Document, doc_id)
    if not doc or not os.path.exists(doc.stored_path):
        raise HTTPException(404, "Documento no encontrado")

    image = Image.open(doc.stored_path).convert("RGB")
    image, fine_angle, osd_angle = ocr.deskew_image(image, fine=fine)
    image.save(doc.stored_path, "PNG")

    words = ocr.run_ocr(image)
    border = ocr.detect_border(image)
    doc.width, doc.height = image.size
    doc.ocr_words = words
    doc.border = border
    doc.signature = ocr.compute_signature(image, words, border)
    db.commit()
    db.refresh(doc)
    return doc


@router.post("/{doc_id}/auto-orient", response_model=schemas.DocumentOut)
def auto_orient_document(
    doc_id: int,
    template_id: int | None = Query(
        None, description="Plantilla de referencia; si se omite, se prueban todas"
    ),
    fine_deskew: bool = Query(
        False,
        description="Aplicar también deskew fino (Hough). OFF por defecto: en "
        "documentos con tablas el Hough confunde las líneas y rota de más.",
    ),
    db: Session = Depends(get_db),
):
    """Endereza el documento ANTES de extraer.

    1) Corrige la orientación del texto (0/90/180/270) con Tesseract OSD, que es
       sensible a la orientación (ORB no sirve: es invariante a rotación).
    2) Aplica deskew fino (Hough) para inclinaciones pequeñas.
    3) Si se pasa una plantilla con anclas de texto, se prueba la orientación
       0/90/180/270 que mejor casa con sus anclas (refuerza/corrige al OSD).
    4) Con 2+ anclas localizadas, se deduce la inclinación fina y se endereza.
    5) Si algo cambió: re-OCR + recálculo de firma/borde y persiste.

    Idempotente: si el documento ya está derecho, no modifica nada.
    """
    doc = db.get(models.Document, doc_id)
    if not doc or not os.path.exists(doc.stored_path):
        raise HTTPException(404, "Documento no encontrado")

    tpl = db.get(models.Template, template_id) if template_id else None

    image = Image.open(doc.stored_path).convert("RGB")
    orig_size = image.size

    # 1) OSD (90° steps por contenido) + deskew fino (Hough) opcional.
    image, fine_angle, osd_angle = ocr.deskew_image(image, fine=fine_deskew)
    changed = (osd_angle % 360 != 0) or image.size != orig_size or abs(fine_angle) >= 0.1

    # 2) Orientación 90° guiada por las anclas de texto de la plantilla
    if tpl is not None and any(a.use_text and a.anchor_text for a in (tpl.anchors or [])):
        try:
            best_angle, _score = anchors.estimate_orientation(image, tpl)
            if best_angle % 360 != 0:
                image = image.rotate(-best_angle, expand=True)
                changed = True
        except Exception:  # noqa: BLE001
            pass

    # Persistir lo acumulado (re-OCR) antes del enderezado fino por anclas
    if changed:
        image.save(doc.stored_path, "PNG")
        words = ocr.run_ocr(image)
        border = ocr.detect_border(image)
        doc.width, doc.height = image.size
        doc.ocr_words = words
        doc.border = border
        doc.signature = ocr.compute_signature(image, words, border)
        db.commit()
        db.refresh(doc)

    # 3) Enderezado fino guiado por anclas (rotación deducida de 2+ anclas)
    if tpl is not None and (tpl.anchors or []):
        try:
            info = anchors.locate_anchors(db, doc, tpl, image)
            transform = anchors.estimate_transform(info["located"])
            angle = transform.get("angle", 0.0) if transform else 0.0
            if transform and 0.7 <= abs(angle) <= 20.0:
                image = Image.open(doc.stored_path).convert("RGB").rotate(
                    -angle, expand=True, fillcolor=(255, 255, 255)
                )
                image.save(doc.stored_path, "PNG")
                words = ocr.run_ocr(image)
                border = ocr.detect_border(image)
                doc.width, doc.height = image.size
                doc.ocr_words = words
                doc.border = border
                doc.signature = ocr.compute_signature(image, words, border)
                db.commit()
                db.refresh(doc)
        except Exception:  # noqa: BLE001
            pass

    return doc


@router.post("/{doc_id}/preprocess")
def preprocess_document(
    doc_id: int,
    steps: str = Query(
        "grayscale,clahe,sharpen",
        description="Filtros separados por coma: grayscale,binary_otsu,adaptive_threshold,clahe,sharpen,denoise,morphology_clean"
    ),
    db: Session = Depends(get_db),
):
    """Aplica filtros de preprocesado y devuelve la imagen resultante (base64)."""
    import base64
    import io as io_mod

    doc = db.get(models.Document, doc_id)
    if not doc or not os.path.exists(doc.stored_path):
        raise HTTPException(404, "Documento no encontrado")

    image = Image.open(doc.stored_path).convert("RGB")
    step_list = [s.strip() for s in steps.split(",") if s.strip()]
    processed, applied = preprocessing.preprocess_pipeline(image, step_list)

    buf = io_mod.BytesIO()
    processed.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "document_id": doc_id,
        "steps_applied": applied,
        "width": processed.size[0],
        "height": processed.size[1],
        "image_base64": b64,
    }


@router.post("/{doc_id}/preview-filters")
def preview_filters(
    doc_id: int,
    db: Session = Depends(get_db),
):
    """Genera todas las variantes de filtro y las devuelve en base64 para comparar."""
    import base64
    import io as io_mod

    doc = db.get(models.Document, doc_id)
    if not doc or not os.path.exists(doc.stored_path):
        raise HTTPException(404, "Documento no encontrado")

    image = Image.open(doc.stored_path).convert("RGB")
    variants = preprocessing.generate_filtered_variants(image)

    result: dict[str, dict] = {}
    for name, variant in variants.items():
        buf = io_mod.BytesIO()
        variant.save(buf, format="PNG")
        result[name] = {
            "width": variant.size[0],
            "height": variant.size[1],
            "image_base64": base64.b64encode(buf.getvalue()).decode(),
        }

    # Incluir también métricas de calidad de la imagen original
    quality = preprocessing.evaluate_image_quality(image)

    return {"document_id": doc_id, "variants": result, "quality": quality}


@router.post("/{doc_id}/detect-angle")
def detect_angle(doc_id: int, db: Session = Depends(get_db)):
    """Detecta el ángulo de rotación del documento (Hough + minAreaRect)."""
    doc = db.get(models.Document, doc_id)
    if not doc or not os.path.exists(doc.stored_path):
        raise HTTPException(404, "Documento no encontrado")

    image = Image.open(doc.stored_path).convert("RGB")
    angle = preprocessing.detect_rotation_angle(image)
    quality = preprocessing.evaluate_image_quality(image)

    return {"document_id": doc_id, "rotation_angle": angle, "quality": quality}


@router.post("/{doc_id}/detect-zones")
def detect_zones(
    doc_id: int,
    template_id: int = Query(..., description="ID de la plantilla para detectar sus zonas"),
    method: str = Query("orb", description="Método: orb, template"),
    db: Session = Depends(get_db),
):
    """Detecta las zonas de la plantilla en el documento usando ORB o template matching."""
    from .. import models as doc_models

    doc = db.get(models.Document, doc_id)
    if not doc or not os.path.exists(doc.stored_path):
        raise HTTPException(404, "Documento no encontrado")

    tpl = db.get(doc_models.Template, template_id)
    if not tpl:
        raise HTTPException(404, "Plantilla no encontrada")

    if not tpl.sample_document_id:
        raise HTTPException(400, "La plantilla no tiene documento de muestra")

    sample_doc = db.get(models.Document, tpl.sample_document_id)
    if not sample_doc or not os.path.exists(sample_doc.stored_path):
        raise HTTPException(404, "Documento de muestra no encontrado")

    doc_img = Image.open(doc.stored_path).convert("RGB")
    tpl_img = Image.open(sample_doc.stored_path).convert("RGB")

    if method == "orb":
        result = preprocessing.detect_zones_orb(doc_img, tpl_img)
    else:
        result = preprocessing.detect_template_zones(doc_img, tpl_img)
        result = {
            "found": result["found"],
            "score": result["score"],
            "region": result["region"],
            "method": "template",
        }

    return {"document_id": doc_id, "template_id": template_id, **result}


@router.post("/{doc_id}/try-orientations")
def try_orientations(
    doc_id: int,
    template_id: int | None = Query(None, help="Plantilla específica; si no, auto-detecta"),
    db: Session = Depends(get_db),
):
    """Prueba el matching en 0°, 90°, 180°, 270° y devuelve el mejor ángulo."""
    from .. import models as doc_models

    doc = db.get(models.Document, doc_id)
    if not doc or not os.path.exists(doc.stored_path):
        raise HTTPException(404, "Documento no encontrado")

    templates = db.query(doc_models.Template).all()
    if not templates:
        raise HTTPException(404, "No hay plantillas registradas")

    if template_id:
        tpl = db.get(doc_models.Template, template_id)
        if not tpl:
            raise HTTPException(404, "Plantilla no encontrada")
        candidates = [tpl]
    else:
        candidates = list(templates)

    best_tpl, best_score, best_angle, _info = matching.best_template_multi_angle(
        doc.signature or {}, candidates, doc.stored_path,
    )

    return {
        "document_id": doc_id,
        "best_template_id": best_tpl.id if best_tpl else None,
        "best_template_name": best_tpl.name if best_tpl else None,
        "best_score": best_score,
        "best_angle": best_angle,
        "angles_tested": [0, 90, 180, 270],
    }


@router.post("/{doc_id}/suggest-fields")
def suggest_fields(
    doc_id: int,
    use_ai: bool = Query(True, description="Usar IA (Ollama) para sugerencias más precisas"),
    db: Session = Depends(get_db),
):
    """Analiza el OCR del documento y sugiere campos de formulario automáticamente.

    - use_ai=true (default): usa Ollama (LLM local) si está disponible.
      Más preciso pero tarda ~5-15s la primera vez.
    - use_ai=false: solo heurísticas (keywords + patrones), instantáneo.
    """
    doc = db.get(models.Document, doc_id)
    if not doc or not doc.ocr_words:
        raise HTTPException(404, "Documento no encontrado o sin OCR")

    source = "heuristic"
    suggestions = []

    if use_ai:
        suggestions = field_suggestions.analyze_words_with_ollama(doc.ocr_words)
        if suggestions:
            source = "ollama"

    if not suggestions:
        suggestions = field_suggestions.analyze_words(doc.ocr_words)

    # Extractor determinista de tablas código→valor (fichas técnicas, ITV…).
    # Se fusiona con lo anterior; el LLM/heurística tiene prioridad en solapes.
    table_fields = field_suggestions.analyze_table_pairs(
        doc.ocr_words, existing_fields=suggestions
    )
    if table_fields:
        suggestions = field_suggestions.merge_suggestions(suggestions, table_fields)
        source = f"{source}+table"

    return {
        "document_id": doc_id,
        "total_suggestions": len(suggestions),
        "source": source,
        "fields": suggestions,
    }


@router.put("/{doc_id}/border", response_model=schemas.Box)
def update_border(doc_id: int, box: schemas.Box, db: Session = Depends(get_db)):
    """Guarda el borde editado manualmente y recalcula la firma con ese borde."""
    doc = db.get(models.Document, doc_id)
    if not doc or not os.path.exists(doc.stored_path):
        raise HTTPException(404, "Documento no encontrado")
    border = box.model_dump()
    doc.border = border
    image = Image.open(doc.stored_path).convert("RGB")
    doc.signature = ocr.compute_signature(image, doc.ocr_words, border)
    db.commit()
    return border
