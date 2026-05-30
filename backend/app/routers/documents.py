"""Subida de documentos: ejecuta OCR, calcula firma y guarda la imagen normalizada."""
from __future__ import annotations

import os
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from sqlalchemy.orm import Session

from .. import models, ocr, schemas
from ..config import settings
from ..database import get_db

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("", response_model=schemas.DocumentOut)
async def upload_document(
    file: UploadFile = File(...), db: Session = Depends(get_db)
):
    content = await file.read()
    if not content:
        raise HTTPException(400, "Fichero vacío")

    try:
        result = ocr.process_upload(content, file.content_type or "")
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
