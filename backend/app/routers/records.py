"""Procesado de documentos contra plantillas y persistencia de registros."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from .. import matching, models, rag, schemas
from ..config import settings
from ..database import get_db

router = APIRouter(prefix="/api", tags=["processing"])


@router.post("/process/{document_id}", response_model=schemas.MatchResult)
def process_document(
    document_id: int,
    template_id: int | None = Query(
        None, description="Forzar una plantilla; si se omite se auto-detecta"
    ),
    db: Session = Depends(get_db),
):
    """Empareja el documento con una plantilla y extrae sus campos (sin guardar nada)."""
    doc = db.get(models.Document, document_id)
    if not doc:
        raise HTTPException(404, "Documento no encontrado")

    if template_id is not None:
        tpl = db.get(models.Template, template_id)
        if not tpl:
            raise HTTPException(404, "Plantilla no encontrada")
        score = matching.similarity(doc.signature or {}, tpl.signature or {})
    else:
        templates = db.query(models.Template).all()
        tpl, score = matching.best_template(doc.signature or {}, templates)
        if tpl and score < settings.match_threshold:
            # Devolvemos la mejor pero marcamos que la confianza es baja
            pass

    fields = (
        matching.extract_all(tpl, doc.ocr_words, doc.border) if tpl else {}
    )

    return schemas.MatchResult(
        document_id=doc.id,
        template_id=tpl.id if tpl else None,
        template_name=tpl.name if tpl else None,
        match_score=score,
        fields=fields,
        width=doc.width,
        height=doc.height,
        ocr_words=doc.ocr_words,
        border=doc.border or {"x": 0, "y": 0, "w": 1, "h": 1},
    )


@router.post("/records", response_model=schemas.RecordOut)
def create_record(payload: schemas.RecordCreate, db: Session = Depends(get_db)):
    rec = models.Record(
        template_id=payload.template_id,
        document_id=payload.document_id,
        data=payload.data,
        match_score=payload.match_score,
        status=payload.status,
    )
    db.add(rec)

    # Aprendizaje (RAG): guarda el ejemplo confirmado y afina la plantilla
    if payload.learn and payload.template_id and payload.document_id:
        tpl = db.get(models.Template, payload.template_id)
        doc = db.get(models.Document, payload.document_id)
        if tpl and doc:
            try:
                rag.learn_from_record(
                    db, tpl, doc, payload.regions or {}, payload.data or {}
                )
            except Exception:  # noqa: BLE001
                pass  # el aprendizaje no debe impedir guardar el registro

    db.commit()
    db.refresh(rec)
    return rec


@router.get("/records", response_model=list[schemas.RecordOut])
def list_records(
    template_id: int | None = None, db: Session = Depends(get_db)
):
    q = db.query(models.Record)
    if template_id is not None:
        q = q.filter_by(template_id=template_id)
    return q.order_by(models.Record.created_at.desc()).all()


@router.get("/records/{record_id}", response_model=schemas.RecordOut)
def get_record(record_id: int, db: Session = Depends(get_db)):
    rec = db.get(models.Record, record_id)
    if not rec:
        raise HTTPException(404, "Registro no encontrado")
    return rec


@router.delete("/records/{record_id}", status_code=204)
def delete_record(record_id: int, db: Session = Depends(get_db)):
    rec = db.get(models.Record, record_id)
    if not rec:
        raise HTTPException(404, "Registro no encontrado")
    db.delete(rec)
    db.commit()
