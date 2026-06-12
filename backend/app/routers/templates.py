"""CRUD de plantillas. Una plantilla se entrena a partir de un documento de muestra."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import get_db

router = APIRouter(prefix="/api/templates", tags=["templates"])


def _signature_from_sample(db: Session, sample_document_id: int | None) -> dict:
    if not sample_document_id:
        return {}
    doc = db.get(models.Document, sample_document_id)
    return doc.signature if doc else {}


def _border_from_sample(db: Session, sample_document_id: int | None) -> dict:
    if not sample_document_id:
        return {}
    doc = db.get(models.Document, sample_document_id)
    return doc.border if doc else {}


def _with_counts(db: Session, tpl: models.Template) -> models.Template:
    """Adjunta el nº de ejemplos aprendidos (muestreo) a la plantilla."""
    tpl.example_count = (
        db.query(models.LearningExample).filter_by(template_id=tpl.id).count()
    )
    return tpl


@router.get("", response_model=list[schemas.TemplateOut])
def list_templates(db: Session = Depends(get_db)):
    rows = db.query(models.Template).order_by(models.Template.created_at.desc()).all()
    return [_with_counts(db, t) for t in rows]


@router.post("", response_model=schemas.TemplateOut)
def create_template(payload: schemas.TemplateCreate, db: Session = Depends(get_db)):
    if db.query(models.Template).filter_by(name=payload.name).first():
        raise HTTPException(409, "Ya existe una plantilla con ese nombre")

    signature = payload.signature or _signature_from_sample(
        db, payload.sample_document_id
    )
    border = payload.border or _border_from_sample(db, payload.sample_document_id)
    tpl = models.Template(
        name=payload.name,
        description=payload.description,
        sample_document_id=payload.sample_document_id,
        signature=signature,
        border=border,
    )
    for f in payload.fields:
        tpl.fields.append(models.TemplateField(**f.model_dump()))
    for a in payload.anchors:
        tpl.anchors.append(models.TemplateAnchor(**a.model_dump()))
    db.add(tpl)
    db.commit()
    db.refresh(tpl)
    return _with_counts(db, tpl)


@router.get("/{template_id}", response_model=schemas.TemplateOut)
def get_template(template_id: int, db: Session = Depends(get_db)):
    tpl = db.get(models.Template, template_id)
    if not tpl:
        raise HTTPException(404, "Plantilla no encontrada")
    return _with_counts(db, tpl)


@router.put("/{template_id}", response_model=schemas.TemplateOut)
def update_template(
    template_id: int, payload: schemas.TemplateUpdate, db: Session = Depends(get_db)
):
    tpl = db.get(models.Template, template_id)
    if not tpl:
        raise HTTPException(404, "Plantilla no encontrada")

    if payload.name is not None:
        tpl.name = payload.name
    if payload.description is not None:
        tpl.description = payload.description
    if payload.signature is not None:
        tpl.signature = payload.signature
    if payload.border is not None:
        tpl.border = payload.border
    if payload.fields is not None:
        tpl.fields.clear()
        db.flush()
        for f in payload.fields:
            tpl.fields.append(models.TemplateField(**f.model_dump()))
    if payload.anchors is not None:
        tpl.anchors.clear()
        db.flush()
        for a in payload.anchors:
            tpl.anchors.append(models.TemplateAnchor(**a.model_dump()))

    db.commit()
    db.refresh(tpl)
    return _with_counts(db, tpl)


@router.delete("/{template_id}", status_code=204)
def delete_template(template_id: int, db: Session = Depends(get_db)):
    tpl = db.get(models.Template, template_id)
    if not tpl:
        raise HTTPException(404, "Plantilla no encontrada")
    if tpl.is_universal:
        raise HTTPException(400, "No se puede eliminar la plantilla universal 'datos IA'")
    # Borrado manual de todos los hijos en orden (por si las FK en la BD
    # no tienen ON DELETE CASCADE / SET NULL tras un create_all inicial).
    # Orden: records (SET NULL), learning_examples (DELETE), template_fields (DELETE)
    db.query(models.Record).filter_by(template_id=template_id).update(
        {"template_id": None}, synchronize_session="fetch"
    )
    db.query(models.LearningExample).filter_by(template_id=template_id).delete(
        synchronize_session="fetch"
    )
    db.query(models.TemplateField).filter_by(template_id=template_id).delete(
        synchronize_session="fetch"
    )
    db.query(models.TemplateAnchor).filter_by(template_id=template_id).delete(
        synchronize_session="fetch"
    )
    db.flush()
    db.delete(tpl)
    db.commit()
