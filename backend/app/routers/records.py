"""Procesado de documentos contra plantillas y persistencia de registros."""
from __future__ import annotations

import os
import sys
import uuid
from typing import Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
)
from PIL import Image
from sqlalchemy.orm import Session

from .. import anchors, matching, models, ocr, preprocessing, rag, schemas
from ..config import settings
from ..database import SessionLocal, get_db

router = APIRouter(prefix="/api", tags=["processing"])


# ---------------------------------------------------------------------------
# Lógica reutilizable de emparejado + extracción (la usan el endpoint /process
# y la tarea de fondo de /jobs).
# ---------------------------------------------------------------------------

def _apply_anchor_selection(
    db: Session,
    doc: models.Document,
    templates: list,
    chosen,
    chosen_score: float,
) -> tuple[Any, float, dict | None]:
    """Refina la elección de plantilla usando las anclas (hitos).

    Para cada plantilla con anclas combina su score base con el score de anclas
    (peso settings.anchor_match_weight). Si la mejor combinada supera a la elegida
    geométrica/visualmente, la sustituye. Devuelve (tpl, score, anchor_info|None)
    donde anchor_info es la localización de anclas de la plantilla final.
    """
    anchored = [t for t in (templates or []) if getattr(t, "anchors", [])]
    if not anchored:
        return chosen, chosen_score, None

    w = settings.anchor_match_weight
    doc_img = None
    if doc and os.path.exists(doc.stored_path):
        try:
            doc_img = Image.open(doc.stored_path).convert("RGB")
        except Exception:  # noqa: BLE001
            doc_img = None

    chosen_id = chosen.id if chosen else None
    best_tpl, best_combined, best_info = chosen, chosen_score, None
    for t in anchored:
        info = anchors.locate_anchors(db, doc, t, doc_img)
        base = chosen_score if t.id == chosen_id else matching.similarity(
            doc.signature or {}, t.signature or {}
        )
        combined = round((1 - w) * base + w * info["score"], 4)
        if t.id == chosen_id:
            best_info = info  # info de la elegida por defecto
        if combined > best_combined:
            best_combined, best_tpl, best_info = combined, t, info

    return best_tpl, best_combined, best_info


def _extract_fields(
    tpl,
    doc: models.Document,
    transform: dict | None,
    tpl_dims: tuple | None = None,
) -> dict:
    """Extrae los campos del documento. Si hay transformación de anclas, mapea las
    regiones con ella (muestra->documento); si no, usa el mapeo por borde habitual."""
    if not tpl:
        return {}
    if not transform:
        return matching.extract_all(tpl, doc.ocr_words, doc.border)

    tpl_w, tpl_h = tpl_dims or (None, None)
    result: dict[str, Any] = {}
    for f in tpl.fields:
        rel = {"x": f.x, "y": f.y, "w": f.w, "h": f.h}
        region = anchors.apply_transform_to_region(
            rel, transform, tpl.border, tpl_w, tpl_h, doc.width, doc.height
        )
        extracted = matching.extract_field(region, doc.ocr_words)
        result[f.key] = {
            "name": f.name,
            "data_type": f.data_type,
            "region": region,
            **extracted,
        }
    return result


def _refine_fields_with_region_ocr(doc: models.Document, fields: dict) -> None:
    """Re-OCR de alta resolución (recorte+ampliación, PSM 6) para campos vacíos o de
    baja confianza. Mejora los valores que el OCR de página completa lee mal. Modifica
    `fields` in-place. Lo usan tanto el procesado interactivo como el job de fondo."""
    if not settings.region_ocr_refine or not fields:
        return
    weak = [
        k
        for k, v in fields.items()
        if v.get("region")
        and (
            not (v.get("value") or "").strip()
            or v.get("confidence", 0) < settings.region_ocr_min_conf
        )
    ]
    if not weak or not os.path.exists(doc.stored_path):
        return
    try:
        image = Image.open(doc.stored_path).convert("RGB")
    except Exception:  # noqa: BLE001
        return
    for k in weak:
        try:
            res = ocr.ocr_region(image, fields[k]["region"])
        except Exception:  # noqa: BLE001
            continue
        txt = (res.get("text") or "").strip()
        if not txt:
            continue
        old_val = (fields[k].get("value") or "").strip()
        old_conf = fields[k].get("confidence", 0) or 0
        new_conf = res.get("confidence", 0) or 0
        # Sustituye solo si mejora: campo vacío, o el re-OCR no es menos fiable
        if not old_val or new_conf >= old_conf:
            fields[k]["value"] = txt
            fields[k]["confidence"] = new_conf


def _match_and_extract(
    db: Session,
    doc: models.Document,
    template_id: int | None = None,
    multi_angle: bool = True,
) -> dict:
    """Empareja el documento con una plantilla y extrae sus campos.

    Devuelve {template, template_id, template_name, score, vis_score, zone,
    anchors, anchor_score, fields}.
    """
    best_angle = 0
    vis_score = 0.0
    tpl = None
    score = 0.0
    anchor_info = None

    if template_id is not None:
        tpl = db.get(models.Template, template_id)
        if not tpl:
            raise HTTPException(404, "Plantilla no encontrada")
        geom_score = matching.similarity(doc.signature or {}, tpl.signature or {})
        if tpl.sample_document_id:
            sample = db.get(models.Document, tpl.sample_document_id)
            if sample and os.path.exists(sample.stored_path):
                vis_score = preprocessing.visual_similarity_score(
                    doc.stored_path, sample.stored_path,
                )
        score = matching.combined_similarity(
            geom_score, vis_score, settings.visual_match_weight,
        )
        if getattr(tpl, "anchors", []):
            anchor_info = anchors.locate_anchors(db, doc, tpl)
    else:
        templates = db.query(models.Template).all()
        if multi_angle and templates:
            tpl, score, best_angle, info = matching.best_template_multi_angle(
                doc.signature or {}, templates, doc.stored_path,
            )
            vis_score = info.get("vis_score", 0.0)
        elif templates:
            tpl, score, _geom, _vis = matching.best_template(
                doc.signature or {}, templates, doc.stored_path,
            )
            vis_score = _vis
        # Las anclas pueden corregir la elección de plantilla
        tpl, score, anchor_info = _apply_anchor_selection(
            db, doc, templates, tpl, score
        )

    # Transformación de alineación a partir de las anclas localizadas
    transform = (
        anchors.estimate_transform(anchor_info["located"]) if anchor_info else None
    )
    tpl_dims = (
        (anchor_info.get("tpl_w"), anchor_info.get("tpl_h")) if anchor_info else None
    )

    fields = _extract_fields(tpl, doc, transform, tpl_dims)

    # Detección de zona parcial (plantilla más pequeña que el documento). Solo si
    # las anclas no han proporcionado ya una alineación.
    zone = None
    if tpl and tpl.sample_document_id and transform is None:
        try:
            sample = db.get(models.Document, tpl.sample_document_id)
            if (
                sample
                and os.path.exists(sample.stored_path)
                and os.path.exists(doc.stored_path)
            ):
                doc_img = Image.open(doc.stored_path).convert("RGB")
                sample_img = Image.open(sample.stored_path).convert("RGB")
                dw, dh = doc_img.size
                sw, sh = sample_img.size
                if sw < dw * 0.85 or sh < dh * 0.85:
                    zone = preprocessing.detect_best_zone(doc_img, sample_img)
                    if zone.get("found") and zone.get("region"):
                        zr = zone["region"]
                        for key, fdata in fields.items():
                            if fdata.get("region"):
                                old_r = fdata["region"]
                                fdata["region"] = {
                                    "x": round(zr["x"] + old_r["x"] * zr["w"], 5),
                                    "y": round(zr["y"] + old_r["y"] * zr["h"], 5),
                                    "w": round(old_r["w"] * zr["w"], 5),
                                    "h": round(old_r["h"] * zr["h"], 5),
                                }
                                extracted = matching.extract_field(
                                    fdata["region"], doc.ocr_words
                                )
                                fdata["value"] = extracted["value"]
                                fdata["confidence"] = extracted["confidence"]
                                fdata["n_words"] = extracted["n_words"]
        except Exception:  # noqa: BLE001
            zone = None

    # Re-OCR de alta resolución de los campos flojos (mismo OCR del botón ↻).
    # Aplica al procesado interactivo y al job de fondo (ambos pasan por aquí).
    _refine_fields_with_region_ocr(doc, fields)

    return {
        "template": tpl,
        "template_id": tpl.id if tpl else None,
        "template_name": tpl.name if tpl else None,
        "score": score,
        "vis_score": vis_score,
        "zone": zone,
        "anchors": anchors.public_anchors(anchor_info["located"]) if anchor_info else None,
        "anchor_score": anchor_info["score"] if anchor_info else 0.0,
        "fields": fields,
    }


@router.post("/process/{document_id}", response_model=schemas.MatchResult)
def process_document(
    document_id: int,
    template_id: int | None = Query(
        None, description="Forzar una plantilla; si se omite se auto-detecta"
    ),
    multi_angle: bool = Query(
        True, description="Probar matching en 0°, 90°, 180°, 270°"
    ),
    db: Session = Depends(get_db),
):
    """Empareja el documento con una plantilla y extrae sus campos (sin guardar nada).

    Usa similitud combinada: geométrica (firma) + visual (ORB multi-ángulo y multi-filtro
    sobre la imagen de muestra de la plantilla)."""
    doc = db.get(models.Document, document_id)
    if not doc:
        raise HTTPException(404, "Documento no encontrado")

    res = _match_and_extract(db, doc, template_id, multi_angle)

    return schemas.MatchResult(
        document_id=doc.id,
        template_id=res["template_id"],
        template_name=res["template_name"],
        match_score=res["score"],
        visual_score=res["vis_score"],
        zone=res["zone"],
        anchors=res["anchors"],
        anchor_score=res["anchor_score"],
        fields=res["fields"],
        width=doc.width,
        height=doc.height,
        ocr_words=doc.ocr_words,
        border=doc.border or {"x": 0, "y": 0, "w": 1, "h": 1},
    )


# ---------------------------------------------------------------------------
# Cola de procesado en segundo plano (jobs)
# ---------------------------------------------------------------------------

def _run_job(
    record_id: int,
    document_id: int,
    raw_path: str,
    content_type: str,
    template_id: int | None,
) -> None:
    """Tarea de fondo: OCR + auto-orientación + match + IA. Actualiza el Record."""
    db = SessionLocal()
    try:
        rec = db.get(models.Record, record_id)
        doc = db.get(models.Document, document_id)
        if not rec or not doc:
            return

        # 1) OCR del fichero subido
        with open(raw_path, "rb") as fh:
            content = fh.read()
        result = ocr.process_upload(
            content, content_type,
            deskew=settings.auto_deskew, multi_filter=settings.multi_filter_ocr,
        )
        result["image"].save(doc.stored_path, "PNG")
        doc.width, doc.height = result["width"], result["height"]
        doc.ocr_words = result["ocr_words"]
        doc.signature = result["signature"]
        doc.border = result["border"]
        db.commit()

        # 2) Auto-orientación (OSD, sin deskew fino para no romper tablas)
        image = Image.open(doc.stored_path).convert("RGB")
        image, _fine, osd_angle = ocr.deskew_image(image, fine=False)
        if osd_angle % 360 != 0:
            image.save(doc.stored_path, "PNG")
            words = ocr.run_ocr(image)
            border = ocr.detect_border(image)
            doc.width, doc.height = image.size
            doc.ocr_words = words
            doc.border = border
            doc.signature = ocr.compute_signature(image, words, border)
            db.commit()

        # 3) Emparejado + extracción geométrica
        m = _match_and_extract(db, doc, template_id)

        # 3b) Orientación guiada por anclas de texto: si la plantilla tiene anclas
        #     pero casan mal, probablemente el documento esté girado. Probamos
        #     0/90/180/270 con las anclas y re-procesamos en el mejor ángulo.
        tpl_obj = m["template"]
        if (
            tpl_obj is not None
            and m.get("anchor_score", 0.0) < 0.5
            and any(a.use_text and a.anchor_text for a in (tpl_obj.anchors or []))
        ):
            try:
                image = Image.open(doc.stored_path).convert("RGB")
                best_angle, a_score = anchors.estimate_orientation(image, tpl_obj)
                if best_angle % 360 != 0 and a_score > m.get("anchor_score", 0.0):
                    image = image.rotate(-best_angle, expand=True)
                    image.save(doc.stored_path, "PNG")
                    words = ocr.run_ocr(image)
                    border = ocr.detect_border(image)
                    doc.width, doc.height = image.size
                    doc.ocr_words = words
                    doc.border = border
                    doc.signature = ocr.compute_signature(image, words, border)
                    db.commit()
                    m = _match_and_extract(db, doc, tpl_obj.id)
            except Exception as exc:  # noqa: BLE001
                print(f"[job {record_id}] reorientación por anclas falló: {exc}", file=sys.stderr)

        values = {k: (v.get("value", "") or "") for k, v in m["fields"].items()}
        confs = [v.get("confidence", 0) for v in m["fields"].values() if v.get("value")]
        avg_conf = sum(confs) / len(confs) if confs else 0.0

        # 4) Extracción con IA (LLM + RAG) si hay plantilla y está disponible
        if m["template"] is not None:
            try:
                ai = rag.extract(db, doc, m["template"])
                if ai.get("available") and ai.get("fields"):
                    for k, fv in ai["fields"].items():
                        if fv.get("value"):
                            values[k] = fv["value"]
            except Exception as exc:  # noqa: BLE001
                print(f"[job {record_id}] IA falló: {exc}", file=sys.stderr)

        # 5) Decidir estado: revisar si no hay plantilla, baja similitud,
        #    sin valores, o confianza media baja
        has_values = any(str(v).strip() for v in values.values())
        needs_review = (
            m["template_id"] is None
            or m["score"] < settings.match_threshold
            or not has_values
            or avg_conf < 40
        )
        rec.template_id = m["template_id"]
        rec.match_score = round(float(m["score"]), 4)
        rec.data = values
        rec.status = "review" if needs_review else "done"
        db.commit()
    except Exception as exc:  # noqa: BLE001
        print(f"[job {record_id}] ERROR: {exc}", file=sys.stderr)
        try:
            rec = db.get(models.Record, record_id)
            if rec:
                rec.status = "error"
                rec.data = {"_error": str(exc)[:300]}
                db.commit()
        except Exception:  # noqa: BLE001
            pass
    finally:
        try:
            os.remove(raw_path)
        except Exception:  # noqa: BLE001
            pass
        db.close()


@router.post("/jobs")
async def create_job(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    template_id: int | None = Query(None, description="Forzar plantilla; si no, auto-detecta"),
    db: Session = Depends(get_db),
):
    """Sube un documento y lo procesa en segundo plano (OCR + IA).

    Devuelve de inmediato el registro en estado 'processing'. El frontend
    sondea GET /api/jobs para ver el progreso.
    """
    content = await file.read()
    if not content:
        raise HTTPException(400, "Fichero vacío")

    os.makedirs(settings.storage_dir, exist_ok=True)
    stored_name = f"{uuid.uuid4().hex}.png"
    stored_path = os.path.join(settings.storage_dir, stored_name)
    raw_path = stored_path + ".raw"
    with open(raw_path, "wb") as fh:
        fh.write(content)

    doc = models.Document(
        filename=file.filename or stored_name,
        stored_path=stored_path,
        width=0, height=0, ocr_words=[], signature={}, border={},
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    rec = models.Record(
        template_id=template_id, document_id=doc.id,
        data={}, match_score=0.0, status="processing",
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)

    background.add_task(
        _run_job, rec.id, doc.id, raw_path, file.content_type or "", template_id,
    )
    return {
        "record_id": rec.id,
        "document_id": doc.id,
        "filename": doc.filename,
        "status": "processing",
    }


@router.get("/jobs")
def list_jobs(db: Session = Depends(get_db)):
    """Lista de documentos procesados/en proceso, con estado para la cola."""
    recs = db.query(models.Record).order_by(models.Record.created_at.desc()).all()
    tpl_names = {t.id: t.name for t in db.query(models.Template).all()}
    doc_names = {d.id: d.filename for d in db.query(models.Document).all()}
    return [
        {
            "record_id": r.id,
            "document_id": r.document_id,
            "filename": doc_names.get(r.document_id, "—"),
            "template_id": r.template_id,
            "template_name": tpl_names.get(r.template_id),
            "status": r.status,
            "match_score": r.match_score,
            "n_fields": len(r.data or {}),
            "created_at": r.created_at,
        }
        for r in recs
    ]


@router.put("/records/{record_id}", response_model=schemas.RecordOut)
def update_record(
    record_id: int, payload: schemas.RecordCreate, db: Session = Depends(get_db)
):
    """Actualiza un registro tras la revisión manual (datos + estado)."""
    rec = db.get(models.Record, record_id)
    if not rec:
        raise HTTPException(404, "Registro no encontrado")
    rec.data = payload.data
    rec.status = payload.status
    if payload.template_id is not None:
        rec.template_id = payload.template_id
    rec.match_score = payload.match_score or rec.match_score

    if payload.learn and rec.template_id and rec.document_id:
        tpl = db.get(models.Template, rec.template_id)
        doc = db.get(models.Document, rec.document_id)
        if tpl and doc:
            try:
                rag.learn_from_record(
                    db, tpl, doc, payload.regions or {}, payload.data or {}
                )
            except Exception:  # noqa: BLE001
                pass

    db.commit()
    db.refresh(rec)
    return rec


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
