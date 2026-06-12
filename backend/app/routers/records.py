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

from .. import anchors, field_suggestions, matching, models, ocr, preprocessing, rag, schemas
from ..config import settings
from ..database import SessionLocal, get_db

router = APIRouter(prefix="/api", tags=["processing"])


def _reocr_persist(db, doc, image) -> None:
    """Guarda la imagen, re-ejecuta OCR y recalcula borde/firma. Persiste en BD."""
    image.save(doc.stored_path, "PNG")
    words = ocr.run_ocr(image)
    border = ocr.detect_border(image)
    doc.width, doc.height = image.size
    doc.ocr_words = words
    doc.border = border
    doc.signature = ocr.compute_signature(image, words, border)
    db.commit()
    db.refresh(doc)


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
            changed = txt != old_val
            fields[k]["value"] = txt
            fields[k]["confidence"] = new_conf
            # Si el valor cambió respecto al OCR de página, la matched_box (calculada
            # sobre el texto antiguo) ya no es válida -> usar la región de búsqueda,
            # que cubre la celda completa del campo en la imagen rectificada.
            if changed:
                fields[k]["matched_box"] = fields[k].get("region")


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

    # -------------------------------------------------------------------
    # Si el score está por debajo del umbral y no se forzó plantilla,
    # descartamos la asignación. Además, si NO hubo confirmación visual
    # (ORB), la firma geométrica sola NO es fiable: dos documentos A4
    # cualesquiera pueden puntuar >0.55 solo por aspect ratio y color.
    # Exigimos confirmación visual para aceptar un match automático.
    # -------------------------------------------------------------------
    if template_id is None and tpl is not None:
        has_visual = vis_score > 0.05  # ORB encontró keypoints comunes reales
        reject = (
            (not has_visual and score < 0.85)
            or (has_visual and score < settings.match_threshold)
        )
        if reject:
            # No se parece a ninguna plantilla guardada → usar la universal "datos IA"
            tpl = db.query(models.Template).filter_by(is_universal=True).first()
            score = 0.0
            vis_score = 0.0
            anchor_info = None

    # -------------------------------------------------------------------
    # Rectificación a la plantilla por anclas: enderezar la IMAGEN (no las
    # coordenadas). Tras rectificar, los campos son proporcionalidad simple.
    # -------------------------------------------------------------------
    aligned = False
    needs_review = False
    anchor_public = None
    anchor_score = 0.0
    border_out = doc.border or dict(matching.FULL_BORDER)
    pipeline: list[str] = []  # traza legible de lo que hizo el pipeline

    has_anchors = bool(tpl and getattr(tpl, "anchors", []) and tpl.sample_document_id)
    if has_anchors and os.path.exists(doc.stored_path):
        try:
            image = Image.open(doc.stored_path).convert("RGB")
            loc = anchors.locate_anchors_robust(db, doc, tpl, image)
            anchor_public = anchors.public_anchors(loc["located"])
            anchor_score = loc["score"]

            if not anchors.required_ok(loc["located"]):
                # Faltan anclas obligatorias: no tocar la imagen, marcar revisión
                needs_review = True
                pipeline.append("⚠ Anclas obligatorias no localizadas")
            else:
                rect_info: dict = {}
                rect, ok = anchors.rectify_to_template(
                    loc["located"], loc["image"], loc.get("sample_size"),
                    loc.get("sample_img"), tpl, rect_info,
                )
                if ok and rect is not None:
                    # Persistir la imagen rectificada: el doc queda encajado a la
                    # plantilla -> borde = borde de la plantilla, extracción proporcional.
                    _reocr_persist(db, doc, rect)
                    doc.border = tpl.border or dict(matching.FULL_BORDER)
                    db.commit()
                    db.refresh(doc)
                    border_out = doc.border
                    aligned = True
                    # Construir la traza legible del pipeline
                    pre = rect_info.get("prerotate", 0)
                    rot = rect_info.get("rotation", 0.0)
                    total = (pre + rot) % 360
                    if total > 180:
                        total -= 360
                    if abs(total) >= 1:
                        pipeline.append(f"Girado/enderezado {total:+.0f}°")
                    sc = rect_info.get("scale")
                    if sc:
                        pipeline.append(f"Escalado ×{sc:.2f}")
                    pipeline.append("Alineado a la plantilla")
                else:
                    needs_review = True
                    pipeline.append("⚠ No se pudo alinear (geometría no plausible)")
        except Exception as exc:  # noqa: BLE001
            print(f"[match] rectificación por anclas falló: {exc}", file=sys.stderr)
            needs_review = True

    # Extracción por PROPORCIONALIDAD simple sobre el borde (la imagen ya está
    # rectificada al marco de la plantilla si aligned=True).
    extract_border = tpl.border if (tpl and aligned) else doc.border
    fields = matching.extract_all(tpl, doc.ocr_words, extract_border) if tpl else {}

    # Re-OCR de alta resolución de los campos flojos (mismo OCR del botón ↻).
    _refine_fields_with_region_ocr(doc, fields)

    return {
        "template": tpl,
        "template_id": tpl.id if tpl else None,
        "template_name": tpl.name if tpl else None,
        "score": score,
        "vis_score": vis_score,
        "zone": None,
        "anchors": anchor_public,
        "anchor_score": anchor_score,
        "aligned": aligned,
        "needs_review": needs_review,
        "aligned_border": border_out,
        "pipeline": pipeline,
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
        needs_review=res.get("needs_review", False),
        aligned=res.get("aligned", False),
        pipeline=res.get("pipeline", []),
        sample_document_id=res["template"].sample_document_id if res["template"] else None,
        is_universal=bool(res["template"] and getattr(res["template"], "is_universal", False)),
        fields=res["fields"],
        width=doc.width,
        height=doc.height,
        ocr_words=doc.ocr_words,
        # El doc ya está rectificado al marco de la plantilla cuando aligned=True
        border=res.get("aligned_border")
        or doc.border
        or {"x": 0, "y": 0, "w": 1, "h": 1},
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

        # 3) Emparejado + rectificación por anclas + extracción (todo en uno).
        #    _match_and_extract endereza la imagen al marco de la plantilla cuando
        #    localiza las anclas (incl. obligatorias) y extrae por proporcionalidad.
        m = _match_and_extract(db, doc, template_id)

        values = {k: (v.get("value", "") or "") for k, v in m["fields"].items()}
        confs = [v.get("confidence", 0) for v in m["fields"].values() if v.get("value")]
        avg_conf = sum(confs) / len(confs) if confs else 0.0

        # 4) Extracción con IA (LLM + RAG). Si la plantilla es la universal
        #    ("datos IA"), usamos extracción libre. Si es una plantilla normal,
        #    extraemos sus campos con few-shot. Si el LLM no está disponible o
        #    falla, usamos field_suggestions como fallback determinista.
        if m["template"] is not None:
            is_universal = getattr(m["template"], "is_universal", False)
            if is_universal:
                # Plantilla universal: IA libre (sin campos predefinidos)
                ai_fields = {}
                try:
                    ai = rag.extract_freeform(db, doc)
                    if ai.get("available") and ai.get("fields"):
                        tipo = ai.get("tipo_documento", "desconocido")
                        values["_tipo_documento"] = tipo
                        ai_fields = ai["fields"]
                    else:
                        print(f"[job {record_id}] IA libre no disponible, usando field_suggestions",
                              file=sys.stderr)
                except Exception as exc:
                    print(f"[job {record_id}] IA libre falló: {exc}", file=sys.stderr)

                # Fallback: si la IA no devolvió nada, usar field_suggestions (reglas)
                if not ai_fields:
                    try:
                        suggestions = field_suggestions.analyze_words(doc.ocr_words or [])
                        for s in suggestions:
                            key = s["key"]
                            val = s.get("sample_text", "")
                            if val:
                                ai_fields[key] = {"value": val, "region": s}
                        if ai_fields:
                            values["_tipo_documento"] = values.get("_tipo_documento", "detectado")
                            print(f"[job {record_id}] field_suggestions extrajo {len(ai_fields)} campos",
                                  file=sys.stderr)
                    except Exception as exc:
                        print(f"[job {record_id}] field_suggestions falló: {exc}", file=sys.stderr)

                for k, fv in ai_fields.items():
                    val = fv.get("value", "") if isinstance(fv, dict) else str(fv)
                    if val:
                        values[k] = val
            else:
                try:
                    ai = rag.extract(db, doc, m["template"])
                    if ai.get("available") and ai.get("fields"):
                        for k, fv in ai["fields"].items():
                            if fv.get("value"):
                                values[k] = fv["value"]
                except Exception as exc:  # noqa: BLE001
                    print(f"[job {record_id}] IA falló: {exc}", file=sys.stderr)

        # 5) Decidir estado: revisar si faltan anclas obligatorias, baja similitud,
        #    sin valores, o confianza media baja. Si no hay plantilla pero la IA
        #    extrajo datos en modo libre, se considera válido.
        has_values = any(
            str(v).strip()
            for k, v in values.items()
            if not k.startswith("_")  # _tipo_documento no cuenta para "sin valores"
        )
        needs_review = (
            m.get("needs_review", False)
            or (m["template_id"] is not None and m["score"] < settings.match_threshold)
            or not has_values
            or (m["template_id"] is not None and avg_conf < 40)
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
            "tipo_documento": (r.data or {}).get("_tipo_documento", ""),
            "status": r.status,
            "match_score": r.match_score,
            "n_fields": sum(1 for k in (r.data or {}) if not k.startswith("_")),
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
