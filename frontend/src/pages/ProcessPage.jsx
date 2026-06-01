import { useEffect, useRef, useState } from "react";
import { api } from "../api.js";
import DocumentViewer from "../components/DocumentViewer.jsx";
import Uploader from "../components/Uploader.jsx";

const toKey = (s) =>
  s
    .toLowerCase()
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_|_$/g, "");

// Metadatos visuales de cada estado de la cola
const STATUS = {
  processing: { label: "Procesando (OCR + IA)…", cls: "badge proc", spin: true },
  review: { label: "⚠ Revisar", cls: "badge warn" },
  done: { label: "✓ Listo", cls: "badge ok" },
  confirmed: { label: "✓ Confirmado", cls: "badge ok" },
  error: { label: "✗ Error", cls: "badge danger" },
};

export default function ProcessPage() {
  const [activeRecord, setActiveRecord] = useState(null); // record abierto en el editor

  if (activeRecord) {
    return (
      <DocumentEditor
        record={activeRecord}
        onBack={() => setActiveRecord(null)}
      />
    );
  }
  return <JobList onOpen={setActiveRecord} />;
}

// ===========================================================================
// Vista LISTA: cola de documentos con estado, subida y sondeo en 2.º plano
// ===========================================================================
function JobList({ onOpen }) {
  const [templates, setTemplates] = useState([]);
  const [forcedTemplate, setForcedTemplate] = useState("");
  const [jobs, setJobs] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [modal, setModal] = useState(null); // { job, data }
  const pollRef = useRef(null);

  // Nombre legible de un campo: el de la plantilla si existe, si no la clave
  function fieldName(job, key) {
    const tpl = templates.find((t) => t.id === job.template_id);
    const f = tpl?.fields?.find((ff) => ff.key === key);
    return f?.name || key;
  }

  async function openData(job, e) {
    e.stopPropagation();
    try {
      const rec = await api.getRecord(job.record_id);
      setModal({ job, data: rec.data || {} });
    } catch (err) {
      setError(err.message);
    }
  }

  async function loadJobs() {
    try {
      const j = await api.listJobs();
      setJobs(j);
    } catch {
      /* silencioso: reintenta en el siguiente tick */
    }
  }

  useEffect(() => {
    api.listTemplates().then(setTemplates).catch(() => {});
    loadJobs();
    pollRef.current = setInterval(loadJobs, 2500);
    return () => clearInterval(pollRef.current);
  }, []);

  async function handleUpload(file) {
    if (!file) return;
    setUploading(true);
    setError("");
    try {
      await api.createJob(file, forcedTemplate ? Number(forcedTemplate) : undefined);
      await loadJobs();
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  }

  async function openJob(job) {
    if (job.status === "processing") return;
    try {
      const rec = await api.getRecord(job.record_id);
      onOpen(rec);
    } catch (err) {
      setError(err.message);
    }
  }

  async function remove(job, e) {
    e.stopPropagation();
    try {
      await api.deleteRecord(job.record_id);
      setJobs((prev) => prev.filter((j) => j.record_id !== job.record_id));
    } catch (err) {
      setError(err.message);
    }
  }

  const processingCount = jobs.filter((j) => j.status === "processing").length;

  return (
    <div className="job-page">
      <div className="job-uploader">
        <h2>Procesar documentos</h2>
        <p className="muted">
          Sube un documento y se procesa en segundo plano (OCR + IA): se corrige
          la orientación, se detecta la plantilla y se extraen los datos. Si algo
          necesita revisión, el estado lo indicará.
        </p>
        <div className="form-row">
          <label>Plantilla (opcional, si no se auto-detecta)</label>
          <select
            value={forcedTemplate}
            onChange={(e) => setForcedTemplate(e.target.value)}
          >
            <option value="">Auto-detectar</option>
            {templates.map((t) => (
              <option key={t.id} value={t.id}>
                {t.name}
              </option>
            ))}
          </select>
        </div>
        <Uploader onFile={handleUpload} busy={uploading} label="Subir documento" />
        {error && <p className="error">{error}</p>}
      </div>

      <div className="job-list">
        <div className="job-list-head">
          <h3>Documentos {jobs.length > 0 && `(${jobs.length})`}</h3>
          {processingCount > 0 && (
            <span className="muted small">⏳ {processingCount} en proceso…</span>
          )}
        </div>

        {jobs.length === 0 ? (
          <p className="muted">Aún no has procesado documentos.</p>
        ) : (
          <table className="job-table">
            <thead>
              <tr>
                <th>Documento</th>
                <th>Plantilla</th>
                <th>Estado</th>
                <th>Similitud</th>
                <th>Campos</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((job) => {
                const st = STATUS[job.status] || STATUS.review;
                return (
                  <tr
                    key={job.record_id}
                    className={"job-row" + (job.status === "processing" ? " disabled" : "")}
                    onClick={() => openJob(job)}
                  >
                    <td className="job-name">{job.filename}</td>
                    <td>{job.template_name || <span className="muted">—</span>}</td>
                    <td>
                      <span className={st.cls}>
                        {st.spin && <span className="mini-spinner" />}
                        {st.label}
                      </span>
                    </td>
                    <td>
                      {job.match_score
                        ? `${Math.round(job.match_score * 100)}%`
                        : "—"}
                    </td>
                    <td>{job.n_fields || 0}</td>
                    <td className="job-actions">
                      {job.status !== "processing" && (
                        <>
                          <button
                            className="btn small"
                            title="Ver datos extraídos"
                            onClick={(e) => openData(job, e)}
                          >
                            👁 Datos
                          </button>
                          <button
                            className="btn small"
                            onClick={(e) => {
                              e.stopPropagation();
                              openJob(job);
                            }}
                          >
                            {job.status === "review" ? "Revisar" : "Abrir"}
                          </button>
                        </>
                      )}
                      <button
                        className="btn danger small"
                        title="Eliminar"
                        onClick={(e) => remove(job, e)}
                      >
                        ✕
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      {modal && (
        <div className="modal-overlay" onClick={() => setModal(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <h3>{modal.job.filename}</h3>
              <button className="btn small" onClick={() => setModal(null)}>
                ✕
              </button>
            </div>
            <p className="muted small">
              {modal.job.template_name || "Sin plantilla"}
              {modal.job.match_score
                ? ` · ${Math.round(modal.job.match_score * 100)}% similitud`
                : ""}
            </p>

            {Object.keys(modal.data).length === 0 ? (
              <p className="muted">Sin datos extraídos.</p>
            ) : (
              <div className="modal-form">
                {Object.entries(modal.data).map(([k, v]) => (
                  <div className="modal-field" key={k}>
                    <label>{fieldName(modal.job, k)}</label>
                    <input readOnly value={String(v ?? "")} />
                  </div>
                ))}
              </div>
            )}

            <details className="json-preview">
              <summary>JSON</summary>
              <pre>{JSON.stringify(modal.data, null, 2)}</pre>
            </details>

            <div className="modal-actions">
              <button className="btn" onClick={() => setModal(null)}>
                Cerrar
              </button>
              <button
                className="btn primary"
                onClick={() => {
                  const j = modal.job;
                  setModal(null);
                  openJob(j);
                }}
              >
                Editar a fondo
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ===========================================================================
// Vista DETALLE: editor de revisión (carga desde un registro existente)
// ===========================================================================
function DocumentEditor({ record, onBack }) {
  const documentId = record.document_id;
  const templateId = record.template_id;

  const [templates, setTemplates] = useState([]);
  const [chosenTemplate, setChosenTemplate] = useState(templateId ?? "");
  const [doc, setDoc] = useState(null);
  const [result, setResult] = useState(null);
  const [fields, setFields] = useState({});
  const [values, setValues] = useState({});
  const [activeKey, setActiveKey] = useState(null);
  const [pending, setPending] = useState(null);
  const [zoom, setZoom] = useState(1);
  const [panMode, setPanMode] = useState(false);
  const [imgRev, setImgRev] = useState(0);
  const [border, setBorder] = useState(null);
  const [borderMode, setBorderMode] = useState(false);
  const [quad, setQuad] = useState(null);
  const [quadMode, setQuadMode] = useState(false);
  const [busy, setBusy] = useState(false);
  const [aiBusy, setAiBusy] = useState(false);
  const [aiStatus, setAiStatus] = useState(null);
  const [error, setError] = useState("");
  const [saved, setSaved] = useState(false);
  const [scanning, setScanning] = useState({});
  const [anchorModal, setAnchorModal] = useState(false);

  // Carga inicial: documento + proceso + valores guardados del registro
  useEffect(() => {
    api.aiStatus().then(setAiStatus).catch(() => setAiStatus({ available: false }));
    api.listTemplates().then(setTemplates).catch(() => {});
    (async () => {
      setBusy(true);
      try {
        const d = await api.getDocument(documentId);
        setDoc(d);
        const res = await api.processDocument(documentId, templateId || undefined);
        applyResult(res, record.data);
      } catch (err) {
        setError(err.message);
      } finally {
        setBusy(false);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [documentId]);

  function applyResult(res, savedData) {
    setResult(res);
    setBorder(res.border || { x: 0, y: 0, w: 1, h: 1 });
    const fs = {};
    const vs = {};
    Object.entries(res.fields).forEach(([k, v]) => {
      // La caja se ajusta al texto realmente capturado (matched_box) cuando lo hay;
      // si no, a la región teórica de la plantilla.
      fs[k] = {
        name: v.name,
        region: v.matched_box || v.region,
        confidence: v.confidence,
        n_words: v.n_words,
      };
      vs[k] = v.value;
    });
    setFields(fs);
    // Los valores guardados (IA del job o revisión previa) tienen prioridad
    setValues(savedData ? { ...vs, ...savedData } : vs);
  }

  async function runAI() {
    if (!result?.template_id || !aiStatus?.ready) return;
    setAiBusy(true);
    setError("");
    try {
      const res = await api.aiExtract(documentId, result.template_id);
      setValues((prev) => {
        const n = { ...prev };
        Object.entries(res.fields).forEach(([k, v]) => (n[k] = v.value));
        return n;
      });
      setFields((prev) => {
        const n = { ...prev };
        Object.entries(res.fields).forEach(([k, v]) => {
          if (n[k] && v.region) n[k] = { ...n[k], region: v.region };
        });
        return n;
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setAiBusy(false);
    }
  }

  async function reprocess(tplId = chosenTemplate) {
    const res = await api.processDocument(documentId, tplId || undefined);
    applyResult(res);
  }

  // Cambiar manualmente la plantilla cuando la auto-detección se equivoca
  async function changeTemplate(value) {
    setChosenTemplate(value);
    setBusy(true);
    setError("");
    try {
      await reprocess(value);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function rotate(deg) {
    if (!doc) return;
    setBusy(true);
    setError("");
    try {
      const d = await api.rotateDocument(documentId, deg);
      setDoc(d);
      setImgRev((v) => v + 1);
      await reprocess();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function autoOrient() {
    if (!doc) return;
    setBusy(true);
    setError("");
    try {
      const d = await api.autoOrient(documentId, chosenTemplate || undefined);
      setDoc(d);
      setImgRev((v) => v + 1);
      await reprocess();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  function textInRegion(box) {
    if (!doc) return "";
    const x1 = box.x + box.w,
      y1 = box.y + box.h;
    return doc.ocr_words
      .filter((w) => {
        const cx = w.box.x + w.box.w / 2;
        const cy = w.box.y + w.box.h / 2;
        return cx >= box.x && cx <= x1 && cy >= box.y && cy <= y1;
      })
      .sort((a, b) => a.box.y - b.box.y || a.box.x - b.box.x)
      .map((w) => w.text)
      .join(" ");
  }

  async function onRegionDraw(box) {
    setPending({ box, text: textInRegion(box), confidence: null, scanning: true });
    try {
      const res = await api.ocrRegion(documentId, box);
      setPending((p) =>
        p ? { ...p, text: res.text || p.text, confidence: res.confidence, scanning: false } : p
      );
    } catch {
      setPending((p) => (p ? { ...p, scanning: false } : p));
    }
  }

  function onWordClick(word) {
    setPending({ box: word.box, text: word.text, confidence: word.conf, scanning: false });
  }

  function assign(target, newName) {
    if (!pending) return;
    let key = target;
    if (target === "__new__") {
      if (!newName?.trim()) return;
      key = toKey(newName);
      setFields((prev) => ({
        ...prev,
        [key]: { name: newName.trim(), region: pending.box, confidence: pending.confidence ?? 0, n_words: 0 },
      }));
    } else {
      setFields((prev) => ({
        ...prev,
        [key]: { ...prev[key], region: pending.box, confidence: pending.confidence ?? prev[key].confidence },
      }));
    }
    setValues((prev) => ({ ...prev, [key]: pending.text }));
    setActiveKey(key);
    setPending(null);
  }

  function onRegionChange(key, box) {
    setFields((prev) => ({ ...prev, [key]: { ...prev[key], region: box } }));
  }

  async function applyBorder() {
    if (!doc || !border) return;
    setBusy(true);
    setError("");
    try {
      await api.updateBorder(documentId, border);
      await reprocess();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  function toggleBorderMode() {
    if (borderMode) applyBorder();
    setBorderMode((m) => !m);
  }

  async function autoDetectBorder() {
    if (!doc) return;
    try {
      const b = await api.detectBorder(documentId);
      setBorder(b);
    } catch (err) {
      setError(err.message);
    }
  }

  function startQuad() {
    const b = border || { x: 0, y: 0, w: 1, h: 1 };
    setQuad({
      tl: { x: b.x, y: b.y },
      tr: { x: b.x + b.w, y: b.y },
      br: { x: b.x + b.w, y: b.y + b.h },
      bl: { x: b.x, y: b.y + b.h },
    });
    setQuadMode(true);
  }

  function cancelQuad() {
    setQuadMode(false);
    setQuad(null);
  }

  async function applyRectify() {
    if (!doc || !quad) return;
    setBusy(true);
    setError("");
    try {
      const d = await api.rectifyDocument(documentId, quad);
      setDoc(d);
      setImgRev((v) => v + 1);
      setQuadMode(false);
      setQuad(null);
      await reprocess();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function rescan(key) {
    const region = fields[key]?.region;
    if (!region) return;
    setScanning((s) => ({ ...s, [key]: true }));
    try {
      const res = await api.ocrRegion(documentId, region);
      setValues((prev) => ({ ...prev, [key]: res.text }));
    } catch (err) {
      setError(err.message);
    } finally {
      setScanning((s) => ({ ...s, [key]: false }));
    }
  }

  function removeField(key) {
    setFields((prev) => {
      const n = { ...prev };
      delete n[key];
      return n;
    });
    setValues((prev) => {
      const n = { ...prev };
      delete n[key];
      return n;
    });
  }

  async function confirm() {
    setBusy(true);
    setError("");
    try {
      const regions = Object.fromEntries(
        Object.entries(fields).map(([k, v]) => [k, v.region])
      );
      await api.updateRecord(record.record_id ?? record.id, {
        template_id: result.template_id,
        document_id: documentId,
        data: values,
        match_score: result.match_score,
        status: "confirmed",
        regions,
        learn: true,
      });
      setSaved(true);
      setTimeout(() => onBack(), 600);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  // Anclas para el overlay. Si el documento está ALINEADO a la plantilla, se
  // dibujan en su posición de plantilla (proporcional, coherente con los campos);
  // si no, en la caja que ORB encontró (para diagnosticar el match).
  const anchorRegions = (result?.anchors || [])
    .map((a, i) => {
      const box = result.aligned ? a.expected_region : a.region;
      if (!box) return null;
      return {
        key: `__anchor_${i}`,
        name: `⚓ ${a.name}${a.found ? " ✓" : ""}`,
        className: "anchor",
        readOnly: true,
        ...box,
      };
    })
    .filter(Boolean);

  const regions = [
    ...Object.entries(fields).map(([k, v]) => ({ key: k, name: v.name, ...v.region })),
    ...anchorRegions,
    ...(pending ? [{ key: "__pending__", name: "Nuevo", ...pending.box }] : []),
  ];

  const lowConfidence = result && result.match_score < 0.55;

  return (
    <div className="editor">
      {(busy || aiBusy) && (
        <div className="loading-overlay">
          <div className="loading-spinner">
            <div className="spinner-ring" />
            <div className="spinner-ring spinner-ring-2" />
            <div className="spinner-ring spinner-ring-3" />
            <div className="spinner-icon">📄</div>
          </div>
          <p className="loading-text">
            {aiBusy ? "🧠 Extrayendo datos con IA…" : busy ? "⏳ Procesando documento…" : ""}
          </p>
          <p className="loading-sub">Esto puede tardar unos segundos</p>
        </div>
      )}
      <div className="editor-left">
        {!doc ? (
          <p className="muted" style={{ padding: 24 }}>Cargando documento…</p>
        ) : (
          <>
            <div className="viewer-toolbar">
              <div className="zoom-controls">
                <button className="btn small" onClick={onBack} title="Volver a la lista">
                  ← Lista
                </button>
                <button className="btn small" onClick={() => setZoom((z) => Math.max(0.25, +(z - 0.25).toFixed(2)))}>
                  −
                </button>
                <span>{Math.round(zoom * 100)}%</span>
                <button className="btn small" onClick={() => setZoom((z) => Math.min(6, +(z + 0.25).toFixed(2)))}>
                  +
                </button>
                <button className="btn small" onClick={() => setZoom(1)} title="Ajustar">
                  ⤢
                </button>
                <button
                  className={"btn small" + (panMode ? " primary" : "")}
                  onClick={() => setPanMode((m) => !m)}
                  title="Mover documento"
                >
                  ✋
                </button>
                <button className="btn small" onClick={() => rotate(-90)} title="Girar 90° izquierda" disabled={busy}>
                  ↺
                </button>
                <button className="btn small" onClick={() => rotate(90)} title="Girar 90° derecha" disabled={busy}>
                  ↻
                </button>
                <button className="btn small" onClick={autoOrient} title="Auto-enderezar según la plantilla" disabled={busy}>
                  🧭 Auto
                </button>
              </div>
              <div className="border-controls">
                {!quadMode && (
                  <button
                    className={"btn small" + (borderMode ? " primary" : "")}
                    onClick={toggleBorderMode}
                    disabled={busy}
                    title="Ajustar el borde rectangular y re-posicionar los campos"
                  >
                    {borderMode ? "✓ Aplicar bordes" : "✎ Ajustar bordes"}
                  </button>
                )}
                {borderMode && (
                  <button className="btn small" onClick={autoDetectBorder}>
                    ⟳ Auto-detectar
                  </button>
                )}
                {!borderMode && !quadMode && (
                  <button
                    className="btn small"
                    onClick={startQuad}
                    disabled={busy}
                    title="Marcar las 4 esquinas y enderezar la perspectiva"
                  >
                    ⬢ Enderezar (4 puntos)
                  </button>
                )}
                {quadMode && (
                  <>
                    <button className="btn small primary" onClick={applyRectify} disabled={busy}>
                      ✓ Enderezar
                    </button>
                    <button className="btn small" onClick={cancelQuad} disabled={busy}>
                      Cancelar
                    </button>
                  </>
                )}
              </div>
            </div>
            <DocumentViewer
              imageUrl={`${api.documentImageUrl(documentId)}?v=${imgRev}`}
              words={doc.ocr_words}
              regions={regions}
              activeKey={activeKey}
              onWordClick={onWordClick}
              onRegionDraw={onRegionDraw}
              onRegionClick={setActiveKey}
              showWords={!borderMode && !quadMode}
              zoom={zoom}
              onZoomChange={setZoom}
              panMode={panMode}
              border={quadMode ? null : border}
              editableBorder={borderMode && !quadMode}
              onBorderChange={setBorder}
              editableRegions={!borderMode && !quadMode}
              onRegionChange={onRegionChange}
              onRegionCommit={rescan}
              quad={quadMode ? quad : null}
              editableQuad={quadMode}
              onQuadChange={setQuad}
            />
          </>
        )}
      </div>

      <aside className="editor-right">
        {!result ? (
          <p className="muted">Cargando…</p>
        ) : (
          <>
            <div className="match-info">
              <h3>{result.template_name || "Sin plantilla"}</h3>
              <div className="form-row template-switch">
                <label>Plantilla</label>
                <select
                  value={chosenTemplate}
                  onChange={(e) => changeTemplate(e.target.value)}
                  disabled={busy}
                  title="Cámbiala si la detección automática se equivocó"
                >
                  <option value="">Auto-detectar</option>
                  {templates.map((t) => (
                    <option key={t.id} value={t.id}>
                      {t.name}
                    </option>
                  ))}
                </select>
              </div>
              <div className="match-scores">
                <span className={"badge " + (lowConfidence ? "warn" : "ok")}>
                  {(result.match_score * 100).toFixed(0)}% similitud
                </span>
                {result.visual_score > 0 && (
                  <span className="badge visual-score" title="Similitud visual (ORB) con la imagen de muestra">
                    👁️ {(result.visual_score * 100).toFixed(0)}% visual
                  </span>
                )}
                {result.anchors && result.anchors.length > 0 && (
                  <span
                    className={
                      "badge " +
                      (result.anchors.every((a) => a.found) ? "ok" : "warn")
                    }
                  >
                    🎯 {result.anchors.filter((a) => a.found).length}/
                    {result.anchors.length} anclas
                  </span>
                )}
                {result.anchors && result.anchors.length > 0 && (
                  <button
                    className="btn small"
                    onClick={() => setAnchorModal(true)}
                    title="Ver la huella de cada ancla: plantilla vs detectada"
                  >
                    ⚓ Ver anclas
                  </button>
                )}
              </div>
              {result.pipeline && result.pipeline.length > 0 && (
                <ul className="pipeline-trace">
                  {result.pipeline.map((step, i) => (
                    <li key={i}>{step}</li>
                  ))}
                </ul>
              )}
            </div>
            {!result.template_id && (
              <p className="warn-text">
                No se auto-detectó plantilla. Puedes seleccionar zonas y crear campos
                manualmente.
              </p>
            )}
            {lowConfidence && result.template_id && (
              <p className="warn-text">Coincidencia baja: revisa los campos antes de confirmar.</p>
            )}
            {result.needs_review && result.template_id && (
              <p className="warn-text">
                ⚠ No se localizaron las anclas obligatorias: el documento no se ha
                enderezado automáticamente. Ajusta la orientación (🧭 Auto / girar) y las
                zonas a mano antes de confirmar.
              </p>
            )}
            {result.aligned && (
              <p className="muted small">
                ✓ Documento enderezado y alineado a la plantilla.
              </p>
            )}

            {result.template_id && (
              <div className="ai-bar">
                <span className="muted small">
                  {aiBusy
                    ? `🧠 Extrayendo con IA (${aiStatus?.model})…`
                    : aiStatus?.ready
                    ? `🧠 IA disponible (${aiStatus.model})`
                    : aiStatus?.available
                    ? `Ollama arriba, falta descargar el modelo — ver README`
                    : "IA local no disponible — solo extracción geométrica"}
                </span>
                {aiStatus?.ready && (
                  <button
                    className="btn small"
                    onClick={runAI}
                    disabled={aiBusy}
                    title="Volver a extraer con IA"
                  >
                    ↻ Re-extraer IA
                  </button>
                )}
              </div>
            )}

            {pending && (
              <AssignPanel
                pending={pending}
                fieldKeys={Object.keys(fields)}
                fields={fields}
                onAssign={assign}
                onCancel={() => setPending(null)}
              />
            )}

            <p className="muted small">
              Revisa los valores, re-escanea (↻) o arrastra una zona en el documento
              para asignarla a un campo o crear uno nuevo.
            </p>

            <div className="field-edit-list">
              {Object.entries(fields).map(([k, v]) => (
                <div
                  key={k}
                  className={"field-edit " + (k === activeKey ? "active" : "")}
                  onClick={() => setActiveKey(k)}
                >
                  <label>
                    {v.name}{" "}
                    <span className="muted small">
                      ({v.confidence}%{v.n_words ? ` · ${v.n_words}w` : ""})
                    </span>
                  </label>
                  <div className="row">
                    <input
                      value={values[k] ?? ""}
                      onChange={(e) => setValues((prev) => ({ ...prev, [k]: e.target.value }))}
                    />
                    <button
                      className="btn small"
                      title="Re-escanear esta región con OCR"
                      disabled={scanning[k]}
                      onClick={(e) => {
                        e.stopPropagation();
                        rescan(k);
                      }}
                    >
                      {scanning[k] ? "…" : "↻ OCR"}
                    </button>
                    <button
                      className="btn danger small"
                      title="Quitar campo de este registro"
                      onClick={(e) => {
                        e.stopPropagation();
                        removeField(k);
                      }}
                    >
                      ✕
                    </button>
                  </div>
                </div>
              ))}
            </div>

            <details className="json-preview">
              <summary>JSON resultado</summary>
              <pre>{JSON.stringify(values, null, 2)}</pre>
            </details>

            {error && <p className="error">{error}</p>}
            {saved ? (
              <p className="success">✓ Registro confirmado.</p>
            ) : (
              <div className="editor-actions">
                <button className="btn" onClick={onBack} disabled={busy}>
                  Cancelar
                </button>
                <button
                  className="btn primary"
                  onClick={confirm}
                  disabled={busy || Object.keys(fields).length === 0}
                >
                  Aceptar y confirmar
                </button>
              </div>
            )}
          </>
        )}
      </aside>

      {anchorModal && result && (
        <AnchorFootprintModal
          anchors={result.anchors || []}
          aligned={result.aligned}
          docImageUrl={`${api.documentImageUrl(documentId)}?v=${imgRev}`}
          sampleImageUrl={
            result.sample_document_id
              ? api.documentImageUrl(result.sample_document_id)
              : null
          }
          onClose={() => setAnchorModal(false)}
        />
      )}
    </div>
  );
}

// Modal de huella de anclas: recorta al vuelo cada ancla de la muestra (esperada)
// y del documento (detectada) usando las coordenadas normalizadas.
function AnchorFootprintModal({ anchors, aligned, docImageUrl, sampleImageUrl, onClose }) {
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal anchor-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <h3>⚓ Huella de anclas</h3>
          <button className="btn small" onClick={onClose}>
            ✕
          </button>
        </div>
        <p className="muted small">
          Comparación de cada ancla: a la izquierda, la zona de la plantilla
          (esperada); a la derecha, la zona localizada en el documento.
        </p>
        <div className="anchor-foot-list">
          {anchors.map((a, i) => (
            <div className="anchor-foot-row" key={i}>
              <div className="anchor-foot-label">
                <strong>⚓ {a.name || "(ancla)"}</strong>
                <span className={"badge " + (a.found ? "ok" : "danger")}>
                  {a.found ? "✓ detectada" : "✗ no encontrada"}
                </span>
                <span className="muted small">
                  {a.text_score > 0 && `texto ${Math.round(a.text_score * 100)}%`}
                  {a.text_score > 0 && a.image_score > 0 && " · "}
                  {a.image_score > 0 && `imagen ${Math.round(a.image_score * 100)}%`}
                </span>
              </div>
              <div className="anchor-foot-crops">
                <figure>
                  <CropView imageUrl={sampleImageUrl} box={a.expected_region} />
                  <figcaption>Plantilla</figcaption>
                </figure>
                <figure>
                  {/* Si el documento está rectificado a la plantilla, la zona del
                      ancla está en su posición de plantilla (expected_region); si
                      no, en la caja que ORB localizó (region). */}
                  <CropView
                    imageUrl={docImageUrl}
                    box={aligned ? a.expected_region : a.region}
                  />
                  <figcaption>Detectada</figcaption>
                </figure>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// Recorta una región normalizada de una imagen usando background-position/size.
function CropView({ imageUrl, box }) {
  if (!imageUrl || !box || !box.w || !box.h) {
    return <div className="anchor-crop empty">—</div>;
  }
  // Mostrar la región con un pequeño margen para contexto
  const pad = 0.4;
  const w = Math.min(1, box.w * (1 + 2 * pad));
  const h = Math.min(1, box.h * (1 + 2 * pad));
  const x = Math.max(0, box.x - box.w * pad);
  const y = Math.max(0, box.y - box.h * pad);
  // Escala: la imagen de fondo se amplía 1/w (o 1/h) para que la región llene la caja
  const bgW = (1 / w) * 100;
  const bgH = (1 / h) * 100;
  const posX = w < 1 ? (x / (1 - w)) * 100 : 0;
  const posY = h < 1 ? (y / (1 - h)) * 100 : 0;
  return (
    <div
      className="anchor-crop"
      style={{
        backgroundImage: `url(${imageUrl})`,
        backgroundSize: `${bgW}% ${bgH}%`,
        backgroundPosition: `${posX}% ${posY}%`,
        backgroundRepeat: "no-repeat",
      }}
    />
  );
}

function AssignPanel({ pending, fieldKeys, fields, onAssign, onCancel }) {
  const [target, setTarget] = useState(fieldKeys[0] || "__new__");
  const [newName, setNewName] = useState("");
  const isNew = target === "__new__" || fieldKeys.length === 0;

  return (
    <div className="pending-field">
      <div className="muted small">
        Valor detectado:{" "}
        {pending.scanning ? (
          <em>reconociendo…</em>
        ) : (
          pending.confidence != null && <span>(OCR {pending.confidence}%)</span>
        )}
      </div>
      <div className="captured">{pending.text || "(zona sin texto)"}</div>

      <label className="muted small">Asignar a:</label>
      <select value={target} onChange={(e) => setTarget(e.target.value)}>
        {fieldKeys.map((k) => (
          <option key={k} value={k}>
            {fields[k].name}
          </option>
        ))}
        <option value="__new__">➕ Crear campo nuevo…</option>
      </select>

      {isNew && (
        <input
          autoFocus
          placeholder="Nombre del nuevo campo"
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && onAssign("__new__", newName)}
        />
      )}

      <div className="row">
        <button
          className="btn primary small"
          onClick={() => onAssign(isNew ? "__new__" : target, newName)}
        >
          {isNew ? "Crear y asignar" : "Asignar"}
        </button>
        <button className="btn small" onClick={onCancel}>
          Cancelar
        </button>
      </div>
    </div>
  );
}
