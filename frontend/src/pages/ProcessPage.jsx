import { useEffect, useState } from "react";
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

export default function ProcessPage() {
  const [templates, setTemplates] = useState([]);
  const [forcedTemplate, setForcedTemplate] = useState("");
  const [doc, setDoc] = useState(null);
  const [result, setResult] = useState(null);
  const [fields, setFields] = useState({}); // {key:{name,region,confidence,n_words}}
  const [values, setValues] = useState({}); // {key: value}
  const [activeKey, setActiveKey] = useState(null);
  const [pending, setPending] = useState(null); // {box,text,confidence,scanning}
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

  useEffect(() => {
    api.listTemplates().then(setTemplates);
    api.aiStatus().then(setAiStatus).catch(() => setAiStatus({ available: false }));
  }, []);

  // Extracción con IA: rellena valores Y actualiza las regiones (proporcionalidad)
  async function runAI(docId, templateId) {
    if (!templateId || !aiStatus?.ready) return;
    setAiBusy(true);
    setError("");
    try {
      const res = await api.aiExtract(docId, templateId);
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

  async function handleUpload(file) {
    if (!file) return;
    setBusy(true);
    setError("");
    setResult(null);
    setSaved(false);
    setPending(null);
    try {
      const d = await api.uploadDocument(file);
      setDoc(d);
      const res = await api.processDocument(
        d.id,
        forcedTemplate ? Number(forcedTemplate) : undefined
      );
      applyResult(res);
      if (res.template_id) await runAI(d.id, res.template_id); // IA automática
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  function applyResult(res) {
    setResult(res);
    setBorder(res.border || { x: 0, y: 0, w: 1, h: 1 });
    const fs = {};
    const vs = {};
    Object.entries(res.fields).forEach(([k, v]) => {
      fs[k] = { name: v.name, region: v.region, confidence: v.confidence, n_words: v.n_words };
      vs[k] = v.value;
    });
    setFields(fs);
    setValues(vs);
  }

  async function rotate(deg) {
    if (!doc) return;
    setBusy(true);
    setError("");
    try {
      const d = await api.rotateDocument(doc.id, deg);
      setDoc(d);
      setImgRev((v) => v + 1);
      const res = await api.processDocument(
        d.id,
        forcedTemplate ? Number(forcedTemplate) : undefined
      );
      applyResult(res);
      if (res.template_id) await runAI(d.id, res.template_id); // IA automática
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

  // Selección de una zona nueva -> re-OCR y abre el panel de asignación
  async function onRegionDraw(box) {
    setPending({ box, text: textInRegion(box), confidence: null, scanning: true });
    try {
      const res = await api.ocrRegion(doc.id, box);
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

  // Asigna el texto detectado a un campo existente o crea uno nuevo
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
      // Actualiza la región resaltada del campo a la zona seleccionada
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

  // Persiste el borde corregido y re-procesa (re-posiciona los campos proporcionalmente)
  async function applyBorder() {
    if (!doc || !border) return;
    setBusy(true);
    setError("");
    try {
      await api.updateBorder(doc.id, border);
      const res = await api.processDocument(
        doc.id,
        result?.template_id || (forcedTemplate ? Number(forcedTemplate) : undefined)
      );
      applyResult(res);
      if (res.template_id) await runAI(doc.id, res.template_id);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  function toggleBorderMode() {
    if (borderMode) applyBorder(); // al salir del modo, aplica y re-procesa
    setBorderMode((m) => !m);
  }

  async function autoDetectBorder() {
    if (!doc) return;
    try {
      const b = await api.detectBorder(doc.id);
      setBorder(b);
    } catch (err) {
      setError(err.message);
    }
  }

  // Enderezar (perspectiva con 4 puntos)
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
      const d = await api.rectifyDocument(doc.id, quad);
      setDoc(d);
      setImgRev((v) => v + 1);
      setQuadMode(false);
      setQuad(null);
      const res = await api.processDocument(
        d.id,
        result?.template_id || (forcedTemplate ? Number(forcedTemplate) : undefined)
      );
      applyResult(res);
      if (res.template_id) await runAI(d.id, res.template_id);
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
      const res = await api.ocrRegion(doc.id, region);
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
      await api.createRecord({
        template_id: result.template_id,
        document_id: result.document_id,
        data: values,
        match_score: result.match_score,
        status: "confirmed",
        regions,
        learn: true,
      });
      setSaved(true);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  const regions = [
    ...Object.entries(fields).map(([k, v]) => ({ key: k, name: v.name, ...v.region })),
    ...(pending ? [{ key: "__pending__", name: "Nuevo", ...pending.box }] : []),
  ];

  const lowConfidence = result && result.match_score < 0.55;

  return (
    <div className="editor">
      <div className="editor-left">
        {!doc ? (
          <div className="upload-zone">
            <h2>Procesar documento</h2>
            <p className="muted">
              Sube un documento. Se corrige su orientación, se detecta la plantilla
              más parecida y se rellenan los campos entrenados.
            </p>
            <div className="form-row">
              <label>Plantilla (opcional, si no se auto-detecta)</label>
              <select value={forcedTemplate} onChange={(e) => setForcedTemplate(e.target.value)}>
                <option value="">Auto-detectar</option>
                {templates.map((t) => (
                  <option key={t.id} value={t.id}>
                    {t.name}
                  </option>
                ))}
              </select>
            </div>
            <Uploader onFile={handleUpload} busy={busy} label="Subir documento" />
          </div>
        ) : (
          <>
            <div className="viewer-toolbar">
              <div className="zoom-controls">
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
                  title="Mover documento (o botón central del ratón)"
                >
                  ✋
                </button>
                <button className="btn small" onClick={() => rotate(-90)} title="Girar 90° izquierda" disabled={busy}>
                  ↺
                </button>
                <button className="btn small" onClick={() => rotate(90)} title="Girar 90° derecha" disabled={busy}>
                  ↻
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
              imageUrl={`${api.documentImageUrl(doc.id)}?v=${imgRev}`}
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
          <p className="muted">Esperando documento…</p>
        ) : (
          <>
            <div className="match-info">
              <h3>{result.template_name || "Sin plantilla"}</h3>
              <span className={"badge " + (lowConfidence ? "warn" : "ok")}>
                Confianza {(result.match_score * 100).toFixed(0)}%
              </span>
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

            {result.template_id && (
              <div className="ai-bar">
                <span className="muted small">
                  {aiBusy
                    ? `🧠 Extrayendo con IA (${aiStatus?.model})…`
                    : aiStatus?.ready
                    ? `🧠 IA aplicada automáticamente (${aiStatus.model})`
                    : aiStatus?.available
                    ? `Ollama arriba, falta descargar el modelo — ver README`
                    : "IA local no disponible — solo extracción geométrica"}
                </span>
                {aiStatus?.ready && (
                  <button
                    className="btn small"
                    onClick={() => runAI(doc.id, result.template_id)}
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
              <p className="success">✓ Registro creado correctamente.</p>
            ) : (
              <div className="editor-actions">
                <button
                  className="btn primary"
                  onClick={confirm}
                  disabled={busy || Object.keys(fields).length === 0}
                >
                  Aceptar y crear registro
                </button>
              </div>
            )}
          </>
        )}
      </aside>
    </div>
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
