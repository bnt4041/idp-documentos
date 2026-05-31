import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api } from "../api.js";
import DocumentViewer from "../components/DocumentViewer.jsx";
import Uploader from "../components/Uploader.jsx";

// Convierte un nombre a clave snake_case (sin acentos)
const toKey = (s) =>
  s
    .toLowerCase()
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_|_$/g, "");

const FULL_BORDER = { x: 0, y: 0, w: 1, h: 1 };
const ANCHOR_PREFIX = "__anchor__";

// Conversión entre coords del escaneo completo (0..1) y coords relativas al borde
const imgToRel = (box, b) => ({
  x: +((box.x - b.x) / b.w).toFixed(5),
  y: +((box.y - b.y) / b.h).toFixed(5),
  w: +(box.w / b.w).toFixed(5),
  h: +(box.h / b.h).toFixed(5),
});
const relToImg = (f, b) => ({
  x: b.x + f.x * b.w,
  y: b.y + f.y * b.h,
  w: f.w * b.w,
  h: f.h * b.h,
});

export default function TemplateEditor() {
  const { id } = useParams();
  const navigate = useNavigate();
  const editing = Boolean(id);

  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [doc, setDoc] = useState(null);
  const [signature, setSignature] = useState({});
  const [border, setBorder] = useState(FULL_BORDER);
  const [borderMode, setBorderMode] = useState(false);
  const [panMode, setPanMode] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [imgRev, setImgRev] = useState(0);
  const [quad, setQuad] = useState(null);
  const [quadMode, setQuadMode] = useState(false);
  const [fields, setFields] = useState([]); // coords RELATIVAS al borde
  const [anchors, setAnchors] = useState([]); // hitos/anclas, coords RELATIVAS al borde
  const [anchorMode, setAnchorMode] = useState(false); // dibujar crea anclas, no campos
  const [pending, setPending] = useState(null); // selección (coords de imagen)
  const [activeKey, setActiveKey] = useState(null);
  const [busy, setBusy] = useState(false);
  const [suggesting, setSuggesting] = useState(false);
  const [suggestSource, setSuggestSource] = useState(null);  // "ollama" | "heuristic" | null
  const [error, setError] = useState("");

  // Carga de plantilla existente
  useEffect(() => {
    if (!editing) return;
    (async () => {
      const t = await api.getTemplate(id);
      setName(t.name);
      setDescription(t.description);
      setSignature(t.signature);
      setFields(
        t.fields.map((f) => ({
          key: f.key,
          name: f.name,
          data_type: f.data_type,
          x: f.x,
          y: f.y,
          w: f.w,
          h: f.h,
          sample_text: f.sample_text,
        }))
      );
      setAnchors(
        (t.anchors || []).map((a, i) => ({
          key: `a${i}_${a.id ?? i}`,
          name: a.name,
          x: a.x,
          y: a.y,
          w: a.w,
          h: a.h,
          anchor_text: a.anchor_text || "",
          use_text: a.use_text,
          use_image: a.use_image,
          weight: a.weight ?? 1.0,
        }))
      );
      if (t.sample_document_id) {
        const d = await fetch(`/api/documents/${t.sample_document_id}`).then((r) =>
          r.json()
        );
        setDoc(d);
        // Usa el borde de referencia de la plantilla (con el que se guardaron los campos)
        const b =
          t.border && t.border.w > 0 ? t.border : validBorder(d.border);
        setBorder(b);
      }
    })();
  }, [id, editing]);

  async function handleSample(file) {
    setBusy(true);
    setError("");
    try {
      const d = await api.uploadDocument(file);
      setDoc(d);
      setSignature(d.signature);
      setBorder(validBorder(d.border));
      setBorderMode(true); // empieza ajustando bordes
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  function validBorder(b) {
    return b && b.w > 0 && b.h > 0 ? b : FULL_BORDER;
  }

  // Texto OCR contenido en una región de imagen (preview rápido cacheado)
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

  function onWordClick(word) {
    setPending({ ...word.box, sample_text: word.text, scanning: false });
  }

  async function onRegionDraw(box) {
    setPending({ ...box, sample_text: textInRegion(box), scanning: true });
    try {
      const res = await api.ocrRegion(doc.id, box);
      setPending((p) =>
        p
          ? { ...p, sample_text: res.text || p.sample_text, confidence: res.confidence, scanning: false }
          : p
      );
    } catch {
      setPending((p) => (p ? { ...p, scanning: false } : p));
    }
  }

  function confirmField(fieldName) {
    if (!pending || !fieldName.trim()) return;
    const key = toKey(fieldName);
    // Guardamos las coords RELATIVAS al borde para que sean proporcionales
    const rel = imgToRel(pending, border);
    const newField = {
      key,
      name: fieldName.trim(),
      data_type: "text",
      ...rel,
      sample_text: pending.sample_text || "",
    };
    setFields((prev) => [...prev.filter((f) => f.key !== key), newField]);
    setPending(null);
    setActiveKey(key);
  }

  function removeField(key) {
    setFields((prev) => prev.filter((f) => f.key !== key));
  }

  // Al cambiar el borde, re-basa los campos para que NO se muevan visualmente
  // (mantienen su posición absoluta en la imagen y quedan coherentes con el borde).
  function changeBorder(nb) {
    setFields((prev) =>
      prev.map((f) => {
        const img = relToImg(f, border);
        return { ...f, ...imgToRel(img, nb) };
      })
    );
    setAnchors((prev) =>
      prev.map((a) => {
        const img = relToImg(a, border);
        return { ...a, ...imgToRel(img, nb) };
      })
    );
    setBorder(nb);
  }

  // Mover/redimensionar una zona ya creada (coords de imagen -> relativas al borde)
  function onFieldRegionChange(key, box) {
    const rel = imgToRel(box, border);
    if (key.startsWith(ANCHOR_PREFIX)) {
      const ak = key.slice(ANCHOR_PREFIX.length);
      setAnchors((prev) => prev.map((a) => (a.key === ak ? { ...a, ...rel } : a)));
      return;
    }
    setFields((prev) => prev.map((f) => (f.key === key ? { ...f, ...rel } : f)));
  }

  // Al soltar, re-OCR de la nueva zona para refrescar el texto de muestra/ancla
  async function onFieldRegionCommit(key) {
    if (key.startsWith(ANCHOR_PREFIX)) {
      const ak = key.slice(ANCHOR_PREFIX.length);
      const a = anchors.find((x) => x.key === ak);
      if (!a || !doc) return;
      const img = relToImg(a, border);
      try {
        const res = await api.ocrRegion(doc.id, img);
        setAnchors((prev) =>
          prev.map((x) => (x.key === ak ? { ...x, anchor_text: res.text || x.anchor_text } : x))
        );
      } catch {
        /* noop */
      }
      return;
    }
    let f;
    setFields((prev) => {
      f = prev.find((x) => x.key === key);
      return prev;
    });
    if (!f || !doc) return;
    const img = relToImg(f, border);
    try {
      const res = await api.ocrRegion(doc.id, img);
      setFields((prev) =>
        prev.map((x) => (x.key === key ? { ...x, sample_text: res.text || x.sample_text } : x))
      );
    } catch {
      /* noop */
    }
  }

  // Crear un ancla a partir de la selección pendiente
  function confirmAnchor(anchorName, useText, useImage) {
    if (!pending || !anchorName.trim()) return;
    const rel = imgToRel(pending, border);
    const key = `a_${Date.now()}`;
    setAnchors((prev) => [
      ...prev,
      {
        key,
        name: anchorName.trim(),
        ...rel,
        anchor_text: pending.sample_text || "",
        use_text: useText,
        use_image: useImage,
        weight: 1.0,
      },
    ]);
    setPending(null);
  }

  function removeAnchor(key) {
    setAnchors((prev) => prev.filter((a) => a.key !== key));
  }

  async function rotate(deg) {
    if (!doc) return;
    try {
      const d = await api.rotateDocument(doc.id, deg);
      setDoc(d);
      setBorder(validBorder(d.border));
      setImgRev((v) => v + 1);
    } catch (err) {
      setError(err.message);
    }
  }

  // Enderezar la muestra (perspectiva con 4 puntos)
  function startQuad() {
    const b = border || FULL_BORDER;
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
      setBorder(validBorder(d.border)); // documento ya recto -> borde completo
      setImgRev((v) => v + 1);
      setQuadMode(false);
      setQuad(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function autoDetectBorder() {
    if (!doc) return;
    try {
      const b = await api.detectBorder(doc.id);
      changeBorder(validBorder(b));
    } catch (err) {
      setError(err.message);
    }
  }

  async function handleSuggest() {
    if (!doc) return;
    setSuggesting(true);
    setError("");
    setSuggestSource(null);
    try {
      const result = await api.suggestFields(doc.id);
      const suggestions = result.fields || [];
      setSuggestSource(result.source || "heuristic");

      if (suggestions.length === 0) {
        setError("No se detectaron campos automáticamente. Puedes crearlos manualmente.");
        return;
      }

      // Convertir sugerencias a campos relativos al borde y añadirlos
      const existingKeys = new Set(fields.map((f) => f.key));
      const newFields = [];
      for (const s of suggestions) {
        let key = s.key;
        let counter = 1;
        while (existingKeys.has(key)) {
          counter++;
          key = `${s.key}_${counter}`;
        }
        existingKeys.add(key);
        // Las sugerencias vienen en coords de imagen (0..1), convertir a relativas al borde
        const rel = imgToRel(
          { x: s.x, y: s.y, w: s.w, h: s.h },
          border,
        );
        newFields.push({
          key,
          name: s.name || key,
          data_type: s.data_type || "text",
          ...rel,
          sample_text: s.sample_text || "",
        });
      }
      setFields((prev) => [...prev, ...newFields]);
    } catch (err) {
      setError("Error al sugerir campos: " + err.message);
    } finally {
      setSuggesting(false);
    }
  }

  async function save() {
    setError("");
    if (!name.trim()) return setError("Ponle un nombre a la plantilla.");
    if (fields.length === 0) return setError("Añade al menos un campo.");
    setBusy(true);
    try {
      // Persiste el borde editado en el documento de muestra (recalcula su firma)
      let sig = signature;
      if (doc) {
        await api.updateBorder(doc.id, border);
        const refreshed = await fetch(`/api/documents/${doc.id}`).then((r) =>
          r.json()
        );
        sig = refreshed.signature;
      }
      const payload = {
        name: name.trim(),
        description,
        sample_document_id: doc?.id ?? null,
        signature: sig,
        border,
        fields,
        anchors: anchors.map((a) => ({
          name: a.name,
          x: a.x,
          y: a.y,
          w: a.w,
          h: a.h,
          anchor_text: a.anchor_text || "",
          use_text: a.use_text,
          use_image: a.use_image,
          weight: a.weight ?? 1.0,
        })),
      };
      if (editing) await api.updateTemplate(id, payload);
      else await api.createTemplate(payload);
      navigate("/");
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  // Regiones para el visor: campos + anclas (rel -> imagen) + pendiente (ya en imagen)
  const regions = [
    ...fields.map((f) => ({ key: f.key, name: f.name, ...relToImg(f, border) })),
    ...anchors.map((a) => ({
      key: ANCHOR_PREFIX + a.key,
      name: "⚓ " + a.name,
      className: "anchor",
      ...relToImg(a, border),
    })),
    ...(pending
      ? [
          {
            key: "__pending__",
            name: anchorMode ? "Ancla" : "Nuevo",
            className: anchorMode ? "anchor" : undefined,
            ...pending,
          },
        ]
      : []),
  ];

  const resultJson = Object.fromEntries(fields.map((f) => [f.key, f.sample_text]));

  return (
    <div className="editor">
      {suggesting && (
        <div className="loading-overlay">
          <div className="loading-spinner">
            <div className="spinner-ring" />
            <div className="spinner-ring spinner-ring-2" />
            <div className="spinner-ring spinner-ring-3" />
            <div className="spinner-icon">🧠</div>
          </div>
          <p className="loading-text">Analizando documento con IA…</p>
          <p className="loading-sub">Buscando campos de formulario automáticamente</p>
        </div>
      )}
      <div className="editor-left">
        {!doc ? (
          <div className="upload-zone">
            <h2>{editing ? "Editar plantilla" : "Entrenar nueva plantilla"}</h2>
            <p className="muted">
              Sube un documento tipo de muestra. Se ejecuta el OCR, se detecta el
              borde del documento y aparecen los marcadores de texto para vincular
              cada campo.
            </p>
            <Uploader onFile={handleSample} busy={busy} label="Subir documento de muestra" />
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
                <button className="btn small" onClick={() => rotate(-90)} title="Girar 90° izquierda">
                  ↺
                </button>
                <button className="btn small" onClick={() => rotate(90)} title="Girar 90° derecha">
                  ↻
                </button>
              </div>
              <div className="border-controls">
                {!quadMode && (
                  <button
                    className={"btn small" + (borderMode ? " primary" : "")}
                    onClick={() => setBorderMode((m) => !m)}
                  >
                    {borderMode ? "✓ Ajustando bordes" : "✎ Ajustar bordes"}
                  </button>
                )}
                {!borderMode && !quadMode && doc && (
                  <button
                    className="btn small"
                    onClick={handleSuggest}
                    disabled={suggesting}
                    title={
                      suggestSource === "ollama"
                        ? "🧠 Campos sugeridos por IA (Ollama)"
                        : "La IA analiza el OCR y propone campos automáticamente"
                    }
                  >
                    {suggesting ? "⏳" : suggestSource === "ollama" ? "🧠✓" : "🧠"} Sugerir campos
                  </button>
                )}
                {!borderMode && !quadMode && doc && (
                  <button
                    className={"btn small" + (anchorMode ? " primary" : "")}
                    onClick={() => {
                      setAnchorMode((m) => !m);
                      setPending(null);
                    }}
                    title="Marca hitos/anclas (texto fijo y/o trozo de imagen): ayudan a elegir plantilla, orientar y alinear"
                  >
                    ⚓ {anchorMode ? "Anclas ✓" : "Anclas"}
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
              showWords={!quadMode}
              zoom={zoom}
              onZoomChange={setZoom}
              panMode={panMode}
              border={quadMode ? null : border}
              editableBorder={borderMode && !quadMode}
              onBorderChange={changeBorder}
              editableRegions={!borderMode && !quadMode}
              onRegionChange={onFieldRegionChange}
              onRegionCommit={onFieldRegionCommit}
              quad={quadMode ? quad : null}
              editableQuad={quadMode}
              onQuadChange={setQuad}
            />
          </>
        )}
      </div>

      <aside className="editor-right">
        <div className="form-row">
          <label>Nombre de la plantilla</label>
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Ej. DNI español" />
        </div>
        <div className="form-row">
          <label>Descripción</label>
          <input value={description} onChange={(e) => setDescription(e.target.value)} placeholder="Opcional" />
        </div>

        <hr />

        {borderMode ? (
          <div className="hint">
            <strong>Modo bordes:</strong> arrastra un recuadro para redefinir el borde
            del documento, o mueve los tiradores de las esquinas. Los campos quedan
            proporcionales a este borde. Desactívalo para vincular campos.
          </div>
        ) : (
          <>
            <h3>Campos ({fields.length})</h3>
            <p className="muted small">
              {anchorMode
                ? "⚓ Modo anclas activo: dibuja un recuadro o pincha una palabra sobre un punto fijo (cabecera, logo, etiqueta) para crear un ancla."
                : "Pincha una palabra o arrastra un recuadro sobre el documento para seleccionar, luego nómbralo."}
            </p>

            {pending &&
              (anchorMode ? (
                <PendingAnchor
                  sampleText={pending.sample_text}
                  scanning={pending.scanning}
                  confidence={pending.confidence}
                  onConfirm={confirmAnchor}
                  onCancel={() => setPending(null)}
                />
              ) : (
                <PendingField
                  sampleText={pending.sample_text}
                  scanning={pending.scanning}
                  confidence={pending.confidence}
                  onConfirm={confirmField}
                  onCancel={() => setPending(null)}
                />
              ))}

            <ul className="field-list">
              {fields.map((f) => (
                <li
                  key={f.key}
                  className={f.key === activeKey ? "active" : ""}
                  onClick={() => setActiveKey(f.key)}
                >
                  <div>
                    <strong>{f.name}</strong>
                    <code>{f.key}</code>
                    <span className="muted small">{f.sample_text || "—"}</span>
                  </div>
                  <button className="btn danger small" onClick={() => removeField(f.key)}>
                    ✕
                  </button>
                </li>
              ))}
            </ul>

            <h3 className="anchor-head">⚓ Anclas / hitos ({anchors.length})</h3>
            <p className="muted small">
              Puntos fijos de referencia. Ayudan a elegir la plantilla correcta, a girar
              y enderezar el documento y a alinear los campos.
            </p>
            {anchors.length === 0 ? (
              <p className="muted small">
                Sin anclas. Activa «⚓ Anclas» arriba y marca puntos fijos.
              </p>
            ) : (
              <ul className="field-list anchor-list">
                {anchors.map((a) => (
                  <li
                    key={a.key}
                    className={ANCHOR_PREFIX + a.key === activeKey ? "active" : ""}
                    onClick={() => setActiveKey(ANCHOR_PREFIX + a.key)}
                  >
                    <div>
                      <strong>⚓ {a.name}</strong>
                      <code>
                        {a.use_text ? "texto" : ""}
                        {a.use_text && a.use_image ? "+" : ""}
                        {a.use_image ? "imagen" : ""}
                      </code>
                      <span className="muted small">{a.anchor_text || "—"}</span>
                    </div>
                    <button
                      className="btn danger small"
                      onClick={(e) => {
                        e.stopPropagation();
                        removeAnchor(a.key);
                      }}
                    >
                      ✕
                    </button>
                  </li>
                ))}
              </ul>
            )}

            <details className="json-preview">
              <summary>JSON resultado (muestra)</summary>
              <pre>{JSON.stringify(resultJson, null, 2)}</pre>
            </details>
          </>
        )}

        {error && <p className="error">{error}</p>}

        <div className="editor-actions">
          <button className="btn" onClick={() => navigate("/")}>
            Cancelar
          </button>
          <button className="btn primary" onClick={save} disabled={busy}>
            {editing ? "Guardar cambios" : "Crear plantilla"}
          </button>
        </div>
      </aside>
    </div>
  );
}

function PendingField({ sampleText, scanning, confidence, onConfirm, onCancel }) {
  const [name, setName] = useState("");
  return (
    <div className="pending-field">
      <div className="muted small">
        Texto capturado:{" "}
        {scanning ? <em>reconociendo…</em> : confidence != null && <span>(OCR {confidence}%)</span>}
      </div>
      <div className="captured">{sampleText || "(región sin texto)"}</div>
      <input
        autoFocus
        placeholder="Nombre del campo (ej. Número documento)"
        value={name}
        onChange={(e) => setName(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && onConfirm(name)}
      />
      <div className="row">
        <button className="btn primary small" onClick={() => onConfirm(name)}>
          Vincular campo
        </button>
        <button className="btn small" onClick={onCancel}>
          Cancelar
        </button>
      </div>
    </div>
  );
}

function PendingAnchor({ sampleText, scanning, confidence, onConfirm, onCancel }) {
  const [name, setName] = useState("");
  const [useText, setUseText] = useState(true);
  const [useImage, setUseImage] = useState(true);
  const hasText = Boolean((sampleText || "").trim());

  return (
    <div className="pending-field anchor-pending">
      <div className="muted small">
        ⚓ Nueva ancla — texto capturado:{" "}
        {scanning ? <em>reconociendo…</em> : confidence != null && <span>(OCR {confidence}%)</span>}
      </div>
      <div className="captured">{sampleText || "(región sin texto — usa imagen)"}</div>
      <input
        autoFocus
        placeholder="Nombre del ancla (ej. Cabecera, Logo)"
        value={name}
        onChange={(e) => setName(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && onConfirm(name, useText, useImage)}
      />
      <div className="anchor-flags">
        <label title="Casar por el texto OCR fijo de esta zona">
          <input
            type="checkbox"
            checked={useText && hasText}
            disabled={!hasText}
            onChange={(e) => setUseText(e.target.checked)}
          />{" "}
          Usar texto
        </label>
        <label title="Casar por el trozo de imagen (logo, sello, gráfico)">
          <input
            type="checkbox"
            checked={useImage}
            onChange={(e) => setUseImage(e.target.checked)}
          />{" "}
          Usar imagen
        </label>
      </div>
      <div className="row">
        <button
          className="btn primary small"
          disabled={!(useImage || (useText && hasText))}
          onClick={() => onConfirm(name, useText && hasText, useImage)}
        >
          Crear ancla
        </button>
        <button className="btn small" onClick={onCancel}>
          Cancelar
        </button>
      </div>
    </div>
  );
}
