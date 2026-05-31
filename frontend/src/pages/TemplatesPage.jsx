import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api.js";

export default function TemplatesPage() {
  const [templates, setTemplates] = useState([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  async function load() {
    setLoading(true);
    try {
      setTemplates(await api.listTemplates());
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function remove(id) {
    if (!confirm("¿Eliminar esta plantilla?")) return;
    try {
      await api.deleteTemplate(id);
      load();
    } catch (err) {
      alert("Error al eliminar: " + err.message);
    }
  }

  return (
    <div className="page">
      <div className="page-head">
        <h1>Plantillas</h1>
        <button className="btn primary" onClick={() => navigate("/plantillas/nueva")}>
          + Nueva plantilla
        </button>
      </div>

      {loading ? (
        <p>Cargando…</p>
      ) : templates.length === 0 ? (
        <div className="empty">
          <p>Todavía no hay plantillas.</p>
          <p>
            Crea una subiendo un documento tipo (DNI, factura, contrato…) y
            marcando sus campos.
          </p>
        </div>
      ) : (
        <div className="card-grid">
          {templates.map((t) => (
            <div className="card" key={t.id}>
              <div className="card-head">
                <h3>{t.name}</h3>
                <span className="badge ok" title="Ejemplos aprendidos (RAG)">
                  🧠 {t.example_count ?? 0}
                </span>
              </div>
              <p className="muted small">{t.description || "Sin descripción"}</p>

              {t.sample_document_id && (
                <img
                  className="tpl-thumb"
                  src={`/api/documents/${t.sample_document_id}/image`}
                  alt="muestra"
                />
              )}

              <div className="tpl-fields">
                {t.fields.map((f) => (
                  <div className="tpl-field" key={f.id}>
                    <code>{f.key}</code>
                    <span className="muted small">{f.sample_text || "—"}</span>
                  </div>
                ))}
              </div>

              <p className="muted small">
                {t.fields.length} campo(s) · {new Date(t.created_at).toLocaleDateString()}
              </p>
              <div className="card-actions">
                <Link className="btn" to={`/plantillas/${t.id}`}>
                  Editar
                </Link>
                <button className="btn danger" onClick={() => remove(t.id)}>
                  Eliminar
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
