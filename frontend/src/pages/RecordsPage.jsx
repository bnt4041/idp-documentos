import { useEffect, useState } from "react";
import { api } from "../api.js";

export default function RecordsPage() {
  const [records, setRecords] = useState([]);
  const [templates, setTemplates] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([api.listRecords(), api.listTemplates()]).then(
      ([recs, tpls]) => {
        setRecords(recs);
        setTemplates(tpls);
        setLoading(false);
      }
    );
  }, []);

  const tplName = (id) =>
    templates.find((t) => t.id === id)?.name || "—";

  return (
    <div className="page">
      <div className="page-head">
        <h1>Registros extraídos</h1>
      </div>
      {loading ? (
        <p>Cargando…</p>
      ) : records.length === 0 ? (
        <div className="empty">
          <p>Aún no hay registros. Procesa un documento para crear el primero.</p>
        </div>
      ) : (
        <table className="records-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Plantilla</th>
              <th>Datos</th>
              <th>Confianza</th>
              <th>Fecha</th>
            </tr>
          </thead>
          <tbody>
            {records.map((r) => (
              <tr key={r.id}>
                <td>{r.id}</td>
                <td>{tplName(r.template_id)}</td>
                <td>
                  <div className="kv">
                    {Object.entries(r.data).map(([k, v]) => (
                      <span key={k}>
                        <code>{k}</code>: {String(v)}
                      </span>
                    ))}
                  </div>
                </td>
                <td>{(r.match_score * 100).toFixed(0)}%</td>
                <td>{new Date(r.created_at).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
