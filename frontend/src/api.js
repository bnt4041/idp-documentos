// Cliente ligero de la API. En dev Vite hace proxy de /api -> backend:8000.
const BASE = "";

async function http(path, options = {}) {
  const res = await fetch(BASE + path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      /* noop */
    }
    throw new Error(detail);
  }
  if (res.status === 204) return null;
  return res.json();
}

export const api = {
  // Documentos
  uploadDocument(file) {
    const fd = new FormData();
    fd.append("file", file);
    return http("/api/documents", { method: "POST", body: fd });
  },
  documentImageUrl: (id) => `/api/documents/${id}/image`,
  ocrRegion: (id, box) =>
    http(`/api/documents/${id}/ocr-region`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(box),
    }),
  rotateDocument: (id, degrees) =>
    http(`/api/documents/${id}/rotate?degrees=${degrees}`, { method: "POST" }),
  rectifyDocument: (id, quad) =>
    http(`/api/documents/${id}/rectify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(quad),
    }),
  detectBorder: (id) =>
    http(`/api/documents/${id}/detect-border`, { method: "POST" }),
  updateBorder: (id, box) =>
    http(`/api/documents/${id}/border`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(box),
    }),

  // Plantillas
  listTemplates: () => http("/api/templates"),
  getTemplate: (id) => http(`/api/templates/${id}`),
  createTemplate: (payload) =>
    http("/api/templates", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  updateTemplate: (id, payload) =>
    http(`/api/templates/${id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  deleteTemplate: (id) => http(`/api/templates/${id}`, { method: "DELETE" }),

  // Procesado y registros
  processDocument: (docId, templateId) =>
    http(
      `/api/process/${docId}` +
        (templateId ? `?template_id=${templateId}` : ""),
      { method: "POST" }
    ),
  createRecord: (payload) =>
    http("/api/records", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  listRecords: (templateId) =>
    http("/api/records" + (templateId ? `?template_id=${templateId}` : "")),

  // IA / RAG (Ollama)
  aiStatus: () => http("/api/ai/status"),
  aiExtract: (docId, templateId) =>
    http(`/api/ai/extract/${docId}?template_id=${templateId}`, { method: "POST" }),
};
