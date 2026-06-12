import { useEffect, useState } from "react";
import { api } from "../api.js";

export default function ConfigPage() {
  const [config, setConfig] = useState(null);
  const [tokens, setTokens] = useState(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState("");

  // Campos editables
  const [deepseekEnabled, setDeepseekEnabled] = useState(false);
  const [deepseekApiKey, setDeepseekApiKey] = useState("");
  const [deepseekModel, setDeepseekModel] = useState("deepseek-chat");
  const [deepseekBaseUrl, setDeepseekBaseUrl] = useState("https://api.deepseek.com");
  const [ollamaModel, setOllamaModel] = useState("llama3.2");
  const [ollamaUrl, setOllamaUrl] = useState("http://ollama:11434");
  const [ollamaGenUrl, setOllamaGenUrl] = useState("");
  const [ollamaGenApiKey, setOllamaGenApiKey] = useState("");
  const [ollamaVision, setOllamaVision] = useState(false);

  async function load() {
    setLoading(true);
    try {
      const [c, t] = await Promise.all([
        api.getAIConfig(),
        api.getTokenStats(),
      ]);
      setConfig(c);
      setTokens(t);
      setDeepseekEnabled(c.deepseek_enabled);
      setDeepseekModel(c.deepseek_model);
      setDeepseekBaseUrl(c.deepseek_base_url);
      setOllamaModel(c.ollama_model);
      setOllamaUrl(c.ollama_url);
      setOllamaGenUrl(c.ollama_gen_url || "");
      setOllamaVision(c.ollama_vision);
    } catch (err) {
      setMsg("Error al cargar configuración: " + err.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { load(); }, []);

  async function handleSave(e) {
    e.preventDefault();
    setSaving(true);
    setMsg("");
    try {
      const payload = {
        deepseek_enabled: deepseekEnabled,
        deepseek_model: deepseekModel || "deepseek-chat",
        deepseek_base_url: deepseekBaseUrl || "https://api.deepseek.com",
        ollama_model: ollamaModel || "llama3.2",
        ollama_url: ollamaUrl || "http://ollama:11434",
        ollama_gen_url: ollamaGenUrl || "",
        ollama_vision: ollamaVision,
      };
      if (deepseekApiKey) payload.deepseek_api_key = deepseekApiKey;
      if (ollamaGenApiKey) payload.ollama_gen_api_key = ollamaGenApiKey;

      const updated = await api.updateAIConfig(payload);
      setConfig(updated);
      if (deepseekApiKey) setDeepseekApiKey("");
      if (ollamaGenApiKey) setOllamaGenApiKey("");
      setMsg("✅ Configuración guardada correctamente.");
      setTimeout(() => setMsg(""), 4000);
    } catch (err) {
      setMsg("❌ Error al guardar: " + err.message);
    } finally {
      setSaving(false);
    }
  }

  async function handleResetTokens() {
    if (!confirm("¿Resetear el contador de tokens a cero?")) return;
    try {
      await api.resetTokenStats();
      const t = await api.getTokenStats();
      setTokens(t);
      setMsg("✅ Contador de tokens reseteado.");
      setTimeout(() => setMsg(""), 3000);
    } catch (err) {
      setMsg("❌ Error: " + err.message);
    }
  }

  if (loading) return <p className="muted">Cargando configuración…</p>;

  const backend = config?.backend || "ollama";
  const genOk = config?.gen_available;
  const embedOk = config?.embed_available;

  return (
    <div className="page">
      <div className="page-head">
        <h1>⚙️ Configuración</h1>
      </div>

      {/* Estado actual */}
      <div className="status-bar">
        <span className={"status-dot " + (genOk ? "on" : "off")} />
        <span>
          Generación: <strong>{backend === "deepseek" ? "DeepSeek ☁️" : "Ollama 🖥️"}</strong>
          {genOk ? " (conectado)" : " (sin conexión)"}
        </span>
        <span className="sep">|</span>
        <span className={"status-dot " + (embedOk ? "on" : "off")} />
        <span>Embeddings: {embedOk ? "conectado" : "sin conexión"}</span>
        {config?.gen_models?.length > 0 && (
          <>
            <span className="sep">|</span>
            <span className="muted small">
              Modelos: {config.gen_models.slice(0, 5).join(", ")}
            </span>
          </>
        )}
      </div>

      {/* ============================================================ */}
      {/* Panel de consumo de tokens */}
      {/* ============================================================ */}
      <div className="token-panel">
        <div className="token-panel-head">
          <h3>📊 Consumo de tokens</h3>
          <button className="btn small" onClick={handleResetTokens} title="Resetear contador">
            ↺ Reset
          </button>
        </div>

        {tokens && tokens.docs > 0 ? (
          <div className="token-cards">
            <div className="token-card">
              <span className="token-value">{tokens.docs}</span>
              <span className="token-label">documentos procesados</span>
            </div>
            <div className="token-card">
              <span className="token-value">{tokens.tokens_in.toLocaleString()}</span>
              <span className="token-label">tokens de entrada</span>
            </div>
            <div className="token-card">
              <span className="token-value">{tokens.tokens_out.toLocaleString()}</span>
              <span className="token-label">tokens de salida</span>
            </div>
            <div className="token-card">
              <span className="token-value">{tokens.avg_in} / {tokens.avg_out}</span>
              <span className="token-label">media in / out por documento</span>
            </div>
            <div className="token-card cost">
              <span className="token-value">{tokens.cost_estimate}</span>
              <span className="token-label">
                coste estimado {backend === "deepseek" ? "(DeepSeek V3)" : "(referencia DeepSeek)"}
              </span>
            </div>
          </div>
        ) : (
          <p className="muted small" style={{ margin: 0 }}>
            Todavía no se ha procesado ningún documento con IA. El contador se actualiza
            automáticamente al extraer campos con el LLM.
          </p>
        )}

        {tokens?.last_model && (
          <p className="muted small" style={{ margin: "0.4rem 0 0" }}>
            Último modelo usado: <code>{tokens.last_model}</code>
          </p>
        )}
      </div>

      <form onSubmit={handleSave}>
        {/* ============================================================ */}
        {/* DeepSeek */}
        {/* ============================================================ */}
        <fieldset className="config-section">
          <legend>
            ☁️ DeepSeek API{" "}
            {backend === "deepseek" && <span className="badge ok">activo</span>}
          </legend>
          <p className="muted small">
            API cloud de DeepSeek (compatible OpenAI). Precio: ~0,27 $/M tokens input,
            ~1,10 $/M tokens output. Actívalo para usar deepseek-chat (V3) en lugar del
            Ollama local o remoto. Los embeddings seguirán usando el Ollama local.
          </p>

          <div className="form-row">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={deepseekEnabled}
                onChange={(e) => setDeepseekEnabled(e.target.checked)}
              />
              Activar DeepSeek como backend de generación
            </label>
          </div>

          <div className="form-row">
            <label>Clave API (sk-…)</label>
            <input
              type="password"
              value={deepseekApiKey}
              onChange={(e) => setDeepseekApiKey(e.target.value)}
              placeholder={config?.deepseek_api_key_set ? "•••••••• (ya configurada)" : "sk-…"}
            />
            {config?.deepseek_api_key_set && !deepseekApiKey && (
              <span className="muted small">🔑 Clave ya configurada (no se muestra)</span>
            )}
          </div>

          <div className="form-row form-row-2col">
            <div>
              <label>Modelo</label>
              <select
                value={deepseekModel}
                onChange={(e) => setDeepseekModel(e.target.value)}
              >
                <option value="deepseek-chat">deepseek-chat (V3)</option>
                <option value="deepseek-reasoner">deepseek-reasoner (R1)</option>
              </select>
            </div>
            <div>
              <label>URL base</label>
              <input
                type="text"
                value={deepseekBaseUrl}
                onChange={(e) => setDeepseekBaseUrl(e.target.value)}
              />
            </div>
          </div>
        </fieldset>

        {/* ============================================================ */}
        {/* Ollama */}
        {/* ============================================================ */}
        <fieldset className="config-section">
          <legend>
            🖥️ Ollama{" "}
            {backend === "ollama" && <span className="badge ok">activo</span>}
          </legend>
          <p className="muted small">
            Configuración del backend Ollama (local o remoto). Si DeepSeek está activo,
            estos valores solo se usan para los embeddings.
          </p>

          <div className="form-row form-row-2col">
            <div>
              <label>URL del Ollama local (embeddings)</label>
              <input
                type="text"
                value={ollamaUrl}
                onChange={(e) => setOllamaUrl(e.target.value)}
              />
            </div>
            <div>
              <label>Modelo de lenguaje</label>
              <input
                type="text"
                value={ollamaModel}
                onChange={(e) => setOllamaModel(e.target.value)}
                placeholder="llama3.2"
              />
            </div>
          </div>

          <div className="form-row form-row-2col">
            <div>
              <label>URL de generación remota (opcional)</label>
              <input
                type="text"
                value={ollamaGenUrl}
                onChange={(e) => setOllamaGenUrl(e.target.value)}
                placeholder="https://bot.dealerbest.com/ollama"
              />
            </div>
            <div>
              <label>API Key generación remota</label>
              <input
                type="password"
                value={ollamaGenApiKey}
                onChange={(e) => setOllamaGenApiKey(e.target.value)}
                placeholder={config?.ollama_gen_api_key_set ? "•••••••• (ya configurada)" : "Bearer…"}
              />
            </div>
          </div>

          <div className="form-row">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={ollamaVision}
                onChange={(e) => setOllamaVision(e.target.checked)}
              />
              Enviar imagen del documento al modelo (solo si el modelo soporta visión)
            </label>
          </div>
        </fieldset>

        {/* Botón guardar */}
        <div className="form-actions">
          <button type="submit" className="btn primary" disabled={saving}>
            {saving ? "⏳ Guardando…" : "💾 Guardar configuración"}
          </button>
          {msg && <span className={msg.startsWith("✅") ? "msg-ok" : "msg-error"}>{msg}</span>}
        </div>
      </form>
    </div>
  );
}
