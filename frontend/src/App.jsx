import { useState } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import TemplatesPage from "./pages/TemplatesPage.jsx";
import TemplateEditor from "./pages/TemplateEditor.jsx";
import ProcessPage from "./pages/ProcessPage.jsx";
import RecordsPage from "./pages/RecordsPage.jsx";
import { api } from "./api.js";

export default function App() {
  const [resetting, setResetting] = useState(false);

  async function handleReset() {
    if (
      !confirm(
        "⚠️ ¿Eliminar TODOS los datos?\n\n" +
          "Se borrarán documentos, plantillas, registros y ejemplos de aprendizaje.\n" +
          "Esta acción NO se puede deshacer."
      )
    )
      return;
    if (
      !confirm(
        "¿Estás COMPLETAMENTE seguro?\n\nPresiona Aceptar para confirmar el reseteo total."
      )
    )
      return;

    setResetting(true);
    try {
      const result = await api.resetAll();
      alert(result.message || "Datos eliminados correctamente.");
      window.location.href = "/";
    } catch (err) {
      alert("Error al resetear: " + err.message);
      setResetting(false);
    }
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">📄 IDP Visual</div>
        <nav>
          <NavLink to="/" end>
            Plantillas
          </NavLink>
          <NavLink to="/procesar">Procesar documento</NavLink>
          <NavLink to="/registros">Registros</NavLink>
        </nav>
        <div className="topbar-right">
          <button
            className="btn-reset"
            onClick={handleReset}
            disabled={resetting}
            title="Resetear todos los datos (MVP)"
          >
            {resetting ? "⏳" : "🗑️"} Reset
          </button>
        </div>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<TemplatesPage />} />
          <Route path="/plantillas/nueva" element={<TemplateEditor />} />
          <Route path="/plantillas/:id" element={<TemplateEditor />} />
          <Route path="/procesar" element={<ProcessPage />} />
          <Route path="/registros" element={<RecordsPage />} />
        </Routes>
      </main>
    </div>
  );
}
