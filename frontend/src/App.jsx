import { NavLink, Route, Routes } from "react-router-dom";
import TemplatesPage from "./pages/TemplatesPage.jsx";
import TemplateEditor from "./pages/TemplateEditor.jsx";
import ProcessPage from "./pages/ProcessPage.jsx";
import RecordsPage from "./pages/RecordsPage.jsx";

export default function App() {
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
