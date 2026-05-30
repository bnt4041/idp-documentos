import { useRef, useState } from "react";

/**
 * Zona de subida con arrastrar y soltar + clic. Acepta imágenes y PDF.
 * Props: onFile(file), busy, label.
 */
export default function Uploader({ onFile, busy = false, label = "Subir documento" }) {
  const inputRef = useRef(null);
  const [over, setOver] = useState(false);

  function pick(file) {
    if (file) onFile(file);
  }

  return (
    <div
      className={"dropzone" + (over ? " over" : "") + (busy ? " busy" : "")}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        if (!busy) pick(e.dataTransfer.files?.[0]);
      }}
      onClick={() => !busy && inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/png,image/jpeg,image/webp,image/tiff,image/bmp,application/pdf"
        hidden
        onChange={(e) => pick(e.target.files?.[0])}
      />
      <div className="dropzone-icon">🖼️</div>
      <div className="dropzone-text">
        {busy ? (
          "Procesando OCR…"
        ) : (
          <>
            <strong>{label}</strong>
            <div className="muted small">
              Arrastra una imagen (JPG, PNG, WEBP, TIFF) o PDF, o haz clic
            </div>
          </>
        )}
      </div>
    </div>
  );
}
