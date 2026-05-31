import { useEffect, useLayoutEffect, useRef, useState } from "react";

const CORNERS = ["nw", "ne", "sw", "se"];
const clamp01 = (v) => Math.max(0, Math.min(1, v));
const clampZoom = (z) => Math.max(0.25, Math.min(6, +z.toFixed(3)));
const pct = (b) => ({
  left: `${b.x * 100}%`,
  top: `${b.y * 100}%`,
  width: `${b.w * 100}%`,
  height: `${b.h * 100}%`,
});

/**
 * Visor con zoom por rueda (anclado al cursor), pan (mano / botón central) y
 * overlays en % (alineados a cualquier zoom). El documento y los campos se mueven
 * y escalan juntos.
 *
 * Props relevantes:
 *  - zoom, onZoomChange: zoom controlado por el padre.
 *  - panMode: si true, arrastrar mueve el documento en vez de dibujar.
 *  - border / editableBorder / onBorderChange: borde del documento.
 *  - onRegionDraw / onWordClick / onRegionClick: selección de campos.
 */
export default function DocumentViewer({
  imageUrl,
  words = [],
  regions = [],
  activeKey = null,
  onWordClick,
  onRegionDraw,
  onRegionClick,
  showWords = true,
  zoom = 1,
  onZoomChange,
  panMode = false,
  border = null,
  editableBorder = false,
  onBorderChange,
  editableRegions = false,
  onRegionChange,
  onRegionCommit,
  quad = null,
  editableQuad = false,
  onQuadChange,
}) {
  const scrollRef = useRef(null);
  const viewerRef = useRef(null);
  const anchorRef = useRef(null); // ancla del zoom por rueda
  const panRef = useRef(null); // estado de paneo
  const [drag, setDrag] = useState(null); // dibujo de región (normalizado)
  const [handle, setHandle] = useState(null); // esquina del borde en arrastre
  const [regionDrag, setRegionDrag] = useState(null); // mover/redimensionar campo
  const [quadHandle, setQuadHandle] = useState(null); // esquina del cuadrilátero
  const [panning, setPanning] = useState(false);

  // Zoom con la rueda, anclado en el punto bajo el cursor
  useEffect(() => {
    const sc = scrollRef.current;
    if (!sc || !onZoomChange) return;
    function onWheel(e) {
      e.preventDefault();
      const v = viewerRef.current;
      const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
      const next = clampZoom(zoom * factor);
      if (next === zoom) return;
      const scRect = sc.getBoundingClientRect();
      const cx = e.clientX - scRect.left;
      const cy = e.clientY - scRect.top;
      anchorRef.current = {
        fracX: (sc.scrollLeft + cx) / v.clientWidth,
        fracY: (sc.scrollTop + cy) / v.clientHeight,
        cx,
        cy,
      };
      onZoomChange(next);
    }
    sc.addEventListener("wheel", onWheel, { passive: false });
    return () => sc.removeEventListener("wheel", onWheel);
  }, [zoom, onZoomChange]);

  // Tras aplicar el nuevo zoom, recoloca el scroll para mantener el ancla
  useLayoutEffect(() => {
    const a = anchorRef.current;
    const sc = scrollRef.current;
    const v = viewerRef.current;
    if (!a || !sc || !v) return;
    sc.scrollLeft = a.fracX * v.clientWidth - a.cx;
    sc.scrollTop = a.fracY * v.clientHeight - a.cy;
    anchorRef.current = null;
  }, [zoom]);

  function relNorm(e) {
    const rect = viewerRef.current.getBoundingClientRect();
    return {
      x: clamp01((e.clientX - rect.left) / rect.width),
      y: clamp01((e.clientY - rect.top) / rect.height),
    };
  }

  function moveHandle(corner, p) {
    if (!border || !onBorderChange) return;
    const right = border.x + border.w;
    const bottom = border.y + border.h;
    let { x, y, w, h } = border;
    if (corner === "nw") {
      x = Math.min(p.x, right - 0.02);
      y = Math.min(p.y, bottom - 0.02);
      w = right - x;
      h = bottom - y;
    } else if (corner === "ne") {
      y = Math.min(p.y, bottom - 0.02);
      w = Math.max(0.02, p.x - border.x);
      h = bottom - y;
    } else if (corner === "sw") {
      x = Math.min(p.x, right - 0.02);
      w = right - x;
      h = Math.max(0.02, p.y - border.y);
    } else if (corner === "se") {
      w = Math.max(0.02, p.x - border.x);
      h = Math.max(0.02, p.y - border.y);
    }
    onBorderChange({
      x: +clamp01(x).toFixed(5),
      y: +clamp01(y).toFixed(5),
      w: +clamp01(w).toFixed(5),
      h: +clamp01(h).toFixed(5),
    });
  }

  function resizeBox(box, corner, p) {
    const right = box.x + box.w;
    const bottom = box.y + box.h;
    let { x, y, w, h } = box;
    if (corner === "nw") {
      x = Math.min(p.x, right - 0.01);
      y = Math.min(p.y, bottom - 0.01);
      w = right - x;
      h = bottom - y;
    } else if (corner === "ne") {
      y = Math.min(p.y, bottom - 0.01);
      w = Math.max(0.01, p.x - box.x);
      h = bottom - y;
    } else if (corner === "sw") {
      x = Math.min(p.x, right - 0.01);
      w = right - x;
      h = Math.max(0.01, p.y - box.y);
    } else {
      w = Math.max(0.01, p.x - box.x);
      h = Math.max(0.01, p.y - box.y);
    }
    return { x, y, w, h };
  }

  function startRegionDrag(key, mode, e) {
    const r = regions.find((rg) => rg.key === key);
    if (!r) return;
    setRegionDrag({
      key,
      mode,
      start: relNorm(e),
      box: { x: r.x, y: r.y, w: r.w, h: r.h },
    });
  }

  function startPan(e) {
    panRef.current = {
      x: e.clientX,
      y: e.clientY,
      sl: scrollRef.current.scrollLeft,
      st: scrollRef.current.scrollTop,
    };
    setPanning(true);
  }

  function handleMouseDown(e) {
    if (handle || quadHandle) return;
    // Botón central o modo mano -> paneo
    if (e.button === 1 || (panMode && e.button === 0)) {
      e.preventDefault();
      startPan(e);
      return;
    }
    if (e.button !== 0) return;
    if (editableQuad) return; // en modo cuadrilátero solo se mueven las esquinas
    if (!editableBorder && !onRegionDraw) return;
    const p = relNorm(e);
    setDrag({ x0: p.x, y0: p.y, x1: p.x, y1: p.y });
  }

  function handleMouseMove(e) {
    if (panRef.current) {
      scrollRef.current.scrollLeft = panRef.current.sl - (e.clientX - panRef.current.x);
      scrollRef.current.scrollTop = panRef.current.st - (e.clientY - panRef.current.y);
      return;
    }
    if (quadHandle) {
      const p = relNorm(e);
      onQuadChange &&
        onQuadChange({ ...quad, [quadHandle]: { x: +p.x.toFixed(5), y: +p.y.toFixed(5) } });
      return;
    }
    if (regionDrag) {
      const p = relNorm(e);
      let box;
      if (regionDrag.mode === "move") {
        const dx = p.x - regionDrag.start.x;
        const dy = p.y - regionDrag.start.y;
        box = {
          x: Math.max(0, Math.min(regionDrag.box.x + dx, 1 - regionDrag.box.w)),
          y: Math.max(0, Math.min(regionDrag.box.y + dy, 1 - regionDrag.box.h)),
          w: regionDrag.box.w,
          h: regionDrag.box.h,
        };
      } else {
        box = resizeBox(regionDrag.box, regionDrag.mode, p);
      }
      const rounded = {
        x: +clamp01(box.x).toFixed(5),
        y: +clamp01(box.y).toFixed(5),
        w: +box.w.toFixed(5),
        h: +box.h.toFixed(5),
      };
      onRegionChange && onRegionChange(regionDrag.key, rounded);
      return;
    }
    if (handle) {
      moveHandle(handle, relNorm(e));
      return;
    }
    if (!drag) return;
    const p = relNorm(e);
    setDrag((d) => ({ ...d, x1: p.x, y1: p.y }));
  }

  function handleMouseUp() {
    if (panRef.current) {
      panRef.current = null;
      setPanning(false);
      return;
    }
    if (regionDrag) {
      const key = regionDrag.key;
      setRegionDrag(null);
      onRegionCommit && onRegionCommit(key);
      return;
    }
    if (quadHandle) {
      setQuadHandle(null);
      return;
    }
    if (handle) {
      setHandle(null);
      return;
    }
    if (!drag) return;
    const box = {
      x: +Math.min(drag.x0, drag.x1).toFixed(5),
      y: +Math.min(drag.y0, drag.y1).toFixed(5),
      w: +Math.abs(drag.x1 - drag.x0).toFixed(5),
      h: +Math.abs(drag.y1 - drag.y0).toFixed(5),
    };
    setDrag(null);
    if (box.w < 0.01 || box.h < 0.01) return;
    if (editableBorder) onBorderChange && onBorderChange(box);
    else onRegionDraw && onRegionDraw(box);
  }

  const interactiveWords = showWords && !editableBorder && !panMode;
  const cursor = panMode ? (panning ? "grabbing" : "grab") : "crosshair";

  return (
    <div className="doc-scroll" ref={scrollRef}>
      <div
        className="doc-viewer"
        ref={viewerRef}
        style={{ width: `${zoom * 100}%`, cursor }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={() =>
          (drag || handle || panRef.current || regionDrag || quadHandle) && handleMouseUp()
        }
      >
        <img src={imageUrl} alt="documento" draggable={false} />

        {/* Cuadrilátero de perspectiva (4 puntos) */}
        {quad && (
          <svg className="quad-overlay" viewBox="0 0 1 1" preserveAspectRatio="none">
            <polygon
              points={["tl", "tr", "br", "bl"]
                .map((k) => `${quad[k].x},${quad[k].y}`)
                .join(" ")}
            />
          </svg>
        )}
        {quad &&
          editableQuad &&
          ["tl", "tr", "br", "bl"].map((k) => (
            <span
              key={k}
              className="quad-handle"
              style={{ left: `${quad[k].x * 100}%`, top: `${quad[k].y * 100}%` }}
              onMouseDown={(e) => {
                e.stopPropagation();
                setQuadHandle(k);
              }}
            />
          ))}

        {border && (
          <div
            className={"doc-border" + (editableBorder ? " editable" : "")}
            style={pct(border)}
          >
            {editableBorder &&
              CORNERS.map((c) => (
                <span
                  key={c}
                  className={"handle " + c}
                  onMouseDown={(e) => {
                    e.stopPropagation();
                    setHandle(c);
                  }}
                />
              ))}
          </div>
        )}

        {showWords &&
          words.map((wd, i) => (
            <div
              key={i}
              className="ocr-word"
              title={`${wd.text} (${wd.conf}%)`}
              style={{ ...pct(wd.box), pointerEvents: interactiveWords ? "auto" : "none" }}
              onClick={(e) => {
                e.stopPropagation();
                interactiveWords && onWordClick && onWordClick(wd);
              }}
            />
          ))}

        {regions.map((r) => {
          const active = r.key === activeKey;
          const editable =
            editableRegions &&
            r.key !== "__pending__" &&
            !r.readOnly &&
            !panMode &&
            !editableBorder;
          return (
            <div
              key={r.key}
              className={
                "region" +
                (active ? " active" : "") +
                (editable ? " editable" : "") +
                (r.className ? " " + r.className : "")
              }
              style={{
                ...pct(r),
                pointerEvents: editableBorder || panMode ? "none" : "auto",
                cursor: editable ? "move" : "pointer",
              }}
              onMouseDown={
                editable
                  ? (e) => {
                      e.stopPropagation();
                      startRegionDrag(r.key, "move", e);
                    }
                  : undefined
              }
              onClick={(e) => {
                e.stopPropagation();
                onRegionClick && onRegionClick(r.key);
              }}
            >
              <span className="region-label">{r.name || r.key}</span>
              {editable &&
                CORNERS.map((c) => (
                  <span
                    key={c}
                    className={"handle " + c}
                    onMouseDown={(e) => {
                      e.stopPropagation();
                      startRegionDrag(r.key, c, e);
                    }}
                  />
                ))}
            </div>
          );
        })}

        {drag && (
          <div
            className={"region drawing" + (editableBorder ? " border-draw" : "")}
            style={pct({
              x: Math.min(drag.x0, drag.x1),
              y: Math.min(drag.y0, drag.y1),
              w: Math.abs(drag.x1 - drag.x0),
              h: Math.abs(drag.y1 - drag.y0),
            })}
          />
        )}
      </div>
    </div>
  );
}
