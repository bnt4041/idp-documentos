# Hitos / anclas de plantilla y rectificación a la plantilla

Documentación de la funcionalidad de **anclas** (hitos) y del nuevo **pipeline de
rectificación**: enderezar la imagen del documento al marco de la plantilla y extraer por
proporcionalidad. Cubre el diseño, el flujo de datos, los ficheros implicados y los
parámetros de configuración.

---

## 1. Idea general

Antes, la extracción dependía de transformar las coordenadas de cada campo con una afín de
anclas — frágil, dependía de un `detect_border` poco fiable y dejaba las cajas torcidas al
editar. El enfoque actual **invierte** el problema:

> Localizar las anclas → estimar la transformación documento → muestra → **rectificar la
> imagen físicamente** (girar, enderezar, escalar al marco de la plantilla) → extraer los
> campos por **proporcionalidad simple** sobre el borde de la plantilla.

Tras rectificar, el documento guardado queda "encajado" a la muestra de la plantilla, así
que:

- el **borde** del documento pasa a ser el borde de la plantilla (`tpl.border`),
- los **campos** se extraen con `field_to_image` + `extract_field` (sin transformar
  coordenadas una a una),
- al **editar**, el documento aparece ya orientado y enderezado, con las zonas
  proporcionadas.

---

## 2. Qué es un ancla

Un ancla (`TemplateAnchor`) es una **zona fija de referencia** de la plantilla (coords
normalizadas 0..1 relativas al borde, igual que un campo). Sirve para elegir la plantilla,
orientar/enderezar el documento y validar la alineación.

Campos del modelo ([backend/app/models.py](../backend/app/models.py)):

| Campo | Significado |
| --- | --- |
| `name` | Nombre legible (p.ej. "OBSERVACIONES", "Logo") |
| `x, y, w, h` | Región normalizada relativa al borde de la plantilla |
| `anchor_text` | Texto fijo esperado en la zona (cabeceras como "OBSERVACIONES:") |
| `use_text` | Casar por texto OCR (preciso si el texto es fijo) |
| `use_image` | Casar por trozo de imagen (logo, sello) con ORB |
| `required` | Si **obligatoria**, debe localizarse para rectificar; si no, revisión manual |
| `weight` | Importancia relativa en el score de anclas |

**Recomendaciones de uso**
- Define **2–3 anclas obligatorias bien repartidas** (esquinas/extremos): cuanta más
  separación, más estable la orientación.
- Prioriza **anclas de texto** sobre cabeceras fijas (idénticas entre documentos del mismo
  tipo) — se localizan con más precisión que un parche de imagen.
- Usa **imagen** para logos/sellos sin texto. Se pueden combinar texto+imagen.

---

## 3. El pipeline de procesado (paso a paso)

Toda la lógica vive en `_match_and_extract` ([backend/app/routers/records.py](../backend/app/routers/records.py)),
que usan **tanto** el endpoint interactivo `POST /api/process/{id}` como el job de fondo
`_run_job`.

1. **Elegir plantilla** — firma geométrica + similitud visual ORB + score de anclas
   (`_apply_anchor_selection`). Las anclas pueden corregir una elección dudosa.
2. **Localizar anclas robusto** — `anchors.locate_anchors_robust` prueba las 4
   orientaciones (0/90/180/270) y, para anclas de imagen, filtros en cascada
   (gris → binario → CLAHE). Elige el ángulo donde más anclas casan **y** la rectificación
   es geométricamente plausible.
3. **¿Obligatorias localizadas?** — `anchors.required_ok`. Si falta alguna obligatoria, no
   se toca la imagen y el registro queda en **`review`** (aviso en el panel).
4. **Rectificar a la plantilla** — `anchors.rectify_to_template`: warp por **similitud
   rígida ORB de página completa** (rotación + escala uniforme + traslación, sin cizalla).
   Corrige el giro automáticamente (incluido 180°). Refinamientos:
   - **Pre-escalado**: si muestra y documento difieren mucho de tamaño (p.ej. DNI 532px vs
     escaneo 2339px), el documento se reescala antes de ORB para que empareje bien.
   - **ECC** (Enhanced Correlation Coefficient): afina el residual de rotación/traslación.
   - **Refinamiento por anclas de texto**: corrige el desplazamiento residual usando la
     posición de las cabeceras fijas.
5. **Persistir** la imagen rectificada (`_reocr_persist`): re-OCR + recálculo de borde y
   firma; `doc.border = tpl.border`.
6. **Extraer campos** — `matching.extract_all` por proporcionalidad sobre `tpl.border`.
7. **Re-OCR de campos flojos** — `_refine_fields_with_region_ocr`: para campos vacíos o de
   baja confianza, re-OCR de alta resolución (recorta la región, la amplía ×4, PSM 6).

El resultado (`MatchResult`) incluye: `aligned`, `needs_review`, `pipeline` (traza
legible), `anchors` (con `region` y `expected_region`), `sample_document_id`.

---

## 4. Detalle de los algoritmos clave (anchors.py)

[backend/app/anchors.py](../backend/app/anchors.py)

- **`find_text_anchor(anchor_text, words, threshold)`** — busca en todo el OCR la ventana de
  palabras que mejor casa con el texto del ancla (`difflib.SequenceMatcher`).
- **`find_image_anchor(...)`** — recorta el patch del ancla de la muestra y lo localiza con
  ORB (invariante a rotación/escala). La región se normaliza al **tamaño esperado** del
  parche centrada en la posición encontrada (el bbox de ORB es inestable y deforma cajas).
- **`locate_anchors_robust(db, doc, tpl, image)`** — multi-ángulo × multi-filtro. Siembra la
  orientación con `preprocessing.best_rotation_by_orb` (aspect ratio + ORB) y valida cada
  ángulo por rectificabilidad. Devuelve la imagen en el mejor ángulo, las anclas
  localizadas, `sample_size` y `sample_img`.
- **`rectify_with_homography(image, sample_img, sample_size, info)`** — el warp principal.
  Usa **similitud rígida** (`estimateAffinePartial2D`), **no** homografía de perspectiva
  (que cizallaría). Rellena `info` con la traza (rotation, scale, inliers).
- **`rectify_to_template(...)`** — orquesta el warp (con pre-giro 90/270 si el aspect ratio
  lo indica) y el refinamiento por anclas de texto. Cae a afín por centros de anclas si el
  warp global falla.
- **`required_ok(located)`** — True si todas las anclas obligatorias se localizaron.

> **Por qué similitud y no homografía**: un documento (permiso, DNI) es una hoja plana, sin
> perspectiva real. Una homografía de 8 grados de libertad, estimada entre dos documentos
> distintos, introduce cizalla/perspectiva falsa (líneas de tabla en diagonal). La similitud
> (4 dof) solo gira, escala y traslada: nunca deforma.

---

## 5. Frontend

### Editor de plantilla — [frontend/src/pages/TemplateEditor.jsx](../frontend/src/pages/TemplateEditor.jsx)
- Botón **⚓ Anclas**: en ese modo, dibujar una zona o pinchar una palabra crea un ancla.
- Panel `PendingAnchor`: nombre + casillas *Usar texto* / *Usar imagen* / *Obligatoria*.
- Las anclas se dibujan en morado punteado y se listan aparte de los campos (badge "oblig.").

### Editor de revisión — [frontend/src/pages/ProcessPage.jsx](../frontend/src/pages/ProcessPage.jsx)
- **Selector de plantilla**: corrige la auto-detección y re-procesa al instante.
- **Traza del pipeline**: chips bajo el estado (p.ej. `Girado/enderezado +180°`,
  `Escalado ×0.23`, `Alineado a la plantilla`).
- **Badge `n/m anclas`** + botón **⚓ Ver anclas** → modal `AnchorFootprintModal`: por cada
  ancla muestra, lado a lado, el recorte de la **plantilla** (esperada) y el del
  **documento** (detectada). Los recortes se generan al vuelo en el cliente (`CropView`).
- **Aviso de revisión** cuando faltan anclas obligatorias.
- Overlay de anclas sobre el documento: si `aligned`, se dibujan en su posición de
  plantilla (`expected_region`); si no, en la caja ORB encontrada (`region`).

---

## 6. Configuración — [backend/app/config.py](../backend/app/config.py)

| Parámetro | Default | Para qué |
| --- | --- | --- |
| `anchor_match_weight` | 0.5 | Peso de las anclas en el score combinado de plantilla |
| `anchor_text_threshold` | 0.7 | Umbral SequenceMatcher para dar texto por casado |
| `anchor_image_threshold` | 0.3 | Umbral de localización del patch de imagen |
| `anchor_fit_max_error` | 0.15 | Error de reproyección relativo máximo de la afín por anclas |
| `anchor_max_anisotropy` | 6.0 | Ratio máx escala_x/escala_y (rechaza afín deformada) |
| `anchor_max_shear` | 0.2 | Cizalla máxima admitida (alta = otro formulario) |
| `anchor_filters` | gris,binario,clahe | Filtros en cascada para anclas de imagen difíciles |
| `region_ocr_refine` | true | Activa el re-OCR por región de campos flojos |
| `region_ocr_min_conf` | 80 | Por debajo de esta confianza se re-OCR el campo |

---

## 7. Base de datos

La tabla `template_anchors` la crea `Base.metadata.create_all` ([backend/app/main.py](../backend/app/main.py)).

> ⚠️ `create_all` **no altera** tablas ya existentes. La columna `required` se añadió en
> caliente con:
> ```sql
> ALTER TABLE template_anchors ADD COLUMN IF NOT EXISTS required BOOLEAN NOT NULL DEFAULT false;
> ```
> En un despliegue desde cero no hace falta; en uno existente, ejecutar ese ALTER (o un
> Alembic equivalente).

`POST /api/reset` borra también las anclas.

---

## 8. Verificación rápida

```bash
docker compose up --build -d backend frontend
```

1. Crear plantilla con muestra recta y **2 anclas obligatorias** de texto bien separadas.
2. Subir el mismo documento girado 90/180/270 y/o inclinado:
   - El documento debe quedar **rectificado** (mismas dims que la muestra),
     `border = tpl.border`, campos extraídos por proporcionalidad.
   - `POST /api/process/{id}?template_id=…` → `aligned: true`, `pipeline` con la traza.
3. Abrir en el editor: documento **derecho**, zonas proporcionadas, "⚓ Ver anclas" muestra
   plantilla vs detectada alineadas.
4. Documento de otro tipo (faltan anclas obligatorias) → estado `review`, imagen sin tocar.

---

## 9. Limitaciones conocidas

- **Filas "flotantes"**: algunos formularios (permiso de circulación) tienen celdas
  superiores de altura variable que desplazan las filas inferiores (marca/modelo) según el
  contenido. La rectificación alinea perfectamente las zonas **fijas** (cabeceras,
  OBSERVACIONES, rejilla), pero el contenido variable puede caer ±1 fila. Mitigaciones:
  definir esas cajas algo más altas, y/o el re-OCR por región recupera el valor aunque la
  caja no clave al píxel.
- **Muestra de baja resolución**: si la muestra de la plantilla es pequeña, los recortes y
  el OCR tras el warp pierden nitidez. Re-entrenar con una muestra de resolución similar a
  los documentos reales mejora el resultado.
- **Orientación del entrenamiento**: conviene entrenar la plantilla en la orientación
  habitual de llegada de los documentos (apaisado/retrato) para un giro más estable.
