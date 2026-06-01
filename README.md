# IDP Visual — Procesamiento Inteligente de Documentos

Plataforma de **IDP (Intelligent Document Processing)** con OCR local que permite:

1. **Entrenar plantillas visualmente**: subes un documento tipo (DNI, factura, contrato…),
   se ejecuta OCR (Tesseract) y aparecen marcadores de texto sobre la imagen. Pinchas o
   arrastras para vincular cada zona a un **campo**, y se va montando el JSON resultado.
2. **Reconocer la forma**: cada plantilla guarda una *firma geométrica* (proporción,
   color de fondo y organización del texto en rejilla) para emparejar documentos futuros.
3. **Procesar documentos**: subes uno nuevo, se auto-detecta la plantilla, se rellenan
   los campos entrenados, los revisas/corriges y creas el registro con sus datos.
4. **APIs REST** para integrar el servicio.

Todo corre en **Docker** (sin enviar datos a servicios externos).

## Arquitectura

| Servicio   | Tecnología                          | Puerto host |
| ---------- | ----------------------------------- | ----------- |
| `frontend` | React + Vite, servido por Nginx     | 8090        |
| `backend`  | FastAPI + Tesseract + OpenCV        | 8088        |
| `db`       | PostgreSQL 16                       | 5433        |
| `ollama`   | LLM local (RAG) — embeddings + chat | 11434       |

```
idp-visual/
├── docker-compose.yml
├── backend/            # FastAPI: OCR, matching, extracción, APIs
│   └── app/
│       ├── ocr.py          # Tesseract + firma geométrica
│       ├── matching.py     # similitud de plantillas + extracción por región
│       ├── models.py       # Document, Template, TemplateField, Record
│       └── routers/        # documents, templates, records/process
└── frontend/           # React: visor visual, editor y procesado
    └── src/
        ├── components/DocumentViewer.jsx
        └── pages/      # TemplatesPage, TemplateEditor, ProcessPage, RecordsPage
```

## Arranque

Requisitos: **Docker** y **Docker Compose**.

```bash
cp .env.example .env      # (Windows PowerShell: copy .env.example .env)
docker compose up --build
```

- Interfaz visual: <http://localhost:8090>
- API + documentación interactiva (Swagger): <http://localhost:8088/docs>

La primera build tarda un poco (descarga Tesseract con idiomas spa+eng).

### IA local (RAG con Ollama)

El extractor con IA usa un **LLM local** vía Ollama (sin enviar datos fuera). Tras
levantar el stack, descarga los modelos una sola vez:

```bash
docker compose exec ollama ollama pull nomic-embed-text   # embeddings (retrieval)
docker compose exec ollama ollama pull llama3.2            # extracción (texto)
```

Comprueba el estado en <http://localhost:8088/api/ai/status> (`"ready": true`).

- Para usar **visión** (enviar la imagen al modelo), pon `OLLAMA_VISION=true` y un
  modelo con visión, p. ej. `OLLAMA_MODEL=llama3.2-vision` (`ollama pull llama3.2-vision`).
- Modelo configurable con `OLLAMA_MODEL` / `OLLAMA_EMBED_MODEL` en `.env`.

**Cómo aprende:** al confirmar un registro, el documento se guarda como ejemplo
(dataset) con su texto OCR y embedding; las zonas que hayas movido **actualizan las
posiciones de los campos de la plantilla** (y crean campos nuevos si los añadiste); y
al procesar otro documento se recuperan los ejemplos confirmados más parecidos como
contexto few-shot para el LLM. Botón **🧠 Extraer con IA (RAG)** en *Procesar documento*.

## Flujo de uso

1. **Plantillas → + Nueva plantilla** → sube un documento de muestra.
2. Sobre la imagen, **pincha una palabra** o **arrastra un recuadro**; nombra el campo
   (ej. *Número documento*) y pulsa *Vincular campo*. Repite por cada dato.
3. Ponle nombre a la plantilla y **Crear plantilla**.
4. **Procesar documento** → sube uno nuevo del mismo tipo. Se detecta la plantilla y se
   rellenan los campos. Corrige si hace falta y **Acepta y crea el registro**.
5. **Registros** muestra todos los JSON extraídos.

## APIs principales

| Método   | Ruta                              | Descripción                                  |
| -------- | --------------------------------- | -------------------------------------------- |
| `POST`   | `/api/documents`                  | Sube documento → OCR + firma                 |
| `GET`    | `/api/documents/{id}`             | Datos del documento (OCR, firma)             |
| `GET`    | `/api/documents/{id}/image`       | Imagen normalizada (PNG)                     |
| `POST`   | `/api/documents/{id}/ocr-region`  | Re-OCR de una región concreta (recorte)      |
| `POST`   | `/api/documents/{id}/detect-border` | Auto-detecta el borde del documento        |
| `PUT`    | `/api/documents/{id}/border`      | Guarda el borde editado (recalcula firma)    |
| `GET`    | `/api/templates`                  | Lista plantillas                             |
| `POST`   | `/api/templates`                  | Crea plantilla (nombre, campos, firma)       |
| `PUT`    | `/api/templates/{id}`             | Actualiza plantilla                          |
| `DELETE` | `/api/templates/{id}`             | Elimina plantilla                            |
| `POST`   | `/api/process/{document_id}`      | Empareja + extrae campos (sin guardar)       |
| `POST`   | `/api/records`                    | Crea registro (+ aprendizaje RAG si `learn`) |
| `GET`    | `/api/records`                    | Lista registros (filtro `?template_id=`)     |
| `GET`    | `/api/ai/status`                  | Estado de Ollama y modelos instalados        |
| `POST`   | `/api/ai/extract/{document_id}`   | Extrae campos con LLM local + RAG            |
| `GET`    | `/api/ai/examples`                | Dataset de ejemplos confirmados (RAG)        |

### Ejemplo end-to-end con `curl`

```bash
# 1) Subir un documento
curl -F file=@dni.jpg http://localhost:8088/api/documents
# -> {"id": 1, "ocr_words": [...], "signature": {...}, ...}

# 2) Procesarlo contra las plantillas (auto-detección)
curl -X POST http://localhost:8088/api/process/1
# -> {"template_id": 3, "match_score": 0.82, "fields": {"numero_documento": {"value": "12345678Z", ...}}}

# 3) Guardar el registro
curl -X POST http://localhost:8088/api/records \
  -H "Content-Type: application/json" \
  -d '{"template_id":3,"document_id":1,"data":{"numero_documento":"12345678Z"},"match_score":0.82}'
```

## Cómo funciona el reconocimiento de la forma

- **Borde del documento** (`detect_border`): al subir, OpenCV separa el documento del
  fondo del escaneo (máscara por diferencia de color + mayor contorno). Las coordenadas
  de los campos se guardan **relativas a ese borde**, así son proporcionales al documento
  y no al escaneo. En el editor el borde es **editable** (tiradores en las esquinas o
  arrastrar para redibujar) y re-detectable. Al procesar, se detecta el borde del nuevo
  documento y los campos se mapean proporcionalmente (funciona con márgenes/escala
  distintos).
- **OCR** (`ocr.py`): Tesseract devuelve palabras con sus cajas; se normalizan a 0..1.
- **Firma** (`compute_signature`): proporción ancho/alto + color de fondo (mediana de
  bordes) + rejilla 8×8 de densidad de texto.
- **Matching** (`matching.py`): similitud ponderada (coseno de densidad + proporción +
  color). Se elige la plantilla con mayor puntuación; por debajo de `MATCH_THRESHOLD`
  se marca como confianza baja.
- **Extracción**: para cada campo se recogen las palabras cuyo centro cae en su región
  (con un pequeño margen), se ordenan en orden de lectura y se concatenan.

## Anclas (hitos) y rectificación a la plantilla

Además de la firma geométrica, una plantilla puede definir **anclas**: zonas fijas de
referencia (texto fijo y/o trozo de imagen) que permiten **enderezar la imagen del
documento al marco de la plantilla** (girar, corregir inclinación, escalar) y luego extraer
los campos por proporcionalidad simple. Al procesar:

- se localizan las anclas (multi-ángulo × multi-filtro),
- se **rectifica la imagen** con una similitud rígida ORB (sin cizalla; corrige giros de
  90/180/270 automáticamente),
- el documento queda alineado a la plantilla → borde = `tpl.border`, campos proporcionales,
- si faltan anclas **obligatorias**, el documento no se toca y queda en `review`.

El editor de revisión muestra una **traza** del pipeline (giro/enderezado/escala) y un modal
**⚓ Ver anclas** que compara cada ancla (plantilla vs detectada).

📄 **Documentación completa**: [docs/anclas-y-rectificacion.md](docs/anclas-y-rectificacion.md)

## Configuración (`.env`)

| Variable            | Por defecto | Descripción                          |
| ------------------- | ----------- | ------------------------------------ |
| `POSTGRES_USER/...` | `idp`       | Credenciales de PostgreSQL           |
| `OCR_LANGS`         | `spa+eng`   | Idiomas Tesseract (deben instalarse) |

Para añadir más idiomas, instala el paquete `tesseract-ocr-<lang>` en
`backend/Dockerfile` y añádelo a `OCR_LANGS`.

## Notas / siguientes pasos

- El esquema de BD se crea con `create_all` (MVP). Para producción, añadir **Alembic**.
  `create_all` **no altera** tablas existentes: tras añadir la columna `required` a las
  anclas, en una BD ya creada hay que ejecutar
  `ALTER TABLE template_anchors ADD COLUMN IF NOT EXISTS required BOOLEAN NOT NULL DEFAULT false;`
- El matching es geométrico; para documentos muy variables se puede mejorar con anclas
  de texto o, si más adelante quieres, un motor híbrido con un modelo de visión.
- Multipágina: actualmente se procesa la **primera página** de cada PDF.
