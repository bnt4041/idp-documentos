from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class Document(Base):
    """Un fichero subido (página) con su OCR cacheado."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filename: Mapped[str] = mapped_column(String(512))
    stored_path: Mapped[str] = mapped_column(String(1024))
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    # Lista de palabras OCR: [{text, conf, box:{x,y,w,h}}] en coords normalizadas 0..1
    ocr_words: Mapped[list] = mapped_column(JSON, default=list)
    # Firma geométrica del documento para el matching
    signature: Mapped[dict] = mapped_column(JSON, default=dict)
    # Borde detectado del documento {x,y,w,h} normalizado al escaneo completo
    border: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Template(Base):
    """Plantilla entrenada: define la forma y los campos a extraer."""

    __tablename__ = "templates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True)
    description: Mapped[str] = mapped_column(Text, default="")
    # Firma de referencia (aspect ratio, color de fondo, rejilla de densidad de texto)
    signature: Mapped[dict] = mapped_column(JSON, default=dict)
    # Borde de referencia del entrenamiento {x,y,w,h}; los campos son relativos a él
    border: Mapped[dict] = mapped_column(JSON, default=dict)
    # Documento de muestra usado para entrenar (para visualizar en el editor)
    sample_document_id: Mapped[int | None] = mapped_column(
        ForeignKey("documents.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    fields: Mapped[list["TemplateField"]] = relationship(
        back_populates="template", cascade="all, delete-orphan"
    )
    anchors: Mapped[list["TemplateAnchor"]] = relationship(
        back_populates="template", cascade="all, delete-orphan"
    )


class TemplateField(Base):
    """Un campo dentro de una plantilla, anclado a una región geométrica."""

    __tablename__ = "template_fields"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    template_id: Mapped[int] = mapped_column(
        ForeignKey("templates.id", ondelete="CASCADE")
    )
    name: Mapped[str] = mapped_column(String(255))
    key: Mapped[str] = mapped_column(String(255))
    data_type: Mapped[str] = mapped_column(String(50), default="text")
    # Región normalizada 0..1
    x: Mapped[float] = mapped_column(Float)
    y: Mapped[float] = mapped_column(Float)
    w: Mapped[float] = mapped_column(Float)
    h: Mapped[float] = mapped_column(Float)
    # Texto de muestra capturado al entrenar (referencia / anclaje)
    sample_text: Mapped[str] = mapped_column(Text, default="")

    template: Mapped[Template] = relationship(back_populates="fields")


class TemplateAnchor(Base):
    """Hito/ancla de una plantilla: zona de referencia con texto fijo y/o trozo de
    imagen que sirve para elegir la plantilla, orientar/enderezar el documento y
    alinear las regiones de los campos."""

    __tablename__ = "template_anchors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    template_id: Mapped[int] = mapped_column(
        ForeignKey("templates.id", ondelete="CASCADE")
    )
    name: Mapped[str] = mapped_column(String(255), default="")
    # Región normalizada 0..1, relativa al borde (igual que TemplateField)
    x: Mapped[float] = mapped_column(Float)
    y: Mapped[float] = mapped_column(Float)
    w: Mapped[float] = mapped_column(Float)
    h: Mapped[float] = mapped_column(Float)
    # Texto fijo esperado en la zona (referencia para el matching por OCR)
    anchor_text: Mapped[str] = mapped_column(Text, default="")
    # Señales activas del ancla
    use_text: Mapped[bool] = mapped_column(Boolean, default=True)
    use_image: Mapped[bool] = mapped_column(Boolean, default=True)
    # Importancia relativa en el score de anclas
    weight: Mapped[float] = mapped_column(Float, default=1.0)

    template: Mapped[Template] = relationship(back_populates="anchors")


class LearningExample(Base):
    """Documento confirmado: base de conocimiento del RAG (dataset + embedding)."""

    __tablename__ = "learning_examples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    template_id: Mapped[int | None] = mapped_column(
        ForeignKey("templates.id", ondelete="CASCADE"), nullable=True
    )
    document_id: Mapped[int | None] = mapped_column(
        ForeignKey("documents.id", ondelete="SET NULL"), nullable=True
    )
    # {key: {value, region:{x,y,w,h} relativa al borde}}
    fields: Mapped[dict] = mapped_column(JSON, default=dict)
    ocr_text: Mapped[str] = mapped_column(Text, default="")
    # Embedding del texto OCR para el retrieval por similitud
    embedding: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Record(Base):
    """Resultado de procesar un documento con una plantilla."""

    __tablename__ = "records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    template_id: Mapped[int | None] = mapped_column(
        ForeignKey("templates.id", ondelete="SET NULL"), nullable=True
    )
    document_id: Mapped[int | None] = mapped_column(
        ForeignKey("documents.id", ondelete="SET NULL"), nullable=True
    )
    # JSON final {clave: valor}
    data: Mapped[dict] = mapped_column(JSON, default=dict)
    match_score: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(50), default="draft")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
