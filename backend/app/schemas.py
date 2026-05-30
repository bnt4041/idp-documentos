from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class Box(BaseModel):
    x: float
    y: float
    w: float
    h: float


class OCRWord(BaseModel):
    text: str
    conf: float
    box: Box


class Point(BaseModel):
    x: float
    y: float


class Quad(BaseModel):
    tl: Point
    tr: Point
    br: Point
    bl: Point


# ---- Documents ----
class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    width: int
    height: int
    ocr_words: list
    signature: dict
    border: dict
    created_at: datetime


# ---- Template fields ----
class FieldIn(BaseModel):
    name: str
    key: str
    data_type: str = "text"
    x: float
    y: float
    w: float
    h: float
    sample_text: str = ""


class FieldOut(FieldIn):
    model_config = ConfigDict(from_attributes=True)

    id: int


# ---- Templates ----
class TemplateCreate(BaseModel):
    name: str
    description: str = ""
    sample_document_id: int | None = None
    signature: dict = {}
    border: dict = {}
    fields: list[FieldIn] = []


class TemplateUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    signature: dict | None = None
    border: dict | None = None
    fields: list[FieldIn] | None = None


class TemplateOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str
    signature: dict
    border: dict
    sample_document_id: int | None
    created_at: datetime
    fields: list[FieldOut]
    example_count: int = 0


# ---- Processing / records ----
class MatchResult(BaseModel):
    document_id: int
    template_id: int | None
    template_name: str | None
    match_score: float
    fields: dict
    width: int
    height: int
    ocr_words: list
    border: dict


class RecordCreate(BaseModel):
    template_id: int | None = None
    document_id: int | None = None
    data: dict
    match_score: float = 0.0
    status: str = "confirmed"
    # Para el aprendizaje (RAG): regiones afinadas {key:{x,y,w,h}} en coords de imagen
    regions: dict | None = None
    learn: bool = True


class RecordOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    template_id: int | None
    document_id: int | None
    data: dict
    match_score: float
    status: str
    created_at: datetime
