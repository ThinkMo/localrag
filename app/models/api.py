from typing import TypeVar
from datetime import datetime
from pydantic import BaseModel

from app.db import DocumentType


T = TypeVar("T")


class PaginatedResponse[T](BaseModel):
    items: list[T]
    total: int


class DocumentRead(BaseModel):
    id: int
    title: str
    document_type: DocumentType
    document_metadata: dict
    content: str  # Changed to string to match frontend
    created_at: datetime