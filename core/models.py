from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class Document(BaseModel):
    id: Optional[int] = None
    content: str
    source: Optional[str] = None


class DocumentResult(BaseModel):
    id: int
    content: str
    source: Optional[str]
    score: float


class Product(BaseModel):
    id: Optional[int] = None
    name: str
    description: str
    category: Optional[str] = None
    price: Optional[float] = None


class ProductResult(BaseModel):
    id: int
    name: str
    description: str
    category: Optional[str]
    price: Optional[float]
    score: float


class Log(BaseModel):
    id: Optional[int] = None
    message: str
    level: Optional[str] = "INFO"
    service: Optional[str] = None
    timestamp: Optional[datetime] = None


class LogResult(BaseModel):
    id: int
    message: str
    level: Optional[str]
    service: Optional[str]
    score: float
