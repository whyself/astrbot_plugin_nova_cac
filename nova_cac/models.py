"""Shared local-document and retrieval models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LocalDocument:
    document_id: str
    title: str
    relative_path: str
    absolute_path: Path
    category: str
    source_url: str
    created_at: str
    updated_at: str
    body: str


@dataclass(frozen=True)
class ChunkResult:
    chunk_id: str
    document_id: str
    title: str
    content_snippet: str
    source_url: str
    vector_raw_score: float = 0.0
    vector_relevance: float = 0.0
    keyword_score: float = 0.0
    final_score: float = 0.0
    retrieval_methods: tuple[str, ...] = ()
    reliable: bool = False
    chunk_index: int = 0
    file_path: str = ""


@dataclass(frozen=True)
class SearchResult:
    source_id: str
    document: LocalDocument
    score: float
    chunk: ChunkResult
    vector_score: float = 0.0
    keyword_score: float = 0.0
    retrieval_methods: tuple[str, ...] = ()
    reliable: bool = False
