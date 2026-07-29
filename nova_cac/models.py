from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Repository:
    namespace: str
    name: str = ""


@dataclass(frozen=True)
class Document:
    yuque_id: str
    title: str
    repository: str
    namespace: str
    slug: str
    url: str
    created_at: str
    updated_at: str
    body: str
    path: Path | None = None


@dataclass(frozen=True)
class ChunkResult:
    chunk_id: str
    document_id: str
    title: str
    content_snippet: str
    source_url: str
    vector_raw_score: float = 0.0
    vector_score_type: str = "cosine_distance"
    vector_relevance: float = 0.0
    keyword_score: float = 0.0
    final_score: float = 0.0
    retrieval_methods: tuple[str, ...] = ()
    reliable: bool = False
    chunk_index: int = 0
    file_path: str = ""
    slug: str = ""
    namespace: str = ""
    repository: str = ""


@dataclass(frozen=True)
class SearchResult:
    source_id: str
    document: Document
    score: float
    chunk: ChunkResult | None = None
    vector_score: float = 0.0
    keyword_score: float = 0.0
    retrieval_methods: tuple[str, ...] = ()
    reliable: bool = False


@dataclass
class SyncResult:
    succeeded: int = 0
    failed: int = 0
    deleted: int = 0
    skipped: int = 0
    chunks_indexed: int = 0
    chunks_failed: int = 0

    def summary(self) -> str:
        return (
            f"同步完成：成功 {self.succeeded}，失败 {self.failed}，删除 {self.deleted}，跳过 {self.skipped}；"
            f"chunk 索引 {self.chunks_indexed}，失败 {self.chunks_failed}。"
        )
