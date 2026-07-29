"""Traceable evidence collected by Agent tool calls."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import LocalDocument, SearchResult


@dataclass
class EvidenceExcerpt:
    evidence_id: str = ""
    document_id: str = ""
    title: str = ""
    url: str = ""
    file_path: str = ""
    line_start: int | None = None
    line_end: int | None = None
    content: str = ""
    evidence_type: str = "read"
    score: float = 0.0
    version_status: str = "unknown"


@dataclass
class SourceTracker:
    candidate_sources: list[SearchResult] = field(default_factory=list)
    evidence_excerpts: list[EvidenceExcerpt] = field(default_factory=list)
    selected_excerpts: list[EvidenceExcerpt] = field(default_factory=list)
    verified_urls: set[str] = field(default_factory=set)
    diagnostics: bool = False

    def add_candidates(self, results: list[SearchResult]) -> None:
        existing = {item.document.document_id for item in self.candidate_sources}
        for result in results:
            self.record_urls(result.chunk.content_snippet)
            if result.document.document_id not in existing:
                self.candidate_sources.append(result)
                existing.add(result.document.document_id)

    def add_read(
        self,
        document: LocalDocument,
        content: str,
        *,
        line_start: int | None = None,
        line_end: int | None = None,
        score: float = 1.0,
    ) -> EvidenceExcerpt:
        key = (
            document.document_id,
            line_start,
            line_end,
            _content_key(content),
        )
        for existing in self.evidence_excerpts:
            existing_key = (
                existing.document_id,
                existing.line_start,
                existing.line_end,
                _content_key(existing.content),
            )
            if existing_key == key:
                return existing

        excerpt = EvidenceExcerpt(
            evidence_id=f"E{len(self.evidence_excerpts) + 1}",
            document_id=document.document_id,
            title=document.title,
            url=document.source_url,
            file_path=document.relative_path,
            line_start=line_start,
            line_end=line_end,
            content=content[:4000],
            evidence_type="read",
            score=score,
            version_status=_version_status(document),
        )
        self.evidence_excerpts.append(excerpt)
        self.record_urls(content)
        return excerpt

    def record_urls(self, content: str) -> None:
        self.verified_urls.update(
            re.findall(r"https?://[^\s<>()，。；：\"']+", content)
        )


def _content_key(content: str) -> str:
    return re.sub(r"\s+", "", content).casefold()[:1000]


def _version_status(document: LocalDocument) -> str:
    text = f"{document.title} {document.relative_path}".casefold()
    if "过程版" in text or "草案" in text:
        return "draft"
    if "2026版" in text and "章程" in text:
        return "current"
    return "unknown"

