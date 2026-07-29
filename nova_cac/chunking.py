"""Stable Markdown-aware chunk construction."""

from __future__ import annotations
import hashlib
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    content: str
    content_hash: str
    title: str
    repository: str
    namespace: str
    slug: str
    file_path: str
    source_url: str
    updated_at: str

    @property
    def embedding_text(self) -> str:
        """Text used for embedding: title + content, avoiding frontmatter tables."""
        parts = []
        if self.title and self.title.strip():
            parts.append(self.title.strip())
        if self.content and self.content.strip():
            parts.append(self.content.strip())
        return "\n".join(parts)

    def with_index(self, new_index: int) -> "Chunk":
        new_id = _stable_chunk_id(self.document_id, new_index, self.content)
        return Chunk(
            chunk_id=new_id,
            document_id=self.document_id,
            chunk_index=new_index,
            content=self.content,
            content_hash=self.content_hash,
            title=self.title,
            repository=self.repository,
            namespace=self.namespace,
            slug=self.slug,
            file_path=self.file_path,
            source_url=self.source_url,
            updated_at=self.updated_at,
        )


def _stable_chunk_id(document_id: str, chunk_index: int, content: str) -> str:
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    return f"{document_id}:{chunk_index}:{digest}"


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _strip_frontmatter_and_meta_table(body: str) -> str:
    """Remove YAML frontmatter and the leading metadata table often seen in Yuque docs."""
    # Frontmatter
    if body.startswith("---\n"):
        parts = body.split("---\n", 2)
        if len(parts) == 3:
            body = parts[2]
    # Leading markdown table: | col1 | col2 |
    #                         | ---- | ---- |
    #                         | val1 | val2 |
    body = re.sub(
        r"\A\s*\|[^\n]*\|\n\|(?:[-: ]+\|)+\n(?:\|[^\n]*\|\n?)*",
        "",
        body,
    ).lstrip()
    return body


def _clean_text(text: str) -> str:
    """Remove images and HTML tags; keep link text and URL."""
    # Drop markdown images entirely.
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    # Keep both link text and URL.
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    # Drop HTML tags (including Yuque font/color spans).
    text = re.sub(r"<[^>]+>", "", text)
    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_by_markdown_boundaries(text: str) -> list[str]:
    """Split on blank lines and markdown headings, preserving heading context."""
    # Keep headings attached to the following paragraph by replacing heading newline with a marker.
    # We split on \n{2,} but treat headings as their own boundary if they stand alone.
    lines = text.splitlines()
    blocks: list[str] = []
    current: list[str] = []

    def flush():
        if current:
            blocks.append("\n".join(current).strip())
            current.clear()

    for line in lines:
        stripped = line.strip()
        if re.fullmatch(r"#{1,6}\s+.*", stripped):
            flush()
            blocks.append(stripped)
        elif not stripped:
            flush()
        else:
            current.append(line)
    flush()
    # Merge lone heading with the next non-empty block unless it is already big enough.
    merged: list[str] = []
    i = 0
    while i < len(blocks):
        block = blocks[i]
        if re.fullmatch(r"#{1,6}\s+.*", block) and i + 1 < len(blocks):
            merged.append(f"{block}\n\n{blocks[i + 1]}")
            i += 2
        else:
            merged.append(block)
            i += 1
    return [b.strip() for b in merged if b.strip()]


def _boundary_cut(text: str, size: int) -> int:
    """Find a soft boundary at or before size."""
    if len(text) <= size:
        return len(text)
    # Prefer sentence end, then punctuation, then line break, then word boundary.
    for pattern in (r"[。！？\.\?\!]", r"[；;]", r"[\n]", r"[\s]"):
        for m in re.finditer(pattern, text[:size]):
            pos = m.end()
            # Ensure we make forward progress; avoid tiny cuts.
            if pos >= size * 0.3:
                return pos
    # Chinese char granularity fallback.
    return size


def split_markdown(
    document_id: str,
    body: str,
    title: str = "",
    repository: str = "",
    namespace: str = "",
    slug: str = "",
    file_path: str = "",
    source_url: str = "",
    updated_at: str = "",
    size: int = 1200,
    overlap: int = 180,
) -> list[Chunk]:
    """Split a Markdown document into stable, metadata-rich chunks.

    The algorithm prefers Markdown headings, paragraphs and list boundaries.  Only
    individual blocks longer than ``size`` are split with a sliding window whose
    step is ``size - overlap``.  Empty bodies produce no chunks.
    """
    if size < 200 or not 0 <= overlap < size // 2:
        raise ValueError("invalid chunk settings")

    body = _strip_frontmatter_and_meta_table(body)
    if not body.strip():
        return []

    blocks = _split_by_markdown_boundaries(body)
    parts: list[str] = []
    current = ""

    def flush_current():
        nonlocal current
        if current.strip():
            parts.append(current.strip())
            current = ""

    for block in blocks:
        if len(block) > size:
            # Long block gets its own sliding split.
            flush_current()
            step = max(1, size - overlap)
            start = 0
            while start < len(block):
                end = _boundary_cut(block[start:], size) + start
                piece = block[start:end].strip()
                if piece:
                    parts.append(piece)
                # Advance by step; ensure progress even if overlap is large.
                next_start = min(len(block), start + step)
                if next_start <= start:
                    next_start = start + step
                start = next_start
            continue

        if current:
            if len(current) + len(block) + 2 <= size:
                current = f"{current}\n\n{block}"
                continue
            flush_current()
        current = block

    flush_current()

    result: list[Chunk] = []
    index = 0
    for part in parts:
        cleaned = _clean_text(part)
        if len(cleaned.strip()) < 3:
            continue
        chunk = Chunk(
            chunk_id=_stable_chunk_id(document_id, index, cleaned),
            document_id=document_id,
            chunk_index=index,
            content=cleaned,
            content_hash=_content_hash(cleaned),
            title=title,
            repository=repository,
            namespace=namespace,
            slug=slug,
            file_path=file_path,
            source_url=source_url,
            updated_at=updated_at,
        )
        result.append(chunk)
        index += 1
    return result


def renumber_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Recompute indices and IDs after deletion/merge."""
    return [c.with_index(i) for i, c in enumerate(chunks)]
