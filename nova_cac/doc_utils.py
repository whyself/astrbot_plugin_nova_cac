"""Shared, safe document contract used by every knowledge-base tool."""

from __future__ import annotations
import re
from pathlib import Path
from urllib.parse import unquote, urlparse
import yaml


def parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---\n"):
        return {}, text
    parts = text.split("---\n", 2)
    return (yaml.safe_load(parts[1]) or {}, parts[2] if len(parts) == 3 else "")


def strip_meta_table(body: str) -> str:
    return re.sub(
        r"\A\s*\|[^\n]*\|\n\|(?:[-: ]+\|)+\n(?:\|[^\n]*\|\n)?", "", body
    ).lstrip()


def clean_document_body(body: str, *, preserve_paragraphs: bool = False) -> str:
    """Return a display-friendly version of a Markdown document body.

    Removes YAML frontmatter, leading metadata tables, markdown images and HTML
    tags, while preserving link text and URLs.
    """
    body = strip_meta_table(parse_frontmatter(body)[1])
    # Drop markdown images entirely.
    body = re.sub(r"!\[.*?\]\(.*?\)", "", body)
    # Keep both link text and URL for Markdown links.
    body = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", body)
    # Preserve HTML anchor links as "text (url)" before stripping other tags.
    body = re.sub(
        r"<a\s+[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>",
        r"\2 (\1)",
        body,
        flags=re.IGNORECASE | re.DOTALL,
    )
    # Drop remaining HTML tags (including Yuque font/color spans).
    body = re.sub(r"<[^>]+>", "", body)
    if preserve_paragraphs:
        # Collapse horizontal whitespace inside each paragraph but keep paragraph
        # breaks, so headings/lists remain readable for grounding.
        paragraphs = [
            re.sub(r"\s+", " ", paragraph).strip()
            for paragraph in body.split("\n\n")
            if paragraph.strip()
        ]
        body = "\n\n".join(paragraphs)
    else:
        # Normalize whitespace for short snippets / grep results.
        body = re.sub(r"\s+", " ", body).strip()
    return body


def read_document_content(
    root: Path,
    file_path: str,
    offset: int = 0,
    limit: int = 12000,
    strip_metadata: bool = True,
) -> dict:
    if offset < 0 or not 1 <= limit <= 20000:
        raise ValueError("invalid pagination")
    root_resolved = root.resolve()
    file_path_obj = Path(file_path)
    if ".." in file_path_obj.parts:
        raise ValueError("invalid document path")

    # file_path may be relative to root, or relative to the project cwd with the
    # root as a prefix. Try root-relative first; fall back to cwd-relative.
    candidate = (root_resolved / file_path_obj).resolve()
    if (
        not candidate.is_file()
        or not candidate.is_relative_to(root_resolved)
    ):
        candidate = file_path_obj.resolve()

    if (
        candidate.is_symlink()
        or not candidate.is_file()
        or not candidate.is_relative_to(root_resolved)
    ):
        raise ValueError("invalid document path")
    metadata, body = parse_frontmatter(candidate.read_text(encoding="utf-8"))
    if strip_metadata:
        body = clean_document_body(body, preserve_paragraphs=True)
    chunk = body[offset : offset + limit]
    return {
        "metadata": metadata,
        "content": chunk,
        "total_chars": len(body),
        "has_more": offset + len(chunk) < len(body),
        "next_offset": offset + len(chunk),
        "offset": offset,
    }


def read_document_lines(
    root: Path,
    file_path: str,
    start_line: int = 0,
    end_line: int | None = None,
    context_lines: int = 0,
    strip_metadata: bool = True,
) -> dict:
    """Read a local Markdown document by 1-based line numbers.

    ``start_line`` and ``end_line`` use the same 1-based numbering shown by the
    line-level grep tool.  Lines are counted after optional metadata stripping.
    """
    if start_line < 0:
        raise ValueError("invalid line range")
    lines = _load_cleaned_document_lines(root, file_path, strip_metadata=strip_metadata)
    total_lines = len(lines)
    start = max(0, start_line - 1)
    end = total_lines if end_line is None else min(end_line, total_lines)
    if context_lines:
        start = max(0, start - context_lines)
        end = min(total_lines, end + context_lines)
    if start >= end:
        return {
            "metadata": _load_document_metadata(root, file_path),
            "content": "",
            "total_lines": total_lines,
            "start_line": start + 1,
            "end_line": end,
            "has_more": False,
            "file_path": file_path,
        }
    content = "\n".join(f"{i + 1}: {lines[i]}" for i in range(start, end))
    return {
        "metadata": _load_document_metadata(root, file_path),
        "content": content,
        "total_lines": total_lines,
        "start_line": start + 1,
        "end_line": end,
        "has_more": end < total_lines,
        "file_path": file_path,
    }


def _resolve_document_path(root: Path, file_path: str) -> Path:
    """Resolve a stored path against the document root, enforcing containment."""
    root_resolved = root.resolve()
    file_path_obj = Path(file_path)
    if ".." in file_path_obj.parts:
        raise ValueError("invalid document path")
    candidate = (root_resolved / file_path_obj).resolve()
    if not candidate.is_file() or not candidate.is_relative_to(root_resolved):
        candidate = file_path_obj.resolve()
    if (
        candidate.is_symlink()
        or not candidate.is_file()
        or not candidate.is_relative_to(root_resolved)
    ):
        raise ValueError("invalid document path")
    return candidate


def _load_document_metadata(root: Path, file_path: str) -> dict:
    candidate = _resolve_document_path(root, file_path)
    metadata, _ = parse_frontmatter(candidate.read_text(encoding="utf-8"))
    return metadata


def _load_cleaned_document_lines(
    root: Path, file_path: str, *, strip_metadata: bool = True
) -> list[str]:
    """Return the cleaned text of a local document as a list of lines."""
    candidate = _resolve_document_path(root, file_path)
    _, body = parse_frontmatter(candidate.read_text(encoding="utf-8"))
    if strip_metadata:
        body = clean_document_body(body, preserve_paragraphs=True)
    return body.splitlines()


def parse_yuque_doc_url(url: str) -> tuple[str, str] | None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return None
    parts = [unquote(x) for x in parsed.path.split("/") if x]
    return ("/".join(parts[:-1]), parts[-1]) if len(parts) >= 2 else None


def doc_record_to_public_dict(row) -> dict:
    return {
        key: row[key]
        for key in (
            "yuque_id",
            "title",
            "repository",
            "namespace",
            "slug",
            "url",
            "created_at",
            "updated_at",
            "path",
        )
    }
