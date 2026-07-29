"""Lightweight knowledge-base structure model derived from document paths.

The repository stores documents in a directory tree:

    <namespace>/<category>/<sub-category>/<document>.md

This module builds a stable, deterministic tree from :class:`DocumentIndex`
rows without requiring a separate configuration file.  It powers structure
navigation tools such as ``list_knowledge_bases``, ``list_repo_tree`` and
scoped document listing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class KnowledgeBaseSummary:
    """High-level summary of one synced knowledge base (namespace)."""

    namespace: str
    repository: str
    document_count: int
    top_level_categories: tuple["KnowledgeTreeCategorySummary", ...]


@dataclass(frozen=True)
class KnowledgeTreeCategorySummary:
    """One top-level category under a namespace."""

    name: str
    document_count: int


@dataclass(frozen=True)
class KnowledgeTreeNode:
    """A node in the repository directory tree.

    * ``name``           -- directory/file segment name.
    * ``path_prefix``    -- full path prefix from the repository root; the root
                            node itself has an empty prefix.
    * ``depth``          -- depth from the namespace root (root = 0).
    * ``document_count`` -- number of documents in this subtree, including
                            documents stored directly at this level.
    * ``children``       -- child nodes (files are leaves with no children).
    * ``is_index``       -- whether this node represents an index document.
    """

    name: str
    path_prefix: str
    depth: int
    document_count: int
    children: tuple["KnowledgeTreeNode", ...] = ()
    is_index: bool = False


def _row_value(row: Any, key: str, default: Any = "") -> Any:
    """Return a value from a dict-like or sqlite3.Row object."""
    if hasattr(row, "get"):
        return row.get(key, default)
    try:
        return row[key]
    except (KeyError, IndexError):
        return default


def normalize_namespace(value: str) -> str:
    """Return a canonical form of a namespace where ``_`` and ``/`` are equivalent.

    Yuque namespaces look like ``qc19gt/fqpid3`` but are often written as
    ``qc19gt_fqpid3`` in configuration or filenames.  This helper unifies the
    two so that structure tools resolve both to the same knowledge base.
    """
    return str(value).replace("_", "/").strip("/")


def _namespace_matches(
    segments: list[str], namespace_segments: list[str]
) -> bool:
    """Return True when ``segments`` start with the normalized namespace."""
    if not namespace_segments:
        return True
    if len(segments) < len(namespace_segments):
        return False
    return segments[: len(namespace_segments)] == namespace_segments


def _normalize_path(path: str | Path | None) -> str:
    """Return a POSIX-style string path with leading/trailing slashes removed."""
    if path is None:
        return ""
    return str(path).replace("\\", "/").strip("/")


def _path_segments(path: str) -> list[str]:
    """Split a normalized path into non-empty segments."""
    return [segment for segment in path.split("/") if segment]


def _is_index_name(name: str) -> bool:
    """Heuristic for directory index / table-of-contents documents."""
    lower = name.lower()
    if lower.startswith("00_"):
        return True
    stem = Path(name).stem.lower()
    if stem in {"index", "目录", "toc"}:
        return True
    if "index" in stem.split("_"):
        return True
    return False


def _matches_prefix(segments: list[str], prefix_segments: list[str]) -> bool:
    """Return True when ``segments`` start with ``prefix_segments``."""
    if not prefix_segments:
        return True
    if len(segments) < len(prefix_segments):
        return False
    return segments[: len(prefix_segments)] == prefix_segments


def _is_archived_path(segments: list[str]) -> bool:
    """Return True when any path segment is the archive directory."""
    return any(segment.lower() == "归档" for segment in segments)


def _sum_trie_counts(trie: dict) -> int:
    """Return the total document count of a trie (node + all descendants)."""
    total = trie.get("count", 0)
    for child in trie.get("children", {}).values():
        total += _sum_trie_counts(child)
    return total


def _build_trie(
    rows: list[Any],
    *,
    namespace: str = "",
    path_prefix: str = "",
    max_depth: int = 0,
    include_archived: bool = True,
) -> dict:
    """Build a mutable trie of path segments with per-node document counts."""
    prefix_segments = _path_segments(path_prefix)
    root: dict = {"count": 0, "children": {}, "is_index": False}

    namespace_segments = _path_segments(normalize_namespace(namespace)) if namespace else []

    for row in rows:
        rel_path = _normalize_path(_row_value(row, "path"))
        segments = _path_segments(rel_path)
        if not segments:
            continue

        # Namespace filter.  The first segments are treated as the namespace.
        if namespace and not _namespace_matches(segments, namespace_segments):
            continue

        if not include_archived and _is_archived_path(segments):
            continue

        # The path_prefix is interpreted relative to the namespace when a
        # namespace is provided; otherwise it is absolute from the repository
        # root.
        anchor = len(namespace_segments) if namespace else 0
        if not _matches_prefix(segments[anchor:], prefix_segments):
            continue

        # Trim the namespace and/or prefix so the trie root represents the
        # requested scope, not an extra level above it.
        start = anchor + len(prefix_segments)
        working_segments = segments[start:]
        if not working_segments:
            root["count"] += 1
            continue

        node = root
        for i, segment in enumerate(working_segments):
            if max_depth > 0 and i >= max_depth:
                # Any deeper segments are collapsed into the current node.
                node["count"] += 1
                break
            is_last = i == len(working_segments) - 1
            if segment not in node["children"]:
                node["children"][segment] = {
                    "count": 0,
                    "children": {},
                    "is_index": False,
                }
            child = node["children"][segment]
            if is_last:
                child["count"] += 1
                if _is_index_name(segment):
                    child["is_index"] = True
            node = child

    # The root node should report the total documents in its scope.
    root["count"] = root["count"] + _sum_trie_counts(
        {"count": 0, "children": root["children"]}
    )
    return root


def _trie_to_node(
    trie: dict,
    *,
    name: str = "",
    path_prefix: str = "",
    depth: int = 0,
) -> KnowledgeTreeNode:
    """Convert a mutable trie into an immutable ``KnowledgeTreeNode``."""
    children = sorted(
        (
            _trie_to_node(
                child_trie,
                name=child_name,
                path_prefix="/".join(
                    [path_prefix, child_name] if path_prefix else [child_name]
                ),
                depth=depth + 1,
            )
            for child_name, child_trie in trie.get("children", {}).items()
        ),
        key=lambda node: (not node.is_index, node.name),
    )
    return KnowledgeTreeNode(
        name=name,
        path_prefix=path_prefix,
        depth=depth,
        document_count=trie.get("count", 0),
        children=tuple(children),
        is_index=trie.get("is_index", False),
    )


def build_knowledge_tree(
    rows: list[Any],
    *,
    namespace: str = "",
    path_prefix: str = "",
    max_depth: int = 0,
    include_archived: bool = True,
) -> KnowledgeTreeNode | None:
    """Build a ``KnowledgeTreeNode`` for the requested scope.

    Parameters
    ----------
    rows:
        SQLite rows or dictionaries with a ``path`` key.
    namespace:
        Optional namespace filter (the first path segment).
    path_prefix:
        Optional path prefix filter (e.g. ``"02_教务与学业"``).
    max_depth:
        Maximum tree depth relative to ``path_prefix``.  ``0`` means unlimited.
    include_archived:
        Whether to include paths containing an ``归档`` segment.

    Returns
    -------
    The root node, or ``None`` when no documents match.
    """
    trie = _build_trie(
        rows,
        namespace=namespace,
        path_prefix=path_prefix,
        max_depth=max_depth,
        include_archived=include_archived,
    )
    if trie["count"] == 0 and not trie["children"]:
        return None
    return _trie_to_node(trie, name=namespace or "root", path_prefix=path_prefix)


def _is_top_level_index_category(name: str) -> bool:
    """Return True when a category segment is a root index/readme document."""
    lower = name.lower()
    stem = Path(name).stem.lower()
    if stem in {"index", "目录", "toc", "readme"}:
        return True
    return lower.startswith("00_")


def build_knowledge_base_summaries(
    rows: list[Any],
    *,
    include_archived: bool = True,
) -> list[KnowledgeBaseSummary]:
    """Return one summary per namespace, with top-level categories and counts."""
    by_namespace: dict[str, dict[str, Any]] = {}

    for row in rows:
        rel_path = _normalize_path(_row_value(row, "path"))
        segments = _path_segments(rel_path)
        if not segments:
            continue
        namespace = segments[0]
        repository = _row_value(row, "repository")
        if namespace not in by_namespace:
            by_namespace[namespace] = {
                "repository": repository,
                "categories": {},
                "document_count": 0,
            }
        info = by_namespace[namespace]
        if repository and not info["repository"]:
            info["repository"] = repository
        if not include_archived and _is_archived_path(segments):
            continue
        info["document_count"] += 1
        if len(segments) > 1:
            category = segments[1]
            if _is_top_level_index_category(category):
                continue
            info["categories"][category] = info["categories"].get(category, 0) + 1

    summaries: list[KnowledgeBaseSummary] = []
    for namespace in sorted(by_namespace):
        info = by_namespace[namespace]
        categories = tuple(
            KnowledgeTreeCategorySummary(name=name, document_count=count)
            for name, count in sorted(info["categories"].items())
        )
        summaries.append(
            KnowledgeBaseSummary(
                namespace=namespace,
                repository=info["repository"],
                document_count=info["document_count"],
                top_level_categories=categories,
            )
        )
    return summaries


def list_documents_under_prefix(
    rows: list[Any],
    *,
    namespace: str = "",
    path_prefix: str = "",
    title_query: str = "",
    include_archived: bool = True,
    include_index: bool = True,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[dict], bool]:
    """Return document records under a path prefix, with pagination metadata.

    Returns
    -------
    A tuple of (documents, has_more).
    """
    prefix_segments = _path_segments(path_prefix)
    namespace_segments = _path_segments(normalize_namespace(namespace)) if namespace else []
    matching: list[dict] = []

    for row in rows:
        rel_path = _normalize_path(_row_value(row, "path"))
        segments = _path_segments(rel_path)
        if not segments:
            continue
        if namespace and not _namespace_matches(segments, namespace_segments):
            continue
        anchor = len(namespace_segments) if namespace else 0
        if not _matches_prefix(segments[anchor:], prefix_segments):
            continue
        if not include_archived and _is_archived_path(segments):
            continue
        public = {
            "yuque_id": _row_value(row, "yuque_id"),
            "title": _row_value(row, "title"),
            "repository": _row_value(row, "repository"),
            "namespace": _row_value(row, "namespace"),
            "slug": _row_value(row, "slug"),
            "url": _row_value(row, "url"),
            "created_at": _row_value(row, "created_at"),
            "updated_at": _row_value(row, "updated_at"),
            "path": rel_path,
            "is_index": _is_index_name(rel_path),
        }
        if not include_index and public["is_index"]:
            continue
        if title_query and title_query.casefold() not in public["title"].casefold():
            continue
        matching.append(public)

    matching.sort(key=lambda d: (d["path"], d["title"]))
    start = max(offset, 0)
    end = start + max(limit, 1)
    return matching[start:end], len(matching) > end


def tree_to_text(
    node: KnowledgeTreeNode,
    *,
    prefix: str = "",
    is_last: bool = True,
    show_counts: bool = True,
) -> str:
    """Render a knowledge tree as a compact text diagram."""
    connector = "└── " if is_last else "├── "
    line = prefix + connector + node.name
    if show_counts:
        line += f" ({node.document_count})"
    lines = [line]
    child_prefix = prefix + ("    " if is_last else "│   ")
    children = list(node.children)
    for i, child in enumerate(children):
        lines.extend(
            tree_to_text(
                child,
                prefix=child_prefix,
                is_last=(i == len(children) - 1),
                show_counts=show_counts,
            ).splitlines()
        )
    return "\n".join(lines)
