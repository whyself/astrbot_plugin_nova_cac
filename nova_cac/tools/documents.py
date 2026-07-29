"""Document navigation tools sharing the SQLite/Markdown data contract."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from astrbot.api import FunctionTool, logger
from astrbot.api.event import AstrMessageEvent

from ..doc_utils import (
    _load_cleaned_document_lines,
    doc_record_to_public_dict,
    parse_yuque_doc_url,
    read_document_content,
    read_document_lines,
)
from ..evidence import (
    _ACTION_SYNONYMS,
    _canonical_action,
    document_from_index_row,
    evaluate_grep_reliability,
    evidence_excerpt_from_text,
    filter_by_required_phrases,
    grep_hits_to_search_results,
    recommended_read_range,
    score_grep_hit,
)
from ..knowledge_structure import (
    _is_archived_path,
    _matches_prefix,
    _namespace_matches,
    _normalize_path,
    _path_segments,
    _row_value,
    build_knowledge_base_summaries,
    build_knowledge_tree,
    list_documents_under_prefix,
    normalize_namespace,
    tree_to_text,
)


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


def _expand_query_term(term: str) -> list[str]:
    """Return the term plus known action synonyms.

    If ``term`` is an action (or one of its synonyms), the whole synonym set is
    included so that "补办" also matches "补卡" / "重办".
    """
    canonical = _canonical_action(term)
    if canonical is None:
        return [term]
    expanded = list(_ACTION_SYNONYMS[canonical])
    if term not in expanded:
        expanded.append(term)
    return expanded


# Read range preferences.  These limits are intentionally conservative so that
# the model receives focused context and does not waste tokens on huge ranges.
_MIN_READ_LINES = 10
_PREFERRED_READ_LINES = 40
_MAX_READ_LINES = 80
_MAX_READ_CHARS = 2400


def _clamp_read_range(
    start_line: int | None,
    end_line: int | None,
    total_lines: int,
) -> tuple[int, int, list[str]]:
    """Clamp a requested 1-based inclusive line range to the preferred window.

    Returns ``(start, end, warnings)``.  Ranges above ``_MAX_READ_LINES`` lines
    are narrowed; ranges above the preferred size receive a warning but are kept.
    """
    start = max(1, start_line or 1)
    end = total_lines if end_line is None else min(end_line, total_lines)
    warnings: list[str] = []
    count = end - start + 1
    if count > _MAX_READ_LINES:
        narrowed_end = start + _PREFERRED_READ_LINES - 1
        warnings.append(
            f"请求读取 {start}-{end}（{count} 行）超过 {_MAX_READ_LINES} 行，"
            f"已收窄到 {start}-{narrowed_end} 行。"
        )
        end = narrowed_end
    elif count > _PREFERRED_READ_LINES:
        warnings.append(
            f"请求读取 {start}-{end}（{count} 行）较大，建议控制在 "
            f"{_MIN_READ_LINES}-{_PREFERRED_READ_LINES} 行以内。"
        )
    return start, end, warnings


def _recompute_line_end(content: str, line_end: int | None) -> int | None:
    """Return the real last line number after content truncation.

    ``read_document_lines`` prefixes every line with ``N: ``.  We parse that
    prefix from the final line so that ``line_end`` matches the actual content.
    """
    if not content:
        return line_end
    last_line = content.rsplit("\n", 1)[-1]
    match = re.match(r"(\d+):", last_line)
    if match:
        return int(match.group(1))
    return line_end


def _truncate_content_at_boundary(content: str, max_chars: int) -> str:
    """Truncate content at the last complete line, sentence, or safe boundary.

    Prefers a full newline so no half line / list item / link is written into
    evidence.  If no newline fits, fall back to sentence-ending punctuation, then
    to the last whitespace, and only then to ``max_chars``.
    """
    if len(content) <= max_chars:
        return content

    # 1) Full line boundary: keep content up to the last newline that fits.
    # If ``max_chars`` already lands exactly at a line end, keep that much.
    if content[max_chars] == "\n" or max_chars == len(content):
        return content[:max_chars]
    last_newline = content.rfind("\n", 0, max_chars)
    if last_newline > 0:
        return content[:last_newline]

    # 2) Sentence-ending punctuation.
    for boundary in ("\n", "。", "！", "？", ".", "!", "?"):
        idx = content.rfind(boundary, 0, max_chars)
        if idx > 0:
            return content[: idx + 1]

    # 3) Last whitespace so we do not split a word/link mid-token.
    last_space = content.rfind(" ", 0, max_chars)
    if last_space > 0:
        return content[:last_space]

    return content[:max_chars]


def _truncate_read_result(result: dict, max_chars: int = _MAX_READ_CHARS) -> dict:
    """Truncate content to ``max_chars`` and update ``end_line`` accordingly."""
    content = result.get("content", "")
    if len(content) <= max_chars:
        return result
    result["content"] = _truncate_content_at_boundary(content, max_chars)
    result["end_line"] = _recompute_line_end(result["content"], result.get("end_line"))
    result["truncated"] = True
    return result


def _row_path_segments(row: Any) -> list[str]:
    """Return path segments for a SQLite document row."""
    return _path_segments(_normalize_path(_row_value(row, "path")))


def _row_matches_scope(
    row: Any,
    *,
    namespace: str = "",
    path_prefix: str = "",
    repository: str = "",
    document_ids: set[str] | None = None,
    include_archived: bool = True,
) -> bool:
    """Return True when a document row matches the requested scope filters."""
    segments = _row_path_segments(row)
    if not segments:
        return False

    if repository and repository.casefold() not in (_row_value(row, "repository")).casefold():
        return False

    namespace_segments = _path_segments(normalize_namespace(namespace)) if namespace else []
    if namespace and not _matches_prefix(segments, namespace_segments):
        return False

    prefix_segments = _path_segments(path_prefix)
    anchor = len(namespace_segments) if namespace else 0
    if not _matches_prefix(segments[anchor:], prefix_segments):
        return False

    if document_ids is not None and row.get("yuque_id") not in document_ids:
        return False

    if not include_archived and _is_archived_path(segments):
        return False

    return True


def _parse_document_ids(value: str | list[str] | None) -> set[str] | None:
    """Convert a string/list parameter into a set of yuque_ids."""
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return set(str(v).strip() for v in value if str(v).strip())
    ids = [v.strip() for v in re.split(r"[,\s]+", str(value)) if v.strip()]
    return set(ids) if ids else None


@dataclass
class _Tool(FunctionTool):
    index: Any = None
    docs_root: Any = None
    tracker: Any = None

    async def call(self, context, **kwargs):
        return await self._run(**kwargs)

    async def run(self, event: AstrMessageEvent, **kwargs):
        return await self._run(**kwargs)

    def _record_navigation_evidence(
        self, *, title: str, file_path: str = "", content: str, url: str = ""
    ) -> None:
        """Record a structure/navigation tool result as candidate evidence."""
        if not self.tracker or not content:
            return
        self.tracker.add_evidence(
            evidence_excerpt_from_text(
                content=content[:2400],
                title=title,
                file_path=file_path,
                url=url,
                evidence_type="navigation",
            )
        )


@dataclass
class GrepLocalDocsTool(_Tool):
    name: str = "grep_local_docs"
    description: str = (
        "按空格分隔的关键词在本地的 Markdown 文档中逐行搜索，返回带行号的匹配片段。"
        "对于具体事实、名称、流程、时间等精确查询，优先使用本工具而不是向量检索。"
        "支持按知识库命名空间、路径前缀、文档 ID 集合和归档开关进行作用域过滤。"
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "string",
                    "description": "空格分隔的 1-4 个核心关键词，尽量用文档中可能出现的词",
                },
                "repo_filter": {
                    "type": "string",
                    "description": "按 repository 名称过滤（已废弃，建议使用 repository）",
                },
                "repository": {"type": "string", "description": "按 repository 名称过滤"},
                "namespace": {
                    "type": "string",
                    "description": "按知识库命名空间过滤（path 的第一段）",
                },
                "path_prefix": {
                    "type": "string",
                    "description": "按路径前缀过滤（相对 namespace，例如 02_教务与学业/课程与选课）",
                },
                "document_ids": {
                    "type": "string",
                    "description": "空格或逗号分隔的文档 ID 集合，限制只在这些文档中搜索",
                },
                "include_archived": {
                    "type": "boolean",
                    "default": True,
                    "description": "是否包含路径中出现“归档”的文档",
                },
                "context_lines": {
                    "type": "integer",
                    "default": 2,
                    "description": "匹配行前后保留的上下文行数",
                },
                "required_phrases": {
                    "type": "string",
                    "description": "空格分隔的短语，命中片段必须同时包含这些短语",
                },
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["keywords"],
        }
    )

    async def _run(
        self,
        keywords: str,
        repo_filter: str = "",
        repository: str = "",
        namespace: str = "",
        path_prefix: str = "",
        document_ids: str = "",
        include_archived: bool = True,
        context_lines: int = 2,
        required_phrases: str = "",
        limit: int = 10,
        **_,
    ) -> dict:
        terms = [x for x in keywords.split() if x]
        required = [x for x in required_phrases.split() if x]
        repo = repository or repo_filter
        ids = _parse_document_ids(document_ids)
        hits = self._search(
            terms,
            repo_filter=repo,
            namespace=namespace,
            path_prefix=path_prefix,
            document_ids=ids,
            include_archived=include_archived,
            context_lines=context_lines,
            limit=limit,
        )
        # Fallback: long Chinese queries often fail exact substring matching.
        # Split into overlapping 2-character terms (e.g. "确认录取" → "确认 录取").
        if not hits:
            cleaned = keywords.strip().replace(" ", "")
            if len(cleaned) >= 4:
                split_terms = [
                    cleaned[i : i + 2] for i in range(0, len(cleaned) - 1, 2)
                ]
                hits = self._search(
                    split_terms,
                    repo_filter=repo,
                    namespace=namespace,
                    path_prefix=path_prefix,
                    document_ids=ids,
                    include_archived=include_archived,
                    context_lines=context_lines,
                    limit=limit,
                )

        if required:
            hits = filter_by_required_phrases(hits, required)

        if self.tracker:
            # Grep hits are candidate evidence only; the model must read the
            # actual document to ground an answer.
            hits = sorted(hits, key=lambda h: score_grep_hit(h, terms), reverse=True)
            reliable_count = sum(
                1 for h in hits if evaluate_grep_reliability(h, terms)[0]
            )
            before = len(self.tracker.candidate_sources)
            self.tracker.add_candidates(grep_hits_to_search_results(hits, terms))
            after = len(self.tracker.candidate_sources)
            logger.info(
                "NJU grep evidence: query=%r hits=%d reliable=%d "
                "candidates_before=%d candidates_after=%d added=%d",
                keywords,
                len(hits),
                reliable_count,
                before,
                after,
                after - before,
            )
            if getattr(self.tracker, "diagnostics", False):
                for i, hit in enumerate(hits[:10], 1):
                    logger.info(
                        "NJU grep candidate %d: title=%s path=%s score=%s "
                        "matched=%s",
                        i,
                        hit.get("title"),
                        hit.get("path"),
                        hit.get("score"),
                        hit.get("matched_keywords"),
                    )

        # Surface a recommended line range so the model can read efficiently.
        for hit in hits:
            hit["recommended_read_range"] = recommended_read_range(hit)

        return {"count": len(hits), "results": hits}

    def _search(
        self,
        terms: list[str],
        repo_filter: str,
        namespace: str,
        path_prefix: str,
        document_ids: set[str] | None,
        include_archived: bool,
        context_lines: int,
        limit: int,
    ) -> list[dict]:
        if not terms:
            return []

        # Expand action terms with known synonyms so "补办" also matches "补卡".
        term_expansions = {term: _expand_query_term(term) for term in terms}

        hits: list[dict] = []
        cf = context_lines
        for row in self.index.all_documents():
            if repo_filter and repo_filter.casefold() not in (row.get("repository") or "").casefold():
                continue
            if not _row_matches_scope(
                row,
                namespace=namespace,
                path_prefix=path_prefix,
                document_ids=document_ids,
                include_archived=include_archived,
            ):
                continue

            rel_path = row["path"]
            if not rel_path:
                continue
            try:
                lines = _load_cleaned_document_lines(self.docs_root, rel_path)
            except (OSError, ValueError):
                continue
            if not lines:
                continue

            matched_indices: set[int] = set()
            matched_terms: set[str] = set()
            expansion_lower = {
                term: [t.casefold() for t in expansions]
                for term, expansions in term_expansions.items()
            }
            for i, line in enumerate(lines):
                line_lower = line.casefold()
                for term, term_lowers in expansion_lower.items():
                    if any(tl in line_lower for tl in term_lowers):
                        matched_indices.add(i)
                        matched_terms.add(term)
            if not matched_indices:
                continue

            # Build contiguous windows around matched lines.
            sorted_idx = sorted(matched_indices)
            windows: list[tuple[int, int]] = []
            cur_start = cur_end = sorted_idx[0]
            for idx in sorted_idx[1:]:
                if idx - cur_end <= cf + 1:
                    cur_end = idx
                else:
                    windows.append((cur_start, cur_end))
                    cur_start = cur_end = idx
            windows.append((cur_start, cur_end))

            # Expand and merge windows.
            expanded = [
                (max(0, s - cf), min(len(lines) - 1, e + cf)) for s, e in windows
            ]
            merged: list[tuple[int, int]] = []
            for s, e in expanded:
                if merged and s <= merged[-1][1] + 1:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                else:
                    merged.append((s, e))

            matches = []
            for s, e in merged:
                snippet = "\n".join(
                    f"{i + 1}: {lines[i]}" for i in range(s, e + 1)
                )
                matches.append(
                    {
                        "line_start": s + 1,
                        "line_end": e + 1,
                        "snippet": snippet,
                    }
                )

            public = doc_record_to_public_dict(row)
            hits.append(
                {
                    **public,
                    "score": round(
                        score_grep_hit(
                            {
                                **public,
                                "matched_keywords": sorted(matched_terms),
                                "matches": matches,
                            },
                            terms,
                        ),
                        3,
                    ),
                    "matched_keywords": sorted(matched_terms),
                    "matches": matches[:3],
                }
            )
        hits.sort(key=lambda x: x["score"], reverse=True)
        return hits[:limit]


@dataclass
class ReadDocTool(_Tool):
    name: str = "read_doc"
    description: str = (
        "通过 search/grep 工具返回的 file_path 安全读取正文。"
        "支持按字符分页（offset/limit）或按行号范围（start_line/end_line）读取。"
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "offset": {"type": "integer", "default": 0},
                "limit": {"type": "integer", "default": 12000},
                "start_line": {
                    "type": "integer",
                    "description": "与 end_line 配合时按行号读取，优先级高于 offset",
                },
                "end_line": {"type": "integer"},
                "context_lines": {"type": "integer", "default": 0},
            },
            "required": ["file_path"],
        }
    )

    async def _run(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 12000,
        start_line: int | None = None,
        end_line: int | None = None,
        context_lines: int = 0,
        **_,
    ) -> dict:
        try:
            if start_line is not None or end_line is not None:
                lines = _load_cleaned_document_lines(self.docs_root, file_path)
                start, end, warnings = _clamp_read_range(
                    start_line, end_line, len(lines)
                )
                result = read_document_lines(
                    self.docs_root,
                    file_path,
                    start_line=start,
                    end_line=end,
                    context_lines=context_lines,
                )
                if warnings:
                    result["warnings"] = warnings
            else:
                result = read_document_content(
                    self.docs_root, file_path, offset, limit
                )
        except (OSError, ValueError) as exc:
            return {"error": str(exc), "file_path": file_path}

        result = _truncate_read_result(result)

        if self.tracker:
            resolved_path = result.get("file_path", file_path)
            line_start = result.get("start_line")
            line_end = result.get("end_line")
            if getattr(self.tracker, "diagnostics", False):
                logger.info(
                    "NJU read_doc: file=%s lines=%s:%s chars=%d truncated=%s",
                    resolved_path,
                    line_start,
                    line_end,
                    len(result.get("content", "")),
                    result.get("truncated", False),
                )
            row = next(
                (
                    r
                    for r in self.index.all_documents()
                    if r["path"] == resolved_path
                ),
                None,
            )
            if row is not None:
                document = document_from_index_row(row, result["content"])
                self.tracker.add_read_document(
                    document,
                    result["content"],
                    line_start=line_start,
                    line_end=line_end,
                )
            else:
                self.tracker.add_evidence(
                    evidence_excerpt_from_text(
                        content=result["content"][:2400],
                        title=resolved_path,
                        file_path=resolved_path,
                        evidence_type="read",
                    )
                )
        return result


@dataclass
class SearchDocsTool(_Tool):
    name: str = "search_docs"
    description: str = (
        "按标题、slug、语雀 ID 或 URL 查询 SQLite 元数据；同名标题会保留全部候选。"
        "支持按 namespace、repository、path_prefix 和归档开关进行作用域过滤。"
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "标题模糊查询关键字"},
                "title_query": {"type": "string", "description": "标题模糊查询关键字（与 query 二选一）"},
                "yuque_id": {"type": "string"},
                "slug": {"type": "string"},
                "url": {"type": "string"},
                "namespace": {"type": "string"},
                "repository": {"type": "string"},
                "path_prefix": {"type": "string"},
                "include_archived": {"type": "boolean", "default": True},
                "limit": {"type": "integer", "default": 20},
                "offset": {"type": "integer", "default": 0},
            },
            "required": [],
        }
    )

    async def _run(
        self,
        query: str = "",
        title_query: str = "",
        yuque_id: str = "",
        slug: str = "",
        url: str = "",
        namespace: str = "",
        repository: str = "",
        path_prefix: str = "",
        include_archived: bool = True,
        limit: int = 20,
        offset: int = 0,
        **_,
    ) -> dict:
        # Fetch a generous candidate set from SQLite then apply scope filters.
        rows = self.index.find(
            query or title_query,
            yuque_id=yuque_id,
            slug=slug,
            url=url,
            limit=1000,
            offset=0,
        )
        filtered = [
            row
            for row in rows
            if _row_matches_scope(
                row,
                namespace=namespace,
                repository=repository,
                path_prefix=path_prefix,
                include_archived=include_archived,
            )
        ]
        filtered.sort(key=lambda r: r["updated_at"] or "", reverse=True)
        start = max(offset, 0)
        end = start + max(limit, 1)
        page = filtered[start:end]
        return {
            "count": len(page),
            "total": len(filtered),
            "results": [doc_record_to_public_dict(row) for row in page],
            "has_more": len(filtered) > end,
            "next_offset": end if len(filtered) > end else None,
        }


@dataclass
class GetDocDetailsTool(_Tool):
    name: str = "get_doc_details"
    description: str = "通过 title、file_path、yuque_id 或 URL 获取元数据和可选正文；同名标题不擅自选择。"
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "file_path": {"type": "string"},
                "yuque_id": {"type": "string"},
                "url": {"type": "string"},
                "include_content": {"type": "boolean", "default": False},
                "offset": {"type": "integer", "default": 0},
                "limit": {"type": "integer", "default": 12000},
            },
            "required": [],
        }
    )

    async def _run(
        self,
        title: str = "",
        file_path: str = "",
        yuque_id: str = "",
        url: str = "",
        include_content: bool = False,
        offset: int = 0,
        limit: int = 12000,
        **_,
    ) -> dict:
        rows = [
            row
            for row in self.index.find(title, yuque_id=yuque_id, url=url, limit=100)
            if not file_path or row["path"] == file_path
        ]
        results = [doc_record_to_public_dict(row) for row in rows]
        if len(results) != 1 or not include_content:
            return {"count": len(results), "results": results}
        try:
            result = {
                "count": 1,
                "document": results[0],
                **read_document_content(self.docs_root, results[0]["path"], offset, limit),
            }
        except ValueError as exc:
            return {
                "count": 1,
                "document": results[0],
                "error": str(exc),
                "file_path": results[0]["path"],
            }
        if self.tracker:
            row = next(
                (r for r in rows if r["path"] == results[0]["path"]),
                None,
            )
            # Details reads are also bounded to the evidence context window.
            content_for_evidence = _truncate_content_at_boundary(
                result.get("content", ""), _MAX_READ_CHARS
            )
            if row is not None:
                document = document_from_index_row(row, content_for_evidence)
                self.tracker.add_read_document(document, content_for_evidence)
            else:
                self.tracker.add_evidence(
                    evidence_excerpt_from_text(
                        content=content_for_evidence,
                        title=results[0].get("title", results[0]["path"]),
                        file_path=results[0]["path"],
                        url=results[0].get("url", ""),
                        evidence_type="details",
                    )
                )
        return result


@dataclass
class ParseYuqueUrlTool(_Tool):
    name: str = "parse_yuque_url"
    description: str = (
        "解析带 query 或 anchor 的语雀文档链接，并定位配置范围内的本地文档。"
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        }
    )

    async def _run(self, url: str, **_) -> dict:
        parsed = parse_yuque_doc_url(url)
        if not parsed:
            return {"count": 0, "results": []}
        namespace, slug = parsed
        # The NOVA adapter keeps local paths/namespaces for corpus navigation,
        # while retaining each document's original Yuque URL. Match that URL
        # first so query strings and anchors still resolve to the mirrored doc.
        rows = [
            row
            for row in self.index.all_documents()
            if parse_yuque_doc_url(str(row["url"] or "")) == parsed
        ]
        if rows:
            return {
                "count": len(rows),
                "results": [doc_record_to_public_dict(row) for row in rows],
            }
        normalized = normalize_namespace(namespace)
        rows = [
            r
            for r in self.index.find(slug=slug, limit=20)
            if normalize_namespace(_row_value(r, "namespace", "") or "") == normalized
            or _namespace_matches(
                _path_segments(_normalize_path(_row_value(r, "path", ""))),
                _path_segments(normalized),
            )
        ]
        return {
            "count": len(rows),
            "results": [doc_record_to_public_dict(row) for row in rows],
        }


@dataclass
class ListKnowledgeBasesTool(_Tool):
    name: str = "list_knowledge_bases"
    description: str = (
        "列出已同步知识库（namespace）、repository、文档数量及顶层分类统计。"
        "返回结果中的 top_level_categories 字段是该知识库的顶层分类列表。"
    )
    parameters: dict = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )

    async def _run(self, **_) -> dict:
        summaries = build_knowledge_base_summaries(
            self.index.all_documents(), include_archived=True
        )
        knowledge_bases = []
        for summary in summaries:
            knowledge_bases.append(
                {
                    "namespace": summary.namespace,
                    "repository": summary.repository,
                    "document_count": summary.document_count,
                    "documents": summary.document_count,
                    "top_level_categories": [
                        {"name": cat.name, "document_count": cat.document_count}
                        for cat in summary.top_level_categories
                    ],
                }
            )
        result = {"knowledge_bases": knowledge_bases}
        if self.tracker:
            self._record_navigation_evidence(
                title="已同步知识库列表",
                content=json.dumps(knowledge_bases, ensure_ascii=False, indent=2),
            )
        return result


@dataclass
class ListRepoDocsTool(_Tool):
    name: str = "list_repo_docs"
    description: str = (
        "列出指定知识库作用域下的文档、直接子分类和分页信息。"
        "path_prefix 相对于 namespace。默认返回 20 条，可通过 offset/limit 翻页。"
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "namespace": {"type": "string"},
                "path_prefix": {
                    "type": "string",
                    "description": "相对于 namespace 的路径前缀，例如 02_教务与学业/课程与选课",
                },
                "title_query": {"type": "string"},
                "include_archived": {"type": "boolean", "default": False},
                "include_index": {"type": "boolean", "default": True},
                "limit": {"type": "integer", "default": 20},
                "offset": {"type": "integer", "default": 0},
            },
            "required": ["namespace"],
        }
    )

    async def _run(
        self,
        namespace: str,
        path_prefix: str = "",
        title_query: str = "",
        include_archived: bool = False,
        include_index: bool = True,
        limit: int = 20,
        offset: int = 0,
        **_,
    ) -> dict:
        rows = self.index.all_documents()
        docs, has_more = list_documents_under_prefix(
            rows,
            namespace=namespace,
            path_prefix=path_prefix,
            title_query=title_query,
            include_archived=include_archived,
            include_index=include_index,
            limit=limit,
            offset=offset,
        )
        tree = build_knowledge_tree(
            rows,
            namespace=namespace,
            path_prefix=path_prefix,
            max_depth=1,
            include_archived=include_archived,
        )
        categories = [
            {
                "name": child.name,
                "path_prefix": child.path_prefix,
                "document_count": child.document_count,
                "is_index": child.is_index,
            }
            for child in (tree.children if tree else ())
        ]
        result = {
            "documents": docs,
            "categories": categories,
            "count": len(docs),
            "has_more": has_more,
            "next_offset": offset + len(docs) if has_more else None,
        }
        if self.tracker:
            self._record_navigation_evidence(
                title=f"知识库 {namespace} 文档列表",
                file_path=path_prefix,
                content=json.dumps(result, ensure_ascii=False, indent=2),
            )
        return result


@dataclass
class ListRepoTreeTool(_Tool):
    name: str = "list_repo_tree"
    description: str = (
        "返回指定知识库作用域下的目录树，包括分类名称和文档数量。"
        "适合在检索前快速了解知识库结构。"
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "namespace": {"type": "string"},
                "path_prefix": {
                    "type": "string",
                    "description": "相对于 namespace 的路径前缀",
                },
                "max_depth": {
                    "type": "integer",
                    "default": 3,
                    "description": "返回的最大树深度（0 表示不限制）",
                },
                "include_archived": {"type": "boolean", "default": False},
            },
            "required": ["namespace"],
        }
    )

    async def _run(
        self,
        namespace: str,
        path_prefix: str = "",
        max_depth: int = 3,
        include_archived: bool = False,
        **_,
    ) -> dict:
        tree = build_knowledge_tree(
            self.index.all_documents(),
            namespace=namespace,
            path_prefix=path_prefix,
            max_depth=max_depth or 0,
            include_archived=include_archived,
        )
        if tree is None:
            return {"tree_text": "", "tree": None, "count": 0}

        def node_to_dict(node):
            return {
                "name": node.name,
                "path_prefix": node.path_prefix,
                "depth": node.depth,
                "document_count": node.document_count,
                "is_index": node.is_index,
                "children": [node_to_dict(child) for child in node.children],
            }

        result = {
            "tree_text": tree_to_text(tree),
            "tree": node_to_dict(tree),
            "count": tree.document_count,
        }
        if self.tracker:
            self._record_navigation_evidence(
                title=f"知识库 {namespace} 目录树",
                file_path=path_prefix,
                content=result["tree_text"],
            )
        return result


@dataclass
class GetDocOutlineTool(_Tool):
    name: str = "get_doc_outline"
    description: str = (
        "读取指定文档的章节大纲（Markdown 标题），返回标题层级、起始/结束行号。"
        "可选按查询词对章节做相关性排序并返回最相关的若干章节。"
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "max_depth": {
                    "type": "integer",
                    "default": 3,
                    "description": "最大标题层级（1-6）",
                },
                "query": {
                    "type": "string",
                    "description": "可选查询词，命中标题或正文的章节会排在前面",
                },
                "limit": {"type": "integer", "default": 20},
            },
            "required": ["file_path"],
        }
    )

    async def _run(
        self,
        file_path: str,
        max_depth: int = 3,
        query: str = "",
        limit: int = 20,
        **_,
    ) -> dict:
        try:
            lines = _load_cleaned_document_lines(self.docs_root, file_path)
        except (OSError, ValueError) as exc:
            return {"error": str(exc), "file_path": file_path}

        headings: list[dict] = []
        for i, line in enumerate(lines):
            match = _HEADING_RE.match(line)
            if not match:
                continue
            level = len(match.group(1))
            if level > max(max_depth, 1):
                continue
            headings.append(
                {
                    "level": level,
                    "title": match.group(2).strip(),
                    "line_number": i + 1,
                }
            )

        # Determine section boundaries.
        for idx, heading in enumerate(headings):
            next_start = (
                headings[idx + 1]["line_number"]
                if idx + 1 < len(headings)
                else len(lines) + 1
            )
            heading["end_line"] = next_start - 1

        if query:
            terms = [t for t in query.split() if t]

            def _relevance(heading: dict) -> int:
                title_score = sum(
                    1 for t in terms if t.casefold() in heading["title"].casefold()
                )
                body = "\n".join(
                    lines[heading["line_number"] - 1 : heading["end_line"]]
                )
                body_score = sum(
                    1 for t in terms if t.casefold() in body.casefold()
                )
                return title_score * 3 + body_score

            headings.sort(key=_relevance, reverse=True)
            headings = headings[:limit]
            # Keep relevance order when a query is provided.
            result = {
                "file_path": file_path,
                "title": headings[0]["title"] if headings else "",
                "section_count": len(headings),
                "sections": headings,
            }
        else:
            headings.sort(key=lambda h: h["line_number"])
            result = {
                "file_path": file_path,
                "title": headings[0]["title"] if headings else "",
                "section_count": len(headings),
                "sections": headings,
            }

        if self.tracker:
            self._record_navigation_evidence(
                title=f"文档大纲：{result.get('title', file_path)}",
                file_path=file_path,
                content=json.dumps(result, ensure_ascii=False, indent=2),
            )
        return result


@dataclass
class DocStatsTool(_Tool):
    name: str = "doc_stats"
    description: str = "返回本地文档、chunk 和向量索引数量。"
    parameters: dict = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )

    async def _run(self, **_) -> dict:
        result = {
            "sqlite_documents": self.index.document_count(),
            "vector_documents": self.index.vector_count(),
        }
        if hasattr(self, "chunk_store") and self.chunk_store is not None:
            result["chunk_documents"] = self.chunk_store.documents_with_chunks()
            result["chunk_total"] = self.chunk_store.chunk_count()
        return result
