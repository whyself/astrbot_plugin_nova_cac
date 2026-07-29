"""Convert grep hits and read documents into a unified evidence model."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

from .models import ChunkResult, Document, SearchResult


@dataclass
class EvidenceExcerpt:
    """A concrete piece of evidence that has been read into the Agent context."""

    evidence_id: str = ""
    document_id: str = ""
    title: str = ""
    url: str = ""
    file_path: str = ""
    line_start: int | None = None
    line_end: int | None = None
    content: str = ""
    evidence_type: str = "read"  # read | details | outline | navigation
    qa_status: str | None = None
    historical: bool = False
    score: float = 0.0
    # Deterministic version / applicability metadata.
    applicable_years: list[int] | None = None
    applicable_cohorts: list[str] | None = None
    document_year: int | None = None
    version_status: str | None = None
    historical_reason: str | None = None


_APPLICABILITY_KEYWORDS: frozenset[str] = frozenset({
    "适用",
    "面向",
    "针对",
    "对象",
    "培养方案",
    "教学计划",
    "方案",
    "年级",
    "级学生",
    "届学生",
    "学年",
    "入学",
})


def _extract_years(text: str) -> list[int]:
    """Return sorted unique 4-digit years (2000-2099) found in ``text``."""
    years = {int(m) for m in re.findall(r"20\d{2}", text)}
    return sorted(years)


def _extract_cohorts(text: str) -> list[str]:
    """Return cohort markers like ``2024级`` or ``2023届`` found in ``text``."""
    markers: list[str] = []
    for m in re.finditer(r"20\d{2}\s*(?:级|届|学年)", text):
        markers.append(re.sub(r"\s+", "", m.group(0)))
    return sorted(set(markers))


def _in_applicability_context(text: str, start: int, length: int) -> bool:
    """Return True when a year/range sits in an explicit applicability statement."""
    window = text[max(0, start - 15) : start + length + 15]
    if any(kw in window for kw in _APPLICABILITY_KEYWORDS):
        return True
    # A standalone cohort marker (e.g. ``2024级``) is itself version information.
    if re.search(r"20\d{2}\s*(?:级|届|学年)", window):
        return True
    return False


def _extract_year_range(text: str) -> list[int]:
    """Expand explicit ranges such as ``2024-2025`` into individual years."""
    years: set[int] = set()
    for match in re.finditer(r"(20\d{2})\s*[-~]\s*(20\d{2})", text):
        start, end = int(match.group(1)), int(match.group(2))
        if end >= start and end - start <= 10:
            if _in_applicability_context(text, match.start(), len(match.group(0))):
                years.update(range(start, end + 1))
    return sorted(years)


def extract_applicable_years(text: str) -> list[int]:
    """Return years the text explicitly applies to.

    Years that merely appear in ordinary narrative (e.g. ``发布于2024年``) are
    ignored unless they sit in an applicability statement.
    """
    years: set[int] = set(_extract_year_range(text))
    for match in re.finditer(r"20\d{2}", text):
        if _in_applicability_context(text, match.start(), 4):
            years.add(int(match.group(0)))
    return sorted(years)


def extract_applicable_cohorts(text: str) -> list[str]:
    """Return cohort markers that the text explicitly applies to."""
    markers: list[str] = []
    for match in re.finditer(r"20\d{2}\s*(?:级|届|学年)", text):
        if _in_applicability_context(text, match.start(), len(match.group(0))):
            markers.append(re.sub(r"\s+", "", match.group(0)))
    return sorted(set(markers))


def extract_document_year(title: str, path: str | None = None) -> int | None:
    """Return the single document-level year from title/path, if unambiguous."""
    years = _extract_years(title)
    if not years and path:
        years = _extract_years(path)
    return years[0] if years else None


def classify_version_status(
    title: str,
    path: str | None,
    content: str,
    applicable_years: list[int] | None,
    document_year: int | None,
) -> tuple[str, str | None]:
    """Return deterministic (version_status, historical_reason).

    Status values: ``current``, ``historical``, ``archived``, ``unknown``.
    A year-bearing title is not promoted to ``current`` by incidental words like
    ``当前`` or ``最新`` in the body; only explicit version claims count.
    """
    title_low = _normalize(title)
    path_low = _normalize(path or "")
    content_low = _normalize(content)

    if "归档" in path_low or "归档" in title_low:
        return "archived", "路径/标题包含“归档”"

    historical_markers = ("旧版", "历史", "往年", "archive", "未更新", "废止", "失效")
    for marker in historical_markers:
        if marker in title_low or marker in path_low or marker in content_low:
            return "historical", f"文本包含“{marker}”"

    # Explicit claims such as "本条例为最新版本" / "现行有效".
    explicit_current = any(m in title_low for m in ("最新", "现行", "当前"))
    current_phrases = (
        "为最新版本",
        "为当前版本",
        "为现行版本",
        "现行有效",
        "当前有效",
        "最新有效",
        "最新版本",
        "当前版本",
        "现行版本",
    )
    if any(p in content_low for p in current_phrases):
        explicit_current = True

    if document_year is not None:
        return (
            ("current", None)
            if explicit_current
            else ("historical", f"文档标题/路径包含年份 {document_year}")
        )

    if applicable_years:
        return (
            ("current", None)
            if explicit_current
            else ("historical", f"正文明确适用年份 {applicable_years[0]}")
        )

    return "unknown", None


def _populate_version_metadata(
    excerpt: EvidenceExcerpt,
    title: str,
    path: str | None,
    content: str,
) -> None:
    """Fill version/applicability fields on an excerpt in place."""
    excerpt.applicable_years = extract_applicable_years(content)
    excerpt.applicable_cohorts = extract_applicable_cohorts(content)
    excerpt.document_year = extract_document_year(title, path)
    excerpt.version_status, excerpt.historical_reason = classify_version_status(
        title,
        path,
        content,
        excerpt.applicable_years,
        excerpt.document_year,
    )


_QA_BLOCK_START_RE = re.compile(
    r"^(?:#{1,6}\s+|---+\s*|\*\*Q\d+\s+|Q\d+[\s:])", re.IGNORECASE
)


def _split_qa_blocks_with_lines(
    content: str,
) -> list[tuple[str, int | None, int | None]]:
    """Split a QA-style document into blocks and preserve each block's line numbers.

    Lines are expected to carry the ``N: `` prefix written by
    ``read_document_lines``.  When prefixes are missing, line numbers fall back
    to ``None``.
    """
    raw_lines = content.splitlines()
    parsed: list[tuple[int | None, str, str]] = []
    for raw in raw_lines:
        match = re.match(r"^(\d+):\s?(.*)$", raw)
        if match:
            parsed.append((int(match.group(1)), match.group(2), raw))
        else:
            parsed.append((None, raw, raw))

    blocks: list[list[tuple[int | None, str, str]]] = []
    for line_no, text, raw in parsed:
        if _QA_BLOCK_START_RE.match(text):
            blocks.append([])
        if not blocks:
            blocks.append([])
        blocks[-1].append((line_no, text, raw))

    if not blocks:
        start = next((n for n, _, _ in parsed if n is not None), None)
        end = next((n for n, _, _ in reversed(parsed) if n is not None), None)
        return [(content.strip(), start, end)]

    # Only treat as a splittable QA document when at least one block carries a
    # status marker.  Headings alone do not fragment ordinary guides.
    has_status = any(
        classify_qa_window("\n".join(t for _, t, _ in block))
        is not QaEvidenceStatus.UNKNOWN
        for block in blocks
    )
    if len(blocks) == 1 or not has_status:
        start = next((n for n, _, _ in parsed if n is not None), None)
        end = next((n for n, _, _ in reversed(parsed) if n is not None), None)
        return [(content.strip(), start, end)]

    result: list[tuple[str, int | None, int | None]] = []
    for block in blocks:
        if not block:
            continue
        if not any(t.strip() for _, t, _ in block):
            continue
        block_text = "\n".join(raw for _, _, raw in block).strip()
        nums = [n for n, _, _ in block if n is not None]
        start = nums[0] if nums else None
        end = nums[-1] if nums else None
        result.append((block_text, start, end))
    return result


def split_qa_blocks(content: str) -> list[str]:
    """Split a QA-style document into independent blocks.

    Normal articles are returned as a single block unless they contain QA
    status markers, so headings alone do not fragment the evidence.
    """
    return [block for block, _, _ in _split_qa_blocks_with_lines(content)]


def evidence_excerpts_from_read(
    document: Document,
    content: str,
    *,
    line_start: int | None = None,
    line_end: int | None = None,
    evidence_type: str = "read",
    score: float = 0.0,
) -> list[EvidenceExcerpt]:
    """Build EvidenceExcerpt(s) from a document read operation.

    QA documents are split into independent blocks so that ``no_answer`` or
    ``partially_resolved`` blocks do not pollute resolved ones.  Each block
    keeps its real line numbers within the original read window.
    """
    blocks = _split_qa_blocks_with_lines(content)
    excerpts: list[EvidenceExcerpt] = []
    for block, block_start, block_end in blocks:
        historical, _ = is_historical_document(
            document.title, str(document.path or "")
        )
        excerpt = EvidenceExcerpt(
            document_id=document.yuque_id,
            title=document.title,
            url=document.url,
            file_path=str(document.path or ""),
            line_start=block_start if block_start is not None else line_start,
            line_end=block_end if block_end is not None else line_end,
            content=block,
            evidence_type=evidence_type,
            qa_status=classify_qa_window(block).value,
            historical=historical,
            score=score,
        )
        _populate_version_metadata(
            excerpt, document.title, str(document.path or ""), block
        )
        excerpt.historical = excerpt.version_status in {"historical", "archived"}
        excerpts.append(excerpt)
    return excerpts


def evidence_excerpt_from_read(
    document: Document,
    content: str,
    *,
    line_start: int | None = None,
    line_end: int | None = None,
    evidence_type: str = "read",
    score: float = 0.0,
) -> EvidenceExcerpt:
    """Build a single EvidenceExcerpt from a document read operation."""
    return evidence_excerpts_from_read(
        document,
        content,
        line_start=line_start,
        line_end=line_end,
        evidence_type=evidence_type,
        score=score,
    )[0]


def evidence_excerpt_from_text(
    content: str,
    *,
    title: str = "",
    file_path: str = "",
    url: str = "",
    document_id: str = "",
    evidence_type: str = "navigation",
) -> EvidenceExcerpt:
    """Build an EvidenceExcerpt from a structured/navigation output."""
    return EvidenceExcerpt(
        document_id=document_id,
        title=title,
        url=url,
        file_path=file_path,
        line_start=None,
        line_end=None,
        content=content,
        evidence_type=evidence_type,
        qa_status=None,
        historical=False,
        score=0.0,
    )


# Lightweight action synonym dictionary.  Keep it small and testable.
_ACTION_SYNONYMS: dict[str, frozenset[str]] = {
    "补办": frozenset({"补办", "补卡", "重办"}),
    "挂失": frozenset({"挂失", "丢失"}),
    "上传": frozenset({"上传", "采集", "提交"}),
    "查询": frozenset({"查询", "查看"}),
    "领取": frozenset({"领取", "发放"}),
    "办理": frozenset({"办理", "申请"}),
}

_MODIFIER_TERMS: frozenset[str] = frozenset({
    "哪里",
    "在哪",
    "地点",
    "位置",
    "怎么",
    "如何",
    "多少",
    "时间",
    "什么时候",
    "要求",
    "流程",
    "方式",
    "办法",
    "入口",
    "网址",
    "地址",
})

# Aliases only affect modifier scoring; they never become part of the core
# coverage that gates reliability.
_MODIFIER_ALIASES: dict[str, tuple[str, ...]] = {
    "地点": ("位于", "地址", "前往", "放置", "服务大厅", "办理处"),
    "时间": ("工作时间", "开放时间", "几点", "时至", "期间"),
    "网址": ("入口", "网站", "链接", "访问", "平台"),
    "怎么": ("如何",),
    "多少": ("费用", "元", "钱"),
}

# Known objects that are often confused with the query subject.  When one of
# these objects takes the action instead of the real subject, the hit must be
# penalised.
_COMPETING_OBJECTS: tuple[str, ...] = (
    "学生证",
    "团员证",
    "身份证",
    "宿舍钥匙",
    "银行卡",
    "火车优惠卡",
)

_QA_SEPARATORS = re.compile(r"(?:^|\n)\s*(?:###|##|---|\*\*Q\d+|Q\d+)\s*", re.IGNORECASE)

_NO_ANSWER_MARKERS: tuple[str, ...] = (
    "no_answer_found",
    "暂无可靠答案",
    "暂无可用答案",
    "没有可靠答案",
    "待补充",
    "原答案已在最终筛选中剔除",
    "不可作为回答材料",
    "无答案",
)

_QA_STATUSES = (
    "resolved",
    "reviewed_resolved",
    "partially_resolved",
    "no_answer_found",
)


@dataclass(frozen=True)
class QueryIntentTerms:
    """A deterministic split of the user's query.

    * subject_terms  -- what the question is about (e.g. 校园卡).
    * action_terms   -- what the user wants to do (e.g. 补办).
    * modifier_terms -- answer-type hints (e.g. 地点, 时间).
    * original_terms -- raw terms from the tool call.
    """

    subject_terms: tuple[str, ...]
    action_terms: tuple[str, ...]
    modifier_terms: tuple[str, ...]
    original_terms: tuple[str, ...]

    @property
    def core_terms(self) -> tuple[str, ...]:
        return self.subject_terms + self.action_terms

    @property
    def action_expanded(self) -> frozenset[str]:
        """All surface forms for the requested actions."""
        result: set[str] = set()
        for canonical in self.action_terms:
            result.update(_ACTION_SYNONYMS.get(canonical, {canonical}))
        return frozenset(result)


class QaEvidenceStatus(Enum):
    RELIABLE = "reliable"
    PARTIAL = "partial"
    NO_ANSWER = "no_answer"
    UNKNOWN = "unknown"


def _normalize(text: str) -> str:
    """Case-insensitive comparison helper."""
    return text.casefold()


def _extract_terms(query: str) -> list[str]:
    """Return non-empty whitespace-separated terms."""
    return [t for t in query.split() if t]


def _canonical_action(term: str) -> str | None:
    """Return the canonical action name if ``term`` is an action synonym."""
    low = _normalize(term)
    for canonical, synonyms in _ACTION_SYNONYMS.items():
        if low == _normalize(canonical) or low in {_normalize(s) for s in synonyms}:
            return canonical
    return None


def parse_query_terms(terms: Sequence[str]) -> QueryIntentTerms:
    """Split raw query terms into subject / action / modifier buckets."""
    subject_terms: list[str] = []
    action_terms: list[str] = []
    modifier_terms: list[str] = []

    for term in terms:
        canonical = _canonical_action(term)
        if canonical is not None:
            if canonical not in action_terms:
                action_terms.append(canonical)
        elif _normalize(term) in {_normalize(m) for m in _MODIFIER_TERMS}:
            if term not in modifier_terms:
                modifier_terms.append(term)
        else:
            if term not in subject_terms:
                subject_terms.append(term)

    return QueryIntentTerms(
        subject_terms=tuple(subject_terms),
        action_terms=tuple(action_terms),
        modifier_terms=tuple(modifier_terms),
        original_terms=tuple(terms),
    )


def _combine_grep_snippets(hit: dict) -> str:
    """Merge grep match windows into one snippet body."""
    snippets = [
        match.get("snippet", "")
        for match in hit.get("matches", [])
        if match.get("snippet")
    ]
    return "\n\n".join(snippets)


def _window_texts(hit: dict) -> list[str]:
    """Return the text of every matched window."""
    return [
        match.get("snippet", "")
        for match in hit.get("matches", [])
        if match.get("snippet")
    ]


def _is_index_document(title: str, path: str | None = None) -> bool:
    """Heuristic for directory / table-of-contents documents."""
    lower = _normalize(title)
    if re.search(r"(^|[/_-])index\b|\b目录\b|\btoc\b|^00[_-]", lower):
        return True
    if path and "index" in Path(path).name.lower():
        return True
    return lower.startswith("00_")


def is_historical_document(title: str, path: str | None = None) -> tuple[bool, float]:
    """Return (is_historical, penalty).

    Historical / archived documents can still be useful, but they must not be
    treated as the primary current answer.
    """
    lower = _normalize(title)
    if re.search(r"\b20\d{2}\b", lower):
        return True, 0.4
    if path and "归档" in path.casefold():
        return True, 0.5
    if "未更新" in lower:
        return True, 0.3
    return False, 0.0


def build_document_from_grep_hit(hit: dict) -> Document:
    """Create a Document from a grep tool hit dict."""
    return Document(
        yuque_id=hit.get("yuque_id") or hit.get("id", ""),
        title=hit.get("title", ""),
        repository=hit.get("repository", ""),
        namespace=hit.get("namespace", ""),
        slug=hit.get("slug", ""),
        url=hit.get("url", ""),
        created_at=hit.get("created_at", ""),
        updated_at=hit.get("updated_at", ""),
        body=_combine_grep_snippets(hit),
        path=Path(hit["path"]) if hit.get("path") else None,
    )


def document_from_index_row(row, body: str = "") -> Document:
    """Create a Document from a DocumentIndex SQLite row."""
    return Document(
        yuque_id=row["yuque_id"],
        title=row["title"],
        repository=row["repository"],
        namespace=row["namespace"],
        slug=row["slug"],
        url=row["url"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        body=body or row["body"] or "",
        path=Path(row["path"]) if row["path"] else None,
    )


def _find_positions(text: str, terms: Sequence[str]) -> list[tuple[int, int]]:
    """Return (start, end) positions of every occurrence of ``terms``."""
    positions: list[tuple[int, int]] = []
    for term in terms:
        if not term:
            continue
        t = _normalize(term)
        start = 0
        while True:
            idx = text.find(t, start)
            if idx == -1:
                break
            positions.append((idx, idx + len(t)))
            start = idx + 1
    return positions


def detect_competing_object(
    text: str,
    *,
    subject_terms: Sequence[str],
    action_terms: Sequence[str],
) -> bool:
    """Return True when an action is bound to a different object than the subject."""
    norm = _normalize(text)
    subjects = [t for t in subject_terms if t]
    actions = [t for t in action_terms if t]
    if not subjects or not actions:
        return False

    subj_positions = _find_positions(norm, subjects)

    for obj in _COMPETING_OBJECTS:
        obj_low = _normalize(obj)
        for obj_match in re.finditer(re.escape(obj_low), norm):
            for action in actions:
                act_low = _normalize(action)
                for act_match in re.finditer(re.escape(act_low), norm):
                    # Adjacent within 4 characters counts as bound.
                    gap = min(
                        abs(obj_match.start() - act_match.end()),
                        abs(act_match.start() - obj_match.end()),
                    )
                    if gap > 4:
                        continue
                    span_start = min(obj_match.start(), act_match.start())
                    span_end = max(obj_match.end(), act_match.end())
                    # If the real subject is inside the same span, this is not
                    # a competing object phrase.
                    if any(
                        span_start <= start and end <= span_end
                        for start, end in subj_positions
                    ):
                        continue
                    return True
    return False


def subject_action_relevance(
    text: str,
    *,
    subject_terms: Sequence[str],
    action_terms: Sequence[str],
) -> float:
    """Score how tightly the subject and action are related in ``text``.

    Returns a value in [0.0, 1.0].  A high score requires the action to be
    close to the real subject, and not bound to a competing object.
    """
    norm = _normalize(text)
    subjects = [t for t in subject_terms if t]
    actions = [t for t in action_terms if t]
    if not subjects or not actions:
        return 0.0

    subj_positions = _find_positions(norm, subjects)
    act_positions = _find_positions(norm, actions)
    if not subj_positions or not act_positions:
        return 0.0

    min_gap = min(
        abs(s[1] - a[0]) for s in subj_positions for a in act_positions
    )

    if min_gap <= 6:
        score = 1.0
    elif min_gap <= 15:
        score = 0.8
    elif min_gap <= 40:
        score = 0.5
    else:
        score = 0.2

    if detect_competing_object(
        text, subject_terms=subjects, action_terms=actions
    ):
        score = max(0.0, score - 0.7)

    return score


def _core_coverage(body_norm: str, intent: QueryIntentTerms) -> float:
    """Fraction of core terms (subject + action) matched anywhere in the body.

    Action synonyms count as matches for the canonical action.
    """
    core_count = len(intent.core_terms)
    if core_count == 0:
        return 0.0

    matched_subjects = sum(
        1 for s in intent.subject_terms if _normalize(s) in body_norm
    )
    matched_actions = sum(
        1
        for a in intent.action_terms
        if any(
            _normalize(syn) in body_norm
            for syn in _ACTION_SYNONYMS.get(a, {a})
        )
    )
    return (matched_subjects + matched_actions) / core_count


def _same_window_core_coverage(
    windows: list[str], intent: QueryIntentTerms
) -> bool:
    """Return True when one window contains all core terms (action synonyms OK)."""
    if not intent.core_terms:
        return False

    for window in windows:
        norm = _normalize(window)
        subjects_ok = all(
            _normalize(s) in norm for s in intent.subject_terms
        )
        actions_ok = all(
            any(
                _normalize(syn) in norm
                for syn in _ACTION_SYNONYMS.get(a, {a})
            )
            for a in intent.action_terms
        )
        if subjects_ok and actions_ok:
            return True
    return False


def _relation_flags(
    text: str, intent: QueryIntentTerms
) -> tuple[bool, bool]:
    """Return (exact_target_phrase, target_phrase) for ``text``.

    An "exact" phrase keeps subject and action within 8 characters.
    A "target" phrase keeps them within 15 characters.
    """
    norm = _normalize(text)
    if not intent.subject_terms or not intent.action_terms:
        return False, False

    subj_positions = _find_positions(norm, intent.subject_terms)
    act_positions = _find_positions(norm, intent.action_expanded)
    if not subj_positions or not act_positions:
        return False, False

    min_gap = min(
        abs(s[1] - a[0]) for s in subj_positions for a in act_positions
    )
    return min_gap <= 8, min_gap <= 15


def classify_qa_window(text: str) -> QaEvidenceStatus:
    """Classify a single QA window, with no-answer markers taking priority."""
    lower = _normalize(text)

    # Negative markers must be checked first.
    for marker in _NO_ANSWER_MARKERS:
        if re.search(re.escape(_normalize(marker)), lower):
            return QaEvidenceStatus.NO_ANSWER

    if re.search(r"\breviewed_resolved\b", lower):
        return QaEvidenceStatus.RELIABLE
    if re.search(r"\bresolved\b", lower):
        return QaEvidenceStatus.RELIABLE
    if re.search(r"\bpartially_resolved\b", lower):
        return QaEvidenceStatus.PARTIAL

    return QaEvidenceStatus.UNKNOWN


def _block_is_relevant(block: str, intent: QueryIntentTerms) -> bool:
    """A QA block is relevant only when it contains all core terms."""
    norm = _normalize(block)
    subjects_ok = all(_normalize(s) in norm for s in intent.subject_terms)
    actions_ok = all(
        any(
            _normalize(syn) in norm
            for syn in _ACTION_SYNONYMS.get(a, {a})
        )
        for a in intent.action_terms
    )
    return subjects_ok and actions_ok


def _qa_status_for_window(
    window: str, intent: QueryIntentTerms
) -> QaEvidenceStatus:
    """Classify a window, splitting on QA boundaries and using relevant blocks."""
    blocks = [b.strip() for b in _QA_SEPARATORS.split(window) if b.strip()]
    if not blocks:
        blocks = [window]

    relevant = [b for b in blocks if _block_is_relevant(b, intent)]
    if not relevant:
        relevant = blocks

    statuses = [classify_qa_window(b) for b in relevant]
    if any(s is QaEvidenceStatus.NO_ANSWER for s in statuses):
        return QaEvidenceStatus.NO_ANSWER
    if any(s is QaEvidenceStatus.RELIABLE for s in statuses):
        return QaEvidenceStatus.RELIABLE
    if any(s is QaEvidenceStatus.PARTIAL for s in statuses):
        return QaEvidenceStatus.PARTIAL
    return QaEvidenceStatus.UNKNOWN


def _overall_qa_status(
    windows: list[str], intent: QueryIntentTerms
) -> QaEvidenceStatus:
    """Aggregate QA status across windows.  One no-answer window disables reliability."""
    if not windows:
        return QaEvidenceStatus.UNKNOWN
    statuses = [_qa_status_for_window(w, intent) for w in windows]
    if any(s is QaEvidenceStatus.NO_ANSWER for s in statuses):
        return QaEvidenceStatus.NO_ANSWER
    if any(s is QaEvidenceStatus.RELIABLE for s in statuses):
        return QaEvidenceStatus.RELIABLE
    if any(s is QaEvidenceStatus.PARTIAL for s in statuses):
        return QaEvidenceStatus.PARTIAL
    return QaEvidenceStatus.UNKNOWN


def _modifier_coverage(body_norm: str, intent: QueryIntentTerms) -> float:
    """Fraction of explicit modifier terms (or their aliases) matched."""
    if not intent.modifier_terms:
        return 0.0
    matched = 0
    for term in intent.modifier_terms:
        if _normalize(term) in body_norm:
            matched += 1
            continue
        aliases = _MODIFIER_ALIASES.get(term, ())
        if any(_normalize(a) in body_norm for a in aliases):
            matched += 1
    return matched / len(intent.modifier_terms)


def evaluate_grep_reliability(hit: dict, query_terms: list[str]) -> tuple[bool, dict]:
    """Return (reliable, diagnostics) for a grep hit.

    Reliability depends on core term coverage, subject-action binding, and
    QA status.  Modifiers are intentionally not required.
    """
    intent = parse_query_terms(query_terms)
    if not intent.core_terms:
        return False, {"reason": "no_core_terms"}

    windows = _window_texts(hit)
    if not windows:
        return False, {"reason": "no_windows"}

    body = _combine_grep_snippets(hit)
    body_norm = _normalize(body)
    title = hit.get("title", "")

    core_coverage = _core_coverage(body_norm, intent)
    subj_action = subject_action_relevance(
        body, subject_terms=intent.subject_terms, action_terms=intent.action_expanded
    )
    title_action = subject_action_relevance(
        title, subject_terms=intent.subject_terms, action_terms=intent.action_expanded
    )
    same_window_core = _same_window_core_coverage(windows, intent)
    qa_status = _overall_qa_status(windows, intent)
    is_index = _is_index_document(title, hit.get("path"))
    historical, hist_penalty = is_historical_document(title, hit.get("path"))

    # A source is reliable when it directly covers the requested core topic,
    # the subject and action are bound together, and it is not a negative QA.
    # Same-window coverage alone is not enough when a competing object has
    # captured the action (e.g. "学生证补办" with 校园卡 only mentioned as an
    # accessory).
    relation_ok = (
        subj_action >= 0.3
        or title_action >= 0.3
        or (same_window_core and not detect_competing_object(
            body,
            subject_terms=intent.subject_terms,
            action_terms=intent.action_expanded,
        ))
    )
    reliable = (
        not is_index
        and core_coverage >= 1.0
        and qa_status is not QaEvidenceStatus.NO_ANSWER
        and relation_ok
    )

    diagnostics = {
        "core_coverage": core_coverage,
        "subject_action_score": subj_action,
        "title_action_score": title_action,
        "same_window_core_coverage": same_window_core,
        "qa_status": qa_status.value,
        "is_index_document": is_index,
        "is_historical": historical,
        "historical_penalty": hist_penalty,
    }
    return reliable, diagnostics


# Scoring weights.  Subject-action binding is the strongest signal.
CORE_COVERAGE_WEIGHT = 3.0
SUBJECT_ACTION_WEIGHT = 4.0
SAME_WINDOW_BONUS = 1.0
EXACT_PHRASE_BONUS = 1.5
HEADING_BONUS = 1.0
TITLE_TARGET_BONUS = 0.8
MODIFIER_WEIGHT = 0.25
WINDOW_BONUS = 0.1
COMPETING_OBJECT_PENALTY = 1.5
INDEX_PENALTY = 0.8
PARTIAL_CORE_PENALTY = 0.8
HISTORICAL_PENALTY = 0.4


def score_grep_hit(hit: dict, query_terms: list[str]) -> float:
    """Rank grep hits so direct answers outrank partial or relationally wrong matches."""
    intent = parse_query_terms(query_terms)
    if not intent.core_terms:
        return 0.0

    windows = _window_texts(hit)
    body = _combine_grep_snippets(hit)
    body_norm = _normalize(body)
    title = hit.get("title", "")

    core_coverage = _core_coverage(body_norm, intent)
    subj_action = subject_action_relevance(
        body, subject_terms=intent.subject_terms, action_terms=intent.action_expanded
    )
    title_action = subject_action_relevance(
        title, subject_terms=intent.subject_terms, action_terms=intent.action_expanded
    )
    same_window_core = _same_window_core_coverage(windows, intent)
    exact_phrase, target_phrase = _relation_flags(body, intent)
    title_exact, title_target = _relation_flags(title, intent)
    modifier_coverage = _modifier_coverage(body_norm, intent)

    relevant_windows = sum(
        1
        for w in windows
        if any(_normalize(t) in _normalize(w) for t in intent.core_terms)
    )

    competing = detect_competing_object(
        body, subject_terms=intent.subject_terms, action_terms=intent.action_expanded
    ) or detect_competing_object(
        title, subject_terms=intent.subject_terms, action_terms=intent.action_expanded
    )

    is_index = _is_index_document(title, hit.get("path"))
    historical, _ = is_historical_document(title, hit.get("path"))

    score = 0.0
    score += core_coverage * CORE_COVERAGE_WEIGHT
    score += subj_action * SUBJECT_ACTION_WEIGHT
    score += title_action * (SUBJECT_ACTION_WEIGHT * 0.35)

    if exact_phrase:
        score += EXACT_PHRASE_BONUS
    if title_exact:
        score += HEADING_BONUS
    if same_window_core:
        score += SAME_WINDOW_BONUS
    if title_target and not title_exact:
        score += TITLE_TARGET_BONUS

    score += modifier_coverage * MODIFIER_WEIGHT
    score += min(relevant_windows, 3) * WINDOW_BONUS

    if competing:
        score -= COMPETING_OBJECT_PENALTY
    if is_index:
        score -= INDEX_PENALTY
    if core_coverage < 1.0:
        score -= PARTIAL_CORE_PENALTY
    if historical:
        score -= HISTORICAL_PENALTY

    # Tiny extra penalty for single-keyword saturation so that documents
    # mentioning only one core term do not float up.
    if len(intent.core_terms) > 1 and core_coverage <= 1.0 / len(intent.core_terms):
        score -= 0.3

    return max(0.0, score)


def build_chunk_from_grep_hit(
    hit: dict, query_terms: list[str], rank: int = 0
) -> ChunkResult:
    """Build a ChunkResult from a grep hit so it enters the unified evidence chain."""
    doc_id = hit.get("yuque_id") or hit.get("id", "")
    body = _combine_grep_snippets(hit)
    score = score_grep_hit(hit, query_terms)
    matches = hit.get("matches", [{}])
    line_start = matches[0].get("line_start", 0)
    line_end = matches[-1].get("line_end", 0)
    reliable, _ = evaluate_grep_reliability(hit, query_terms)
    return ChunkResult(
        chunk_id=f"grep:{doc_id}:{line_start}:{line_end}",
        document_id=doc_id,
        title=hit.get("title", ""),
        content_snippet=body,
        source_url=hit.get("url", ""),
        keyword_score=score,
        final_score=score,
        retrieval_methods=("grep",),
        reliable=reliable,
        file_path=hit.get("path", ""),
        slug=hit.get("slug", ""),
        namespace=hit.get("namespace", ""),
    )


def grep_hits_to_search_results(
    hits: list[dict], query_terms: list[str]
) -> list[SearchResult]:
    """Convert ranked grep hits into SearchResult evidence objects."""
    ranked = sorted(
        hits, key=lambda hit: score_grep_hit(hit, query_terms), reverse=True
    )
    results: list[SearchResult] = []
    for i, hit in enumerate(ranked):
        chunk = build_chunk_from_grep_hit(hit, query_terms, rank=i)
        document = build_document_from_grep_hit(hit)
        results.append(
            SearchResult(
                source_id=f"G{i + 1}",
                document=document,
                score=chunk.final_score,
                chunk=chunk,
                keyword_score=chunk.keyword_score,
                retrieval_methods=("grep",),
                reliable=chunk.reliable,
            )
        )
    return results


def select_grounding_sources(
    sources: list[SearchResult], *, max_sources: int = 7
) -> list[SearchResult]:
    """Choose the best reliable sources for the grounded prompt.

    Unreliable sources are never promoted into the grounded set just to fill
    the cap.
    """
    seen: set[str] = set()
    reliable: list[SearchResult] = []
    for source in sources:
        if not source.reliable:
            continue
        key = source.document.yuque_id or source.document.url or source.document.title
        if key in seen:
            continue
        seen.add(key)
        reliable.append(source)

    reliable.sort(key=lambda s: s.score, reverse=True)
    return reliable[:max_sources]


def merge_search_results(a: SearchResult, b: SearchResult) -> SearchResult:
    """Merge two results for the same document, keeping the best of each field."""
    methods = tuple(
        dict.fromkeys([*a.retrieval_methods, *b.retrieval_methods])
    )
    score = max(a.score, b.score)
    keyword_score = max(a.keyword_score, b.keyword_score)
    vector_score = max(a.vector_score, b.vector_score)
    reliable = a.reliable or b.reliable

    # Prefer the document with more body text (usually the read/full version).
    document = a.document
    if len(b.document.body) > len(a.document.body):
        document = b.document

    # Choose the richer chunk, then overlay merged metadata.
    if a.chunk and b.chunk:
        chunk = (
            a.chunk
            if len(a.chunk.content_snippet) >= len(b.chunk.content_snippet)
            else b.chunk
        )
        chunk = replace(
            chunk,
            retrieval_methods=methods,
            final_score=max(a.chunk.final_score, b.chunk.final_score),
            keyword_score=max(a.chunk.keyword_score, b.chunk.keyword_score),
            reliable=reliable,
        )
    else:
        chunk = a.chunk or b.chunk
        if chunk:
            chunk = replace(chunk, retrieval_methods=methods, reliable=reliable)

    return SearchResult(
        source_id=a.source_id,
        document=document,
        score=score,
        chunk=chunk,
        vector_score=vector_score,
        keyword_score=keyword_score,
        retrieval_methods=methods,
        reliable=reliable,
    )


def recommended_read_range(hit: dict) -> dict[str, int]:
    """Return the inclusive line range that covers all matched windows in ``hit``."""
    matches = hit.get("matches", [])
    if not matches:
        return {}
    starts = [m.get("line_start") for m in matches if m.get("line_start") is not None]
    ends = [m.get("line_end") for m in matches if m.get("line_end") is not None]
    if not starts or not ends:
        return {}
    return {"start_line": min(starts), "end_line": max(ends)}


def filter_by_required_phrases(hits: list[dict], phrases: list[str]) -> list[dict]:
    """Keep only hits whose combined snippets contain every required phrase.

    Phrases are matched case-insensitively.  An empty phrase list is a no-op.
    """
    if not phrases:
        return hits
    required = [p.casefold() for p in phrases if p.strip()]
    filtered: list[dict] = []
    for hit in hits:
        body = _combine_grep_snippets(hit).casefold()
        if all(phrase in body for phrase in required):
            filtered.append(hit)
    return filtered
