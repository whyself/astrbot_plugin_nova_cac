"""Chunk-level keyword search with a simple BM25-like scorer."""

from __future__ import annotations
import math
import re
from collections import Counter
from dataclasses import dataclass
from .chunking import Chunk


_STOP_WORDS = frozenset(
    "的 是 了 在 和 有 我 他 她 它 你 我们 你们 他们 它们 这 那 这些 那些 "
    "一个 一些 什么 怎么 吗 呢 吧 啊 哦 嗯 对 不 没有 就是 可以 这个 那个 "
    "与 及 或 但 而 因为 所以 如果 即使 虽然 然而 因此 于是 之 其 所 被 把 让 "
    "to be a an the and or but in on at for with of is are was were".split()
)


def _is_url(token: str) -> bool:
    return token.startswith(("http://", "https://"))


def _tokenize(text: str) -> list[str]:
    """Tokenize query/document text.

    Returns lowercased English/number/URL tokens, Chinese single chars and bigrams.
    """
    text = text.lower()
    tokens: list[str] = []
    # URLs are preserved first.
    url_pattern = r"https?://[^\s<>()，。；：\"'（）]+"
    urls = re.findall(url_pattern, text)
    # Replace URLs with placeholders so they are not broken by later splitting.
    placeholder_map: dict[str, str] = {}
    for i, url in enumerate(urls):
        placeholder = f"\x00URL{i}\x00"
        placeholder_map[placeholder] = url.lower()
        text = text.replace(url, placeholder, 1)

    # English words, numbers, Chinese characters.
    pieces = re.findall(r"[a-z]+|[0-9]+|\x00URL\d+\x00|[一-鿿]", text)
    for piece in pieces:
        if piece in placeholder_map:
            tokens.append(placeholder_map[piece])
        elif re.fullmatch(r"[一-鿿]", piece):
            tokens.append(piece)
        else:
            tokens.append(piece)

    # Bigrams for consecutive Chinese characters.
    ch_chars = re.findall(r"[一-鿿]", text)
    tokens.extend("".join(ch_chars[i : i + 2]) for i in range(len(ch_chars) - 1))
    return tokens


def _extract_query_terms(query: str) -> list[str]:
    """Extract search terms, keeping order and dropping stop words and dupes."""
    raw = _tokenize(query)
    seen: set[str] = set()
    terms: list[str] = []
    for t in raw:
        if t in _STOP_WORDS or len(t) == 1 and not _is_url(t):
            continue
        if t not in seen:
            seen.add(t)
            terms.append(t)
    return terms


@dataclass
class _Posting:
    chunk_id: str
    term_frequency: int
    title_hits: int
    path_hits: int
    positions: list[int]


@dataclass
class KeywordHit:
    chunk: Chunk
    score: float
    matched_terms: list[str]
    title_match: bool
    phrase_match: bool


class ChunkKeywordIndex:
    """In-memory BM25-style index over a collection of chunks.

    The index is rebuilt from :class:`ChunkStore` on demand.  It is cheap to
    reconstruct for typical knowledge-base sizes (a few thousand chunks).
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._chunks: dict[str, Chunk] = {}
        self._index: dict[str, list[_Posting]] = {}
        self._avg_len = 0.0
        self._total_chunks = 0

    def build(self, chunks: list[Chunk]) -> None:
        self._chunks = {}
        self._index = {}
        total_len = 0
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
            title_tokens = _tokenize(chunk.title)
            body_tokens = _tokenize(chunk.content)
            path_tokens = _tokenize(chunk.file_path)
            positions = body_tokens + title_tokens + path_tokens
            # title tokens are appended after body tokens for positional phrase scoring.
            term_counts: Counter[str] = Counter(positions)
            title_counter: Counter[str] = Counter(title_tokens)
            path_counter: Counter[str] = Counter(path_tokens)
            length = len(body_tokens)
            total_len += max(length, 1)
            for term, freq in term_counts.items():
                posting = _Posting(
                    chunk_id=chunk.chunk_id,
                    term_frequency=freq,
                    title_hits=title_counter.get(term, 0),
                    path_hits=path_counter.get(term, 0),
                    positions=[i for i, t in enumerate(positions) if t == term],
                )
                self._index.setdefault(term, []).append(posting)
        self._total_chunks = len(self._chunks)
        self._avg_len = total_len / max(self._total_chunks, 1)

    def extract_terms(self, query: str) -> list[str]:
        return _extract_query_terms(query)

    def search(self, query: str, top_k: int = 20) -> list[KeywordHit]:
        terms = _extract_query_terms(query)
        if not terms or not self._chunks:
            return []

        # Document frequency for IDF.
        df = {term: len(self._index.get(term, [])) for term in terms}
        N = max(self._total_chunks, 1)
        idf = {
            term: math.log((N - df_t + 0.5) / (df_t + 0.5) + 1.0)
            for term, df_t in df.items()
        }

        # Aggregate per chunk.
        chunk_scores: dict[str, float] = {}
        chunk_terms: dict[str, set[str]] = {chunk_id: set() for chunk_id in self._chunks}
        chunk_title_hits: dict[str, int] = {}
        chunk_path_hits: dict[str, int] = {}
        chunk_phrase_hits: dict[str, int] = {}

        for term in terms:
            for posting in self._index.get(term, []):
                chunk = self._chunks[posting.chunk_id]
                body_len = max(len(_tokenize(chunk.content)), 1)
                denom = posting.term_frequency + self.k1 * (
                    1 - self.b + self.b * body_len / self._avg_len
                )
                if denom <= 0:
                    continue
                bm25 = idf[term] * (posting.term_frequency * (self.k1 + 1)) / denom
                # Title boost: title hits are strong signals; give them a large bonus.
                title_boost = 0.0
                if posting.title_hits:
                    title_boost = idf[term] * posting.title_hits * 2.0
                # Path boost: file path often contains categorization keywords.
                path_boost = 0.0
                if posting.path_hits:
                    path_boost = idf[term] * posting.path_hits * 1.0
                chunk_scores[chunk.chunk_id] = chunk_scores.get(chunk.chunk_id, 0.0) + bm25 + title_boost + path_boost
                chunk_terms[chunk.chunk_id].add(term)
                chunk_title_hits[chunk.chunk_id] = chunk_title_hits.get(chunk.chunk_id, 0) + posting.title_hits
                chunk_path_hits[chunk.chunk_id] = chunk_path_hits.get(chunk.chunk_id, 0) + posting.path_hits

        # Phrase bonus: consecutive query terms appear in order in the chunk.
        if len(terms) > 1:
            for chunk_id, chunk in self._chunks.items():
                positions_all: list[tuple[int, str]] = []
                tokens = _tokenize(chunk.title + " " + chunk.content + " " + chunk.file_path)
                for i, t in enumerate(tokens):
                    if t in terms:
                        positions_all.append((i, t))
                for i in range(len(positions_all) - 1):
                    a, b = positions_all[i], positions_all[i + 1]
                    if a[1] == terms[0] and b[1] == terms[1] and b[0] - a[0] == 1:
                        chunk_phrase_hits[chunk_id] = chunk_phrase_hits.get(chunk_id, 0) + 1

        # Apply coverage bonus and phrase bonus, then normalize.
        max_score = max(chunk_scores.values()) if chunk_scores else 0.0
        normalized: dict[str, float] = {}
        for chunk_id, raw in chunk_scores.items():
            coverage = len(chunk_terms[chunk_id]) / len(terms)
            coverage_bonus = coverage * 0.3 * raw if max_score > 0 else 0.0
            phrase_bonus = chunk_phrase_hits.get(chunk_id, 0) * 0.2 * raw if max_score > 0 else 0.0
            normalized[chunk_id] = raw + coverage_bonus + phrase_bonus

        max_norm = max(normalized.values()) if normalized else 0.0
        scale = max(max_norm, 1.0)

        hits = []
        for chunk_id, score in sorted(
            normalized.items(), key=lambda x: x[1], reverse=True
        )[:top_k]:
            coverage = len(chunk_terms[chunk_id]) / len(terms)
            if coverage < 0.25:
                continue
            hits.append(
                KeywordHit(
                    chunk=self._chunks[chunk_id],
                    score=min(1.0, score / scale),
                    matched_terms=sorted(chunk_terms[chunk_id]),
                    title_match=chunk_title_hits.get(chunk_id, 0) > 0,
                    phrase_match=chunk_phrase_hits.get(chunk_id, 0) > 0,
                )
            )
        return hits

    def count(self) -> int:
        return len(self._chunks)
