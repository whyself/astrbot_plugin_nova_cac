from __future__ import annotations
import json
import math
import re
import sqlite3
from pathlib import Path
from .models import Document


class DocumentIndex:
    def __init__(self, path: Path):
        self.path, self.conn = path, None

    def open(self) -> None:
        if self.conn:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS documents (yuque_id TEXT PRIMARY KEY,title TEXT,repository TEXT,namespace TEXT,slug TEXT,url TEXT,created_at TEXT,updated_at TEXT,body TEXT,path TEXT,embedding TEXT)"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY,value TEXT)"
        )
        self.conn.commit()

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None

    def upsert(self, doc: Document, embedding: list[float] | None = None) -> None:
        self.open()
        self.conn.execute(
            "INSERT INTO documents VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(yuque_id) DO UPDATE SET title=excluded.title,repository=excluded.repository,namespace=excluded.namespace,slug=excluded.slug,url=excluded.url,created_at=excluded.created_at,updated_at=excluded.updated_at,body=excluded.body,path=excluded.path,embedding=COALESCE(excluded.embedding,documents.embedding)",
            (
                doc.yuque_id,
                doc.title,
                doc.repository,
                doc.namespace,
                doc.slug,
                doc.url,
                doc.created_at,
                doc.updated_at,
                doc.body,
                str(doc.path or ""),
                json.dumps(embedding) if embedding else None,
            ),
        )
        self.conn.commit()

    def delete_missing(self, namespace: str, ids: set[str]) -> list[sqlite3.Row]:
        self.open()
        rows = self.conn.execute(
            "SELECT * FROM documents WHERE namespace=?", (namespace,)
        ).fetchall()
        doomed = [r for r in rows if r["yuque_id"] not in ids]
        self.conn.executemany(
            "DELETE FROM documents WHERE yuque_id=?", [(r["yuque_id"],) for r in doomed]
        )
        self.conn.commit()
        return doomed

    def all_documents(self):
        self.open()
        return self.conn.execute("SELECT * FROM documents").fetchall()

    def document_count(self) -> int:
        return len(self.all_documents())

    def vector_count(self) -> int:
        self.open()
        return self.conn.execute(
            "SELECT count(*) FROM documents WHERE embedding IS NOT NULL"
        ).fetchone()[0]

    def find(
        self,
        query: str = "",
        *,
        title: str = "",
        yuque_id: str = "",
        slug: str = "",
        url: str = "",
        limit: int = 20,
        offset: int = 0,
    ):
        self.open()
        clauses = []
        values = []
        for column, value in (
            ("title", title or query),
            ("yuque_id", yuque_id),
            ("slug", slug),
            ("url", url),
        ):
            if value:
                clauses.append(f"{column} LIKE ?")
                values.append(f"%{value}%")
        sql = (
            "SELECT * FROM documents"
            + (" WHERE " + " AND ".join(clauses) if clauses else "")
            + " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        )
        return self.conn.execute(
            sql, (*values, min(max(limit, 1), 100), max(offset, 0))
        ).fetchall()

    def keyword(self, query: str, limit: int):
        self.open()
        terms = self._terms(query)
        rows = self.all_documents()
        ranked = []
        for row in rows:
            text = (row["title"] + " " + row["body"]).casefold()
            title = row["title"].casefold()
            matched = [term for term in terms if term in text]
            if matched:
                # Coverage is bounded: repeated boilerplate cannot saturate a score.
                score = len(matched) / max(len(terms), 1)
                score += 0.25 * sum(term in title for term in matched) / len(matched)
                ranked.append((row, min(score, 1.0)))
        return sorted(ranked, key=lambda item: item[1], reverse=True)[:limit]

    @staticmethod
    def _terms(query: str) -> list[str]:
        words = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]{2,}", query.casefold())
        terms = []
        for word in words:
            terms.append(word)
            if len(word) > 2 and re.fullmatch(r"[\u4e00-\u9fff]+", word):
                terms.extend(word[i : i + 2] for i in range(len(word) - 1))
        return list(dict.fromkeys(terms))

    def vector(self, vector: list[float], limit: int):
        scored = []
        for row in self.all_documents():
            if not row["embedding"]:
                continue
            other = json.loads(row["embedding"])
            den = math.sqrt(sum(x * x for x in vector)) * math.sqrt(
                sum(x * x for x in other)
            )
            scored.append(
                (row, sum(a * b for a, b in zip(vector, other)) / den if den else 0)
            )
        return sorted(scored, key=lambda x: x[1], reverse=True)[:limit]

    def set_state(self, key: str, value: str):
        self.open()
        self.conn.execute(
            "INSERT INTO state VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self.conn.commit()

    def get_state(self, key: str):
        self.open()
        row = self.conn.execute(
            "SELECT value FROM state WHERE key=?", (key,)
        ).fetchone()
        return row[0] if row else None
