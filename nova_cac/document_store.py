from __future__ import annotations
import re
from pathlib import Path
from typing import Iterable
import yaml
from .models import Document


class DocumentStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def safe_name(value: str) -> str:
        value = (
            re.sub(r"[\\/:*?\"<>|\x00-\x1f]+", "_", value)
            .replace("..", "_")
            .strip(" .")
        )
        return (value or "untitled")[:120]

    def path_for(
        self,
        namespace: str,
        parents: Iterable[str],
        title: str,
        doc_id: str,
        used: set[Path],
    ) -> Path:
        base = self.root / self.safe_name(namespace)
        for parent in parents:
            base /= self.safe_name(parent)
        base.mkdir(parents=True, exist_ok=True)
        stem = self.safe_name(title)
        candidate = base / f"{stem}.md"
        # Used set contains root-relative paths; compare in the same space.
        rel = candidate.resolve().relative_to(self.root.resolve())
        suffix = 2
        while (
            rel in used
            or candidate.exists()
            and not self._has_id(candidate, doc_id)
        ):
            candidate = base / f"{stem}_{suffix}.md"
            rel = candidate.resolve().relative_to(self.root.resolve())
            suffix += 1
        # Persist relative paths so the SQLite records stay valid regardless of cwd.
        return rel

    def _has_id(self, path: Path, doc_id: str) -> bool:
        try:
            return str(self.read(path).yuque_id) == str(doc_id)
        except (OSError, ValueError):
            return False

    @staticmethod
    def _inside_root(target: Path, root: Path) -> bool:
        try:
            target.resolve().relative_to(root.resolve())
            return True
        except ValueError:
            return False

    def _to_target(self, path: Path) -> Path:
        """Resolve a stored path (absolute or relative to root) into a filesystem path."""
        target = path if path.is_absolute() else self.root / path
        if not self._inside_root(target, self.root):
            raise ValueError("document path outside document root")
        return target

    def write(self, doc: Document) -> None:
        if not doc.path:
            raise ValueError("document path required")
        target = self._to_target(doc.path)
        fm = {
            "yuque_id": doc.yuque_id,
            "title": doc.title,
            "repository": doc.repository,
            "namespace": doc.namespace,
            "slug": doc.slug,
            "url": doc.url,
            "created_at": doc.created_at,
            "updated_at": doc.updated_at,
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "---\n"
            + yaml.safe_dump(fm, allow_unicode=True, sort_keys=False)
            + "---\n\n"
            + doc.body,
            encoding="utf-8",
        )

    def read(self, path: Path) -> Document:
        target = self._to_target(path)
        raw = target.read_text(encoding="utf-8")
        if not raw.startswith("---\n"):
            raise ValueError("missing frontmatter")
        _, head, body = raw.split("---\n", 2)
        fm = yaml.safe_load(head) or {}
        required = (
            "yuque_id",
            "title",
            "repository",
            "namespace",
            "slug",
            "url",
            "created_at",
            "updated_at",
        )
        if any(k not in fm for k in required):
            raise ValueError("incomplete frontmatter")
        return Document(
            **{k: str(fm[k]) for k in required}, body=body.lstrip("\n"), path=path
        )

    def remove(self, path: Path) -> None:
        target = self._to_target(path)
        target.unlink(missing_ok=True)
