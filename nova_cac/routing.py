"""Small, testable routing helpers for the AstrBot adapter."""

from __future__ import annotations

import re

_CAC_COMMAND = re.compile(
    r"^\s*/cac(?:\s+(.*?))?\s*$",
    flags=re.IGNORECASE | re.DOTALL,
)


def extract_cac_query(message: str) -> str | None:
    """Extract a query from a complete /cac message, or return ``None``."""

    matched = _CAC_COMMAND.fullmatch(message)
    if matched is None:
        return None
    return (matched.group(1) or "").strip()
