"""Convert Markdown-flavoured text into a plain-text style suitable for QQ."""

from __future__ import annotations

import re


def markdown_to_plaintext(text: str) -> str:
    """Remove or rewrite Markdown markup so the result looks fine in plain text.

    Headings, bold/italic/strikethrough, code fences, blockquotes, tables,
    images and most list markers are normalised. URLs inside links are preserved.
    """
    if not text:
        return text

    # Tables must be handled before generic emphasis/link rewrites split rows.
    text = _convert_tables(text)

    # Fenced code blocks: keep content, drop fences and language hint.
    text = re.sub(
        r"```[\s\S]*?```",
        lambda m: _clean_codeblock(m.group(0)),
        text,
    )

    # Inline code -> plain text (remove backticks).
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # Images -> remove or replace with a short marker.
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"[图片: \2]", text)

    # Links -> text (url).
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)

    # Headings -> remove # markers.
    text = re.sub(r"^#{1,6}\s+(.*)$", r"\1", text, flags=re.MULTILINE)

    # Horizontal rules.
    text = re.sub(r"^[\*\-_]{3,}\s*$", "", text, flags=re.MULTILINE)

    # Blockquotes -> prefix with "| ".
    text = re.sub(r"^>\s?", "| ", text, flags=re.MULTILINE)

    # Bold / italic / strikethrough (must come after list-marker handling).
    # Double emphasis first.
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"~~([^~]+)~~", r"\1", text)
    # Single emphasis; avoid list items starting with "* " by requiring the
    # closing marker not to be followed by whitespace + end of line.
    text = re.sub(r"(?m)\*([^*\n]+)\*(?!\s*$)", r"\1", text)
    text = re.sub(r"(?m)_([^_\n]+)_(?!\s*$)", r"\1", text)

    # Unordered list markers: normalise to "- ".
    text = re.sub(r"^[\*\+]\s+", "- ", text, flags=re.MULTILINE)

    # Collapse multiple blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _clean_codeblock(block: str) -> str:
    """Drop the opening/closing ``` fences and any language hint."""
    lines = block.splitlines()
    if len(lines) >= 2:
        # Opening fence may contain language, e.g. ```python
        content = lines[1:-1]
    else:
        content = lines
    return "\n".join(content)


def _convert_tables(text: str) -> str:
    """Convert Markdown tables to plain-text rows separated by '|'."""
    lines = text.splitlines()
    output: list[str] = []
    table_lines: list[str] = []

    def flush():
        if not table_lines:
            return
        converted = _render_table(table_lines)
        if converted is not None:
            output.extend(converted)
        else:
            output.extend(table_lines)
        table_lines.clear()

    for line in lines:
        stripped = line.strip()
        if _is_table_row(stripped):
            table_lines.append(stripped)
        else:
            flush()
            output.append(line)
    flush()
    return "\n".join(output)


def _is_table_row(line: str) -> bool:
    return bool(line.startswith("|") and line.endswith("|") and line.count("|") >= 2)


def _render_table(lines: list[str]) -> list[str] | None:
    rows = [_parse_table_row(line) for line in lines]
    if not rows or len(rows) < 2:
        return None
    # A valid Markdown table must contain a separator row of dashes/colons.
    sep_index = next(
        (
            i
            for i, row in enumerate(rows)
            if all(re.fullmatch(r"[\s\-:]+", cell) for cell in row)
        ),
        None,
    )
    if sep_index is None:
        return None

    rendered: list[str] = []
    for i, row in enumerate(rows):
        if i == sep_index:
            rendered.append(" | ".join("---" for _ in row))
        else:
            rendered.append(" | ".join(_clean_cell(cell) for cell in row))
    return rendered


def _parse_table_row(line: str) -> list[str]:
    # Strip outer pipes, split on unescaped pipes.
    inner = line[1:-1]
    return [cell.strip() for cell in inner.split("|")]


def _clean_cell(text: str) -> str:
    """Remove inline emphasis/code from a table cell."""
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"~~([^~]+)~~", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"[图片: \2]", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    return text.strip()
