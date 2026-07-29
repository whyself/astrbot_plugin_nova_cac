from nju_qa.chunking import split_markdown


def _make_doc(body, doc_id="42", title="测试"):
    return split_markdown(
        doc_id,
        body,
        title=title,
        repository="新生手册",
        namespace="nju/guide",
        slug="test",
        file_path="test.md",
        source_url="https://example.test/doc",
        updated_at="2026-01-01",
        size=300,
        overlap=60,
    )


def test_long_markdown_creates_stable_overlapping_chunks():
    body = (
        "# 前言\n\n"
        + "无关内容。" * 120
        + "\n\n## 常用平台\n\n南京大学官网、网上办事服务大厅、统一身份认证与教务系统。"
        + "补充说明。" * 120
    )
    first = _make_doc(body)
    second = _make_doc(body)
    assert len(first) > 2 and [c.chunk_id for c in first] == [
        c.chunk_id for c in second
    ]
    assert any("统一身份认证" in c.content for c in first)


def test_empty_body_returns_no_chunks():
    assert split_markdown("1", "   \n\n  ") == []
    assert split_markdown("1", "") == []


def test_frontmatter_and_meta_table_stripped():
    body = "---\ntitle: t\n---\n\n|x|y|\n|---|---|\n|a|b|\n\n正文开始。\n\n第二段。"
    chunks = _make_doc(body, title="t")
    assert chunks and "正文开始" in chunks[0].content
    assert "|x|" not in chunks[0].content


def test_invalid_settings_raise():
    import pytest

    with pytest.raises(ValueError):
        split_markdown("1", "x", size=100)
    with pytest.raises(ValueError):
        split_markdown("1", "x", overlap=200, size=300)


def test_chunk_metadata_present():
    chunks = _make_doc("# 标题\n\n正文。", doc_id="99", title="标题")
    assert chunks
    c = chunks[0]
    assert c.document_id == "99"
    assert c.title == "标题"
    assert c.namespace == "nju/guide"
    assert c.source_url == "https://example.test/doc"


def test_image_only_chunks_are_dropped():
    body = "# 标题\n\n" + "![](https://example.com/a.png)" * 5
    chunks = _make_doc(body)
    assert all("![" not in c.content for c in chunks)


def test_html_tags_and_images_are_cleaned():
    body = (
        '# 标题\n\n'
        '![](https://example.com/a.png)\n\n'
        '<font style="color:red">正文</font> [链接](https://nju.edu.cn) 结束。'
    )
    chunks = _make_doc(body)
    assert chunks
    text = chunks[0].content
    assert "![" not in text
    assert "<font" not in text
    assert "https://nju.edu.cn" in text
    assert "链接" in text
    body = "# A\n\n" + "内容。" * 50
    a = _make_doc(body, doc_id="7")
    b = _make_doc(body, doc_id="7")
    assert {c.chunk_id for c in a} == {c.chunk_id for c in b}
    assert len(a) == len(b)


def test_sliding_window_does_not_loop():
    body = "A" * 5000
    chunks = _make_doc(body)
    assert len(chunks) > 3
    assert sum(len(c.content) for c in chunks) >= 5000
    assert all(c.content for c in chunks)
