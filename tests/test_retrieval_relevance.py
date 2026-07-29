"""Retrieval relevance, query intent, and evidence classification tests."""

from __future__ import annotations

from pathlib import Path

from nju_qa.evidence import (
    QueryIntentTerms,
    classify_qa_window,
    detect_competing_object,
    evaluate_grep_reliability,
    is_historical_document,
    parse_query_terms,
    score_grep_hit,
    subject_action_relevance,
)


def _doc_hit(
    title: str,
    yuque_id: str,
    matched_keywords: list[str],
    snippet: str,
    path: str = "doc.md",
) -> dict:
    return {
        "yuque_id": yuque_id,
        "title": title,
        "repository": "guide",
        "namespace": "nju/guide",
        "slug": Path(path).stem,
        "url": f"https://yuque.test/nju/guide/{Path(path).stem}",
        "created_at": "a",
        "updated_at": "b",
        "path": path,
        "matched_keywords": matched_keywords,
        "matches": [{"line_start": 1, "line_end": 1, "snippet": snippet}],
    }


# ---------------------------------------------------------------------------
# Query intent parsing
# ---------------------------------------------------------------------------


def test_parse_query_terms_separates_subject_action_modifier():
    intent = parse_query_terms(["校园卡", "补办", "地点"])
    assert isinstance(intent, QueryIntentTerms)
    assert intent.subject_terms == ("校园卡",)
    assert intent.action_terms == ("补办",)
    assert intent.modifier_terms == ("地点",)


def test_action_synonyms_are_expanded():
    intent = parse_query_terms(["校园卡", "补办"])
    assert "补卡" in intent.action_expanded
    assert "重办" in intent.action_expanded


def test_unknown_term_becomes_subject():
    intent = parse_query_terms(["校园卡", "照片", "上传"])
    assert intent.subject_terms == ("校园卡", "照片")
    assert intent.action_terms == ("上传",)


# ---------------------------------------------------------------------------
# Subject-action binding
# ---------------------------------------------------------------------------


def test_subject_action_relevance_direct_binding_is_high():
    score = subject_action_relevance(
        "校园卡的挂失与补办可在信息化建设管理服务中心办理。",
        subject_terms=["校园卡"],
        action_terms=["补办", "补卡", "重办"],
    )
    assert score >= 0.8


def test_subject_action_relevance_competing_object_is_low():
    score = subject_action_relevance(
        "学生证补办流程：办理时需携带校园卡。办理地点为综合服务大厅。",
        subject_terms=["校园卡"],
        action_terms=["补办", "补卡", "重办"],
    )
    assert score < 0.5


def test_detect_competing_object_recognises_wrong_object():
    assert detect_competing_object(
        "学生证补办需要携带校园卡。",
        subject_terms=["校园卡"],
        action_terms=["补办"],
    )
    assert not detect_competing_object(
        "校园卡可在中心补办。",
        subject_terms=["校园卡"],
        action_terms=["补办"],
    )


# ---------------------------------------------------------------------------
# Grep ranking
# ---------------------------------------------------------------------------


def test_direct_campus_card_doc_outranks_student_id_doc():
    terms = ["校园卡", "补办", "地点"]
    direct = _doc_hit(
        "常用补办说明",
        "common",
        ["校园卡", "补办", "地点"],
        "信息化建设管理服务中心位于仙林校区。校园卡可在该中心自助补卡。",
        "common.md",
    )
    wrong = _doc_hit(
        "学生证补办和火车优惠卡充磁流程",
        "student",
        ["校园卡", "补办", "地点"],
        "学生证补办流程：办理时需携带校园卡。办理地点为综合服务大厅。",
        "student.md",
    )
    index_hit = _doc_hit(
        "00_index",
        "idx",
        ["校园卡", "补办"],
        "- 校园卡\n- 宿舍\n- 教务",
        "00_index.md",
    )
    direct_score = score_grep_hit(direct, terms)
    wrong_score = score_grep_hit(wrong, terms)
    index_score = score_grep_hit(index_hit, terms)
    assert direct_score > wrong_score
    assert wrong_score > index_score
    assert evaluate_grep_reliability(direct, terms)[0] is True
    assert evaluate_grep_reliability(wrong, terms)[0] is False


def test_modifier_not_required_for_core_coverage():
    hit = _doc_hit(
        "常用补办说明",
        "common",
        ["校园卡", "补办"],
        "校园卡可在信息化建设管理服务中心补办。",
        "common.md",
    )
    reliable, diag = evaluate_grep_reliability(hit, ["校园卡", "补办", "地点"])
    assert reliable is True
    assert diag["core_coverage"] == 1.0


def test_action_synonym_in_document_is_scored_high():
    terms = ["校园卡", "补办"]
    hit = _doc_hit(
        "常用补办说明",
        "common",
        ["校园卡", "补卡"],
        "校园卡自助补卡设备放置在一楼大厅。",
        "common.md",
    )
    score = score_grep_hit(hit, terms)
    assert score >= 3.0
    assert evaluate_grep_reliability(hit, terms)[0] is True


def test_competing_object_docs_score_lower_than_direct():
    terms = ["校园卡", "补办"]
    direct = _doc_hit(
        "常用补办说明",
        "common",
        ["校园卡", "补办"],
        "校园卡可在中心补办。",
        "common.md",
    )
    competing_titles = [
        "学生证补办需校园卡",
        "团员证补办地点",
        "宿舍钥匙补办需要校园卡",
        "银行卡补办后重新绑定校园卡",
    ]
    direct_score = score_grep_hit(direct, terms)
    for title in competing_titles:
        hit = _doc_hit(
            title,
            title,
            ["校园卡", "补办"],
            title,
            f"{title}.md",
        )
        assert score_grep_hit(hit, terms) < direct_score, title


# ---------------------------------------------------------------------------
# QA status classification
# ---------------------------------------------------------------------------


def test_qa_no_answer_markers_take_priority():
    snippet = "回答（resolved）：暂无可用答案（原答案已在最终筛选中剔除）"
    reliable, diag = evaluate_grep_reliability(
        _doc_hit("QA", "q1", ["校园卡", "补办"], snippet),
        ["校园卡", "补办"],
    )
    assert reliable is False
    assert diag["qa_status"] == "no_answer"


def test_qa_no_answer_found_class_is_unreliable():
    snippet = "回答（no_answer_found）：暂无可靠答案，待补充。"
    reliable, diag = evaluate_grep_reliability(
        _doc_hit("QA", "q2", ["校园卡", "过期"], snippet),
        ["校园卡", "过期"],
    )
    assert reliable is False
    assert diag["qa_status"] == "no_answer"


def test_partially_resolved_is_not_treated_as_fully_resolved():
    status = classify_qa_window("回答（partially_resolved）：需要根据具体情况确认")
    assert status.value == "partial"
    # Make sure it is not mistaken for plain resolved.
    assert "resolved" not in ("reliable",)


def test_adjacent_qa_statuses_do_not_bleed():
    snippet = (
        "### Q31 校园卡怎么充值？\n"
        "回答（resolved）：可在线充值。\n"
        "### Q32 校园卡过期是什么意思？\n"
        "回答（no_answer_found）：暂无可靠答案，待补充。\n"
        "### Q33 校园卡丢了怎么办？\n"
        "回答（resolved）：可挂失补办。"
    )
    reliable, diag = evaluate_grep_reliability(
        _doc_hit("常见疑问汇总", "faq", ["校园卡", "过期"], snippet),
        ["校园卡", "过期"],
    )
    assert reliable is False
    assert diag["qa_status"] == "no_answer"


# ---------------------------------------------------------------------------
# Historical / archive penalty
# ---------------------------------------------------------------------------


def test_historical_title_is_detected_and_penalised():
    historical = _doc_hit(
        "★新生校园卡照片采集（2024.07.25-2024.08.05）",
        "hist",
        ["校园卡", "照片"],
        "请在 2024 年 7 月 25 日至 8 月 5 日上传照片。",
        "archive/hist.md",
    )
    current = _doc_hit(
        "校园卡照片采集指南",
        "curr",
        ["校园卡", "照片"],
        "新生入学后需要上传校园卡照片，请关注本年度通知。",
        "guide/photo.md",
    )
    terms = ["校园卡", "照片"]
    assert score_grep_hit(current, terms) > score_grep_hit(historical, terms)
    assert is_historical_document(historical["title"], historical["path"])[0] is True
    assert is_historical_document(current["title"], current["path"])[0] is False


def test_archive_path_is_historical():
    assert is_historical_document("旧文档", "归档/old.md")[0] is True
