"""
Unit tests for PedagogyEvaluator (structure evaluation, flow coherence, curriculum check).
"""

import pytest
from system_b.analyzer import MaterialAnalyzer, ParsedLessonMaterial, ParsedSlideContent
from system_b.pedagogy import PedagogyEvaluator


def test_pedagogy_evaluation_complete_flow():
    material = ParsedLessonMaterial(
        source_type="lesson_json",
        lesson_title="順列の総数",
        subject="数学",
        unit="場合の数と確率 - 順列",
        learning_objectives=["順列の計算公式 nPr を理解する"],
        introduction="曲の演奏順を考える",
        summary="順序を区別するときは nPr を使う",
        slides=[
            ParsedSlideContent(slide_number=1, badge="本時の目標", title="目標", body_text="目標"),
            ParsedSlideContent(slide_number=2, badge="導入", title="導入問題", body_text="4曲から3曲選ぶ"),
            ParsedSlideContent(slide_number=3, badge="公式", title="順列の定義", body_text="nPr公式"),
            ParsedSlideContent(slide_number=4, badge="例題", title="例題1", body_text="計算例", has_example=True),
            ParsedSlideContent(slide_number=5, badge="練習", title="問1", body_text="練習問題", has_exercise=True),
            ParsedSlideContent(slide_number=6, badge="まとめ", title="まとめ", body_text="まとめ"),
        ],
    )

    evaluator = PedagogyEvaluator()
    res = evaluator.evaluate(material)

    assert res.structure_score >= 85
    assert res.has_introduction is True
    assert res.has_clear_objectives is True
    assert res.has_example is True
    assert res.has_exercise is True
    assert res.has_summary is True
    assert "対応しています" in res.curriculum_alignment_analysis
