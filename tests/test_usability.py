"""
Unit tests for UsabilityEvaluator (readability, split recommendation, cognitive load).
"""

import pytest
from system_b.analyzer import ParsedLessonMaterial, ParsedSlideContent, MaterialAnalyzer
from system_b.usability import UsabilityEvaluator


def test_usability_split_recommendation():
    # Construct a dense slide containing both problem, long solution, and multiple formulas
    dense_slide = ParsedSlideContent(
        slide_number=2,
        badge="例題と解説",
        title="平方完成の全ステップ",
        body_text="問題文: 2次関数 y=2x^2-4x+5 を平方完成せよ。解法ステップ: 係数2で括ると y=2(x^2-2x)+5 となる。さらにかっこの中で半分の2乗を作ると y=2{(x-1)^2-1}+5 となる。分配法則により y=2(x-1)^2-2+5 となり、最終的に y=2(x-1)^2+3 と求まる。答え: 頂点 (1,3), 軸 x=1。非常に多くの文字数が含まれています。",
        formulas=["y=2(x^2-2x)+5", "y=2{(x-1)^2-1}+5", "y=2(x-1)^2+3"],
        has_example=True,
    )
    material = ParsedLessonMaterial(
        source_type="lesson_json",
        lesson_title="平方完成",
        subject="数学",
        unit="2次関数",
        slides=[
            ParsedSlideContent(slide_number=1, badge="目標", title="目標", body_text="目標"),
            dense_slide,
        ],
    )

    metrics = MaterialAnalyzer.analyze_material(material)
    evaluator = UsabilityEvaluator()
    res = evaluator.evaluate(material, metrics)

    assert res.usability_score >= 60
    assert 2 in res.slide_split_recommended_slides
    assert "分割" in res.layout_balance_analysis
