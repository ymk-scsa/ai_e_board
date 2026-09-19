"""
Unit tests for ImprovementGenerator and ReportGenerator.
"""

import pytest
from system_b.analyzer import ParsedLessonMaterial, ParsedSlideContent, MaterialAnalyzer
from system_b.pedagogy import PedagogyEvaluator
from system_b.usability import UsabilityEvaluator
from system_b.improvement import ImprovementGenerator
from system_b.report import ReportGenerator
from system_b.quality_evaluator import QualityEvaluator


def test_improvement_and_teaching_guide_generation():
    material = ParsedLessonMaterial(
        source_type="lesson_json",
        lesson_title="順列",
        subject="数学",
        unit="場合の数",
        slides=[
            ParsedSlideContent(slide_number=1, badge="本時の目標", title="目標", body_text="目標"),
            ParsedSlideContent(slide_number=2, badge="例題", title="例題1", body_text="例題", has_example=True),
        ],
    )
    metrics = MaterialAnalyzer.analyze_material(material)
    ped_eval = PedagogyEvaluator().evaluate(material)
    usa_eval = UsabilityEvaluator().evaluate(material, metrics)

    suggestions = ImprovementGenerator.generate_suggestions(material, metrics, ped_eval, usa_eval)
    guides = ImprovementGenerator.generate_teaching_guides(material)

    assert len(suggestions) >= 1
    assert len(guides) == 2
    assert guides[0].slide_number == 1
    assert "発問" in guides[0].questioning_prompt or "目標" in guides[0].key_point_to_emphasize


def test_report_generation_html_and_json():
    material = ParsedLessonMaterial(
        source_type="lesson_json",
        lesson_title="順列の計算",
        subject="数学",
        unit="場合の数と確率",
        slides=[
            ParsedSlideContent(slide_number=1, badge="本時の目標", title="目標", body_text="目標"),
        ],
    )
    evaluator = QualityEvaluator()
    result = evaluator.evaluate_material(material, use_llm=False)

    report_gen = ReportGenerator()
    html_content = report_gen.render_html_report(result)

    assert "<!DOCTYPE html>" in html_content
    assert "順列の計算" in html_content
    assert str(result.overall_score) in html_content

    json_p, html_p = report_gen.save_evaluation_outputs(result, {"test_run": True})
    assert json_p.exists()
    assert html_p.exists()
