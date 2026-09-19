"""
Unit tests for System B Pydantic data schemas in models/evaluation_schemas.py.
"""

import pytest
from pydantic import ValidationError
from models.evaluation_schemas import (
    SlideMetrics,
    PedagogicalEvaluation,
    UsabilityEvaluation,
    ImprovementSuggestion,
    TeachingGuideItem,
    CategoryScores,
    EvaluationResult,
)


def test_slide_metrics_creation():
    m = SlideMetrics(
        slide_number=1,
        title="本時の目標",
        slide_type="目標",
        char_count=120,
        formula_count=2,
        bullet_count=3,
        has_problem=False,
        has_solution=False,
        density_level="Moderate",
        attention_flags=["文字数参考: 120文字"],
    )
    assert m.slide_number == 1
    assert m.density_level == "Moderate"
    assert len(m.attention_flags) == 1


def test_evaluation_result_validation():
    res = EvaluationResult(
        evaluation_id="eval_test_001",
        evaluated_at="2026-09-02T12:00:00",
        input_source_type="lesson_json",
        lesson_title="順列の計算",
        unit="場合の数と確率",
        subject="数学",
        overall_score=88,
        category_scores=CategoryScores(
            pedagogical_structure=90,
            objective_alignment=92,
            blackboard_usability=85,
            cognitive_load_balance=85,
        ),
        slide_metrics=[
            SlideMetrics(
                slide_number=1,
                title="導入",
                slide_type="intro",
                char_count=90,
                formula_count=1,
            )
        ],
        pedagogical_evaluation=PedagogicalEvaluation(
            structure_score=90,
            has_introduction=True,
            has_clear_objectives=True,
            has_explanation=True,
            has_example=True,
            has_exercise=True,
            has_summary=True,
            flow_coherence_analysis="良好な展開",
            curriculum_alignment_analysis="指導要領適合",
        ),
        usability_evaluation=UsabilityEvaluation(
            usability_score=85,
            readability_analysis="視認性良好",
            layout_balance_analysis="バランス良好",
            math_visibility_analysis="KaTeX表示明瞭",
            slide_split_recommended_slides=[],
        ),
        improvements=[
            ImprovementSuggestion(
                id="imp_1",
                target_slide=1,
                category="Pedagogy",
                issue="導入の問いかけ",
                rationale="動機づけ",
                concrete_action="挙手を促す",
                priority="Low",
            )
        ],
        teaching_guides=[
            TeachingGuideItem(
                slide_number=1,
                timing="導入時",
                key_point_to_emphasize="目標共有",
                questioning_prompt="何通りあるか？",
                student_activity_guide="ノート記入",
            )
        ],
        strengths=["構成の一貫性"],
        points_for_attention=["演習時間の確保"],
        executive_summary="総合的に極めて優れた構成です。",
    )

    assert res.overall_score == 88
    assert res.category_scores.pedagogical_structure == 90
    assert len(res.improvements) == 1
    assert len(res.teaching_guides) == 1


def test_score_range_validation():
    with pytest.raises(ValidationError):
        CategoryScores(
            pedagogical_structure=150,  # Invalid: must be <= 100
            objective_alignment=90,
            blackboard_usability=80,
            cognitive_load_balance=80,
        )
