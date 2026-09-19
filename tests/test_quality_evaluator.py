"""
Unit tests for QualityEvaluator and EvaluationPipeline (fallback simulation, unified scoring).
"""

import json
import pytest
from models.schemas import Lesson, Section, Formula, ExampleProblem
from models.evaluation_schemas import EvaluationResult
from system_b.pipeline import EvaluationPipeline


@pytest.fixture
def sample_lesson():
    return Lesson(
        subject="数学",
        grade="高校1年",
        unit="2次関数",
        lesson_title="平方完成と頂点",
        learning_objectives=["平方完成の変形手順を習得する"],
        introduction="グラフの頂点を求めるための式変形を学ぶ",
        sections=[
            Section(
                section_id="sec_1",
                section_type="formula",
                title="基本形",
                content="y = a(x-p)^2 + q の頂点は (p, q)",
                formulas=[Formula(raw_text="y=a(x-p)^2+q", latex="y = a(x-p)^2 + q", description="基本形")],
                order=1,
            ),
            Section(
                section_id="sec_2",
                section_type="example",
                title="例題1",
                content="y = 2x^2 - 4x + 5 を平方完成する",
                example=ExampleProblem(
                    title="例題1",
                    problem="y = 2x^2 - 4x + 5 を平方完成せよ",
                    solution_steps=["y = 2(x^2 - 2x) + 5", "y = 2(x-1)^2 + 3"],
                    answer="頂点 (1, 3), 軸 x = 1",
                ),
                order=2,
            ),
        ],
        summary="3ステップで確実に変形する",
    )


def test_pipeline_evaluate_from_lesson(sample_lesson):
    pipeline = EvaluationPipeline()
    result, json_p, html_p = pipeline.evaluate_from_lesson(sample_lesson, use_llm=False)

    assert isinstance(result, EvaluationResult)
    assert 0 <= result.overall_score <= 100
    assert result.lesson_title == "平方完成と頂点"
    assert len(result.slide_metrics) >= 3
    assert json_p.exists()
    assert html_p.exists()


def test_pipeline_evaluate_from_json_string(sample_lesson):
    pipeline = EvaluationPipeline()
    json_str = json.dumps(sample_lesson.model_dump())
    result, json_p, html_p = pipeline.evaluate_from_json_string(json_str, use_llm=False)

    assert result.lesson_title == "平方完成と頂点"
    assert result.overall_score >= 60


def test_pipeline_evaluate_from_html():
    pipeline = EvaluationPipeline()
    html_sample = """
    <!DOCTYPE html>
    <html>
    <head><title>順列 - AI電子黒板</title></head>
    <body>
        <div class="slides-wrapper">
            <div class="slide"><div class="slide-title">順列</div><div class="slide-body">解説</div></div>
        </div>
    </body>
    </html>
    """
    result, json_p, html_p = pipeline.evaluate_from_html_content(html_sample, use_llm=False)
    assert result.lesson_title == "順列"
    assert result.input_source_type == "blackboard_html"


def test_pipeline_evaluate_invalid_json_raises_error():
    pipeline = EvaluationPipeline()
    with pytest.raises(Exception):
        pipeline.evaluate_from_json_string("INVALID JSON {broken...", use_llm=False)

