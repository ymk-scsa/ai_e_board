"""
Unit tests for MaterialAnalyzer (HTML & JSON parsing, metrics calculation, attention flags).
"""

import pytest
from models.schemas import Lesson, Section, Formula, ExampleProblem
from system_b.analyzer import MaterialAnalyzer, ParsedLessonMaterial


@pytest.fixture
def sample_lesson():
    return Lesson(
        subject="数学",
        grade="高校1年",
        unit="2次関数",
        lesson_title="平方完成",
        learning_objectives=["平方完成の手順を理解する"],
        sections=[
            Section(
                section_id="sec_1",
                section_type="formula",
                title="基本公式",
                content="標準形 y = a(x-p)^2 + q の解説",
                formulas=[Formula(raw_text="y=a(x-p)^2+q", latex="y = a(x-p)^2 + q", description="標準形")],
                order=1,
            ),
            Section(
                section_id="sec_2",
                section_type="example",
                title="例題1",
                content="y = 2x^2 - 4x + 5 を変形する",
                example=ExampleProblem(
                    title="例題1",
                    problem="y = 2x^2 - 4x + 5 を平方完成せよ",
                    solution_steps=["y = 2(x^2 - 2x) + 5", "y = 2(x-1)^2 + 3"],
                    answer="頂点 (1, 3)",
                ),
                order=2,
            ),
        ],
        summary="3ステップで変形する",
    )


def test_parse_from_lesson(sample_lesson):
    material = MaterialAnalyzer.parse_from_lesson(sample_lesson)
    assert material.lesson_title == "平方完成"
    assert len(material.slides) == 4  # Objective + 2 Sections + Summary
    assert material.slides[0].badge == "本時の目標"
    assert material.slides[1].title == "基本公式"


def test_parse_from_html():
    sample_html = """
    <!DOCTYPE html>
    <html>
    <head><title>順列の考え方 - AI電子黒板教材</title></head>
    <body class="theme-chalkboard">
        <div class="board-header">
            <div class="lesson-meta"><span>数学</span><span>•</span><span>順列</span></div>
        </div>
        <div class="slides-wrapper">
            <div class="slide active">
                <div class="slide-title"><span class="slide-badge">本時の目標</span>順列の考え方</div>
                <div class="slide-body">
                    <div class="objective-item">順列の計算ができる</div>
                </div>
            </div>
            <div class="slide">
                <div class="slide-title"><span class="slide-badge">公式</span>順列の公式</div>
                <div class="slide-body">
                    <p>公式は $${}_{n}P_{r} = n(n-1)...(n-r+1)$$ です。</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    material = MaterialAnalyzer.parse_from_html(sample_html)
    assert material.lesson_title == "順列の考え方"
    assert material.unit == "順列"
    assert len(material.slides) == 2
    assert "{}_{n}P_{r}" in material.slides[1].formulas[0]


def test_compute_slide_metrics_and_flags():
    material = MaterialAnalyzer.parse_from_html("""
    <div class="slide">
        <div class="slide-title">例題と練習</div>
        <div class="slide-body">
            <p>ここに非常に長い文章が入ります。高校数学の例題解説として、生徒に考え方を提示します。${}_{4}P_{3} = 24$ です。さらに別の公式 $n! = n(n-1)...1$ もあります。また $4 \\times 3 = 12$ も使います。さらに計算ステップを進めます。問題文と解答が混在しています。</p>
        </div>
    </div>
    """)
    metrics = MaterialAnalyzer.analyze_material(material)
    assert len(metrics) == 1
    assert metrics[0].formula_count == 3
    assert len(metrics[0].attention_flags) >= 1
