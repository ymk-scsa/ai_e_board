"""
Unit tests for electronic blackboard presentation generation and HTML export.
"""

from pathlib import Path
import pytest
from ai.generator import ElectronicBoardGenerator
from models.schemas import Lesson, Section, Formula, ExampleProblem, Exercise
import config


@pytest.fixture
def sample_lesson():
    return Lesson(
        subject="数学",
        grade="高校1年",
        unit="2次関数",
        lesson_title="平方完成とグラフの頂点",
        learning_objectives=["平方完成の変形手順を理解する", "頂点と軸を求めることができる"],
        introduction="一般形から頂点を求める方法を学ぶ",
        sections=[
            Section(
                section_id="sec_01",
                section_type="formula",
                title="基本形と公式",
                content="標準形 y = a(x-p)^2 + q の頂点は (p, q)",
                formulas=[
                    Formula(
                        raw_text="y = a(x-p)^2 + q",
                        latex="y = a(x-p)^2 + q",
                        description="標準形",
                        is_key_formula=True,
                    )
                ],
                order=1,
            ),
            Section(
                section_id="sec_02",
                section_type="example",
                title="例題1の解法",
                content="2次関数を平方完成する",
                example=ExampleProblem(
                    title="例題1",
                    problem="y = 2x^2 - 4x + 5 を平方完成せよ",
                    approach="2で括って半分の2乗を作る",
                    solution_steps=["y = 2(x^2 - 2x) + 5", "y = 2(x-1)^2 + 3"],
                    answer="頂点 (1, 3), 軸 x = 1",
                ),
                order=2,
            ),
        ],
        summary="3つの変形ステップを覚える",
        notes_for_teacher="符号ミスに注意させる",
    )


def test_generate_slides(sample_lesson):
    generator = ElectronicBoardGenerator(theme="chalkboard")
    slides = generator.generate_slides_from_lesson(sample_lesson)

    # Title slide + 2 section slides + 1 summary slide = 4 slides
    assert len(slides) == 4
    assert slides[0].slide_type == "title"
    assert "平方完成の変形手順を理解する" in slides[0].content_html
    assert slides[1].badge == "公式"
    assert slides[2].badge == "例題"
    assert slides[3].slide_type == "summary"


def test_build_and_render_presentation(sample_lesson):
    generator = ElectronicBoardGenerator(theme="chalkboard")
    pres = generator.build_presentation(sample_lesson)
    html_content = generator.render_presentation_html(pres)

    assert "<!DOCTYPE html>" in html_content
    assert "katex" in html_content
    assert "平方完成とグラフの頂点" in html_content
    assert "theme-chalkboard" in html_content


def test_save_outputs(sample_lesson):
    generator = ElectronicBoardGenerator()
    pres = generator.build_presentation(sample_lesson)
    html_content = generator.render_presentation_html(pres)

    json_path, html_path = generator.save_outputs(
        lesson=sample_lesson,
        presentation=pres,
        html_content=html_content,
        research_metadata={"test": True},
    )

    assert json_path.exists()
    assert html_path.exists()
    assert json_path.suffix == ".json"
    assert html_path.suffix == ".html"
