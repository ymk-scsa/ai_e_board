"""
Unit tests for Pydantic data schemas in models/schemas.py.
"""

import pytest
from pydantic import ValidationError

from models.schemas import (
    Lesson,
    Section,
    Formula,
    ExampleProblem,
    Exercise,
    VisualAnnotation,
    Slide,
    ElectronicBoardPresentation,
)


def test_formula_creation():
    f = Formula(
        raw_text="4P3 = 24",
        latex="{}_{4}P_{3} = 24",
        description="順列の計算例",
        is_key_formula=True,
    )
    assert f.raw_text == "4P3 = 24"
    assert f.latex == "{}_{4}P_{3} = 24"
    assert f.is_key_formula is True


def test_example_problem_creation():
    ex = ExampleProblem(
        title="例題1",
        problem="4曲から3曲選ぶ曲順は何通りか",
        approach="1曲目から順に選択肢を掛ける",
        solution_steps=["1曲目: 4通り", "2曲目: 3通り", "3曲目: 2通り", "4*3*2 = 24"],
        answer="24通り",
    )
    assert ex.answer == "24通り"
    assert len(ex.solution_steps) == 4


def test_lesson_full_validation():
    lesson_data = {
        "subject": "数学",
        "grade": "高校1年",
        "unit": "場合の数と確率",
        "lesson_title": "順列の考え方",
        "learning_objectives": ["順列の意味を理解する", "nPrを計算できる"],
        "introduction": "曲の演奏順を考える",
        "sections": [
            {
                "section_id": "sec_01",
                "section_type": "formula",
                "title": "公式 nPr",
                "content": "順列の計算公式",
                "formulas": [
                    {
                        "raw_text": "nPr = n(n-1)...(n-r+1)",
                        "latex": "{}_{n}P_{r} = n(n-1)\\cdots(n-r+1)",
                        "description": "一般式",
                        "is_key_formula": True,
                    }
                ],
                "order": 1,
            }
        ],
        "summary": "順列は並べる順序を考慮する",
    }

    lesson = Lesson.model_validate(lesson_data)
    assert lesson.subject == "数学"
    assert lesson.lesson_title == "順列の考え方"
    assert len(lesson.sections) == 1
    assert lesson.sections[0].formulas[0].latex == "{}_{n}P_{r} = n(n-1)\\cdots(n-r+1)"


def test_lesson_missing_required_fields():
    # 'unit' and 'lesson_title' are required
    with pytest.raises(ValidationError):
        Lesson.model_validate({"subject": "数学"})
