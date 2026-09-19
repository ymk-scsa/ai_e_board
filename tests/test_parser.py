"""
Unit tests for AI response parsing, algorithmic JSON repair, and validation.
"""

import json
import pytest
from ai.parser import LessonParser
from models.schemas import Lesson


def test_extract_json_from_markdown():
    parser = LessonParser()
    raw = """
Here is the lesson structure you requested:

```json
{
  "subject": "数学",
  "grade": "高校1年",
  "unit": "2次関数",
  "lesson_title": "平方完成",
  "learning_objectives": ["平方完成ができる"],
  "sections": []
}
```

Hope this helps!
"""
    extracted = parser.extract_json_string(raw)
    assert extracted.startswith("{")
    assert extracted.endswith("}")
    parsed = json.loads(extracted)
    assert parsed["lesson_title"] == "平方完成"


def test_repair_trailing_commas():
    parser = LessonParser()
    broken_json = """
    {
      "subject": "数学",
      "grade": "高校",
      "unit": "順列",
      "lesson_title": "順列の総数",
      "learning_objectives": [
        "目標1",
        "目標2",
      ],
      "sections": [],
    }
    """
    repaired_dict = parser.repair_json_string(broken_json)
    assert repaired_dict["unit"] == "順列"
    assert len(repaired_dict["learning_objectives"]) == 2


def test_parse_to_lesson_success():
    parser = LessonParser()
    raw_llm_response = """
```json
{
  "subject": "数学",
  "grade": "高校1年",
  "unit": "場合の数と確率 - 順列",
  "lesson_title": "順列の考え方",
  "learning_objectives": ["順列の意味の理解"],
  "sections": [
    {
      "section_id": "sec_01",
      "section_type": "introduction",
      "title": "導入課題",
      "content": "4曲から3曲選ぶ",
      "formulas": [],
      "order": 1
    }
  ]
}
```
"""
    lesson, err = parser.parse_to_lesson(raw_llm_response, source_images=["test.png"])
    assert err is None
    assert isinstance(lesson, Lesson)
    assert lesson.lesson_title == "順列の考え方"
    assert lesson.source_images == ["test.png"]
