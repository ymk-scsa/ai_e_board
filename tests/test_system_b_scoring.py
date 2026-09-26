"""
Regression tests for System B scoring (M7): consistent scores across input routes, no canned praise,
correct density / solution / exercise detection, curriculum matching, and validated LLM feedback.
"""

import json
from pathlib import Path

import pytest

from ai.board_pipeline import build_lesson
from ai.generator import ElectronicBoardGenerator
from ai.llm_client import LLMResult
from models.samples import get_mock_lesson_for_sample
from models.schemas import Lesson, Section
from system_b.analyzer import MaterialAnalyzer, ParsedSlideContent
from system_b.curriculum import describe, match_unit
from system_b.pipeline import EvaluationPipeline
from system_b.quality_evaluator import QualityEvaluator

FIX = Path(__file__).parent / "fixtures"


def _board_lesson(name):
    lesson, _, _ = build_lesson((FIX / f"transcript_{name}.txt").read_text(encoding="utf-8"), "", [name])
    return lesson


LESSONS = {
    "mock_permutation": lambda: get_mock_lesson_for_sample("permutation"),
    "mock_quadratic": lambda: get_mock_lesson_for_sample("quadratic"),
    "board_permutation": lambda: _board_lesson("permutation"),
    "board_quadratic": lambda: _board_lesson("quadratic"),
}


def _scores(result):
    return (result.overall_score, result.category_scores.model_dump(),
            result.usability_evaluation.slide_split_recommended_slides,
            [m.density_level for m in result.slide_metrics])


@pytest.mark.parametrize("name", list(LESSONS))
def test_same_lesson_scores_identically_on_every_input_route(name):
    lesson = LESSONS[name]()
    pipeline = EvaluationPipeline()
    gen = ElectronicBoardGenerator()
    pres = gen.build_presentation(lesson)
    html_export = gen.render_presentation_html(pres, lesson=lesson)
    json_export = json.dumps({"lesson": lesson.model_dump(), "presentation": pres.model_dump()}, ensure_ascii=False)

    via_lesson = pipeline.evaluate_from_lesson(lesson, use_llm=False)[0]
    via_lesson_pres = pipeline.evaluate_from_lesson(lesson, presentation=pres, use_llm=False)[0]
    via_json = pipeline.evaluate_from_json_string(json_export, use_llm=False)[0]
    via_html = pipeline.evaluate_from_html_content(html_export, use_llm=False)[0]
    assert _scores(via_lesson) == _scores(via_lesson_pres) == _scores(via_json) == _scores(via_html)
    assert via_html.lesson_title == lesson.lesson_title


def test_empty_lesson_is_not_praised():
    empty = Lesson(unit="要確認", lesson_title="空の授業", sections=[])
    result = QualityEvaluator().evaluate_material(MaterialAnalyzer.parse_from_lesson(empty), use_llm=False)
    assert result.pedagogical_evaluation.structure_score == 0
    assert result.category_scores.objective_alignment == 40
    assert result.overall_score < 60
    assert "完成度の高い" not in result.executive_summary and "おおむね良好" not in result.executive_summary
    assert not any("目標が明示" in s for s in result.strengths)
    assert any("目標がありません" in p for p in result.points_for_attention)


def test_complete_lesson_scores_high_and_objective_score_varies():
    full = QualityEvaluator().evaluate_material(
        MaterialAnalyzer.parse_from_lesson(get_mock_lesson_for_sample("permutation")), use_llm=False)
    assert full.pedagogical_evaluation.structure_score == 100
    assert full.overall_score >= 85
    lesson = get_mock_lesson_for_sample("permutation")
    lesson.learning_objectives = []
    none = QualityEvaluator().evaluate_material(MaterialAnalyzer.parse_from_lesson(lesson), use_llm=False)
    assert none.category_scores.objective_alignment < full.category_scores.objective_alignment


def test_objective_alignment_recognises_japanese_titles():
    lesson = _board_lesson("quadratic")
    lesson.lesson_title = "平方完成による二次関数のグラフの描画"
    lesson.learning_objectives = ["平方完成の仕組みを理解し、二次関数のグラフをかく。"]
    assert QualityEvaluator.objective_alignment_score(MaterialAnalyzer.parse_from_lesson(lesson)) == 85
    lesson.learning_objectives = ["計算ができる。"]
    assert QualityEvaluator.objective_alignment_score(MaterialAnalyzer.parse_from_lesson(lesson)) == 75


def test_density_levels_and_detection_rules():
    long_text = ParsedSlideContent(slide_number=1, title="長文", body_text="あ" * 2000)
    assert MaterialAnalyzer.compute_slide_metrics(long_text).density_level == "Dense"
    many_formulas = ParsedSlideContent(slide_number=1, title="式", body_text="短い", formulas=["x"] * 8)
    assert MaterialAnalyzer.compute_slide_metrics(many_formulas).density_level == "Dense"
    # "理解" / "解説" are not answers
    prose = ParsedSlideContent(slide_number=1, title="解説", body_text="公式の意味を理解し、解説を聞く。")
    assert MaterialAnalyzer.compute_slide_metrics(prose).has_solution is False
    answered = ParsedSlideContent(slide_number=1, title="例題", body_text="問題文 答: 24通り")
    assert MaterialAnalyzer.compute_slide_metrics(answered).has_solution is True


def test_exercise_detection_uses_slide_type_not_title_words():
    lesson = Lesson(unit="順列", lesson_title="順列", sections=[
        Section(section_id="s1", section_type="concept", title="導入問題の考え方", content="問題を考える")])
    material = MaterialAnalyzer.parse_from_lesson(lesson)
    assert not any(s.has_exercise for s in material.slides)


def test_curriculum_matching():
    assert match_unit("", "") == [] and match_unit("要確認", "") == []
    assert "照合は行っていません" in describe([], "要確認")
    by_topic = match_unit("順列", "順列の総数")
    assert by_topic and by_topic[0].unit == "場合の数と確率" and by_topic[0].matched_by == "topic"
    assert match_unit("2次関数", "")[0].subject == "数学I"
    unknown = match_unit("微分法", "接線の方程式")
    assert unknown == [] and "見つかりませんでした" in describe(unknown, "微分法")


class _FakeLLM:
    name = "fake"

    def __init__(self, text):
        self.text = text

    def generate(self, model, prompt, **kw):
        return LLMResult(text=self.text, backend="fake", model=model, device="gpu", elapsed_seconds=0.1)


def test_llm_feedback_is_validated_and_cannot_change_scores():
    material = MaterialAnalyzer.parse_from_lesson(get_mock_lesson_for_sample("quadratic"))
    base = QualityEvaluator().evaluate_material(material, use_llm=False)
    reply = json.dumps({"overall_score": 3, "executive_summary": "LLMの総評",
                        "strengths": ["良い点", 5, ""], "points_for_attention": ["注意点"]}, ensure_ascii=False)
    qe = QualityEvaluator(llm_backend=_FakeLLM(f"```json\n{reply}\n```"), model="m")
    result = qe.evaluate_material(material, use_llm=True)
    assert result.overall_score == base.overall_score  # rule-based score is kept
    assert result.executive_summary == "LLMの総評"
    assert result.strengths == ["良い点"] and result.points_for_attention == ["注意点"]
    assert result.evaluated_model == "m"


def test_llm_prompt_is_compact_and_long_answers_are_clipped():
    material = MaterialAnalyzer.parse_from_lesson(get_mock_lesson_for_sample("permutation"))
    base = QualityEvaluator().evaluate_material(material, use_llm=False)
    prompt = QualityEvaluator.build_llm_prompt(material, base)
    assert len(prompt) < 2000  # prompt length dominates latency on small GPUs
    assert "S1［本時の目標］" in prompt and "点数は書かない" in prompt
    reply = json.dumps({"executive_summary": "あ" * 1000, "strengths": ["い" * 500] * 9}, ensure_ascii=False)
    result = QualityEvaluator(llm_backend=_FakeLLM(reply), model="m").evaluate_material(material, use_llm=True)
    assert result.evaluated_model == "m"  # a long answer is clipped, not thrown away
    assert len(result.executive_summary) == 300 and len(result.strengths) == 5 and len(result.strengths[0]) == 120


def test_invalid_llm_feedback_falls_back_to_rules():
    material = MaterialAnalyzer.parse_from_lesson(get_mock_lesson_for_sample("quadratic"))
    qe = QualityEvaluator(llm_backend=_FakeLLM('{"strengths": "not a list"}'), model="m")
    result = qe.evaluate_material(material, use_llm=True)
    assert result.evaluated_model == "deterministic_rules"
    assert result.strengths  # rule-based strengths remain


def test_html_scraping_fallback_for_foreign_html():
    html = """<html><head><title>順列 - 基本公式 nPr - AI電子黒板教材</title></head><body>
    <div class="lesson-meta"><span>数学</span><span>•</span><span>場合の数と確率</span></div>
    <div class="slide"><div class="slide-title"><span class="slide-badge">本時の目標</span>順列</div>
      <div class="slide-body"><div class="objective-item"><span class="objective-icon">🎯</span><span>順列を理解する</span></div>
      <p>曲順を考える導入</p></div></div>
    <div class="slide"><div class="slide-title"><span class="slide-badge">まとめ</span>まとめ</div>
      <div class="slide-body"><div class="objective-box"><div>順序を区別するときは順列</div></div>
      <div>📌 指導メモ: 組合せと混同させない</div></div></div></body></html>"""
    material = MaterialAnalyzer.parse_from_html(html)
    assert material.lesson_title == "順列 - 基本公式 nPr"
    assert material.learning_objectives == ["順列を理解する"]
    assert material.introduction == "曲順を考える導入"
    assert material.summary == "順序を区別するときは順列" and "指導メモ" not in material.summary


def test_embedded_lesson_json_is_script_safe():
    lesson = get_mock_lesson_for_sample("permutation")
    lesson.lesson_title = "順列</script><script>alert(1)</script>"
    gen = ElectronicBoardGenerator()
    html_out = gen.render_presentation_html(gen.build_presentation(lesson), lesson=lesson)
    assert "</script><script>alert(1)" not in html_out
    material = MaterialAnalyzer.parse_from_html(html_out)
    assert material.lesson_title == lesson.lesson_title  # round-trips exactly


def test_upload_decoding_accepts_bom():
    from ui.evaluation import EvaluationUI

    class _Upload:
        name = "lesson.json"

        def __init__(self, data):
            self._data = data

        def getvalue(self):
            return self._data

    assert EvaluationUI._decode_upload(_Upload("﻿{}".encode("utf-8"))) == "{}"
