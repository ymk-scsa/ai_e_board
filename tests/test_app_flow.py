"""
Streamlit AppTest for the STEP3 → STEP4 → STEP5 flow (M6: data preservation and stale-state handling).
No LLM is called: lessons are injected into session state. Outputs go to tmp_path.
"""

from pathlib import Path

import pytest

import config

APP = str(Path(__file__).resolve().parent.parent / "app.py")
pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402


@pytest.fixture
def isolated(monkeypatch):
    """Outputs are redirected by tests/conftest.py; here backend discovery is kept offline-safe
    (an unreachable Ollama just shows a sidebar error)."""
    monkeypatch.setattr(config, "LLM_BACKEND", "ollama")
    monkeypatch.setattr(config, "OLLAMA_HOST", "http://127.0.0.1:9")
    return config.OUTPUT_DIR


def _mock(kind):
    import app
    return app.get_mock_lesson_for_sample(kind)


def _start(step, lesson):
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["llm_backend_kind"] = "ollama"
    at.session_state["ollama_host"] = "http://127.0.0.1:9"
    at.run()
    at.session_state["parsed_lesson"] = lesson
    at.session_state["current_step"] = step
    return at.run()


def _text_input(at, label):
    return next(w for w in at.text_input if w.label == label)


def _submit(at):
    btn = next(b for b in at.button if "編集を確定" in b.label)
    btn.click()
    return at.run()


def test_step3_shows_new_lesson_after_switch(isolated):
    at = _start(3, _mock("quadratic"))
    assert not at.exception
    assert _text_input(at, "授業タイトル").value == "2次関数の平方完成とグラフの頂点"
    title_inputs = [w.value for w in at.text_input if w.label == "見出し"]

    at.session_state["parsed_lesson"] = _mock("permutation")
    at.run()
    assert _text_input(at, "授業タイトル").value == "順列の考え方と計算公式 nPr"
    # section headings follow the new lesson, not the old widget state
    assert [w.value for w in at.text_input if w.label == "見出し"] != title_inputs
    assert [w.value for w in at.text_input if w.label == "見出し"][0] == "導入課題：曲の演奏順"


def test_step3_submit_preserves_example_and_exercise_details(isolated):
    lesson = _mock("quadratic")
    lesson.sections[1].example.formulas = [lesson.sections[1].formulas[0]]
    at = _start(3, lesson)
    at = _submit(at)
    assert not at.exception
    assert at.session_state["current_step"] == 4
    saved = at.session_state["parsed_lesson"]
    ex = saved.sections[1].example
    assert ex.teaching_notes and "符号ミス" in ex.teaching_notes
    assert ex.formulas and ex.formulas[0].latex == "y = 2(x - 1)^2 + 3"
    exe = saved.sections[2].exercise
    assert len(exe.solution_steps) == 2


def test_custom_section_type_is_kept(isolated):
    lesson = _mock("permutation")
    lesson.sections[0].section_type = "custom"
    at = _submit(_start(3, lesson))
    assert at.session_state["parsed_lesson"].sections[0].section_type == "custom"


def test_editing_lesson_invalidates_generated_slides(isolated):
    at = _start(4, _mock("permutation"))
    next(b for b in at.button if "電子黒板教材を生成する" in b.label).click()
    at.run()
    assert at.session_state["current_step"] == 5
    assert at.session_state["presentation_html"]
    assert list(isolated.glob("lesson_*.html"))  # written to tmp, not the real outputs/

    # Back to STEP3, change the title, confirm → old slides must be discarded
    at.session_state["current_step"] = 3
    at.run()
    _text_input(at, "授業タイトル").input("順列（改訂版）")
    at = _submit(at)
    assert at.session_state["parsed_lesson"].lesson_title == "順列（改訂版）"
    assert at.session_state["presentation_html"] is None
    assert at.session_state["generated_presentation"] is None


def test_system_b_preset_evaluation_renders(isolated):
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["llm_backend_kind"] = "ollama"
    at.session_state["ollama_host"] = "http://127.0.0.1:9"
    at.session_state["app_mode"] = "📊 教育品質評価・運用支援 (システムB)"
    at.run()
    at.radio[0].set_value("④ 研究用プリセット教材").run()
    next(b for b in at.button if "順列" in b.label).click()
    at.run()
    next(c for c in at.checkbox if "LLM" in c.label).uncheck()
    next(b for b in at.button if "教育品質評価を実行する" in b.label).click()
    at.run()
    assert not at.exception
    result = at.session_state["eval_result"]
    assert result is not None and result.overall_score >= 85
    assert list((isolated / "evaluations").glob("eval_*.html"))


def test_unchanged_confirm_keeps_generated_slides(isolated):
    at = _start(4, _mock("permutation"))
    next(b for b in at.button if "電子黒板教材を生成する" in b.label).click()
    at.run()
    html_before = at.session_state["presentation_html"]
    at.session_state["current_step"] = 3
    at.run()
    at = _submit(at)
    assert at.session_state["presentation_html"] == html_before
