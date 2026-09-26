"""
Security tests: upload path safety, Ollama host allowlist (SSRF), HTML escaping, and no silent mock fallback.
"""

import json
from pathlib import Path

import pytest

import config
from ai.security import safe_upload_path, validate_ollama_host
from ai.generator import ElectronicBoardGenerator
from models.schemas import Lesson, Section, Formula, ExampleProblem, Exercise
from system_b.analyzer import MaterialAnalyzer
from system_b.quality_evaluator import QualityEvaluator
from system_b.report import ReportGenerator


XSS = "<script>alert(1)</script>"
IMG_XSS = '<img src=x onerror="alert(1)">'


# ---------------------------------------------------------------------------
# safe_upload_path
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name",
    [
        "../../x.png",
        "..\\..\\x.png",
        "C:/evil.png",
        "C:\\Windows\\evil.png",
        "/etc/passwd.png",
        "/abs/path/board.PNG",
        "normal.jpg",
    ],
)
def test_safe_upload_path_stays_inside_base(tmp_path, name):
    base = tmp_path / "uploads"
    p = safe_upload_path(name, base, session_id="abc123")
    assert base.resolve() in p.parents
    assert p.parent == (base / "abc123").resolve()
    assert p.parent.exists()
    # Name is a generated uuid hex + lowercase suffix, never the original name
    assert len(p.stem) == 32 and all(c in "0123456789abcdef" for c in p.stem)
    assert p.suffix in (".png", ".jpg")
    assert ".." not in p.name


def test_safe_upload_path_unique_names(tmp_path):
    names = {safe_upload_path("board.png", tmp_path).name for _ in range(20)}
    assert len(names) == 20


@pytest.mark.parametrize("name", ["evil.exe", "noext", "x.png.html", "../../x"])
def test_safe_upload_path_rejects_bad_extension(tmp_path, name):
    with pytest.raises(ValueError):
        safe_upload_path(name, tmp_path)


@pytest.mark.parametrize("sid", ["../x", "a/b", "..", "a\\b"])
def test_safe_upload_path_rejects_bad_session_id(tmp_path, sid):
    with pytest.raises(ValueError):
        safe_upload_path("x.png", tmp_path, session_id=sid)


# ---------------------------------------------------------------------------
# validate_ollama_host
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "host,expected",
    [
        ("http://localhost:11434", "http://localhost:11434"),
        ("localhost:11434", "http://localhost:11434"),
        ("http://127.0.0.1:11434", "http://127.0.0.1:11434"),
        ("127.0.0.1:11434", "http://127.0.0.1:11434"),
        ("https://localhost", "https://localhost"),
        ("http://[::1]:11434", "http://[::1]:11434"),
        ("  http://LOCALHOST:11434  ", "http://localhost:11434"),
    ],
)
def test_validate_ollama_host_accepts_local(host, expected):
    assert validate_ollama_host(host) == expected


@pytest.mark.parametrize(
    "host",
    [
        "http://169.254.169.254",
        "http://169.254.169.254/latest/meta-data/",
        "http://example.com",
        "example.com:11434",
        "file:///etc/passwd",
        "ftp://localhost",
        "gopher://127.0.0.1:11434",
        "http://user:pass@localhost:11434",
        "http://localhost.evil.com:11434",
        "http://localhost:notaport",
        "",
    ],
)
def test_validate_ollama_host_rejects(host):
    with pytest.raises(ValueError):
        validate_ollama_host(host)


def test_validate_ollama_host_env_allowlist(monkeypatch):
    monkeypatch.setattr(config, "OLLAMA_ALLOWED_HOSTS", ["gpu-box"])
    assert validate_ollama_host("http://gpu-box:11434") == "http://gpu-box:11434"
    # Default OLLAMA_HOST hostname (localhost) always allowed
    assert validate_ollama_host("http://localhost:11434") == "http://localhost:11434"
    with pytest.raises(ValueError):
        validate_ollama_host("http://127.0.0.2:11434")


def test_vision_analyzer_rejects_bad_host_without_crash():
    from ai.vision import VisionAnalyzer

    va = VisionAnalyzer(host="http://169.254.169.254", backend_kind="ollama")
    assert va.backend is None
    assert va.host_error
    ok, msg, models = va.check_connection()
    assert ok is False and models == []


def test_quality_evaluator_rejects_bad_host_without_crash(tmp_path):
    qe = QualityEvaluator(host="http://example.com")
    assert qe.client is None and qe.host_error
    material = MaterialAnalyzer.parse_from_lesson(_xss_lesson())
    # LLM enrichment fails safely and falls back to deterministic evaluation
    result = qe.evaluate_material(material, use_llm=True)
    assert result.overall_score >= 0


# ---------------------------------------------------------------------------
# HTML escaping
# ---------------------------------------------------------------------------
def _xss_lesson() -> Lesson:
    return Lesson(
        subject="数学",
        grade="高校1年",
        unit=f"単元{XSS}",
        lesson_title=f"タイトル{XSS}",
        learning_objectives=[f"目標{IMG_XSS}"],
        introduction=f"導入{XSS}",
        sections=[
            Section(
                section_id="sec_01",
                section_type="formula",
                title=f"見出し{XSS}",
                content=f"本文 $a<b$ {IMG_XSS}",
                formulas=[Formula(raw_text="a<b", latex="a<b", description=f"説明{XSS}", is_key_formula=True)],
                order=1,
            ),
            Section(
                section_id="sec_02",
                section_type="example",
                title="例題",
                content="例題の本文",
                example=ExampleProblem(
                    title=f"例題{XSS}",
                    problem=f"問題{IMG_XSS}",
                    approach=f"考え方{XSS}",
                    solution_steps=[f"ステップ{XSS}"],
                    answer=f"答え{XSS}",
                ),
                exercise=Exercise(title=f"練習{XSS}", problem=f"練習問題{XSS}", hint=f"ヒント{XSS}", answer=f"解答{XSS}"),
                order=2,
            ),
        ],
        summary=f"まとめ{XSS}",
        notes_for_teacher=f"メモ{IMG_XSS}",
    )


def test_generator_html_escapes_user_content():
    gen = ElectronicBoardGenerator()
    pres = gen.build_presentation(_xss_lesson())
    html_out = gen.render_presentation_html(pres)

    assert XSS not in html_out
    assert "<img src=x" not in html_out
    assert "onerror=\"alert" not in html_out
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_out
    # Math with '<' survives as escaped text (KaTeX reads textContent)
    assert "$a&lt;b$" in html_out
    assert "$$a&lt;b$$" in html_out
    # No double escaping
    assert "&amp;lt;" not in html_out
    # Template's own script remains intact
    assert "function nextSlide()" in html_out


def test_analyzer_parses_escaped_html_back_to_text():
    gen = ElectronicBoardGenerator()
    pres = gen.build_presentation(_xss_lesson())
    html_out = gen.render_presentation_html(pres)
    material = MaterialAnalyzer.parse_from_html(html_out)

    assert material.lesson_title.startswith("タイトル<script>")
    assert material.unit == f"単元{XSS}"
    assert any("$a<b$" in s.body_text or "a<b" in s.body_text for s in material.slides)
    assert any("a<b" in f for s in material.slides for f in s.formulas)


def test_report_html_escapes_llm_and_lesson_text():
    material = MaterialAnalyzer.parse_from_lesson(_xss_lesson())
    result = QualityEvaluator().evaluate_material(material, use_llm=False)
    result.executive_summary = f"LLM要約{XSS}"
    result.strengths = [f"強み{IMG_XSS}"]
    html_out = ReportGenerator().render_html_report(result)

    assert XSS not in html_out
    assert "<img src=x" not in html_out
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_out
    assert "&amp;lt;" not in html_out


# ---------------------------------------------------------------------------
# Mock lesson is never silent
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(config, "RESEARCH_LOG_PATH", tmp_path / "research_log.jsonl")
    return tmp_path


@pytest.mark.parametrize("is_mock,expected_model", [(True, "mock"), (False, "qwen3-vl:8b")])
def test_save_outputs_records_is_mock(tmp_outputs, is_mock, expected_model):
    gen = ElectronicBoardGenerator()
    lesson = _xss_lesson()
    pres = gen.build_presentation(lesson)
    json_p, html_p = gen.save_outputs(
        lesson=lesson,
        presentation=pres,
        html_content=gen.render_presentation_html(pres),
        research_metadata={"model": "qwen3-vl:8b", "source_images": ["a.png"]},
        is_mock=is_mock,
    )
    assert json_p.parent == tmp_outputs
    entry = json.loads((tmp_outputs / "research_log.jsonl").read_text(encoding="utf-8").strip().splitlines()[-1])
    assert entry["is_mock"] is is_mock
    assert entry["model"] == expected_model
    assert json.loads(json_p.read_text(encoding="utf-8"))["is_mock"] is is_mock


class _FakeAnalyzer:
    backend_name = "fake"

    def __init__(self, result):
        self._result = result

    def analyze_board_to_lesson(self, **kwargs):
        return self._result


def test_analyze_and_parse_does_not_substitute_mock_on_vision_failure():
    import app
    from ai.board_pipeline import BoardAnalysis

    fake = _FakeAnalyzer(BoardAnalysis(None, "板書の書き起こしに失敗しました: model not found", []))
    lesson, err, raw = app.analyze_and_parse(fake, [Path("x.png")], ["x.png"])
    assert lesson is None
    assert "model not found" in err
    assert raw["is_mock"] is False


def test_analyze_and_parse_does_not_substitute_mock_on_unreadable_board():
    import app
    from ai.board_pipeline import BoardAnalysis

    fake = _FakeAnalyzer(BoardAnalysis(None, "板書から文字を読み取れませんでした。", [], transcript="（空白）"))
    lesson, err, raw = app.analyze_and_parse(fake, [Path("x.png")], ["x.png"])
    assert lesson is None
    assert err
    assert raw["raw_text"] == "（空白）"
