"""
Tests for ai/llm_client.py (backend abstraction). No real LLM server is contacted:
HTTP streaming is replaced with in-memory fakes.
"""

import io
import json
from contextlib import contextmanager

import pytest
from PIL import Image

import config
from ai import llm_client
from ai.llm_client import (
    BackendStatus,
    FoundryLocalBackend,
    LLMError,
    ModelInfo,
    OllamaBackend,
    OpenAICompatBackend,
    RepetitionGuard,
    prepare_image,
    select_backend,
    strip_thinking,
)


class _FakeResponse:
    def __init__(self, lines, status_code=200):
        self._lines = lines
        self.status_code = status_code

    def iter_lines(self):
        yield from self._lines

    def read(self):
        return b"error body"


def _fake_stream(lines, captured):
    @contextmanager
    def _stream(method, url, json=None, headers=None, timeout=None):
        captured.update({"method": method, "url": url, "json": json})
        yield _FakeResponse(lines)

    return _stream


def _sse(events):
    return [f"data: {json.dumps(e, ensure_ascii=False)}" for e in events] + ["data: [DONE]"]


def _png_bytes(size=(2000, 1000)):
    buf = io.BytesIO()
    Image.new("RGB", size, (10, 80, 40)).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_prepare_image_downscales_and_converts_to_jpeg(tmp_path):
    p = tmp_path / "board.png"
    p.write_bytes(_png_bytes((3000, 2250)))
    out = prepare_image(p, max_side=1280)
    with Image.open(io.BytesIO(out)) as im:
        assert im.format == "JPEG"
        assert max(im.size) == 1280 and im.size == (1280, 960)


def test_prepare_image_accepts_bytes_and_keeps_small_images():
    out = prepare_image(_png_bytes((640, 480)), max_side=1280)
    with Image.open(io.BytesIO(out)) as im:
        assert im.size == (640, 480)


def test_strip_thinking_variants():
    assert strip_thinking("<think>reasoning</think>\n\n答え") == "答え"
    assert strip_thinking("dangling reasoning</think>答え") == "答え"
    assert strip_thinking("答えのみ") == "答えのみ"


def _guard():
    return RepetitionGuard(check_every=1)


def test_repetition_guard_detects_inline_loop():
    assert not _guard().looping("普通の書き起こしテキストです。" * 2)
    assert _guard().looping("計算: " + "$x^2-2px=(x-p)^2-p^2$、" * 30)


def test_repetition_guard_detects_numbered_line_loop():
    """Small models sometimes invent 例7, 例8, ... endlessly: numbers differ, lines are the same shape."""
    lines = [f"例{i}〔5〕人の生徒のうち、3人が（列に）並ぶと、並ぶ方の総数は何通りか。\n〔5〕P〔3〕=〔5×4×3〕=60\n〔60通り〕\n"
             for i in range(7, 20)]
    assert _guard().looping("## 列3\n" + "".join(lines))


def test_repetition_guard_ignores_json_boilerplate():
    """Real JSON output repeats lines like '"is_key_formula": false' — that must not stop generation."""
    block = (
        '          {{\n'
        '            "raw_text": "{f}",\n'
        '            "latex": "{f}",\n'
        '            "description": "{d}",\n'
        '            "is_key_formula": false\n'
        '          }},\n'
    )
    formulas = [("2x^2-4x+5=2(x^2-2x)+5", "括る"), ("x^2-2x=(x-1)^2-1^2", "平方完成"),
                ("2(x-1)^2-2+5", "展開"), ("2(x-1)^2+3", "結果"), ("2x^2+8x+7=2(x^2+4x)+7", "括る"),
                ("x^2+4x=(x+2)^2-2^2", "平方完成"), ("2(x+2)^2-8+7", "展開"), ("2(x+2)^2-1", "結果")]
    text = '{\n  "sections": [\n' + "".join(block.format(f=f, d=d) for f, d in formulas)
    assert not _guard().looping(text)


def test_repetition_guard_ignores_repeated_json_section_skeletons():
    """Several sections share keys and short values (null, [, "arrow") and even a repeated note."""
    section = (
        '    {{\n      "section_id": "sec_0{i}",\n      "title": "例題{i}: {t}",\n'
        '      "exercise": null,\n      "visual_annotations": [\n        {{\n'
        '          "element_type": "arrow",\n          "target_text": "{t}",\n'
        '          "note": "平方完成の計算式を矢印で示す"\n        }}\n      ],\n'
        '      "teaching_notes": "平方完成の計算式と、p の求め方を強調する。"\n    }},\n'
    )
    targets = ["2x^2-4x+5", "2x^2+8x+7", "x^2-x-2", "3x^2+6x-1", "x^2+4x+1"]
    text = '{\n  "sections": [\n' + "".join(section.format(i=i, t=t) for i, t in enumerate(targets, 1))
    assert not _guard().looping(text)


# ---------------------------------------------------------------------------
# Foundry Local backend
# ---------------------------------------------------------------------------

def _foundry(monkeypatch, device="gpu"):
    b = FoundryLocalBackend(cli="foundry")
    monkeypatch.setattr(b, "base_url", lambda refresh=False: "http://127.0.0.1:5555")
    monkeypatch.setattr(b, "list_models", lambda: [
        ModelInfo(id="qwen3.5-4b-generic-gpu:4", alias="qwen3.5-4b", device=device, vision=True, loaded=True)])
    return b


def test_foundry_generate_uses_responses_api_with_images(monkeypatch):
    captured = {}
    events = [
        {"type": "response.created"},
        {"type": "response.output_text.delta", "delta": "## 列1\n"},
        {"type": "response.output_text.delta", "delta": "$y=a(x-p)^2+q$"},
        {"type": "response.completed", "response": {"status": "completed",
                                                    "usage": {"input_tokens": 900, "output_tokens": 12}}},
    ]
    monkeypatch.setattr(llm_client.httpx, "stream", _fake_stream(_sse(events), captured))
    b = _foundry(monkeypatch)
    deltas = []
    res = b.generate("qwen3.5-4b", "書き起こして", images=[_png_bytes()], max_tokens=100,
                     on_delta=lambda d, n: deltas.append(n))

    assert res.text == "## 列1\n$y=a(x-p)^2+q$"
    assert res.device == "gpu" and res.backend == "foundry"
    assert res.input_tokens == 900 and res.output_tokens == 12 and not res.truncated
    assert deltas and deltas[-1] == len(res.text)
    assert captured["url"].endswith("/v1/responses")
    content = captured["json"]["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "書き起こして"}
    assert content[1]["type"] == "input_image" and content[1]["media_type"] == "image/jpeg"
    assert captured["json"]["model"] == "qwen3.5-4b-generic-gpu:4"


def test_foundry_budget_includes_prompt_and_caps_output(monkeypatch):
    """Foundry treats max_output_tokens as a whole-sequence limit: add the prompt estimate, cap output locally."""
    captured = {}
    events = [{"type": "response.output_text.delta", "delta": f"t{i} "} for i in range(50)]
    monkeypatch.setattr(llm_client.httpx, "stream", _fake_stream(_sse(events), captured))
    res = _foundry(monkeypatch).generate("qwen3.5-4b", "あ" * 100, images=[_png_bytes((1280, 960))], max_tokens=10)
    # 1280x960 image → 40*30 = 1200 blocks; prompt 100 chars; plus margins
    assert captured["json"]["max_output_tokens"] >= 10 + 1200 + 100
    assert res.truncated and res.output_tokens == 10 and res.text.count("t") == 10


def test_foundry_requires_gpu_by_default(monkeypatch):
    monkeypatch.setattr(config, "LLM_REQUIRE_GPU", True)
    monkeypatch.setattr(config, "LLM_ALLOW_CPU", False)
    b = _foundry(monkeypatch, device="cpu")
    with pytest.raises(LLMError, match="GPU"):
        b.generate("qwen3.5-4b", "hi")


def test_foundry_cpu_allowed_with_override(monkeypatch):
    monkeypatch.setattr(config, "LLM_ALLOW_CPU", True)
    events = [{"type": "response.output_text.delta", "delta": "ok"},
              {"type": "response.completed", "response": {"usage": {"output_tokens": 1}}}]
    monkeypatch.setattr(llm_client.httpx, "stream", _fake_stream(_sse(events), {}))
    assert _foundry(monkeypatch, device="cpu").generate("qwen3.5-4b", "hi").text == "ok"


def test_foundry_missing_model_gives_setup_hint(monkeypatch):
    b = _foundry(monkeypatch)
    with pytest.raises(LLMError, match="setup_llm"):
        b.ensure_model("not-downloaded-model")


def test_generation_stops_on_repetition(monkeypatch):
    events = [{"type": "response.output_text.delta", "delta": "=(x-1/2)^2-9/4、"} for _ in range(200)]
    monkeypatch.setattr(llm_client.httpx, "stream", _fake_stream(_sse(events), {}))
    res = _foundry(monkeypatch).generate("qwen3.5-4b", "hi", max_tokens=4000)
    assert res.truncated
    assert len(res.text) < 200 * 15


def test_callback_can_abort(monkeypatch):
    events = [{"type": "response.output_text.delta", "delta": f"行{i}\n"} for i in range(50)]
    monkeypatch.setattr(llm_client.httpx, "stream", _fake_stream(_sse(events), {}))
    res = _foundry(monkeypatch).generate("qwen3.5-4b", "hi", on_delta=lambda d, n: False if n > 10 else None)
    assert res.truncated and len(res.text) <= 12


def test_foundry_failed_event_raises(monkeypatch):
    events = [{"type": "response.failed", "response": {"error": {"message": "boom"}}}]
    monkeypatch.setattr(llm_client.httpx, "stream", _fake_stream(_sse(events), {}))
    with pytest.raises(LLMError, match="失敗"):
        _foundry(monkeypatch).generate("qwen3.5-4b", "hi")


def test_foundry_recovers_once_from_lost_gpu_device(monkeypatch):
    calls = {"n": 0, "run": []}

    @contextmanager
    def _stream(method, url, json=None, headers=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            yield _FakeResponse(_sse([{"type": "response.failed", "response": {"error": {
                "message": "Failed to download data from buffer: [Device] is lost."}}}]))
        else:
            yield _FakeResponse(_sse([{"type": "response.output_text.delta", "delta": "ok"},
                                      {"type": "response.completed", "response": {"usage": {"output_tokens": 1}}}]))

    monkeypatch.setattr(llm_client.httpx, "stream", _stream)
    b = _foundry(monkeypatch)
    monkeypatch.setattr(b, "_run", lambda *a, **k: calls["run"].append(a) or {"success": True})
    assert b.generate("qwen3.5-4b", "hi").text == "ok"
    assert calls["n"] == 2 and calls["run"][0][:2] == ("server", "restart")


def test_foundry_base_url_is_validated(monkeypatch):
    b = FoundryLocalBackend(cli="foundry")
    monkeypatch.setattr(b, "_run", lambda *a, **k: {"running": True, "webUrls": ["http://evil.example.com:80"]})
    with pytest.raises(ValueError):
        b.base_url(refresh=True)


# ---------------------------------------------------------------------------
# Ollama / OpenAI-compatible parsing
# ---------------------------------------------------------------------------

def test_ollama_stream_parsing_and_think_disabled(monkeypatch):
    captured = {}
    lines = [json.dumps({"message": {"content": "こん"}, "done": False}),
             json.dumps({"message": {"content": "にちは"}, "done": False}),
             json.dumps({"message": {"content": ""}, "done": True, "done_reason": "stop",
                         "prompt_eval_count": 50, "eval_count": 3})]
    monkeypatch.setattr(llm_client.httpx, "stream", _fake_stream(lines, captured))
    b = OllamaBackend(host="http://localhost:11434")
    monkeypatch.setattr(b, "ensure_model", lambda m: ModelInfo(id=m, alias=m, device="gpu", loaded=True))
    res = b.generate("qwen3-vl:4b-instruct", "hi", images=[_png_bytes()])
    assert res.text == "こんにちは" and res.input_tokens == 50 and res.output_tokens == 3
    assert captured["json"]["think"] is False
    assert captured["json"]["messages"][-1]["images"]


def test_openai_compat_stream_parsing(monkeypatch):
    captured = {}
    events = [{"choices": [{"delta": {"content": "A"}}]},
              {"choices": [{"delta": {"content": "B"}, "finish_reason": "stop"}]},
              {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 2}}]
    monkeypatch.setattr(llm_client.httpx, "stream", _fake_stream(_sse(events), captured))
    b = OpenAICompatBackend(base_url="http://127.0.0.1:1234/v1", device="gpu")
    res = b.generate("some-vl-model", "hi", images=[_png_bytes()])
    assert res.text == "AB" and res.output_tokens == 2
    assert captured["url"] == "http://127.0.0.1:1234/v1/chat/completions"
    assert captured["json"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured["json"]["messages"][0]["content"][1]["type"] == "image_url"


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

class _StubBackend:
    def __init__(self, name, status):
        self.name = name
        self._status = status

    def status(self):
        return self._status


def test_select_backend_prefers_gpu_foundry(monkeypatch):
    stubs = {
        "foundry": _StubBackend("foundry", BackendStatus("foundry", True, "ok", models=[
            ModelInfo(id="qwen3.5-4b-generic-gpu:4", alias=config.FOUNDRY_VISION_MODEL, device="gpu")])),
        "ollama": _StubBackend("ollama", BackendStatus("ollama", True, "ok", models=[
            ModelInfo(id=config.OLLAMA_VISION_MODEL_GPU, alias=config.OLLAMA_VISION_MODEL_GPU, device="unknown")])),
    }
    monkeypatch.setattr(llm_client, "create_backend", lambda kind, host=None: stubs[kind])
    backend, statuses = select_backend("auto")
    assert backend is stubs["foundry"]


def test_select_backend_falls_back_to_ollama_when_foundry_missing(monkeypatch):
    stubs = {
        "foundry": _StubBackend("foundry", BackendStatus("foundry", False, "not installed")),
        "ollama": _StubBackend("ollama", BackendStatus("ollama", True, "ok", models=[
            ModelInfo(id=config.OLLAMA_VISION_MODEL_GPU, alias=config.OLLAMA_VISION_MODEL_GPU, device="unknown")])),
    }
    monkeypatch.setattr(llm_client, "create_backend", lambda kind, host=None: stubs[kind])
    backend, statuses = select_backend("auto")
    assert backend is stubs["ollama"]
    assert [s.backend for s in statuses] == ["foundry", "ollama"]


def test_select_backend_returns_none_when_nothing_available(monkeypatch):
    stubs = {k: _StubBackend(k, BackendStatus(k, False, "down")) for k in ("foundry", "ollama")}
    monkeypatch.setattr(llm_client, "create_backend", lambda kind, host=None: stubs[kind])
    backend, statuses = select_backend("auto")
    assert backend is None and len(statuses) == 2
