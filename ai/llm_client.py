"""
Backend-agnostic local LLM client for ai_e_board.

Supported backends (all run locally):
- foundry : Microsoft Foundry Local (Windows / macOS). Picks the GPU variant of a model automatically
            (WebGPU / Metal / CUDA / OpenVINO ...). Vision via the OpenAI-style Responses API.
- ollama  : Ollama (GPU on NVIDIA/AMD/Apple Silicon; CPU only on Windows ARM64).
- openai  : Any OpenAI-compatible server (LM Studio, llama-server, vLLM ...).

The app talks only to `LLMBackend.generate()`; backend differences (endpoint discovery, model loading,
image encoding, thinking suppression, streaming formats) are handled here.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Union

import httpx

import config
from ai.security import validate_ollama_host

logger = logging.getLogger("ai_e_board.llm_client")

ImageInput = Union[str, Path, bytes]
DeltaCallback = Callable[[str, int], Optional[bool]]  # (delta_text, output_chars_so_far) -> False to abort


class LLMError(RuntimeError):
    """Raised when a backend is unavailable or a generation request fails."""


@dataclass
class ModelInfo:
    id: str
    alias: str
    device: str = "unknown"  # "gpu" | "cpu" | "npu" | "unknown"
    vision: bool = False
    loaded: bool = False


@dataclass
class BackendStatus:
    backend: str
    available: bool
    message: str
    base_url: str = ""
    models: List[ModelInfo] = field(default_factory=list)


@dataclass
class LLMResult:
    text: str
    backend: str
    model: str
    device: str
    elapsed_seconds: float
    first_token_seconds: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    truncated: bool = False  # stopped by token limit, repetition guard, or callback

    @property
    def tokens_per_second(self) -> Optional[float]:
        if self.output_tokens and self.first_token_seconds is not None and self.elapsed_seconds > self.first_token_seconds:
            return self.output_tokens / (self.elapsed_seconds - self.first_token_seconds)
        return None


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def prepare_image(image: ImageInput, max_side: int = config.LLM_IMAGE_MAX_SIDE, quality: int = 88) -> bytes:
    """Load an image (path or bytes), fix EXIF orientation, downscale to max_side, return JPEG bytes."""
    from PIL import Image, ImageOps

    src = io.BytesIO(image) if isinstance(image, (bytes, bytearray)) else Path(image)
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((max_side, max_side), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks (and a dangling unterminated <think> prefix)."""
    text = _THINK_RE.sub("", text)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.lstrip()


class RepetitionGuard:
    """
    Detects degenerate looping output (common with small models under greedy decoding) without
    tripping on legitimately repetitive structure such as JSON boilerplate lines.

    - Line loops: the last `lines` meaningful lines (>= min_len chars, digits normalized so that
      "例7…", "例8…" count as the same line) have each already appeared >= `repeats` times.
    - Inline loops: one long line whose tail is the same chunk repeated back-to-back `repeats` times.
    """

    _DIGITS = re.compile(r"\d+")
    _JSON_KEY = re.compile(r'^"[A-Za-z_][A-Za-z0-9_]*"\s*:\s*')

    def __init__(self, lines: int = 6, repeats: int = 3, min_len: int = 12, check_every: int = 16):
        self.lines = lines
        self.repeats = repeats
        self.min_len = min_len
        self.check_every = check_every
        self._calls = 0

    def looping(self, text: str) -> bool:
        self._calls += 1
        if self._calls % self.check_every:
            return False
        return self._line_loop(text) or self._inline_loop(text)

    def _line_loop(self, text: str) -> bool:
        complete = text.split("\n")[:-1][-300:]
        # JSON keys are structural boilerplate: compare only the values (short values like null / [ are ignored)
        norm = [self._DIGITS.sub("#", self._JSON_KEY.sub("", ln.strip())) for ln in complete]
        meaningful = [ln for ln in norm if len(ln) >= self.min_len]
        if len(meaningful) < self.lines * self.repeats:
            return False
        counts: dict = {}
        for ln in meaningful:
            counts[ln] = counts.get(ln, 0) + 1
        return all(counts[ln] >= self.repeats for ln in meaningful[-self.lines:])

    def _inline_loop(self, text: str) -> bool:
        tail = text[-1200:].rsplit("\n", 1)[-1]
        if len(tail) < 300:
            return False
        for period in range(10, len(tail) // self.repeats + 1):
            unit = tail[-period:]
            if tail.endswith(unit * self.repeats):
                return True
        return False


# ---------------------------------------------------------------------------
# Backend base class
# ---------------------------------------------------------------------------

class LLMBackend(ABC):
    name = "base"

    def __init__(self, timeout: float = config.LLM_TIMEOUT_SECONDS):
        self.timeout = timeout

    @abstractmethod
    def status(self) -> BackendStatus: ...

    @abstractmethod
    def ensure_model(self, model: str) -> ModelInfo:
        """Make sure the model is ready (downloaded / loaded). Returns its info."""

    @abstractmethod
    def _stream(self, model_id: str, prompt: str, images: List[bytes], system: Optional[str],
                max_tokens: int, temperature: float) -> Iterable[dict]:
        """Yield {"delta": str} events and finally {"usage": {...}}."""

    def generate(
        self,
        model: str,
        prompt: str,
        images: Sequence[ImageInput] = (),
        system: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        on_delta: Optional[DeltaCallback] = None,
        image_max_side: int = config.LLM_IMAGE_MAX_SIDE,
        repetition_guard: bool = True,
    ) -> LLMResult:
        info = self.ensure_model(model)
        if config.LLM_REQUIRE_GPU and info.device != "gpu" and not config.LLM_ALLOW_CPU:
            raise LLMError(
                f"モデル {info.id} は {info.device.upper()} で動作しています。GPUでの実行が必須です"
                "（開発時のみ LLM_ALLOW_CPU=1 で許可できます）。"
            )
        imgs = [prepare_image(i, image_max_side) for i in images]
        guard = RepetitionGuard() if repetition_guard else None
        start = time.time()
        first = None
        parts: List[str] = []
        usage: dict = {}
        truncated = False
        try:
            for ev in self._stream(info.id, prompt, imgs, system, max_tokens, temperature):
                if "usage" in ev:
                    usage = ev["usage"] or {}
                    truncated = truncated or bool(ev.get("truncated"))
                    continue
                delta = ev.get("delta") or ""
                if not delta:
                    continue
                if first is None:
                    first = time.time() - start
                parts.append(delta)
                so_far = "".join(parts)
                if on_delta is not None and on_delta(delta, len(so_far)) is False:
                    truncated = True
                    break
                if guard and guard.looping(so_far):
                    logger.warning("Repetition detected; stopping generation early.")
                    truncated = True
                    break
        except httpx.TimeoutException as e:
            raise LLMError(f"LLMの応答がタイムアウトしました（{self.timeout:.0f}秒）: {e}") from e
        except httpx.HTTPError as e:
            raise LLMError(f"LLMサーバーとの通信に失敗しました: {e}") from e

        text = strip_thinking("".join(parts))
        out_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or usage.get("eval_count")
        if out_tokens and out_tokens >= max_tokens:
            truncated = True
        return LLMResult(
            text=text,
            backend=self.name,
            model=info.id,
            device=info.device,
            elapsed_seconds=time.time() - start,
            first_token_seconds=first,
            input_tokens=usage.get("input_tokens") or usage.get("prompt_tokens") or usage.get("prompt_eval_count"),
            output_tokens=out_tokens,
            truncated=truncated,
        )


def _iter_sse(response: httpx.Response) -> Iterable[dict]:
    for line in response.iter_lines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            logger.debug(f"Skipping non-JSON SSE line: {data[:80]}")


def _raise_for_status(r: httpx.Response) -> None:
    if r.status_code >= 400:
        body = r.read().decode("utf-8", "replace")[:500]
        raise LLMError(f"LLMサーバーがエラーを返しました（HTTP {r.status_code}）: {body}")


# ---------------------------------------------------------------------------
# Foundry Local
# ---------------------------------------------------------------------------

class FoundryLocalBackend(LLMBackend):
    """Microsoft Foundry Local via its CLI (discovery / load) and OpenAI-compatible web service."""

    name = "foundry"

    def __init__(self, timeout: float = config.LLM_TIMEOUT_SECONDS, cli: Optional[str] = None):
        super().__init__(timeout)
        self.cli = cli or shutil.which("foundry")
        self._base_url: Optional[str] = None

    def _run(self, *args: str, timeout: float = 120) -> dict:
        if not self.cli:
            raise LLMError("Foundry Local（foundry コマンド）が見つかりません。")
        proc = subprocess.run([self.cli, *args, "-o", "json"], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout)
        lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip().startswith("{")]
        if not lines:
            raise LLMError(f"foundry {' '.join(args)} が失敗しました: {(proc.stderr or proc.stdout).strip()[:300]}")
        return json.loads(lines[-1])

    def base_url(self, refresh: bool = False) -> str:
        if self._base_url and not refresh:
            return self._base_url
        st = self._run("server", "status")
        if not st.get("running"):
            st = self._run("server", "start", timeout=180)
        urls = st.get("webUrls") or []
        if not urls:
            raise LLMError("Foundry Local のサーバーURLを取得できませんでした。")
        self._base_url = validate_ollama_host(urls[0].rstrip("/"))
        return self._base_url

    def list_models(self) -> List[ModelInfo]:
        data = self._run("cache", "list")
        return [
            ModelInfo(
                id=m.get("id", ""),
                alias=m.get("alias", ""),
                device=(m.get("device") or "unknown").lower(),
                vision=(m.get("type") or "").lower() == "multimodal",
                loaded=bool(m.get("loaded")),
            )
            for m in data.get("models", [])
        ]

    def status(self) -> BackendStatus:
        if not self.cli:
            return BackendStatus(self.name, False, "Foundry Local がインストールされていません。")
        try:
            url = self.base_url(refresh=True)
            models = self.list_models()
            return BackendStatus(self.name, True, f"Foundry Local に接続しました（{url}）。", url, models)
        except Exception as e:
            return BackendStatus(self.name, False, f"Foundry Local に接続できません: {e}")

    def ensure_model(self, model: str) -> ModelInfo:
        models = self.list_models()
        info = next((m for m in models if model in (m.alias, m.id)), None)
        if info is None:
            raise LLMError(
                f"モデル「{model}」がダウンロードされていません。"
                f"`python tools/setup_llm.py` を実行して準備してください。"
            )
        if not info.loaded:
            logger.info(f"Loading Foundry model {info.id} ...")
            res = self._run("model", "load", info.alias or info.id, timeout=900)
            if not res.get("success", True):
                raise LLMError(f"モデルの読み込みに失敗しました: {res.get('message')}")
            info.loaded = True
            self.base_url(refresh=True)  # load may (re)start the daemon
        return info

    _RECOVERABLE_GPU_ERRORS = ("device] is lost", "device is lost", "device lost", "dxgi_error_device")

    def generate(self, model: str, prompt: str, *args, **kwargs) -> LLMResult:
        """
        Same as LLMBackend.generate, but recovers once from a lost GPU device (driver reset / TDR).
        A lost WebGPU device stays lost for the whole Foundry daemon process, so the daemon is restarted
        (reloading the model alone is not enough), then the request is retried.
        """
        try:
            return super().generate(model, prompt, *args, **kwargs)
        except LLMError as e:
            if not any(s in str(e).lower() for s in self._RECOVERABLE_GPU_ERRORS):
                raise
            logger.warning(f"GPU device lost; restarting Foundry Local and retrying once: {e}")
            try:
                self._run("server", "restart", timeout=300)
            except LLMError as restart_err:
                raise LLMError(f"GPUデバイスが失われ、Foundry Local の再起動にも失敗しました: {restart_err}") from e
            self._base_url = None
            return super().generate(model, prompt, *args, **kwargs)

    @staticmethod
    def estimate_input_tokens(prompt: str, images: List[bytes], system: Optional[str] = None) -> int:
        """
        Conservative prompt-length estimate. Foundry Local treats `max_output_tokens` as the limit of the
        *whole sequence* (prompt + output), so the prompt length must be added to the output budget.
        Qwen-VL style models use about one token per 32x32 pixel block.
        """
        from PIL import Image

        n = len(prompt) + len(system or "") + 128
        for b in images:
            with Image.open(io.BytesIO(b)) as im:
                w, h = im.size
            n += (-(-w // 32)) * (-(-h // 32)) + 16
        return n

    def _stream(self, model_id, prompt, images, system, max_tokens, temperature):
        content = [{"type": "input_text", "text": prompt}]
        content += [{"type": "input_image", "image_data": base64.b64encode(b).decode(), "media_type": "image/jpeg"}
                    for b in images]
        body = {
            "model": model_id,
            "stream": True,
            "max_output_tokens": max_tokens + self.estimate_input_tokens(prompt, images, system),
            "temperature": temperature,
            "input": [{"type": "message", "role": "user", "content": content}],
        }
        if system:
            body["instructions"] = system
        with httpx.stream("POST", self.base_url() + "/v1/responses", json=body, timeout=self.timeout) as r:
            _raise_for_status(r)
            emitted = 0  # each output_text.delta event carries one token
            for ev in _iter_sse(r):
                et = ev.get("type", "")
                if et == "response.output_text.delta":
                    emitted += 1
                    yield {"delta": ev.get("delta", "")}
                    if emitted >= max_tokens:
                        yield {"usage": {"output_tokens": emitted}, "truncated": True}
                        return
                elif et == "response.completed":
                    resp = ev.get("response") or {}
                    yield {"usage": resp.get("usage"), "truncated": resp.get("status") == "incomplete"}
                elif et in ("response.failed", "error"):
                    err = (ev.get("response") or {}).get("error") or ev.get("error") or ev
                    detail = err.get("message") if isinstance(err, dict) and err.get("message") else json.dumps(err, ensure_ascii=False)
                    raise LLMError(f"Foundry Local の生成が失敗しました: {str(detail)[:500]}")


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

class OllamaBackend(LLMBackend):
    name = "ollama"

    def __init__(self, host: Optional[str] = None, timeout: float = config.LLM_TIMEOUT_SECONDS):
        super().__init__(timeout)
        self.host = validate_ollama_host(host or config.OLLAMA_HOST)

    def _get(self, path: str) -> dict:
        r = httpx.get(self.host + path, timeout=10)
        r.raise_for_status()
        return r.json()

    def status(self) -> BackendStatus:
        try:
            tags = self._get("/api/tags")
            loaded = {m.get("name"): m for m in self._get("/api/ps").get("models", [])}
            models = []
            for m in tags.get("models", []):
                name = m.get("name") or m.get("model")
                ps = loaded.get(name)
                device = ("gpu" if ps.get("size_vram", 0) > 0 else "cpu") if ps else "unknown"
                models.append(ModelInfo(id=name, alias=name, device=device, loaded=bool(ps)))
            return BackendStatus(self.name, True, f"Ollama に接続しました（{self.host}）。", self.host, models)
        except Exception as e:
            return BackendStatus(self.name, False, f"Ollama に接続できません（{self.host}）: {e}")

    def ensure_model(self, model: str) -> ModelInfo:
        try:
            names = [m.get("name") for m in self._get("/api/tags").get("models", [])]
        except Exception as e:
            raise LLMError(f"Ollama に接続できません（{self.host}）: {e}") from e
        if model not in names:
            raise LLMError(f"Ollama にモデル「{model}」がありません（ollama pull {model}）。")
        # Load with an empty request, then read where it lives (GPU/CPU) from /api/ps.
        httpx.post(self.host + "/api/generate", json={"model": model, "prompt": "", "keep_alive": "30m"}, timeout=self.timeout)
        ps = {m.get("name"): m for m in self._get("/api/ps").get("models", [])}.get(model)
        device = ("gpu" if ps and ps.get("size_vram", 0) > 0 else "cpu") if ps else "unknown"
        return ModelInfo(id=model, alias=model, device=device, loaded=bool(ps))

    def _stream(self, model_id, prompt, images, system, max_tokens, temperature):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        msg = {"role": "user", "content": prompt}
        if images:
            msg["images"] = [base64.b64encode(b).decode() for b in images]
        messages.append(msg)
        body = {"model": model_id, "messages": messages, "stream": True, "think": False, "keep_alive": "30m",
                "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": config.LLM_NUM_CTX}}
        with httpx.stream("POST", self.host + "/api/chat", json=body, timeout=self.timeout) as r:
            _raise_for_status(r)
            for line in r.iter_lines():
                if not line.strip():
                    continue
                ch = json.loads(line)
                if ch.get("error"):
                    raise LLMError(f"Ollama の生成が失敗しました: {ch['error']}")
                delta = (ch.get("message") or {}).get("content") or ""
                if delta:
                    yield {"delta": delta}
                if ch.get("done"):
                    yield {"usage": {"prompt_eval_count": ch.get("prompt_eval_count"), "eval_count": ch.get("eval_count")},
                           "truncated": ch.get("done_reason") == "length"}


# ---------------------------------------------------------------------------
# Generic OpenAI-compatible server
# ---------------------------------------------------------------------------

class OpenAICompatBackend(LLMBackend):
    name = "openai"

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 timeout: float = config.LLM_TIMEOUT_SECONDS, device: str = config.OPENAI_COMPAT_DEVICE):
        super().__init__(timeout)
        self.base = validate_ollama_host((base_url or config.OPENAI_COMPAT_BASE_URL).rstrip("/").removesuffix("/v1"))
        self.api_key = api_key or config.OPENAI_COMPAT_API_KEY
        self.device = device

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _models(self) -> List[str]:
        r = httpx.get(self.base + "/v1/models", headers=self._headers(), timeout=10)
        r.raise_for_status()
        return [m.get("id") for m in r.json().get("data", [])]

    def status(self) -> BackendStatus:
        try:
            models = [ModelInfo(id=m, alias=m, device=self.device) for m in self._models()]
            return BackendStatus(self.name, True, f"OpenAI互換サーバーに接続しました（{self.base}）。", self.base, models)
        except Exception as e:
            return BackendStatus(self.name, False, f"OpenAI互換サーバーに接続できません（{self.base}）: {e}")

    def ensure_model(self, model: str) -> ModelInfo:
        return ModelInfo(id=model, alias=model, device=self.device, loaded=True)

    def _stream(self, model_id, prompt, images, system, max_tokens, temperature):
        content: list = [{"type": "text", "text": prompt}]
        content += [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b).decode()}}
                    for b in images]
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
        body = {"model": model_id, "messages": messages, "stream": True, "max_tokens": max_tokens,
                "temperature": temperature, "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False}}
        with httpx.stream("POST", self.base + "/v1/chat/completions", json=body, headers=self._headers(),
                          timeout=self.timeout) as r:
            _raise_for_status(r)
            for ch in _iter_sse(r):
                choice = (ch.get("choices") or [{}])[0]
                delta = (choice.get("delta") or {}).get("content") or ""
                if delta:
                    yield {"delta": delta}
                if ch.get("usage"):
                    yield {"usage": ch["usage"], "truncated": choice.get("finish_reason") == "length"}


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

def create_backend(kind: str, ollama_host: Optional[str] = None) -> LLMBackend:
    kind = (kind or "auto").lower()
    if kind == "foundry":
        return FoundryLocalBackend()
    if kind == "ollama":
        return OllamaBackend(host=ollama_host)
    if kind == "openai":
        return OpenAICompatBackend()
    raise ValueError(f"未知のLLMバックエンドです: {kind}")


def default_model_for(backend: LLMBackend) -> str:
    return {"foundry": config.FOUNDRY_VISION_MODEL, "ollama": config.OLLAMA_VISION_MODEL_GPU,
            "openai": config.OPENAI_COMPAT_MODEL}.get(backend.name, "")


def select_backend(preferred: str = config.LLM_BACKEND, ollama_host: Optional[str] = None) -> tuple[Optional[LLMBackend], List[BackendStatus]]:
    """
    Pick a usable backend. With "auto", try Foundry Local → Ollama → OpenAI-compatible and prefer one
    whose vision model runs on the GPU. Returns (backend or None, statuses checked).
    """
    order = [preferred] if preferred != "auto" else ["foundry", "ollama", "openai"]
    statuses: List[BackendStatus] = []
    fallback: Optional[LLMBackend] = None
    for kind in order:
        if kind == "openai" and preferred == "auto" and not config.OPENAI_COMPAT_BASE_URL_SET:
            continue
        try:
            backend = create_backend(kind, ollama_host)
        except ValueError as e:
            statuses.append(BackendStatus(kind, False, str(e)))
            continue
        st = backend.status()
        statuses.append(st)
        if not st.available:
            continue
        model = default_model_for(backend)
        info = next((m for m in st.models if model in (m.alias, m.id)), None)
        if info and (info.device == "gpu" or kind == "openai"):
            return backend, statuses
        if info and fallback is None:
            fallback = backend
    return fallback, statuses
