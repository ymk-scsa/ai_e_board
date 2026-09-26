"""
Vision AI module for analyzing blackboard images with a local vision LLM.
The actual runtime (Foundry Local / Ollama / OpenAI-compatible) is abstracted by ai.llm_client.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import config
from ai.board_pipeline import BoardAnalysis, analyze_board
from ai.llm_client import (
    BackendStatus,
    LLMBackend,
    LLMError,
    create_backend,
    default_model_for,
    select_backend,
)

logger = logging.getLogger("ai_e_board.vision")

ProgressCallback = Callable[[str, int], Optional[bool]]


class VisionAnalyzer:
    """Vision AI analyzer for multi-image blackboard analysis on a local (GPU) LLM backend."""

    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = config.LLM_TIMEOUT_SECONDS,
        backend: Optional[LLMBackend] = None,
        backend_kind: Optional[str] = None,
    ):
        """
        host: Ollama host (used only when the Ollama backend is selected).
        backend: an already-created backend (preferred; the app caches it across reruns).
        backend_kind: "auto" | "foundry" | "ollama" | "openai" (default: config.LLM_BACKEND).
        """
        self.host = host or config.OLLAMA_HOST
        self.timeout = timeout
        self.host_error: Optional[str] = None
        self.statuses: List[BackendStatus] = []
        self.backend: Optional[LLMBackend] = backend
        if self.backend is None:
            kind = (backend_kind or config.LLM_BACKEND).lower()
            try:
                if kind == "auto":
                    self.backend, self.statuses = select_backend("auto", ollama_host=self.host)
                    if self.backend is None:
                        self.host_error = "利用できるローカルLLM（Foundry Local / Ollama）が見つかりません。" + " / ".join(
                            s.message for s in self.statuses)
                else:
                    self.backend = create_backend(kind, ollama_host=self.host)
            except ValueError as e:  # SSRF対策: 許可リスト外のホスト
                self.host_error = str(e)
                logger.error(f"LLM backend rejected: {e}")
        self.model = model or (default_model_for(self.backend) if self.backend else config.DEFAULT_VISION_MODEL)

    @property
    def backend_name(self) -> str:
        return self.backend.name if self.backend else "none"

    def _require_backend(self) -> LLMBackend:
        if self.backend is None:
            raise ConnectionError(self.host_error or "LLMバックエンドが初期化されていません。")
        return self.backend

    def check_connection(self) -> Tuple[bool, str, List[str]]:
        """
        Check the backend and list available models.
        Returns (is_connected, message, available_models).
        """
        try:
            st = self._require_backend().status()
        except Exception as e:
            return False, str(e), []
        return st.available, st.message, [m.alias or m.id for m in st.models]

    def model_device(self, model: Optional[str] = None) -> str:
        """Return "gpu" / "cpu" / "npu" / "unknown" for the model as reported by the backend."""
        model = model or self.model
        try:
            st = self._require_backend().status()
        except Exception:
            return "unknown"
        info = next((m for m in st.models if model in (m.alias, m.id)), None)
        return info.device if info else "unknown"

    def analyze_board_to_lesson(
        self,
        image_paths: List[Path],
        custom_instructions: Optional[str] = None,
        model_override: Optional[str] = None,
        on_phase: Optional[Callable[[str, str], None]] = None,
        on_progress: Optional[ProgressCallback] = None,
    ) -> BoardAnalysis:
        """Full board analysis: transcribe (vision) → structure (text) → Lesson. See ai/board_pipeline.py."""
        for p in image_paths:
            if not Path(p).exists():
                raise FileNotFoundError(f"Image not found: {p}")
        if not image_paths:
            raise ValueError("No images provided for analysis.")
        try:
            backend = self._require_backend()
        except ConnectionError as e:
            return BoardAnalysis(None, str(e), [], model=model_override or self.model)
        return analyze_board(backend, model_override or self.model, list(image_paths),
                             custom_instructions=custom_instructions, on_phase=on_phase, on_delta=on_progress)

    def load_analyze_prompt(self, name: str = config.TRANSCRIBE_PROMPT_FILE) -> str:
        """Load a prompt from prompts/ (default: the board transcription prompt)."""
        prompt_file = config.PROMPTS_DIR / name
        if prompt_file.exists():
            return prompt_file.read_text(encoding="utf-8")
        raise FileNotFoundError(f"解析プロンプトが見つかりません: {prompt_file}")

    def analyze_board_images(
        self,
        image_paths: List[Path],
        custom_instructions: Optional[str] = None,
        model_override: Optional[str] = None,
        on_progress: Optional[ProgressCallback] = None,
        prompt_name: str = config.TRANSCRIBE_PROMPT_FILE,
    ) -> Dict[str, Any]:
        """
        Run one vision LLM pass over the images with the given prompt (default: transcription).
        Returns a dict containing raw LLM response text, timing, device, and metadata.
        """
        model_to_use = model_override or self.model
        base_prompt = self.load_analyze_prompt(prompt_name)
        prompt = f"{base_prompt}\n\n【教員からの個別指示】\n{custom_instructions}" if custom_instructions else base_prompt

        image_names = []
        for p in image_paths:
            path_obj = Path(p)
            if not path_obj.exists():
                raise FileNotFoundError(f"Image not found: {path_obj}")
            image_names.append(path_obj.name)
        if not image_names:
            raise ValueError("No images provided for analysis.")

        base = {"model": model_to_use, "backend": self.backend_name, "source_images": image_names, "is_mock": False}
        try:
            backend = self._require_backend()
            logger.info(f"Sending {len(image_names)} image(s) to {backend.name} ({model_to_use}) for analysis...")
            res = backend.generate(
                model_to_use,
                prompt,
                images=list(image_paths),
                max_tokens=config.VISION_MAX_OUTPUT_TOKENS,
                on_delta=on_progress,
            )
        except (LLMError, ConnectionError) as e:
            logger.error(f"Vision LLM error: {e}")
            return {**base, "success": False, "error": str(e), "raw_text": "", "elapsed_seconds": 0.0}

        return {
            **base,
            "success": True,
            "raw_text": res.text,
            "model": res.model,
            "device": res.device,
            "elapsed_seconds": res.elapsed_seconds,
            "first_token_seconds": res.first_token_seconds,
            "input_tokens": res.input_tokens,
            "output_tokens": res.output_tokens,
            "truncated": res.truncated,
        }

    def repair_json_with_llm(self, broken_json_text: str, error_details: str) -> str:
        """
        Ask LLM to fix invalid JSON syntax while preserving exact math and pedagogical content.
        """
        repair_prompt = f"""You are a JSON repair assistant.
The following JSON output has a syntax error:
Error details: {error_details}

Input text:
{broken_json_text}

Instructions:
1. Fix syntax errors (missing commas, unescaped quotes in LaTeX, unbalanced brackets).
2. Do NOT change any mathematical formulas or pedagogical contents.
3. Return ONLY the valid RFC 8259 JSON, enclosed in ```json ... ``` code block.
"""
        try:
            return self._require_backend().generate(
                self.model, repair_prompt, max_tokens=config.VISION_MAX_OUTPUT_TOKENS
            ).text
        except Exception as e:
            logger.error(f"JSON repair request failed: {e}")
            return broken_json_text
