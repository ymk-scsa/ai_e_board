"""
Vision AI module for analyzing blackboard images using Ollama Vision LLMs.
"""

import base64
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import httpx
import ollama

import config

logger = logging.getLogger("ai_e_board.vision")
logging.basicConfig(level=logging.INFO)


class VisionAnalyzer:
    """Vision AI analyzer wrapping Ollama client for multi-image blackboard analysis."""

    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = config.VISION_TIMEOUT_SECONDS,
    ):
        self.host = host or config.OLLAMA_HOST
        self.model = model or config.DEFAULT_VISION_MODEL
        self.timeout = timeout
        self.client = ollama.Client(host=self.host)

    def check_connection(self) -> Tuple[bool, str, List[str]]:
        """
        Check if Ollama service is reachable and retrieve available model names.
        Returns (is_connected, message, available_models).
        """
        try:
            response = self.client.list()
            # response can be a ListResponse or dict depending on SDK version
            models = []
            if hasattr(response, "models"):
                for m in response.models:
                    name = getattr(m, "model", None) or getattr(m, "name", str(m))
                    models.append(name)
            elif isinstance(response, dict) and "models" in response:
                for m in response["models"]:
                    name = m.get("model") or m.get("name") or str(m)
                    models.append(name)
            
            msg = f"Connected to Ollama at {self.host}. {len(models)} model(s) found."
            return True, msg, models
        except Exception as e:
            logger.warning(f"Failed to connect to Ollama at {self.host}: {e}")
            return False, f"Ollamaサーバー（{self.host}）に接続できませんでした: {str(e)}", []

    def encode_image_to_base64(self, image_path: Path) -> str:
        """Read and encode an image file to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def load_analyze_prompt(self) -> str:
        """Load the analysis prompt from prompts/analyze.txt."""
        prompt_file = config.PROMPTS_DIR / "analyze.txt"
        if prompt_file.exists():
            return prompt_file.read_text(encoding="utf-8")
        return "Analyze the blackboard image and extract the lesson structure in JSON format."

    def analyze_board_images(
        self,
        image_paths: List[Path],
        custom_instructions: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze one or more blackboard images using the Vision LLM.
        Returns a dict containing raw LLM response text, timing, and metadata.
        """
        model_to_use = model_override or self.model
        base_prompt = self.load_analyze_prompt()

        if custom_instructions:
            prompt = f"{base_prompt}\n\n【教員からの個別指示】\n{custom_instructions}"
        else:
            prompt = base_prompt

        # Encode images
        encoded_images = []
        image_names = []
        for p in image_paths:
            path_obj = Path(p)
            if not path_obj.exists():
                raise FileNotFoundError(f"Image not found: {path_obj}")
            encoded_images.append(self.encode_image_to_base64(path_obj))
            image_names.append(path_obj.name)

        if not encoded_images:
            raise ValueError("No images provided for analysis.")

        logger.info(
            f"Sending {len(encoded_images)} image(s) to Ollama ({model_to_use}) for analysis..."
        )

        start_time = time.time()
        try:
            response = self.client.generate(
                model=model_to_use,
                prompt=prompt,
                images=encoded_images,
                options={
                    "temperature": 0.1,  # Low temperature for deterministic structure extraction
                    "top_p": 0.9,
                },
            )
            raw_response = response.get("response", "") if isinstance(response, dict) else getattr(response, "response", "")
            elapsed = time.time() - start_time

            return {
                "success": True,
                "raw_text": raw_response,
                "model": model_to_use,
                "elapsed_seconds": elapsed,
                "source_images": image_names,
                "is_mock": False,
            }
        except Exception as e:
            elapsed = time.time() - start_time
            err_msg = str(e)
            logger.error(f"Ollama Vision API error: {err_msg}")
            
            # Check for common model not found or connection failure
            return {
                "success": False,
                "error": err_msg,
                "raw_text": "",
                "model": model_to_use,
                "elapsed_seconds": elapsed,
                "source_images": image_names,
                "is_mock": False,
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
            response = self.client.generate(
                model=self.model,
                prompt=repair_prompt,
                options={"temperature": 0.0},
            )
            return response.get("response", "") if isinstance(response, dict) else getattr(response, "response", "")
        except Exception as e:
            logger.error(f"JSON repair request failed: {e}")
            return broken_json_text
