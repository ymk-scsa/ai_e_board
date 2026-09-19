"""
Parser module for extracting, repairing, and validating Lesson structures with Pydantic.
"""

import json
import logging
import re
from typing import Any, Dict, Optional, Tuple, Union

from pydantic import ValidationError

from models.schemas import Lesson, Section, Formula, ExampleProblem, Exercise

logger = logging.getLogger("ai_e_board.parser")


class LessonParser:
    """Robust JSON parser and Pydantic validator with multi-stage fallback repairing."""

    @staticmethod
    def extract_json_string(raw_text: str) -> str:
        """Extract the JSON substring from raw LLM output containing markdown or conversational text."""
        if not raw_text or not raw_text.strip():
            raise ValueError("LLM output is empty.")

        text = raw_text.strip()

        # Try markdown code block ```json ... ```
        json_block_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
        if json_block_match:
            return json_block_match.group(1).strip()

        # Fallback: Find outermost '{' and '}'
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return text[start_idx : end_idx + 1].strip()

        return text

    @staticmethod
    def sanitize_latex_escapes(json_str: str) -> str:
        """
        Fix unescaped backslashes in LaTeX strings inside JSON strings.
        In JSON, backslashes must be escaped (e.g. \\times -> \\\\times), but LLMs often output single backslashes.
        """
        # Replace unescaped single backslashes that are not valid JSON escape sequences (\", \\, \/, \b, \f, \n, \r, \t, \uXXXX)
        def replace_slash(match):
            char = match.group(1)
            if char in ['"', '\\', '/', 'b', 'f', 'n', 'r', 't']:
                return match.group(0)
            if char == 'u' and re.match(r'\\u[0-9a-fA-F]{4}', match.group(0)):
                return match.group(0)
            return r"\\" + char

        # Match single backslash followed by a character
        sanitized = re.sub(r'\\(.)', replace_slash, json_str)
        return sanitized

    @staticmethod
    def clean_trailing_commas(json_str: str) -> str:
        """Remove trailing commas before closing braces/brackets."""
        json_str = re.sub(r",\s*([\]}])", r"\1", json_str)
        return json_str

    @staticmethod
    def balance_brackets(json_str: str) -> str:
        """Add missing closing brackets/braces if JSON was truncated by token limit."""
        open_curly = json_str.count("{")
        close_curly = json_str.count("}")
        open_sq = json_str.count("[")
        close_sq = json_str.count("]")

        if open_sq > close_sq:
            json_str += "]" * (open_sq - close_sq)
        if open_curly > close_curly:
            json_str += "}" * (open_curly - close_curly)
        return json_str

    def repair_json_string(self, raw_json_str: str) -> Dict[str, Any]:
        """
        Multi-step repair pipeline for imperfect JSON strings.
        """
        # Step 1: Direct parse
        try:
            return json.loads(raw_json_str)
        except json.JSONDecodeError:
            pass

        # Step 2: Clean trailing commas
        cleaned = self.clean_trailing_commas(raw_json_str)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Step 3: Sanitize LaTeX backslashes
        sanitized = self.sanitize_latex_escapes(cleaned)
        try:
            return json.loads(sanitized)
        except json.JSONDecodeError:
            pass

        # Step 4: Balance brackets
        balanced = self.balance_brackets(sanitized)
        try:
            return json.loads(balanced)
        except json.JSONDecodeError as e:
            logger.warning(f"Algorithmic JSON repair failed: {e}")
            raise

    def parse_to_lesson(
        self,
        raw_llm_text: str,
        source_images: Optional[list] = None,
        vision_analyzer: Optional[Any] = None,
    ) -> Tuple[Optional[Lesson], Optional[str]]:
        """
        Parse raw LLM output into a validated Lesson Pydantic object.
        Returns (Lesson, None) on success, or (None, error_message) on failure.
        """
        try:
            json_str = self.extract_json_string(raw_llm_text)
        except Exception as e:
            return None, f"JSON文字列の抽出に失敗しました: {str(e)}"

        # Attempt algorithmic parsing and repair
        parsed_dict = None
        try:
            parsed_dict = self.repair_json_string(json_str)
        except Exception as e:
            logger.info("Attempting LLM-based repair if vision analyzer is available...")
            if vision_analyzer:
                try:
                    repaired_text = vision_analyzer.repair_json_with_llm(json_str, str(e))
                    repaired_json_str = self.extract_json_string(repaired_text)
                    parsed_dict = self.repair_json_string(repaired_json_str)
                except Exception as repair_err:
                    return None, f"JSON構文の修復に失敗しました: {str(e)} (LLM修復エラー: {str(repair_err)})"
            else:
                return None, f"JSON構文エラー（無効なフォーマット）: {str(e)}"

        if not isinstance(parsed_dict, dict):
            return None, "解析結果がJSONオブジェクト（辞書）ではありません。"

        # If source images provided, attach to lesson
        if source_images:
            parsed_dict.setdefault("source_images", source_images)

        # Validate with Pydantic
        try:
            lesson = Lesson.model_validate(parsed_dict)
            return lesson, None
        except ValidationError as val_err:
            logger.error(f"Pydantic Validation Error: {val_err}")
            # Format validation errors for human readability
            error_messages = []
            for err in val_err.errors():
                loc = " -> ".join(str(l) for l in err.get("loc", []))
                msg = err.get("msg", "")
                error_messages.append(f"【{loc}】: {msg}")
            
            detailed_err = "構造化データの検証（Validation）に失敗しました:\n" + "\n".join(error_messages)
            return None, detailed_err
        except Exception as e:
            return None, f"Pydantic変換中の予期せぬエラー: {str(e)}"
