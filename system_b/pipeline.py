"""
Unified Pipeline Facade for System B (Educational Quality Evaluation and Classroom Operation).
Handles all entry points: direct Lesson schemas, raw JSON, and standalone Blackboard HTML.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple

import config
from ai.llm_client import LLMBackend
from models.schemas import Lesson, ElectronicBoardPresentation
from models.evaluation_schemas import EvaluationResult
from system_b.analyzer import MaterialAnalyzer, ParsedLessonMaterial
from system_b.quality_evaluator import QualityEvaluator
from system_b.report import ReportGenerator

logger = logging.getLogger("ai_e_board.system_b.pipeline")


class EvaluationPipeline:
    """Facade orchestrating material ingestion, quality evaluation, and report export."""

    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        llm_backend: Optional[LLMBackend] = None,
    ):
        self.evaluator = QualityEvaluator(host=host, model=model, llm_backend=llm_backend)
        self.report_generator = ReportGenerator()

    def evaluate_from_lesson(
        self,
        lesson: Lesson,
        presentation: Optional[ElectronicBoardPresentation] = None,
        custom_focus: Optional[str] = None,
        use_llm: bool = True,
    ) -> Tuple[EvaluationResult, Path, Path]:
        """Evaluate directly from Lesson schema and optional presentation."""
        material = MaterialAnalyzer.parse_from_lesson(lesson, presentation)
        result = self.evaluator.evaluate_material(material, custom_focus=custom_focus, use_llm=use_llm)
        json_path, html_path = self.report_generator.save_evaluation_outputs(result)
        return result, json_path, html_path

    def evaluate_from_json_string(
        self,
        json_text: str,
        custom_focus: Optional[str] = None,
        use_llm: bool = True,
    ) -> Tuple[EvaluationResult, Path, Path]:
        """Evaluate from raw JSON text (containing Lesson structure or exported presentation)."""
        data = json.loads(json_text)
        if "lesson" in data and isinstance(data["lesson"], dict):
            lesson = Lesson.model_validate(data["lesson"])
        else:
            lesson = Lesson.model_validate(data)
        
        return self.evaluate_from_lesson(lesson, custom_focus=custom_focus, use_llm=use_llm)

    def evaluate_from_html_content(
        self,
        html_content: str,
        custom_focus: Optional[str] = None,
        use_llm: bool = True,
    ) -> Tuple[EvaluationResult, Path, Path]:
        """Evaluate from standalone Electronic Blackboard HTML string."""
        material = MaterialAnalyzer.parse_from_html(html_content)
        result = self.evaluator.evaluate_material(material, custom_focus=custom_focus, use_llm=use_llm)
        json_path, html_path = self.report_generator.save_evaluation_outputs(result)
        return result, json_path, html_path
