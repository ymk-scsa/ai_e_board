"""
Evaluator module providing extensible interfaces for pedagogical quality assessment,
curriculum alignment check, and instructional feedback (Future expansion hooks).
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pathlib import Path

import config
from models.schemas import Lesson, ElectronicBoardPresentation

logger = logging.getLogger("ai_e_board.evaluator")


class BaseLessonEvaluator(ABC):
    """Abstract base evaluator for pedagogical and technical evaluation."""

    @abstractmethod
    def evaluate(self, lesson: Lesson, presentation: Optional[ElectronicBoardPresentation] = None) -> Dict[str, Any]:
        """Perform evaluation and return structured metrics and improvement suggestions."""
        pass


class CurriculumAlignmentEvaluator(BaseLessonEvaluator):
    """Evaluates alignment between lesson content and Course of Study (学習指導要領) database."""

    def __init__(self, curriculum_data_path: Optional[Path] = None):
        self.curriculum_path = curriculum_data_path or (config.CURRICULUM_DIR / "high_school_math_stub.json")
        self.curriculum_db = self._load_curriculum_db()

    def _load_curriculum_db(self) -> Dict[str, Any]:
        if self.curriculum_path.exists():
            try:
                with open(self.curriculum_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load curriculum DB: {e}")
        return {}

    def evaluate(self, lesson: Lesson, presentation: Optional[ElectronicBoardPresentation] = None) -> Dict[str, Any]:
        """Check if unit and key formulas match curriculum standards (Stub implementation for Phase 1)."""
        matched_subjects = []
        curriculum_found = False

        for subject_info in self.curriculum_db.get("subjects", []):
            for unit in subject_info.get("units", []):
                if unit["unit_name"] in lesson.unit or lesson.unit in unit["unit_name"]:
                    matched_subjects.append({
                        "subject": subject_info["subject_name"],
                        "unit": unit["unit_name"],
                        "recommended_topics": unit.get("topics", []),
                        "standard_formulas": unit.get("key_formulas", []),
                    })
                    curriculum_found = True

        return {
            "evaluation_type": "curriculum_alignment",
            "is_aligned": curriculum_found,
            "matched_curriculum_entries": matched_subjects,
            "status": "整合性確認完了" if curriculum_found else "学習指導要領データベース照合（要拡張）",
            "future_extensions": [
                "学習指導要領コード（Guideline Code）との自動マッピング",
                "単元内における既習事項・未習事項の依存関係グラフ解析",
            ]
        }


class PedagogicalQualityEvaluator(BaseLessonEvaluator):
    """Evaluates instructional flow, slide readability, cognitive load, and formula presentation."""

    def evaluate(self, lesson: Lesson, presentation: Optional[ElectronicBoardPresentation] = None) -> Dict[str, Any]:
        """Calculates basic pedagogical quality metrics."""
        has_objectives = len(lesson.learning_objectives) > 0
        has_examples = any(s.example is not None for s in lesson.sections)
        has_exercises = any(s.exercise is not None for s in lesson.sections)
        has_summary = lesson.summary is not None

        score = 0
        if has_objectives: score += 25
        if has_examples: score += 25
        if has_exercises: score += 25
        if has_summary: score += 25

        return {
            "evaluation_type": "pedagogical_quality",
            "completeness_score": score,
            "metrics": {
                "has_clear_objectives": has_objectives,
                "has_example_problems": has_examples,
                "has_practice_exercises": has_exercises,
                "has_lesson_summary": has_summary,
                "total_sections": len(lesson.sections),
                "total_slides": len(presentation.slides) if presentation else 0,
            },
            "suggestions": [
                "例題の解法ステップが明確に記述されているか確認してください。" if has_examples else "例題を追加すると生徒の理解が深まります。",
                "まとめスライドで本時の最重要公式を再度強調すると効果的です。" if has_summary else "授業の最後にまとめを追加することを推奨します。",
            ]
        }
