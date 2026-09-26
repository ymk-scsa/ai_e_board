"""
Evaluator module providing extensible interfaces for pedagogical quality assessment,
curriculum alignment check, and instructional feedback (Future expansion hooks).
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path

from models.schemas import Lesson, ElectronicBoardPresentation
from system_b.curriculum import describe, match_unit

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
        self.curriculum_path = curriculum_data_path  # None → data/curriculum/high_school_math_stub.json

    def evaluate(self, lesson: Lesson, presentation: Optional[ElectronicBoardPresentation] = None) -> Dict[str, Any]:
        """Match the lesson's unit / title against the bundled Course of Study data (system_b/curriculum.py)."""
        matches = match_unit(lesson.unit, lesson.lesson_title, self.curriculum_path)
        matched_subjects = [
            {"subject": m.subject, "unit": m.unit, "recommended_topics": m.topics, "standard_formulas": m.key_formulas}
            for m in matches
        ]
        curriculum_found = bool(matches)
        return {
            "evaluation_type": "curriculum_alignment",
            "is_aligned": curriculum_found,
            "matched_curriculum_entries": matched_subjects,
            "message": describe(matches, lesson.unit),
            "status": "整合性確認完了" if curriculum_found else "該当単元なし（収録範囲外または単元名未確定）",
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
        has_summary = bool(lesson.summary and lesson.summary.strip())

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
