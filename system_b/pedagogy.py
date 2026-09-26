"""
Pedagogical Evaluation Module for System B.
Evaluates instructional sequence, objective alignment, worked examples, and curriculum standards.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

from models.evaluation_schemas import PedagogicalEvaluation
from system_b.analyzer import ParsedLessonMaterial
from system_b.curriculum import describe, match_unit

logger = logging.getLogger("ai_e_board.system_b.pedagogy")

# Points per lesson component (sum = 100). No floor: a lesson missing everything scores 0.
COMPONENT_POINTS = {
    "objectives": 15,
    "introduction": 15,
    "explanation": 15,
    "example": 20,
    "exercise": 20,
    "summary": 15,
}
_EXPLANATION_BADGES = ("ポイント", "定義", "公式", "解説")


class PedagogyEvaluator:
    """Evaluates the educational structure, flow coherence, and curriculum alignment of lesson materials."""

    def __init__(self, curriculum_path: Optional[Path] = None):
        self.curriculum_path = curriculum_path  # None → bundled data/curriculum/high_school_math_stub.json

    def check_curriculum_match(self, unit_name: str, subject: str, lesson_title: str = "") -> Tuple[bool, str]:
        """Check whether the unit matches the bundled Course of Study entries."""
        matches = match_unit(unit_name, lesson_title, self.curriculum_path)
        return bool(matches), describe(matches, unit_name)

    def evaluate(self, material: ParsedLessonMaterial) -> PedagogicalEvaluation:
        """
        Evaluate lesson structure from what the material actually contains (objective text, introduction,
        explanation / example / exercise slides, summary). The generator always adds an objective slide,
        so only real objective text counts.
        """
        has_objectives = any(o.strip() for o in material.learning_objectives)
        has_intro = bool(material.introduction and material.introduction.strip()) or any(
            "導入" in s.badge for s in material.slides)
        has_explanation = any(any(b in s.badge for b in _EXPLANATION_BADGES) for s in material.slides)
        has_example = any(s.has_example for s in material.slides)
        has_exercise = any(s.has_exercise for s in material.slides)
        has_summary = bool(material.summary and material.summary.strip())

        present = {"objectives": has_objectives, "introduction": has_intro, "explanation": has_explanation,
                   "example": has_example, "exercise": has_exercise, "summary": has_summary}
        structure_score = sum(COMPONENT_POINTS[k] for k, ok in present.items() if ok)

        is_curriculum_matched, curriculum_msg = self.check_curriculum_match(
            material.unit, material.subject, material.lesson_title)

        flow_parts = []
        if has_objectives:
            flow_parts.append("本時の目標が明示されており、生徒が見通しを持って学習に臨める構成です。")
        else:
            flow_parts.append("本時の目標が明示されていません。授業の冒頭で到達目標を示すと見通しが持ちやすくなります。")
        if has_intro:
            flow_parts.append("導入で学習内容への動機づけが用意されています。")
        if has_example and has_exercise:
            flow_parts.append("例題による解法の確認から練習問題による自力演習へと接続されています。")
        elif has_example:
            flow_parts.append("例題は用意されていますが、生徒の定着を確かめる練習問題がありません。")
        elif has_exercise:
            flow_parts.append("練習問題はありますが、解法を示す例題がありません。")
        else:
            flow_parts.append("例題・練習問題がありません。")
        if has_summary:
            flow_parts.append("授業終盤に本時の重要事項を振り返るまとめが用意されています。")
        missing = [k for k, ok in present.items() if not ok]
        if not missing:
            flow_parts.append("授業の基本要素（目標・導入・解説・例題・練習・まとめ）がすべて揃っています。")

        return PedagogicalEvaluation(
            structure_score=structure_score,
            has_introduction=has_intro,
            has_clear_objectives=has_objectives,
            has_explanation=has_explanation,
            has_example=has_example,
            has_exercise=has_exercise,
            has_summary=has_summary,
            flow_coherence_analysis=" ".join(flow_parts),
            curriculum_alignment_analysis=curriculum_msg,
        )
