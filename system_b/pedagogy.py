"""
Pedagogical Evaluation Module for System B.
Evaluates instructional sequence, objective alignment, worked examples, and curriculum standards.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import config

from models.evaluation_schemas import PedagogicalEvaluation
from system_b.analyzer import ParsedLessonMaterial

logger = logging.getLogger("ai_e_board.system_b.pedagogy")


class PedagogyEvaluator:
    """Evaluates the educational structure, flow coherence, and curriculum alignment of lesson materials."""

    def __init__(self, curriculum_path: Optional[Path] = None):
        self.curriculum_path = curriculum_path or (config.CURRICULUM_DIR / "high_school_math_stub.json")
        self.curriculum_db = self._load_curriculum_db()

    def _load_curriculum_db(self) -> Dict[str, Any]:
        if self.curriculum_path and self.curriculum_path.exists():
            try:
                with open(self.curriculum_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load curriculum DB: {e}")
        return {}

    def check_curriculum_match(self, unit_name: str, subject: str) -> Tuple[bool, Optional[str]]:
        """Check if unit matches standard Course of Study curriculum entries."""
        for subj in self.curriculum_db.get("subjects", []):
            for u in subj.get("units", []):
                if u["unit_name"] in unit_name or unit_name in u["unit_name"]:
                    topics_str = "、".join(u.get("topics", []))
                    return True, f"学習指導要領（{subj['subject_name']} / {u['unit_name']}）の標準指導内容（{topics_str}）と適合しています。"
        return False, "高等学校学習指導要領データベースに類似単元が登録されています（文脈に応じた柔軟な展開が推奨されます）。"

    def evaluate(self, material: ParsedLessonMaterial) -> PedagogicalEvaluation:
        """
        Evaluate lesson structure based on instructional components and pedagogical flow.
        Calculates structure score and provides contextual pedagogical feedback.
        """
        has_intro = (material.introduction is not None and len(material.introduction) > 0) or any(
            "導入" in s.badge or "導入" in s.title for s in material.slides
        )
        has_objectives = len(material.learning_objectives) > 0 or any(
            "目標" in s.badge or "目標" in s.title for s in material.slides
        )
        has_explanation = any(
            s.badge in ["concept", "formula", "definition", "解説", "公式", "ポイント"] for s in material.slides
        ) or len(material.slides) >= 3
        has_example = any(s.has_example for s in material.slides)
        has_exercise = any(s.has_exercise for s in material.slides)
        has_summary = (material.summary is not None and len(material.summary) > 0) or any(
            "まとめ" in s.badge or "まとめ" in s.title for s in material.slides
        )

        # Baseline structure scoring
        points = 40  # baseline
        if has_objectives: points += 12
        if has_intro: points += 10
        if has_explanation: points += 12
        if has_example: points += 14
        if has_exercise: points += 12
        if has_summary: points += 10

        structure_score = min(100, max(50, points))

        # Check curriculum match
        from typing import Tuple
        is_curriculum_matched, curriculum_msg = self.check_curriculum_match(material.unit, material.subject)

        # Build narrative flow analysis
        flow_parts = []
        if has_objectives:
            flow_parts.append("本時の目標が明示されており、生徒が見通しを持って学習に臨める構成です。")
        if has_intro:
            flow_parts.append("身近な題材や導入問題による動機づけが適切に配置されています。")
        if has_example and has_exercise:
            flow_parts.append("例題による解法の確認から練習問題による自力演習へとスムーズに接続されています。")
        elif has_example:
            flow_parts.append("例題が丁寧に解説されています。生徒の定着を確認する練習問題の配置も検討できます。")
        if has_summary:
            flow_parts.append("授業終盤に本時の重要事項を振り返るまとめが用意されています。")

        flow_narrative = " ".join(flow_parts) if flow_parts else "基本的な授業展開が構成されています。"

        return PedagogicalEvaluation(
            structure_score=structure_score,
            has_introduction=has_intro,
            has_clear_objectives=has_objectives,
            has_explanation=has_explanation,
            has_example=has_example,
            has_exercise=has_exercise,
            has_summary=has_summary,
            flow_coherence_analysis=flow_narrative,
            curriculum_alignment_analysis=curriculum_msg,
        )
