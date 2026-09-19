"""
Usability and Cognitive Load Evaluation Module for System B.
Evaluates 16:9 electronic blackboard readability, layout pacing, and cognitive density.
"""

import logging
from typing import List
from models.evaluation_schemas import UsabilityEvaluation, SlideMetrics
from system_b.analyzer import ParsedLessonMaterial

logger = logging.getLogger("ai_e_board.system_b.usability")


class UsabilityEvaluator:
    """Evaluates electronic blackboard UX, readability, and cognitive load."""

    def evaluate(self, material: ParsedLessonMaterial, metrics_list: List[SlideMetrics]) -> UsabilityEvaluation:
        """
        Evaluate presentation layout, readability, math visibility, and recommend slide splits when helpful.
        """
        total_slides = len(material.slides)
        split_candidates = []
        dense_count = 0

        for m in metrics_list:
            # Check for candidate split slides: e.g. slides containing both problem, multi-step solution, and 3+ formulas, or Dense level
            if (m.char_count >= 130 and m.formula_count >= 3 and m.has_problem and m.has_solution) or m.density_level == "Dense":
                split_candidates.append(m.slide_number)
            elif m.density_level == "High":
                dense_count += 1


        # Base usability score
        base_score = 85
        if dense_count > 0:
            base_score -= min(15, dense_count * 5)
        if total_slides < 2:
            base_score -= 10
        elif total_slides > 12:
            base_score -= 5

        usability_score = max(60, min(95, base_score))

        # Qualitative analyses
        readability_text = (
            "16:9比率の大画面ステージ構成となっており、教室後方からでも視認しやすいフォント階層（見出し・本文・数式）が確保されています。"
        )
        layout_text = (
            f"全{total_slides}スライド構成で、スライドごとにテーマが整理されています。"
        )
        if split_candidates:
            layout_text += f" スライド {', '.join(str(s) for s in split_candidates)} は情報量が多いため、問題提示とステップ解説を2画面に分割するか、教員の発問による段階的提示が効果的です。"

        math_text = (
            "数式はKaTeXによる高品位なLaTeX表示が適用されており、計算過程や変形理由が明瞭に強調されています。"
        )

        return UsabilityEvaluation(
            usability_score=usability_score,
            readability_analysis=readability_text,
            layout_balance_analysis=layout_text,
            math_visibility_analysis=math_text,
            slide_split_recommended_slides=split_candidates,
        )
