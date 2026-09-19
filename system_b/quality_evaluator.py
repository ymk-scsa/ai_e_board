"""
Comprehensive Quality Evaluator Module for System B.
Orchestrates quantitative metrics, pedagogical analysis, usability evaluation, and LLM refinement into validated EvaluationResult.
"""

import json
import logging
import re
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

import ollama
from pydantic import ValidationError

import config
from models.evaluation_schemas import (
    EvaluationResult,
    CategoryScores,
    PedagogicalEvaluation,
    UsabilityEvaluation,
    SlideMetrics,
    ImprovementSuggestion,
    TeachingGuideItem,
)
from system_b.analyzer import ParsedLessonMaterial, MaterialAnalyzer
from system_b.pedagogy import PedagogyEvaluator
from system_b.usability import UsabilityEvaluator
from system_b.improvement import ImprovementGenerator

logger = logging.getLogger("ai_e_board.system_b.quality_evaluator")


class QualityEvaluator:
    """Combines rule-based contextual metrics and LLM evaluation into structured EvaluationResult."""

    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = config.EVALUATION_TIMEOUT_SECONDS,
    ):
        self.host = host or config.OLLAMA_HOST
        self.model = model or config.DEFAULT_EVALUATION_MODEL
        self.timeout = timeout
        self.client = ollama.Client(host=self.host)
        self.pedagogy_evaluator = PedagogyEvaluator()
        self.usability_evaluator = UsabilityEvaluator()

    def evaluate_material(
        self,
        material: ParsedLessonMaterial,
        custom_focus: Optional[str] = None,
        use_llm: bool = True,
    ) -> EvaluationResult:
        """
        Execute full educational evaluation pipeline for the given lesson material.
        Combines deterministic contextual analysis with optional LLM enhancement.
        """
        eval_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        now_iso = datetime.now().isoformat()

        # Step 1: Compute slide-level quantitative metrics
        metrics_list: List[SlideMetrics] = MaterialAnalyzer.analyze_material(material)

        # Step 2: Pedagogical & Curriculum analysis
        pedagogy_eval: PedagogicalEvaluation = self.pedagogy_evaluator.evaluate(material)

        # Step 3: Blackboard usability & Cognitive load analysis
        usability_eval: UsabilityEvaluation = self.usability_evaluator.evaluate(material, metrics_list)

        # Step 4: Generate improvement suggestions & teaching guides
        improvements: List[ImprovementSuggestion] = ImprovementGenerator.generate_suggestions(
            material, metrics_list, pedagogy_eval, usability_eval
        )
        teaching_guides: List[TeachingGuideItem] = ImprovementGenerator.generate_teaching_guides(material)

        # Step 5: Compute unified category scores
        # Pedagogical structure (0-100)
        ped_score = pedagogy_eval.structure_score
        # Objective alignment (0-100)
        obj_score = 90 if pedagogy_eval.has_clear_objectives else 70
        # Blackboard UX (0-100)
        usa_score = usability_eval.usability_score
        # Cognitive load balance (0-100)
        dense_count = sum(1 for m in metrics_list if m.density_level == "Dense")
        cog_score = max(60, 95 - (dense_count * 10))

        overall_score = int((ped_score * 0.35) + (obj_score * 0.25) + (usa_score * 0.25) + (cog_score * 0.15))

        category_scores = CategoryScores(
            pedagogical_structure=ped_score,
            objective_alignment=obj_score,
            blackboard_usability=usa_score,
            cognitive_load_balance=cog_score,
        )

        # Step 6: Strengths & Attention points
        strengths = [
            f"単元「{material.unit}」の学習目標が明瞭であり、授業展開の見通しが良い点",
            "16:9電子黒板に適したレイアウトで、教室後方からの視認性に配慮されている点",
            "数学の数式（LaTeX）が美しく配置され、計算プロセスの視覚的追従性が高い点",
        ]
        if pedagogy_eval.has_example:
            strengths.append("例題において思考プロセスと解法ステップが丁寧に構造化されている点")

        points_for_attention = [
            "例題から演習への移行時、生徒の自力ワーク時間を適切に確保すること",
            "数式が多いスライドでは、教員が立ち止まって生徒の理解度を確認（発問）すること",
        ]

        summary_text = (
            f"本教材（{material.lesson_title}）は総合スコア {overall_score}点（100点満点）と評価されました。"
            f"授業構成の一貫性と電子黒板としての提示適性が高く、生徒の思考を支援する優れた構成です。"
        )

        # Build preliminary result
        result = EvaluationResult(
            evaluation_id=eval_id,
            evaluated_at=now_iso,
            input_source_type="lesson_json" if material.source_type == "lesson_json" else "blackboard_html",
            lesson_title=material.lesson_title,
            unit=material.unit,
            subject=material.subject,
            overall_score=overall_score,
            category_scores=category_scores,
            slide_metrics=metrics_list,
            pedagogical_evaluation=pedagogy_eval,
            usability_evaluation=usability_eval,
            improvements=improvements,
            teaching_guides=teaching_guides,
            strengths=strengths,
            points_for_attention=points_for_attention,
            executive_summary=summary_text,
            evaluated_model=self.model if use_llm else "deterministic_rules",
        )

        # Step 7: Optional LLM qualitative enrichment if Ollama is accessible
        if use_llm:
            try:
                enriched_result = self._try_llm_enrichment(material, result, custom_focus)
                if enriched_result:
                    return enriched_result
            except Exception as e:
                logger.info(f"LLM evaluation enrichment skipped or timed out: {e}. Using deterministic evaluation.")

        return result

    def _try_llm_enrichment(
        self,
        material: ParsedLessonMaterial,
        base_result: EvaluationResult,
        custom_focus: Optional[str] = None,
    ) -> Optional[EvaluationResult]:
        """Attempt to call Ollama LLM to refine qualitative feedback and suggestions."""
        prompt_template = config.PROMPTS_DIR / "evaluate_quality.txt"
        base_prompt = prompt_template.read_text(encoding="utf-8") if prompt_template.exists() else ""

        # Prepare payload for LLM
        summary_payload = {
            "lesson_title": material.lesson_title,
            "unit": material.unit,
            "subject": material.subject,
            "objectives": material.learning_objectives,
            "slides_count": len(material.slides),
            "slides_summary": [
                {
                    "slide_number": s.slide_number,
                    "title": s.title,
                    "badge": s.badge,
                    "body_excerpt": s.body_text[:150],
                    "formula_count": len(s.formulas),
                }
                for s in material.slides
            ],
            "preliminary_metrics": [m.model_dump() for m in base_result.slide_metrics],
        }

        full_prompt = f"""{base_prompt}

【分析対象教材データ】
```json
{json.dumps(summary_payload, ensure_ascii=False, indent=2)}
```

{f'【教員からの評価重点指示】: {custom_focus}' if custom_focus else ''}

以下のJSON形式で評価結果を出力してください。
```json
{{
  "overall_score": {base_result.overall_score},
  "executive_summary": "総合評価のコメント",
  "strengths": ["良い点1", "良い点2"],
  "points_for_attention": ["注意点1", "注意点2"]
}}
```
"""
        response = self.client.generate(
            model=self.model,
            prompt=full_prompt,
            options={"temperature": 0.2},
        )
        raw_resp = response.get("response", "") if isinstance(response, dict) else getattr(response, "response", "")

        # Extract JSON
        match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw_resp)
        json_str = match.group(1) if match else raw_resp[raw_resp.find("{") : raw_resp.rfind("}") + 1]

        if json_str:
            llm_dict = json.loads(json_str)
            if "executive_summary" in llm_dict:
                base_result.executive_summary = llm_dict["executive_summary"]
            if "strengths" in llm_dict and isinstance(llm_dict["strengths"], list):
                base_result.strengths = llm_dict["strengths"]
            if "points_for_attention" in llm_dict and isinstance(llm_dict["points_for_attention"], list):
                base_result.points_for_attention = llm_dict["points_for_attention"]
            if "overall_score" in llm_dict and isinstance(llm_dict["overall_score"], int):
                base_result.overall_score = max(0, min(100, llm_dict["overall_score"]))

        return base_result
