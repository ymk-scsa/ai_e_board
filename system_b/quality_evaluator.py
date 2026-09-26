"""
Comprehensive Quality Evaluator Module for System B.
Orchestrates quantitative metrics, pedagogical analysis, usability evaluation, and LLM refinement into validated EvaluationResult.
"""

import json
import logging
import re
from difflib import SequenceMatcher
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

from pydantic import BaseModel, Field, ValidationError, field_validator

import config
from ai.llm_client import LLMBackend, OllamaBackend, default_model_for
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
        llm_backend: Optional[LLMBackend] = None,
    ):
        self.host = host or config.OLLAMA_HOST
        self.timeout = timeout
        # LLM補強に使うバックエンド（ai/llm_client.py）。未指定なら Ollama を使う
        # SSRF対策: 許可リスト外のホストにはクライアントを作成しない（LLM補強はスキップされる）
        self.host_error: Optional[str] = None
        self.client: Optional[LLMBackend] = llm_backend
        if self.client is None:
            try:
                self.client = OllamaBackend(host=self.host, timeout=timeout)
            except ValueError as e:
                self.host_error = str(e)
                logger.error(f"Invalid Ollama host rejected: {e}")
        default_model = default_model_for(llm_backend) if llm_backend else config.DEFAULT_EVALUATION_MODEL
        self.model = model or default_model
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
        ped_score = pedagogy_eval.structure_score
        obj_score = self.objective_alignment_score(material)
        usa_score = usability_eval.usability_score
        dense_count = sum(1 for m in metrics_list if m.density_level == "Dense")
        cog_score = max(60, 95 - (dense_count * 10))

        overall_score = int(round((ped_score * 0.35) + (obj_score * 0.25) + (usa_score * 0.25) + (cog_score * 0.15)))

        category_scores = CategoryScores(
            pedagogical_structure=ped_score,
            objective_alignment=obj_score,
            blackboard_usability=usa_score,
            cognitive_load_balance=cog_score,
        )

        # Step 6: Strengths, attention points and summary — derived from the results, not canned
        strengths, points_for_attention = self.describe_results(
            material, metrics_list, pedagogy_eval, usability_eval, obj_score)
        summary_text = self.executive_summary(material.lesson_title, overall_score, pedagogy_eval, usability_eval)

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
            # LLM補強が成功した場合のみモデル名を記録する（_try_llm_enrichment で上書き）
            evaluated_model="deterministic_rules",
        )

        # Step 7: Optional LLM qualitative enrichment if Ollama is accessible
        if use_llm:
            try:
                enriched_result = self._try_llm_enrichment(material, result, custom_focus)
                if enriched_result:
                    return enriched_result
            except Exception as e:
                logger.warning(f"LLM evaluation enrichment failed: {e}. Using deterministic evaluation.")

        return result

    # ------------------------------------------------------------------
    # Rule-based scoring helpers
    # ------------------------------------------------------------------

    @staticmethod
    def objective_alignment_score(material: ParsedLessonMaterial) -> int:
        """0 objectives → 40; objectives present → 75, +10 if they refer to the unit / title, +5 for 2+ items."""
        objectives = [o for o in material.learning_objectives if o.strip()]
        if not objectives:
            return 40
        score = 75
        # the objectives refer to the lesson when they share a 3+ character phrase with the unit / title
        # (Japanese titles have no spaces, so a word split would miss "平方完成による…")
        reference = f"{material.unit} {material.lesson_title}".replace("要確認", "")
        joined = " ".join(objectives)
        match = SequenceMatcher(None, joined, reference, autojunk=False).find_longest_match(0, len(joined), 0, len(reference))
        if match.size >= 3 and joined[match.a:match.a + match.size].strip():
            score += 10
        if len(objectives) >= 2:
            score += 5
        return min(100, score)

    @staticmethod
    def describe_results(material, metrics_list, pedagogy_eval, usability_eval, obj_score):
        strengths: List[str] = []
        attention: List[str] = []
        p = pedagogy_eval
        if p.has_clear_objectives:
            strengths.append("本時の目標が明示され、授業の到達点が共有しやすい" if obj_score >= 85
                             else "本時の目標が示されている")
        else:
            attention.append("本時の目標がありません。冒頭で到達目標を示してください")
        if p.has_example and p.has_exercise:
            strengths.append("例題で解法を示したあと、練習問題で自力演習につなげる流れがある")
        elif p.has_example:
            attention.append("練習問題がありません。例題の直後に類題を1〜2問加えると定着を確認できます")
        elif p.has_exercise:
            attention.append("解法を示す例題がありません。練習の前に1問、解き方を示すと取り組みやすくなります")
        else:
            attention.append("例題・練習問題がありません")
        if p.has_summary:
            strengths.append("まとめで本時の要点を振り返れる")
        else:
            attention.append("まとめがありません。最後に要点を1〜2文で確認してください")
        if not p.has_introduction:
            attention.append("導入（問いかけ・動機づけ）がありません")
        split = usability_eval.slide_split_recommended_slides
        if split:
            attention.append(f"スライド {', '.join(map(str, split))} は情報量が多いため、分割か段階的な提示を検討してください")
        elif metrics_list:
            strengths.append("各スライドの情報量が1画面に収まる範囲に抑えられている")
        formula_slides = [m.slide_number for m in metrics_list if m.formula_count >= 3]
        if formula_slides:
            attention.append(f"数式の多いスライド（{', '.join(map(str, formula_slides))}）では、途中で発問して理解を確認してください")
        return strengths, attention

    @staticmethod
    def executive_summary(title: str, overall: int, pedagogy_eval, usability_eval) -> str:
        if overall >= 85:
            verdict = "授業の基本要素が揃い、電子黒板としての提示にも適した完成度の高い教材です。"
        elif overall >= 70:
            verdict = "おおむね良好な構成です。下記の注意点を確認すると、さらに使いやすくなります。"
        else:
            verdict = "授業の要素が不足しているため、改善提案を参考に補ってください。"
        return f"本教材（{title}）の総合スコアは {overall} 点（100点満点）です。{verdict}"

    @staticmethod
    def build_llm_prompt(material: ParsedLessonMaterial, base_result: EvaluationResult,
                         custom_focus: Optional[str] = None) -> str:
        prompt_file = config.PROMPTS_DIR / "evaluate_quality.txt"
        base_prompt = prompt_file.read_text(encoding="utf-8") if prompt_file.exists() else ""
        metrics = {m.slide_number: m for m in base_result.slide_metrics}
        digest = []
        for s in material.slides:
            m = metrics.get(s.slide_number)
            info = f"（{m.char_count}字・式{m.formula_count}・{m.density_level}）" if m else ""
            excerpt = re.sub(r"\s+", " ", s.body_text)[:60]
            digest.append(f"S{s.slide_number}［{s.badge}］{s.title}{info}: {excerpt}")
        objectives = " / ".join(material.learning_objectives) or "（なし）"
        focus = f"\n【教員の重点】{custom_focus}" if custom_focus else ""
        return (
            f"{base_prompt}\n"
            f"【教材】{material.lesson_title}（{material.unit}）\n【目標】{objectives}\n"
            f"【スライド】\n" + "\n".join(digest) + focus + "\n\n"
            "次のJSONだけを出力してください（総評は120字以内、各項目は60字以内・3項目まで）:\n"
            '{"executive_summary": "総評", "strengths": ["良い点"], "points_for_attention": ["注意点"]}'
        )

    def _try_llm_enrichment(
        self,
        material: ParsedLessonMaterial,
        base_result: EvaluationResult,
        custom_focus: Optional[str] = None,
    ) -> Optional[EvaluationResult]:
        """
        Ask the local LLM for a short qualitative review. The input is a compact one-line-per-slide digest
        and the output is length-limited, because on small GPUs time grows with prompt and answer length.
        """
        if self.client is None:
            raise ConnectionError(self.host_error or "LLMクライアントが初期化されていません。")
        full_prompt = self.build_llm_prompt(material, base_result, custom_focus)
        raw_resp = self.client.generate(self.model, full_prompt, max_tokens=config.EVALUATION_MAX_OUTPUT_TOKENS,
                                        temperature=0.2).text

        # Extract JSON
        match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw_resp)
        json_str = match.group(1) if match else raw_resp[raw_resp.find("{") : raw_resp.rfind("}") + 1]

        if not json_str:
            return None
        # Validate the model output; it may refine the wording only. Scores stay rule-based so that the
        # overall score always agrees with the category scores.
        feedback = LLMFeedback.model_validate(json.loads(json_str))
        if feedback.executive_summary:
            base_result.executive_summary = feedback.executive_summary
        if feedback.strengths:
            base_result.strengths = feedback.strengths
        if feedback.points_for_attention:
            base_result.points_for_attention = feedback.points_for_attention
        base_result.evaluated_model = self.model
        return base_result


class LLMFeedback(BaseModel):
    """
    Accepted shape of the LLM's qualitative feedback. Extra keys (e.g. a score) are ignored; over-long
    text is clipped rather than rejected so that a long but valid answer is not thrown away.
    """
    executive_summary: Optional[str] = None
    strengths: List[str] = Field(default_factory=list)
    points_for_attention: List[str] = Field(default_factory=list)

    @field_validator("executive_summary", mode="before")
    @classmethod
    def _clip_summary(cls, v):
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("must be a string")
        return v.strip()[:300] or None

    @field_validator("strengths", "points_for_attention", mode="before")
    @classmethod
    def _non_empty_strings(cls, v):
        """Keep non-empty strings only (small models sometimes emit numbers or blanks); a non-list is invalid."""
        if not isinstance(v, list):
            raise ValueError("must be a list")
        return [s.strip()[:120] for s in v if isinstance(s, str) and s.strip()][:5]
