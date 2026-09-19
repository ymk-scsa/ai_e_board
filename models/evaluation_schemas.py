"""
Pydantic data schemas for Educational Quality Evaluation and Classroom Operation System (System B).
"""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


class SlideMetrics(BaseModel):
    """Objective quantitative metrics and attention flags for a single slide."""
    slide_number: int = Field(..., description="Slide page number (1-based)")
    title: str = Field(..., description="Slide title")
    slide_type: str = Field("content", description="Slide category badge/type")
    char_count: int = Field(0, description="Total visible character count in slide body")
    formula_count: int = Field(0, description="Count of mathematical formula elements ($...$ or $$...$$)")
    bullet_count: int = Field(0, description="Count of list / bullet items")
    has_problem: bool = Field(False, description="Whether slide contains problem statement")
    has_solution: bool = Field(False, description="Whether slide contains solution/answer")
    density_level: Literal["Low", "Moderate", "High", "Dense"] = Field(
        "Moderate", description="Information density indicator"
    )
    attention_flags: List[str] = Field(
        default_factory=list,
        description="Objective contextual flags (e.g. '文字数多め: 提示順序に配慮', '数式ステップ展開: 発問推奨')",
    )


class PedagogicalEvaluation(BaseModel):
    """Evaluation of instructional structure, flow coherence, and curriculum/objective alignment."""
    structure_score: int = Field(..., ge=0, le=100, description="Instructional structure quality score (0-100)")
    has_introduction: bool = Field(True, description="Presence of introductory context/problem")
    has_clear_objectives: bool = Field(True, description="Presence of explicit learning objectives")
    has_explanation: bool = Field(True, description="Presence of concept explanation/formulas")
    has_example: bool = Field(True, description="Presence of worked example problem")
    has_exercise: bool = Field(True, description="Presence of practice exercise for students")
    has_summary: bool = Field(True, description="Presence of lesson summary")
    flow_coherence_analysis: str = Field(..., description="Qualitative evaluation of lesson narrative and transitions")
    curriculum_alignment_analysis: str = Field(..., description="Alignment with Course of Study standards")


class UsabilityEvaluation(BaseModel):
    """Evaluation of electronic blackboard presentation usability, legibility, and cognitive load."""
    usability_score: int = Field(..., ge=0, le=100, description="Electronic blackboard UX score (0-100)")
    readability_analysis: str = Field(..., description="Font size, contrast, and visibility from back of classroom")
    layout_balance_analysis: str = Field(..., description="Information density and visual hierarchy")
    math_visibility_analysis: str = Field(..., description="Clarity and prominent placement of math expressions")
    slide_split_recommended_slides: List[int] = Field(
        default_factory=list, description="Slide numbers where 2-screen split is recommended for better pacing"
    )


class ImprovementSuggestion(BaseModel):
    """Concrete, actionable improvement suggestion for teachers."""
    id: str = Field(..., description="Unique suggestion identifier")
    target_slide: int = Field(..., description="Target slide number (0 for overall lesson)")
    category: Literal["Pedagogy", "Usability", "CognitiveLoad", "Content"] = Field(
        ..., description="Category of improvement"
    )
    issue: str = Field(..., description="Specific problem or potential obstacle for students")
    rationale: str = Field(..., description="Educational or cognitive rationale for this suggestion")
    concrete_action: str = Field(..., description="Specific, actionable step teacher or system can take")
    priority: Literal["High", "Medium", "Low"] = Field("Medium", description="Recommended priority")


class TeachingGuideItem(BaseModel):
    """Instructional guidance and operation support for classroom delivery."""
    slide_number: int = Field(..., description="Slide page number")
    timing: str = Field(..., description="Recommended timing during lesson (e.g. 導入時, 例題解説時, 机間指導時)")
    key_point_to_emphasize: str = Field(..., description="Core concept or caution to highlight for students")
    questioning_prompt: str = Field(..., description="Concrete question teacher can ask students (発問例)")
    student_activity_guide: str = Field(..., description="Instructions for student notebook work / thinking time")


class CategoryScores(BaseModel):
    """Sub-scores broken down by educational and technical dimensions."""
    pedagogical_structure: int = Field(..., ge=0, le=100, description="Lesson flow & structure (0-100)")
    objective_alignment: int = Field(..., ge=0, le=100, description="Target objective match (0-100)")
    blackboard_usability: int = Field(..., ge=0, le=100, description="16:9 Blackboard UX & visibility (0-100)")
    cognitive_load_balance: int = Field(..., ge=0, le=100, description="Cognitive load & pacing (0-100)")


class EvaluationResult(BaseModel):
    """Comprehensive evaluation package generated by System B."""
    evaluation_id: str = Field(..., description="Unique ID of evaluation report")
    evaluated_at: str = Field(..., description="Timestamp of evaluation (ISO format)")
    input_source_type: Literal["lesson_json", "blackboard_html", "direct_input"] = Field(
        ..., description="Source format evaluated"
    )
    lesson_title: str = Field(..., description="Lesson title")
    unit: str = Field(..., description="Unit or chapter")
    subject: str = Field("数学", description="Subject")
    overall_score: int = Field(..., ge=0, le=100, description="Overall pedagogical quality score (0-100)")
    category_scores: CategoryScores = Field(..., description="Detailed category score breakdown")
    slide_metrics: List[SlideMetrics] = Field(default_factory=list, description="Quantitative metrics per slide")
    pedagogical_evaluation: PedagogicalEvaluation = Field(..., description="Pedagogical structure evaluation")
    usability_evaluation: UsabilityEvaluation = Field(..., description="Blackboard UX evaluation")
    improvements: List[ImprovementSuggestion] = Field(default_factory=list, description="Actionable improvement suggestions")
    teaching_guides: List[TeachingGuideItem] = Field(default_factory=list, description="Classroom delivery guide items")
    strengths: List[str] = Field(default_factory=list, description="Key strengths and positive design points")
    points_for_attention: List[str] = Field(default_factory=list, description="Important points for classroom attention")
    executive_summary: str = Field(..., description="High-level evaluation summary for teacher")
    evaluated_model: Optional[str] = Field(None, description="LLM model used for evaluation")
