"""
Data schemas for educational structure extraction and electronic blackboard generation.
"""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


class Formula(BaseModel):
    """Mathematical formula extracted from blackboard with raw text and LaTeX notation."""
    raw_text: str = Field(..., description="Original formula text as written on board")
    latex: str = Field(..., description="Standard LaTeX representation of the formula")
    description: Optional[str] = Field(None, description="Explanation or context of the formula")
    is_key_formula: bool = Field(False, description="Whether this is a core highlight formula in this lesson")


class VisualAnnotation(BaseModel):
    """Visual elements such as boxes, underlines, arrows, colors, and layout indicators."""
    element_type: Literal["box", "underline", "arrow", "color_highlight", "table", "diagram", "other"] = Field(
        "other", description="Type of visual annotation"
    )
    target_text: str = Field(..., description="Text or element being emphasized or linked")
    color: Optional[str] = Field(None, description="Color used (red, yellow, white, blue, etc.) if detectable")
    note: Optional[str] = Field(None, description="Pedagogical meaning (e.g., 'Important definition', 'Key step')")


class ExampleProblem(BaseModel):
    """Example problem (例題) and its step-by-step solution."""
    title: str = Field("例題", description="Title of example (e.g. 例題1, 導入問題)")
    problem: str = Field(..., description="The problem statement")
    approach: Optional[str] = Field(None, description="Thinking process or concept hint (考え方)")
    solution_steps: List[str] = Field(default_factory=list, description="Step-by-step solution expressions/text")
    answer: str = Field(..., description="Final answer or result")
    formulas: List[Formula] = Field(default_factory=list, description="Formulas used in solution")
    teaching_notes: Optional[str] = Field(None, description="Key points teacher emphasizes during explanation")


class Exercise(BaseModel):
    """Practice exercise (練習問題) for students."""
    title: str = Field("練習問題", description="Title of exercise (e.g. 問1, 練習1)")
    problem: str = Field(..., description="The exercise problem statement")
    hint: Optional[str] = Field(None, description="Hint for students (ヒント)")
    answer: Optional[str] = Field(None, description="Answer if written on board, or '要確認/未記載' if omitted")
    solution_steps: List[str] = Field(default_factory=list, description="Solution steps if available")


class BoardElement(BaseModel):
    """Raw structural component found on the board."""
    position: Optional[str] = Field(None, description="Position on board (top-left, center, right-bottom, etc.)")
    content: str = Field(..., description="Textual or formula content")
    category: Literal["title", "objective", "intro", "concept", "formula", "example", "exercise", "summary", "note"] = Field(
        "concept", description="Category of the element"
    )


class Section(BaseModel):
    """Structured educational section within a lesson."""
    section_id: str = Field(..., description="Unique section identifier (e.g. 'sec_intro', 'sec_01')")
    section_type: Literal["introduction", "concept", "definition", "formula", "example", "exercise", "summary", "custom"] = Field(
        ..., description="Type of educational section"
    )
    title: str = Field(..., description="Section heading (e.g. '1. 順列の考え方', '2. 公式の導出')")
    content: str = Field(..., description="Main explanatory text or narrative for this section")
    formulas: List[Formula] = Field(default_factory=list, description="Formulas belonging to this section")
    example: Optional[ExampleProblem] = Field(None, description="Example problem if applicable")
    exercise: Optional[Exercise] = Field(None, description="Practice exercise if applicable")
    visual_annotations: List[VisualAnnotation] = Field(default_factory=list, description="Visual emphasis notes")
    order: int = Field(1, description="Display and instructional sequence order")


class VisualStructure(BaseModel):
    """Overview of visual board layout features extracted by Vision AI."""
    layout_description: Optional[str] = Field(None, description="Description of the 2D layout and flow of the board")
    color_scheme: List[str] = Field(default_factory=list, description="Colors identified on board (white, red, yellow, etc.)")
    diagrams_present: bool = Field(False, description="Whether tree diagrams, graphs, coordinate planes exist")
    table_present: bool = Field(False, description="Whether data tables exist")


class Lesson(BaseModel):
    """Complete lesson structure extracted from blackboard plans."""
    subject: str = Field("数学", description="Subject name (e.g. 数学, 理科, etc.)")
    grade: Optional[str] = Field("高校", description="Target grade/level (e.g. 高校1年, 高校数学I, 高校数学A)")
    unit: str = Field(..., description="Unit or chapter name (e.g. 場合の数と確率 - 順列, 2次関数)")
    lesson_title: str = Field(..., description="Lesson title or theme (e.g. 順列の総数と公式)")
    learning_objectives: List[str] = Field(
        default_factory=list, description="Target educational objectives (本時の目標 / ねらい)"
    )
    introduction: Optional[str] = Field(None, description="Introductory context or motivation (導入)")
    sections: List[Section] = Field(default_factory=list, description="Structured instructional sections")
    summary: Optional[str] = Field(None, description="Final summary or takeaway (まとめ)")
    visual_structure: Optional[VisualStructure] = Field(None, description="Visual layout and annotation details")
    source_images: List[str] = Field(default_factory=list, description="Filenames of source board plan images")
    notes_for_teacher: Optional[str] = Field(None, description="Pedagogical hints or cautionary notes for the teacher")


class Slide(BaseModel):
    """A single presentation slide designed for electronic blackboard display (16:9)."""
    slide_number: int = Field(..., description="Slide page number (1-based)")
    total_slides: int = Field(..., description="Total count of slides in lesson")
    slide_type: Literal["title", "objective", "intro", "concept", "formula", "example", "exercise", "summary"] = Field(
        ..., description="Category of slide"
    )
    badge: str = Field(..., description="Short category badge (e.g. '本時の目標', '公式', '例題', '練習')")
    title: str = Field(..., description="Main heading of the slide")
    content_html: str = Field(..., description="Rendered HTML content with KaTeX math markers")
    speaker_notes: Optional[str] = Field(None, description="Teacher's instructional guidance notes for this slide")
    has_formula: bool = Field(False, description="Whether slide features major mathematical formulas")


class ElectronicBoardPresentation(BaseModel):
    """Complete presentation data model for electronic blackboard UI."""
    lesson_id: str = Field(..., description="Unique ID for this lesson package")
    lesson_title: str = Field(..., description="Lesson title")
    subject: str = Field("数学", description="Subject")
    unit: str = Field(..., description="Unit name")
    slides: List[Slide] = Field(default_factory=list, description="Ordered list of display slides")
    theme: str = Field("chalkboard", description="Visual theme ('chalkboard' or 'whiteboard')")
    created_at: str = Field(..., description="Generation timestamp (ISO format)")
