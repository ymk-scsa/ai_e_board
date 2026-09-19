"""
Material Analyzer module for System B.
Parses Lesson JSON models, raw JSON dicts, or standalone Electronic Blackboard HTML into structured intermediate slides
and computes objective quantitative metrics (character count, formula count, attention flags).
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from models.schemas import Lesson, ElectronicBoardPresentation, Slide
from models.evaluation_schemas import SlideMetrics

logger = logging.getLogger("ai_e_board.system_b.analyzer")


class ParsedSlideContent(BaseModel):
    """Intermediate representation of a single slide extracted from JSON or HTML."""
    slide_number: int = Field(..., description="Slide page number (1-based)")
    badge: str = Field("解説", description="Slide category badge (本時の目標, 公式, 例題, 練習, etc.)")
    title: str = Field(..., description="Slide title")
    body_text: str = Field("", description="Plain text extracted from slide")
    raw_html: str = Field("", description="Raw HTML content of slide body")
    formulas: List[str] = Field(default_factory=list, description="Extracted LaTeX math formulas ($...$ or $$...$$)")
    has_example: bool = Field(False, description="Whether slide contains example problem")
    has_exercise: bool = Field(False, description="Whether slide contains practice exercise")
    speaker_notes: Optional[str] = Field(None, description="Speaker notes if available")


class ParsedLessonMaterial(BaseModel):
    """Normalized educational material representation regardless of source format."""
    source_type: str = Field(..., description="Source format ('lesson_json' or 'blackboard_html')")
    lesson_title: str = Field(..., description="Lesson title")
    subject: str = Field("数学", description="Subject")
    unit: str = Field(..., description="Unit or chapter")
    grade: Optional[str] = Field("高校", description="Grade level")
    learning_objectives: List[str] = Field(default_factory=list, description="Target educational objectives")
    introduction: Optional[str] = Field(None, description="Introductory context")
    summary: Optional[str] = Field(None, description="Lesson summary")
    slides: List[ParsedSlideContent] = Field(default_factory=list, description="Ordered parsed slides")
    raw_source: Optional[Dict[str, Any]] = Field(None, description="Raw source payload if JSON")


class MaterialAnalyzer:
    """Extracts instructional content and quantitative metrics from Lesson JSON or HTML."""

    @classmethod
    def parse_from_lesson(cls, lesson: Lesson, presentation: Optional[ElectronicBoardPresentation] = None) -> ParsedLessonMaterial:
        """Parse structured Lesson schema (and optional presentation) into ParsedLessonMaterial."""
        slides: List[ParsedSlideContent] = []

        # If presentation is already built, use its slides
        if presentation and presentation.slides:
            for s in presentation.slides:
                body_clean, formulas = cls._extract_text_and_formulas(s.content_html)
                slides.append(
                    ParsedSlideContent(
                        slide_number=s.slide_number,
                        badge=s.badge,
                        title=s.title,
                        body_text=body_clean,
                        raw_html=s.content_html,
                        formulas=formulas,
                        has_example="例題" in s.badge or "例題" in s.title or "problem-card" in s.content_html,
                        has_exercise="練習" in s.badge or "問" in s.title,
                        speaker_notes=s.speaker_notes,
                    )
                )
        else:
            # Reconstruct from Lesson sections
            slide_idx = 1
            # 1. Title & Objectives Slide
            obj_text = "\n".join(lesson.learning_objectives)
            slides.append(
                ParsedSlideContent(
                    slide_number=slide_idx,
                    badge="本時の目標",
                    title=lesson.lesson_title,
                    body_text=f"【本時の目標】\n{obj_text}\n{lesson.introduction or ''}",
                    raw_html=f"<div>{obj_text}</div>",
                    formulas=[],
                    has_example=False,
                    has_exercise=False,
                )
            )
            slide_idx += 1

            for sec in lesson.sections:
                sec_formulas = [f.latex for f in sec.formulas]
                sec_text = sec.content
                if sec.example:
                    sec_text += f"\n【例題】{sec.example.problem} 解法: {' '.join(sec.example.solution_steps)} 答: {sec.example.answer}"
                if sec.exercise:
                    sec_text += f"\n【練習】{sec.exercise.problem} 解答: {sec.exercise.answer or ''}"

                slides.append(
                    ParsedSlideContent(
                        slide_number=slide_idx,
                        badge=sec.section_type,
                        title=sec.title,
                        body_text=sec_text,
                        raw_html="",
                        formulas=sec_formulas,
                        has_example=sec.example is not None,
                        has_exercise=sec.exercise is not None,
                    )
                )
                slide_idx += 1

            if lesson.summary:
                slides.append(
                    ParsedSlideContent(
                        slide_number=slide_idx,
                        badge="まとめ",
                        title="本時のまとめ",
                        body_text=lesson.summary,
                        raw_html="",
                        formulas=[],
                        has_example=False,
                        has_exercise=False,
                    )
                )

        return ParsedLessonMaterial(
            source_type="lesson_json",
            lesson_title=lesson.lesson_title,
            subject=lesson.subject,
            unit=lesson.unit,
            grade=lesson.grade,
            learning_objectives=lesson.learning_objectives,
            introduction=lesson.introduction,
            summary=lesson.summary,
            slides=slides,
            raw_source=lesson.model_dump(),
        )

    @classmethod
    def parse_from_html(cls, html_content: str) -> ParsedLessonMaterial:
        """Parse standalone Electronic Blackboard HTML into ParsedLessonMaterial."""
        soup = BeautifulSoup(html_content, "html.parser")

        # Extract title and unit from header/meta
        title_tag = soup.find("title")
        page_title = title_tag.get_text(strip=True) if title_tag else "電子黒板教材"
        lesson_title = page_title.split("-")[0].strip()

        meta_div = soup.find("div", class_="lesson-meta")
        subject = "数学"
        unit = "単元"
        if meta_div:
            spans = meta_div.find_all("span")
            if len(spans) >= 3:
                subject = spans[0].get_text(strip=True)
                unit = spans[2].get_text(strip=True)
            elif spans:
                unit = spans[-1].get_text(strip=True)

        # Extract slides
        slide_divs = soup.find_all("div", class_="slide")
        slides: List[ParsedSlideContent] = []
        objectives: List[str] = []
        intro: Optional[str] = None
        summary: Optional[str] = None

        for idx, s_div in enumerate(slide_divs, start=1):
            badge_tag = s_div.find("span", class_="slide-badge")
            badge = badge_tag.get_text(strip=True) if badge_tag else "解説"

            title_span = s_div.find("div", class_="slide-title")
            title_text = "スライド"
            if title_span:
                # remove badge text
                if badge_tag:
                    badge_tag.extract()
                title_text = title_span.get_text(strip=True)

            body_div = s_div.find("div", class_="slide-body")
            body_html = str(body_div) if body_div else ""
            body_clean, formulas = cls._extract_text_and_formulas(body_html)

            # Check if this slide has objectives or summary
            if "目標" in badge or idx == 1:
                obj_items = s_div.find_all("div", class_="objective-item")
                for oi in obj_items:
                    objectives.append(oi.get_text(strip=True))

            if "まとめ" in badge or "まとめ" in title_text:
                summary = body_clean

            slides.append(
                ParsedSlideContent(
                    slide_number=idx,
                    badge=badge,
                    title=title_text,
                    body_text=body_clean,
                    raw_html=body_html,
                    formulas=formulas,
                    has_example="例題" in badge or "例題" in title_text or "problem-card" in body_html,
                    has_exercise="練習" in badge or "問" in title_text,
                )
            )

        return ParsedLessonMaterial(
            source_type="blackboard_html",
            lesson_title=lesson_title,
            subject=subject,
            unit=unit,
            grade="高校",
            learning_objectives=objectives,
            introduction=intro,
            summary=summary,
            slides=slides,
        )

    @classmethod
    def _extract_text_and_formulas(cls, html_str: str) -> Tuple[str, List[str]]:
        """Clean HTML tags and extract LaTeX formulas ($...$ or $$...$$)."""
        if not html_str:
            return "", []

        # Find all formulas
        display_formulas = re.findall(r"\$\$(.*?)\$\$", html_str, re.DOTALL)
        inline_formulas = re.findall(r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)", html_str)
        all_formulas = [f.strip() for f in display_formulas + inline_formulas if f.strip()]

        # Clean HTML to plain text
        soup = BeautifulSoup(html_str, "html.parser")
        clean_text = soup.get_text(separator=" ", strip=True)

        return clean_text, all_formulas

    @classmethod
    def compute_slide_metrics(cls, slide: ParsedSlideContent) -> SlideMetrics:
        """
        Compute objective quantitative metrics and generate non-punitive attention flags.
        Attention flags serve as objective focal points for the teacher and LLM, not rigid deduction rules.
        """
        char_count = len(slide.body_text)
        formula_count = len(slide.formulas)
        bullet_count = slide.body_text.count("・") + slide.body_text.count("- ") + slide.body_text.count("\n")

        # Determine qualitative density level based on combined characteristics
        if char_count < 80 and formula_count <= 1:
            density_level = "Low"
        elif char_count <= 160 and formula_count <= 3:
            density_level = "Moderate"
        elif char_count <= 260 or formula_count <= 5:
            density_level = "High"
        else:
            density_level = "Dense"

        attention_flags = []
        # Informational flags (objective cues without assuming it is inherently bad)
        if char_count > 180:
            attention_flags.append(f"文字数参考: {char_count}文字（教室後方からの視認性・提示順序に留意）")
        if formula_count >= 3:
            attention_flags.append(f"数式要素多め: {formula_count}式（展開ステップごとの発問や立ち止まりを推奨）")
        if slide.has_example and slide.has_exercise:
            attention_flags.append("例題と練習問題が同一スライドに配置（段階的な提示を検討）")

        return SlideMetrics(
            slide_number=slide.slide_number,
            title=slide.title,
            slide_type=slide.badge,
            char_count=char_count,
            formula_count=formula_count,
            bullet_count=bullet_count,
            has_problem=slide.has_example or slide.has_exercise,
            has_solution="答" in slide.body_text or "解" in slide.body_text,
            density_level=density_level,
            attention_flags=attention_flags,
        )

    @classmethod
    def analyze_material(cls, material: ParsedLessonMaterial) -> List[SlideMetrics]:
        """Compute metrics for all slides in the material."""
        return [cls.compute_slide_metrics(s) for s in material.slides]
