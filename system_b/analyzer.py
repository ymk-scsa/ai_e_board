"""
Material Analyzer module for System B.
Parses Lesson JSON models, raw JSON dicts, or standalone Electronic Blackboard HTML into structured intermediate slides
and computes objective quantitative metrics (character count, formula count, attention flags).
"""

import html
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
    def parse_from_lesson(
        cls,
        lesson: Lesson,
        presentation: Optional[ElectronicBoardPresentation] = None,
        source_type: str = "lesson_json",
    ) -> ParsedLessonMaterial:
        """
        Parse a Lesson into ParsedLessonMaterial. Slides always come from System A's generator
        (the given presentation, or one built from the lesson), so a lesson scores the same whether it
        arrives as a Lesson, an exported JSON, or an exported HTML.
        """
        if not presentation or not presentation.slides:
            from ai.generator import ElectronicBoardGenerator  # local import: ai ↔ system_b packages
            presentation = ElectronicBoardGenerator().build_presentation(lesson)

        slides: List[ParsedSlideContent] = []
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
                    has_example=s.slide_type == "example" or (s.slide_type == "intro" and "problem-card" in s.content_html),
                    has_exercise=s.slide_type == "exercise",
                    speaker_notes=s.speaker_notes,
                )
            )

        return ParsedLessonMaterial(
            source_type=source_type,
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
        """
        Parse standalone Electronic Blackboard HTML into ParsedLessonMaterial.
        HTML exported by System A embeds the lesson as JSON; it is used when present so that the HTML is
        evaluated exactly like the lesson. Other HTML is scraped from the slide markup.
        """
        soup = BeautifulSoup(html_content, "html.parser")

        data_tag = soup.find("script", id="lesson-data")
        if data_tag and data_tag.string:
            try:
                lesson = Lesson.model_validate(json.loads(data_tag.string))
                return cls.parse_from_lesson(lesson, source_type="blackboard_html")
            except (ValueError, TypeError) as e:
                logger.warning(f"Embedded lesson data is invalid; falling back to HTML scraping: {e}")

        # Extract title and unit from header/meta ("<title>{lesson title} - AI電子黒板教材</title>")
        title_tag = soup.find("title")
        page_title = title_tag.get_text(strip=True) if title_tag else "電子黒板教材"
        lesson_title = re.sub(r"\s+-\s+AI電子黒板.*$", "", page_title).strip() or page_title

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

            # Objectives / introduction live on the first slide, the summary on the まとめ slide
            if "目標" in badge or idx == 1:
                for oi in s_div.find_all("div", class_="objective-item"):
                    icon = oi.find("span", class_="objective-icon")
                    if icon:
                        icon.extract()  # drop the 🎯 marker
                    objectives.append(oi.get_text(strip=True))
                intro_p = body_div.find("p") if body_div else None
                if intro_p and intro_p.get_text(strip=True):
                    intro = intro_p.get_text(" ", strip=True)

            if "まとめ" in badge or "まとめ" in title_text:
                box = body_div.find("div", class_="objective-box") if body_div else None
                summary = box.get_text(" ", strip=True) if box else body_clean  # teacher notes are outside the box

            slides.append(
                ParsedSlideContent(
                    slide_number=idx,
                    badge=badge,
                    title=title_text,
                    body_text=body_clean,
                    raw_html=body_html,
                    formulas=formulas,
                    has_example="例題" in badge or ("導入" in badge and "problem-card" in body_html),
                    has_exercise="練習" in badge,
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
        # Generated HTML escapes math text (e.g. $a&lt;b$), so unescape entities in extracted formulas
        all_formulas = [html.unescape(f).strip() for f in display_formulas + inline_formulas if f.strip()]

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
        html_items = slide.raw_html.count("<li") + slide.raw_html.count("objective-item")
        bullet_count = html_items if html_items else slide.body_text.count("\n") + slide.body_text.count("・")

        # Qualitative density: each level requires BOTH the text amount and the formula count to stay in range
        if char_count < 80 and formula_count <= 1:
            density_level = "Low"
        elif char_count <= 160 and formula_count <= 3:
            density_level = "Moderate"
        elif char_count <= 260 and formula_count <= 5:
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
            # a shown answer (answer box / "答:" / "解答:"), not any text containing 解 (理解, 解説, ...)
            has_solution="answer-box" in slide.raw_html or bool(re.search(r"(?:^|[\s。])(?:答え?|解答)\s*[:：]", slide.body_text)),
            density_level=density_level,
            attention_flags=attention_flags,
        )

    @classmethod
    def analyze_material(cls, material: ParsedLessonMaterial) -> List[SlideMetrics]:
        """Compute metrics for all slides in the material."""
        return [cls.compute_slide_metrics(s) for s in material.slides]
