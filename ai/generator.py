"""
Generator module for creating 16:9 electronic blackboard presentation materials from structured Lesson data.
"""

import html
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from jinja2 import Environment

import config
from models.schemas import (
    Lesson,
    Slide,
    ElectronicBoardPresentation,
    Section,
    Formula,
    ExampleProblem,
    Exercise,
)

logger = logging.getLogger("ai_e_board.generator")


def _e(value: Any) -> str:
    """HTML-escape LLM/teacher-derived text before embedding it into generated HTML.
    KaTeX auto-render reads textContent, so escaped math (e.g. $a&lt;b$) still renders correctly."""
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


HTML_PRESENTATION_TEMPLATE = """<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ presentation.lesson_title }} - AI電子黒板教材</title>
    
    <!-- KaTeX for Beautiful Math Rendering -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
        onload="renderMathInElement(document.body, {
            delimiters: [
                {left: '$$', right: '$$', display: true},
                {left: '$', right: '$', display: false}
            ],
            throwOnError: false
        });"></script>
    
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=BIZ+UDPGothic:wght@400;700&family=Yuji+Boku&family=Noto+Sans+JP:wght@400;600;800&display=swap" rel="stylesheet">

    <style>
        :root {
            --ratio-width: 16;
            --ratio-height: 9;
        }

        /* Chalkboard Theme (Default) */
        body.theme-chalkboard {
            --bg-color: #1a3328;
            --bg-frame: #3a2312;
            --text-main: #f0f7f4;
            --text-sub: #d0ded6;
            --chalk-yellow: #fce38a;
            --chalk-pink: #f38181;
            --chalk-blue: #a8d8ea;
            --card-bg: rgba(255, 255, 255, 0.08);
            --card-border: rgba(255, 255, 255, 0.2);
            --badge-bg: #fce38a;
            --badge-text: #1a3328;
            --highlight-box-bg: rgba(252, 227, 138, 0.15);
            --highlight-box-border: #fce38a;
        }

        /* Whiteboard Theme */
        body.theme-whiteboard {
            --bg-color: #f8fafc;
            --bg-frame: #94a3b8;
            --text-main: #0f172a;
            --text-sub: #334155;
            --chalk-yellow: #b45309;
            --chalk-pink: #dc2626;
            --chalk-blue: #0284c7;
            --card-bg: #ffffff;
            --card-border: #cbd5e1;
            --badge-bg: #0284c7;
            --badge-text: #ffffff;
            --highlight-box-bg: #e0f2fe;
            --highlight-box-border: #0284c7;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            user-select: none;
        }

        body {
            font-family: 'BIZ UDPGothic', 'Noto Sans JP', sans-serif;
            background-color: #0b0f14;
            color: var(--text-main);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            overflow: hidden;
        }

        /* 16:9 Aspect Ratio Presentation Stage */
        .stage-container {
            width: 100vw;
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 10px;
        }

        .board-stage {
            width: min(100vw - 20px, calc((100vh - 20px) * 16 / 9));
            height: min(calc((100vw - 20px) * 9 / 16), 100vh - 20px);
            background-color: var(--bg-color);
            border: 14px solid var(--bg-frame);
            border-radius: 12px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.8), inset 0 0 100px rgba(0, 0, 0, 0.4);
            display: flex;
            flex-direction: column;
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
        }

        /* Header / Meta Bar */
        .board-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 18px 36px 10px;
            border-bottom: 2px dashed rgba(255, 255, 255, 0.15);
        }

        .lesson-meta {
            font-size: 1.25rem;
            color: var(--text-sub);
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .slide-badge {
            background-color: var(--badge-bg);
            color: var(--badge-text);
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 1.15rem;
            font-weight: 800;
            letter-spacing: 0.05em;
        }

        .theme-toggle-btn, .fullscreen-btn {
            background: rgba(255, 255, 255, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.3);
            color: var(--text-main);
            padding: 6px 14px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1rem;
            margin-left: 8px;
            transition: background 0.2s;
        }

        .theme-toggle-btn:hover, .fullscreen-btn:hover {
            background: rgba(255, 255, 255, 0.25);
        }

        /* Slide Content Area */
        .slides-wrapper {
            flex: 1;
            position: relative;
            width: 100%;
            height: 100%;
        }

        .slide {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            padding: 30px 50px;
            display: none;
            flex-direction: column;
            justify-content: flex-start;
            opacity: 0;
            transition: opacity 0.25s ease-in-out;
            overflow-y: auto;
        }

        .slide.active {
            display: flex;
            opacity: 1;
        }

        .slide-title {
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 24px;
            color: var(--chalk-yellow);
            line-height: 1.3;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.5);
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .slide-body {
            font-size: 1.8rem;
            line-height: 1.8;
            color: var(--text-main);
            flex: 1;
        }

        /* Structured Elements */
        .objective-box {
            background: var(--highlight-box-bg);
            border: 3px solid var(--highlight-box-border);
            border-radius: 12px;
            padding: 24px 32px;
            margin: 20px 0;
        }

        .objective-item {
            font-size: 2.0rem;
            font-weight: 700;
            margin: 14px 0;
            display: flex;
            align-items: baseline;
            gap: 14px;
        }

        .objective-icon {
            color: var(--chalk-yellow);
            font-size: 1.8rem;
        }

        .formula-highlight {
            background: rgba(0, 0, 0, 0.25);
            border: 3px solid var(--chalk-pink);
            border-radius: 12px;
            padding: 24px;
            margin: 24px 0;
            text-align: center;
            font-size: 2.6rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }

        .formula-desc {
            font-size: 1.4rem;
            color: var(--text-sub);
            margin-top: 10px;
            text-align: center;
        }

        .problem-card {
            background: var(--card-bg);
            border: 2px solid var(--card-border);
            border-radius: 12px;
            padding: 22px 28px;
            margin-bottom: 20px;
        }

        .problem-header {
            font-size: 1.6rem;
            font-weight: 800;
            color: var(--chalk-blue);
            margin-bottom: 12px;
        }

        .problem-text {
            font-size: 1.85rem;
            font-weight: 600;
            line-height: 1.6;
            margin-bottom: 16px;
        }

        .step-list {
            margin: 16px 0;
            padding-left: 20px;
        }

        .step-item {
            font-size: 1.7rem;
            margin: 10px 0;
            line-height: 1.6;
        }

        .answer-box {
            background: var(--highlight-box-bg);
            border-left: 8px solid var(--chalk-yellow);
            padding: 14px 20px;
            font-size: 2.1rem;
            font-weight: 800;
            color: var(--chalk-yellow);
            margin-top: 16px;
            border-radius: 4px;
        }

        .visual-note-tag {
            display: inline-block;
            background: rgba(255, 255, 255, 0.15);
            border-radius: 6px;
            padding: 4px 12px;
            font-size: 1.2rem;
            margin-right: 8px;
            color: var(--chalk-yellow);
        }

        /* Footer / Navigation Control Bar */
        .board-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 36px 16px;
            background: rgba(0, 0, 0, 0.2);
            border-top: 1px solid rgba(255, 255, 255, 0.15);
        }

        .nav-btn {
            background: rgba(255, 255, 255, 0.15);
            border: 2px solid rgba(255, 255, 255, 0.4);
            color: var(--text-main);
            padding: 10px 28px;
            border-radius: 8px;
            font-size: 1.4rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .nav-btn:hover:not(:disabled) {
            background: rgba(255, 255, 255, 0.35);
            transform: translateY(-2px);
        }

        .nav-btn:disabled {
            opacity: 0.3;
            cursor: not-allowed;
        }

        .slide-counter {
            font-size: 1.6rem;
            font-weight: 800;
            color: var(--chalk-yellow);
            letter-spacing: 0.1em;
        }

        .progress-bar-container {
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.1);
        }

        .progress-bar-fill {
            height: 100%;
            background: var(--chalk-yellow);
            transition: width 0.3s ease;
        }

        .katex {
            font-size: 1.25em !important;
        }
    </style>
</head>
<body class="theme-{{ presentation.theme }}">

<div class="stage-container">
    <div class="board-stage" id="boardStage">
        <!-- Top Bar -->
        <div class="board-header">
            <div class="lesson-meta">
                <span>{{ presentation.subject }}</span>
                <span>•</span>
                <span>{{ presentation.unit }}</span>
            </div>
            <div>
                <button class="theme-toggle-btn" onclick="toggleTheme()">🎨 テーマ切替</button>
                <button class="fullscreen-btn" onclick="toggleFullScreen()">⛶ 全画面</button>
            </div>
        </div>

        <!-- Slides Wrapper -->
        <div class="slides-wrapper" onclick="handleStageClick(event)">
            {% for slide in presentation.slides %}
            <div class="slide {% if loop.first %}active{% endif %}" id="slide-{{ loop.index }}" data-index="{{ loop.index }}">
                <div class="slide-title">
                    <span class="slide-badge">{{ slide.badge }}</span>
                    <span>{{ slide.title }}</span>
                </div>
                <div class="slide-body">
                    {{ slide.content_html | safe }}
                </div>
            </div>
            {% endfor %}
        </div>

        <!-- Footer / Controls -->
        <div class="board-footer">
            <button class="nav-btn" id="prevBtn" onclick="prevSlide()">◀ 前へ</button>
            <div class="slide-counter">
                <span id="currentSlideNum">1</span> / <span id="totalSlideNum">{{ presentation.slides|length }}</span>
            </div>
            <button class="nav-btn" id="nextBtn" onclick="nextSlide()">次へ ▶</button>
        </div>

        <!-- Bottom Progress Line -->
        <div class="progress-bar-container">
            <div class="progress-bar-fill" id="progressFill" style="width: {{ (1 / presentation.slides|length) * 100 }}%;"></div>
        </div>
    </div>
</div>

<script>
    let currentSlide = 1;
    const totalSlides = {{ presentation.slides|length }};

    function updateSlide() {
        // Update slide visibility
        document.querySelectorAll('.slide').forEach((el, idx) => {
            if (idx + 1 === currentSlide) {
                el.classList.add('active');
            } else {
                el.classList.remove('active');
            }
        });

        // Update indicators
        document.getElementById('currentSlideNum').innerText = currentSlide;
        document.getElementById('prevBtn').disabled = (currentSlide === 1);
        document.getElementById('nextBtn').disabled = (currentSlide === totalSlides);
        
        const progressPct = (currentSlide / totalSlides) * 100;
        document.getElementById('progressFill').style.width = progressPct + '%';

        // Re-render KaTeX in newly active slide
        if (window.renderMathInElement) {
            renderMathInElement(document.getElementById('slide-' + currentSlide), {
                delimiters: [
                    {left: '$$', right: '$$', display: true},
                    {left: '$', right: '$', display: false}
                ],
                throwOnError: false
            });
        }
    }

    function nextSlide() {
        if (currentSlide < totalSlides) {
            currentSlide++;
            updateSlide();
        }
    }

    function prevSlide() {
        if (currentSlide > 1) {
            currentSlide--;
            updateSlide();
        }
    }

    function handleStageClick(event) {
        // If clicking on buttons or interactive items, ignore
        if (event.target.closest('button')) return;
        
        const clickX = event.clientX;
        const width = window.innerWidth;
        if (clickX > width * 0.6) {
            nextSlide();
        } else if (clickX < width * 0.4) {
            prevSlide();
        }
    }

    // Keyboard Navigation: Left/Right arrows, Space, PageUp/PageDown, F11
    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') {
            e.preventDefault();
            nextSlide();
        } else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {
            e.preventDefault();
            prevSlide();
        }
    });

    function toggleFullScreen() {
        const elem = document.getElementById('boardStage');
        if (!document.fullscreenElement) {
            if (elem.requestFullscreen) {
                elem.requestFullscreen();
            } else if (elem.webkitRequestFullscreen) {
                elem.webkitRequestFullscreen();
            }
        } else {
            if (document.exitFullscreen) {
                document.exitFullscreen();
            }
        }
    }

    function toggleTheme() {
        const body = document.body;
        if (body.classList.contains('theme-chalkboard')) {
            body.classList.remove('theme-chalkboard');
            body.classList.add('theme-whiteboard');
        } else {
            body.classList.remove('theme-whiteboard');
            body.classList.add('theme-chalkboard');
        }
    }

    // Initial trigger
    window.addEventListener('DOMContentLoaded', () => {
        updateSlide();
    });
</script>

</body>
</html>
"""


class ElectronicBoardGenerator:
    """Transforms validated Lesson schema into presentation slides and exportable HTML/JSON."""

    def __init__(self, theme: str = config.DEFAULT_THEME):
        self.theme = theme
        # autoescape=True: template variables are escaped; slide.content_html is pre-escaped via _e() and marked |safe
        self.template = Environment(autoescape=True).from_string(HTML_PRESENTATION_TEMPLATE)

    def generate_slides_from_lesson(self, lesson: Lesson) -> List[Slide]:
        """Convert a Lesson model into an ordered list of 16:9 slides."""
        slides: List[Slide] = []
        slide_num = 1

        # 1. Slide: Lesson Title & Objectives
        obj_items = "".join(
            f'<div class="objective-item"><span class="objective-icon">🎯</span><span>{_e(obj)}</span></div>'
            for obj in lesson.learning_objectives
        )
        if not obj_items:
            obj_items = '<div class="objective-item"><span>本時の学習内容を理解する</span></div>'

        intro_text = ""
        if lesson.introduction:
            intro_text = f'<p style="margin-top: 18px; font-size: 1.6rem; color: var(--text-sub);">{_e(lesson.introduction)}</p>'

        slides.append(
            Slide(
                slide_number=slide_num,
                total_slides=0,  # updated later
                slide_type="title",
                badge="本時の目標",
                title=lesson.lesson_title,
                content_html=f"""
                <div class="objective-box">
                    <div style="font-size: 1.5rem; color: var(--chalk-yellow); font-weight: 800; margin-bottom: 12px;">【学習のねらい】</div>
                    {obj_items}
                </div>
                {intro_text}
                """,
                speaker_notes="本時の目標を提示し、生徒に学習の見通しを持たせます。",
            )
        )
        slide_num += 1

        # 2. Iterate through sections
        for sec in sorted(lesson.sections, key=lambda s: s.order):
            content_html_parts = []

            # Visual annotations tag
            if sec.visual_annotations:
                tags = "".join(
                    f'<span class="visual-note-tag">📌 {_e(va.target_text)} ({_e(va.note or va.element_type)})</span>'
                    for va in sec.visual_annotations
                )
                content_html_parts.append(f'<div style="margin-bottom: 14px;">{tags}</div>')

            # Main narrative content
            if sec.content:
                content_html_parts.append(
                    f'<p style="font-size: 1.8rem; margin-bottom: 20px; line-height: 1.7;">{_e(sec.content)}</p>'
                )

            # Formulas
            for form in sec.formulas:
                content_html_parts.append(
                    f"""
                    <div class="formula-highlight">
                        $${_e(form.latex)}$$
                        {f'<div class="formula-desc">{_e(form.description)}</div>' if form.description else ''}
                    </div>
                    """
                )

            # Example problem
            if sec.example:
                ex = sec.example
                steps_html = "".join(
                    f'<li class="step-item">{_e(st)}</li>' for st in ex.solution_steps
                )
                approach_html = (
                    f'<div style="color: var(--chalk-yellow); font-size: 1.5rem; margin-bottom: 10px;">💡 <b>考え方:</b> {_e(ex.approach)}</div>'
                    if ex.approach
                    else ""
                )
                content_html_parts.append(
                    f"""
                    <div class="problem-card">
                        <div class="problem-header">📝 {_e(ex.title)}</div>
                        <div class="problem-text">{_e(ex.problem)}</div>
                        {approach_html}
                        {f'<div style="font-size: 1.5rem; font-weight: bold; margin-top: 10px;">【解法】</div><ol class="step-list">{steps_html}</ol>' if steps_html else ''}
                        <div class="answer-box">答: {_e(ex.answer)}</div>
                    </div>
                    """
                )

            # Practice exercise
            if sec.exercise:
                exe = sec.exercise
                hint_html = (
                    f'<div style="color: var(--chalk-blue); font-size: 1.4rem; margin-top: 10px;">💬 <b>ヒント:</b> {_e(exe.hint)}</div>'
                    if exe.hint
                    else ""
                )
                ans_html = (
                    f'<div class="answer-box" style="font-size: 1.7rem;">解答: {_e(exe.answer)}</div>'
                    if exe.answer and exe.answer != "要確認/未記載"
                    else '<div style="margin-top: 14px; font-size: 1.4rem; color: var(--text-sub);">※ 解答はノートに記入しましょう</div>'
                )
                content_html_parts.append(
                    f"""
                    <div class="problem-card" style="border-color: var(--chalk-blue);">
                        <div class="problem-header" style="color: var(--chalk-yellow);">✏️ {_e(exe.title)}</div>
                        <div class="problem-text">{_e(exe.problem)}</div>
                        {hint_html}
                        {ans_html}
                    </div>
                    """
                )

            badge_label = {
                "introduction": "導入",
                "concept": "ポイント",
                "definition": "定義",
                "formula": "公式",
                "example": "例題",
                "exercise": "練習",
                "summary": "まとめ",
            }.get(sec.section_type, "解説")

            slides.append(
                Slide(
                    slide_number=slide_num,
                    total_slides=0,
                    slide_type={"introduction": "intro", "formula": "formula", "example": "example",
                                "exercise": "exercise", "summary": "summary"}.get(sec.section_type, "concept"),
                    badge=badge_label,
                    title=sec.title,
                    content_html="".join(content_html_parts),
                    has_formula=len(sec.formulas) > 0,
                )
            )
            slide_num += 1

        # 3. Slide: Final Summary (if available)
        if lesson.summary:
            slides.append(
                Slide(
                    slide_number=slide_num,
                    total_slides=0,
                    slide_type="summary",
                    badge="まとめ",
                    title="本時のまとめと振り返り",
                    content_html=f"""
                    <div class="objective-box" style="border-color: var(--chalk-pink); background: rgba(243, 129, 129, 0.12);">
                        <div style="font-size: 2.0rem; font-weight: 700; line-height: 1.8;">
                            {_e(lesson.summary)}
                        </div>
                    </div>
                    {f'<div style="margin-top: 20px; font-size: 1.4rem; color: var(--text-sub);">📌 <b>指導メモ:</b> {_e(lesson.notes_for_teacher)}</div>' if lesson.notes_for_teacher else ''}
                    """,
                    speaker_notes="授業全体の重要事項を振り返り、次回の授業につなげます。",
                )
            )

        # Update total_slides count
        total = len(slides)
        for s in slides:
            s.total_slides = total

        return slides

    def build_presentation(self, lesson: Lesson) -> ElectronicBoardPresentation:
        """Construct a complete ElectronicBoardPresentation object."""
        slides = self.generate_slides_from_lesson(lesson)
        lesson_id = f"lesson_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        
        return ElectronicBoardPresentation(
            lesson_id=lesson_id,
            lesson_title=lesson.lesson_title,
            subject=lesson.subject,
            unit=lesson.unit,
            slides=slides,
            theme=self.theme,
            created_at=datetime.now().isoformat(),
        )

    def render_presentation_html(self, presentation: ElectronicBoardPresentation, lesson: Optional[Lesson] = None) -> str:
        """
        Render standalone HTML string from presentation model. When the lesson is given, it is embedded as
        machine-readable JSON (<script type="application/json" id="lesson-data">) so that System B evaluates
        an exported HTML exactly like the lesson itself.
        """
        html_out = self.template.render(presentation=presentation)
        if lesson is None:
            return html_out
        payload = json.dumps(lesson.model_dump(), ensure_ascii=False)
        payload = payload.replace("<", "\\u003c")  # valid JSON escape; the text can never close the <script>

        block = f'<script type="application/json" id="lesson-data">{payload}</script>\n'
        return html_out.replace("</body>", block + "</body>", 1)

    def save_outputs(
        self,
        lesson: Lesson,
        presentation: ElectronicBoardPresentation,
        html_content: str,
        research_metadata: Optional[Dict[str, Any]] = None,
        is_mock: bool = False,
    ) -> Tuple[Path, Path]:
        """
        Save JSON and HTML files to outputs/ directory, and record research log.
        is_mock=True marks outputs built from the demo sample lesson (not a real model run);
        the research log then records model="mock".
        Returns (json_path, html_path).
        """
        base_name = presentation.lesson_id
        json_path = config.OUTPUT_DIR / f"{base_name}.json"
        html_path = config.OUTPUT_DIR / f"{base_name}.html"

        # Save structured Lesson JSON + Presentation data
        full_export = {
            "lesson": lesson.model_dump(),
            "presentation": presentation.model_dump(),
            "is_mock": bool(is_mock),
        }
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(full_export, jf, ensure_ascii=False, indent=2)

        # Save standalone presentation HTML
        with open(html_path, "w", encoding="utf-8") as hf:
            hf.write(html_content)

        # Append to research log
        if research_metadata:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "lesson_id": base_name,
                "lesson_title": lesson.lesson_title,
                "unit": lesson.unit,
                "slides_count": len(presentation.slides),
                **research_metadata,
                "is_mock": bool(is_mock),
            }
            if is_mock:
                log_entry["model"] = "mock"
            try:
                with open(config.RESEARCH_LOG_PATH, "a", encoding="utf-8") as log_file:
                    log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.warning(f"Could not write to research log: {e}")

        logger.info(f"Saved lesson outputs: JSON -> {json_path}, HTML -> {html_path}")
        return json_path, html_path
