"""
Two-stage board analysis pipeline (image → faithful transcript → structured Lesson).

Stage 1 (vision, GPU): transcribe the board line by line (prompts/transcribe_board.txt).
Segmentation (Python): the transcript is split deterministically into blocks at column changes,
         numbered headings (例5 / 練習11 / (2)), definition sentences (〜という。) and the objective box.
Stage 2 (text only, GPU): the model only *labels blocks* (section type, heading, hint / note) and writes the
         short lesson-level texts (title, objectives, intro, summary, teacher note) — prompts/structure_board.txt.
Assembly (Python): sections are built from the original transcript lines. Board content can therefore never
         be paraphrased, "corrected" or invented by the model; the model's labels are also checked against
         the block's shape (computation lines, definition sentences, question sentences, board numbers).

Why this split: in one combined pass the 4B model summarised or rewrote problems (e.g. "fixed" a wrong
answer written on the board, invented exercises), and it cannot track dozens of line numbers reliably.
Stage 1 alone transcribes accurately, and labelling a handful of blocks is an easy, short task.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import config
from ai.board_markup import (
    UNCONFIRMED,
    _as_math_text,
    _canon_key,
    _canon_section,
    _COLOR_WORDS,
    normalize_inline_math,
    normalize_latex,
    strip_wrapping,
)
from ai.llm_client import LLMBackend, LLMError
from models.schemas import ExampleProblem, Exercise, Formula, Lesson, Section, VisualAnnotation

logger = logging.getLogger("ai_e_board.board_pipeline")

# (phase, message) progress callback; phase is "transcribe" | "structure"
PhaseCallback = Callable[[str, str], None]


# ---------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------

@dataclass
class TLine:
    no: int            # 1-based line number
    text: str          # content without decoration prefixes
    column: int = 1
    color: Optional[str] = None
    boxed: bool = False


_DATE_TOKEN = r"(?:\d{1,2}\s*/\s*\d{1,2}|[(（][月火水木金土日][)）]|[pP]\.?\s*\d+|\d{1,2}月\d{1,2}日)"
_DATE_PAGE = re.compile(rf"^(?:{_DATE_TOKEN}\s*)+$")
_LEADING_DATE = re.compile(rf"^(?:{_DATE_TOKEN}\s+)+")
_COLOR_PREFIX = re.compile(r"^【(赤|黄|青|白|緑|オレンジ|橙|ピンク|黄色)】\s*")
_BOX_START = re.compile(r"^【\s*囲み(開始)?\s*】$")
_BOX_END = re.compile(r"^【\s*囲み終了\s*】$")
_COLUMN = re.compile(r"^#{1,3}\s*列\s*(\d+)")
_BULLET = re.compile(r"^\s*(?:[-*・]\s+|\d+\.\s+)")
_CIRCLED = re.compile(r"[①-⑳]")  # ①..⑳ (teachers circle numbers for emphasis)


def parse_transcript(text: str) -> List[TLine]:
    """Parse the stage-1 transcript: columns, color / box markers, date & page tokens, circled numbers."""
    lines: List[TLine] = []
    column, boxed = 1, False
    for raw in strip_wrapping(text).splitlines():
        s = raw.strip()
        if not s:
            continue
        m = _COLUMN.match(s)
        if m:
            column = int(m.group(1))
            continue
        if s.startswith("#"):
            s = s.lstrip("#").strip()
        if _BOX_START.match(s):
            boxed = True
            continue
        if _BOX_END.match(s):
            boxed = False
            continue
        s = _BULLET.sub("", s)
        color = None
        cm = _COLOR_PREFIX.match(s)
        if cm:
            color = _COLOR_WORDS.get(cm.group(1).replace("黄色", "黄"))
            s = s[cm.end():]
        s = _CIRCLED.sub(lambda c: str(ord(c.group(0)) - 0x245F), s)  # ④P3 → 4P3
        if _DATE_PAGE.match(s.strip("$ ")):
            continue  # date / weekday / page lines are not lesson content
        s = _LEADING_DATE.sub("", s).strip()  # "6/3 順列" → "順列", "p.18 吹奏楽部…" → "吹奏楽部…"
        if s:
            lines.append(TLine(no=len(lines) + 1, text=s, column=column, color=color, boxed=boxed))
    return lines


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

_HEADING = re.compile(r"^(例題?\s*\d+|練習?\s*\d+|問\s*\d+|\(\d+\)|（\d+）)")
_MAIN_LABEL = re.compile(r"^(例題?\s*\d+|練習?\s*\d+|問\s*\d+)")
_SUB_LABEL = re.compile(r"^(\(\d+\)|（\d+）)")
_OBJECTIVE = re.compile(r"めあて|ねらい|目標")
_DEFINITION_END = re.compile(r"(という|といい|と呼ぶ|と表す)[。.]?\s*$")
_MATH_ONLY = re.compile(r"^\$[^$]+\$$")
_CONTINUATION = re.compile(r"^\$?\s*=")
_QUESTION_END = ("か。", "か.", "か", "か？", "せよ。", "せよ", "なさい。", "求めよ。")


@dataclass
class Block:
    id: int
    lines: List[TLine]
    role: str = "content"   # "title" | "objective" | "content"
    label: str = ""         # board numbering, e.g. "練12 (2)"

    @property
    def start(self) -> int:
        return self.lines[0].no

    @property
    def end(self) -> int:
        return self.lines[-1].no

    @property
    def continuations(self) -> int:
        return sum(1 for ln in self.lines if _CONTINUATION.match(ln.text.strip()))

    @property
    def has_question(self) -> bool:
        return any(ln.text.strip().endswith(_QUESTION_END) for ln in self.lines)

    @property
    def has_definition(self) -> bool:
        return any(_DEFINITION_END.search(ln.text) for ln in self.lines)


def _is_title_line(t: str) -> bool:
    """A short heading such as "順列" or "$y=ax^2+bx+c$のグラフ" (not a formula line or a question)."""
    t = t.strip()
    return (len(t) <= 30 and not _HEADING.match(t) and not _OBJECTIVE.search(t)
            and not _MATH_ONLY.match(t) and not _CONTINUATION.match(t) and not t.endswith(_QUESTION_END))


def segment_blocks(lines: Sequence[TLine]) -> List[Block]:
    """
    Split the transcript into blocks. A new block starts at a column change, a numbered heading,
    the objective (めあて) text, a box boundary, after a definition sentence, and where the objective text
    reaches its first formula line. The first short line of the board is its title.
    """
    blocks: List[Block] = []
    cur: List[TLine] = []

    def flush():
        if cur:
            blocks.append(Block(len(blocks) + 1, list(cur)))
            cur.clear()

    prev: Optional[TLine] = None
    for ln in lines:
        t = ln.text.strip()
        in_objective = bool(cur) and bool(_OBJECTIVE.search(cur[0].text))
        new_block = (
            prev is None
            or ln.column != prev.column
            or bool(_HEADING.match(t))
            or bool(_OBJECTIVE.search(t))
            or ln.boxed != prev.boxed
            or bool(_DEFINITION_END.search(prev.text))
            or (in_objective and bool(_MATH_ONLY.match(t)))
            or (len(blocks) == 0 and len(cur) == 1 and _is_title_line(cur[0].text))  # board title line
        )
        # "練習11" alone followed by "(2) ..." belongs to the sub-problem block
        if new_block and prev is not None and cur == [prev] and _MAIN_LABEL.fullmatch(prev.text.strip()) \
                and ln.column == prev.column:
            new_block = False
        if new_block:
            flush()
        cur.append(ln)
        prev = ln
    flush()

    # roles and board labels; a sub-problem "(3)" inherits the last main label ("練習11"), which may
    # have been written in the previous column
    main_label = ""
    for i, b in enumerate(blocks):
        first = b.lines[0].text.strip()
        if _OBJECTIVE.search(first):
            b.role = "objective"
        elif i == 0 and len(b.lines) == 1 and _is_title_line(first):
            b.role = "title"
        labels: List[str] = []
        for ln in b.lines[:2]:
            m = _MAIN_LABEL.match(ln.text.strip())
            if m:
                main_label = m.group(1).replace(" ", "")
                labels.append(main_label)
            s = _SUB_LABEL.match(ln.text.strip())
            if s:
                if not labels and main_label:
                    labels.append(main_label)
                labels.append(s.group(1))
                break
            if not m:
                break
        b.label = " ".join(labels)
    return blocks


def numbered_blocks(blocks: Sequence[Block]) -> str:
    out = []
    for b in blocks:
        note = {"title": "（板書のタイトル）", "objective": "（めあて）"}.get(b.role, "")
        out.append(f"B{b.id}（列{b.lines[0].column}）{note}:")
        out.extend(f"  {ln.text}" for ln in b.lines)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Stage-2 output parsing
# ---------------------------------------------------------------------------

@dataclass
class SecSpec:
    kind: str
    title: str
    block_ids: List[int]
    hints: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


_SEC_LINE = re.compile(r"^\s*#{1,3}\s*([A-Za-z_]+|[぀-ヿ一-鿿]+)\s*[:：]\s*(.*?)\s*(?:[|｜]\s*(.+))?$")
_KV = re.compile(r"^\s*(?:[-*]\s*)?([A-Za-z_]+|[぀-ヿ一-鿿]+)\s*[:：]\s*(.*)$")
_BLOCK_REF = re.compile(r"B\s*(\d+)(?:\s*[-–〜~～+＋]\s*B?\s*(\d+))?")
_HEADER_KEYS = ("subject", "grade", "unit", "title", "objective", "intro", "summary", "teacher_note")


def _block_ids(value: str, valid: set) -> List[int]:
    ids: List[int] = []
    for a, b in _BLOCK_REF.findall(value):
        lo, hi = sorted((int(a), int(b or a)))
        ids.extend(i for i in range(lo, hi + 1) if i in valid and i not in ids)
    return ids


def parse_structure(text: str, blocks: Sequence[Block]) -> Tuple[Dict[str, List[str]], List[SecSpec]]:
    """
    Parse the stage-2 output ("# 種類: 見出し | B3"). Each content block ends up in exactly one section:
    blocks referenced twice keep their first section, and blocks the model did not mention are added
    back with a type inferred from their shape, so no board content is lost.
    """
    header: Dict[str, List[str]] = {}
    specs: List[SecSpec] = []
    content_ids = {b.id for b in blocks if b.role == "content"}
    used: set = set()
    cur: Optional[SecSpec] = None
    for raw in strip_wrapping(text).splitlines():
        line = raw.strip()
        if not line:
            continue
        sm = _SEC_LINE.match(line)
        if sm and line.startswith("#"):
            kind = _canon_section(sm.group(1)) or _canon_section(sm.group(2)) or "concept"
            ids = [i for i in _block_ids(sm.group(3) or "", content_ids) if i not in used]
            if kind == "summary" or not ids:
                cur = None
                continue
            used.update(ids)
            cur = SecSpec(kind=kind, title=sm.group(2).strip() or sm.group(1), block_ids=ids)
            specs.append(cur)
            continue
        km = _KV.match(line)
        if not km:
            continue
        key, value = _canon_key(km.group(1)), km.group(2).strip()
        if not value:
            continue
        if cur is not None and key == "hint":
            cur.hints.append(value)
        elif cur is not None and key == "note":
            cur.notes.append(value)
        elif key in _HEADER_KEYS:
            header.setdefault(key, []).append(value)

    by_id = {b.id: b for b in blocks}
    for bid in sorted(content_ids - used):
        b = by_id[bid]
        specs.append(SecSpec(kind=infer_kind(b, first_content=bid == min(content_ids)), title="", block_ids=[bid]))
    specs.sort(key=lambda s: min(s.block_ids))
    for s in specs:
        s.kind = reconcile_kind(s.kind, [by_id[i] for i in s.block_ids])
        s.title = section_title(s, [by_id[i] for i in s.block_ids])
    return header, specs


# ---------------------------------------------------------------------------
# Section type / title rules (the block's shape wins over an implausible model label)
# ---------------------------------------------------------------------------

_KIND_NAMES = {"introduction": "導入", "concept": "解説", "definition": "定義", "formula": "公式",
               "example": "例題", "exercise": "練習"}


def infer_kind(b: Block, first_content: bool = False) -> str:
    if b.label.startswith(("練", "問")) or (b.label.startswith("(") and not b.label.startswith("例")):
        return "exercise"
    if b.label.startswith("例"):
        return "example"
    if b.has_question:
        return "introduction" if first_content else "example"
    if b.continuations >= 2:
        return "example"
    if b.has_definition:
        return "definition"
    if b.continuations == 1 or any(_MATH_ONLY.match(ln.text) for ln in b.lines):
        return "formula"
    return "concept"


def reconcile_kind(kind: str, blocks: Sequence[Block]) -> str:
    label = next((b.label for b in blocks if b.label), "")
    conts = sum(b.continuations for b in blocks)
    question = any(b.has_question for b in blocks)
    definition = any(b.has_definition for b in blocks)
    if label.startswith(("練", "問")) or (label.startswith("(") and kind not in ("example", "introduction")):
        return "exercise"
    if label.startswith("例") and kind not in ("example", "introduction"):
        return "example"
    if kind in ("formula", "definition", "concept") and (conts >= 2 or question):
        return "example"  # a worked computation / a question is not a definition
    if kind in ("example", "exercise") and not label and not question and conts == 0:
        return "definition" if definition else "formula"
    if kind == "definition" and not definition:
        return "formula"
    return kind


def section_title(spec: SecSpec, blocks: Sequence[Block]) -> str:
    """Board numbering beats model headings; model headings with numbers not on the board are dropped."""
    label = next((b.label for b in blocks if b.label), "")
    if label:
        return label
    title = spec.title.strip()
    text = " ".join(ln.text for b in blocks for ln in b.lines)
    fabricated = re.match(r"^(例題?|練習?|問)\s*(\d+)", title)
    if fabricated and fabricated.group(0).replace(" ", "") not in text.replace(" ", ""):
        title = re.sub(r"^(例題?|練習?|問)\s*\d+\s*[:：]?\s*", "", title)
    if not title:
        first = re.sub(r"\$[^$]*\$", "", blocks[0].lines[0].text).strip(" 、。:：")
        title = first[:18] if first else _KIND_NAMES.get(spec.kind, "板書の内容")
    return normalize_inline_math(title)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

_STEP_START = re.compile(r"^(\$?\s*=|=|\$?\s*[-−]?\s*\d|\$)")
_INLINE_ANSWER = re.compile(r"^(?P<math>.*?=\s*\S+?)\s+(?P<ans>[\d,]+\s*通り)\s*$")
_RESULT_HINT = re.compile(r"通り|=")


def _strip_label(text: str) -> str:
    """Remove numbering (例5, 練習11, (2)) from the start of a line."""
    t = text.strip()
    while True:
        m = _HEADING.match(t)
        if not m:
            return t
        t = t[m.end():].strip(" 　:：.")


def _mark_for(ln: TLine) -> Optional[VisualAnnotation]:
    if not ln.color and not ln.boxed:
        return None
    return VisualAnnotation(element_type="box" if ln.boxed else "color_highlight",
                            target_text=normalize_inline_math(ln.text), color=ln.color,
                            note="囲み" if ln.boxed else "色チョーク")


def _split_inline_answer(text: str) -> Tuple[str, Optional[str]]:
    """'4P2=4×3=12 12通り' → ('4P2=4×3=12', '12通り'); '=24 24通り' → ('=24', '24通り')."""
    m = _INLINE_ANSWER.match(text.strip().strip("$"))
    if m:
        return m.group("math").strip(), m.group("ans").replace(" ", "")
    return text, None


def split_problem(lines: Sequence[TLine]) -> Tuple[List[str], List[str], Optional[str]]:
    """
    (problem parts, solution steps, answer) of an example / exercise block, taken verbatim from the board.
    - problem: lines up to a question sentence; else the numbered first line (or the expression right
      before the first "= ..." line); a one-line "4P2=4×3=12" problem keeps only its left-hand side.
    - answer: an inline "… 12通り" result, else the last line that looks like a result.
    """
    texts = [_strip_label(ln.text) for ln in lines]
    idx = [i for i, t in enumerate(texts) if t]  # heading-only lines carry no content
    if not idx:
        return [], [], None
    problem_idx: List[int] = []
    q = next((k for k in idx[:4] if lines[k].text.strip().endswith(_QUESTION_END)), None)
    if q is not None:
        problem_idx = [k for k in idx if k <= q]
    elif _HEADING.match(lines[idx[0]].text.strip()) or (idx[0] > 0 and _HEADING.match(lines[0].text.strip())):
        problem_idx = [idx[0]]
    else:
        # the expression being transformed: nearest line with math before the first "= ..." line
        # (skipping side notes such as "3個の計算")
        cont = next((k for k in idx if k > idx[0] and _CONTINUATION.match(texts[k]) and not _CONTINUATION.match(texts[k - 1])), None)
        target = None
        if cont is not None:
            target = next((k for k in range(cont - 1, -1, -1) if k in idx and ("=" in texts[k] or "$" in texts[k])), None)
        problem_idx = [target if target is not None else idx[0]]

    problem, steps, answer = [], [], None
    for k in idx:
        t = texts[k]
        if k in problem_idx:
            if "=" in t and q is None:
                lhs = t.split("=", 1)[0].strip()
                math, inline_ans = _split_inline_answer(t)
                problem.append(_as_math_text(lhs) if lhs else _as_math_text(t))
                steps.append(_as_math_text(math))
                answer = inline_ans or answer
            else:
                problem.append(_as_math_text(t) if not re.search(r"[぀-ヿ一-鿿]{3,}", t) else normalize_inline_math(t))
            continue
        math, inline_ans = _split_inline_answer(t)
        steps.append(_as_math_text(math))
        if inline_ans:
            answer = inline_ans
    if answer is None:
        candidates = [texts[k] for k in idx if k not in problem_idx and _RESULT_HINT.search(texts[k])]
        if candidates:
            answer = _as_math_text(_split_inline_answer(candidates[-1])[0])
    return problem, steps, answer


def _build_section(i: int, spec: SecSpec, blocks: Sequence[Block]) -> Section:
    lines = [ln for b in blocks for ln in b.lines]
    marks = [m for m in (_mark_for(ln) for ln in lines) if m]
    title, kind = spec.title, spec.kind
    notes = " ".join(spec.notes) or None

    if kind in ("example", "exercise", "introduction"):
        problem_parts, steps, answer = split_problem(lines)
        problem = " ".join(problem_parts) or title
        if kind == "exercise":
            exercise = Exercise(title=title, problem=problem, hint=" ".join(spec.hints) or None,
                                answer=answer or f"{UNCONFIRMED}/未記載", solution_steps=steps)
            return Section(section_id=f"sec_{i:02d}", section_type="exercise", title=title,
                           content=notes or problem, exercise=exercise, visual_annotations=marks, order=i)
        example = ExampleProblem(title=title, problem=problem, solution_steps=steps,
                                 answer=answer or UNCONFIRMED, teaching_notes=notes)
        return Section(section_id=f"sec_{i:02d}", section_type=kind, title=title, content=problem,
                       example=example, visual_annotations=marks, order=i)

    prose, formulas = [], []
    for ln in lines:
        t = ln.text.strip()
        if _MATH_ONLY.match(t):
            latex = normalize_latex(t.strip("$"))
            formulas.append(Formula(raw_text=latex, latex=latex, is_key_formula=bool(ln.boxed or ln.color)))
        else:
            prose.append(_as_math_text(t) if not re.search(r"[぀-ヿ一-鿿]{2,}", t) else normalize_inline_math(t))
    if notes:
        prose.append("（指導のポイント）" + notes)
    content = "\n".join(prose) or "\n".join(f"${f.latex}$" for f in formulas) or title
    if kind == "formula" and formulas and not any(f.is_key_formula for f in formulas):
        formulas[0].is_key_formula = True
    return Section(section_id=f"sec_{i:02d}", section_type=kind if kind in ("concept", "definition", "formula") else "concept",
                   title=title, content=content, formulas=formulas, visual_annotations=marks, order=i)


def assemble_lesson(blocks: Sequence[Block], header: Dict[str, List[str]], specs: List[SecSpec],
                    source_images: Sequence[str]) -> Tuple[Lesson, List[str]]:
    warnings: List[str] = []
    by_id = {b.id: b for b in blocks}
    sections = [_build_section(i, s, [by_id[j] for j in s.block_ids]) for i, s in enumerate(specs, 1)]
    for s in sections:
        if (s.example and s.example.answer == UNCONFIRMED) or (s.exercise and (s.exercise.answer or "").startswith(UNCONFIRMED)):
            warnings.append(f"「{s.title}」の答えが板書から読み取れませんでした（要確認）。")

    def first(key, default=None):
        vals = header.get(key) or []
        return vals[0].strip() if vals else default

    title_block = next((b for b in blocks if b.role == "title"), None)
    board_title = normalize_inline_math(title_block.lines[0].text) if title_block else None
    objective_block = next((b for b in blocks if b.role == "objective"), None)
    objectives = [normalize_inline_math(o) for o in header.get("objective", [])][:2]
    if not objectives and objective_block:
        body = [ln.text for ln in objective_block.lines if not re.fullmatch(r".{0,8}(めあて|ねらい|目標).{0,2}", ln.text)]
        if body:
            objectives = [normalize_inline_math("".join(body))]
    unit = first("unit") or board_title or UNCONFIRMED
    if unit == UNCONFIRMED:
        warnings.append("単元名が読み取れませんでした（要確認）。")
    lines = [ln for b in blocks for ln in b.lines]
    marks = [a for s in sections for a in s.visual_annotations]
    lesson = Lesson(
        subject=first("subject", "数学") or "数学",
        grade=first("grade", UNCONFIRMED),
        unit=normalize_inline_math(unit),
        lesson_title=normalize_inline_math(first("title") or board_title or unit),
        learning_objectives=[o for o in objectives if o],
        introduction=normalize_inline_math(" ".join(header.get("intro", []))) or None,
        sections=sections,
        summary=normalize_inline_math(" ".join(header.get("summary", []))) or None,
        notes_for_teacher=normalize_inline_math(" ".join(header.get("teacher_note", []))) or None,
        source_images=list(source_images),
        visual_structure={
            "layout_description": f"{max((ln.column for ln in lines), default=1)} 列構成の板書",
            "color_scheme": sorted({a.color for a in marks if a.color}),
            "diagrams_present": False,
            "table_present": False,
        },
    )
    return lesson, warnings


def build_lesson(transcript: str, structure: str, source_images: Sequence[str]) -> Tuple[Optional[Lesson], List[str], bool]:
    """transcript + stage-2 output → (lesson, warnings, used_fallback). Also used to re-assemble saved outputs."""
    lines = parse_transcript(transcript)
    if not lines:
        return None, [], False
    blocks = segment_blocks(lines)
    header, specs = parse_structure(structure, blocks)
    fallback = not header and not any(s.hints or s.notes for s in specs)
    lesson, warnings = assemble_lesson(blocks, header, specs, source_images)
    if fallback:
        warnings.append("授業構成は見出しから自動で区切りました（STEP3で確認してください）。")
    return lesson, warnings, fallback


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

@dataclass
class BoardAnalysis:
    lesson: Optional[Lesson]
    error: Optional[str]
    warnings: List[str]
    transcript: str = ""
    structure: str = ""
    timings: Dict[str, float] = field(default_factory=dict)
    model: str = ""
    device: str = "unknown"
    structure_fallback: bool = False


def _load_prompt(name: str) -> str:
    return (config.PROMPTS_DIR / name).read_text(encoding="utf-8")


def analyze_board(
    backend: LLMBackend,
    model: str,
    image_paths: Sequence[Path],
    custom_instructions: Optional[str] = None,
    on_phase: Optional[PhaseCallback] = None,
    on_delta=None,
) -> BoardAnalysis:
    notify = on_phase or (lambda phase, msg: None)
    names = [Path(p).name for p in image_paths]
    timings: Dict[str, float] = {}

    # Stage 1: transcription (vision)
    notify("transcribe", "板書を書き起こしています（GPU）…")
    t0 = time.time()
    prompt = _load_prompt(config.TRANSCRIBE_PROMPT_FILE)
    if custom_instructions:
        prompt += f"\n\n【教員からの補足（書き起こしの参考）】\n{custom_instructions}"
    try:
        tr = backend.generate(model, prompt, images=list(image_paths),
                              max_tokens=config.TRANSCRIBE_MAX_TOKENS, on_delta=on_delta)
    except (LLMError, ConnectionError) as e:
        return BoardAnalysis(None, f"板書の書き起こしに失敗しました: {e}", [], model=model)
    timings["transcribe"] = time.time() - t0
    lines = parse_transcript(tr.text)
    if not lines:
        return BoardAnalysis(None, "板書から文字を読み取れませんでした。画像を確認してください。", [],
                             transcript=tr.text, timings=timings, model=tr.model, device=tr.device)
    warnings: List[str] = []
    if tr.truncated:
        warnings.append("書き起こしが途中で打ち切られました（繰り返しの検出または長さの上限）。")

    # Stage 2: label blocks (text only)
    notify("structure", "授業の構成を整理しています（GPU）…")
    t1 = time.time()
    blocks = segment_blocks(lines)
    structure_text = ""
    try:
        st = backend.generate(model, _load_prompt(config.STRUCTURE_PROMPT_FILE) + "\n\n【書き起こし】\n" + numbered_blocks(blocks),
                              max_tokens=config.STRUCTURE_MAX_TOKENS, on_delta=on_delta)
        structure_text = st.text
    except (LLMError, ConnectionError) as e:
        warnings.append(f"授業構成の整理に失敗しました（{e}）。")
    timings["structure"] = time.time() - t1

    lesson, w2, fallback = build_lesson(tr.text, structure_text, names)
    timings["total"] = timings["transcribe"] + timings["structure"]
    return BoardAnalysis(lesson, None, warnings + w2, transcript=tr.text, structure=structure_text,
                         timings=timings, model=tr.model, device=tr.device, structure_fallback=fallback)
