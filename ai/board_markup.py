"""
Helpers for the line-based board formats used by ai/board_pipeline.py:
key/section-name canonicalization and LaTeX normalization of transcribed math.
"""

from __future__ import annotations

import re
from typing import Optional

UNCONFIRMED = "要確認"

SECTION_TYPES = {"introduction", "concept", "definition", "formula", "example", "exercise", "summary"}
_SECTION_ALIASES = {
    "intro": "introduction", "導入": "introduction", "概念": "concept", "解説": "concept",
    "def": "definition", "定義": "definition", "公式": "formula", "例題": "example", "例": "example",
    "練習": "exercise", "練習問題": "exercise", "問": "exercise", "問題": "exercise", "まとめ": "summary",
}
_KEY_ALIASES = {
    "教科": "subject", "学年": "grade", "単元": "unit", "タイトル": "title", "授業タイトル": "title",
    "目標": "objective", "めあて": "objective", "objectives": "objective", "導入": "intro", "introduction": "intro",
    "問題": "problem", "答え": "answer", "解答": "answer", "ヒント": "hint", "指導": "note",
    "teaching_note": "note", "まとめ": "summary", "注意": "teacher_note", "notes_for_teacher": "teacher_note",
}
_COLOR_WORDS = {"赤": "red", "黄": "yellow", "青": "blue", "白": "white", "緑": "green", "オレンジ": "orange",
                "橙": "orange", "ピンク": "pink"}


def _canon_key(key: str) -> str:
    k = key.strip().lower()
    return _KEY_ALIASES.get(k, _KEY_ALIASES.get(key.strip(), k))


def _canon_section(kind: str) -> Optional[str]:
    k = kind.strip().lower()
    if k in SECTION_TYPES:
        return k
    for alias, canon in _SECTION_ALIASES.items():
        if kind.strip().startswith(alias):
            return canon
    return None


def strip_wrapping(text: str) -> str:
    """Remove markdown code fences / <think> blocks the model may add around its output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^```[a-zA-Z]*\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


# ---------------------------------------------------------------------------
# LaTeX normalization
# ---------------------------------------------------------------------------

_PERM_PLAIN = re.compile(r"(?<![\w{}_])(\d+|[a-zA-Z])\s*([PC])\s*(\d+|[a-zA-Z])(?![\w{])")
_PERM_SUB = re.compile(r"(?<![\w{}])(\d+|[a-zA-Z])\s*([PC])_\{?(\d+|[a-zA-Z])\}?")
_PERM_BAD_PREFIX = re.compile(r"(?<!\{\})_\{?(\d+|[a-zA-Z])\}?\s*([PC])_\{?(\d+|[a-zA-Z])\}?")


def normalize_latex(expr: str) -> str:
    """Canonicalize notations the model writes inconsistently (inside math): ×, ÷, 4P3 → {}_{4}P_{3}."""
    s = expr.strip()
    s = s.replace("×", r"\times ").replace("÷", r"\div ").replace("−", "-").replace("＝", "=")
    s = s.replace("（", "(").replace("）", ")").replace("｛", r"\{").replace("｝", r"\}")
    s = _PERM_BAD_PREFIX.sub(r"{}_{\1}\2_{\3}", s)
    s = _PERM_SUB.sub(r"{}_{\1}\2_{\3}", s)
    s = _PERM_PLAIN.sub(r"{}_{\1}\2_{\3}", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()


def normalize_inline_math(text: str) -> str:
    """Normalize $...$ segments inside prose and drop an unbalanced trailing '$'."""
    if not text:
        return text
    if text.count("$") % 2 == 1:
        idx = text.rfind("$")
        text = text[:idx] + text[idx + 1:]
    parts = re.split(r"(\$\$.+?\$\$|\$.+?\$)", text)
    out = []
    for p in parts:
        if p.startswith("$$") and p.endswith("$$") and len(p) > 4:
            out.append("$$" + normalize_latex(p[2:-2]) + "$$")
        elif p.startswith("$") and p.endswith("$") and len(p) > 2:
            out.append("$" + normalize_latex(p[1:-1]) + "$")
        else:
            out.append(p)
    return "".join(out).strip()


def _as_math_text(value: str) -> str:
    """A step/answer line: keep prose with $..$ as is; a bare formula gets wrapped in $..$."""
    v = value.strip()
    if "$" in v:
        return normalize_inline_math(v)
    looks_math = bool(re.search(r"[=^_\\]|\d\s*[+\-×x*/]\s*\d|^\d+\s*[PC]\s*\d+$|^\d+!$", v)) \
        and not re.search(r"[぀-ヿ一-鿿]{3,}", v)
    return f"${normalize_latex(v)}$" if looks_math else v
