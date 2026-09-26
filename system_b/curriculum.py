"""
Course of Study (学習指導要領) lookup shared by System A (ai/evaluator.py) and System B (pedagogy.py).
The bundled database (data/curriculum/*.json) covers only part of high-school mathematics.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import config

logger = logging.getLogger("ai_e_board.system_b.curriculum")

DEFAULT_PATH = config.CURRICULUM_DIR / "high_school_math_stub.json"
_UNSET = {"", "要確認", "単元", "unknown"}


@dataclass
class CurriculumMatch:
    subject: str
    unit: str
    matched_by: str                      # "unit" | "topic"
    topics: List[str] = field(default_factory=list)
    key_formulas: List[str] = field(default_factory=list)


@lru_cache(maxsize=4)
def load_curriculum(path: Optional[Path] = None) -> dict:
    p = Path(path or DEFAULT_PATH)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.warning(f"Failed to load curriculum DB {p}: {e}")
        return {}


def _core(topic: str) -> str:
    """'順列 (nPr)' → '順列', '場合の数（和の法則・積の法則）' → '場合の数'."""
    return re.split(r"\s*[(（]", topic, 1)[0].strip()


def _contains(a: str, b: str) -> bool:
    return len(b) >= 2 and b in a


def match_unit(unit_text: Optional[str], lesson_title: str = "", path: Optional[Path] = None) -> List[CurriculumMatch]:
    """
    Find curriculum units for a lesson. Matches the unit name first, then topic names (so a lesson titled
    "順列" maps to 数学A「場合の数と確率」). An empty / unconfirmed unit never matches everything.
    """
    unit = (unit_text or "").strip()
    text = " ".join(t for t in (unit, lesson_title) if t and t not in _UNSET)
    if not text:
        return []
    db = load_curriculum(path)
    matches: List[CurriculumMatch] = []
    for subj in db.get("subjects", []):
        for u in subj.get("units", []):
            name = u.get("unit_name", "")
            topics = u.get("topics", [])
            if _contains(text, name) or (unit not in _UNSET and _contains(name, unit)):
                matches.append(CurriculumMatch(subj["subject_name"], name, "unit", topics, u.get("key_formulas", [])))
            elif any(_contains(text, _core(t)) for t in topics):
                matches.append(CurriculumMatch(subj["subject_name"], name, "topic", topics, u.get("key_formulas", [])))
    return matches


def describe(matches: List[CurriculumMatch], unit_text: Optional[str]) -> str:
    """Human-readable alignment message (never claims a match that did not happen)."""
    if (unit_text or "").strip() in _UNSET:
        return "単元名が未確定のため、学習指導要領との照合は行っていません（STEP3で単元名を入力してください）。"
    if not matches:
        return ("収録している学習指導要領データ（高校数学の一部）には該当する単元が見つかりませんでした。"
                "対象範囲外の単元か、単元名の表記が異なる可能性があります。")
    m = matches[0]
    return f"学習指導要領（{m.subject} / {m.unit}）の指導内容（{'、'.join(m.topics)}）に対応しています。"
