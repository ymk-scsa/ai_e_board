"""
板書解析の品質評価ツール（M5）

    python tools/eval_boards.py                 # data/eval/boards.json の全板書を GPU で解析して採点
    python tools/eval_boards.py --only quadratic
    python tools/eval_boards.py --from-raw outputs/eval/<run>   # 保存済みの書き起こし・構成から組み立て直して再採点（GPU不要）

採点:
  - content: 板書に書かれている式・語句（expected）が授業データに含まれている割合（表記揺れは正規化）
  - structure: タイトル/単元/目標/導入/例題(解法2手順以上)/練習/まとめ/指導メモ の充足率
  - forbidden: 板書に無い内容（ループで作られた例題など）が出ていないか
  - time: 解析にかかった秒数
合格ライン: content >= 0.85, structure >= 0.85, forbidden = 0
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from ai import board_pipeline  # noqa: E402
from models.schemas import Lesson  # noqa: E402

PASS_CONTENT = 0.85
PASS_STRUCTURE = 0.85


def normalize_for_match(text: str) -> str:
    """Collapse notation differences: LaTeX commands, braces, multiplication signs, spaces, full-width chars."""
    s = text
    s = re.sub(r"\\d?frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", s)
    s = re.sub(r"\\(times|cdot|ast)\b", "*", s)
    s = s.replace("×", "*").replace("・", "*").replace("·", "*").replace("⋅", "*").replace("、", "*")
    s = re.sub(r"\\(left|right|,|;|!|quad|qquad)", "", s)
    s = s.replace("\\{", "(").replace("\\}", ")").replace("{", "").replace("}", "")
    s = s.replace("−", "-").replace("＝", "=").replace("（", "(").replace("）", ")").replace("！", "!")
    s = s.replace("$", "").replace("\\", "")
    s = re.sub(r"(?<![_0-9A-Za-z])(\d+)P_?(\d+)", r"_\1P_\2", s)  # 4P3 / 4P_3 → _4P_3 (also after kanji)
    s = re.sub(r"\s+", "", s)
    return s


def lesson_text(lesson: Lesson) -> str:
    return json.dumps(lesson.model_dump(), ensure_ascii=False)


def structure_checks(lesson: Lesson) -> dict:
    examples = [s.example for s in lesson.sections if s.example]
    return {
        "title": bool(lesson.lesson_title and lesson.lesson_title != "要確認"),
        "unit": bool(lesson.unit and lesson.unit != "要確認"),
        "objectives": len(lesson.learning_objectives) >= 1,
        "introduction": bool(lesson.introduction),
        "example_with_steps": any(len(e.solution_steps) >= 2 for e in examples),
        "exercise": any(s.exercise for s in lesson.sections),
        "summary": bool(lesson.summary),
        "teacher_notes": bool(lesson.notes_for_teacher),
        "sections>=3": len(lesson.sections) >= 3,
    }


def score(lesson: Lesson, board: dict) -> dict:
    haystack = normalize_for_match(lesson_text(lesson))
    found = {e: normalize_for_match(e) in haystack for e in board["expected"]}
    forbidden = [f for f in board.get("forbidden", []) if normalize_for_match(f) in haystack]
    checks = structure_checks(lesson)
    content = sum(found.values()) / len(found)
    structure = sum(checks.values()) / len(checks)
    return {
        "content": round(content, 3),
        "structure": round(structure, 3),
        "missing": [e for e, ok in found.items() if not ok],
        "failed_checks": [k for k, ok in checks.items() if not ok],
        "forbidden_found": forbidden,
        "sections": len(lesson.sections),
        "passed": content >= PASS_CONTENT and structure >= PASS_STRUCTURE and not forbidden,
    }


def run_board(board: dict, out_dir: Path, analyzer=None, raw_dir: Path | None = None) -> dict:
    """Analyze one board (or re-assemble saved stage outputs) and score the resulting lesson."""
    stem = board["id"]
    meta: dict = {}
    if raw_dir:
        # re-assemble from the saved stage-1 transcript and stage-2 structure (no GPU)
        transcript = (raw_dir / f"{stem}.transcript.txt").read_text(encoding="utf-8")
        structure = (raw_dir / f"{stem}.structure.txt").read_text(encoding="utf-8")
        meta_path = raw_dir / f"{stem}.meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        lesson, warnings, _ = board_pipeline.build_lesson(transcript, structure, [Path(board["image"]).name])
        if lesson is None:
            return {"id": stem, "error": "empty transcript", **meta}
    else:
        t = time.time()
        res = analyzer.analyze_board_to_lesson([ROOT / board["image"]])
        meta = {"model": res.model, "device": res.device, "timings": {k: round(v, 1) for k, v in res.timings.items()},
                "structure_fallback": res.structure_fallback, "wall_seconds": round(time.time() - t, 1)}
        (out_dir / f"{stem}.transcript.txt").write_text(res.transcript, encoding="utf-8")
        (out_dir / f"{stem}.structure.txt").write_text(res.structure, encoding="utf-8")
        (out_dir / f"{stem}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        if res.lesson is None:
            return {"id": stem, "error": res.error, **meta}
        lesson, warnings = res.lesson, res.warnings

    (out_dir / f"{stem}.lesson.json").write_text(
        json.dumps(lesson.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
    return {"id": stem, **score(lesson, board), "warnings": warnings, **meta}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="評価する板書 id（カンマ区切り）")
    ap.add_argument("--from-raw", help="保存済みの出力ディレクトリを再採点する（GPU不要）")
    ap.add_argument("--model", help="モデルを上書き（例: qwen3.5-2b）")
    args = ap.parse_args()

    boards = json.loads((ROOT / "data" / "eval" / "boards.json").read_text(encoding="utf-8"))["boards"]
    if args.only:
        wanted = set(args.only.split(","))
        boards = [b for b in boards if b["id"] in wanted]

    raw_dir = Path(args.from_raw) if args.from_raw else None
    out_dir = raw_dir or (config.OUTPUT_DIR / "eval" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    analyzer = None
    if not raw_dir:
        from ai.vision import VisionAnalyzer
        analyzer = VisionAnalyzer(model=args.model)
        print(f"backend={analyzer.backend_name} model={analyzer.model}", flush=True)

    results = []
    for b in boards:
        r = run_board(b, out_dir, analyzer, raw_dir)
        results.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
    (out_dir / "report.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n出力: {out_dir}")
    ok = all(r.get("passed") for r in results)
    print("総合: " + ("合格" if ok else "不合格"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
