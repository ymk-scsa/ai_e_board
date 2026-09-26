"""
Tests for ai/board_pipeline.py. Fixtures are real stage-1 transcripts produced on GPU from
uploads/IMG_0938.jpeg (quadratic) and uploads/IMG_0939.png (permutation).
"""

import re
from pathlib import Path

from ai import board_pipeline as bp
from ai.llm_client import LLMError, LLMResult

FIX = Path(__file__).parent / "fixtures"


def _fixture(name):
    return (FIX / name).read_text(encoding="utf-8")


def _blocks(name):
    return bp.segment_blocks(bp.parse_transcript(_fixture(f"transcript_{name}.txt")))


# ---------------------------------------------------------------------------
# Transcript & segmentation
# ---------------------------------------------------------------------------

def test_parse_transcript_columns_colors_boxes_dates():
    lines = bp.parse_transcript(
        "## 列1\n4/23\n(木)\np.85\n6/3 順列\n【囲み開始】\n今回のめあて\n平方完成を理解する\n【囲み終了】\n"
        "【オレンジ】$y=a(x-p)^2+q$\n## 列2\n練習11\n- (2) $2x^2+8x+7$\n⑥P4=360\n")
    assert [ln.text for ln in lines] == ["順列", "今回のめあて", "平方完成を理解する", "$y=a(x-p)^2+q$",
                                         "練習11", "(2) $2x^2+8x+7$", "6P4=360"]
    assert lines[1].boxed and lines[2].boxed and not lines[3].boxed
    assert lines[3].color == "orange"
    assert [ln.column for ln in lines] == [1, 1, 1, 1, 2, 2, 2]


def test_segment_blocks_roles_and_labels_permutation():
    blocks = _blocks("permutation")
    assert blocks[0].role == "title" and blocks[0].lines[0].text == "順列"
    labels = [b.label for b in blocks if b.label]
    assert "例5" in labels and "例6" in labels and "練13" in labels and "練14" in labels
    # sub-problems inherit the main label; "練12" stays with "(1)"
    assert {"練12 (1)", "練12 (2)", "練12 (3)", "練12 (4)"} <= set(labels)


def test_segment_blocks_quadratic_objective_and_definition():
    blocks = _blocks("quadratic")
    assert blocks[0].role == "title"
    objective = next(b for b in blocks if b.role == "objective")
    assert objective.lines[-1].text.startswith("グラフ")  # objective text ends before the first formula
    assert any(b.lines[-1].text.endswith("という。") for b in blocks)
    # "(3)" in column 3 inherits "練習11" from column 2
    assert {"練習11 (2)", "練習11 (3)"} <= {b.label for b in blocks}


# ---------------------------------------------------------------------------
# Structure parsing and kind reconciliation
# ---------------------------------------------------------------------------

def test_parse_structure_blocks_merge_and_backfill():
    blocks = _blocks("permutation")
    ex5 = next(b for b in blocks if b.label == "例5")
    text = (f"unit: 順列\ntitle: 順列の総数\n# definition: 例題1 | B{ex5.id}\nnote: 並び順に注意\n"
            f"# exercise: 練習 | B{ex5.id}\nsummary: まとめ文\nteacher_note: 注意\n")
    header, specs = bp.parse_structure(text, blocks)
    assert header["unit"] == ["順列"] and header["summary"] == ["まとめ文"]
    ex5_spec = next(s for s in specs if ex5.id in s.block_ids)
    # board number beats the model heading; the block shape (question) beats "definition"
    assert ex5_spec.title == "例5" and ex5_spec.kind == "example" and ex5_spec.notes == ["並び順に注意"]
    # every content block is used exactly once, in board order
    used = [i for s in specs for i in s.block_ids]
    assert sorted(used) == used and len(used) == len(set(used))
    assert set(used) == {b.id for b in blocks if b.role == "content"}


def test_fabricated_heading_numbers_are_dropped():
    blocks = _blocks("quadratic")
    definition = next(b for b in blocks if b.has_definition)
    _, specs = bp.parse_structure(f"# example: 例2 平方完成 | B{definition.id}\n", blocks)
    spec = next(s for s in specs if definition.id in s.block_ids)
    assert spec.kind == "definition"      # no computation / question → not an example
    assert spec.title == "平方完成"        # "例2" is not on the board


def test_worked_computation_is_an_example_even_if_labelled_formula():
    blocks = _blocks("quadratic")
    worked = next(b for b in blocks if b.continuations >= 2 and not b.label)
    _, specs = bp.parse_structure(f"# formula: 平方完成の式 | B{worked.id}\n", blocks)
    assert next(s for s in specs if worked.id in s.block_ids).kind == "example"


# ---------------------------------------------------------------------------
# Problem / step / answer extraction
# ---------------------------------------------------------------------------

def _lines(text):
    return bp.parse_transcript(text)


def test_split_problem_question_sentence():
    p, steps, ans = bp.split_problem(_lines("例6 4人の生徒全員が1列に並ぶとき、\n並び方の総数は何通りか。\n4!=4×3×2×1\n=24 24通り"))
    assert p == ["4人の生徒全員が1列に並ぶとき、", "並び方の総数は何通りか。"]
    assert steps == [r"$4!=4\times 3\times 2\times 1$", "$=24$"] and ans == "24通り"


def test_split_problem_one_line_solution():
    p, steps, ans = bp.split_problem(_lines("練12\n(1) 4P2=4×3=12 12通り"))
    assert p == ["${}_{4}P_{2}$"] and steps == [r"${}_{4}P_{2}=4\times 3=12$"] and ans == "12通り"


def test_split_problem_skips_side_notes_and_keeps_board_answer():
    p, steps, ans = bp.split_problem(_lines("4P3=4×3×2\n3個の計算\n=24"))
    assert p == ["${}_{4}P_{3}$"] and ans == "$=24$"
    # the board's (wrong) result is kept verbatim
    p, steps, ans = bp.split_problem(_lines("練13\n5P3=5×4×3\n=120 120通り"))
    assert p == ["${}_{5}P_{3}$"] and ans == "120通り"


def test_split_problem_expression_before_first_continuation():
    p, steps, ans = bp.split_problem(_lines("$y=a(x-p)^2+q$\n$2x^2-4x+5$\n$=2(x^2-2x)+5$\n$=2(x-1)^2+3$"))
    assert p == ["$2x^2-4x+5$"] and ans == "$=2(x-1)^2+3$"


# ---------------------------------------------------------------------------
# End-to-end assembly on real transcripts (no model output → deterministic fallback)
# ---------------------------------------------------------------------------

def test_fallback_lesson_permutation_keeps_board_content_verbatim():
    lesson, warnings, fallback = bp.build_lesson(_fixture("transcript_permutation.txt"), "", ["IMG_0939.png"])
    assert fallback and any("自動で区切りました" in w for w in warnings)
    assert lesson.unit == "順列"  # from the board title line
    dump = lesson.model_dump_json()
    assert "{}_{5}P_{3}" in dump and not re.search(r"(?<!\d)60通り", dump)  # the board's 5P3=…=120 is not "corrected"
    by_title = {s.title: s for s in lesson.sections}
    assert by_title["練12 (2)"].exercise.answer == "210通り"
    assert by_title["例5"].section_type == "example"
    assert lesson.sections[0].section_type == "introduction"  # the opening question


def test_fallback_lesson_quadratic_structure():
    lesson, _, _ = bp.build_lesson(_fixture("transcript_quadratic.txt"), "", ["IMG_0938.jpeg"])
    kinds = [s.section_type for s in lesson.sections]
    assert kinds[0] == "example" and "definition" in kinds and kinds.count("exercise") == 2
    assert lesson.learning_objectives and "平方完成" in lesson.learning_objectives[0]
    assert all("めあて" not in s.content for s in lesson.sections)
    ex = next(s for s in lesson.sections if s.title == "練習11 (2)").exercise
    assert ex.problem == "$2x^2+8x+7$" and ex.answer == "$=2(x+2)^2-1$"


# ---------------------------------------------------------------------------
# Orchestration with a fake backend
# ---------------------------------------------------------------------------

class _FakeBackend:
    name = "fake"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def generate(self, model, prompt, images=(), max_tokens=0, on_delta=None, **kw):
        self.calls.append({"images": list(images), "prompt": prompt})
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return LLMResult(text=r, backend="fake", model=model, device="gpu", elapsed_seconds=1.0)


def _img(tmp_path):
    p = tmp_path / "b.png"
    p.write_bytes(b"x")
    return p


def test_analyze_board_two_stages(tmp_path):
    img = _img(tmp_path)
    backend = _FakeBackend([_fixture("transcript_permutation.txt"),
                            "unit: 順列\ntitle: 順列の総数\nsummary: まとめ\nteacher_note: 注意\n# example: 例5 | B4\nnote: 指導"])
    phases = []
    res = bp.analyze_board(backend, "m", [img], on_phase=lambda p, m: phases.append(p))
    assert res.lesson is not None and res.error is None and not res.structure_fallback
    assert res.lesson.lesson_title == "順列の総数" and res.lesson.summary == "まとめ"
    assert phases == ["transcribe", "structure"]
    assert backend.calls[0]["images"] == [img] and backend.calls[1]["images"] == []
    assert "【書き起こし】" in backend.calls[1]["prompt"] and "B1（列1）（板書のタイトル）:" in backend.calls[1]["prompt"]
    assert set(res.timings) == {"transcribe", "structure", "total"}


def test_analyze_board_falls_back_when_structure_fails(tmp_path):
    backend = _FakeBackend([_fixture("transcript_permutation.txt"), LLMError("timeout")])
    res = bp.analyze_board(backend, "m", [_img(tmp_path)])
    assert res.lesson is not None and res.structure_fallback
    assert any("失敗" in w for w in res.warnings)


def test_analyze_board_reports_transcription_failure(tmp_path):
    res = bp.analyze_board(_FakeBackend([LLMError("GPU busy")]), "m", [_img(tmp_path)])
    assert res.lesson is None and "書き起こしに失敗" in res.error
