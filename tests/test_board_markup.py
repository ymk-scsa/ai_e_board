"""Tests for ai/board_markup.py (LaTeX normalization and key/section canonicalization)."""

from ai.board_markup import _as_math_text, _canon_key, _canon_section, normalize_inline_math, normalize_latex, strip_wrapping


def test_normalize_permutation_and_symbols():
    assert normalize_latex("4P3") == "{}_{4}P_{3}"
    assert normalize_latex("_{15}P_{1}=15") == "{}_{15}P_{1}=15"
    assert normalize_latex("{}_{5}P_{5}") == "{}_{5}P_{5}"  # already canonical: unchanged
    assert normalize_latex("nPr=n(n-1)") == "{}_{n}P_{r}=n(n-1)"
    assert normalize_latex("5×4÷2") == r"5\times 4\div 2"


def test_normalize_inline_math():
    assert normalize_inline_math("答えは $4P3$ 通り") == "答えは ${}_{4}P_{3}$ 通り"
    assert normalize_inline_math("閉じていない $x^2") == "閉じていない x^2"
    assert normalize_inline_math("$$2x+1$$ と $y$") == "$$2x+1$$ と $y$"


def test_as_math_text_wraps_bare_formulas_only():
    assert _as_math_text("4×3×2=24") == r"$4\times 3\times 2=24$"
    assert _as_math_text("$=2(x-1)^2+3$") == "$=2(x-1)^2+3$"
    assert _as_math_text("1曲目: 4通り") == "1曲目: 4通り"
    assert _as_math_text("24通り") == "24通り"


def test_canonicalization_and_wrapping():
    assert _canon_key("答え") == "answer" and _canon_key("Title") == "title"
    assert _canon_section("例題") == "example" and _canon_section("練習問題") == "exercise"
    assert _canon_section("exercise") == "exercise" and _canon_section("xyz") is None
    assert strip_wrapping("```\nunit: 順列\n```") == "unit: 順列"
    assert strip_wrapping("<think>考え中</think>unit: 順列") == "unit: 順列"
