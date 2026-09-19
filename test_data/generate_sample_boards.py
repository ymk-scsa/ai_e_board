"""
Generates high-resolution simulated blackboard images for testing.
Creates 2 sample board plans:
1. Permutations (順列の考え方と公式 nPr)
2. Quadratic Function Completing the Square (2次関数の平方完成)
"""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from PIL import Image, ImageDraw, ImageFont
import config


TEST_DATA_DIR = config.TEST_DATA_DIR


def create_blackboard_base(width: int = 1600, height: int = 900) -> Image.Image:
    """Create a high-res dark green chalkboard texture canvas."""
    img = Image.new("RGB", (width, height), color=(26, 52, 40))
    draw = ImageDraw.Draw(img)

    # Add chalkboard wooden frame
    draw.rectangle([0, 0, width, height], outline=(68, 42, 22), width=18)
    draw.rectangle([18, 18, width - 18, height - 18], outline=(120, 90, 60), width=4)

    # Subtle chalk dust effect / grid line
    for y in range(80, height - 80, 160):
        draw.line([(30, y), (width - 30, y)], fill=(32, 60, 48), width=1)

    return img


def draw_permutation_board() -> Path:
    """Generate Board 1: Permutations (高校数学A 順列)."""
    img = create_blackboard_base()
    draw = ImageDraw.Draw(img)

    # Colors
    c_white = (245, 250, 248)
    c_yellow = (255, 230, 110)
    c_red = (255, 130, 130)
    c_blue = (160, 220, 240)

    # Title & Objective Box (Top-Left)
    draw.rectangle([40, 35, 750, 140], outline=c_white, width=3)
    draw.text((60, 48), "数学A  単元: 場合の数と確率", fill=c_white)
    draw.text((60, 75), "【本時の目標】", fill=c_yellow)
    draw.text((60, 100), "・順列の意味を理解し、nPr の計算ができるようになる", fill=c_white)

    # Date / Class Info
    draw.text((1350, 45), "9月1日 (火) 第3限", fill=c_white)

    # Section 1: Introduction Problem (Left column)
    draw.text((50, 160), "【導入問題】", fill=c_yellow)
    draw.text((60, 200), "4曲 a, b, c, d から 異なる3曲を選んで演奏する。", fill=c_white)
    draw.text((60, 235), "演奏する「曲の順序」を考えると、何通りあるか？", fill=c_white)

    # Tree / Sequential Choice illustration
    draw.rectangle([60, 280, 720, 430], outline=c_blue, width=2)
    draw.text((80, 300), "〈考え方〉", fill=c_blue)
    draw.text((100, 335), "・1曲目 の選び方  →  4 通り (a, b, c, d)", fill=c_white)
    draw.text((100, 365), "・2曲目 の選び方  →  3 通り (残り3つ)", fill=c_white)
    draw.text((100, 395), "・3曲目 の選び方  →  2 通り (残り2つ)", fill=c_white)

    # Multiplication principle formula
    draw.text((100, 455), "積の法則より:", fill=c_white)
    draw.rectangle([250, 445, 560, 500], outline=c_yellow, width=3)
    draw.text((270, 460), "4 × 3 × 2 = 24 (通り)", fill=c_yellow)

    # Section 2: General Formula (Right Column)
    draw.text((820, 160), "【定義・公式】 順列 (Permutation)", fill=c_yellow)
    draw.rectangle([820, 200, 1530, 430], outline=c_red, width=4)
    draw.text((850, 220), "異なる n 個のものから 異なる r 個を選んで並べる順列の総数", fill=c_white)
    draw.text((950, 270), "nPr = n × (n - 1) × (n - 2) × … × (n - r + 1)", fill=c_yellow)
    draw.text((1050, 320), "＼─────── r 個の積 ───────／", fill=c_red)
    draw.text((850, 370), "例: 4P3 = 4 × 3 × 2 = 24", fill=c_white)

    # Section 3: Exercise Problem (Bottom)
    draw.text((50, 540), "【練習問題】", fill=c_yellow)
    draw.rectangle([40, 580, 1530, 760], outline=c_white, width=2)
    draw.text((70, 605), "問1. 次の値を求めよ。", fill=c_white)
    draw.text((100, 645), "(1) 5P2  = 5 × 4 = 20", fill=c_white)
    draw.text((600, 645), "(2) 6P3  = 6 × 5 × 4 = 120", fill=c_white)
    draw.text((1100, 645), "(3) 4P4  = 4! = 24", fill=c_white)

    draw.text((70, 695), "問2. 5人の生徒の中から、走る順番を決めて3人のリレー選手を選ぶ方法は何通りか。", fill=c_white)
    draw.text((100, 725), "解: 5P3 = 5 × 4 × 3 = 60通り", fill=c_yellow)

    # Section 4: Summary (Bottom Bar)
    draw.rectangle([40, 780, 1530, 860], outline=c_yellow, width=2)
    draw.text((60, 800), "【まとめ】", fill=c_yellow)
    draw.text((180, 800), "・「並べる順序」を区別するときは「順列 nPr」を使う！", fill=c_white)
    draw.text((180, 828), "・n からスタートして 1 ずつ減らしながら r 個の数を掛け算する。", fill=c_white)

    out_path = TEST_DATA_DIR / "sample_permutation_board.png"
    img.save(out_path)
    print(f"Generated sample board 1: {out_path}")
    return out_path


def draw_completing_square_board() -> Path:
    """Generate Board 2: Quadratic Completing the Square (高校数学I 2次関数の平方完成)."""
    img = create_blackboard_base()
    draw = ImageDraw.Draw(img)

    # Colors
    c_white = (245, 250, 248)
    c_yellow = (255, 230, 110)
    c_red = (255, 130, 130)
    c_blue = (160, 220, 240)

    # Title & Objective Box (Top-Left)
    draw.rectangle([40, 35, 780, 140], outline=c_white, width=3)
    draw.text((60, 48), "数学I  単元: 2次関数とグラフ", fill=c_white)
    draw.text((60, 75), "【本時の目標】", fill=c_yellow)
    draw.text((60, 100), "・2次式を平方完成し、グラフの頂点と軸の座標を求める", fill=c_white)

    draw.text((1350, 45), "9月2日 (水) 第2限", fill=c_white)

    # Standard Form & Vertex Form Box (Right Top)
    draw.rectangle([840, 35, 1530, 140], outline=c_red, width=3)
    draw.text((860, 50), "基本形: y = a(x - p)² + q", fill=c_yellow)
    draw.text((860, 85), "→ 頂点 (p, q),  軸の方程式 x = p", fill=c_white)

    # Section 1: Example Problem
    draw.text((50, 170), "【例題】 2次関数 y = 2x² - 4x + 5 を平方完成し、頂点と軸を求めよ。", fill=c_yellow)
    
    # Step-by-step box
    draw.rectangle([50, 215, 820, 520], outline=c_blue, width=2)
    draw.text((70, 235), "〈平方完成の変形手順〉", fill=c_blue)
    draw.text((70, 270), "y = 2(x² - 2x) + 5", fill=c_white)
    draw.text((100, 300), "↑ x² の係数 2 で括る", fill=c_yellow)
    draw.text((70, 335), "  = 2{(x - 1)² - 1²} + 5", fill=c_white)
    draw.text((100, 365), "↑ x の係数の半分 (-1) の2乗", fill=c_red)
    draw.text((70, 400), "  = 2(x - 1)² - 2 + 5", fill=c_white)
    draw.text((70, 440), "  = 2(x - 1)² + 3", fill=c_yellow)
    draw.text((70, 480), "答: 頂点 (1, 3),  軸: 直線 x = 1", fill=c_yellow)

    # Section 2: General Method Summary (Right middle)
    draw.rectangle([870, 215, 1530, 520], outline=c_yellow, width=3)
    draw.text((900, 240), "【平方完成の鉄則 3ステップ】", fill=c_yellow)
    draw.text((920, 285), "① x² の係数 a で x の項までを括る", fill=c_white)
    draw.text((920, 335), "② かっこの中を (x + □/2)² - (□/2)² の形に変形", fill=c_white)
    draw.text((920, 385), "③ 分配法則で a を外に出し、定数項を整理する", fill=c_white)
    draw.text((920, 440), "※ 符号ミスに最大の注意！", fill=c_red)

    # Section 3: Practice Problem (Bottom)
    draw.text((50, 550), "【練習問題】", fill=c_yellow)
    draw.rectangle([40, 590, 1530, 770], outline=c_white, width=2)
    draw.text((70, 615), "問. 次の2次関数のグラフの頂点と軸を求めよ。", fill=c_white)
    draw.text((100, 660), "(1) y = x² - 6x + 2  →  y = (x - 3)² - 7   頂点 (3, -7), 軸 x = 3", fill=c_white)
    draw.text((100, 715), "(2) y = 3x² + 12x - 1 →  y = 3(x + 2)² - 13 頂点 (-2, -13), 軸 x = -2", fill=c_white)

    # Section 4: Summary (Bottom Bar)
    draw.rectangle([40, 790, 1530, 870], outline=c_yellow, width=2)
    draw.text((60, 810), "【まとめ】", fill=c_yellow)
    draw.text((180, 810), "・平方完成によって、一般形 y=ax²+bx+c からグラフの頂点と軸が一目で分かる！", fill=c_white)

    out_path = TEST_DATA_DIR / "sample_quadratic_board.png"
    img.save(out_path)
    print(f"Generated sample board 2: {out_path}")
    return out_path


if __name__ == "__main__":
    draw_permutation_board()
    draw_completing_square_board()
