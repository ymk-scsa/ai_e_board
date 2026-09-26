"""
Demo lessons (hand-made from the real boards uploads/IMG_0939.png and uploads/IMG_0938.jpeg).
Used only when the teacher explicitly chooses the demo in STEP 2 and as System B presets; also the
quality reference for tools/eval_boards.py.
"""

from models.schemas import Lesson, Section, Formula, ExampleProblem, Exercise, VisualAnnotation


def get_mock_lesson_for_sample(image_name: str) -> Lesson:
    """Generate high-fidelity structured Lesson when running in offline/mock simulation mode."""
    if "quadratic" in image_name.lower():
        return Lesson(
            subject="数学",
            grade="高校1年 (数学I)",
            unit="2次関数とグラフ",
            lesson_title="2次関数の平方完成とグラフの頂点",
            learning_objectives=[
                "2次式 ax² + bx + c を平方完成の形 a(x-p)² + q に変形できる。",
                "平方完成により放物線の頂点 (p, q) と軸の方程式 x=p を求めることができる。",
            ],
            introduction="一般形 y = 2x² - 4x + 5 からグラフの頂点を直接読み取ることは困難であるため、標準形に変形する平方完成を学ぶ。",
            sections=[
                Section(
                    section_id="sec_01",
                    section_type="definition",
                    title="基本形と頂点・軸の関係",
                    content="2次関数を $y = a(x-p)^2 + q$ の形に直すと、頂点の座標が $(p, q)$、軸の方程式が $x = p$ であることが直ちにわかる。",
                    formulas=[
                        Formula(
                            raw_text="y = a(x - p)^2 + q",
                            latex="y = a(x - p)^2 + q",
                            description="2次関数の標準形（頂点・軸表示）",
                            is_key_formula=True,
                        )
                    ],
                    visual_annotations=[
                        VisualAnnotation(element_type="box", target_text="y = a(x - p)^2 + q", color="red", note="最重要基本形")
                    ],
                    order=1,
                ),
                Section(
                    section_id="sec_02",
                    section_type="example",
                    title="例題：平方完成の計算",
                    content="2次関数 $y = 2x^2 - 4x + 5$ を平方完成し、頂点と軸を求めよう。",
                    formulas=[
                        Formula(
                            raw_text="y = 2(x - 1)^2 + 3",
                            latex="y = 2(x - 1)^2 + 3",
                            description="変形完了した式",
                            is_key_formula=True,
                        )
                    ],
                    example=ExampleProblem(
                        title="例題1",
                        problem="2次関数 $y = 2x^2 - 4x + 5$ を平方完成し、放物線の頂点と軸を求めよ。",
                        approach="x² の係数 2 で x の項までを括り、かっこの中で $(x - 1)^2 - 1$ の形を作る。",
                        solution_steps=[
                            "$y = 2(x^2 - 2x) + 5$ (係数2で括る)",
                            r"$y = 2\{(x - 1)^2 - 1^2\} + 5$ (xの係数の半分の2乗を引く)",
                            "$y = 2(x - 1)^2 - 2 + 5$ (分配法則)",
                            "$y = 2(x - 1)^2 + 3$",
                        ],
                        answer="頂点 $(1, 3)$, 軸の直線 $x = 1$",
                        teaching_notes="符号ミス（-1の2乗の引き忘れ、カッコの外への展開時の掛け忘れ）に注意を促す。",
                    ),
                    order=2,
                ),
                Section(
                    section_id="sec_03",
                    section_type="exercise",
                    title="練習問題",
                    content="各自でノートに平方完成の計算を行い、頂点と軸を求めましょう。",
                    exercise=Exercise(
                        title="練習1",
                        problem="次の2次関数のグラフの頂点と軸を求めよ。\n(1) $y = x^2 - 6x + 2$\n(2) $y = 3x^2 + 12x - 1$",
                        hint="(1) はそのまま $(x-3)^2$ を作る。(2) はまず 3 で括る。",
                        answer="(1) 頂点 $(3, -7)$, 軸 $x = 3$ / (2) 頂点 $(-2, -13)$, 軸 $x = -2$",
                        solution_steps=[
                            "(1) $y = (x-3)^2 - 9 + 2 = (x-3)^2 - 7$",
                            r"(2) $y = 3(x^2+4x) - 1 = 3\{(x+2)^2-4\} - 1 = 3(x+2)^2 - 13$",
                        ],
                    ),
                    order=3,
                ),
            ],
            summary="平方完成の3ステップ（括る → 半分の2乗を作る → 定数項を整理する）を確実に身につけ、一般形からグラフを正確に描けるようにしよう。",
            notes_for_teacher="特に分配法則で係数aを外に出す際の符号と定数の計算ミスが頻発するため、机間巡視で確認すること。",
        )

    # Default: Permutation board
    return Lesson(
        subject="数学",
        grade="高校1年 (数学A)",
        unit="場合の数と確率 - 順列",
        lesson_title="順列の考え方と計算公式 nPr",
        learning_objectives=[
            "異なるものからいくつかを選んで並べる「順列」の意味を理解する。",
            "積の法則を活用して順列の総数を求め、記号 nPr の計算ができる。",
        ],
        introduction="4曲から3曲を選んで演奏順を決める身近な問題を題材に、順序を区別する並べ方の総数を考えます。",
        sections=[
            Section(
                section_id="sec_01",
                section_type="introduction",
                title="導入課題：曲の演奏順",
                content="4曲 a, b, c, d から異なる3曲を選んで演奏するとき、演奏する「曲の順序」を考えると何通りあるだろうか？",
                example=ExampleProblem(
                    title="導入問題",
                    problem="4曲 a, b, c, d から異なる3曲を選んで曲順を決める方法は何通りあるか。",
                    approach="1曲目、2曲目、3曲目と順番に選ぶときの選択肢の数を考える。",
                    solution_steps=[
                        "1曲目: 4通り (a, b, c, d のいずれか)",
                        "2曲目: 3通り (1曲目で選んだものを除く3通り)",
                        "3曲目: 2通り (残り2通り)",
                        "積の法則より: $4 \\times 3 \\times 2 = 24$",
                    ],
                    answer="24 通り",
                ),
                order=1,
            ),
            Section(
                section_id="sec_02",
                section_type="formula",
                title="順列の定義と計算公式",
                content="異なる $n$ 個のものから異なる $r$ 個を取り出して1列に並べる並べ方を **順列 (Permutation)** といい、その総数を ${}_{n}P_{r}$ で表す。",
                formulas=[
                    Formula(
                        raw_text="nPr = n * (n-1) * (n-2) * ... * (n-r+1)",
                        latex="{}_{n}P_{r} = n(n-1)(n-2)\\cdots(n-r+1)",
                        description="順列の総数（nから1ずつ減らしてr個の数を掛け算する）",
                        is_key_formula=True,
                    ),
                    Formula(
                        raw_text="4P3 = 4 * 3 * 2 = 24",
                        latex="{}_{4}P_{3} = 4 \\times 3 \\times 2 = 24",
                        description="4曲から3曲選ぶ順列の計算例",
                        is_key_formula=True,
                    ),
                ],
                visual_annotations=[
                    VisualAnnotation(element_type="box", target_text="{}_{n}P_{r}", color="red", note="最重要公式")
                ],
                order=2,
            ),
            Section(
                section_id="sec_03",
                section_type="exercise",
                title="練習問題",
                content="順列の計算公式を使って、次の値を求めましょう。",
                exercise=Exercise(
                    title="練習問題（問1・問2）",
                    problem="問1. 次の値を求めよ。\n(1) ${}_{5}P_{2}$\n(2) ${}_{6}P_{3}$\n(3) ${}_{4}P_{4}$\n\n問2. 5人の生徒の中から走る順番を決めて3人のリレー選手を選ぶ方法は何通りか。",
                    hint="問1: 公式通り掛け算する。問2: 順番を区別するので順列を利用する。",
                    answer="問1: (1) 20, (2) 120, (3) 24 / 問2: 60通り (${}_{5}P_{3} = 5 \\times 4 \\times 3 = 60$)",
                ),
                order=3,
            ),
        ],
        summary="「並べる順序」を区別するときは順列 ${}_{n}P_{r}$ を用いる。$n$ からスタートして 1 ずつ減らしながら $r$ 個の数を掛け合わせる！",
        notes_for_teacher="生徒が組合せ(nCr)と混同しないよう、「順序を区別する」点（リレーの走順など）を強く意識づけること。",
    )
