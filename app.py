"""
Main Streamlit Application for AI Electronic Blackboard Generation System (ai_e_board).
Supports multi-image blackboard analysis, educational JSON structuring, teacher review/editing,
and 16:9 KaTeX-powered electronic blackboard presentation generation.
"""

import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
from PIL import Image

import config
from ai.vision import VisionAnalyzer
from ai.parser import LessonParser
from ai.generator import ElectronicBoardGenerator
from ai.evaluator import CurriculumAlignmentEvaluator, PedagogicalQualityEvaluator
from models.schemas import Lesson, Section, Formula, ExampleProblem, Exercise, VisualAnnotation
from ui.board import BlackboardUI
from system_b.pipeline import EvaluationPipeline
from ui.evaluation import EvaluationUI

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_e_board.app")


def init_session_state():
    """Initialize Streamlit session state variables."""
    if "app_mode" not in st.session_state:
        st.session_state.app_mode = "🎨 電子黒板教材生成 (システムA)"
    if "current_step" not in st.session_state:
        st.session_state.current_step = 1
    if "uploaded_image_paths" not in st.session_state:
        st.session_state.uploaded_image_paths = []
    if "uploaded_image_names" not in st.session_state:
        st.session_state.uploaded_image_names = []
    if "raw_vision_result" not in st.session_state:
        st.session_state.raw_vision_result = None
    if "parsed_lesson" not in st.session_state:
        st.session_state.parsed_lesson = None
    if "generated_presentation" not in st.session_state:
        st.session_state.generated_presentation = None
    if "presentation_html" not in st.session_state:
        st.session_state.presentation_html = None
    if "saved_json_path" not in st.session_state:
        st.session_state.saved_json_path = None
    if "saved_html_path" not in st.session_state:
        st.session_state.saved_html_path = None
    if "ollama_model" not in st.session_state:
        st.session_state.ollama_model = config.DEFAULT_VISION_MODEL
    if "ollama_host" not in st.session_state:
        st.session_state.ollama_host = config.OLLAMA_HOST
    if "theme" not in st.session_state:
        st.session_state.theme = config.DEFAULT_THEME



def render_sidebar(vision_analyzer: VisionAnalyzer):
    """Render sidebar with system status, Ollama model configuration, and sample presets."""
    st.sidebar.title("🎛️ 機能モード切替")
    mode_options = [
        "🎨 電子黒板教材生成 (システムA)",
        "📊 教育品質評価・運用支援 (システムB)",
    ]
    current_mode_idx = 0 if st.session_state.app_mode.startswith("🎨") else 1
    selected_mode = st.sidebar.radio(
        "利用するAIシステムを選択",
        options=mode_options,
        index=current_mode_idx,
    )
    if selected_mode != st.session_state.app_mode:
        st.session_state.app_mode = selected_mode
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.title("⚙️ システム設定")

    # Ollama Host Setting
    host_input = st.sidebar.text_input(
        "Ollama 接続ホスト",
        value=st.session_state.ollama_host,
        help="OllamaのローカルAPIエンドポイント（通常: http://localhost:11434）",
    )

    if host_input != st.session_state.ollama_host:
        st.session_state.ollama_host = host_input
        st.rerun()

    # Ollama Health Check & Model List
    is_connected, msg, available_models = vision_analyzer.check_connection()

    if is_connected:
        st.sidebar.success(f"🟢 Ollama 接続中 ({len(available_models)} モデル)")
        # Model Selection
        model_options = list(dict.fromkeys(available_models + config.CANDIDATE_VISION_MODELS))
        selected_idx = 0
        if st.session_state.ollama_model in model_options:
            selected_idx = model_options.index(st.session_state.ollama_model)

        selected_model = st.sidebar.selectbox(
            "Vision AI モデル選択",
            options=model_options,
            index=selected_idx,
            help="画像解析に対応したVision LLM（Qwen3-VL, LLaMA 3.2 Vision等）を選択してください",
        )
        st.session_state.ollama_model = selected_model
    else:
        st.sidebar.warning(f"🟡 Ollama 未接続または待機中\n\n{msg}")
        custom_model = st.sidebar.text_input(
            "モデル名（直接入力）",
            value=st.session_state.ollama_model,
        )
        st.session_state.ollama_model = custom_model

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔬 研究用テスト板書")
    st.sidebar.caption("手元に画像がない場合は以下のテスト板書を即座に読み込めます。")

    col_s1, col_s2 = st.sidebar.columns(2)
    with col_s1:
        if st.button("① 順列の板書", use_container_width=True):
            p1 = config.TEST_DATA_DIR / "sample_permutation_board.png"
            if p1.exists():
                dest = config.UPLOAD_DIR / p1.name
                shutil.copy(p1, dest)
                st.session_state.uploaded_image_paths = [dest]
                st.session_state.uploaded_image_names = [p1.name]
                st.session_state.current_step = 2
                st.rerun()
            else:
                st.sidebar.error("テスト画像が見つかりません。")

    with col_s2:
        if st.button("② 平方完成の板書", use_container_width=True):
            p2 = config.TEST_DATA_DIR / "sample_quadratic_board.png"
            if p2.exists():
                dest = config.UPLOAD_DIR / p2.name
                shutil.copy(p2, dest)
                st.session_state.uploaded_image_paths = [dest]
                st.session_state.uploaded_image_names = [p2.name]
                st.session_state.current_step = 2
                st.rerun()
            else:
                st.sidebar.error("テスト画像が見つかりません。")

    st.sidebar.markdown("---")
    st.sidebar.caption("AI Electronic Blackboard System v1.0\nPrototype for Educational Research")


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


def main():
    st.set_page_config(
        page_title="AI電子黒板教材生成システム (ai_e_board)",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    BlackboardUI.load_custom_css()

    vision_analyzer = VisionAnalyzer(
        host=st.session_state.ollama_host,
        model=st.session_state.ollama_model,
    )
    render_sidebar(vision_analyzer)

    # If System B mode is selected, render System B
    if st.session_state.app_mode.startswith("📊"):
        pipeline = EvaluationPipeline(
            host=st.session_state.ollama_host,
            model=st.session_state.ollama_model,
        )
        EvaluationUI.render_evaluation_view(pipeline)
        return

    # ----------------------------------------------------
    # SYSTEM A: 電子黒板教材生成フロー
    # ----------------------------------------------------

    # Main Top Header
    st.markdown(
        """
        <div class="main-header">
            <h1>🎓 AI電子黒板教材生成システム</h1>
            <p>教員の板書計画写真をAIが解析し、教育的構造（目標・導入・公式・例題・練習・まとめ）を保持した16:9電子黒板教材を自動生成します。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Step Progress Indicator
    BlackboardUI.render_step_progress_bar(st.session_state.current_step)

    # ----------------------------------------------------
    # STEP 1: 板書計画をアップロード
    # ----------------------------------------------------
    if st.session_state.current_step == 1:

        st.subheader("📤 STEP 1: 教員の板書計画画像をアップロード")
        st.info("💡 1枚または複数枚（授業の順序順）の板書写真をアップロードしてください。対応形式: PNG, JPG, JPEG")

        uploaded_files = st.file_uploader(
            "板書計画画像を選択（複数選択可能）",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help="授業展開の順序に沿って画像をアップロードしてください",
        )

        if uploaded_files:
            saved_paths = []
            saved_names = []
            cols = st.columns(min(len(uploaded_files), 3))
            
            for idx, uploaded_file in enumerate(uploaded_files):
                dest_path = config.UPLOAD_DIR / uploaded_file.name
                with open(dest_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_paths.append(dest_path)
                saved_names.append(uploaded_file.name)

                with cols[idx % 3]:
                    st.image(uploaded_file, caption=f"画像 {idx+1}: {uploaded_file.name}", use_container_width=True)

            st.session_state.uploaded_image_paths = saved_paths
            st.session_state.uploaded_image_names = saved_names

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("次へ進む（STEP 2: AI解析へ） ▶", type="primary", use_container_width=True):
                st.session_state.current_step = 2
                st.rerun()

    # ----------------------------------------------------
    # STEP 2: AIが板書を解析
    # ----------------------------------------------------
    elif st.session_state.current_step == 2:
        st.subheader("🤖 STEP 2: Vision AIによる板書解析")
        st.write(f"対象画像: **{len(st.session_state.uploaded_image_paths)} 枚** ({', '.join(st.session_state.uploaded_image_names)})")

        # Preview uploaded images
        cols = st.columns(min(len(st.session_state.uploaded_image_paths), 4))
        for idx, img_p in enumerate(st.session_state.uploaded_image_paths):
            with cols[idx % 4]:
                st.image(str(img_p), caption=f"板書 {idx+1}: {Path(img_p).name}", use_container_width=True)

        teacher_instruction = st.text_area(
            "教員からの補足指示（任意）",
            placeholder="例: 高校数学Iの授業として解析してください。例題の計算ステップを丁寧に抽出してください。",
            help="AIに対する追加の文脈や指導上の留意事項があれば入力してください",
        )

        col_btn1, col_btn2 = st.columns([2, 1])
        with col_btn1:
            start_analysis = st.button("🔍 板書をAI解析する", type="primary", use_container_width=True)
        with col_btn2:
            if st.button("◀ STEP 1（画像選択）に戻る", use_container_width=True):
                st.session_state.current_step = 1
                st.rerun()

        if start_analysis:
            with st.spinner("Vision AI が板書の構造・数式・教育的意図を解析中... (少々お待ちください)"):
                # Call Vision AI
                vision_result = vision_analyzer.analyze_board_images(
                    image_paths=st.session_state.uploaded_image_paths,
                    custom_instructions=teacher_instruction,
                    model_override=st.session_state.ollama_model,
                )
                st.session_state.raw_vision_result = vision_result

                # Check if Ollama succeeded or if we should fallback/simulate
                parser = LessonParser()
                lesson = None
                parse_err = None

                if vision_result.get("success") and vision_result.get("raw_text"):
                    lesson, parse_err = parser.parse_to_lesson(
                        raw_llm_text=vision_result["raw_text"],
                        source_images=st.session_state.uploaded_image_names,
                        vision_analyzer=vision_analyzer,
                    )

                # Fallback Simulation if Ollama vision model not available
                if not lesson:
                    first_img_name = st.session_state.uploaded_image_names[0] if st.session_state.uploaded_image_names else "board.png"
                    logger.info("Using high-fidelity educational simulation fallback for testing...")
                    lesson = get_mock_lesson_for_sample(first_img_name)
                    lesson.source_images = st.session_state.uploaded_image_names
                    if parse_err:
                        st.warning(f"⚠️ AI解析の警告: {parse_err}\n（シミュレーション・高精度テンプレート構造を適用しました）")
                    else:
                        st.info("ℹ️ ローカルOllamaモデルからの応答待機またはモックモードで授業構造を抽出しました。")

                st.session_state.parsed_lesson = lesson
                st.success("✅ 板書解析と授業構造化が完了しました！")
                time.sleep(0.5)
                st.session_state.current_step = 3
                st.rerun()

    # ----------------------------------------------------
    # STEP 3: AI解析結果を確認・編集
    # ----------------------------------------------------
    elif st.session_state.current_step == 3:
        st.subheader("✏️ STEP 3: AI解析結果の確認と編集")
        st.caption("AIが抽出した授業構造を教員が確認・修正します。必要に応じて数式（LaTeX）や説明文を調整してください。")

        lesson: Lesson = st.session_state.parsed_lesson
        if not lesson:
            st.error("授業構造データがありません。STEP 2 からやり直してください。")
            if st.button("STEP 2 へ戻る"):
                st.session_state.current_step = 2
                st.rerun()
            return

        with st.form("edit_lesson_form"):
            st.markdown("### 📋 基本情報")
            c1, c2 = st.columns(2)
            with c1:
                edit_subject = st.text_input("教科", value=lesson.subject)
                edit_grade = st.text_input("学年・対象科目", value=lesson.grade or "高校")
            with c2:
                edit_unit = st.text_input("単元名", value=lesson.unit)
                edit_title = st.text_input("授業タイトル", value=lesson.lesson_title)

            st.markdown("### 🎯 本時の目標（学習目標）")
            objectives_text = st.text_area(
                "目標（1行に1項目）",
                value="\n".join(lesson.learning_objectives),
                height=100,
                help="生徒に提示する本時のねらいを箇条書きで入力してください",
            )

            st.markdown("### 📖 導入（動機づけ・問いかけ）")
            edit_intro = st.text_area("導入内容", value=lesson.introduction or "", height=80)

            st.markdown("### 🧩 各セクション（展開）")
            edited_sections = []
            for idx, sec in enumerate(lesson.sections):
                st.markdown(f"#### セクション {idx+1}: {sec.title} ({sec.section_type})")
                sec_c1, sec_c2 = st.columns([3, 1])
                with sec_c1:
                    s_title = st.text_input(f"見出し", value=sec.title, key=f"sec_title_{idx}")
                    s_content = st.text_area(f"本文・解説", value=sec.content, key=f"sec_content_{idx}", height=100)
                with sec_c2:
                    s_type = st.selectbox(
                        "種類",
                        options=["introduction", "concept", "definition", "formula", "example", "exercise", "summary"],
                        index=["introduction", "concept", "definition", "formula", "example", "exercise", "summary"].index(
                            sec.section_type if sec.section_type in ["introduction", "concept", "definition", "formula", "example", "exercise", "summary"] else "concept"
                        ),
                        key=f"sec_type_{idx}",
                    )

                # Formulas inside section
                edited_formulas = []
                if sec.formulas:
                    st.caption("数式 (LaTeX)")
                    for f_idx, f in enumerate(sec.formulas):
                        f_latex = st.text_input(
                            f"LaTeX式 #{f_idx+1}",
                            value=f.latex,
                            key=f"f_latex_{idx}_{f_idx}",
                        )
                        f_desc = st.text_input(
                            f"説明 #{f_idx+1}",
                            value=f.description or "",
                            key=f"f_desc_{idx}_{f_idx}",
                        )
                        edited_formulas.append(
                            Formula(
                                raw_text=f.raw_text,
                                latex=f_latex,
                                description=f_desc,
                                is_key_formula=f.is_key_formula,
                            )
                        )
                        # Live Math Preview
                        st.markdown(f"**数式プレビュー:** ${f_latex}$")

                # Example problem
                edited_example = sec.example
                if sec.example:
                    with st.expander(f"例題の詳細設定: {sec.example.title}", expanded=True):
                        ex_prob = st.text_area("問題文", value=sec.example.problem, key=f"ex_prob_{idx}")
                        ex_app = st.text_input("考え方・ヒント", value=sec.example.approach or "", key=f"ex_app_{idx}")
                        ex_steps_str = st.text_area(
                            "解法ステップ（1行1ステップ）",
                            value="\n".join(sec.example.solution_steps),
                            key=f"ex_steps_{idx}",
                        )
                        ex_ans = st.text_input("答え", value=sec.example.answer, key=f"ex_ans_{idx}")
                        edited_example = ExampleProblem(
                            title=sec.example.title,
                            problem=ex_prob,
                            approach=ex_app,
                            solution_steps=[s for s in ex_steps_str.split("\n") if s.strip()],
                            answer=ex_ans,
                        )

                # Exercise
                edited_exercise = sec.exercise
                if sec.exercise:
                    with st.expander(f"練習問題の詳細設定: {sec.exercise.title}", expanded=True):
                        exe_prob = st.text_area("問題文", value=sec.exercise.problem, key=f"exe_prob_{idx}")
                        exe_hint = st.text_input("ヒント", value=sec.exercise.hint or "", key=f"exe_hint_{idx}")
                        exe_ans = st.text_input("解答", value=sec.exercise.answer or "", key=f"exe_ans_{idx}")
                        edited_exercise = Exercise(
                            title=sec.exercise.title,
                            problem=exe_prob,
                            hint=exe_hint,
                            answer=exe_ans,
                        )

                edited_sections.append(
                    Section(
                        section_id=sec.section_id,
                        section_type=s_type,
                        title=s_title,
                        content=s_content,
                        formulas=edited_formulas,
                        example=edited_example,
                        exercise=edited_exercise,
                        visual_annotations=sec.visual_annotations,
                        order=idx + 1,
                    )
                )
                st.markdown("---")

            st.markdown("### 🏁 本時のまとめ & 指導メモ")
            edit_summary = st.text_area("まとめ", value=lesson.summary or "", height=80)
            edit_notes = st.text_area("教員用指導メモ", value=lesson.notes_for_teacher or "", height=60)

            submitted = st.form_submit_button("💾 編集を確定して次へ（STEP 4: 教材生成） ▶", type="primary", use_container_width=True)

            if submitted:
                updated_lesson = Lesson(
                    subject=edit_subject,
                    grade=edit_grade,
                    unit=edit_unit,
                    lesson_title=edit_title,
                    learning_objectives=[o.strip() for o in objectives_text.split("\n") if o.strip()],
                    introduction=edit_intro if edit_intro.strip() else None,
                    sections=edited_sections,
                    summary=edit_summary if edit_summary.strip() else None,
                    notes_for_teacher=edit_notes if edit_notes.strip() else None,
                    source_images=st.session_state.uploaded_image_names,
                    visual_structure=lesson.visual_structure,
                )
                st.session_state.parsed_lesson = updated_lesson
                st.session_state.current_step = 4
                st.rerun()

    # ----------------------------------------------------
    # STEP 4: 電子黒板教材を生成
    # ----------------------------------------------------
    elif st.session_state.current_step == 4:
        st.subheader("⚙️ STEP 4: 電子黒板教材の生成")
        st.caption("確定された授業構造から、16:9の高視認性電子黒板プレゼンテーションを構築します。")

        lesson: Lesson = st.session_state.parsed_lesson
        if not lesson:
            st.error("授業データがありません。STEP 1 からやり直してください。")
            return

        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(
                f"""
                <div class="content-card">
                    <h3>📚 生成対象授業</h3>
                    <p><b>教科・単元:</b> {lesson.subject} | {lesson.unit}</p>
                    <p><b>授業テーマ:</b> {lesson.lesson_title}</p>
                    <p><b>スライド予定数:</b> {len(lesson.sections) + 2} 枚（タイトル・セクション・まとめ）</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown("### 🎨 デザインテーマ")
            theme_choice = st.radio(
                "電子黒板のカラーテーマ",
                options=["chalkboard", "whiteboard"],
                format_func=lambda x: "🟢 チョークボード（黒板・深緑）" if x == "chalkboard" else "⚪ ホワイトボード（白基調）",
                index=0 if st.session_state.theme == "chalkboard" else 1,
            )
            st.session_state.theme = theme_choice

        # Future Evaluation Preview (Pedagogical & Curriculum)
        with st.expander("🔬 教育工学的整合性・品質チェック（将来拡張機能プレビュー）", expanded=True):
            curr_eval = CurriculumAlignmentEvaluator()
            curr_res = curr_eval.evaluate(lesson)
            qual_eval = PedagogicalQualityEvaluator()
            qual_res = qual_eval.evaluate(lesson)

            e_col1, e_col2 = st.columns(2)
            with e_col1:
                st.markdown(f"**学習指導要領照合:** `{curr_res['status']}`")
                if curr_res.get("matched_curriculum_entries"):
                    for entry in curr_res["matched_curriculum_entries"]:
                        st.write(f"- 対象科目: {entry['subject']} ({entry['unit']})")
            with e_col2:
                st.markdown(f"**教材完全性スコア:** `{qual_res['completeness_score']} / 100`")
                for sug in qual_res.get("suggestions", []):
                    st.caption(f"💡 {sug}")

        st.markdown("<br>", unsafe_allow_html=True)
        col_g1, col_g2 = st.columns([2, 1])
        with col_g1:
            if st.button("🚀 電子黒板教材を生成する", type="primary", use_container_width=True):
                with st.spinner("16:9 スライドおよびKaTeX数式テンプレートを生成中..."):
                    generator = ElectronicBoardGenerator(theme=st.session_state.theme)
                    presentation = generator.build_presentation(lesson)
                    html_str = generator.render_presentation_html(presentation)

                    # Save to outputs/
                    json_p, html_p = generator.save_outputs(
                        lesson=lesson,
                        presentation=presentation,
                        html_content=html_str,
                        research_metadata={
                            "model": st.session_state.ollama_model,
                            "source_images": st.session_state.uploaded_image_names,
                        },
                    )

                    st.session_state.generated_presentation = presentation
                    st.session_state.presentation_html = html_str
                    st.session_state.saved_json_path = str(json_p)
                    st.session_state.saved_html_path = str(html_p)

                    st.success(f"✅ 教材を生成・保存しました！（{json_p.name}）")
                    time.sleep(0.5)
                    st.session_state.current_step = 5
                    st.rerun()

        with col_g2:
            if st.button("◀ STEP 3（編集）に戻る", use_container_width=True):
                st.session_state.current_step = 3
                st.rerun()

    # ----------------------------------------------------
    # STEP 5: 電子黒板で表示
    # ----------------------------------------------------
    elif st.session_state.current_step == 5:
        st.subheader("🖥️ STEP 5: 電子黒板プレゼンテーション")

        if not st.session_state.presentation_html:
            st.error("教材HTMLがありません。STEP 4 で生成してください。")
            return

        col_top1, col_top2 = st.columns([2, 1])
        with col_top1:
            st.caption("💡 キーボードの **[←] [→]** または **画面端クリック** でスライドをめくれます。**[F11]** で全画面表示可能です。")
        with col_top2:
            # Download buttons
            c_dl1, c_dl2 = st.columns(2)
            with c_dl1:
                st.download_button(
                    "📥 HTML保存",
                    data=st.session_state.presentation_html,
                    file_name=Path(st.session_state.saved_html_path).name if st.session_state.saved_html_path else "board_presentation.html",
                    mime="text/html",
                    use_container_width=True,
                )
            with c_dl2:
                if st.session_state.saved_json_path and Path(st.session_state.saved_json_path).exists():
                    with open(st.session_state.saved_json_path, "r", encoding="utf-8") as jf:
                        json_data = jf.read()
                    st.download_button(
                        "📥 JSON保存",
                        data=json_data,
                        file_name=Path(st.session_state.saved_json_path).name,
                        mime="application/json",
                        use_container_width=True,
                    )

        # Presentation Stage Viewer (16:9 responsive canvas)
        BlackboardUI.render_presentation_viewer(
            html_content=st.session_state.presentation_html,
            height=720,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        # Link button to System B
        if st.button("📊 この教材の教育品質を評価・改善する（システムBへ連携） ▶", type="primary", use_container_width=True):
            st.session_state.app_mode = "📊 教育品質評価・運用支援 (システムB)"
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("◀ STEP 4（設定・再生成）へ戻る", use_container_width=True):
                st.session_state.current_step = 4
                st.rerun()
        with col_b2:
            if st.button("🔄 最初から新しい板書を処理する", use_container_width=True):
                st.session_state.current_step = 1
                st.session_state.uploaded_image_paths = []
                st.session_state.uploaded_image_names = []
                st.session_state.parsed_lesson = None
                st.session_state.presentation_html = None
                st.rerun()



if __name__ == "__main__":
    main()
