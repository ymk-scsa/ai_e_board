"""
Main Streamlit Application for AI Electronic Blackboard Generation System (ai_e_board).
Supports multi-image blackboard analysis, educational JSON structuring, teacher review/editing,
and 16:9 KaTeX-powered electronic blackboard presentation generation.
"""

import html
import json
import logging
import os
import shutil
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
from PIL import Image

import config
from ai.vision import VisionAnalyzer
from ai.generator import ElectronicBoardGenerator
from ai.evaluator import CurriculumAlignmentEvaluator, PedagogicalQualityEvaluator
from ai.security import safe_upload_path, validate_ollama_host
from models.schemas import Lesson, Section, Formula, ExampleProblem, Exercise, VisualAnnotation, lesson_fingerprint
from models.samples import get_mock_lesson_for_sample
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
    # 生成済みスライドがどの授業データから作られたか（授業データの変更後に古いスライドを使わないため）
    if "presentation_lesson_hash" not in st.session_state:
        st.session_state.presentation_lesson_hash = None
    if "llm_backend_kind" not in st.session_state:
        st.session_state.llm_backend_kind = config.LLM_BACKEND
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = None  # None: バックエンドの既定モデル
    if "ollama_host" not in st.session_state:
        st.session_state.ollama_host = config.OLLAMA_HOST
    if "theme" not in st.session_state:
        st.session_state.theme = config.DEFAULT_THEME
    # デモ用サンプル授業（解析結果ではない）を使用中かどうか
    if "is_mock" not in st.session_state:
        st.session_state.is_mock = False
    # 直近のAI解析失敗情報 {"message": str, "raw_text": str}
    if "analysis_error" not in st.session_state:
        st.session_state.analysis_error = None
    # アップロード保存先を分離するためのセッションID
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
    # 再実行ごとの再保存を避けるためのキャッシュ {upload_key: saved_path}
    if "upload_cache" not in st.session_state:
        st.session_state.upload_cache = {}


SECTION_TYPE_OPTIONS = ["introduction", "concept", "definition", "formula", "example", "exercise", "summary", "custom"]

DERIVED_STATE_KEYS = (
    "generated_presentation", "presentation_html", "saved_json_path", "saved_html_path",
    "presentation_lesson_hash", "eval_result", "eval_json_path", "eval_html_path",
)


def find_unconfirmed_fields(lesson: Lesson) -> List[str]:
    """Human-readable locations of fields the AI marked as 要確認 (unreadable / not on the board)."""
    found = []
    if "要確認" in (lesson.grade or ""):
        found.append("学年")
    if "要確認" in lesson.unit:
        found.append("単元名")
    for sec in lesson.sections:
        if sec.example and "要確認" in sec.example.answer:
            found.append(f"{sec.title} の答え")
        if sec.exercise and "要確認" in (sec.exercise.answer or ""):
            found.append(f"{sec.title} の解答")
    return found


def reset_derived_state() -> None:
    """Clear everything generated from the current lesson (slides, saved paths, evaluation)."""
    for key in DERIVED_STATE_KEYS:
        st.session_state[key] = None


def set_lesson(lesson: Optional[Lesson]) -> None:
    """Replace the working lesson; derived artifacts are invalidated whenever the content changes."""
    if lesson_fingerprint(lesson) != lesson_fingerprint(st.session_state.get("parsed_lesson")):
        reset_derived_state()
    st.session_state.parsed_lesson = lesson


def analyze_and_parse(
    vision_analyzer: VisionAnalyzer,
    image_paths: List[Path],
    image_names: List[str],
    custom_instructions: Optional[str] = None,
    model_override: Optional[str] = None,
    on_progress=None,
    on_phase=None,
) -> Tuple[Optional[Lesson], Optional[str], Dict[str, Any]]:
    """
    Run the two-stage board analysis (transcribe → structure). Never substitutes a mock lesson.
    Returns (lesson or None, error message or None, analysis info dict).
    """
    try:
        res = vision_analyzer.analyze_board_to_lesson(
            image_paths=image_paths,
            custom_instructions=custom_instructions,
            model_override=model_override,
            on_phase=on_phase,
            on_progress=on_progress,
        )
    except (FileNotFoundError, ValueError) as e:
        return None, f"Vision AI の呼び出しに失敗しました: {e}", {"success": False, "error": str(e), "raw_text": ""}

    info: Dict[str, Any] = {
        "success": res.lesson is not None,
        "error": res.error,
        "raw_text": res.transcript,          # 書き起こし（デバッグ表示用）
        "structure_text": res.structure,
        "structure_fallback": res.structure_fallback,
        "parse_warnings": res.warnings,
        "model": res.model,
        "backend": vision_analyzer.backend_name,
        "device": res.device,
        "elapsed_seconds": res.timings.get("total"),
        "timings": res.timings,
        "source_images": image_names,
        "is_mock": False,
    }
    if res.lesson is None:
        return None, res.error or "板書の解析に失敗しました。", info
    return res.lesson, None, info


def _copy_sample_to_uploads(src: Path) -> Path:
    """Copy a bundled test image into the per-session upload dir with a safe generated name."""
    dest = safe_upload_path(
        src.name, config.UPLOAD_DIR, st.session_state.session_id, config.ALLOWED_IMAGE_EXTENSIONS
    )
    shutil.copy(src, dest)
    return dest



@st.cache_resource(show_spinner="AI実行環境を確認しています…")
def get_vision_analyzer(backend_kind: str, ollama_host: str) -> VisionAnalyzer:
    """Backend discovery spawns CLI processes; cache the analyzer across reruns."""
    return VisionAnalyzer(host=ollama_host, backend_kind=backend_kind)


@st.cache_data(ttl=30, show_spinner=False)
def get_backend_status(_vision_analyzer: VisionAnalyzer, cache_key: tuple) -> Tuple[bool, str, List[str], str]:
    """(is_connected, message, models, device of the selected model). cache_key = (kind, host, model)."""
    ok, msg, models = _vision_analyzer.check_connection()
    device = _vision_analyzer.model_device(cache_key[2]) if ok else "unknown"
    return ok, msg, models, device


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

    # AI実行環境（ローカルLLMバックエンド）の選択
    backend_labels = {
        "auto": "自動（GPUで動く環境を優先）",
        "foundry": "Foundry Local",
        "ollama": "Ollama",
        "openai": "OpenAI互換サーバー",
    }
    kinds = list(backend_labels)
    selected_kind = st.sidebar.selectbox(
        "AI実行環境",
        options=kinds,
        index=kinds.index(st.session_state.llm_backend_kind) if st.session_state.llm_backend_kind in kinds else 0,
        format_func=lambda k: backend_labels[k],
        help="画像解析に使うローカルLLMの実行環境。自動では Foundry Local → Ollama の順にGPU動作を確認します。",
    )
    if selected_kind != st.session_state.llm_backend_kind:
        st.session_state.llm_backend_kind = selected_kind
        st.session_state.llm_model = None
        st.rerun()

    if selected_kind == "ollama":
        host_input = st.sidebar.text_input(
            "Ollama 接続ホスト",
            value=st.session_state.ollama_host,
            help="OllamaのローカルAPIエンドポイント（通常: http://localhost:11434）",
        )
        if host_input != st.session_state.ollama_host:
            # SSRF対策: 許可リスト外のホストには接続しない（直前の接続先を維持）
            try:
                validated_host = validate_ollama_host(host_input)
            except ValueError as e:
                st.sidebar.error(f"⛔ {e}\n\n直前の接続先（{st.session_state.ollama_host}）を維持します。")
            else:
                st.session_state.ollama_host = validated_host
                st.rerun()

    is_connected, msg, available_models, device = get_backend_status(
        vision_analyzer, (selected_kind, st.session_state.ollama_host, st.session_state.llm_model or vision_analyzer.model)
    )

    if is_connected:
        backend_title = backend_labels.get(vision_analyzer.backend_name, vision_analyzer.backend_name)
        if device == "gpu":
            st.sidebar.success(f"🟢 {backend_title} 接続中（GPU で実行）")
        elif device == "unknown":
            st.sidebar.info(f"🔵 {backend_title} 接続中（実行デバイスは初回解析時に確認します）")
        else:
            st.sidebar.warning(f"🟡 {backend_title} 接続中ですが **{device.upper()}** で実行されます。GPU実行が必須の設定では解析できません。")
        model_options = list(dict.fromkeys(([vision_analyzer.model] if vision_analyzer.model else []) + available_models))
        current = st.session_state.llm_model or vision_analyzer.model
        selected_model = st.sidebar.selectbox(
            "Vision AI モデル選択",
            options=model_options,
            index=model_options.index(current) if current in model_options else 0,
            help="画像解析に対応したモデルを選択してください（Foundry Local の既定: qwen3.5-4b）",
        )
        st.session_state.llm_model = selected_model
    else:
        st.sidebar.error(f"🔴 AI実行環境に接続できません\n\n{msg}\n\n`python tools/setup_llm.py` で診断できます。")

    if st.sidebar.button("🔄 AI実行環境を再確認", use_container_width=True):
        get_vision_analyzer.clear()
        get_backend_status.clear()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔬 研究用テスト板書")
    st.sidebar.caption("手元に画像がない場合は以下のテスト板書を即座に読み込めます。")

    col_s1, col_s2 = st.sidebar.columns(2)
    with col_s1:
        if st.button("① 順列の板書", use_container_width=True):
            p1 = config.TEST_DATA_DIR / "sample_permutation_board.png"
            if p1.exists():
                dest = _copy_sample_to_uploads(p1)
                st.session_state.is_mock = False
                st.session_state.analysis_error = None
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
                dest = _copy_sample_to_uploads(p2)
                st.session_state.is_mock = False
                st.session_state.analysis_error = None
                st.session_state.uploaded_image_paths = [dest]
                st.session_state.uploaded_image_names = [p2.name]
                st.session_state.current_step = 2
                st.rerun()
            else:
                st.sidebar.error("テスト画像が見つかりません。")

    st.sidebar.markdown("---")
    st.sidebar.caption("AI Electronic Blackboard System v1.0\nPrototype for Educational Research")


def main():
    st.set_page_config(
        page_title="AI電子黒板教材生成システム (ai_e_board)",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    BlackboardUI.load_custom_css()

    vision_analyzer = get_vision_analyzer(st.session_state.llm_backend_kind, st.session_state.ollama_host)
    render_sidebar(vision_analyzer)

    # If System B mode is selected, render System B
    if st.session_state.app_mode.startswith("📊"):
        pipeline = EvaluationPipeline(
            host=st.session_state.ollama_host,
            model=st.session_state.llm_model or vision_analyzer.model,
            llm_backend=vision_analyzer.backend,
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

    # デモ用サンプル授業を使用中は常に警告を表示（STEP3〜5）
    if st.session_state.current_step >= 3 and st.session_state.is_mock:
        st.warning("⚠️ **デモ用サンプル授業を表示中です。** これはアップロードされた板書画像のAI解析結果ではありません。研究ログには `is_mock: true` として記録されます。")

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
                # パストラバーサル対策: 元のファイル名は表示用のみ。保存名はUUIDで生成する
                upload_key = getattr(uploaded_file, "file_id", None) or f"{uploaded_file.name}:{uploaded_file.size}"
                cached = st.session_state.upload_cache.get(upload_key)
                if cached and Path(cached).exists():
                    dest_path = Path(cached)
                else:
                    try:
                        dest_path = safe_upload_path(
                            uploaded_file.name,
                            config.UPLOAD_DIR,
                            st.session_state.session_id,
                            config.ALLOWED_IMAGE_EXTENSIONS,
                        )
                    except ValueError as e:
                        st.error(f"「{uploaded_file.name}」を保存できません: {e}")
                        continue
                    with open(dest_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.session_state.upload_cache[upload_key] = str(dest_path)
                saved_paths.append(dest_path)
                saved_names.append(uploaded_file.name)

                with cols[idx % 3]:
                    st.image(uploaded_file, caption=f"画像 {idx+1}: {uploaded_file.name}", use_container_width=True)

            st.session_state.uploaded_image_paths = saved_paths
            st.session_state.uploaded_image_names = saved_names

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("次へ進む（STEP 2: AI解析へ） ▶", type="primary", use_container_width=True, disabled=not saved_paths):
                st.session_state.is_mock = False
                st.session_state.analysis_error = None
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
                display_names = st.session_state.uploaded_image_names
                disp = display_names[idx] if idx < len(display_names) else Path(img_p).name
                st.image(str(img_p), caption=f"板書 {idx+1}: {disp}", use_container_width=True)

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
            with st.status("Vision AI が板書を読み取っています…（画像の読み込み中）", expanded=False) as status_box:
                # 新しい解析を開始するので、デモ状態と前回エラーをリセット
                st.session_state.is_mock = False
                st.session_state.analysis_error = None
                t_start = time.time()
                last_update = [0.0]
                phase_label = ["① 板書を書き起こしています"]

                def _phase(phase: str, message: str):
                    phase_label[0] = "① 板書を書き起こしています" if phase == "transcribe" else "② 授業の構成を整理しています"
                    status_box.update(label=f"{phase_label[0]}…（{time.time() - t_start:.0f} 秒経過）")
                    st.write(f"{phase_label[0]}（開始 {time.time() - t_start:.0f} 秒）")

                def _progress(_delta: str, chars: int):
                    now = time.time()
                    if now - last_update[0] >= 1.0:  # 1秒ごとに進捗表示を更新
                        last_update[0] = now
                        status_box.update(label=f"{phase_label[0]}… {chars} 文字（{now - t_start:.0f} 秒経過）")

                # 書き起こし → 構成の整理（失敗時にモック授業へ自動差し替えはしない）
                lesson, analysis_err, vision_result = analyze_and_parse(
                    vision_analyzer=vision_analyzer,
                    image_paths=st.session_state.uploaded_image_paths,
                    image_names=st.session_state.uploaded_image_names,
                    custom_instructions=teacher_instruction,
                    model_override=st.session_state.llm_model or vision_analyzer.model,
                    on_progress=_progress,
                    on_phase=_phase,
                )
                st.session_state.raw_vision_result = vision_result
                status_box.update(
                    label=f"解析処理が終了しました（{time.time() - t_start:.0f} 秒 / "
                    f"{(vision_result or {}).get('device', '?').upper()}）",
                    state="complete" if lesson else "error",
                )

            if lesson:
                set_lesson(lesson)
                st.success("✅ 板書解析と授業構造化が完了しました！")
                time.sleep(0.5)
                st.session_state.current_step = 3
                st.rerun()
            else:
                logger.warning(f"Board analysis failed: {analysis_err}")
                st.session_state.analysis_error = {
                    "message": analysis_err,
                    "raw_text": vision_result.get("raw_text", "") if vision_result else "",
                }

        # 解析失敗時: エラー内容・生出力を表示し、STEP2に留まる（再試行可能）
        if st.session_state.analysis_error:
            err_info = st.session_state.analysis_error
            st.error(f"❌ 板書のAI解析に失敗しました。\n\n{err_info.get('message')}\n\nモデル・接続設定を確認して再試行してください。")
            with st.expander("🔎 LLMの生出力（デバッグ用）", expanded=False):
                if err_info.get("raw_text"):
                    st.code(err_info["raw_text"], language="text")
                else:
                    st.caption("（生出力はありません）")

            st.caption("※ 以下はOllamaが使えない環境での動作確認用です。アップロード画像の解析結果ではありません。")
            if st.button("デモ用サンプル授業で続行（解析結果ではありません）", use_container_width=True):
                first_img_name = st.session_state.uploaded_image_names[0] if st.session_state.uploaded_image_names else "board.png"
                mock_lesson = get_mock_lesson_for_sample(first_img_name)
                mock_lesson.source_images = st.session_state.uploaded_image_names
                set_lesson(mock_lesson)
                st.session_state.is_mock = True
                st.session_state.analysis_error = None
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

        # AIが読み取れなかった・推測できなかった項目（要確認）を先に示す
        review_items = list((st.session_state.get("raw_vision_result") or {}).get("parse_warnings") or [])
        review_items += [f"「{loc}」が要確認になっています。" for loc in find_unconfirmed_fields(lesson)]
        if review_items:
            st.warning("⚠️ **確認が必要な項目があります**（板書と照らし合わせて修正してください）\n\n"
                       + "\n".join(f"- {w}" for w in dict.fromkeys(review_items)))
        raw_info = st.session_state.get("raw_vision_result") or {}
        if raw_info.get("elapsed_seconds"):
            st.caption(f"AI解析: {raw_info.get('model')}（{str(raw_info.get('device', '?')).upper()}）"
                       f" {raw_info['elapsed_seconds']:.0f} 秒")

        # 入力欄のキーに授業データのハッシュを含める（別の授業を読み込んだとき前の入力値が残らないように）
        kp = f"{lesson_fingerprint(lesson)}_"

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
                    s_title = st.text_input(f"見出し", value=sec.title, key=kp + f"sec_title_{idx}")
                    s_content = st.text_area(f"本文・解説", value=sec.content, key=kp + f"sec_content_{idx}", height=100)
                with sec_c2:
                    s_type = st.selectbox(
                        "種類",
                        options=SECTION_TYPE_OPTIONS,
                        index=SECTION_TYPE_OPTIONS.index(sec.section_type) if sec.section_type in SECTION_TYPE_OPTIONS else 1,
                        key=kp + f"sec_type_{idx}",
                    )

                # Formulas inside section
                edited_formulas = []
                if sec.formulas:
                    st.caption("数式 (LaTeX)")
                    for f_idx, f in enumerate(sec.formulas):
                        f_latex = st.text_input(
                            f"LaTeX式 #{f_idx+1}",
                            value=f.latex,
                            key=kp + f"f_latex_{idx}_{f_idx}",
                        )
                        f_desc = st.text_input(
                            f"説明 #{f_idx+1}",
                            value=f.description or "",
                            key=kp + f"f_desc_{idx}_{f_idx}",
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
                        ex_prob = st.text_area("問題文", value=sec.example.problem, key=kp + f"ex_prob_{idx}")
                        ex_app = st.text_input("考え方・ヒント", value=sec.example.approach or "", key=kp + f"ex_app_{idx}")
                        ex_steps_str = st.text_area(
                            "解法ステップ（1行1ステップ）",
                            value="\n".join(sec.example.solution_steps),
                            key=kp + f"ex_steps_{idx}",
                        )
                        ex_ans = st.text_input("答え", value=sec.example.answer, key=kp + f"ex_ans_{idx}")
                        ex_notes = st.text_input("指導のポイント", value=sec.example.teaching_notes or "", key=kp + f"ex_notes_{idx}")
                        # 編集項目以外（例題内の数式など）は元の値を引き継ぐ
                        edited_example = sec.example.model_copy(update={
                            "problem": ex_prob,
                            "approach": ex_app or None,
                            "solution_steps": [s for s in ex_steps_str.split("\n") if s.strip()],
                            "answer": ex_ans,
                            "teaching_notes": ex_notes or None,
                        })

                # Exercise
                edited_exercise = sec.exercise
                if sec.exercise:
                    with st.expander(f"練習問題の詳細設定: {sec.exercise.title}", expanded=True):
                        exe_prob = st.text_area("問題文", value=sec.exercise.problem, key=kp + f"exe_prob_{idx}")
                        exe_hint = st.text_input("ヒント", value=sec.exercise.hint or "", key=kp + f"exe_hint_{idx}")
                        exe_ans = st.text_input("解答", value=sec.exercise.answer or "", key=kp + f"exe_ans_{idx}")
                        exe_steps_str = st.text_area(
                            "解答の手順（1行1ステップ）",
                            value="\n".join(sec.exercise.solution_steps),
                            key=kp + f"exe_steps_{idx}",
                        )
                        edited_exercise = sec.exercise.model_copy(update={
                            "problem": exe_prob,
                            "hint": exe_hint or None,
                            "answer": exe_ans or None,
                            "solution_steps": [s for s in exe_steps_str.split("\n") if s.strip()],
                        })

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
                set_lesson(updated_lesson)
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
                    <p><b>教科・単元:</b> {html.escape(lesson.subject)} | {html.escape(lesson.unit)}</p>
                    <p><b>授業テーマ:</b> {html.escape(lesson.lesson_title)}</p>
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
                    html_str = generator.render_presentation_html(presentation, lesson=lesson)

                    # Save to outputs/
                    json_p, html_p = generator.save_outputs(
                        lesson=lesson,
                        presentation=presentation,
                        html_content=html_str,
                        research_metadata={
                            "model": "mock" if st.session_state.is_mock else (st.session_state.llm_model or vision_analyzer.model),
                            "backend": vision_analyzer.backend_name,
                            "device": (st.session_state.get("raw_vision_result") or {}).get("device", "unknown"),
                            "elapsed_seconds": (st.session_state.get("raw_vision_result") or {}).get("elapsed_seconds"),
                            "source_images": st.session_state.uploaded_image_names,
                        },
                        is_mock=st.session_state.is_mock,
                    )

                    st.session_state.generated_presentation = presentation
                    st.session_state.presentation_html = html_str
                    st.session_state.saved_json_path = str(json_p)
                    st.session_state.saved_html_path = str(html_p)
                    st.session_state.presentation_lesson_hash = lesson_fingerprint(lesson)

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
        if st.session_state.presentation_lesson_hash != lesson_fingerprint(st.session_state.parsed_lesson):
            st.warning("授業データが変更されたため、表示中の教材は古い内容です。STEP 4 で再生成してください。")
            if st.button("STEP 4 へ戻って再生成する", type="primary"):
                st.session_state.current_step = 4
                st.rerun()
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
                set_lesson(None)
                reset_derived_state()
                st.session_state.raw_vision_result = None
                st.session_state.is_mock = False
                st.session_state.analysis_error = None
                st.rerun()



if __name__ == "__main__":
    main()
