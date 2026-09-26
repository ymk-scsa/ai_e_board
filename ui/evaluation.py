"""
Streamlit UI Component for System B (Educational Quality Evaluation and Classroom Operation).
"""

import html
import json
from pathlib import Path
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components

import config
from pydantic import ValidationError

from models.samples import get_mock_lesson_for_sample
from models.schemas import Lesson, ElectronicBoardPresentation, lesson_fingerprint
from models.evaluation_schemas import EvaluationResult, ImprovementSuggestion
from system_b.pipeline import EvaluationPipeline


class EvaluationUI:
    """Renders the comprehensive educational evaluation and classroom operation dashboard."""

    @classmethod
    def render_evaluation_view(cls, pipeline: EvaluationPipeline):
        """Render the complete System B UI."""
        st.subheader("📊 電子黒板教材 教育品質評価・授業運用支援システム (システムB)")
        st.caption("電子黒板教材（Lesson JSON または HTML）を教育工学・認知負荷・16:9視認性の観点から多角的に分析し、具体的な改善提案と授業運用ガイドを提供します。")

        # Session state for System B
        if "eval_result" not in st.session_state:
            st.session_state.eval_result = None
        if "eval_json_path" not in st.session_state:
            st.session_state.eval_json_path = None
        if "eval_html_path" not in st.session_state:
            st.session_state.eval_html_path = None

        # Input Mode Selector
        input_mode = st.radio(
            "評価対象教材の入力方法を選択",
            options=["① システムAで生成済みの教材を評価", "② Lesson JSON ファイルをアップロード", "③ 電子黒板 HTML ファイルをアップロード", "④ 研究用プリセット教材"],
            horizontal=True,
        )

        target_lesson: Optional[Lesson] = None
        target_html: Optional[str] = None
        target_json_str: Optional[str] = None

        if input_mode == "① システムAで生成済みの教材を評価":
            if st.session_state.get("parsed_lesson"):
                target_lesson = st.session_state.parsed_lesson
                st.success(f"✅ システムAで生成・確定された教材「{target_lesson.lesson_title}」が読み込まれています。")
                if st.session_state.get("is_mock"):
                    st.warning("⚠️ この教材はデモ用サンプル授業です（板書画像のAI解析結果ではありません）。")
            else:
                st.info("ℹ️ システムAで教材をまだ生成していません。画面上部のモード切替で「🎨 電子黒板教材生成」を行うか、他のアップロード方法を選択してください。")

        elif input_mode == "② Lesson JSON ファイルをアップロード":
            uploaded_json = st.file_uploader("Lesson JSON ファイルを選択", type=["json"], key="b_json_uploader")
            if uploaded_json:
                target_json_str = cls._decode_upload(uploaded_json)
                if target_json_str is not None:
                    st.success(f"✅ JSONファイル「{uploaded_json.name}」がアップロードされました。")

        elif input_mode == "③ 電子黒板 HTML ファイルをアップロード":
            uploaded_html = st.file_uploader("電子黒板 HTML ファイルを選択", type=["html", "htm"], key="b_html_uploader")
            if uploaded_html:
                target_html = cls._decode_upload(uploaded_html)
                if target_html is not None:
                    st.success(f"✅ HTMLファイル「{uploaded_html.name}」がアップロードされました。")

        elif input_mode == "④ 研究用プリセット教材":
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                if st.button("📚 プリセット: 高校数学A「順列」", use_container_width=True):
                    target_lesson = get_mock_lesson_for_sample("permutation")
                    st.session_state.preset_lesson = target_lesson
            with col_p2:
                if st.button("📚 プリセット: 高校数学I「平方完成」", use_container_width=True):
                    target_lesson = get_mock_lesson_for_sample("quadratic")
                    st.session_state.preset_lesson = target_lesson

            if "preset_lesson" in st.session_state and st.session_state.preset_lesson:
                target_lesson = st.session_state.preset_lesson
                st.success(f"✅ プリセット教材「{target_lesson.lesson_title}」を選択中")

        st.markdown("---")

        # Evaluation Configuration
        eval_c1, eval_c2 = st.columns([3, 2])
        with eval_c1:
            eval_focus = st.text_input(
                "教員からの評価重点指示（任意）",
                placeholder="例: 高校1年生向けの授業です。例題の難易度とスライドの情報量を重点的に見てください。",
            )
        with eval_c2:
            use_llm_toggle = st.checkbox(
                "ローカルLLMによる講評を追加（GPUで約1分）", value=False,
                help="オフのときは採点ルールだけで即座に評価します。オンにすると総評・良い点・注意点の文章をAIが書き直します（点数は変わりません）。",
            )

        start_eval_btn = st.button("🚀 教育品質評価を実行する", type="primary", use_container_width=True)

        if start_eval_btn:
            if not target_lesson and not target_html and not target_json_str:
                st.error("評価対象の教材が指定されていません。JSONまたはHTMLをアップロードしてください。")
                return

            with st.spinner("教材の構成、教育目標整合性、16:9視認性、認知負荷を多角的に分析中..."):
                try:
                    result, json_p, html_p = cls._run_evaluation(
                        pipeline, target_lesson, target_json_str, target_html, eval_focus, use_llm_toggle)
                except (ValueError, ValidationError) as e:
                    st.error(f"教材ファイルを読み取れませんでした。システムAで出力した JSON / HTML か確認してください。\n\n詳細: {e}")
                    return
                st.session_state.eval_result = result
                st.session_state.eval_json_path = str(json_p)
                st.session_state.eval_html_path = str(html_p)
                st.success("✅ 評価が完了し、レポートを生成しました！")
                if use_llm_toggle and result.evaluated_model == "deterministic_rules":
                    st.warning("AIによる講評を取得できなかったため、採点ルールによる評価のみを表示しています。")

        # Render Results Dashboard
        res: Optional[EvaluationResult] = st.session_state.eval_result
        if res:
            cls._render_results(res)

    @staticmethod
    def _decode_upload(uploaded) -> Optional[str]:
        """Decode an uploaded text file (UTF-8 with or without BOM)."""
        try:
            return uploaded.getvalue().decode("utf-8-sig")
        except UnicodeDecodeError:
            st.error(f"「{uploaded.name}」は UTF-8 のテキストではないため読み込めません。")
            return None

    @staticmethod
    def _run_evaluation(pipeline, target_lesson, target_json_str, target_html, eval_focus, use_llm_toggle):
        if target_lesson:
            # 生成済みスライドは、同じ授業データから作られたものだけを使う（古いスライドの評価を防ぐ）
            presentation = st.session_state.get("generated_presentation")
            if st.session_state.get("presentation_lesson_hash") != lesson_fingerprint(target_lesson):
                presentation = None
            return pipeline.evaluate_from_lesson(
                lesson=target_lesson, presentation=presentation, custom_focus=eval_focus, use_llm=use_llm_toggle)
        if target_json_str:
            return pipeline.evaluate_from_json_string(
                json_text=target_json_str, custom_focus=eval_focus, use_llm=use_llm_toggle)
        return pipeline.evaluate_from_html_content(
            html_content=target_html, custom_focus=eval_focus, use_llm=use_llm_toggle)

    @classmethod
    def _render_results(cls, res: EvaluationResult):
        st.markdown("---")
        cls._render_evaluation_dashboard(res)

    @classmethod
    def _render_evaluation_dashboard(cls, res: EvaluationResult):
        """Render the rich evaluation dashboard and scores."""
        st.markdown(f"## 📋 評価結果: {res.lesson_title} ({res.unit})")

        # Score Row
        score_cols = st.columns([2, 1, 1, 1, 1])
        with score_cols[0]:
            st.markdown(
                f"""
                <div style="text-align: center; padding: 20px; background: #eff6ff; border: 2px solid #3b82f6; border-radius: 12px;">
                    <div style="font-size: 0.95rem; font-weight: 700; color: #1e40af;">総合教育品質スコア</div>
                    <div style="font-size: 3.5rem; font-weight: 900; color: #1e3a8a; line-height: 1.1;">{html.escape(str(res.overall_score))}</div>
                    <div style="font-size: 0.85rem; color: #64748b;">/ 100点</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with score_cols[1]:
            st.metric("🧩 授業構成", f"{res.category_scores.pedagogical_structure}点")
        with score_cols[2]:
            st.metric("🎯 目標整合性", f"{res.category_scores.objective_alignment}点")
        with score_cols[3]:
            st.metric("🖥️ 電子黒板UX", f"{res.category_scores.blackboard_usability}点")
        with score_cols[4]:
            st.metric("🧠 認知負荷", f"{res.category_scores.cognitive_load_balance}点")

        # Executive Summary & Strengths/Attentions
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📝 総合評価サマリーと特徴", expanded=True):
            st.write(res.executive_summary)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🌟 主な優れた点")
                for s in res.strengths:
                    st.markdown(f"- ✅ {s}")
            with c2:
                st.markdown("#### ⚠️ 授業での留意点")
                for p in res.points_for_attention:
                    st.markdown(f"- 📌 {p}")

        # Quantitative Metrics Table
        with st.expander("📐 スライド別定量的メトリクスと着目フラグ", expanded=True):
            st.caption("※文字数や数式数は客観的な参考情報（着目フラグ）であり、単独で機械的な減点を行うものではありません。")
            metrics_data = []
            for m in res.slide_metrics:
                flags_str = "、".join(m.attention_flags) if m.attention_flags else "適正"
                metrics_data.append({
                    "スライド": f"#{m.slide_number}",
                    "見出し": m.title,
                    "種別": m.slide_type,
                    "文字数": f"{m.char_count}字",
                    "数式数": f"{m.formula_count}式",
                    "情報密度": m.density_level,
                    "着目ポイント・参考情報": flags_str,
                })
            st.table(metrics_data)

        # Improvement Suggestions (Human-in-the-loop Editing)
        with st.expander("💡 教員向け具体的改善提案（Human-in-the-Loop 確認・修正）", expanded=True):
            for idx, imp in enumerate(res.improvements):
                target_str = f"スライド {imp.target_slide}" if imp.target_slide > 0 else "全体構成"
                badge_color = {"High": "red", "Medium": "orange", "Low": "green"}.get(imp.priority, "blue")
                
                st.markdown(f"#### 提案 {idx+1}: {target_str} [{imp.category}] （優先度: :{badge_color}[{imp.priority}]）")
                st.write(f"**問題点:** {imp.issue}")
                st.write(f"**教育的理由:** {imp.rationale}")
                st.info(f"👉 **推奨アクション:** {imp.concrete_action}")
                st.markdown("---")

        # Classroom Teaching Guide
        with st.expander("👨‍🏫 授業運用ガイド（発問・指示・生徒活動のヒント）", expanded=True):
            for g in res.teaching_guides:
                st.markdown(f"**スライド #{g.slide_number}**（{g.timing}）")
                st.write(f"- 🎯 **強調ポイント:** {g.key_point_to_emphasize}")
                st.write(f"- 🗣️ **発問例:** `{g.questioning_prompt}`")
                st.write(f"- ✍️ **生徒の活動指示:** {g.student_activity_guide}")
                st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)

        # Download and Preview Section
        st.markdown("---")
        st.subheader("📥 評価レポートのエクスポート")
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            if st.session_state.eval_html_path and Path(st.session_state.eval_html_path).exists():
                with open(st.session_state.eval_html_path, "r", encoding="utf-8") as hf:
                    html_data = hf.read()
                st.download_button(
                    "📥 評価レポート（HTML）をダウンロード",
                    data=html_data,
                    file_name=Path(st.session_state.eval_html_path).name,
                    mime="text/html",
                    use_container_width=True,
                )
        with dl_col2:
            if st.session_state.eval_json_path and Path(st.session_state.eval_json_path).exists():
                with open(st.session_state.eval_json_path, "r", encoding="utf-8") as jf:
                    json_data = jf.read()
                st.download_button(
                    "📥 評価データ（JSON）をダウンロード",
                    data=json_data,
                    file_name=Path(st.session_state.eval_json_path).name,
                    mime="application/json",
                    use_container_width=True,
                )

        with st.expander("📄 レポートHTMLプレビュー", expanded=False):
            if st.session_state.eval_html_path and Path(st.session_state.eval_html_path).exists():
                with open(st.session_state.eval_html_path, "r", encoding="utf-8") as hf:
                    components.html(hf.read(), height=650, scrolling=True)
