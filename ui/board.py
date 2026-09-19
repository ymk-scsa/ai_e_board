"""
Electronic Blackboard UI rendering component for Streamlit.
"""

from pathlib import Path
from typing import Optional
import streamlit as st
import streamlit.components.v1 as components

import config
from models.schemas import ElectronicBoardPresentation


class BlackboardUI:
    """Renders interactive 16:9 electronic blackboard display directly inside Streamlit."""

    @staticmethod
    def load_custom_css():
        """Load external CSS stylesheet into Streamlit app."""
        css_file = config.UI_DIR / "styles.css"
        if css_file.exists():
            css_content = css_file.read_text(encoding="utf-8")
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

    @staticmethod
    def render_presentation_viewer(
        html_content: str,
        height: int = 700,
        scrolling: bool = False,
    ):
        """
        Embed the presentation HTML with KaTeX, keyboard controls, and 16:9 container inside Streamlit.
        """
        components.html(html_content, height=height, scrolling=scrolling)

    @staticmethod
    def render_step_progress_bar(current_step: int):
        """Render modern step progression header (STEP 1 ~ STEP 5)."""
        steps = [
            (1, "板書アップロード"),
            (2, "AI板書解析"),
            (3, "構造確認・編集"),
            (4, "教材生成"),
            (5, "電子黒板表示"),
        ]

        cols = st.columns(len(steps))
        for idx, (num, label) in enumerate(steps):
            with cols[idx]:
                if num < current_step:
                    st.markdown(
                        f"""
                        <div style="text-align: center; padding: 8px; border-radius: 8px; background: #ecfdf5; border: 1px solid #a7f3d0;">
                            <div style="font-weight: 800; color: #059669; font-size: 0.85rem;">✓ STEP {num}</div>
                            <div style="font-size: 0.95rem; font-weight: 700; color: #065f46;">{label}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                elif num == current_step:
                    st.markdown(
                        f"""
                        <div style="text-align: center; padding: 8px; border-radius: 8px; background: #eff6ff; border: 2px solid #3b82f6; box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.2);">
                            <div style="font-weight: 800; color: #2563eb; font-size: 0.85rem;">▶ STEP {num} (進行中)</div>
                            <div style="font-size: 0.95rem; font-weight: 800; color: #1e40af;">{label}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div style="text-align: center; padding: 8px; border-radius: 8px; background: #f8fafc; border: 1px solid #e2e8f0;">
                            <div style="font-weight: 600; color: #94a3b8; font-size: 0.85rem;">STEP {num}</div>
                            <div style="font-size: 0.95rem; font-weight: 600; color: #64748b;">{label}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
