"""
Report Generator Module for System B.
Generates structured JSON output and printable, beautifully styled HTML evaluation reports.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict, Any
from jinja2 import Environment

import config
from models.evaluation_schemas import EvaluationResult

logger = logging.getLogger("ai_e_board.system_b.report")


HTML_REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>教育品質評価レポート - {{ result.lesson_title }}</title>
    
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=BIZ+UDPGothic:wght@400;700&family=Noto+Sans+JP:wght@400;600;800&display=swap" rel="stylesheet">

    <style>
        :root {
            --primary: #1e3a8a;
            --primary-light: #eff6ff;
            --secondary: #059669;
            --secondary-light: #ecfdf5;
            --warning: #d97706;
            --warning-light: #fffbeb;
            --danger: #dc2626;
            --danger-light: #fef2f2;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text-main: #0f172a;
            --text-sub: #475569;
            --border: #e2e8f0;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'BIZ UDPGothic', 'Noto Sans JP', sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            line-height: 1.6;
            padding: 30px 20px;
        }

        .report-container {
            max-width: 1000px;
            margin: 0 auto;
            background: var(--card-bg);
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
            border: 1px solid var(--border);
            overflow: hidden;
        }

        .report-header {
            background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%);
            color: #ffffff;
            padding: 36px 40px;
        }

        .report-header h1 {
            font-size: 2.0rem;
            font-weight: 800;
            margin-bottom: 8px;
        }

        .report-meta {
            display: flex;
            gap: 20px;
            font-size: 0.95rem;
            color: #cbd5e1;
        }

        .report-body {
            padding: 36px 40px;
        }

        /* Score Summary Banner */
        .score-banner {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: #f1f5f9;
            border-radius: 10px;
            padding: 24px 32px;
            margin-bottom: 30px;
            border: 1px solid var(--border);
        }

        .overall-score-box {
            text-align: center;
        }

        .overall-score-num {
            font-size: 3.8rem;
            font-weight: 800;
            color: var(--primary);
            line-height: 1;
        }

        .overall-score-label {
            font-size: 0.9rem;
            font-weight: 700;
            color: var(--text-sub);
            margin-top: 6px;
        }

        .category-score-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px 24px;
            flex: 1;
            margin-left: 40px;
        }

        .cat-score-item {
            background: #ffffff;
            border-radius: 6px;
            padding: 10px 16px;
            border: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .cat-score-name {
            font-size: 0.9rem;
            font-weight: 700;
            color: var(--text-main);
        }

        .cat-score-val {
            font-size: 1.25rem;
            font-weight: 800;
            color: var(--primary);
        }

        /* Section Cards */
        .section-card {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 24px;
        }

        .section-title {
            font-size: 1.3rem;
            font-weight: 800;
            color: var(--primary);
            margin-bottom: 16px;
            padding-bottom: 8px;
            border-bottom: 2px solid #e2e8f0;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        /* Lists and Bullets */
        .highlight-list {
            list-style: none;
            margin: 10px 0;
        }

        .highlight-list li {
            position: relative;
            padding-left: 24px;
            margin-bottom: 10px;
            font-size: 0.95rem;
        }

        .highlight-list.strengths li::before {
            content: "✓";
            position: absolute;
            left: 0;
            color: var(--secondary);
            font-weight: 800;
        }

        .highlight-list.attentions li::before {
            content: "!";
            position: absolute;
            left: 0;
            color: var(--warning);
            font-weight: 800;
        }

        /* Metrics Table */
        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
            margin-top: 12px;
        }

        .metrics-table th, .metrics-table td {
            padding: 10px 14px;
            border: 1px solid var(--border);
            text-align: left;
        }

        .metrics-table th {
            background: #f8fafc;
            font-weight: 700;
            color: var(--text-sub);
        }

        .flag-tag {
            display: inline-block;
            background: var(--warning-light);
            color: var(--warning);
            border: 1px solid #fde68a;
            border-radius: 4px;
            padding: 2px 8px;
            font-size: 0.8rem;
            margin: 2px 0;
        }

        /* Improvement Cards */
        .imp-card {
            background: #fafafa;
            border-left: 5px solid var(--primary);
            border-radius: 6px;
            padding: 16px 20px;
            margin-bottom: 16px;
            border-top: 1px solid var(--border);
            border-right: 1px solid var(--border);
            border-bottom: 1px solid var(--border);
        }

        .imp-card.priority-High {
            border-left-color: var(--danger);
        }

        .imp-card.priority-Medium {
            border-left-color: var(--warning);
        }

        .imp-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }

        .badge-priority {
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 800;
        }

        .badge-priority.High {
            background: var(--danger-light);
            color: var(--danger);
        }

        .badge-priority.Medium {
            background: var(--warning-light);
            color: var(--warning);
        }

        .badge-priority.Low {
            background: var(--secondary-light);
            color: var(--secondary);
        }

        /* Teaching Guide Cards */
        .guide-box {
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 14px;
        }

        .guide-timing {
            font-size: 0.85rem;
            font-weight: 800;
            color: var(--secondary);
            margin-bottom: 6px;
        }

        .guide-prompt {
            font-size: 1.0rem;
            font-weight: 700;
            color: #166534;
            margin: 6px 0;
        }

        .report-footer {
            background: #f8fafc;
            border-top: 1px solid var(--border);
            padding: 20px 40px;
            text-align: center;
            font-size: 0.85rem;
            color: var(--text-sub);
        }
    </style>
</head>
<body>

<div class="report-container">
    <div class="report-header">
        <h1>📊 AI電子黒板 教育品質評価レポート</h1>
        <div class="report-meta">
            <span><b>教材:</b> {{ result.lesson_title }}</span>
            <span><b>単元:</b> {{ result.unit }}</span>
            <span><b>教科:</b> {{ result.subject }}</span>
            <span><b>評価日時:</b> {{ result.evaluated_at[:16] }}</span>
        </div>
    </div>

    <div class="report-body">
        <!-- Score Banner -->
        <div class="score-banner">
            <div class="overall-score-box">
                <div class="overall-score-num">{{ result.overall_score }}</div>
                <div class="overall-score-label">総合教育品質スコア / 100</div>
            </div>
            <div class="category-score-grid">
                <div class="cat-score-item">
                    <span class="cat-score-name">🧩 授業構成・指導展開</span>
                    <span class="cat-score-val">{{ result.category_scores.pedagogical_structure }}</span>
                </div>
                <div class="cat-score-item">
                    <span class="cat-score-name">🎯 教育目標整合性</span>
                    <span class="cat-score-val">{{ result.category_scores.objective_alignment }}</span>
                </div>
                <div class="cat-score-item">
                    <span class="cat-score-name">🖥️ 電子黒板UX・視認性</span>
                    <span class="cat-score-val">{{ result.category_scores.blackboard_usability }}</span>
                </div>
                <div class="cat-score-item">
                    <span class="cat-score-name">🧠 認知負荷・情報密度</span>
                    <span class="cat-score-val">{{ result.category_scores.cognitive_load_balance }}</span>
                </div>
            </div>
        </div>

        <!-- Executive Summary -->
        <div class="section-card">
            <div class="section-title">📝 総合評価サマリー</div>
            <p style="font-size: 1.05rem; line-height: 1.8;">{{ result.executive_summary }}</p>
            <div style="margin-top: 16px; display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div>
                    <h4 style="color: var(--secondary); font-size: 0.95rem; margin-bottom: 6px;">🌟 主な優れた点</h4>
                    <ul class="highlight-list strengths">
                        {% for s in result.strengths %}
                        <li>{{ s }}</li>
                        {% endfor %}
                    </ul>
                </div>
                <div>
                    <h4 style="color: var(--warning); font-size: 0.95rem; margin-bottom: 6px;">⚠️ 授業での留意点</h4>
                    <ul class="highlight-list attentions">
                        {% for p in result.points_for_attention %}
                        <li>{{ p }}</li>
                        {% endfor %}
                    </ul>
                </div>
            </div>
        </div>

        <!-- Quantitative Slide Metrics -->
        <div class="section-card">
            <div class="section-title">📐 スライド別定量的メトリクスと着目フラグ</div>
            <table class="metrics-table">
                <thead>
                    <tr>
                        <th>スライド</th>
                        <th>見出し</th>
                        <th>種別</th>
                        <th>文字数</th>
                        <th>数式数</th>
                        <th>情報密度</th>
                        <th>着目ポイント・参考情報</th>
                    </tr>
                </thead>
                <tbody>
                    {% for m in result.slide_metrics %}
                    <tr>
                        <td><b>#{{ m.slide_number }}</b></td>
                        <td>{{ m.title }}</td>
                        <td>{{ m.slide_type }}</td>
                        <td>{{ m.char_count }} 字</td>
                        <td>{{ m.formula_count }} 式</td>
                        <td><b>{{ m.density_level }}</b></td>
                        <td>
                            {% if m.attention_flags %}
                                {% for f in m.attention_flags %}
                                <div class="flag-tag">📌 {{ f }}</div>
                                {% endfor %}
                            {% else %}
                                <span style="color: #94a3b8;">適正</span>
                            {% endif %}
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <!-- Improvement Suggestions -->
        <div class="section-card">
            <div class="section-title">💡 教員向け具体的改善提案</div>
            {% for imp in result.improvements %}
            <div class="imp-card priority-{{ imp.priority }}">
                <div class="imp-header">
                    <div>
                        <b>{% if imp.target_slide > 0 %}スライド {{ imp.target_slide }}{% else %}全体構成{% endif %}</b>
                        <span style="color: var(--text-sub); font-size: 0.85rem; margin-left: 8px;">[{{ imp.category }}]</span>
                    </div>
                    <span class="badge-priority {{ imp.priority }}">{{ imp.priority }} 優先度</span>
                </div>
                <div style="margin: 6px 0;"><b>問題点:</b> {{ imp.issue }}</div>
                <div style="font-size: 0.9rem; color: var(--text-sub); margin-bottom: 6px;"><b>理由:</b> {{ imp.rationale }}</div>
                <div style="background: #ffffff; border: 1px dashed var(--border); border-radius: 6px; padding: 10px; font-weight: 700; color: var(--primary);">
                    👉 <b>推奨アクション:</b> {{ imp.concrete_action }}
                </div>
            </div>
            {% endfor %}
        </div>

        <!-- Teaching Operation Guide -->
        <div class="section-card">
            <div class="section-title">👨‍🏫 授業運用ガイド（発問と指導のヒント）</div>
            {% for g in result.teaching_guides %}
            <div class="guide-box">
                <div class="guide-timing">⏱️ スライド {{ g.slide_number }} / {{ g.timing }}</div>
                <div><b>強調ポイント:</b> {{ g.key_point_to_emphasize }}</div>
                <div class="guide-prompt">🗣️ <b>発問例:</b> {{ g.questioning_prompt }}</div>
                <div style="font-size: 0.9rem; color: var(--text-sub); margin-top: 4px;">✍️ <b>生徒の活動:</b> {{ g.student_activity_guide }}</div>
            </div>
            {% endfor %}
        </div>
    </div>

    <div class="report-footer">
        AI Electronic Blackboard Quality Evaluation System (System B) • Generated for Educational Research
    </div>
</div>

</body>
</html>
"""


class ReportGenerator:
    """Generates standalone HTML evaluation reports and structured JSON files."""

    def __init__(self):
        # autoescape=True: LLM/教材由来のテキストはすべてHTMLエスケープされる（テンプレート内に事前生成HTMLは無い）
        self.template = Environment(autoescape=True).from_string(HTML_REPORT_TEMPLATE)

    def render_html_report(self, result: EvaluationResult) -> str:
        """Render complete HTML report from EvaluationResult."""
        return self.template.render(result=result)

    def save_evaluation_outputs(
        self,
        result: EvaluationResult,
        research_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Path, Path]:
        """
        Save JSON and HTML evaluation reports to outputs/evaluations/ directory.
        Returns (json_path, html_path).
        """
        config.EVALUATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        base_name = result.evaluation_id

        json_path = config.EVALUATION_OUTPUT_DIR / f"{base_name}.json"
        html_path = config.EVALUATION_OUTPUT_DIR / f"{base_name}.html"

        # Save JSON
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(result.model_dump(), jf, ensure_ascii=False, indent=2)

        # Save HTML
        html_content = self.render_html_report(result)
        with open(html_path, "w", encoding="utf-8") as hf:
            hf.write(html_content)

        # Append to evaluation research log
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "evaluation_id": base_name,
            "lesson_title": result.lesson_title,
            "unit": result.unit,
            "overall_score": result.overall_score,
            "input_source_type": result.input_source_type,
            "evaluated_model": result.evaluated_model,
            **(research_metadata or {}),
        }
        try:
            with open(config.EVALUATION_LOG_PATH, "a", encoding="utf-8") as lf:
                lf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Could not write to evaluation log: {e}")

        logger.info(f"Saved evaluation outputs: JSON -> {json_path}, HTML -> {html_path}")
        return json_path, html_path
