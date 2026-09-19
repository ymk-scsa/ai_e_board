"""
Improvement and Teaching Guide Generator Module for System B.
Generates concrete, actionable suggestions for teachers and classroom instructional guides.
"""

import logging
from typing import List, Dict, Any, Optional
from models.evaluation_schemas import (
    ImprovementSuggestion,
    TeachingGuideItem,
    SlideMetrics,
    PedagogicalEvaluation,
    UsabilityEvaluation,
)
from system_b.analyzer import ParsedLessonMaterial

logger = logging.getLogger("ai_e_board.system_b.improvement")


class ImprovementGenerator:
    """Generates concrete instructional improvements and practical classroom operation guides."""

    @classmethod
    def generate_suggestions(
        cls,
        material: ParsedLessonMaterial,
        metrics_list: List[SlideMetrics],
        pedagogy_eval: PedagogicalEvaluation,
        usability_eval: UsabilityEvaluation,
    ) -> List[ImprovementSuggestion]:
        """Generate a list of prioritized, constructive improvement suggestions."""
        suggestions: List[ImprovementSuggestion] = []
        sug_idx = 1

        # Check for split-recommended slides
        for slide_num in usability_eval.slide_split_recommended_slides:
            suggestions.append(
                ImprovementSuggestion(
                    id=f"imp_{sug_idx}",
                    target_slide=slide_num,
                    category="CognitiveLoad",
                    issue=f"スライド{slide_num}には問題文、考え方、複数の解法ステップが集約されており、1画面の情報密度が高くなっています。",
                    rationale="生徒が解法ステップを先読みしてしまい、自力で考える時間が短縮されるリスクや、視線が散漫になるのを防ぐため。",
                    concrete_action=f"スライド{slide_num}を『問題提示・考え方画面』と『ステップ別解法・答え画面』の2枚に分割するか、教員が途中で画面を止めて発問を挟む運用を推奨します。",
                    priority="High",
                )
            )
            sug_idx += 1

        # Check pedagogy completeness suggestions
        if not pedagogy_eval.has_exercise:
            suggestions.append(
                ImprovementSuggestion(
                    id=f"imp_{sug_idx}",
                    target_slide=0,
                    category="Pedagogy",
                    issue="生徒が自力で公式や解法を適用する練習問題（演習）が含まれていません。",
                    rationale="数学の概念定着には、例題の理解直後に自力で類題を解くアウトプット活動が不可欠であるため。",
                    concrete_action="例題の後に数値を変えた練習問題を1〜2問追加し、生徒の個人ワーク時間を設けることを推奨します。",
                    priority="Medium",
                )
            )
            sug_idx += 1

        if not pedagogy_eval.has_summary:
            suggestions.append(
                ImprovementSuggestion(
                    id=f"imp_{sug_idx}",
                    target_slide=0,
                    category="Pedagogy",
                    issue="授業の最後に本時の学びを振り返るまとめスライドがありません。",
                    rationale="授業終盤に本時の最重要公式や考え方を再確認することで、長期記憶への定着と次回の見通しが強化されるため。",
                    concrete_action="本時の最重要公式や重要キーワードを整理したまとめスライドを末尾に追加することを推奨します。",
                    priority="Low",
                )
            )
            sug_idx += 1

        # Check attention flags on individual slides
        for m in metrics_list:
            if m.char_count > 200 and m.slide_number not in usability_eval.slide_split_recommended_slides:
                suggestions.append(
                    ImprovementSuggestion(
                        id=f"imp_{sug_idx}",
                        target_slide=m.slide_number,
                        category="Usability",
                        issue=f"スライド{m.slide_number}（{m.title}）のテキスト量が比較的多めです（約{m.char_count}文字）。",
                        rationale="教室後方からの速読性を高め、要点に素早く視線を誘導するため。",
                        concrete_action="長文の解説を短い箇条書きに整理するか、教員の口頭説明に委ねて画面上の文字数を絞り込むことを推奨します。",
                        priority="Low",
                    )
                )
                sug_idx += 1

        # Fallback default suggestion if presentation is already very solid
        if not suggestions:
            suggestions.append(
                ImprovementSuggestion(
                    id="imp_1",
                    target_slide=1,
                    category="Pedagogy",
                    issue="スライド全体のバランスは良好です。",
                    rationale="さらなる授業活性化のため。",
                    concrete_action="導入スライドで生徒に問いかける際、挙手やペアワークを促す指示を追加するとより効果的です。",
                    priority="Low",
                )
            )

        return suggestions

    @classmethod
    def generate_teaching_guides(cls, material: ParsedLessonMaterial) -> List[TeachingGuideItem]:
        """Generate practical classroom delivery tips for teachers."""
        guides: List[TeachingGuideItem] = []

        for s in material.slides:
            timing = "授業展開中"
            key_point = f"{s.title}のポイントを強調"
            question = "『ここまでの内容で疑問点はありますか？』"
            student_act = "ノートへの記録を促す"

            if "目標" in s.badge or s.slide_number == 1:
                timing = "授業開始直後（導入）"
                key_point = "本時の学習目標と到達点を生徒全員で共有する"
                question = "『今日できるようになりたい目標は何でしょう？』"
                student_act = "ノートの先頭に本時のテーマと日付を書かせる"
            elif "公式" in s.badge or "定義" in s.badge:
                timing = "概念解説・公式提示時"
                key_point = "公式の文字や記号が何を意味しているかを具体例と結びつける"
                question = "『なぜこの公式の形で計算できるか説明できますか？』"
                student_act = "重要公式を赤枠や黄チョーク風に色分けしてノートに書かせる"
            elif "例題" in s.badge or s.has_example:
                timing = "例題解説・思考プロセス提示時"
                key_point = "いきなり答えを見せず、第1ステップ（方針）を生徒に考えさせる"
                question = "『この問題を見たとき、まず何から着手すべきでしょうか？』"
                student_act = "解法の一行目を各自でノートに書かせてから画面を進める"
            elif "練習" in s.badge or s.has_exercise:
                timing = "生徒活動・自力演習時"
                key_point = "机間巡視を行い、つまずいている生徒にヒントを与える"
                question = "『例題と同じ考え方で解ける部分はどこですか？』"
                student_act = "3分間の個人ワーク後、隣の生徒と答え合わせを行わせる"
            elif "まとめ" in s.badge:
                timing = "授業終了前（振り返り）"
                key_point = "本時のキーワードと公式を再度全員で確認する"
                question = "『今日の授業で最も重要だったポイントを一言で言うと何ですか？』"
                student_act = "本日の自己評価・振り返りを1行書かせる"

            guides.append(
                TeachingGuideItem(
                    slide_number=s.slide_number,
                    timing=timing,
                    key_point_to_emphasize=key_point,
                    questioning_prompt=question,
                    student_activity_guide=student_act,
                )
            )

        return guides
