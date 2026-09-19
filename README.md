# AI電子黒板教材生成 & 教育品質評価システム (ai_e_board)

教員が作成した手書き・図表・数式を含む**板書計画写真**をAI（Vision LLM）が解析し、教育的構造（単元・目標・導入・公式・例題・練習・まとめ）を保持した**16:9電子黒板教材**を自動生成する**「システムA」**と、生成教材や既存教材の教育効果・16:9視認性・認知負荷を多角的に分析し改善提案と授業運用ガイドを提供する**「システムB」**からなる教育工学研究プラットフォームです。

---

## 1. 研究背景と目的

- **研究テーマ**: 「教員が作成した板書計画を基にAIが電子黒板教材を自動生成し、教育品質を評価・改善する支援システムの開発」
- **今後の発展テーマ**: 「AIシステムを活用した電子黒板の効果的運用と教育の質向上に関する研究」
- **目的**: 
  - 教員の教材準備負担を大幅に軽減しながら、板書計画に込められた指導意図（導入の動機づけ、例題の解法ステップ、重要公式の強調等）を損なうことなく、教室後方からも視認性の高い電子黒板教材へ即座に変換・提示する（**システムA**）。
  - 教材の授業構成、教育目標整合性、16:9画面としての情報密度・認知負荷を客観的・定性的に分析し、具体的な改善提案と教員向け発問・授業運用ガイドを提供する（**システムB**）。
  - AIが勝手に教員を置き換えるのではなく、教員が解析結果や改善案を確認・修正できる**Human-in-the-Loop（協調型）**システムとして設計。

---

## 2. システムアーキテクチャ

システムAとシステムBは**完全に独立して動作可能**でありながら、**シームレスに連携**できます。

```mermaid
flowchart TD
    subgraph システムA [システムA: 電子黒板教材生成]
        A1[板書計画画像\nuploads/] --> A2[ai/vision.py\nOllama Vision AI]
        A2 --> A3[ai/parser.py\nPydantic JSON Parser]
        A3 --> A4[Lesson JSON\n教員確認・編集]
        A4 --> A5[ai/generator.py\n16:9 Slide Generator]
        A5 --> A6[電子黒板HTML\noutputs/lesson_*.html]
    end

    subgraph システムB [システムB: 教育品質評価 & 授業運用支援]
        B1[入力: Lesson JSON または HTML\nまたは システムAからの直接連携] --> B2[system_b/analyzer.py\n教材解析 & 定量的メトリクス抽出]
        B2 --> B3[system_b/pedagogy.py\n授業構成 & 指導目標整合性評価]
        B2 --> B4[system_b/usability.py\n電子黒板UX & 認知負荷評価]
        B3 --> B5[system_b/quality_evaluator.py\n統合品質スコアリング]
        B4 --> B5
        B5 --> B6[system_b/improvement.py\n具体的改善提案 & 授業運用ガイド]
        B6 --> B7[system_b/report.py\n評価レポート出力 (JSON & HTML)]
    end

    A4 -.->|直接連携| B1
    A6 -.->|直接連携| B1
```

### 3つの運用パターン
1. **[システムA 単独]**: 板書計画写真 → AI解析 → 編集 → 16:9 電子黒板教材生成・提示
2. **[システムB 単独]**: 任意の Lesson JSON または 電子黒板 HTML を直接アップロード → 教育品質評価・改善提案
3. **[システムA + B 連携]**: システムAで生成した教材からワンクリック（「この教材を評価する」）でシステムBへ引き渡し、即座に評価レポートと授業運用ガイドを取得

---

## 3. ディレクトリ構成

```
ai_e_board/
├── app.py                             # Streamlit メインアプリケーション（システムA/B 切替対応）
├── config.py                          # 設定ファイル（Ollama Host, Models, 各種パス）
├── requirements.txt                   # 依存パッケージ定義
├── pytest.ini                         # テスト設定
├── README.md                          # システム仕様・運用マニュアル
├── .gitignore                         # バージョン管理除外設定
│
├── system_b/                          # 【NEW】システムB（教育品質評価・授業運用支援）
│   ├── __init__.py
│   ├── analyzer.py                    # JSON/HTML教材パーサー & 定量的メトリクス算出
│   ├── pedagogy.py                    # 授業構成・学習指導要領照合評価
│   ├── usability.py                   # 16:9電子黒板UX & 認知負荷・画面分割評価
│   ├── quality_evaluator.py           # 統合品質スコアリング (0-100) & LLM定性補強
│   ├── improvement.py                 # 具体的改善提案 & 授業運用ガイド生成
│   ├── report.py                      # JSON & HTML評価レポート生成
│   └── pipeline.py                    # システムB実行ファサード
│
├── ai/                                # システムA（教材生成）コアAI層
│   ├── __init__.py
│   ├── vision.py                      # Ollama Vision API呼び出し・複数画像処理
│   ├── parser.py                      # JSON自動抽出・構文修復・Pydantic検証
│   ├── generator.py                   # 16:9スライド構築 & KaTeX HTML生成
│   └── evaluator.py                   # 評価用拡張インターフェース
│
├── models/                            # データモデル層
│   ├── __init__.py
│   ├── schemas.py                     # システムA用スキーマ (Lesson, Section, Formula, Slide等)
│   └── evaluation_schemas.py          # 【NEW】システムB用スキーマ (EvaluationResult, SlideMetrics等)
│
├── ui/                                # UI・表示層
│   ├── __init__.py
│   ├── board.py                       # 電子黒板表示コンポーネント (HTML/JS)
│   ├── evaluation.py                  # 【NEW】システムB評価ダッシュボードUI
│   └── styles.css                     # アプリケーションスタイルシート
│
├── prompts/                           # プロンプト管理
│   ├── analyze.txt                    # 板書解析プロンプト
│   ├── generate.txt                   # スライド構成プロンプト
│   ├── evaluate.txt                   # 教育評価プロンプト
│   ├── evaluate_quality.txt           # 【NEW】総合品質評価プロンプト
│   ├── evaluate_pedagogy.txt          # 【NEW】授業構成評価プロンプト
│   ├── evaluate_usability.txt         # 【NEW】電子黒板UX評価プロンプト
│   └── generate_improvement.txt       # 【NEW】改善提案・運用ガイドプロンプト
│
├── data/curriculum/                   # 学習指導要領データ
│   └── high_school_math_stub.json     # 高等学校学習指導要領スタブ（数学I・数学A）
│
├── test_data/                         # 研究用テスト板書画像
│   ├── sample_permutation_board.png   # 順列板書
│   ├── sample_quadratic_board.png     # 平方完成板書
│   └── generate_sample_boards.py      # テスト画像自動生成スクリプト
│
├── uploads/                           # アップロード画像保存先
├── outputs/                           # 生成成果物
│   ├── lesson_*.json / .html          # システムA生成教材
│   ├── evaluations/                   # システムB生成評価レポート (.json, .html)
│   ├── research_log.jsonl             # システムA研究ログ
│   └── evaluation_log.jsonl           # システムB評価ログ
│
└── tests/                             # 自動テストスイート（全24件）
    ├── test_schemas.py
    ├── test_parser.py
    ├── test_generator.py
    ├── test_evaluation_schemas.py
    ├── test_analyzer.py
    ├── test_pedagogy.py
    ├── test_usability.py
    ├── test_improvement.py
    └── test_quality_evaluator.py
```

---

## 4. スコアリング構造とハイブリッド評価設計

システムBでは、機械的な減点ルールを排除し、**客観的な定量的メトリクス**と**授業文脈をふまえたLLMによる定性評価**をハイブリッドで組み合わせます。

1. **定量的指標（着目フラグ）**:
   - スライドごとの文字数、数式数、箇条書き数、問題/解答混在フラグを自動算出。
   - 例: 「文字数多め（180字）: 教室後方からの視認性・提示順序に留意」「数式ステップ展開: 発問推奨」等の客観的な着目ポイントとして提示（機械的な減点理由にはしない）。
2. **定性的文脈評価（LLM + 教育工学ルール）**:
   - 高校数学の平方完成や証明問題など、式の展開過程が多くなる教育的意図を正しく加味。
   - 学習目標の明確さ、導入の動機づけ、例題と練習の接続性、まとめの有無を総合評価。
3. **スコア構造 (0〜100点)**:
   - **総合教育品質スコア (`overall_score`)**
   - 🧩 授業構成・指導展開スコア (`pedagogical_structure`)
   - 🎯 教育目標整合性スコア (`objective_alignment`)
   - 🖥️ 電子黒板UX・視認性スコア (`blackboard_usability`)
   - 🧠 認知負荷・情報密度スコア (`cognitive_load_balance`)

---

## 5. 起動方法と使い方

### ① 仮想環境のセットアップとライブラリインストール
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### ② Ollamaモデルの準備（ローカル実行）
```powershell
# Visionモデル (システムA)
ollama pull qwen3-vl:8b

# テキスト評価モデル (システムB)
ollama pull qwen2.5:7b
```

### ③ アプリケーションの起動
```powershell
streamlit run app.py
```

### ④ アプリケーションの使い方
- 左サイドバーの「🎛️ 機能モード切替」から、**「🎨 電子黒板教材生成 (システムA)」** または **「📊 教育品質評価・運用支援 (システムB)」** を自由に選択できます。
- システムAで教材を生成後、STEP 5 画面下部の **「📊 この教材の教育品質を評価・改善する（システムBへ連携）」** をクリックすると、生成された教材データを引き継いで即座に品質評価・授業運用ガイドを確認できます。

---

## 6. テストの実行

```powershell
python -m pytest tests/ -v
```
全24件のユニットテスト（スキーマ検証、HTML/JSON解析、スコアリング、改善提案生成、HTMLレポート出力、Ollamaフォールバック等）が全て正常にパスします。

