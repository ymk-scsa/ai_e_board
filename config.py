"""
Configuration settings for AI Electronic Blackboard Generation System (ai_e_board).
Supports flexible Ollama host and model configuration for research experimentation.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
PROMPTS_DIR = BASE_DIR / "prompts"
DATA_DIR = BASE_DIR / "data"
CURRICULUM_DIR = DATA_DIR / "curriculum"
TEST_DATA_DIR = BASE_DIR / "test_data"
UI_DIR = BASE_DIR / "ui"

EVALUATION_OUTPUT_DIR = OUTPUT_DIR / "evaluations"

# Ensure all essential directories exist
for directory in [UPLOAD_DIR, OUTPUT_DIR, PROMPTS_DIR, CURRICULUM_DIR, TEST_DATA_DIR, EVALUATION_OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Ollama Connection Configuration
# Can be overridden via environment variables
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# SSRF対策: 接続を許可するOllamaホスト名（カンマ区切り）
# OLLAMA_HOST のホスト名は ai.security.validate_ollama_host 側で常に許可される
OLLAMA_ALLOWED_HOSTS = [
    h.strip().lower()
    for h in os.getenv("OLLAMA_ALLOWED_HOSTS", "localhost,127.0.0.1,::1").split(",")
    if h.strip()
]

# アップロード許可拡張子
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Vision LLM Model Configuration（バックエンドが見つからない場合の表示用。実際の既定は下の LLM 設定）
DEFAULT_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "qwen3-vl:4b-instruct")

# Evaluation LLM Model Configuration for System B（Ollama を直接使う場合）
DEFAULT_EVALUATION_MODEL = os.getenv("OLLAMA_EVALUATION_MODEL", "qwen3:8b")

# ----------------------------------------------------------------------
# Local LLM runtime (ai/llm_client.py)
# ----------------------------------------------------------------------
# "auto": Foundry Local → Ollama → OpenAI互換サーバー の順に、GPUで動くものを選ぶ
LLM_BACKEND = os.getenv("LLM_BACKEND", "auto").lower()
# GPU実行を必須にする（CPUで動くバックエンドはエラーにする）。開発・テスト時のみ LLM_ALLOW_CPU=1
LLM_REQUIRE_GPU = os.getenv("LLM_REQUIRE_GPU", "1") not in ("0", "false", "False")
LLM_ALLOW_CPU = os.getenv("LLM_ALLOW_CPU", "0") in ("1", "true", "True")
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "600"))
# 画像の長辺（px）。Qwen3.5 は内部で約 960x672 に揃えるため、これ以上大きくしても精度は上がらない
LLM_IMAGE_MAX_SIDE = int(os.getenv("LLM_IMAGE_MAX_SIDE", "1280"))
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "8192"))  # Ollama のみ
VISION_MAX_OUTPUT_TOKENS = int(os.getenv("VISION_MAX_OUTPUT_TOKENS", "3000"))
# 板書解析（ai/board_pipeline.py）: ①画像の書き起こし → ②行番号による構成の整理 → ③Pythonで授業データを組み立て
TRANSCRIBE_PROMPT_FILE = os.getenv("TRANSCRIBE_PROMPT_FILE", "transcribe_board.txt")
STRUCTURE_PROMPT_FILE = os.getenv("STRUCTURE_PROMPT_FILE", "structure_board.txt")
TRANSCRIBE_MAX_TOKENS = int(os.getenv("TRANSCRIBE_MAX_TOKENS", "1800"))
STRUCTURE_MAX_TOKENS = int(os.getenv("STRUCTURE_MAX_TOKENS", "800"))

# Foundry Local（Windows / macOS。PCに合ったGPU版を自動選択）
FOUNDRY_VISION_MODEL = os.getenv("FOUNDRY_VISION_MODEL", "qwen3.5-4b")
# Ollama（NVIDIA/AMD/Apple Silicon でGPU動作。推論しない instruct 版を使う）
OLLAMA_VISION_MODEL_GPU = os.getenv("OLLAMA_VISION_MODEL_GPU", "qwen3-vl:4b-instruct")
# OpenAI互換サーバー（LM Studio / llama-server など）
OPENAI_COMPAT_BASE_URL_SET = "OPENAI_COMPAT_BASE_URL" in os.environ
OPENAI_COMPAT_BASE_URL = os.getenv("OPENAI_COMPAT_BASE_URL", "http://127.0.0.1:1234")
OPENAI_COMPAT_API_KEY = os.getenv("OPENAI_COMPAT_API_KEY", "")
OPENAI_COMPAT_MODEL = os.getenv("OPENAI_COMPAT_MODEL", "")
# OpenAI互換サーバーは実行デバイスを報告しないため、利用者が申告する（gpu / cpu）
OPENAI_COMPAT_DEVICE = os.getenv("OPENAI_COMPAT_DEVICE", "gpu").lower()

# Request Configuration
VISION_TIMEOUT_SECONDS = int(os.getenv("VISION_TIMEOUT_SECONDS", "180"))
EVALUATION_TIMEOUT_SECONDS = int(os.getenv("EVALUATION_TIMEOUT_SECONDS", "180"))
# システムBのLLM講評の出力上限（短いほど速い。Snapdragon X で約350トークン ≒ 30秒）
EVALUATION_MAX_OUTPUT_TOKENS = int(os.getenv("EVALUATION_MAX_OUTPUT_TOKENS", "350"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))

# Display Settings for Electronic Blackboard
ASPECT_RATIO = "16:9"
DEFAULT_THEME = "chalkboard"  # "chalkboard" (green/dark) or "whiteboard" (clean bright)
ENABLE_KEYBOARD_NAVIGATION = True

# Research Logging
RESEARCH_LOG_PATH = OUTPUT_DIR / "research_log.jsonl"
EVALUATION_LOG_PATH = OUTPUT_DIR / "evaluation_log.jsonl"

