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

# Vision LLM Model Configuration
# Priority: Qwen3-VL > LLaMA 3.2 Vision > LLaVA > Custom
DEFAULT_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "qwen3-vl:8b")

# Evaluation LLM Model Configuration for System B
DEFAULT_EVALUATION_MODEL = os.getenv("OLLAMA_EVALUATION_MODEL", "qwen3:8b")

# Fallback/Candidate Models for Selection UI
CANDIDATE_VISION_MODELS = [
    "qwen3-vl:8b",
    "qwen3-vl:2b",
    "llama3.2-vision:11b",
    "llama3.2-vision:latest",
    "llava:7b",
    "llava:13b",
    "minicpm-v:latest",
    "qwen2.5-coder:7b", # fallback for text processing
    "qwen2.5:7b",
]

# Request Configuration
VISION_TIMEOUT_SECONDS = int(os.getenv("VISION_TIMEOUT_SECONDS", "180"))
EVALUATION_TIMEOUT_SECONDS = int(os.getenv("EVALUATION_TIMEOUT_SECONDS", "180"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))

# Display Settings for Electronic Blackboard
ASPECT_RATIO = "16:9"
DEFAULT_THEME = "chalkboard"  # "chalkboard" (green/dark) or "whiteboard" (clean bright)
ENABLE_KEYBOARD_NAVIGATION = True

# Research Logging
RESEARCH_LOG_PATH = OUTPUT_DIR / "research_log.jsonl"
EVALUATION_LOG_PATH = OUTPUT_DIR / "evaluation_log.jsonl"

