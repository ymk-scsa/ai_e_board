"""
Shared test configuration.

Every test writes generated lessons, evaluation reports, research / evaluation logs and uploads into its
own temporary directory, so running the suite never touches the real outputs/ and uploads/ folders.
"""

import pytest

import config


@pytest.fixture(autouse=True)
def _isolate_outputs(tmp_path, monkeypatch):
    out = tmp_path / "outputs"
    (out / "evaluations").mkdir(parents=True)
    monkeypatch.setattr(config, "OUTPUT_DIR", out)
    monkeypatch.setattr(config, "EVALUATION_OUTPUT_DIR", out / "evaluations")
    monkeypatch.setattr(config, "RESEARCH_LOG_PATH", out / "research_log.jsonl")
    monkeypatch.setattr(config, "EVALUATION_LOG_PATH", out / "evaluation_log.jsonl")
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "uploads")
    yield out
