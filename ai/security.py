"""
Security helpers for ai_e_board.
- safe_upload_path: アップロードファイルの保存先を安全に生成（パストラバーサル対策）
- validate_ollama_host: Ollama接続先ホストの許可リスト検証（SSRF対策）
"""

import uuid
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlsplit

import config

DEFAULT_ALLOWED_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg"})


def safe_upload_path(
    original_name: str,
    base_dir: Path,
    session_id: Optional[str] = None,
    allowed_extensions: Optional[Iterable[str]] = None,
) -> Path:
    """
    Build a safe, unique save path for an uploaded file.

    The original filename is never used as a path component; only its (lower-cased)
    suffix is kept after validation. The file is stored as ``<uuid4hex><suffix>`` under
    ``base_dir / (session_id or "default")``. Raises ValueError on a disallowed extension
    or if the resolved path would escape ``base_dir``.
    """
    allowed = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in (allowed_extensions or DEFAULT_ALLOWED_EXTENSIONS)}

    # Normalize both separator styles so "..\\x.png" or "C:\\evil.png" only yield a bare name
    name_only = Path(str(original_name).replace("\\", "/")).name
    suffix = Path(name_only).suffix.lower()
    if suffix not in allowed:
        raise ValueError(f"許可されていないファイル形式です: {suffix or '(拡張子なし)'}")

    subdir = session_id or "default"
    # session_id must be a plain token (no separators / traversal)
    if not subdir.replace("-", "").replace("_", "").isalnum():
        raise ValueError("不正なセッションIDです。")

    base_resolved = Path(base_dir).resolve()
    target_dir = base_resolved / subdir
    target = (target_dir / f"{uuid.uuid4().hex}{suffix}").resolve()

    if base_resolved != target and base_resolved not in target.parents:
        raise ValueError("保存先がアップロードディレクトリ外を指しています。")

    target_dir.mkdir(parents=True, exist_ok=True)
    return target


def _allowed_hostnames() -> set:
    allowed = {h.strip().lower().strip("[]") for h in getattr(config, "OLLAMA_ALLOWED_HOSTS", []) if h.strip()}
    default_host = urlsplit(config.OLLAMA_HOST if "://" in config.OLLAMA_HOST else f"http://{config.OLLAMA_HOST}").hostname
    if default_host:
        allowed.add(default_host.lower())
    return allowed


def validate_ollama_host(host: str) -> str:
    """
    Validate an Ollama host string against the OLLAMA_ALLOWED_HOSTS allowlist.
    Accepts "http(s)://host[:port]" or bare "host:port". Returns the normalized URL
    (scheme://host[:port]). Raises ValueError (Japanese message) if not allowed.
    """
    if not host or not str(host).strip():
        raise ValueError("Ollama接続ホストが空です。")

    raw = str(host).strip()
    url = raw if "://" in raw else f"http://{raw}"

    try:
        parts = urlsplit(url)
        port = parts.port  # raises ValueError on invalid port
    except ValueError as e:
        raise ValueError(f"Ollama接続ホストの形式が不正です: {raw}") from e

    if parts.scheme not in ("http", "https"):
        raise ValueError(f"Ollama接続ホストのスキームは http または https のみ許可されています: {raw}")

    hostname = (parts.hostname or "").lower()
    if not hostname:
        raise ValueError(f"Ollama接続ホストのホスト名を解釈できません: {raw}")

    if parts.username or parts.password:
        raise ValueError("Ollama接続ホストに認証情報を含めることはできません。")

    if hostname not in _allowed_hostnames():
        raise ValueError(
            f"Ollama接続ホスト「{hostname}」は許可されていません。"
            f"環境変数 OLLAMA_ALLOWED_HOSTS に追加してください。"
        )

    host_part = f"[{hostname}]" if ":" in hostname else hostname
    return f"{parts.scheme}://{host_part}" + (f":{port}" if port is not None else "")
