"""
ai_e_board ローカルLLM セットアップツール（全PC共通）

使い方:
    python tools/setup_llm.py            # 診断 → モデル準備 → 互換パッチ → GPU動作テスト
    python tools/setup_llm.py --check    # 診断のみ（何も変更しない）
    python tools/setup_llm.py --restore  # 互換パッチを元に戻す
    python tools/setup_llm.py --model qwen3.5-2b --yes

処理内容:
  1. このPCのOS / CPU / GPU を表示する
  2. Foundry Local（Windows / macOS）を確認する。無ければインストール方法を表示する
     （Foundry Local が使えない環境では Ollama / OpenAI互換サーバーの準備方法を表示する）
  3. 画像対応モデルのGPU版がこのPCで選ばれるか確認し、未取得ならダウンロードする
  4. Qwen3.5 のモデルファイルに互換パッチを当てる（ai/model_patches.py。.orig に原本を保存）
  5. GPUで実際に画像を読めるか・速度はどれくらいかをテストする
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from ai import model_patches  # noqa: E402
from ai.llm_client import FoundryLocalBackend, LLMError, OllamaBackend  # noqa: E402


def _print(title: str, msg: str = "") -> None:
    print(f"\n=== {title} ===" + (f"\n{msg}" if msg else ""))


def _run(cmd: list, timeout: float = 60) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=timeout).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def gpu_names() -> list:
    system = platform.system()
    if system == "Windows":
        out = _run(["powershell", "-NoProfile", "-Command",
                    "(Get-CimInstance Win32_VideoController).Name"])
        return [ln.strip() for ln in out.splitlines() if ln.strip()]
    if system == "Darwin":
        out = _run(["system_profiler", "SPDisplaysDataType"])
        return [ln.split(":", 1)[1].strip() for ln in out.splitlines() if "Chipset Model" in ln]
    out = _run(["sh", "-c", "lspci 2>/dev/null | grep -Ei 'vga|3d|display'"])
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def foundry_install_hint() -> str:
    system = platform.system()
    if system == "Windows":
        return "winget install Microsoft.FoundryLocal"
    if system == "Darwin":
        return "brew tap microsoft/foundrylocal && brew install foundrylocal"
    return "（Linux 版 Foundry Local は https://github.com/microsoft/Foundry-Local を参照）"


def foundry_json(cli: str, *args: str, timeout: float = 120) -> dict:
    out = _run([cli, *args, "-o", "json"], timeout=timeout)
    lines = [ln for ln in out.splitlines() if ln.strip().startswith("{")]
    return json.loads(lines[-1]) if lines else {}


def foundry_cache_dir(cli: str | None) -> Path:
    default = Path.home() / ".foundry" / "cache" / "models"
    if not cli:
        return default
    settings = foundry_json(cli, "config", "show").get("settings", [])
    return Path(next((s["value"] for s in settings if s.get("key") == "cache-directory"), default))


def make_test_image() -> bytes:
    """Draw a simple board-like test image with ASCII math (no font dependency)."""
    from PIL import Image, ImageDraw, ImageFont

    im = Image.new("RGB", (960, 540), (30, 60, 40))
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("arial.ttf", 64)
    except OSError:
        font = ImageFont.load_default(size=64) if hasattr(ImageFont, "load_default") else None
    d.text((60, 80), "2x + 3 = 11", fill="white", font=font)
    d.text((60, 220), "x = 4", fill=(255, 220, 90), font=font)
    d.rectangle((40, 200, 400, 310), outline=(255, 220, 90), width=5)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def smoke_test(model: str) -> bool:
    backend = FoundryLocalBackend()
    _print("GPU動作テスト", f"モデル {model} で画像を読み取ります（初回はモデルの読み込みに時間がかかります）")
    try:
        # Reload so that freshly patched files are used.
        subprocess.run([backend.cli, "model", "unload", model], capture_output=True, timeout=120)
        res = backend.generate(model, "画像に書かれている式を、そのまま2行で書き写してください。",
                               images=[make_test_image()], max_tokens=64)
    except LLMError as e:
        print(f"  失敗: {e}")
        return False
    ok = "11" in res.text and "4" in res.text
    tps = res.tokens_per_second
    print(f"  デバイス: {res.device.upper()} / モデル: {res.model}")
    print(f"  応答: {res.text.strip()[:120]!r}")
    print(f"  所要時間: {res.elapsed_seconds:.1f}秒（最初の出力まで {res.first_token_seconds or 0:.1f}秒"
          + (f"、生成 {tps:.1f} トークン/秒" if tps else "") + "）")
    print("  結果: " + ("OK（GPUで画像を読めています）" if ok and res.device == "gpu" else "NG"))
    return ok and res.device == "gpu"


def setup_foundry(args) -> int:
    cli = shutil.which("foundry")
    if not cli:
        _print("Foundry Local", "見つかりません。次のコマンドでインストールしてから再実行してください:\n  "
               + foundry_install_hint())
        return setup_ollama_hint()

    _print("Foundry Local", f"CLI: {cli} / バージョン: {_run([cli, '--version'])}")
    info = foundry_json(cli, "model", "info", args.model)
    m = info.get("model") or {}
    if not m:
        print(f"  モデル「{args.model}」がカタログにありません。")
        return 1
    gpu_variants = [v for v in m.get("variants", []) if (v.get("device") or "").lower() == "gpu"]
    print(f"  モデル: {m.get('alias')} → このPCで選ばれるバリアント: {m.get('id')}（{m.get('device')}）")
    for v in m.get("variants", []):
        print(f"    - {v['id']:<40} {v['device']:<4} {v.get('executionProvider', ''):<28} "
              f"{v.get('fileSizeMb', 0) / 1024:.1f}GB {'取得済み' if v.get('cached') else '未取得'}")
    if not gpu_variants or (m.get("device") or "").lower() != "gpu":
        print("  警告: このPCではGPU版が選ばれません。GPUドライバを更新するか、Ollama（GPU対応）を使ってください。")
        if args.check:
            return 1

    cache = foundry_cache_dir(cli)
    if args.check:
        for d in model_patches.find_model_dirs(cache):
            rep = model_patches.patch_model_dir(d, dry_run=True)
            print(f"  パッチ状態 {d.parent.name}: embedding={rep.embedding} / template={rep.template}")
        return 0

    if not m.get("cached"):
        size = m.get("fileSizeMb", 0) / 1024
        if not args.yes:
            ans = input(f"  {m.get('id')}（約{size:.1f}GB）をダウンロードします。よろしいですか？ [y/N] ")
            if ans.strip().lower() not in ("y", "yes"):
                print("  中止しました。")
                return 1
        _print("ダウンロード", f"{m.get('id')} を取得しています…")
        subprocess.run([cli, "model", "download", args.model])

    _print("互換パッチ", f"キャッシュ: {cache}")
    dirs = model_patches.find_model_dirs(cache)
    if not dirs:
        print("  対象の Qwen3.5 画像対応モデルはありません。")
    failed = False
    for d in dirs:
        rep = model_patches.patch_model_dir(d)
        print(f"  {d.parent.name}/{d.name}: embedding={rep.embedding} / template={rep.template}")
        failed |= not rep.ok
    if failed:
        return 1
    return 0 if smoke_test(args.model) else 1


def setup_ollama_hint() -> int:
    _print("代替: Ollama")
    try:
        st = OllamaBackend().status()
    except ValueError as e:
        print(f"  {e}")
        return 1
    print(f"  {st.message}")
    if st.available:
        print(f"  GPU対応のPC（NVIDIA / AMD / Apple Silicon）では次のモデルを使えます:\n"
              f"    ollama pull {config.OLLAMA_VISION_MODEL_GPU}\n"
              f"  アプリ起動時に LLM_BACKEND=ollama を指定してください。")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description="ai_e_board ローカルLLM セットアップ")
    ap.add_argument("--model", default=config.FOUNDRY_VISION_MODEL, help="Foundry Local のモデル別名")
    ap.add_argument("--check", action="store_true", help="診断のみ（変更しない）")
    ap.add_argument("--restore", action="store_true", help="互換パッチを元に戻す")
    ap.add_argument("--yes", action="store_true", help="確認なしでダウンロードする")
    args = ap.parse_args()

    _print("このPC", f"OS: {platform.system()} {platform.release()} / アーキテクチャ: {platform.machine()}\n"
           f"CPU: {platform.processor() or '-'}\nGPU: {', '.join(gpu_names()) or '（検出できません）'}\n"
           f"Python: {sys.version.split()[0]} ({platform.architecture()[0]})")

    if args.restore:
        for d in model_patches.find_model_dirs(foundry_cache_dir(shutil.which("foundry"))):
            for p in model_patches.restore_model_dir(d):
                print(f"  元に戻しました: {p}")
        return 0

    t = time.time()
    code = setup_foundry(args)
    print(f"\n完了（{time.time() - t:.0f}秒）: " + ("成功" if code == 0 else "要対応"))
    return code


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.exit(main())
