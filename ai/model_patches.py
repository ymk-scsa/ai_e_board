"""
Compatibility patches for Foundry Local Qwen3.5 multimodal models.

1. embedding.onnx — "bool Expand" rewrite
   The exported graph builds the image-token mask as
       Equal(input_ids, image_token) → Unsqueeze → Expand → Expand   (bool [B, S, hidden])
       → NonZero → Transpose ([B*S*hidden, 3] indices) → ScatterND(flattened image features)
   On some GPUs (e.g. Qualcomm Adreno via WebGPU/D3D12) the WebGPU EP fails to compile the bool Expand
   shader once an image is present ("Failed to create a WebGPU compute pipeline: Invalid ShaderModule
   'Expand'"). The rewrite is mathematically equivalent and avoids Expand entirely:
       Equal → NonZero → Transpose ([N, 2] row indices) → ScatterND(image_features[:N])
   (ONNX ScatterND with indices of last-dim 2 on rank-3 data updates whole hidden-size rows.)
   It also shrinks the scatter index tensor by a factor of `hidden_size * 3/2`.

2. chat_template.jinja — non-thinking default
   Some Foundry Qwen3.5 variants (e.g. 4B) ship a template that opens "<think>" by default, while the
   Foundry web service cannot pass `enable_thinking=false`. The model then writes long reasoning into
   the answer text. The patch makes non-thinking the default (same as the official small Qwen3.5 models
   and Foundry's own 2B template); `enable_thinking=true` still enables reasoning.

Both patches are detected structurally (not by model name), are idempotent, keep a `.orig` backup, and
can be reverted with `restore()`.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Minimal protobuf reader/writer (enough for ONNX ModelProto graph surgery)
# ---------------------------------------------------------------------------

Field = Tuple[int, int, object]  # (field_number, wire_type, value)


def _read_varint(b: bytes, i: int) -> Tuple[int, int]:
    result = shift = 0
    while True:
        c = b[i]
        i += 1
        result |= (c & 0x7F) << shift
        shift += 7
        if c < 0x80:
            return result, i


def parse_fields(b: bytes) -> List[Field]:
    i, out = 0, []
    while i < len(b):
        key, i = _read_varint(b, i)
        num, wire = key >> 3, key & 7
        if wire == 0:
            val, i = _read_varint(b, i)
        elif wire == 1:
            val, i = b[i:i + 8], i + 8
        elif wire == 2:
            n, i = _read_varint(b, i)
            val, i = b[i:i + n], i + n
        elif wire == 5:
            val, i = b[i:i + 4], i + 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        out.append((num, wire, val))
    return out


def _enc_varint(n: int) -> bytes:
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        if n:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def encode_fields(fields: List[Field]) -> bytes:
    out = bytearray()
    for num, wire, val in fields:
        out += _enc_varint((num << 3) | wire)
        if wire == 0:
            out += _enc_varint(val)  # type: ignore[arg-type]
        elif wire == 2:
            out += _enc_varint(len(val)) + val  # type: ignore[arg-type]
        else:
            out += val  # type: ignore[operator]
    return bytes(out)


def _s(v: object) -> str:
    return bytes(v).decode("utf-8", "replace")  # type: ignore[arg-type]


# ONNX field numbers
_MODEL_GRAPH = 7
_GRAPH_NODE, _GRAPH_VALUE_INFO = 1, 13
_NODE_INPUT, _NODE_OUTPUT, _NODE_NAME, _NODE_OP = 1, 2, 3, 4


@dataclass
class _Node:
    raw: bytes
    op: str
    name: str
    inputs: List[str]
    outputs: List[str]


def _node(raw: bytes) -> _Node:
    f = parse_fields(raw)
    return _Node(
        raw=raw,
        op=next((_s(v) for n, w, v in f if n == _NODE_OP), ""),
        name=next((_s(v) for n, w, v in f if n == _NODE_NAME), ""),
        inputs=[_s(v) for n, w, v in f if n == _NODE_INPUT],
        outputs=[_s(v) for n, w, v in f if n == _NODE_OUTPUT],
    )


def _make_node(op: str, name: str, inputs: List[str], outputs: List[str]) -> bytes:
    f: List[Field] = [(_NODE_INPUT, 2, i.encode()) for i in inputs]
    f += [(_NODE_OUTPUT, 2, o.encode()) for o in outputs]
    f += [(_NODE_NAME, 2, name.encode()), (_NODE_OP, 2, op.encode())]
    return encode_fields(f)


def _with_inputs(raw: bytes, inputs: List[str]) -> bytes:
    f = [x for x in parse_fields(raw) if x[0] != _NODE_INPUT]
    return encode_fields([(_NODE_INPUT, 2, i.encode()) for i in inputs] + f)


def graph_nodes(model_bytes: bytes) -> List[_Node]:
    for n, w, v in parse_fields(model_bytes):
        if n == _MODEL_GRAPH:
            return [_node(gv) for gn, gw, gv in parse_fields(v) if gn == _GRAPH_NODE]  # type: ignore[arg-type]
    return []


# ---------------------------------------------------------------------------
# Patch 1: embedding.onnx bool-Expand rewrite
# ---------------------------------------------------------------------------

class PatchError(RuntimeError):
    pass


def _find_expand_pattern(nodes: List[_Node]) -> Optional[dict]:
    """Locate Equal→Unsqueeze→Expand→Expand→NonZero→Transpose and ScatterND fed by Slice(Reshape(image_features))."""
    by_out = {o: n for n in nodes for o in n.outputs}
    scatter = next((n for n in nodes if n.op == "ScatterND"), None)
    if scatter is None or len(scatter.inputs) != 3:
        return None
    transpose = by_out.get(scatter.inputs[1])
    slice_ = by_out.get(scatter.inputs[2])
    if not transpose or transpose.op != "Transpose" or not slice_ or slice_.op != "Slice":
        return None
    nonzero = by_out.get(transpose.inputs[0])
    if not nonzero or nonzero.op != "NonZero":
        return None
    exp2 = by_out.get(nonzero.inputs[0])
    exp1 = by_out.get(exp2.inputs[0]) if exp2 and exp2.op == "Expand" else None
    unsq = by_out.get(exp1.inputs[0]) if exp1 and exp1.op == "Expand" else None
    equal = by_out.get(unsq.inputs[0]) if unsq and unsq.op == "Unsqueeze" else None
    reshape = by_out.get(slice_.inputs[0])
    if not equal or equal.op != "Equal" or not reshape or reshape.op != "Reshape":
        return None
    return {"equal": equal, "unsqueeze": unsq, "expand1": exp1, "expand2": exp2, "nonzero": nonzero,
            "transpose": transpose, "slice": slice_, "reshape": reshape, "scatter": scatter,
            "image_features": reshape.inputs[0]}


def is_embedding_patched(model_bytes: bytes) -> bool:
    nodes = graph_nodes(model_bytes)
    return any(n.name == "/NonZero_rows" for n in nodes) and not any(n.op == "Expand" for n in nodes)


def needs_embedding_patch(model_bytes: bytes) -> bool:
    return _find_expand_pattern(graph_nodes(model_bytes)) is not None


def rewrite_embedding(model_bytes: bytes) -> bytes:
    nodes = graph_nodes(model_bytes)
    pat = _find_expand_pattern(nodes)
    if pat is None:
        raise PatchError("embedding.onnx に想定したグラフ構造（bool Expand）が見つかりません。")
    drop_names = {pat[k].name for k in ("unsqueeze", "expand1", "expand2", "nonzero", "reshape")}
    # Producers used only by the dropped nodes (Constants / Shape) are removed too.
    consumers = {}
    for n in nodes:
        for i in n.inputs:
            consumers.setdefault(i, set()).add(n.name)
    changed = True
    while changed:
        changed = False
        for n in nodes:
            if n.name in drop_names or n.op not in ("Constant", "Shape"):
                continue
            users = set().union(*(consumers.get(o, set()) for o in n.outputs))
            if users and users <= drop_names:
                drop_names.add(n.name)
                changed = True
    dropped_tensors = {o for n in nodes if n.name in drop_names for o in n.outputs}

    new_model: List[Field] = []
    for num, wire, val in parse_fields(model_bytes):
        if num != _MODEL_GRAPH:
            new_model.append((num, wire, val))
            continue
        new_graph: List[Field] = []
        for gn, gw, gv in parse_fields(val):  # type: ignore[arg-type]
            if gn == _GRAPH_NODE:
                node = _node(gv)  # type: ignore[arg-type]
                if node.name in drop_names:
                    continue
                if node.name == pat["transpose"].name:
                    # mask [B,S] → indices [2,N]; Transpose(perm=[1,0]) → [N,2] (row indices)
                    new_graph.append((_GRAPH_NODE, 2, _make_node(
                        "NonZero", "/NonZero_rows", [pat["equal"].outputs[0]], [pat["nonzero"].outputs[0]])))
                if node.name == pat["slice"].name:
                    # take rows [0:N] of image_features [P, hidden] instead of the flattened features
                    gv = _with_inputs(gv, [pat["image_features"]] + node.inputs[1:])  # type: ignore[arg-type]
                new_graph.append((gn, gw, gv))
            elif gn == _GRAPH_VALUE_INFO:
                name = next((_s(v) for n, w, v in parse_fields(gv) if n == 1), "")  # type: ignore[arg-type]
                if name in dropped_tensors or name.startswith("/Cast"):
                    continue
                new_graph.append((gn, gw, gv))
            else:
                new_graph.append((gn, gw, gv))
        new_model.append((num, wire, encode_fields(new_graph)))
    out = encode_fields(new_model)
    _verify_rewritten(out, pat)
    return out


def _verify_rewritten(model_bytes: bytes, pat: dict) -> None:
    nodes = graph_nodes(model_bytes)
    if any(n.op == "Expand" for n in nodes):
        raise PatchError("書き換え後に Expand が残っています。")
    by_out = {o: n for n in nodes for o in n.outputs}
    scatter = next(n for n in nodes if n.op == "ScatterND")
    t = by_out[scatter.inputs[1]]
    nz = by_out[t.inputs[0]]
    if nz.op != "NonZero" or nz.inputs != [pat["equal"].outputs[0]]:
        raise PatchError("書き換え後の NonZero の入力が想定と異なります。")
    if by_out[scatter.inputs[2]].inputs[0] != pat["image_features"]:
        raise PatchError("書き換え後の Slice の入力が image_features ではありません。")
    defined = {o for n in nodes for o in n.outputs} | _graph_input_and_initializer_names(model_bytes)
    for n in nodes:
        for i in n.inputs:
            if i and i not in defined:
                raise PatchError(f"書き換え後のグラフで未定義のテンソルを参照しています: {i}")


def _graph_input_and_initializer_names(model_bytes: bytes) -> set:
    names = set()
    for n, w, v in parse_fields(model_bytes):
        if n != _MODEL_GRAPH:
            continue
        for gn, gw, gv in parse_fields(v):  # type: ignore[arg-type]
            if gn == 11:  # graph.input (ValueInfoProto.name = 1)
                names |= {_s(x) for f, ww, x in parse_fields(gv) if f == 1}  # type: ignore[arg-type]
            elif gn == 5:  # graph.initializer (TensorProto.name = 8)
                names |= {_s(x) for f, ww, x in parse_fields(gv) if f == 8}  # type: ignore[arg-type]
    return names


# ---------------------------------------------------------------------------
# Patch 2: chat template non-thinking default
# ---------------------------------------------------------------------------

_THINK_DEFAULT_ON = (
    "{%- if enable_thinking is defined and enable_thinking is false %}\n"
    "        {{- '<think>\\n\\n</think>\\n\\n' }}\n"
    "    {%- else %}\n"
    "        {{- '<think>\\n' }}\n"
    "    {%- endif %}"
)
_THINK_DEFAULT_OFF = (
    "{%- if enable_thinking is defined and enable_thinking is true %}\n"
    "        {{- '<think>\\n' }}\n"
    "    {%- else %}\n"
    "        {{- '<think>\\n\\n</think>\\n\\n' }}\n"
    "    {%- endif %}"
)


def _norm(t: str) -> str:
    return t.replace("\r\n", "\n")


def template_thinking_default(text: str) -> Optional[bool]:
    t = _norm(text)
    if _THINK_DEFAULT_ON in t:
        return True
    if _THINK_DEFAULT_OFF in t:
        return False
    return None


def rewrite_template(text: str) -> str:
    t = _norm(text)
    if _THINK_DEFAULT_ON not in t:
        raise PatchError("chat_template.jinja に想定した thinking 既定値のブロックが見つかりません。")
    return t.replace(_THINK_DEFAULT_ON, _THINK_DEFAULT_OFF)


# ---------------------------------------------------------------------------
# Model directory level API
# ---------------------------------------------------------------------------

@dataclass
class PatchReport:
    model_dir: Path
    embedding: str  # "patched" | "already" | "not_needed" | "missing" | "error: ..."
    template: str

    @property
    def ok(self) -> bool:
        return not (self.embedding.startswith("error") or self.template.startswith("error"))


def is_qwen35_multimodal(model_dir: Path) -> bool:
    cfg = model_dir / "genai_config.json"
    if not cfg.exists():
        return False
    try:
        m = json.loads(cfg.read_text(encoding="utf-8")).get("model", {})
    except (OSError, ValueError):
        return False
    return str(m.get("type", "")).startswith("qwen3_5") and "vision" in m and "embedding" in m


def find_model_dirs(cache_dir: Path) -> List[Path]:
    return sorted({p.parent for p in Path(cache_dir).rglob("genai_config.json") if is_qwen35_multimodal(p.parent)})


def _backup(path: Path) -> Path:
    bak = path.with_name(path.name + ".orig")
    if not bak.exists():
        shutil.copy2(path, bak)
    return bak


def patch_model_dir(model_dir: Path, dry_run: bool = False) -> PatchReport:
    model_dir = Path(model_dir)
    emb_status = tmpl_status = "missing"

    emb = model_dir / "embedding.onnx"
    if emb.exists():
        try:
            data = emb.read_bytes()
            if is_embedding_patched(data):
                emb_status = "already"
            elif needs_embedding_patch(data):
                new = rewrite_embedding(data)
                if not dry_run:
                    _backup(emb)
                    tmp = emb.with_name(emb.name + ".tmp")
                    tmp.write_bytes(new)
                    tmp.replace(emb)
                emb_status = "patched" if not dry_run else "needs_patch"
            else:
                emb_status = "not_needed"
        except Exception as e:  # noqa: BLE001 — report per file, keep going
            emb_status = f"error: {e}"

    tmpl = model_dir / "chat_template.jinja"
    if tmpl.exists():
        try:
            text = tmpl.read_text(encoding="utf-8")
            default = template_thinking_default(text)
            if default is False:
                tmpl_status = "already"
            elif default is True:
                if not dry_run:
                    _backup(tmpl)
                    tmpl.write_text(rewrite_template(text), encoding="utf-8", newline="\n")
                tmpl_status = "patched" if not dry_run else "needs_patch"
            else:
                tmpl_status = "not_needed"
        except Exception as e:  # noqa: BLE001
            tmpl_status = f"error: {e}"

    return PatchReport(model_dir, emb_status, tmpl_status)


def restore_model_dir(model_dir: Path) -> List[Path]:
    restored = []
    for name in ("embedding.onnx", "chat_template.jinja"):
        bak = Path(model_dir) / (name + ".orig")
        if bak.exists():
            shutil.copy2(bak, Path(model_dir) / name)
            restored.append(Path(model_dir) / name)
    return restored
