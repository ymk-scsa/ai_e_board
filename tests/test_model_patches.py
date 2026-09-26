"""
Tests for ai/model_patches.py using a synthetic ONNX graph with the same structure as
Foundry Local's Qwen3.5 embedding.onnx (no onnx package required).
"""

import json

import pytest

from ai import model_patches as mp


def _node(op, name, inputs, outputs):
    return mp._make_node(op, name, inputs, outputs)


def _value_info(name):
    return mp.encode_fields([(1, 2, name.encode())])


def _synthetic_embedding(with_expand=True) -> bytes:
    """Equal→Unsqueeze→Expand→Expand→NonZero→Transpose→ScatterND, as exported for Qwen3.5."""
    E = "/embed_tokens/Gather_output_0"
    nodes = [
        _node("Gather", "/embed_tokens/Gather", ["embed_tokens.weight", "input_ids"], [E]),
        _node("Constant", "/Constant", [], ["/Constant_output_0"]),
        _node("Equal", "/Equal", ["input_ids", "/Constant_output_0"], ["/Equal_output_0"]),
    ]
    if with_expand:
        nodes += [
            _node("Constant", "/Constant_1", [], ["/Constant_1_output_0"]),
            _node("Unsqueeze", "/Unsqueeze", ["/Equal_output_0", "/Constant_1_output_0"], ["/Unsqueeze_output_0"]),
            _node("Shape", "/Shape", [E], ["/Shape_output_0"]),
            _node("Expand", "/Expand", ["/Unsqueeze_output_0", "/Shape_output_0"], ["/Expand_output_0"]),
            _node("Shape", "/Shape_1", [E], ["/Shape_1_output_0"]),
            _node("Expand", "/Expand_1", ["/Expand_output_0", "/Shape_1_output_0"], ["/Expand_1_output_0"]),
            _node("NonZero", "/NonZero", ["/Expand_1_output_0"], ["/NonZero_output_0"]),
        ]
    nodes += [
        _node("Transpose", "/Transpose", ["/NonZero_output_0"], ["/Transpose_output_0"]),
        _node("Constant", "/Constant_2", [], ["/Constant_2_output_0"]),
        _node("Reshape", "/Reshape", ["image_features", "/Constant_2_output_0"], ["/Reshape_output_0"]),
        _node("Shape", "/Shape_2", ["/Transpose_output_0"], ["/Shape_2_output_0"]),
        _node("Constant", "/Constant_3", [], ["/Constant_3_output_0"]),
        _node("Gather", "/Gather", ["/Shape_2_output_0", "/Constant_3_output_0"], ["/Gather_output_0"]),
        _node("Constant", "/Constant_4", [], ["/Constant_4_output_0"]),
        _node("Constant", "/Constant_5", [], ["/Constant_5_output_0"]),
        _node("Constant", "/Constant_6", [], ["/Constant_6_output_0"]),
        _node("Unsqueeze", "/Unsqueeze_1", ["/Gather_output_0", "/Constant_6_output_0"], ["/Unsqueeze_1_output_0"]),
        _node("Slice", "/Slice", ["/Reshape_output_0", "/Constant_5_output_0", "/Unsqueeze_1_output_0",
                                  "/Constant_4_output_0"], ["/Slice_output_0"]),
        _node("ScatterND", "/ScatterND", [E, "/Transpose_output_0", "/Slice_output_0"], ["inputs_embeds"]),
    ]
    graph = [(1, 2, n) for n in nodes]
    graph += [(5, 2, mp.encode_fields([(1, 0, 248320), (1, 0, 2560), (2, 0, 10), (8, 2, b"embed_tokens.weight")]))]
    graph += [(11, 2, _value_info("input_ids")), (11, 2, _value_info("image_features")),
              (12, 2, _value_info("inputs_embeds"))]
    graph += [(13, 2, _value_info(n)) for n in ("/Expand_output_0", "/Equal_output_0", "/Reshape_output_0")]
    model = [(1, 0, 8), (7, 2, mp.encode_fields(graph)), (8, 2, mp.encode_fields([(1, 2, b""), (2, 0, 20)]))]
    return mp.encode_fields(model)


def _ops(model_bytes):
    return [(n.op, n.name) for n in mp.graph_nodes(model_bytes)]


def test_protobuf_roundtrip_is_lossless():
    data = _synthetic_embedding()
    assert mp.encode_fields(mp.parse_fields(data)) == data


def test_rewrite_removes_bool_expand_and_rewires_scatter():
    data = _synthetic_embedding()
    assert mp.needs_embedding_patch(data) and not mp.is_embedding_patched(data)
    new = mp.rewrite_embedding(data)

    nodes = mp.graph_nodes(new)
    ops = [n.op for n in nodes]
    assert "Expand" not in ops and "Reshape" not in ops
    by_name = {n.name: n for n in nodes}
    assert by_name["/NonZero_rows"].inputs == ["/Equal_output_0"]
    assert by_name["/Transpose"].inputs == ["/NonZero_output_0"]
    assert by_name["/Slice"].inputs[0] == "image_features"
    assert by_name["/ScatterND"].inputs == ["/embed_tokens/Gather_output_0", "/Transpose_output_0", "/Slice_output_0"]
    # Constants/Shapes that only fed the removed nodes are dropped
    for dropped in ("/Constant_1", "/Shape", "/Shape_1", "/Constant_2", "/Unsqueeze"):
        assert dropped not in by_name
    # Topological order: NonZero before Transpose
    names = [n.name for n in nodes]
    assert names.index("/NonZero_rows") < names.index("/Transpose")
    # value_info for removed tensors is dropped, others kept; model-level fields preserved
    top = mp.parse_fields(new)
    assert top[0] == (1, 0, 8) and top[-1][0] == 8
    assert mp.is_embedding_patched(new) and not mp.needs_embedding_patch(new)


def test_rewrite_refuses_unexpected_graph():
    with pytest.raises(mp.PatchError):
        mp.rewrite_embedding(_synthetic_embedding(with_expand=False))


TEMPLATE_ON = (
    "{%- if add_generation_prompt %}\n"
    "    {{- '<|im_start|>assistant\\n' }}\n"
    "    {%- if enable_thinking is defined and enable_thinking is false %}\n"
    "        {{- '<think>\\n\\n</think>\\n\\n' }}\n"
    "    {%- else %}\n"
    "        {{- '<think>\\n' }}\n"
    "    {%- endif %}\n"
    "{%- endif %}"
)


def test_template_patch_makes_non_thinking_default():
    assert mp.template_thinking_default(TEMPLATE_ON) is True
    new = mp.rewrite_template(TEMPLATE_ON)
    assert mp.template_thinking_default(new) is False
    assert "enable_thinking is true" in new
    # CRLF input is handled too
    assert mp.template_thinking_default(mp.rewrite_template(TEMPLATE_ON.replace("\n", "\r\n"))) is False


def _model_dir(tmp_path, template=TEMPLATE_ON):
    d = tmp_path / "Microsoft" / "qwen3.5-4b-generic-gpu-4" / "v4"
    d.mkdir(parents=True)
    (d / "genai_config.json").write_text(json.dumps(
        {"model": {"type": "qwen3_5", "vision": {}, "embedding": {}, "decoder": {}}}), encoding="utf-8")
    (d / "embedding.onnx").write_bytes(_synthetic_embedding())
    (d / "chat_template.jinja").write_text(template, encoding="utf-8")
    return d


def test_patch_model_dir_is_idempotent_and_restorable(tmp_path):
    d = _model_dir(tmp_path)
    original = (d / "embedding.onnx").read_bytes()
    assert mp.find_model_dirs(tmp_path) == [d]

    dry = mp.patch_model_dir(d, dry_run=True)
    assert (dry.embedding, dry.template) == ("needs_patch", "needs_patch")
    assert (d / "embedding.onnx").read_bytes() == original  # dry run changes nothing

    rep = mp.patch_model_dir(d)
    assert (rep.embedding, rep.template) == ("patched", "patched") and rep.ok
    assert (d / "embedding.onnx.orig").read_bytes() == original
    patched = (d / "embedding.onnx").read_bytes()

    again = mp.patch_model_dir(d)
    assert (again.embedding, again.template) == ("already", "already")
    assert (d / "embedding.onnx").read_bytes() == patched

    restored = mp.restore_model_dir(d)
    assert len(restored) == 2
    assert (d / "embedding.onnx").read_bytes() == original
    assert mp.template_thinking_default((d / "chat_template.jinja").read_text(encoding="utf-8")) is True


def test_non_qwen_models_are_ignored(tmp_path):
    d = tmp_path / "other" / "v1"
    d.mkdir(parents=True)
    (d / "genai_config.json").write_text(json.dumps({"model": {"type": "phi3"}}), encoding="utf-8")
    assert mp.find_model_dirs(tmp_path) == []
