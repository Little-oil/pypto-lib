# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Static contracts for the DeepSeek-V4 MoE pre-normalization fusion.

These checks intentionally parse source instead of importing the model.  Importing
the model freezes the selected EP configuration and requires the PyPTO device
environment, neither of which is needed to protect the fusion ABI and barriers.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _REPO_ROOT / "models" / "deepseek" / "v4-flash"
_BRIDGE = _MODEL_DIR / "fused_pre_norm_cce.py"
_MOE = _MODEL_DIR / "moe.py"
_HC_PRE = _MODEL_DIR / "hc_pre.py"
_GATE = _MODEL_DIR / "gate.py"
_KERNEL_DIR = _MODEL_DIR / "kernels" / "fused_pre_norm_cce"
_FUSED_BODY = _KERNEL_DIR / "kernel" / "fused_body.hpp"
_PRODUCTION_ENTRY = _KERNEL_DIR / "entry.cpp"
_DEBUG_ENTRY = _KERNEL_DIR / "debug" / "entry.cpp"

_TENSOR_ARGS = (
    "x_mixed",
    "x_flat",
    "inv_rms",
    "mixes_raw",
    "hc_base",
    "norm_w",
    "pre_val_store",
    "post",
    "xg_buf",
    "ffn_inv_rms_buf",
    "xn_scale_buf",
    "x_norm_scale",
)
_SCALAR_ARGS = ("scale0", "scale1", "num_tokens")
_PHASE_OUTPUTS = (
    "x_mixed",
    "pre_val_store",
    "post",
    "xg_buf",
    "ffn_inv_rms_buf",
    "xn_scale_buf",
    "x_norm_scale",
)
_DUMP_INPUTS = ("x_flat", "inv_rms", "mixes_raw", "hc_base", "norm_w")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse(path: Path) -> ast.Module:
    return ast.parse(_read(path), filename=str(path))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _qualified_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _qualified_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _keyword(call: ast.Call, name: str) -> ast.AST:
    return next(keyword.value for keyword in call.keywords if keyword.arg == name)


def _annotation_kind(annotation: ast.AST | None) -> str:
    if isinstance(annotation, ast.Subscript):
        return _qualified_name(annotation.value).rsplit(".", 1)[-1]
    return _qualified_name(annotation).rsplit(".", 1)[-1]


def _extern_decorator(function: ast.FunctionDef) -> ast.Call:
    return next(
        decorator
        for decorator in function.decorator_list
        if isinstance(decorator, ast.Call)
        and _qualified_name(decorator.func) == "pl.jit.extern"
    )


def _calls(function: ast.FunctionDef, qualified_name: str) -> list[ast.Call]:
    return sorted(
        (
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and _qualified_name(node.func) == qualified_name
        ),
        key=lambda node: (node.lineno, node.col_offset),
    )


def _named_spmd_with(function: ast.FunctionDef, name_hint: str) -> list[ast.With]:
    result = []
    for node in ast.walk(function):
        if not isinstance(node, ast.With) or len(node.items) != 1:
            continue
        context = node.items[0].context_expr
        if (
            isinstance(context, ast.Call)
            and _qualified_name(context.func) == "pl.spmd"
            and ast.literal_eval(_keyword(context, "name_hint")) == name_hint
        ):
            result.append(node)
    return result


def _assigned_constant(tree: ast.Module, name: str) -> object:
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and (
            any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            )
            if isinstance(node, ast.Assign)
            else isinstance(node.target, ast.Name) and node.target.id == name
        )
    )
    return ast.literal_eval(assignment.value)


def _strip_cpp_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", source)


def _enum_symbols(source: str, enum_name: str) -> list[str]:
    match = re.search(
        rf"enum(?:\s+class)?\s+{re.escape(enum_name)}[^{{]*\{{(.*?)\}};",
        source,
        flags=re.DOTALL,
    )
    assert match is not None, f"missing C++ enum {enum_name}"
    return re.findall(r"\b(k[A-Za-z0-9_]+)\s*(?:=[^,\n]+)?\s*,", match.group(1))


def _dump_calls(function: ast.FunctionDef) -> list[tuple[int, str]]:
    result = []
    for call in _calls(function, "pl.dump_tag"):
        if call.args and isinstance(call.args[0], ast.Name):
            result.append((call.lineno, call.args[0].id))
    return result


def _argument_call(tree: ast.Module, option: str) -> ast.Call:
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _qualified_name(node.func).endswith(".add_argument")
        and any(
            isinstance(arg, ast.Constant) and arg.value == option
            for arg in node.args
        )
    )


def test_fused_extern_python_and_cpp_abis_stay_in_lockstep() -> None:
    tree = _parse(_BRIDGE)
    expected_kinds = (
        "Out",
        "Tensor",
        "Tensor",
        "Tensor",
        "Tensor",
        "Tensor",
        "Out",
        "Out",
        "Out",
        "Out",
        "Out",
        "Out",
        "Scalar",
        "Scalar",
        "Scalar",
    )

    for function_name, source_name, trailing_args in (
        ("fused_pre_norm_cce", "_ENTRY", ()),
        ("fused_pre_norm_debug_cce", "_DEBUG_ENTRY", ("stop_after",)),
    ):
        function = _function(tree, function_name)
        arguments = function.args.args
        assert tuple(argument.arg for argument in arguments) == (
            *_TENSOR_ARGS,
            *_SCALAR_ARGS,
            *trailing_args,
        )
        assert tuple(_annotation_kind(argument.annotation) for argument in arguments) == (
            *expected_kinds,
            *(("Scalar",) if trailing_args else ()),
        )

        output_like = [
            argument.arg
            for argument in arguments
            if _annotation_kind(argument.annotation) in {"Out", "InOut"}
        ]
        assert output_like == list(_PHASE_OUTPUTS)
        assert output_like[0] == "x_mixed", (
            "the first extern result must bind to the first pl.Out argument"
        )
        assert _annotation_kind(arguments[0].annotation) == "Out"

        decorator = _extern_decorator(function)
        assert ast.literal_eval(_keyword(decorator, "core_type")) == "aiv"
        assert ast.unparse(_keyword(decorator, "source")) == source_name

    # Direct tests poison-initialize every result buffer, so their public ABI
    # must preserve those allocations across the extern call.
    for function_name in ("fused_pre_norm_test", "fused_pre_norm_debug_test"):
        function = _function(tree, function_name)
        annotations = {
            argument.arg: _annotation_kind(argument.annotation)
            for argument in function.args.args
        }
        assert {name: annotations[name] for name in _PHASE_OUTPUTS} == {
            name: "InOut" for name in _PHASE_OUTPUTS
        }

    body = _read(_FUSED_BODY)
    assert _enum_symbols(body, "TensorArg") == [
        "kXMixed",
        "kXFlat",
        "kInvRms",
        "kMixesRaw",
        "kHcBase",
        "kNormWeight",
        "kPreValue",
        "kPost",
        "kXg",
        "kFfnInvRms",
        "kXnScale",
        "kXNormScale",
        "kTensorArgCount",
    ]
    assert re.search(r"\bkTensorArgCount\s*=\s*12\s*,", body)
    assert _enum_symbols(body, "ScalarArg") == [
        "kScale0",
        "kScale1",
        "kNumTokens",
        "kProductionArgCount",
        "kDebugStopAfter",
        "kDebugArgCount",
    ]
    assert re.search(r"\bkScale0\s*=\s*kTensorArgCount\s*,", body)
    assert re.search(r"\bkDebugStopAfter\s*=\s*kProductionArgCount\s*,", body)


def test_every_fused_launch_is_one_synchronously_started_48_aiv_wave() -> None:
    tree = _parse(_BRIDGE)
    assert _assigned_constant(tree, "FUSED_AIV_CORES") == 48

    for function_name, extern_name in (
        ("fused_pre_norm_test", "fused_pre_norm_cce"),
        ("fused_pre_norm_debug_test", "fused_pre_norm_debug_cce"),
    ):
        launches = _calls(_function(tree, function_name), "pl.spmd_submit")
        assert len(launches) == 1
        launch = launches[0]
        assert ast.unparse(launch.args[0]) == f"self.{extern_name}"
        assert ast.unparse(_keyword(launch, "core_num")) == "FUSED_AIV_CORES"
        assert ast.literal_eval(_keyword(launch, "sync_start")) is True

    model_launch = _calls(_function(_parse(_MOE), "moe"), "pl.spmd_submit")
    assert len(model_launch) == 1
    assert ast.unparse(model_launch[0].args[0]) == "self.fused_pre_norm_cce"
    assert ast.unparse(_keyword(model_launch[0], "core_num")) == (
        "FUSED_AIV_CORES"
    )
    assert ast.literal_eval(_keyword(model_launch[0], "sync_start")) is True
    assert ast.unparse(_keyword(model_launch[0], "deps")) == (
        "[hc_pre_rms_tid, hc_pre_linear_tid]"
    )

    body = _read(_FUSED_BODY)
    assert re.search(r"\bkAivLanes\s*=\s*48\s*;", body)


def test_moe_passes_producer_and_fused_task_ids_across_scheduler_boundaries() -> None:
    moe_tree = _parse(_MOE)
    function = _function(moe_tree, "moe")
    producer_call = _calls(function, "hc_pre_moe_producers")
    fused_call = _calls(function, "fused_pre_norm_cce")
    gate_call = _calls(function, "gate_precomputed")
    dispatch_call = _calls(function, "dispatch")
    comb_call = _calls(function, "hc_pre_moe_comb")
    assert (
        len(producer_call)
        == len(fused_call)
        == len(gate_call)
        == len(dispatch_call)
        == len(comb_call)
        == 1
    )

    producer_assignment = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign) and node.value is producer_call[0]
    )
    assert ast.unparse(producer_assignment.targets[0]) == (
        "(hc_pre_rms_tid, hc_pre_linear_tid)"
    )

    launch = _calls(function, "pl.spmd_submit")
    assert len(launch) == 1
    launch_assignment = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign) and node.value is launch[0]
    )
    assert ast.unparse(launch_assignment.targets[0]) == (
        "((x_mixed, pre_val_store, post_ffn, xg_buf, gate_inv_rms_buf, "
        "xn_scale_buf, x_norm_scale), fused_pre_norm_tid)"
    )

    fused_assignment = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign) and node.value is fused_call[0]
    )
    assert ast.unparse(fused_assignment.targets[0]) == (
        "(x_mixed, pre_val_store, post_ffn, xg_buf, gate_inv_rms_buf, "
        "xn_scale_buf, x_norm_scale)"
    )
    assert ast.unparse(gate_call[0].args[-1]) == "fused_pre_norm_tid"

    dispatch_assignment = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign) and node.value is dispatch_call[0]
    )
    assert ast.unparse(dispatch_assignment.targets[0]) == "dispatch_gather_tid"
    assert ast.unparse(comb_call[0].args[-1]) == "dispatch_gather_tid"
    assert dispatch_call[0].lineno < comb_call[0].lineno

    dispatch_function = _function(moe_tree, "dispatch")
    gather_scopes = _named_spmd_with(dispatch_function, "dispatch_gather")
    assert len(gather_scopes) == 1
    assert ast.unparse(gather_scopes[0].items[0].optional_vars) == "_gather_tid"
    assert any(
        isinstance(node, ast.Return)
        and ast.unparse(node.value) == "_gather_tid"
        for node in dispatch_function.body
    )


def test_comb_sinkhorn_uses_with_spmd_and_waits_directly_on_dispatch_gather() -> None:
    tree = _parse(_HC_PRE)

    standalone = _function(tree, "_hc_pre_separate")
    standalone_comb = _named_spmd_with(standalone, "comb_sinkhorn")
    assert len(standalone_comb) == 1
    assert isinstance(standalone_comb[0].body[0], ast.Assign)
    assert ast.unparse(standalone_comb[0].body[0]) == (
        "ob = pl.tile.get_block_idx()"
    )

    producers = _function(tree, "hc_pre_moe_producers")
    assert not _named_spmd_with(producers, "comb_sinkhorn")
    assert tuple(argument.arg for argument in producers.args.args) == (
        "x",
        "hc_fn",
        "inv_rms",
        "mixes_raw",
    )

    delayed_comb = _function(tree, "hc_pre_moe_comb")
    comb_scopes = _named_spmd_with(delayed_comb, "comb_sinkhorn")
    assert len(comb_scopes) == 1
    comb_call = comb_scopes[0].items[0].context_expr
    assert isinstance(comb_call, ast.Call)
    assert ast.unparse(_keyword(comb_call, "deps")) == "[dispatch_gather_tid]"
    assert "hc_pre_rms_tid" not in ast.unparse(delayed_comb)
    assert "hc_pre_linear_tid" not in ast.unparse(delayed_comb)

    moe = _function(_parse(_MOE), "moe")
    for tensor_name in ("hc_inv_rms", "mixes_raw"):
        create = next(
            node.value
            for node in ast.walk(moe)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == tensor_name
                for target in node.targets
            )
        )
        assert isinstance(create, ast.Call)
        assert _qualified_name(create.func) == "pl.create_tensor"
        assert ast.literal_eval(_keyword(create, "manual_dep")) is True


def test_aiv_fusion_uses_exactly_two_hardware_syncall_barriers() -> None:
    body = _strip_cpp_comments(_read(_FUSED_BODY))
    sync_calls = re.findall(
        r"AscendC::SyncAll\s*<\s*>\s*\(\s*\)\s*;",
        body,
    )
    assert len(sync_calls) == 2
    assert len(re.findall(r"AscendC::SyncAll", body)) == 2

    all_kernel_code = "\n".join(
        _strip_cpp_comments(_read(path))
        for path in sorted(_KERNEL_DIR.rglob("*"))
        if path.suffix in {".cpp", ".h", ".hpp"}
    )
    forbidden = {
        "SyncAll<false>": r"SyncAll\s*<\s*false\s*>",
        "software synchronization": (
            r"\b(?:soft(?:ware)?[_\s-]*(?:sync|barrier)|"
            r"(?:sync|barrier)[_\s-]*soft(?:ware)?)\b"
        ),
        "dcci": r"\bdcci\b",
        "dsb": r"\bdsb\b",
    }
    for description, pattern in forbidden.items():
        assert re.search(pattern, all_kernel_code, flags=re.IGNORECASE) is None, (
            f"pure-AIV fusion must not add {description}"
        )


def test_generated_phases_keep_grid_stride_mapping_and_barrier_order() -> None:
    body = _strip_cpp_comments(_read(_FUSED_BODY))
    phase_specs = (
        (
            "split_work",
            "deepseek_fused_pre_norm_split_generated::split_pre_post",
        ),
        (
            "mix_work",
            "deepseek_fused_pre_norm_mix_generated::mix_x",
        ),
        (
            "ffn_work",
            "deepseek_fused_pre_norm_ffn_generated::ffn_norm",
        ),
    )

    phase_positions = []
    for work_count, callee in phase_specs:
        loop = re.search(
            rf"for\s*\(\s*int32_t\s+logical_block\s*=\s*lane\s*;"
            rf"\s*logical_block\s*<\s*{work_count}\s*;"
            rf"\s*logical_block\s*\+=\s*kAivLanes\s*\)\s*\{{"
            rf"\s*{re.escape(callee)}\s*\((.*?)\)\s*;\s*\}}",
            body,
            flags=re.DOTALL,
        )
        assert loop is not None, f"{callee} must retain a 48-lane grid-stride loop"
        assert re.search(
            rf"logical_block\s*,\s*{work_count}\s*$",
            loop.group(1),
        ), f"{callee} must receive its original logical block count"
        phase_positions.append(body.index(f"{callee}("))

    barrier_positions = [
        match.start()
        for match in re.finditer(r"AscendC::SyncAll\s*<\s*>", body)
    ]
    assert (
        phase_positions[0]
        < barrier_positions[0]
        < phase_positions[1]
        < barrier_positions[1]
        < phase_positions[2]
    )


def test_debug_entry_can_stop_on_each_side_of_both_barriers() -> None:
    bridge_tree = _parse(_BRIDGE)
    expected_python_stops = {
        "STOP_SPLIT_BEFORE_BARRIER1": 0,
        "STOP_AFTER_BARRIER1": 1,
        "STOP_MIX_BEFORE_BARRIER2": 2,
        "STOP_AFTER_BARRIER2": 3,
        "STOP_FULL": 4,
    }
    assert {
        name: _assigned_constant(bridge_tree, name)
        for name in expected_python_stops
    } == expected_python_stops

    body = _read(_FUSED_BODY)
    assert _enum_symbols(body, "StopAfter") == [
        "kSplitBeforeBarrier1",
        "kAfterBarrier1",
        "kMixBeforeBarrier2",
        "kAfterBarrier2",
        "kFull",
    ]
    cpp_stops = dict(
        re.findall(
            r"\b(k(?:SplitBeforeBarrier1|AfterBarrier1|MixBeforeBarrier2|"
            r"AfterBarrier2|Full))\s*=\s*(\d+)",
            body,
        )
    )
    assert cpp_stops == {
        "kSplitBeforeBarrier1": "0",
        "kAfterBarrier1": "1",
        "kMixBeforeBarrier2": "2",
        "kAfterBarrier2": "3",
        "kFull": "4",
    }

    production_entry = _strip_cpp_comments(_read(_PRODUCTION_ENTRY))
    assert "StopAfter::kFull" in production_entry
    assert "kDebugStopAfter" not in production_entry
    assert "switch" not in production_entry

    debug_entry = _strip_cpp_comments(_read(_DEBUG_ENTRY))
    assert re.search(r"args\s*\[\s*deepseek_fused_pre_norm::kDebugStopAfter\s*\]", debug_entry)
    assert re.search(r"switch\s*\(\s*stop_after\s*\)", debug_entry)
    for stop_name in cpp_stops:
        assert f"StopAfter::{stop_name}" in debug_entry

    debug_map = next(
        node.value
        for node in ast.walk(bridge_tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "debug_stops"
            for target in node.targets
        )
    )
    assert isinstance(debug_map, ast.Dict)
    assert {
        ast.literal_eval(key): ast.unparse(value)
        for key, value in zip(debug_map.keys, debug_map.values)
    } == {
        "split": "STOP_SPLIT_BEFORE_BARRIER1",
        "barrier1": "STOP_AFTER_BARRIER1",
        "mix": "STOP_MIX_BEFORE_BARRIER2",
        "barrier2": "STOP_AFTER_BARRIER2",
        "full": "STOP_FULL",
    }


def test_dump_tags_bracket_the_extern_and_cover_consumer_side_buffers() -> None:
    bridge_tree = _parse(_BRIDGE)
    tag_helper = _function(bridge_tree, "_tag_test_tensors")
    assert {name for _, name in _dump_calls(tag_helper)} == set(
        (*_DUMP_INPUTS, *_PHASE_OUTPUTS),
    )
    for function_name, extern_name in (
        ("fused_pre_norm_test", "fused_pre_norm_cce"),
        ("fused_pre_norm_debug_test", "fused_pre_norm_debug_cce"),
    ):
        function = _function(bridge_tree, function_name)
        tag_calls = _calls(function, "_tag_test_tensors")
        extern_calls = _calls(function, extern_name)
        assert len(tag_calls) == 2
        assert len(extern_calls) == 1
        assert tag_calls[0].lineno < extern_calls[0].lineno < tag_calls[1].lineno

    model = _function(_parse(_MOE), "moe")
    model_extern = _calls(model, "fused_pre_norm_cce")
    assert len(model_extern) == 1
    model_extern_line = model_extern[0].lineno
    model_dumps = _dump_calls(model)
    before = {name for line, name in model_dumps if line < model_extern_line}
    after = {name for line, name in model_dumps if line > model_extern_line}
    assert {
        "x_flat",
        "hc_inv_rms",
        "mixes_raw",
        "hc_ffn_base",
        "norm_w",
        "x_mixed",
        "pre_val_store",
        "post_ffn",
        "xg_buf",
        "gate_inv_rms_buf",
        "xn_scale_buf",
        "x_norm_scale",
    } <= before
    assert {
        "x_mixed",
        "pre_val_store",
        "post_ffn",
        "xg_buf",
        "gate_inv_rms_buf",
        "xn_scale_buf",
        "x_norm_scale",
    } <= after

    gate = _function(_parse(_GATE), "gate_precomputed")
    assert {
        "xg_buf",
        "inv_rms_buf",
        "xn_scale_buf",
        "x_norm_scale",
    } <= {name for _, name in _dump_calls(gate)}


def test_model_and_standalone_clis_forward_partial_dump_level() -> None:
    for path in (_MOE, _BRIDGE):
        tree = _parse(path)
        argument = _argument_call(tree, "--dump-args")
        assert ast.literal_eval(_keyword(argument, "nargs")) == "?"
        assert ast.literal_eval(_keyword(argument, "const")) == 1
        assert ast.literal_eval(_keyword(argument, "default")) == 0
        assert ast.unparse(_keyword(argument, "type")) == "int"
        assert ast.literal_eval(_keyword(argument, "choices")) == (0, 1, 2, 3)

        forwarded = [
            keyword.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for keyword in node.keywords
            if keyword.arg == "enable_dump_args"
        ]
        assert forwarded
        assert {
            ast.unparse(value)
            for value in forwarded
        } == {"args.dump_args"}


def test_ep_expert_formula_keeps_sixteen_experts_per_rank() -> None:
    tree = _parse(_MOE)
    replacement = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and _qualified_name(target) == "config.FLASH"
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
        and _qualified_name(node.value.func) == "dataclasses.replace"
    )
    assert isinstance(replacement, ast.Call)
    expert_count = _keyword(replacement, "n_routed_experts")
    assert ast.unparse(expert_count) == (
        "config.FLASH.n_routed_experts // 16 * EP"
    )


def test_balanced_routing_default_has_three_routes_per_expert() -> None:
    for ep in (2, 4, 8):
        expert_count = 16 * ep
        route_count = ep * 8 * 6
        routes_per_expert, remainder = divmod(route_count, expert_count)
        assert remainder == 0
        assert routes_per_expert == 3
