# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the A3 EP2 CI device-candidate selector."""

import ast
from pathlib import Path

import pytest


def _load_selector():
    source_path = (
        Path(__file__).resolve().parents[2]
        / "models"
        / "deepseek_v4_flash_mtp"
        / "prefill_mtp.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_select_device_ids"
    )
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), source_path, "exec"), namespace)
    return namespace["_select_device_ids"]


@pytest.mark.parametrize(
    ("candidates", "expected"),
    [
        ([8, 9, 10], [8, 10]),
        ([9, 10, 11], [9, 10]),
        ([0, 2, 1], [0, 2]),
    ],
)
def test_a2a3_ep2_selects_cross_chip_pair(candidates, expected):
    original = list(candidates)

    assert _load_selector()(candidates, 2, "a2a3") == expected
    assert candidates == original


def test_exact_two_device_request_preserves_same_chip_pair():
    assert _load_selector()([8, 9], 2, "a2a3") == [8, 9]


def test_non_a2a3_platform_ignores_spare_candidate():
    assert _load_selector()([8, 9, 10], 2, "a5") == [8, 9]


def test_too_few_candidates_is_rejected():
    with pytest.raises(ValueError, match="need at least 2 devices"):
        _load_selector()([8], 2, "a2a3")


def test_spare_candidates_must_span_physical_chips():
    with pytest.raises(ValueError, match="requires candidates from two physical chips"):
        _load_selector()([8, 9, 8], 2, "a2a3")
