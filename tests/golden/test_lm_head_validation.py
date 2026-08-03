# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for composed DeepSeek-V4 LM-head validation."""

import importlib.util
from pathlib import Path

import pytest
import torch


_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "deepseek"
    / "v4-flash"
    / "lm_head_validation.py"
)
_SPEC = importlib.util.spec_from_file_location("lm_head_validation", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
compute_lm_head_logits = _MODULE.compute_lm_head_logits
device_hidden_logits_allclose = _MODULE.device_hidden_logits_allclose


def _fixture(rank_count=2):
    tp_size = 2
    hidden = torch.arange(rank_count * 3 * 4, dtype=torch.float32).reshape(rank_count, 3, 4) / 10
    shards = torch.arange(tp_size * 3 * 4, dtype=torch.float32).reshape(tp_size, 3, 4) / 20
    weights = torch.stack([shards[rank % tp_size] for rank in range(rank_count)])
    row_indices = torch.tensor([[0, 2]] * rank_count, dtype=torch.int32)
    logits = compute_lm_head_logits(hidden, weights, row_indices, tp_size)
    return tp_size, hidden, weights, row_indices, logits


def _compare(cmp, actual, expected, hidden, weights, row_indices):
    return cmp(
        actual,
        expected,
        actual_outputs={"hidden_out": hidden, "logits": actual},
        expected_outputs={"logits": expected},
        inputs={"lm_head_weight": weights, "logit_row_indices": row_indices},
        rtol=1e-3,
        atol=1e-3,
    )


def test_logits_reference_uses_actual_hidden_output():
    tp_size, hidden, weights, row_indices, actual = _fixture()
    old_expected = compute_lm_head_logits(hidden + 0.1, weights, row_indices, tp_size)
    cmp = device_hidden_logits_allclose(
        tp_size,
        atol=1e-5,
        rtol=1e-5,
        max_error_ratio=0.0,
        max_e2e_abs_diff=0.5,
    )

    ok, detail = _compare(cmp, actual, old_expected, hidden, weights, row_indices)

    assert ok, detail


def test_logits_reference_still_catches_lm_head_error():
    tp_size, hidden, weights, row_indices, actual = _fixture()
    corrupted = actual.clone()
    corrupted[0, 0].add_(1.0)
    cmp = device_hidden_logits_allclose(
        tp_size,
        atol=1e-5,
        rtol=1e-5,
        max_error_ratio=0.0,
        max_e2e_abs_diff=0.5,
    )

    ok, detail = _compare(cmp, corrupted, actual, hidden, weights, row_indices)

    assert not ok
    assert "ratio_allclose fail" in detail


def test_logits_reference_rejects_nan():
    tp_size, hidden, weights, row_indices, actual = _fixture()
    actual[0, 0, 0] = torch.nan
    cmp = device_hidden_logits_allclose(
        tp_size,
        atol=1e-5,
        rtol=1e-5,
        max_error_ratio=1.0,
        max_e2e_abs_diff=0.5,
    )

    ok, detail = _compare(cmp, actual, actual, hidden, weights, row_indices)

    assert not ok
    assert "NaN=1" in detail


@pytest.mark.parametrize("nonfinite", [torch.nan, torch.inf])
def test_logits_reference_rejects_nonfinite_reference(nonfinite):
    tp_size, hidden, weights, row_indices, expected = _fixture()
    actual_hidden = hidden.clone()
    actual_hidden[0, 0, 0] = nonfinite
    cmp = device_hidden_logits_allclose(
        tp_size,
        atol=1e-5,
        rtol=1e-5,
        max_error_ratio=0.0,
        max_e2e_abs_diff=0.5,
    )

    ok, detail = _compare(cmp, expected, expected, actual_hidden, weights, row_indices)

    assert not ok
    assert "non-finite LM-head reference logits" in detail


def test_logits_reference_catches_catastrophic_upstream_error():
    tp_size, golden_hidden, weights, row_indices, expected = _fixture()
    actual_hidden = golden_hidden.clone()
    actual_hidden[0, 0, 0] = 1000.0
    actual = compute_lm_head_logits(actual_hidden, weights, row_indices, tp_size)
    cmp = device_hidden_logits_allclose(
        tp_size,
        atol=1e-5,
        rtol=1e-5,
        max_error_ratio=0.01,
        max_e2e_abs_diff=0.5,
    )

    ok, detail = _compare(cmp, actual, expected, actual_hidden, weights, row_indices)

    assert not ok
    assert "end-to-end logits max abs diff" in detail


def test_reference_matches_explicit_projection():
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    weights = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [1.0, -1.0]],
        ]
    )
    row_indices = torch.tensor([[1, -1]], dtype=torch.int32)

    actual = compute_lm_head_logits(hidden, weights, row_indices, tp_size=2)

    assert torch.equal(actual, torch.tensor([[[3.0, 4.0, 7.0, -1.0], [0.0, 0.0, 0.0, 0.0]]]))


def test_logits_reference_honors_one_percent_outlier_budget():
    tp_size = 2
    hidden = torch.ones(1, 1, 2)
    weights = torch.ones(2, 50, 2)
    row_indices = torch.zeros(1, 1, dtype=torch.int32)
    reference = compute_lm_head_logits(hidden, weights, row_indices, tp_size)
    cmp = device_hidden_logits_allclose(
        tp_size,
        atol=1e-5,
        rtol=1e-5,
        max_error_ratio=0.01,
        max_e2e_abs_diff=0.5,
    )
    one_outlier = reference.clone()
    one_outlier[0, 0, 0] += 0.1
    two_outliers = one_outlier.clone()
    two_outliers[0, 0, 1] += 0.1

    one_ok, one_detail = _compare(
        cmp, one_outlier, reference, hidden, weights, row_indices,
    )
    two_ok, _ = _compare(
        cmp, two_outliers, reference, hidden, weights, row_indices,
    )

    assert one_ok, one_detail
    assert not two_ok


@pytest.mark.parametrize("rank_count", [2, 4, 8])
def test_reference_uses_runtime_owner_count(rank_count):
    tp_size, hidden, weights, row_indices, logits = _fixture(rank_count)

    assert logits.shape == (rank_count, row_indices.shape[1], tp_size * weights.shape[1])
    assert torch.equal(
        logits[-1],
        compute_lm_head_logits(hidden[-1:], weights, row_indices[-1:], tp_size)[0],
    )
