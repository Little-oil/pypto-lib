# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU references and composed-output validation for the DeepSeek-V4 LM head."""


def compute_lm_head_logits(hidden_states, lm_head_weight, logit_row_indices, tp_size):
    """Project selected hidden rows against the rank-local vocabulary shards."""
    import torch

    hidden = hidden_states.float()
    # Card r holds shard r % tp_size, so the first group contains one copy of
    # every shard in global vocabulary order.
    full_weight = lm_head_weight[:tp_size].float().reshape(-1, hidden.shape[-1])
    max_logit_rows = logit_row_indices.shape[1]
    hidden_size = hidden.shape[2]
    full_logits = []
    for owner_rank in range(hidden.shape[0]):
        selected = torch.zeros((max_logit_rows, hidden_size), dtype=torch.float32)
        for row in range(max_logit_rows):
            source_row = int(logit_row_indices[owner_rank, row])
            if source_row >= 0:
                source_row = min(source_row, hidden.shape[1] - 1)
                selected[row].copy_(hidden[owner_rank, source_row])
        full_logits.append(torch.matmul(selected, full_weight.t()))
    return torch.stack(full_logits, dim=0)


def device_hidden_logits_allclose(
    tp_size,
    atol,
    rtol,
    max_error_ratio,
    max_e2e_abs_diff,
):
    """Validate LM-head accuracy while retaining a catastrophic end-to-end guard."""
    from golden import ratio_allclose
    import torch

    if max_e2e_abs_diff <= 0:
        raise ValueError(f"max_e2e_abs_diff must be > 0, got {max_e2e_abs_diff}")

    base_cmp = ratio_allclose(
        atol=atol,
        rtol=rtol,
        max_error_ratio=max_error_ratio,
    )

    def cmp(
        actual,
        _expected,
        *,
        actual_outputs,
        expected_outputs,
        inputs,
        rtol,
        atol,
    ):
        reference = compute_lm_head_logits(
            actual_outputs["hidden_out"].cpu(),
            inputs["lm_head_weight"].cpu(),
            inputs["logit_row_indices"].cpu(),
            tp_size,
        )
        if not bool(torch.isfinite(reference).all()):
            return False, "    non-finite LM-head reference logits"
        lm_head_ok, lm_head_detail = base_cmp(
            actual,
            reference,
            actual_outputs=actual_outputs,
            expected_outputs=expected_outputs,
            inputs=inputs,
            rtol=rtol,
            atol=atol,
        )
        if not lm_head_ok:
            return False, lm_head_detail

        e2e_diff = (actual.cpu().to(torch.float32) - _expected.cpu().to(torch.float32)).abs()
        if not bool(torch.isfinite(e2e_diff).all()):
            return False, "    non-finite end-to-end logits difference"
        max_diff, flat_max_pos = torch.max(e2e_diff.flatten(), dim=0)
        if float(max_diff.item()) > max_e2e_abs_diff:
            max_pos = tuple(int(i.item()) for i in torch.unravel_index(flat_max_pos, e2e_diff.shape))
            return False, (
                f"    end-to-end logits max abs diff={max_diff.item():.6g} "
                f"at {max_pos}, allowed<={max_e2e_abs_diff:.6g}"
            )
        return True, ""

    cmp.__name__ = (
        f"device_hidden_{base_cmp.__name__}"
        f"_with_e2e_max_abs_diff({max_e2e_abs_diff})"
    )
    return cmp
