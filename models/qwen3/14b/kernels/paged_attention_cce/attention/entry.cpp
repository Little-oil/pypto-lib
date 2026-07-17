/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <cstdint>

#include "tensor.h"

#ifdef __CPU_SIM
#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__ [aicore]
#endif

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) { (void)args; }

#else

#include <pto/pto-inst.hpp>

#include "../kernel/fai_body.hpp"

namespace {

static __aicore__ __attribute__((always_inline)) __gm__ int32_t *qwen_fai_barrier_data(GM_ADDR metadata) {
    uint64_t raw_barrier = reinterpret_cast<uint64_t>(metadata + qwen_fai_metadata::kBarrierAlignmentOffset);
    uint64_t aligned_barrier = (raw_barrier + qwen_fai_metadata::kBarrierAlignmentBytes - 1) &
                               ~(static_cast<uint64_t>(qwen_fai_metadata::kBarrierAlignmentBytes) - 1);
    return reinterpret_cast<__gm__ int32_t *>(aligned_barrier);
}

// AIV lanes finish the rope GM writes, barrier among themselves over the rope-ready
// metadata slots, then one sub-lane per AIC pair signals its AIC. The AIC waits for
// that flag before starting attention (which reads the rope-produced q/k/v cache).
// CrossCoreSetFlag uses PIPE_MTE3 so the signal fires only after rope's UB->GM
// stores have flushed — the cross-pipe GM visibility that a plain SYNCALL<Mix> lacks.
static __aicore__ __attribute__((always_inline)) void sync_qwen_rope_producers(
    __gm__ int64_t *args,
    __gm__ int32_t *fai_barrier
) {
    pipe_barrier(PIPE_ALL);
#if defined(__DAV_CUBE__)
    AscendC::CrossCoreWaitFlag(qwen_fai_metadata::kRopeReadyCrossCoreFlag);
#elif defined(__DAV_VEC__)
    __gm__ int32_t *ready = reinterpret_cast<__gm__ int32_t *>(
        reinterpret_cast<__gm__ uint8_t *>(fai_barrier) + qwen_fai_metadata::kBarrierBytes
    );
    uint32_t physical_lane = static_cast<uint32_t>(get_block_idx(args)) * 2 +
                             static_cast<uint32_t>(get_sub_block_id(args));
    __ubuf__ int32_t *ub_workspace = reinterpret_cast<__ubuf__ int32_t *>(0x3000);
    pto::SYNCALL_SOFT_AIV_BARRIER(
        ready,
        ub_workspace,
        qwen_fai_metadata::kRopeReadySlotCount,
        physical_lane
    );
    if (get_sub_block_id(args) == 0) {
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(qwen_fai_metadata::kRopeReadyCrossCoreFlag);
    }
#endif
    pipe_barrier(PIPE_ALL);
}

}  // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    GM_ADDR metadata = tensor_data<uint8_t>(args, kMetadataArg);
    acquire_qwen_fai_metadata(metadata);
    run_qwen_rope_qkv(args);
    __gm__ int32_t *fai_barrier = qwen_fai_barrier_data(metadata);
    sync_qwen_rope_producers(args, fai_barrier);
    __gm__ const FAInferTilingData *tiling = reinterpret_cast<__gm__ const FAInferTilingData *>(metadata);
    if (tiling->needCoreNum != 0) {
        run_qwen_fai<true>(args, fai_barrier);
    } else {
        run_qwen_fai<false>(args);
    }
}

#endif
