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

#ifndef PYPTO_QWEN_FAI_METADATA_LAYOUT_H
#define PYPTO_QWEN_FAI_METADATA_LAYOUT_H

#include <cstdint>

namespace qwen_fai_metadata {

constexpr uint32_t kTilingOffset = 0;
constexpr uint32_t kTilingBytes = 2488;
constexpr uint32_t kCumulativeQOffset = 2488;
constexpr uint32_t kLengthArrayBytes = 16 * sizeof(int64_t);
constexpr uint32_t kKvLengthsOffset = kCumulativeQOffset + kLengthArrayBytes;
constexpr uint32_t kBarrierAlignmentOffset = kKvLengthsOffset + kLengthArrayBytes;
constexpr uint32_t kDcciLineBytes = 64;
constexpr uint32_t kBarrierAlignmentBytes = 512;
constexpr uint32_t kBarrierSlotBytes = 512;
constexpr uint32_t kBarrierSlotWords = kBarrierSlotBytes / sizeof(int32_t);
constexpr uint32_t kBarrierSlotCount = 48;
constexpr uint32_t kBarrierBytes = kBarrierSlotCount * kBarrierSlotBytes;
// Producer-only RoPE-ready soft-barrier region, laid out right after the aligned
// FAI barrier. The fused kernel's AIV lanes barrier among themselves here, then one
// sub-lane per AIC pair signals the AIC attention via CrossCoreSetFlag.
constexpr uint32_t kRopeReadySlotBytes = 32;
constexpr uint32_t kRopeReadySlotWords = kRopeReadySlotBytes / sizeof(int32_t);
constexpr uint32_t kRopeReadySlotCount = 48;
constexpr uint32_t kRopeReadyBytes = kRopeReadySlotCount * kRopeReadySlotBytes;
constexpr uint16_t kRopeReadyCrossCoreFlag = 0;
constexpr uint32_t kMetadataBytes = 29376;

static_assert(
    kBarrierAlignmentOffset + kBarrierAlignmentBytes - 1 + kBarrierBytes + kRopeReadyBytes <= kMetadataBytes,
    "metadata buffer does not cover the aligned FAI and RoPE-ready regions"
);

}  // namespace qwen_fai_metadata

#endif
