/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0.
 */

#ifndef PYPTO_DEEPSEEK_FUSED_PRE_NORM_BODY_HPP
#define PYPTO_DEEPSEEK_FUSED_PRE_NORM_BODY_HPP

#include <cstdint>

#include "intrinsic.h"
#include "kernel_operator.h"
#include "tensor.h"

#include "ffn_norm_generated.hpp"
#include "mix_x_generated.hpp"
#include "split_pre_post_generated.hpp"

namespace deepseek_fused_pre_norm {

// This entry is a pure-AIV extern. On Ascend A2/A3 it must be launched as one
// synchronously-started 48-AIV wave. Every lane reaches both SyncAll barriers,
// including lanes with no logical work in the surrounding phase.
constexpr int32_t kAivLanes = 48;
constexpr int64_t kTokenTile = 8;
constexpr int64_t kMixSlicesPerTokenTile = 4;
constexpr int64_t kGateTokenTile = 16;

// PyPTO packs every Tensor argument before every scalar argument. Keep this
// table in lockstep with the Python @pl.jit.extern signature.
enum TensorArg : int32_t {
  kXMixed = 0,       // BF16 [T, 4096], first Out and return[0]
  kXFlat = 1,        // FP32 [T, 16384]
  kInvRms = 2,       // FP32 [t_linear, 1]
  kMixesRaw = 3,     // FP32 [t_linear, 32]
  kHcBase = 4,       // FP32 [24]
  kNormWeight = 5,   // BF16 [4096] or [1, 4096]
  kPreValue = 6,     // FP32 [t_linear, 8]
  kPost = 7,         // FP32 [T, 4]
  kXg = 8,           // FP32 [T_PAD, 4096]
  kFfnInvRms = 9,    // FP32 [T_PAD, 1]
  kXnScale = 10,     // FP32 [T_PAD, 1]
  kXNormScale = 11,  // FP32 [T, 1]
  kTensorArgCount = 12,
};

enum ScalarArg : int32_t {
  kScale0 = kTensorArgCount,  // FP32 bit pattern
  kScale1,                    // FP32 bit pattern
  kNumTokens,                 // INT32 in the low 32 bits
  kProductionArgCount,
  kDebugStopAfter = kProductionArgCount,
  kDebugArgCount,
};

// The debug-only entry selects one of these compile-time bodies with one
// uniform scalar. Production always instantiates kFull and contains no
// stop-mode branch.
enum class StopAfter : int32_t {
  kSplitBeforeBarrier1 = 0,
  kAfterBarrier1 = 1,
  kMixBeforeBarrier2 = 2,
  kAfterBarrier2 = 3,
  kFull = 4,
};

template <typename T>
static __aicore__ __attribute__((always_inline)) __gm__ T *
tensor_data(__gm__ int64_t *args, int32_t index) {
  __gm__ Tensor *tensor = reinterpret_cast<__gm__ Tensor *>(args[index]);
  return reinterpret_cast<__gm__ T *>(tensor->buffer.addr) +
         tensor->start_offset;
}

static __aicore__ __attribute__((always_inline)) int64_t
tensor_dim(__gm__ int64_t *args, int32_t index, int32_t axis) {
  __gm__ Tensor *tensor = reinterpret_cast<__gm__ Tensor *>(args[index]);
  return static_cast<int64_t>(tensor->shapes[axis]);
}

static __aicore__ __attribute__((always_inline)) float
unpack_float_scalar(__gm__ int64_t *args, int32_t index) {
  union {
    uint64_t bits;
    float value;
  } scalar;
  scalar.bits = static_cast<uint64_t>(args[index]);
  return scalar.value;
}

static __aicore__ __attribute__((always_inline)) int32_t
compute_ffn_work(int64_t num_tokens, int64_t tokens) {
  int64_t active_tokens = num_tokens;
  if (active_tokens < 0) {
    active_tokens = 0;
  }
  if (active_tokens > tokens) {
    active_tokens = tokens;
  }

  int64_t active_gate_tokens =
      ((active_tokens + kGateTokenTile - 1) / kGateTokenTile) *
      kGateTokenTile;
  if (active_gate_tokens > tokens) {
    active_gate_tokens = tokens;
  }
  return static_cast<int32_t>(active_gate_tokens);
}

template <StopAfter Stop>
static __aicore__ __attribute__((always_inline)) void
run_fused_pre_norm(__gm__ int64_t *args) {
#ifdef __DAV_C220_VEC__
  const int32_t lane = static_cast<int32_t>(get_block_idx(args));
  const int64_t tokens = tensor_dim(args, kXMixed, 0);
  const int64_t t_linear = tensor_dim(args, kPreValue, 0);
  const int32_t split_work =
      static_cast<int32_t>(tokens / kTokenTile);
  const int32_t mix_work =
      static_cast<int32_t>(split_work * kMixSlicesPerTokenTile);
  const int64_t num_tokens =
      static_cast<int64_t>(static_cast<int32_t>(args[kNumTokens]));
  const int32_t ffn_work = compute_ffn_work(num_tokens, tokens);

  __gm__ bfloat16_t *x_mixed = tensor_data<bfloat16_t>(args, kXMixed);
  __gm__ float *x_flat = tensor_data<float>(args, kXFlat);
  __gm__ float *inv_rms = tensor_data<float>(args, kInvRms);
  __gm__ float *mixes_raw = tensor_data<float>(args, kMixesRaw);
  __gm__ float *hc_base = tensor_data<float>(args, kHcBase);
  __gm__ bfloat16_t *norm_weight =
      tensor_data<bfloat16_t>(args, kNormWeight);
  __gm__ float *pre_value = tensor_data<float>(args, kPreValue);
  __gm__ float *post = tensor_data<float>(args, kPost);
  __gm__ float *xg = tensor_data<float>(args, kXg);
  __gm__ float *ffn_inv_rms = tensor_data<float>(args, kFfnInvRms);
  __gm__ float *xn_scale = tensor_data<float>(args, kXnScale);
  __gm__ float *x_norm_scale = tensor_data<float>(args, kXNormScale);

  const float scale0 = unpack_float_scalar(args, kScale0);
  const float scale1 = unpack_float_scalar(args, kScale1);

  // Preserve the standalone split_pre_post logical block mapping. Physical
  // lanes with lane >= split_work do no work but still reach barrier 1.
  for (int32_t logical_block = lane; logical_block < split_work;
       logical_block += kAivLanes) {
    deepseek_fused_pre_norm_split_generated::split_pre_post(
        inv_rms, hc_base, mixes_raw, pre_value, post, scale0, scale1, t_linear,
        tokens, logical_block, split_work);
  }

  if constexpr (Stop == StopAfter::kSplitBeforeBarrier1) {
    return;
  }

  // AIV-only global barrier: publish every pre_value GM write before mix_x
  // repartitions work by (token tile, 1024-wide D slice).
  AscendC::SyncAll<>();

  if constexpr (Stop == StopAfter::kAfterBarrier1) {
    return;
  }

  // Preserve mix_x's logical block count rather than passing the physical
  // 48-lane count. Prefill T=128 has 64 tasks, hence the grid-stride loop.
  for (int32_t logical_block = lane; logical_block < mix_work;
       logical_block += kAivLanes) {
    deepseek_fused_pre_norm_mix_generated::mix_x(
        pre_value, x_mixed, x_flat, t_linear, tokens, tokens, logical_block,
        mix_work);
  }

  if constexpr (Stop == StopAfter::kMixBeforeBarrier2) {
    return;
  }

  // AIV-only global barrier: publish all four BF16 D slices for each token
  // before ffn_norm reloads the complete 4096-element row.
  AscendC::SyncAll<>();

  if constexpr (Stop == StopAfter::kAfterBarrier2) {
    return;
  }

  // ffn_norm preserves the gate's clamp/round-to-16/clamp-to-T semantics.
  // With num_tokens=0 this is a zero-trip loop after all 48 lanes have crossed
  // both barriers; T=128 can require multiple logical tokens per AIV lane.
  for (int32_t logical_block = lane; logical_block < ffn_work;
       logical_block += kAivLanes) {
    deepseek_fused_pre_norm_ffn_generated::ffn_norm(
        x_mixed, norm_weight, xg, ffn_inv_rms, x_norm_scale, xn_scale,
        logical_block, ffn_work);
  }
#else
  (void)args;
#endif  // __DAV_C220_VEC__
}

}  // namespace deepseek_fused_pre_norm

#endif  // PYPTO_DEEPSEEK_FUSED_PRE_NORM_BODY_HPP
