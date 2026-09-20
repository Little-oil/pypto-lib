# Compensated Cube scoring for the DSpark CSA decode Indexer

The opt-in `cube_compensated` implementation contracts the 64-head score on
Cube and sends one FP32 score per candidate to Vector. It preserves the
original score gate with compensated FP16 operands and FP32 accumulation.

## Measured result

TP=1, runtime B=16, eight queries per request, all 16 start positions 131072.
Both implementations use the same frozen original inputs and stock comparators.
Measurements used baseline `f3167ecbf8838ecff0e169ee8d854baafaa0236c`
on 2026-09-17, before rebasing this change onto newer upstream preprocessing.
They are not a new performance comparison against the rebased main branch.
Unprofiled device timings use five warmups and 100 measured rounds:

| Complete operation | Original median (us) | Compensated median (us) | Reduction |
|---|---:|---:|---:|
| Indexer | 1533.6705 | 1377.4795 | 10.18% |
| CSA | 2198.3500 | 2021.4905 | 8.05% |

A separate same-device, standalone Indexer L4 capture measured
`indexer_score_topk_leaf` at **1381.88 -> 1176.90 us (-14.83%)**.
This is one task-span observation per implementation, including on-core
waits and start skew; it is not a 100-round median or pure Cube compute time.
The full-CSA mean was 2396.99187 -> 2084.97455 us; medians above are the
primary latency metric.

The measured toolchain was PyPTO `2f892f96564d`, runtime `097735888e6d`,
PTOAS 0.61, PTO ISA `03e45c4bda48`, and CANN 9.0.0 on A2/A3.

The 65,536 selected Indexer scores have zero outliers against the actual
original Vector outputs. Selected sets match the Torch golden; comparison
with the actual Vector implementation retains one pre-existing boundary
selection difference. The stock index comparator passes. The complete CSA
output is bitwise identical to the original device output (2,097,152 FP32
values). Existing B=2 mixed-start, B=1, and B=16 shorter-context fixtures also
pass unchanged gates. These measurements correspond to the retained v8r
kernel; the kernel arithmetic is unchanged by the selector rename and style cleanup.

The actual maximum C2V improvement is a payload reduction, not a proportional
latency claim. For each 256-candidate tile, the original sends 64*256 INT32
values (64 KiB); this implementation sends 256 FP32 values (1 KiB), exactly
1/64. The target issues 16,464 such tiles, including padding: 1029 MiB versus
16.078125 MiB in each direction. This is logical GM payload, not measured
physical HBM traffic. Precision compensation, Key loading, local transfers,
synchronization, and TopK still take time.

## Execution

1. Query preprocessing, Key compression/cache writes, and head-weight
   projection retain their original dependencies.
2. An AIV preparation task computes `c = FP32(query_scale * weight)`, scales
   it by 16384, and writes three FP16 coefficient residuals. It also prepares
   21 KiB of reusable Cube constants. There is no full-cache Key-scale
   packing/gather pass.
3. The mixed leaf task uses 24 Cube / 48 Vector workers. Cube computes exact
   INT32 QK, forms two exact nonnegative FP16 R limbs locally, and contracts
   them with the three coefficient limbs using six FP32-accumulating GEMVs.
   The large R tensor never visits GM. This is the mathematical Matmul
   replacement, implemented with multiple Cube operations for precision.
4. Key tiles use two L1 buffers. The next Key DMA overlaps current math;
   low-limb Fixpipe movement overlaps high-limb GEMVs. Constants remain
   resident. Immutable page-table cache lines are invalidated once per leaf.
5. An eight-slot compact-score GM pipe feeds both paired AIVs. Vector applies
   the original FP32 Key scale to the compact score and accumulates each
   half-leaf in UB. TopK starts as soon as that leaf arrives, while Cube
   advances to the next leaf. Sort width is 512/1024/2048/4096 according to
   valid length; partial padding retains the original finite `-FLT_MAX` sentinel.
6. The unchanged query merge combines half-leaf Top-512 pairs.

Vector therefore performs compact-score scaling and TopK; the original
head-sized Broadcast/Mul/ReduceSum is eliminated. See [NUMERICS.md](NUMERICS.md)
for the bit-alias construction, coefficient bound, signed-cancellation
regression, and FP32 accumulation-order limitations.

## Selection and validation

Set `DSPARK_INDEXER_SCORE_IMPL=cube_compensated` before importing the model.
The default remains `vector`. The Cube path requires the DSpark 64-head,
128-wide, 32-row-page, S=8 configuration and the tested A2/A3 toolchain.
There is no automatic runtime coefficient-range guard or fallback. Arbitrary
signed FP32 inputs are not promised to be bitwise equivalent; inputs outside
the documented numerical contract should use `vector`.

From an initialized worktree, portable arithmetic tests need only PyTorch:

```bash
python models/deepseek_v4_flash_dspark/kernels/indexer_cube_score/test_compensated_numerics.py
```

The mixed-kernel device smoke checks shuffled physical pages, empty queries,
all important tail lengths, untouched storage, signed cancellation, and true
INT8 extremes. Run it on a device allocated by the normal task queue:

```bash
python models/deepseek_v4_flash_dspark/kernels/indexer_cube_score/test_fused.py -d "$TASK_DEVICE"
```

For the complete TP1 CSA target, pass all 16 positions explicitly; a single
scalar start position selects batch one in the CSA runner:

```bash
DSPARK_INDEXER_SCORE_IMPL=cube_compensated python models/deepseek_v4_flash_dspark/decode_csa.py \
  --tp 1 -d "$TASK_DEVICE" \
  --start-pos 131072,131072,131072,131072,131072,131072,131072,131072,131072,131072,131072,131072,131072,131072,131072,131072
```

[validation_results.json](validation_results.json) records the benchmark
configuration, pinned toolchain, source hashes, and measured results.
