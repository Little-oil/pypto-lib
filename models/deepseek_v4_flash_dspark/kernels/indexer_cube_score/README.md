# Compensated Cube scoring for the DSpark CSA decode Indexer

The opt-in `cube_compensated` implementation contracts the 64-head score on
Cube and sends one FP32 score per candidate to Vector. It preserves the
original score gate with compensated FP16 operands and FP32 accumulation.

## Measured result

TP=1, runtime B=16, eight queries per request, all 16 start positions 131072.
The fresh 2026-09-19 comparison uses exact main commit
`eacafcfd77187e8cb9fc5d3ae03dfec844f4d78d` as the Vector baseline and PR
implementation `af71111a85bf258581bad87a87f73fde2a402642` on the same device 0.
Both implementations replay the same frozen inputs and stock comparators.
Profiling is disabled for complete-operation timings below.

Indexer uses A-B-B-A order, five warmups and 100 timed rounds per session,
for 200 samples per implementation. CSA first uses the same protocol,
then a separate five-warmup, 1000-round A/B confirmation because the initial
baseline session medians cross between two timing modes. All samples are
retained; CSA rows below report the 1000-round confirmation.

| Operation and metric | Baseline (us) | Compensated (us) | Reduction |
|---|---:|---:|---:|
| Indexer median, 200 samples each | 1520.12 | 1358.84 | 10.61% |
| Indexer mean, 200 samples each | 1530.01 | 1363.63 | 10.87% |
| CSA mean, 1000 samples each | 2639.10 | 2401.31 | 9.01% |
| CSA median, 1000 samples each | 2167.30 | 2016.21 | 6.97% |
| CSA p95, nearest rank | 3320.78 | 3088.62 | 6.99% |

CSA has a bimodal distribution: 446/1000 baseline and 380/1000 compensated
samples exceed 2800 us. The initial 100-round baseline session medians were
2170.11 and 3148.52 us, while their means were 2675.17 and 2718.40 us.
The initial A-B-B-A pooled means were 2696.78 -> 2384.52 us. The confirmation
supports an observed mean reduction; the median alone is sensitive to the
fraction in each mode and is not a stable standalone speedup estimate.
This measurement does not identify the cause of the two modes.

A separate same-device, standalone Indexer L4 capture measured
`indexer_score_topk_leaf` at **1354.04 -> 1185.56 us (-12.44%)**.
This is one task-span observation per implementation, including on-core
waits and start skew; it is not a benchmark median or pure Cube compute time.
The coefficient preparation task spans 15.50 us in the compensated capture.

The measured toolchain was PyPTO `2f892f96564d`, runtime `097735888e6d`,
PTOAS 0.61, PTO ISA `03e45c4bda48`, and CANN 9.0.0 on A2/A3.
Each fresh process begins with the frozen fixture. The standard benchmark
repeats the same fixed positions without restoring tensors each round;
metadata checks confirm disjoint historical-state reads and current writes.
This is a repeated fixed-step measurement, not consecutive decode positions.

All fresh stock precision gates pass. The 65,536 ranked Indexer scores have
zero outliers against the actual current Vector outputs, with maximum absolute
difference 2.2888184e-5. The compensated selected sets match the Torch golden;
comparison with Vector has one boundary selection difference at query 80
(Vector selects 32712, Cube selects 28162). The original index comparator
passes. Complete CSA output is bitwise identical to Vector across all
2,097,152 FP32 values, and both implementations repeat consistently.
The committed validation record preserves these fresh results and the older
2026-09-17 measurements under a separate historical entry.

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
