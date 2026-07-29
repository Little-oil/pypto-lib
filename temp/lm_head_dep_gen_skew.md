# lm_head 跨卡启动偏差：dep-gen 影响记录

日期：2026-07-29

工作树：`pypto-lib-worktrees/test-lm-head`
场景：`models/deepseek/v4-flash/decode_fwd.py -p a2a3 -d 0,2,4,6 --ep 4 --tp 4 --enable-l2-swimlane`

## 现象

在 lm_head 前的 d0 (`decode_fwd`) 末尾，各卡的 `runner.end` 有明显偏差。旧的
L2 swimlane 采集结果中，四卡 d0 结束时间的最大跨度为 **21.277345 ms**。
对应的 `dep_gen.stop` 时长为：

| rank | `dep_gen.stop` 时长 |
| --- | ---: |
| 0 | 947.549570 ms |
| 1 | 931.632527 ms |
| 2 | 928.744439 ms |
| 3 | 932.151062 ms |

rank 0 与 rank 2 的 `dep_gen.stop` 相差 **18.805131 ms**，与 d0 跨卡结束偏差同量级。
这会把后续 d1/lm_head 的提交和开始时间推迟，表现为 lm_head 之前有几十 ms 的卡间错位。

## 原因

`DistributedWorker`（prepared/resident 路径）在启用 L2 swimlane 时，会自动同时打开
dependency generation，即使业务侧没有显式设置 `enable_dep_gen=True`：

```python
call_config.enable_dep_gen = bool(
    dfx.enable_dep_gen or (co_enable_swimlane_dep_gen and dfx.enable_l2_swimlane)
)
```

`_make_call_config()` 的 `co_enable_swimlane_dep_gen` 默认值是 `True`。prepared worker
是单次 dispatch 采集，因此会在同一次运行里执行 dep-gen；d0 收尾时的
`dep_gen.stop()` 会等待、汇总和写出依赖信息 (`deps.json`)。各卡该工作量和主机调度不同，
故导致 d0 结束时间错位。

普通 one-shot 路径已经采用两次执行：第一次只生成 deps，第二次只计时，并在计时轮传入
`co_enable_swimlane_dep_gen=False`。prepared/resident 路径此前没有这样做，因为它不能在
两轮之间重新 fork 子进程（会触发 `halHostRegister` 上限）。

## 对照实验：不记录 deps

保留 `--enable-l2-swimlane 2`，但在 prepared 路径调用 `_make_call_config()` 时传入：

```python
co_enable_swimlane_dep_gen=False
```

且不显式开启 `enable_dep_gen`。结果：

| 指标 | 原先记录 deps | 不记录 deps |
| --- | ---: | ---: |
| d0 `runner.end` 最大跨卡跨度 | 21.277345 ms | 5.883096 ms |
| d0 `runner` 时长 | 1180.955–1204.690 ms | 246.236–254.037 ms |
| `dep_gen.stop/reconcile/replay` | 存在 | 不存在 |

关闭 deps 后，跨卡偏差减少 **15.394249 ms（72.4%）**；原先的“几十 ms”问题消失，
但仍有约 5.9 ms 的残余差异，需归因到 d0 计算/主机调度等非 dep-gen 因素。

本次无 dep-gen 产物：
`build_output/_jit_l3_decode_fwd_20260729_005532`。

## 如何用于计时

对 prepared/resident 运行路径，在构造每次 dispatch 的 `CallConfig` 时传入
`co_enable_swimlane_dep_gen=False`。这仍保留 L2 task timing，但不生成 `deps.json`；
因此转换后的泳道图没有依赖箭头，也不能把任务 token 映射成 kernel 名称。

若需要依赖图，单独运行一次 dep-gen 采集；若要比较真实性能，使用关闭 dep-gen 的 L2
计时结果。
