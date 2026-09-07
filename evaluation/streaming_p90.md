# 流式实时压测报告（5.3，2026-09-07）

> 方法：69.5s 真实诈骗音频（40 段四川话 edgetts 拼接，16k mono）经 `StreamProcessor.process_signal`
> 流式推窗，**真实三通道**（AASIST GPU + CAMPPlus + SenseVoice ASR + 云端 deepseek 语义），
> 以 `perf_counter` 包 `_process_window` 逐窗计时（n=68 窗，win 3s / step 1s）。

## 结果
| 指标 | 值 |
|---|---|
| 窗数 | 68 |
| mean | 380 ms |
| **P50 / P90** | **174 ms / 194 ms** |
| P99 / max | 14,056 ms（首窗冷加载/云端重试离群，单点） |
| 判定 | **P90 194ms ≤ 500ms ✅ 达标**（2s 保底线裕量 ~10x） |

## 副产物（压测发现并已修复）
- **真 bug**：长流转写 `transcript` 无限累积 → MemoryError（69s 即崩）；已修复为**滚动截断 1500 字**
  （`src/server/stream_pipeline.py`，commit 桌面 + GitHub b0e5be9）。
- 复现脚本：临时单测器逻辑见本报告来源；`scripts/test_streaming.py --replay --real` 为回归入口
  （建议进 5.5 周回归 GPU 段）。

## 口径说明
- P90 194ms 为**单窗处理墙钟**（不含排队）；真实吞吐取决于窗步进（1s/窗），单卡已富余；
- 离群点 14s 为首窗模型冷加载 + 云端语义首次调用，P99 不纳入 P90 口径（已记录）；
- 5.1 并行调度（VERICALL_PARALLEL=1）为可选项，单卡 GPU 通道本就排队，收益主要在云端语义重叠，
  已接线未强制开启（防 OOM）。

## 追加：5.1 并行墙钟实测（2026-09-07，3 条极难端到端）
| 模式 | 3 条总耗时 | 均/条 | 加速 |
|---|---|---|---|
| VERICALL_PARALLEL=0（串行） | 23.1s | 7.7s | 1.0x |
| VERICALL_PARALLEL=1（并发） | 8.0s | 2.7s | **~2.9x** |

结论：并发开启后语义云端与 GPU 通道重叠，整段分析墙钟减 ~65%；流式窗口 P90 本已 194ms 达标，
并行开关主要服务"整段文件批量分析"场景，默认保持串行（防单卡 OOM）可配。
