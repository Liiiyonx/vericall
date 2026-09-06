# 红蓝第 2 轮 · 三通道联合复测（1.1④）

> 生成：2026-09-06 19:49 · `evaluation/round2_joint3ch.py`
> 输入：SET-B 中声学未拦(spoof<0.5) 89 条，抽样 10 条
> 通道：声学 XLS-R wide（第2轮同款）+ 声纹 CAMPPlus（家人库）+ 语义 SenseVoice→deepseek-r1:8b

## 结果

| 指标 | 值 |
|---|---|
| 端到端拦截(block) | 10 |
| 警惕(caution) | 0 |
| 放行(allow) | 0 |
| 拦截/警惕率 | 100% |

### 兜底通道分布（声学未拦但被拦下的样本，谁 score≥0.5）

| 通道 | 兜底次数 |
|---|---|
| voiceprint | 10 |
| semantic | 4 |

## 逐条裁决

| # | 文件 | 信道 | 声学spoof | 最终 | 融合分 | 置信 | 耗时s |
|---|---|---|---|---|---|---|---|
| 1 | `seg_edgetts_sichuan_zh-CN-YunxiNeural_SC-000409_01.amr.wav` | amr | 0.331 | **block** | 1.0 | 0.85 | 22.8 |
| 2 | `seg_edgetts_mandarin_zh-CN-YunyangNeural_SC-000159_01.amr.wav` | amr | 0.185 | **block** | 1.0 | 0.85 | 26.9 |
| 3 | `seg_edgetts_sichuan_zh-CN-YunxiNeural_SC-000576_00.amr.wav` | amr | 0.42 | **block** | 1.0 | 0.85 | 14.7 |
| 4 | `seg_edgetts_mandarin_zh-CN-YunyangNeural_SC-000036_03.phone8k.wav` | phone8k | 0.339 | **block** | 1.0 | 0.85 | 19.2 |
| 5 | `seg_edgetts_mandarin_zh-CN-YunxiNeural_SC-000511_00.phone8k.wav` | phone8k | 0.353 | **block** | 1.0 | 0.85 | 38.5 |
| 6 | `edgetts_mandarin_zh-CN-YunyangNeural_SC-000085.amr.wav` | amr | 0.312 | **block** | 1.0 | 0.85 | 11.7 |
| 7 | `seg_edgetts_sichuan_zh-CN-YunxiNeural_SC-000429_01.phone8k.wav` | phone8k | 0.412 | **block** | 1.0 | 0.85 | 11.2 |
| 8 | `seg_edgetts_mandarin_zh-CN-YunxiNeural_SC-000327_00.phone8k.wav` | phone8k | 0.213 | **block** | 1.0 | 0.85 | 14.8 |
| 9 | `seg_edgetts_mandarin_zh-CN-YunyangNeural_SC-000036_03.amr.wav` | amr | 0.444 | **block** | 1.0 | 0.85 | 19.9 |
| 10 | `seg_edgetts_mandarin_zh-CN-YunxiNeural_SC-000480_00.mp3_16k.mp3` | mp3_16k | 0.355 | **block** | 1.0 | 0.85 | 14.6 |

### 裁决依据摘录

- seg_edgetts_sichuan_zh-CN-YunxiNeural_SC-000409_01.amr.wav → 纵深防御拦截：voiceprint 通道高置信命中（mismatch，最似「8342」相似度=0.27）
- seg_edgetts_mandarin_zh-CN-YunyangNeural_SC-000159_01.amr.wav → 纵深防御拦截：voiceprint 通道高置信命中（mismatch，最似「5072」相似度=0.27）
- seg_edgetts_sichuan_zh-CN-YunxiNeural_SC-000576_00.amr.wav → 纵深防御拦截：voiceprint 通道高置信命中（mismatch，最似「8342」相似度=0.35）
- seg_edgetts_mandarin_zh-CN-YunyangNeural_SC-000036_03.phone8k.wav → 纵深防御拦截：voiceprint 通道高置信命中（mismatch，最似「5072」相似度=0.22）
- seg_edgetts_mandarin_zh-CN-YunxiNeural_SC-000511_00.phone8k.wav → 纵深防御拦截：voiceprint 通道高置信命中（mismatch，最似「8342」相似度=0.30）
- edgetts_mandarin_zh-CN-YunyangNeural_SC-000085.amr.wav → 纵深防御拦截：voiceprint 通道高置信命中（mismatch，最似「5072」相似度=0.21）
- seg_edgetts_sichuan_zh-CN-YunxiNeural_SC-000429_01.phone8k.wav → 纵深防御拦截：voiceprint 通道高置信命中（mismatch，最似「3443」相似度=0.26）
- seg_edgetts_mandarin_zh-CN-YunxiNeural_SC-000327_00.phone8k.wav → 纵深防御拦截：voiceprint 通道高置信命中（mismatch，最似「8342」相似度=0.27）
- seg_edgetts_mandarin_zh-CN-YunyangNeural_SC-000036_03.amr.wav → 纵深防御拦截：voiceprint 通道高置信命中（mismatch，最似「5072」相似度=0.24）
- seg_edgetts_mandarin_zh-CN-YunxiNeural_SC-000480_00.mp3_16k.mp3 → 纵深防御拦截：voiceprint 通道高置信命中（mismatch，最似「8342」相似度=0.09）

## 结论

声学单通道未拦的 10 条重编码样本走三通道联合后，最终拦截/警惕 10 条（100%），放行 0 条；兜底通道分布：{'voiceprint': 10, 'semantic': 4}。→ 纵深防御实证：语义(话术风险)/声纹(非家人) 对声学漏检样本有效兜底，单通道击穿 ≠ 系统击穿。

## 关联
- `redblue_round2.md`（SET-B 声学击穿 23.5% 的动机）
- `redblue_round2_attack_surface.md`（建议 3：声学+语义联合评测）
