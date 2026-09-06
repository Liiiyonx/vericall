# 红蓝第 2 轮 · 对抗退化矩阵报告

> 生成：2026-09-06 19:21 · `evaluation/round2_degrade_matrix.py`
> 输入：34 条 wide 极难样本（score<0.3）× 4 预设退化 = 136 目标变体

## 结果

- 成功生成 **136** 条（新生成 34 / 复用 102），失败 0，校验异常 0；
- 文件可读性/时长校验：全部可解码且 ≥0.5s（异常 0 条）；amr 档为 8kHz（amr-nb 12.2k 语义），其余 16kHz；

### 分预设

| 预设 | 含义 | 条数 |
|---|---|---|
| amr | AMR-NB 12.2kbps 移动信道 | 34 |
| mp3_16k | MP3 32kbps 编解码 | 34 |
| noise | 8k + 粉噪 SNR15dB(免提外放) | 34 |
| phone8k | 8k 电话信道(μ-law+削顶) | 34 |

### 分方言

| 方言 | 条数 |
|---|---|
| cantonese | 4 |
| mandarin | 92 |
| minnan | 16 |
| sichuan | 24 |

## 后续
- 变体已入 `round2_degrade/manifest.json`，供三通道联合复测（1.1④）与击穿率分信道表（1.1③）使用。
