# 电话信道仿真（P1-4）

> 退化脚本：`scripts/degrade_audio.py`（--preset phone8k/mp3_16k/amr/noise）。
> **链路已验证**（2026-09-02）：在 Python 3.13 下对合成 16k 音频实跑 phone8k / noise（纯 numpy μ-law）与 mp3_16k（ffmpeg），均成功产出退化 wav + manifest，证明管线可用。
> 数字待数据：对中文测试集与 ASVspoof 英文 dev 各生成退化版后重测 EER，填下表。

## 退化档位
| preset | 含义 | 实现 | 依赖 |
|---|---|---|---|
| phone8k | 16k→8k→16k + μ-law 压扩 + 轻度削顶 | 纯 numpy 重采样 + μ-law | 无（仅 numpy） |
| mp3_16k | ffmpeg 转 mp3 32kbps 再转回（网络电话编解码） | ffmpeg 中转 | ffmpeg |
| amr | ffmpeg 转 amr-nb 12.2kbps（移动语音信道） | ffmpeg 中转 | ffmpeg |
| noise | phone8k + 加性粉噪 SNR 15dB（免提外放） | phone8k + 加性粉噪 | 无（仅 numpy） |

> 注：`_ulaw` 原用 `audioop`，已在 Python 3.13（audioop 已移除）改为纯 numpy 实现，CI 3.13 矩阵可正常跑。

## 干净 vs 四档退化对比
| 测试域 | 干净 | phone8k | mp3_16k | amr | noise |
|---|---|---|---|---|---|
| 中文集 EER | TODO | TODO | TODO | TODO | TODO |
| 英文 dev EER | 3.49% | TODO | TODO | TODO | TODO |

## 填充命令（真实数据到位后）
```bash
# 1) 对 ASVspoof dev 与中文集各跑四档退化
python scripts/degrade_audio.py <英文dev目录> --preset phone8k --out data/degraded/phone8k
python scripts/degrade_audio.py <中文集目录>  --preset phone8k --out data/degraded/phone8k
# （mp3_16k / amr / noise 同理）
# 2) 退化集重测 EER（复用 eval_aasist_eer / attack_breakdown_eval）
python scripts/eval_aasist_eer.py --split dev --limit 2000
# 3) 把结果填回上表
```

## 结论（待填）
明确回答"电话信道下系统还能不能用、掉多少"——评委对通话场景的第一疑问。
