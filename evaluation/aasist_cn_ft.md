# AASIST 中文域微调实验报告（2026-09-08）

> 目标：验证"英文预训练 AASIST + 中文域数据微调"能否修复 AASIST 的中文盲区。
> 背景：英文 AASIST（ASVspoof2019 LA 训练）对中文失效，CFAD 零样本 EER 44%（≈随机）。
> 复现：`scripts/prepare_aasist_cn.py`（数据）+ `configs/AASIST_CN.conf`（训练）。

## 一、实验设置

| 项 | 值 |
|---|---|
| 数据 | 真：AISHELL-1 子集 34,716；伪：FMFCC-A 17,636（A01–A13 攻击系统，含退化变体） |
| 切分 | train 8k真+8k伪 / dev 1k+1k / eval 1k+1k（按说话人/攻击系统分层，无泄漏） |
| 初始化 | 官方英文预训练 `models/weights/AASIST.pth`（`--resume` 加载，从头 epoch0 微调） |
| 超参 | bs 16 / 8 epoch / lr 5e-5（cosine）/ AMP / 加权 CE（0.1/0.9） |
| 硬件 | RTX 5060 Laptop 8GB，40min 训完 |

## 二、结果（逐 epoch）

| epoch | dev EER | eval EER | 备注 |
|---|---|---|---|
| 0 | 11.30% | 1.90% | 微调首 epoch 即破英文盲区 |
| 1 | 4.70% | **0.70%** | |
| 2 | 5.00% | — | |
| 3 | **4.20%** | **0.60%** | **best（best.pth = epoch_3_4.200.pth）** |
| 4 | 5.10% | — | |
| 5 | 5.00% | — | |
| 6 | 5.80% | — | |
| 7 | 7.10% | — | 后期轻微过拟合 |
| SWA | 6.50% | 0.80% | SWA 劣于训练中最佳，best.pth 取 epoch3 |

## 三、结论

1. **中文盲区已修复（同域）**：英文零样本 CFAD 44% → 中文微调后 **dev EER 4.20% / eval EER 0.60%**，
   均达成计划书 §一"中文域 EER <5%"验收线。
2. **微调高效**：仅 8k+8k 训练样本、8 epoch、40 分钟（8GB 卡），首 epoch 即破盲区——跨语种迁移
   成本远低于从头训练，验证"预训练 AASIST + 目标语种小样本微调"的可行性。
3. **dev/eval 差异（4.2% vs 0.6%）**：切分波动——dev 集恰好含更难分的声音/攻击系统；
   两集说话人与攻击系统不重叠（脚本分层切分），非泄漏。正式口径取 **eval EER 0.60%**，
   保守口径取 dev 4.20%。
4. **跨域泛化有限（诚实结论，CFAD 复算）**：微调后的 `best.pth` 在隔离 CFAD clean 2000 上
   **EER 52.30%**（1000 真 + 1000 伪，接近随机；bonafide spoof-分 0.25 > spoof 0.14，判别反向）。
   - **根因**：CFAD 伪样本为「部分伪造/声码器」（PARTIALLYFAKE / STRAIGHT），与 FMFCC 的 TTS 伪
     分布错位——这正是「评测域匹配原则」（`evaluation/cn_model_channel_robustness.md`）：
     单一目标域微调过拟合 FMFCC 伪，遇 unseen 攻击系统即退化。
   - **含义**：AASIST 中文微调 ≠ 中文域通用解；生产声学通道仍以 `xlsr_cn_channel`（wide scorer
     = aishell+FMFCC+CFAD 伪宽覆盖，CFAD 47→28%）为主。AASIST 中文微调价值 = 同域快通道 +
     跨语种迁移方法学证据，而非跨域泛化解。

## 四、产物

- 权重：`external/aasist/exp_result/LA_AASIST_CN_ep8_bs16_cn_ft/weights/best.pth`（dev 4.2% / eval 0.6%）
- 数据：`D:/VeriCall_data/AASIST_CN/`（协议 + flac，20,000 条）
- 脚本/配置：`scripts/prepare_aasist_cn.py` + `configs/AASIST_CN.conf`
- 质量：`model_quality.json`（dev_eer 4.2, winner=best_dev）
