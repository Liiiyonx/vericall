# 电话信道增强训练实验（exp_channel_robust，红蓝第 4 演化点前置）

> 生成：2026-09-08 01:51 · `evaluation/exp_channel_robust.py`
> 设计：退化增强训练：clean(aishell真+FMFCC伪+CFAD伪1000) + 退化副本(aishell真500/说话人K=5 + FMFCC伪2000, ×amr/phone8k/mp3_16k)
> 对照：m_base（复现 wide，仅 clean） vs m_enh（clean + 退化增强）；defender 均 XLS-R + LR。

## SET-D 退化信道击穿率（<0.3，越低越好）

| 信道 | base | enh | 变化 | 第3轮参照(wide) |
|---|---|---|---|---|
| amr | 22.73% | 9.09% | -13.64 | 22.73% |
| phone8k | 4.55% | 0.0% | -4.55 | 4.55% |
| mp3_16k | 9.09% | 9.09% | 0.00（未变） | 9.09% |
| noise | 0.0% | 0.0% | — | 0.0% |

## 能力保持检查（enh 不得显著回退）

| 指标 | base | enh | 参照 |
|---|---|---|---|
| SET-C clean 极难击穿率 | 100.0% | 34.78% | base=复现 wide（该清单即 wide<0.3 选出） |
| aishell 真样本判真率(≈1-FAR) | 99.98% | 99.98% | 第3轮 aishell 判真 100% |
| CFAD EER（乐观口径） | 29.2% | 29.2% | wide ~29% |

## CFAD 退化信道 EER（额外参照）

| 信道 | base | enh |
|---|---|---|
| amr | 26.9% | 25.8% |
| phone8k | 30.8% | 30.2% |
| mp3_16k | 27.0% | 26.6% |
| noise | 36.8% | 36.1% |

## 结论

- **SET-D 四信道合计击穿 8/88（9.1%）→ 4/88（4.5%）**：主盲区 amr 22.73% → 9.09%（-13.6pp）、phone8k 4.55% → 0%（清零）；残余漏网 = amr 2/22 + mp3_16k 2/22（特定母本重编码后仍高隐蔽，见 json 明细）；
- **增强泛化回 clean 域**：SET-C（round3 极难 23，base 定义性 100% 击穿）→ enh 仅 8/23（34.8%）——即退化增强让 15/23 个 clean 极难样本也不再击穿（决策面整体右移）；
- **能力零回退**：aishell 真样本判真率 99.98%（base=enh，≈FAR 0.02%）、CFAD clean EER 29.2%（=wide 基线）；域外退化（CFAD-deg 4 信道）EER 全线微降（-0.4~-1.1pp）——说明增强学的是"退化≠伪"的通用不变性而非记忆信道；
- 训练侧成本：+7,500 条退化副本（真 1,500 + 伪 6,000，占 clean 14%），无新数据采集，纯计算增强；
- **第 4 演化点叙事**：第 3 轮 SET-D amr 22.7% 盲区（defender=wide 不变）→ 训练侧退化增强（defender=chenh）收敛：SET-D 合计 4.5%、amr 9.1%；11 月"电话信道增强训练立项"原基于盲区数据成立，本实验直接给出低成本收敛路径（无需新数据/新模型架构）；
- 残余项：mp3_16k 与 amr 各 2 条顽固样本 → 下轮可加权重或对该 4 条母本做针对性重编码增强（增量微调）。

## 复现与产物

- 一键复现：`python -u evaluation/exp_channel_robust.py --stage degrade|feat|train|eval`（种子 2026）
- 模型：`data/redteam/factory/cn_lr_scorer_wide_chbase.pkl`（对照，复现 wide 口径）/ `cn_lr_scorer_wide_chenh.pkl`（增强版）
- 特征缓存：`.tmp_ssl/chrobust/`（manifest/deg_feats.npz，可复用免重提）
- 明细：`evaluation/exp_channel_robust.json`（SET-C/D 逐条分、CFAD EER 全表）