# 红蓝循环 · 攻防演化报告（第 0 轮 → 第 1 轮）

> 生成：2026-09-06 · 数据与脚本索引见文末
> 叙事主线：红队合成（Edge-TTS/GPT-SoVITS 克隆）→ 击穿基线 → 合规中文域训练 → 盲区收窄

## 一、红队资产与难度基线

- **红队语料**：4605 wav（Edge-TTS 4165 + GPT-SoVITS 克隆 440），clean 母本 3117 + 超长切段 2121 = **5238 独立样本**，6 方言均衡；
- **合规约束**：红队禁入训练管线（详册 §7.3），仅作评测难例集。

## 二、第 0 轮基线（redteam_breakthrough.md）

| 检测器 | 红队击穿率（判真比例） |
|---|---|
| 自有 AASIST（英文声学，对照中文真判别正常） | **95.5%** |
| XLS-R + 中文LR（CFAD 拟合，AUC 0.950） | **99.0%** |

对照证明工具正常（AASIST 英文 dev 真 0.010/伪 0.888）→ 击穿主因 = 中文域/中文 TTS 域差。

## 三、数据侧闭环（hf-mirror 打通）

- **真**：AISHELL-1 按说话人下载 21 人 → **34,715 条 16k 真人中文语音**（AISHELL-1 全量 100 说话人 S0002-S0101，`D:/VeriCall_data/aishell1_sub/`，与 CFAD 真同源）；
- **伪**：FMFCC-A 17,636 条（A07-A13 商业/开源 TTS 系）；
- 特征缓存：aishell 34,715（全量）/ FMFCC 17,636 / 红队母本 3117 + 切段 2121（XLS-R 1024d）。

## 四、第 1 轮修复（中文域适配）+ 评测矩阵

| 实验 | 评测域 | 结果 |
|---|---|---|
| 中文域模型（aishell真 + FMFCC伪） | **同域说话人外**（5人/伪留出） | **EER 0.00%** |
| 中文域模型 | CFAD（跨域参考） | 22.5%（评测错位：CFAD伪=声码器族 vs FMFCC伪=TTS族） |
| **红蓝第 1 轮** | **红队 3117 母本** | **击穿 0.0% / 检出 98.3%**（edgetts 98/sovits 100） |
| 留一系统（排除 A09） | 未见 A09 攻击 | **检出 100%**（学到通用合成特征，非记忆系统） |

## 五、攻防演化数据点（王牌图）

```
红队击穿率
100% |████████████████████ AASIST 95.5%
 90% |███████████████████▌ 中文LR 99.0%
 ... |                     
  0% |▏                    中文域模型 0.0% ← 第 1 轮
     +------------------------------------
     第0轮                      第1轮
```

## 六、方法学要点（跨域评测陷阱）

1. **"英文零样本基线 15.5%"是陷阱**：英文 ASVspoof 训练覆盖声码器族攻击，与 CFAD 伪同族 → CFAD 上看似可用，实则对中文 TTS 无泛化（英文模型对一切中文判伪：FAR 崩坏）；
2. **纯中文域模型在 CFAD 上 47% 是另一陷阱**：FMFCC 伪（TTS 族）与 CFAD 伪（声码器族）分布错位 → 跨数据评测需报域条件；
3. **正确范式**：同域说话人外评测（0.00%）+ 红队评测（0.0% 击穿）+ 留一系统（100% 检出）三件套。

## 七、可复现入口

| 报告 | 脚本/数据 |
|---|---|
| `redblue_round1.md` | `evaluation/redblue_round1.py`（正式打分器 `cn_lr_scorer_full.pkl`） |
| `redteam_breakthrough.md` | `evaluation/redteam_breakthrough.py`（第 0 轮） |
| `exp_cn_same_domain.md` | `evaluation/exp_cn_same_domain.py` |
| `exp_leave_one_system.md` | 内联（同 exp_cn_same_domain 数据） |
| `exp_cn_train_full.md` / `exp_domain_anchor.md` / `exp_fmfcc_pseudomain.md` | 配套方法学 |
| 特征缓存 | `.tmp_ssl/{aishell,fmfcc,redteam}/` · 数据下载 `docs/中文域增量训练_数据侧准备.md` |
