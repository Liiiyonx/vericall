# 红队难度表体系说明（2026-09-06）

> 用途：四份难度表并存，本文说明各自口径与适用场景，避免误用/混淆。

## 难度语义

难度分 = `difficulty_spoof_prob`：**检测器判该音频为真(bonafide)的概率**。
- 分越低 → 检测器越判"真" → 对检测器越隐蔽、越难检 → 难度越高（作增量训练难例越有价值）；
- 阈值：<0.3 隐蔽(难检) / 0.3~0.5 边界 / ≥0.5 可检。

⚠️ **关键：同一条红队音频，在不同检测器视角下难度相反**（红队对弱检测器全隐蔽、对强中文域检测器全可检）——
这是"检测器能力"的度量而非红队固有属性。报难度必须带打分器口径。

## 四表对照

| 文件 | 覆盖 | 打分器 | 分布 | 语义 |
|---|---|---|---|---|
| `redteam_difficulty.csv` | 母本 3117 | cfad 拟合 LR（AUC 0.950，弱口径） | 隐蔽 3090/边界 26/可检 1 | 早期基线："红队对 CFAD 域检测器极隐蔽" |
| `redteam_difficulty_full.csv` | 母本 3117 | **中文域 v4**（aishell+FMFCC） | 可检 3063/边界 52/隐蔽 2 | 红蓝第 1 轮后："红队已被中文域检测器检出" |
| `redteam_seg_difficulty.csv` | 切段 2121 | cfad 拟合 LR | 隐蔽 1488/边界 581/可检 52 | 切段早期难度（母本 99.1%→切段 70% 分布更宽） |
| `redteam_seg_difficulty_full.csv` | 切段 2121 | **中文域 v4** | 可检 2025/边界 85/隐蔽 11 | 切段在强检测器下同样可检 |

## 推荐用法

1. **论文/答辩叙事**：
   - 用 cfad 表（+ redteam_breakthrough）讲"红队曾击穿最强英文/CFAD 检测器 95-99%"——红队是顶级难例；
   - 用 full 表（+ redblue_round1）讲"中文域适配后红队击穿归零"——模型能力跃迁；
2. **增量训练选样**：以 full 表为准——但红队禁入训练管线（合规约束），
   选样仅用于**评测**分层报告；训练难例取自 FMFCC 伪（17.6k 特征已备）；
3. **质量分层报告**：切段表（full）显示 2121 段 95.5% 可检——细粒度评测分层用。

## 复现

```bash
# cfad 口径（默认）
python scripts/redteam_factory/score_redteam_xlsr.py              # → difficulty.csv
python scripts/redteam_factory/score_redteam_xlsr.py --seg-only  # → seg_difficulty.csv
# full 口径（中文域 v4）
python scripts/redteam_factory/score_redteam_xlsr.py --scorer full
python scripts/redteam_factory/score_redteam_xlsr.py --seg-only --scorer full
# 快速重打（特征已缓存）用 .tmp_ssl 内联脚本，见历史 commit d34db94
```

## 追加（09-06 14:10）：wide 版难度表（生产默认口径）

| 文件 | 覆盖 | 打分器 | 分布 | 语义 |
|---|---|---|---|---|
| `redteam_difficulty_wide.csv` | 母本 3117 | **wide**（aishell+FMFCC+CFAD伪） | 可检 2990/边界 121/隐蔽 6 | 生产默认口径：TTS+声码器双域 |
| `redteam_seg_difficulty_wide.csv` | 切段 2121 | **wide** | 可检 1989/边界 104/隐蔽 28 | 同上 |

**最难样本定位**：wide 版中仍"隐蔽"的 6（母本）+ 28（切段）= 34 条是 wide 也难检的
**极难样本**——红蓝第 2 轮攻击升级的候选靶标（红队禁训练，仅作评测分层）。

## 追加（09-07）：极难集口径升级 + 红蓝第 2 轮回填

- **极难集口径升级（任务 1.2）**：34 条极难（6 母本 + 28 切段）中，28 条切段已过 `evaluation/round2_seg_qa.py` 机器预筛——时长 ≥1s、有效语音 ≥0.7s 且占比 ≥35%、RMS/峰值 ≥-42/-36 dBFS、削波 ≤5e-04、首尾静音 ≤0.6/0.9s，**28/28 pass_struct、0 suspect**。口径由"wide<0.3"升级为 **"wide<0.3 且音频结构有效（round2_seg_qa）"**，消除"分数低但切坏/静音/过短"的假难例；
- **红蓝第 2 轮回填（defender = 同款 wide 不变）**：34 条极难 → ① 男声强化 SET-A（196 条，wide 分 0.659–0.936 音色域）击穿 1.53%；② 退化矩阵 SET-B（136 条 = 34×{phone8k,mp3_16k,amr,noise}）击穿 23.53%（分信道 amr 最高 29.41%——**真实电话信道是主要盲区**，为第 3 轮退化信道训练弹药）；③ 三通道联合 10/10 拦截（语义兜底实证）。明细见 `evaluation/redblue_round2*.md/.json`；
- **wide 表在红队评测体系中的位置**：母本 3117 / 切段 2121 的 wide 难度分布（可检 2990+1989 / 边界 121+104 / 隐蔽 6+28）即"生产口径红蓝评测"的分层依据，与 cfad/v4 表并列不复用混用。
