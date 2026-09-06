# 中文域基线（P1-3）· 状态更新 2026-09-06

> 本文原为"待数据"规划；**2026-09-06 数据全部到位、决策树已走通**，更新如下。

## 原决策树（保留，已执行）

- 英文模型直测中文 EER 10–25%（典型跨域劣化）→ 启动中文域微调；
- 结论已落地：见下文"实际结果"与 `redblue_evolution.md` 总叙事。

## 数据到位（2026-09-06 实测）

| 数据集 | 状态 | 备注 |
|---|---|---|
| FMFCC-A | 17,636 伪 wav（A07-A13 系）本地 | label 0=伪/1=真（⚠️ 与 ASVspoof 相反，09-06 纠正） |
| aishell1 | **100 人全量 34,715 条真**（hf-mirror 按说话人） | CFAD 真同源 |
| CFAD | 2,000 + 4 退化布局 | 评测集（隔离纪律不入训练） |
| 红队 | 母本 3,117 + 切段 2,121 | 禁入训练，评测难例 |

## 实际结果（决策树已走通 → 第 1 轮）

1. **跨域劣化确认**：英文 AASIST 对中文 CFAD EER 44%、红队击穿 95.5% → 中文域适配必要；
2. **适配执行**：aishell 真 + FMFCC 伪训练中文域 LR（XLS-R 特征）→ v4 打分器；
3. **修复验证**：同域说话人外 EER **0.00%**；红队击穿 **0.3%**（基线 95-99%）；FAR 100% 判真；
4. **评测域澄清（重要）**：CFAD（声码器伪）与 FMFCC/红队（TTS 伪）分布错位——
   中文 TTS 检测正确评测在红队域，CFAD 仅作跨域参考（详见 `cn_model_cross_eval.md`）。

## 协议转换（保留，兼容）

`scripts/convert_fmfcc_protocol.py`：FMFCC 标签→ASVspoof 协议（含 --selftest）。
⚠️ 09-06 核实：本地 FMFCC 文件为数字命名（2000xxxx.wav），协议 key 含 .wav 后缀勿去。

## 入口导航

核心叙事：`evaluation/redblue_evolution.md`；复现：`redblue_round1.py`、`exp_cn_same_domain.py`；
正式打分器：`data/redteam/factory/cn_lr_scorer_full.pkl`（v4）。
