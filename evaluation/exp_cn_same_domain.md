# 中文域模型 · 同域说话人外评测

> 生成：2026-09-06 12:19 · `evaluation/exp_cn_same_domain.py`
> 训练：aishell真(S0002-5, 1414) + FMFCC伪(12k)；测试：aishell真(S0006-7, 708,**说话人外**) + FMFCC伪(5.6k)

| 模型 | 同域 EER |
|---|---|
| 中文域模型 (aishell+FMFCC) | 0.00% |
| 英文模型 (ASVspoof 对照) | 2.70% |

## 结论

中文域模型在同域(说话人外+同族伪)评测 EER 0.00%，英文模型 2.70%。→ 中文域模型在其目标域上表现优异；此前 CFAD 上 47% 系评测错位(CFAD伪=声码器族 vs FMFCC伪=商业TTS族)。**结论：中文域模型真实水平需同域评测，CFAD/退化矩阵仅作跨域参考；红队(同属TTS域)是中文域模型的自然评测集。**

## 配套实验索引
- `exp_cn_train_full.md`：同模型在 CFAD(跨域)上的表现 → 揭示评测错位；
- `exp_domain_anchor.md`：目标域锚点适配路径有效；
- `exp_fmfcc_pseudomain.md`：伪域扩充(异构)反效果。
