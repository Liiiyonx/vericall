# SSL 前端选型报告（通道①升级实验）

> 生成：2026-09-04T13:27:31 · 脚本 `scripts/eval_ssl_frontend.py`
> 对应：极致化计划书 §一「模型矩阵」增量 1

## 实验矩阵

| 编号 | 方案 | 特征 | 训练成本 | 备注 |
|---|---|---|---|---|
| A0 | AASIST（现有基线） | 原始波形 | 已训练 best.pth | 证据链已存在，直接引用 |
| B1 | wav2vec2-XLS-R-300M + LR 头 | SSL 冻结特征 | train 子集 2h 训练 | 选型主力 |
| B2 | WavLM-Base+ + LR 头 | SSL 冻结特征 | 同上 | 备选，域偏置更小 |
| B3 | XLS-R + AASIST 后端（完整训练） | SSL+AASIST | 多卡 1-2 天 | B1 胜出后立项 |

## 结果

| 方案 | dev EER | FAR@EER | 样本数 | 备注 |
|---|---|---|---|---|
| A0 AASIST（现有基线） | 3.49% | 0.00% | 0 | 引用 LA_AASIST_5060_ep24_bs16/model_quality.json（eval 集核验值，不重复推理） |
| B1 wav2vec2-xls-r-300m + LR | 10.51% | 10.64% | 500 | train 1000 条, device=cuda |

## 选型结论

B1（10.51%）相对 A0（3.49%）提升不足 30%，维持 AASIST 主力线，SSL 转离线深评候选。
