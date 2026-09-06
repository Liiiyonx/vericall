# 融合模型中文域零样本复验（CFAD）

> 生成：2026-09-06T10:05:52 · 脚本 `scripts/eval_fusion_crossdomain.py`
> CFAD 2000 条 · 布局 `asvspoof_layout_deg_noise` · 融合器在英文 ASVspoof train 上拟合，零样本应用

| 方案 | CFAD EER |
|---|---|
| AASIST 单模型 | 48.90% |
| XLS-R 冻结 + LR | 19.30% |
| 等权平均融合 | 35.40% |
| LR stacking 融合 | 33.90% |

## 结论

中文域（CFAD 零样本）最优 **XLS-R 冻结 + LR EER 19.30%**，⚠️ 未达到计划书 §一「中文域 EER<5%」验收线。融合相对单模型的增益见上表。
