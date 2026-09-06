# 融合模型中文域零样本复验（CFAD）

> 生成：2026-09-06T10:41:31 · 脚本 `scripts/eval_fusion_crossdomain.py`
> CFAD 2000 条 · 布局 `asvspoof_layout` · 融合器在英文 ASVspoof train 上拟合，零样本应用

| 方案 | CFAD EER |
|---|---|
| AASIST 单模型 | 44.20% |
| XLS-R 冻结 + LR | 15.50% |
| 等权平均融合 | 19.00% |
| LR stacking 融合 | 18.30% |

## 结论

中文域（CFAD 零样本）最优 **XLS-R 冻结 + LR EER 15.50%**，⚠️ 未达到计划书 §一「中文域 EER<5%」验收线。融合相对单模型的增益见上表。
