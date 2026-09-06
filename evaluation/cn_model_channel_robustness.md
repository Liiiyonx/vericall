# 中文域模型（v4/wide）× 信道退化鲁棒性

> 生成：2026-09-06 · 评测 CFAD 5 布局（跨域参考，说明评测域匹配原则下数字含义）
> 打分器：v4（aishell+FMFCC）vs wide（+CFAD伪）——见 `cn_model_cross_eval.md`/`exp_wide_coverage.md`

## 结果（CFAD EER %，排除训练伪锚点评测口径）

| 布局 | v4 | wide | wide 退化 Δvs原始 |
|---|---|---|---|
| 原始 | 47.50 | 29.20 | — |
| amr | 41.00 | **26.90** | -2.3pp |
| mp3_16k | 43.10 | **27.00** | -2.2pp |
| noise | 44.60 | **36.80** | +7.6pp（最劣） |
| phone8k | 46.90 | **30.80** | +1.6pp |

## 结论

1. **wide 全面优于 v4**：含 CFAD 声码器伪训练 → 所有布局 EER 显著更低（-10~-18pp），
   覆盖优势在退化信道后保持（这是 CFAD 域对声码器攻击的本征能力）；
2. **wide 退化鲁棒**：amr/mp3/phone8k 退化损失 ≤2.3pp（且两版略优于原始，可能是退化
   抹平部分合成伪迹差异），noise 最劣 +7.6pp → 噪声仍是中文域最劣信道（与英文模型结论一致）；
3. **口径提醒**：CFAD 是跨域参考（中文域模型训练于 FMFCC TTS 伪），CFAD 上绝对 EER 不代表
   红队/TTS 场景真实水平（红队域：v4 检出 97.9%/wide ~96%）；产品选 wide 可同时覆盖
   TTS/克隆 + 声码器类（含退化电话信道）双场景。

## 关联
- `fusion_crossdomain_matrix.md`：英文模型跨域矩阵（补中文域行参考本节）
- `exp_wide_coverage.md`：wide 严谨 3 折验证
- 复现：特征缓存 `.tmp_ssl/crossdomain/xlsr_cfad_deg_*`
