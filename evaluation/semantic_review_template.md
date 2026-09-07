# 语义深评复盘报告模板（A3，2026-09-07）

> 用途：云端深评线（deepseek-chat / SemanticChannel）对**拦截事件**与**抽检样本**的
> 拦截后复盘，形成标准化记录。评审/论文/答辩引用时保证"每个拦截决策可回溯"。
> 与 A4 F1 评测（批量、自动）互补：A4 出数字，A3 出个案解释。

## 一、报告结构（每次复盘一个 md）

```markdown
# 语义复盘 · {日期} {批次标识}

## 1. 批次概览
| 项 | 值 |
|---|---|
| 样本来源 | 线上拦截 / eval_split 抽检 / 红蓝第 2 轮 SET-* |
| 样本数 | N（scam n1 / benign n2 / 未知 n3） |
| LLM 后端 | cloud deepseek-chat（SCAM_LLM_*）|
| 平均延迟 / P90 | x.x s / x.x s |

## 2. 逐条记录（表）
| id | 类别(标) | 阶段 | ASR/文本摘录 | risk | category(判) | 裁决 | 复盘栏(人工) |
|---|---|---|---|---|---|---|---|
| ... | | | | | | block/pass | |

## 3. 错误聚类（重点）
### 3.1 误拦（benign 判 block）
- 模式 / 根因（如 near_boundary 近边界） / 建议
### 3.2 漏拦（scam 判 pass/低风险）
- 模式 / 根因（如铺垫段无索要动作） / 建议
### 3.3 类别错分
- 混淆对（family↔authority↔refund_cs 等）

## 4. 结论与动作
- 阈值是否需动（引用 voiceprint_calibration 同类方法）；提示词是否需补（防回归到 drop 区误杀/漏网）
```

## 二、字段口径
- risk：0-1（LLM 语义风险分）；category：SemanticChannel CATEGORIES 键 + normal；
- 裁决：>0.5 视为 block（与三通道融合一致，最终以 FusionOrchestrator decide 为准）；
- 类别粒度注意：channel 的 impersonation 为**合并类**，复盘点开细分（family/authority/refund_cs）需人工/二次 prompt。

## 三、触发方式
- 线上：拦截事件落盘后由 `evaluation/review_3round_audit.md` 类流程人工抽复盘；
- 批量：`scripts/scam_corpus/` 下 A4 F1 评测输出的 fail 子集直接进本模板。
