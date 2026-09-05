# 语料抽检 LLM 预筛摘要

- 预筛时间：2026-09-06 00:37（云端 deepseek-chat）
- 样本：677 条（正 scam + 负 benign），可疑候选 30 条，其中 drop 建议 23 条
- 预筛 drop 率 3.4%（人工复核后才定论；>30% 才触发整轮重做）

| 类型 | verdict | 条数 |
|---|---|---|
| scam | pass | 403 |
| scam | fix | 7 |
| scam | drop | 23 |
| benign | pass | 244 |
| benign | fix | 0 |
| benign | drop | 0 |

## 人工复核指引
- 打开 `review_prefilter_suspect.csv`，逐条在抽检原表 verdict 列填 pass/fix/drop；
- 可疑候选之外，建议再随机抽 5~10% pass 条复核防漏；
- 若 drop（含 fix 中需删者）占比 >30%，整轮语料重做。