# 话术分类评测集切分（A1）——语料 v0.2 定稿后冻结

> 日期：2026-09-07 · 脚本：`scripts/scam_corpus/make_eval_split.py`（--seed 2026，--ratio 0.2，可复现）

## 输入（定稿后）
- scam：`corpus_v0.1.jsonl` 11,934 条（8 类，1,345–1,572/类，2026-09-07 定稿，删 19/标 fix 15）
- benign：`benign_corpus.jsonl` 2,404 条（6 类，397–408/类）

## 切分（按 category 分层 80/20）
| 集 | train | eval | 隔离 |
|---|---|---|---|
| scam | 9,547 | 2,387（20.0%，8 类各恰 20%） | ✅ train∩eval=∅ |
| benign | 1,923 | 481（20.0%，6 类各恰 20%） | ✅ |

## 冻结方式
只冻结 **id 清单**（`data/scam_corpus/eval_split/*_train_ids.json / *_eval_ids.json`），正文按 id 从语料现取，避免双份数据漂移；`split_manifest.json` 记录 seed/比例/各类分布，可复现重切。

## 用途
线 A 语义模型（A4 多分类 F1，8 诈骗类 + benign）：train 供微调/快筛校准，eval 冻结做终评。
