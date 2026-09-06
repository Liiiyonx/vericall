# 语料抽检三轮 LLM 复核审计（2026-09-06）

> 方法学记录：首轮预筛 → 假阴猎手（过严）→ 校准终审（可靠）。三轮相互印证，
> 最终以校准终审为准。产出均落盘可复现。

## 一、三轮设置与结果

| 轮次 | 脚本 | Prompt 要点 | 覆盖 | 判定 |
|---|---|---|---|---|
| ① 首轮预筛 | `pre_review.py` | 宽松，强调"单段独立是设计使然" | 全部 433 scam + 244 benign | pass 403 + fix 7 + drop 23；benign 244 全 pass |
| ② 假阴猎手 | `pre_review_pass2.py` | **缺单段独立原则**，专挑毛病 | pass 区 65（含钱词）→ 全量 403 | drop 136 / flag 25（**误杀率高**） |
| ③ 校准终审 | `pre_review_pass2_calib.py` | 单段独立原则 + 猎手怀疑 + 三值 | 全部 403 pass | **ok 402 / flag 1 / drop 0** |

## 二、关键结论

1. **首轮 pass 区质量合格**：403 条经校准终审 99.75% ok（402），唯一 flag 为 SC-004891
   （类别混杂：福利诱导+转账+保密要求混杂，阶段标注不明，**需人工定夺**）。
2. **猎手 prompt 系统性误杀**：136 drop 中 135 条（99%）被校准判 ok——原因是 prompt 未声明
   "话术库按 8 类 × 4 阶段构建、单段独立是设计使然"，把 pressure 段威胁施压、ask 段索要
   等**合理单段话术**误判为"缺诈骗逻辑/穿帮"。
3. **教训（方法论）**：LLM 复核的 verdict 分布对 prompt 措辞极度敏感；作"质检"用途的 prompt
   必须显式注入数据集的生成约束（本案例：网格化单段话术），否则会得到虚假的高 drop 率。
   多轮复核必须交叉验证，不能单独采信某一轮激进 prompt 的结论。

## 三、人工复核清单（最终版）

`evaluation/review_candidates_worksheet.csv`（**31 行**）：
- 首轮候选 30 条（drop 23 / fix 7，来自 review_prefilter_suspect.csv，LLM 初判）；
- 校准终审 flag 1 条（SC-004891）；
- 每行含全文 + LLM 建议 + `human_verdict`/`human_note` 待填列。

## 四、审计文件

| 文件 | 内容 |
|---|---|
| `evaluation/review_prefilter.csv` | ①全量 677 行（scam 433 + benign 244） |
| `evaluation/review_prefilter_suspect.csv` | ①候选 30 行 |
| `evaluation/review_pass2_flagged.csv` | ②猎手标记（**存疑，勿直接采信**） |
| `evaluation/review_pass2_calib.csv` | ③校准全量 403 行（权威） |
| `evaluation/review_pass2_calib_flagged.csv` | ③候选（flag 1） |
| `evaluation/review_candidates_worksheet.csv` | **人工复核工作表 31 行（唯一入口）** |

## 五、建议

人工复核仅需处理 31 行（远小于整轮 433）。若确认后 drop+fix-需删 ≤30% 则语料 v0.1 可定稿。
下一步可直接在 ③ 的 ok 402 基础上继续（pass 区的 ok 是语料主体）。
