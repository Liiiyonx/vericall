# 中文场景首次验证（P1-3）

> 状态：**待数据**。FMFCC-A / CFAD 获取有周期，数据到位后跑 `scripts/eval_aasist_eer.py`
> （或本目录的 `chinese_baseline` 流程）填充下表。

## 方法
1. 数据：FMFCC-A（GitHub 直接获取，1 万真 / 4 万假，含噪声与编解码）；CFAD 作第二集。
2. 协议对齐：`scripts/convert_fmfcc_protocol.py` 已实装——把 FMFCC-A 标签转成 ASVspoof 5 列协议 + ASVspoof 风格目录布局，可经 `VERICALL_ASVSPOOF_LA` 直接喂给 `eval_aasist_eer.py` / `cross_domain_eval.py`。内置 `--selftest` 可在无真实数据下验证协议生成。
3. 小规模首测：子采样 1000 真 + 1000 假，跑 AASIST 直测中文 EER + 分数分布。

## 三个关键数字（后续所有材料的引用源）
| 指标 | 数值 |
|---|---|
| 中文 EER | TODO |
| 英文 eval EER | 3.49% |
| 劣化倍数 | TODO |

## 决策树（按实际结果走）
- <10%：跨语种泛化尚可 → 直接进入电话信道实验（P1-4），中文混训列为 P2 可选项。
- 10–25%：典型跨域劣化 → 启动 FMFCC-A 微调（冻结大部分层，5060 上 5 epoch 内）。
- >25%：崩了 → 微调 + 考虑 SSL 前端升级（wav2vec2-base + AASIST 头）。

## 误判样本抽样
TODO（哪些攻击类型漏了）
