# 谛听 VeriCall · 参赛证据一键报告

> 聚合生成：2026-09-07 11:36 · `scripts/eval/competition_report.py`
> 原则：本页所有数字均可沿「来源」回链到脚本与产物复现（门禁 #3：复算不出 = 没做）。

## 1. 语料资产（v0.2 定稿 2026-09-07）
- scam **11934** 条（8 类 1345-1572/类均衡；
  4 阶段 2599-3229）+ benign 2,404（6 类）
- 定稿流程：2 轮 LLM 预筛（二审纠偏后真实 drop 率 1.0%）+ fix 改写 15/15 回填
- 评测集冻结：scam train/eval 9,547/2,387、benign 1,923/481（8+6 类各恰 20%，隔离 OK）
- 来源：`make_eval_split.py` / `apply_review_verdicts.py --llm-fill`

## 2. 号码第 0 层（先验闸门）
- 已知特征号命中 10/10、正常号误报 0/10、本地查表 0ms；命中即短路跳过三通道
- 来源：`src/fusion/number_channel.py` + `evaluation/number_channel_eval.py`

## 3. 语义侧（线 A）
- **二分类诈骗/正常 F1 98.7%**（P 99.9% /
  R 97.5%，n=2868）；8 类加权 75.8%
- 校准 **ECE 7.2%**；语义 block 阈值建议 risk≥0.6（P99.9/R97.4）
- 来源：`eval_semantic_f1.py` / `calib_semantic.py`（2,868 行级 risk 落盘 semantic_f1_rows.csv）

## 4. 声纹侧（线 B）
- **EER 0.51%** @0.47180950514078135（200 人合库：AISHELL-1 100 + AISHELL-3 100，
  13068 同人对 / 60000 异人对）
- 工作点 κ 1:1/3:1/1:3 → 0.471/0.443/0.520；代码阈值已替换 0.520/0.443
- 三板斧：双信道注册（互验≥0.52）+ challenge-response 骨架 + TTS 门控（joint3ch 10/10）
- 来源：`voiceprint_det_scan.py` / `voiceprint_calibration.md` / `b4_antiattack_design.md`

## 5. 声学/红蓝（攻防演化 0→2）
- 红蓝：第0轮击穿 95.5/99.0% → 第1轮 0.3% → 第2轮 SET-A 1.53% /
  SET-B 23.53%（amr 信道 29.41%）
- 三通道联合复测 10/10 拦截；wide 极难集口径 = score<0.3 且 round2_seg_qa 结构有效（28/28）
- AASIST dev EER 0.745%（best 0.316%）/ eval 3.49%（`v0.2-repro` 复现）
- 来源：`redblue_round2.py` 等 + 演化图 `redblue_evolution.md`

## 6. 工程/质量
- pytest **88 passed**（2026-09-07）；CI：`.github/workflows/pytest.yml`
- 隔离检查 `check_isolation.py` 通过（红队集禁入训练）；降级模式 VERICALL_OFFLINE=1 保底演示

## 7. 口径锚点（答辩防翻车速查）
- 终端形态：独立终端（座机/一体机）为主 + iPhone 号码预警（iOS 生态无法 App 内实时分析）
- 与官方关系：号码=先验加速（可接官方库做第 0 层）；官方离线单条 AIGC 鉴定 vs 我们通话中实时流式融合——互补
- 对照图：`assets/comparison_official_vs_us.svg`；tag：`v0.8-redteam`（全链可复现）
