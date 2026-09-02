# 评测与标定总览（P1）

本目录承载任务书 P1 的"中文首验 + 信道仿真 + 阈值标定"三项硬指标，全部为
**评委必问**内容。各脚本不依赖 GPU/Ollama 即可导入，缺数据时生成骨架报告。

| 文档 / 脚本 | 对应任务 | 说明 |
|---|---|---|
| `chinese_baseline.md` | P1-3 | 英文模型直测中文的 EER + 决策树 |
| `channel_degradation.md` | P1-4 | 干净 vs 四档信道退化的 EER 对比（链路已验证） |
| `fusion_calibration.md` | P1-5 | 非对称代价网格搜索最优阈值 + 消融表（由脚本生成） |
| `scenario_manifest.csv` | P1-5 | 100 条带期望标签的场景集（由 `scripts/build_scenarios.py` 生成） |
| `scenario_scores.csv` | P1-5 | 每条场景的三通道分数矩阵（全链路推理填充后供标定） |
| `redteam_adversarial.md` / `.json` | P2-5 | 红队对抗实验报告（攻击面框架，由 `make_redteam_adversarial.py` 生成） |
| `cross_domain_matrix.md` | P2-2 | 跨域评测矩阵（即插即用，由 `cross_domain_eval.py` 生成） |
| `attack_breakdown_eval.py` | 通道① | 跨攻击域（已知 A01–A06 vs 未知 A07–A19）细分 EER |
| `metrics.py` | 全部 | 零依赖 EER/FAR/FRR 指标 |

## 跑通顺序
1. `python scripts/build_scenarios.py` → 生成 `scenario_manifest.csv`
2. 全链路推理（需模型）→ 填 `scenario_scores.csv`
3. `python scripts/fusion_calibration.py` → 出 `fusion_calibration.md`
4. `python scripts/degrade_audio.py <音频> --preset phone8k` → 退化集
5. 退化集重测 EER → 填 `channel_degradation.md`
6. 中文集（FMFCC-A/CFAD）跑 EER → 填 `chinese_baseline.md`

## 中文数据获取
见 `docs/数据集获取指南.md` P1 节（FMFCC-A GitHub 直取；CFAD 第二集）。
