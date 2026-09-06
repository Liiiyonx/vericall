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
| `ssl_selection.md` / `.json` | 通道①升级 | SSL 前端选型报告（AASIST vs XLS-R/WavLM，由 `scripts/eval_ssl_frontend.py` 生成，缺依赖出骨架） |
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

---

## 红蓝循环 / 中文域检测（2026-09-06 重大里程碑，P2 验收线）

> 主线：红队合成（Edge-TTS/GPT-SoVITS 克隆）→ 第 0 轮击穿基线 → 合规中文域训练 →
> 第 1 轮击穿归零 → 产品级声学通道。**核心叙事 + 王牌图见 `redblue_evolution.md`。**

### 数据资产（全部合规：aishell/FMFCC 训练，CFAD/红队仅评测）

| 数据集 | 真/伪 | 规模 | 特征缓存 |
|---|---|---|---|
| AISHELL-1（hf-mirror 2026-09-06） | 真 | **100 人 34,715 条**（`D:/VeriCall_data/aishell1_sub/`） | `.tmp_ssl/aishell/` |
| FMFCC-A | 伪 | 17,636 条（A07-A13 TTS/VC 系） | `.tmp_ssl/fmfcc/` |
| 红队（自产，禁入训练） | 伪 | 母本 3117 + 切段 2121 = 5,238 | `.tmp_ssl/redteam/` |
| CFAD | 真伪 | 2,000 + 4 退化布局 | `.tmp_ssl/crossdomain/` |

### 报告导航（从底层证据到总叙事）

| 文档 | 内容 | 关键数字 |
|---|---|---|
| **`redblue_evolution.md`** | **总叙事 + 攻防演化王牌图（先读这个）** | 击穿 95.5%→0.3% |
| `redblue_round1.md` | 红蓝第 1 轮（修复后对比） | 检出 98.3% |
| `redteam_breakthrough.md` | 第 0 轮击穿基线（双口径） | AASIST 95.5% / 中文LR 99.0% |
| `exp_cn_same_domain.md` | 同域说话人外评测 | EER 0.00% |
| `exp_leave_one_system.md` | 未见攻击系统泛化 | 检出 100% |
| `cn_model_cross_eval.md` | 跨评测集行为（评测域匹配原则） | CFAD 47.5% = 域错配 |
| `exp_wide_coverage.md` | 宽覆盖（TTS+声码器伪） | CFAD 47.5→29.2% |
| `exp_cn_train_full.md` / `exp_domain_anchor.md` / `exp_fmfcc_pseudomain.md` | 方法学三件套 | 域适配 > 盲加 |
| `fmfcc_redteam_proximity.md` | 红队 vs FMFCC 特征空间 | 红队最像真(0.078) |

### 代码资产

| 模块 | 作用 |
|---|---|
| `src/fusion/xlsr_cn_channel.py` | **产品级中文域声学通道**（红队克隆→block / aishell 真→allow） |
| `src/api/server.py` | `VERICALL_ACOUSTIC=xlsr_cn` 开关（默认 AASIST 兼容） |
| `data/redteam/factory/cn_lr_scorer_full.pkl` | 正式中文域打分器 v4（aishell 100人 + FMFCC） |
| `scripts/eval/extract_{aishell,fmfcc,redteam,redteam_seg}_features.py` | 四套特征提取（断点续跑） |
| `scripts/redteam_factory/score_redteam_xlsr.py` | 难度打分（`--scorer cfad|full`） |

### 评测域原则（评委必问，务必自洽）

1. CFAD 伪=声码器族（STRAIGHT/GL/HIFIGAN）vs FMFCC 伪=商业/开源 TTS 族 → **不同伪造分布**；
2. "英文零样本 CFAD 15.5%"是攻击族意外重叠的假象（英文模型对一切中文判伪）；
3. 中文 TTS/克隆检测的正确评测 = 红队域（同族），辅以同域说话人外 + 留一系统三件套；
4. 复现：`exp_cn_same_domain.py`、`redblue_round1.py`、`evaluation/*.py` 全部可跑。
