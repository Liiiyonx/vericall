# 红队伪造语料工厂（redteam_factory）

极致化计划书 §四的落地：网格化批量合成「引擎 × 方言 × 话术 × 信道」伪造语音，
目标规模 ≥50,000 条（含方言 ≥5,000），全部带 meta.csv 元数据与质量门禁。

## 组成

| 文件 | 职责 |
|---|---|
| `factory_config.json` | 网格配置：引擎 cmd 模板、方言目录、信道预设、每格上限 |
| `run_factory.py` | 编排器：探测引擎 → 生成网格计划 → 合成 → 信道退化 → meta.csv |
| `quality_gate.py` | 质量门禁：物理检查（时长/削波/静音）+ 通道①反向筛选难例 |
| `check_isolation.py` | 隔离检查（详册 §7.3）：训练链路禁引 redteam/ 等目录，CI 挂接 |

## 使用流程

```bash
# 0. 每次评估/训练前（CI 周回归）
python scripts/redteam_factory/check_isolation.py

# 1. 干跑：看引擎可用性与网格规模
python scripts/redteam_factory/run_factory.py --dry-run

# 2. 合成（断点续跑，已存在自动跳过）
python scripts/redteam_factory/run_factory.py --engine sovits --dialect minnan --per-cell 5

# 3. 质量门禁
python scripts/redteam_factory/quality_gate.py                  # 物理检查
python scripts/redteam_factory/quality_gate.py --reverse-screen # +难例筛选（需 GPU）
```

## 依赖准备（当前状态）

- **Edge-TTS（已接入，开箱即用）**：`pip install edge-tts` + ffmpeg，无需参考音。
  音色已按 `--list-voices` 实际目录校准：东北=辽宁腔 XiaobeiNeural，河南以陕西腔
  XiaoniNeural 近似，粤=zh-HK 音色，闽南以台湾国语近似，川渝无原生音色回落普通话
  （方言感由话术文本承担）。零成本补齐方言维度，是 GPT-SoVITS 就位前的主力引擎。
- **TTS 克隆引擎**（未安装，探测到后自动启用）：GPT-SoVITS / CosyVoice / Seed-VC
  克隆到 `external/`，并在 `factory_config.json` 校准 `probe_path` 与 `cmd` 模板。
  获取方式见 `docs/数据获取方式详册.docx` 第 6 章（许可证约束见 §8.1：CosyVoice 仅学术）。
- **方言参考音**（仅克隆引擎需要）：放入 `data/redteam/{mandarin,minnan,...}/*.wav`
  （志愿者招募见任务书 P2-1，知情同意书模板已备）。当前为时间瓶颈项。
- **话术文本**：`data/scam_corpus/corpus_v0.1.jsonl`（话术库管线产出）或
  `data/redteam/scam_scripts.jsonl`（旧 80 条小样本），方言匹配样本优先。

## meta.csv 约定

前 6 列与《数据获取方式详册》§7.2 一致：
`path, label, attack_id, channel, source, license`
扩展列：`engine, dialect, script_id, speaker_ref, duration_s`
标签约定与 AASIST 训练配置一致：**bonafide=1 / spoof=0**，工厂产出全部为 spoof。
