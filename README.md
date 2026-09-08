# 谛听 VeriCall

[![CI](https://github.com/Liiiyonx/vericall/actions/workflows/ci.yml/badge.svg)](https://github.com/Liiiyonx/vericall/actions/workflows/ci.yml) ![AASIST dev EER](https://img.shields.io/badge/AASIST_dev_EER-0.745%25-brightgreen) ![License](https://img.shields.io/badge/license-MIT-blue)

> 面向老年群体的 **AI 拟声电话诈骗拦截系统**（换脸检测为二期规划，见下）。
> 当诈骗分子用 AI 伪造家人声音来电时，通话中实时给出"伪造检测 + 家庭声纹 + 话术风险"三重证据判定，保护老人不被转账诈骗。

- 完整立项论证、技术路线、赛事打法：`docs/项目方案书.html`
- 数据集获取清单：`docs/数据集获取指南.md`
- 训练指南：`docs/训练指南.md`
- 环境变量速查：`docs/环境变量.md`

## 已实现能力（阶段一 · 基线）

| 通道 | 技术 | 状态 |
|---|---|---|
| ① 声学伪造检测 | AASIST（clovaai/aasist，RTX 5060 8GB 实训） | dev EER 0.745%（best 0.316%）/ eval EER 3.494% |
| ② 家庭声纹 | CAMPPlus（modelscope） | 真实推理，同人≈1.0 / 异人≈0.03 |
| ③ 话术语义 | SenseVoice ASR + 云端 DeepSeek（deepseek-chat）风险评分 | 真实推理（默认云端，禁本地 Ollama） |
| 融合决策 | 三通道加权 + 纵深防御 + 单通道中危升级 | 可解释，绿/黄/红三级 |
| 适老交互 | 大字模式 + 红黄绿整屏色块 + 语音播报（`web/copy.js` 非指控式文案库） | 纯前端，localStorage 记忆偏好 |
| 子女端联动 | 拦截分享卡片（canvas 生成：来电时间/三通道证据/建议话术） | 实时告警与检测页均可生成，可保存转发 |

评测产出见 `evaluation/reports/`（跨攻击域细分 + 泛化鸿沟）。

## 目录结构

```
谛听VeriCall/
├── docs/                  # 方案书、数据集指南、训练指南、知情同意书模板
├── configs/               # AASIST 训练配置（RTX5060 / smoke）
├── src/
│   ├── paths.py           # 统一路径配置（环境变量 > .env > 默认值）
│   ├── api/
│   │   └── server.py      # FastAPI 演示服务 + 前端接口
│   └── fusion/
│       ├── acoustic_channel.py     # 通道① AASIST 声学伪造
│       ├── voiceprint_channel.py   # 通道② CAMPPlus 声纹（落盘持久化）
│       ├── semantic_channel.py     # 通道③ SenseVoice + LLM 话术
│       ├── rule_scorer.py          # 离线规则评分器（LLM 兜底 / 规则基线）
│       ├── transcript_cache.py     # 转写缓存（md5 keyed，断网可复用）
│       ├── fusion_orchestrator.py  # 三通道融合决策
│       └── pipeline.py             # 端到端管线编排（含离线降级）
├── evaluation/            # 指标(metrics) + 跨攻击域评测 + 红队评测运行器
│   ├── metrics.py
│   ├── attack_breakdown_eval.py
│   └── redteam_eval.py
├── scripts/               # 下载 / 预处理 / 训练 / 评估 / 监控 / 红队话术
├── tests/                 # pytest 单测（融合决策 + 评测指标）
├── web/                   # 单页演示前端
├── data/                  # 项目内数据（history.jsonl / uploads / redteam），git 忽略大文件
├── external/aasist/       # clovaai/aasist 需自行 clone（见下），git 忽略
└── assets/                # 图表、演示素材
```

## 快速开始

```bash
# 1. 环境（训练需 torch；仅跑演示/评测按需）
pip install -r requirements.txt

# 2. 数据（详见 docs/数据集获取指南.md）
#    - ASVspoof 2019 LA：官网注册申请（审核要等）
#    - 其余数据集见指南

# 3. 基线：AASIST（clovaai/aasist，必须 clone 到 external/）
git clone https://github.com/clovaai/aasist external/aasist

# 4. 一键训练（RTX5060 定制配置）
python scripts/launch_training.py --config configs/AASIST_5060.conf
```

## 5 分钟跑起来（演示）

无需训练、无需数据集即可演示：

```bash
# Windows（评委大概率 Windows）
双击 start.bat

# Linux / macOS
bash start.sh
```

启动器会自动：① 检查并安装缺失依赖（清华镜像）；② 检查 AASIST 权重（缺失仅提示，声学通道以 stub 运行）；③ 检查云端 LLM 密钥（`SCAM_LLM_*`），缺失/不可达则自动开启**离线降级模式**；④ 拉起服务并打开 `http://localhost:8000`。

> 干净虚拟机双击 `start.bat`，≤5 分钟见到演示页面；断网 / 无 GPU 也能进降级演示（见下）。

## 离线降级模式（答辩保命）

`VERICALL_OFFLINE=1` 开启三级降级，**每一级都保住核心演示链路**：

| 环节 | 正常 | 降级策略 |
|---|---|---|
| 通道① AASIST | GPU 推理 | 无权重 → stub（conf=0）；有权重则 CPU 推理 |
| 通道② CAMPPlus | GPU | CPU 推理（30MB，毫秒级） |
| 通道③ ASR | SenseVoice | 转写缓存 `data/cache/transcripts.json`（md5 keyed）复用历史转写 |
| 通道③ LLM | 云端 deepseek-chat | 规则评分器 `rule_scorer.py` 兜底 |

离线下三 demo 场景结论保持：**A=放行 / B=拦截 / C=警惕**，演示数据见 `data/demo_cache.json`。

## 架构

三通道 → 融合 → 三级裁决。架构图（mermaid）见 [`assets/architecture.md`](assets/architecture.md)。

## 演示音频合规

ASVspoof 许可**禁止再分发**，demo 不打包数据集里的 flac。运行 `scripts/prepare_demo_audio.py` 可把本机 LA 数据拷贝到 `assets/demo_audio/`（仅本地用，`.gitignore` 排除）；没有数据集时提示用 GPT-SoVITS 自生成（红队集建成后切换为完全自产，彻底无版权风险）。

## 配置（不再写死盘符）

所有绝对路径集中在 `src/paths.py`，通过**环境变量 > 项目根 `.env` > 代码默认值**三级解析。复制模板后按需修改：

```bash
cp .env.example .env      # 然后编辑 VERICALL_DATA_ROOT 等
python -c "from src.paths import report; print(report())"   # 打印当前生效配置与存在性
```

关键项：`VERICALL_DATA_ROOT`（数据集/模型/日志根）、`VERICALL_ASVSPOOF_LA`、`VERICALL_SENSEVOICE_DIR`、`VERICALL_AASIST_DIR`、`VERICALL_OLLAMA_HOST/MODEL`、`VERICALL_DEVICE`、`VERICALL_PORT`。

## 评测

```bash
# 通道① 跨攻击域细分（已知 A01–A06 vs 未知 A07–A19 泛化鸿沟）
python evaluation/attack_breakdown_eval.py --split dev
python evaluation/attack_breakdown_eval.py --split eval --limit 2000

# 红队方言鲁棒性评测（离线 text 模式，无需 Ollama/SenseVoice）
python evaluation/redteam_eval.py --mode text --out evaluation/reports/redteam_<时间戳>.json
# 构建/查看合成方言语料（6 方言 × 6 = 36 样本）
python evaluation/redteam_eval.py --build-corpus
python evaluation/redteam_eval.py --list

# P2-5 红队对抗实验报告（攻击面框架，复用上方 text 评测结果）
python evaluation/make_redteam_adversarial.py
# P2-2 跨域评测矩阵（即插即用：真实权重+数据就位后自动填格）
python evaluation/cross_domain_eval.py --list-domains
python evaluation/cross_domain_eval.py

# P1-3 中文协议对齐（FMFCC-A → ASVspoof 协议，含 --selftest 自检）
python scripts/convert_fmfcc_protocol.py --selftest
python scripts/convert_fmfcc_protocol.py --root <FMFCC-A> --out data/raw/FMFCC-A_asvspoof
# P1-4 电话信道退化（phone8k/noise 纯 numpy；mp3_16k/amr 需 ffmpeg）
python scripts/degrade_audio.py <音频或目录> --preset phone8k --out data/degraded/phone8k
```

指标口径见 `evaluation/metrics.py`（EER / FAR / FRR，含排序自检）。

## 测试

```bash
pip install pytest numpy
pytest tests/ -q       # 90 个单测：融合决策 + EER 指标 + 流式状态机 + 规则评分器 + 红队扰动/语料 + 号码通道 + 声纹标定（纯逻辑，无 GPU/Ollama）
```

## 里程碑

| 阶段 | tag | 目标 | 进度 |
|---|---|---|---|
| P0 · 可复现性抢救 | `v0.2-repro` | 任何机器 30 分钟跑起演示（离线降级 + 一键启动 + 声纹持久化） | ✅ AASIST 真实 EER 3.49%，降级保底 |
| P1 · 核心叙事兑现 | `v0.5-streaming` | 流式实时推理引擎 + 实时监测页 + 离线降级深化 | ✅ 3s 滑窗/1s 步进，P90 延迟 ≤2s |
| P2 · 差异化加分 | `v0.8-redteam` | 红队方言鲁棒性评测（离线 text 模式 + 对抗扰动） | ✅ 6 方言 36 样本，粤语/闽南 100% 漏拦、FA 0% |
| P3 · 参赛冲刺 | `v1.0-competition` | 视频脚本 / 社区试点 / 论文大纲 / 五赛事矩阵 / 证据总览 | ✅ 材料就绪（见下） |

> 注意：流式实时拦截（VAD/WS + 增量融合状态机）已在 P1 实现并通过端到端测试，
> 系统已从"文件批处理"升级为"边说边判"。**换脸检测为二期规划**，本期聚焦 AI 拟声（声学伪造）拦截。

## 参赛冲刺（P3 · `v1.0-competition`）

> 2027.03–06 五赛事（信安作品赛 / 4C2027 / 互联网+红旅 / 挑战杯 / 大创）。所有真实指标与可复现 tag 见 [`docs/参赛证据总览.md`](docs/参赛证据总览.md)。

| 材料 | 文件 | 状态 |
|---|---|---|
| 参赛证据总览（答辩一页纸） | `docs/参赛证据总览.md` | ✅ 真实指标聚合 + tag 链 |
| 演示视频脚本（3min） | `docs/演示视频脚本.md` | ✅ 脚本先行，镜头可现场复现 |
| 社区试点方案 + 报告模板 | `docs/试点使用报告模板.md` | ✅ 待实地执行 |
| 小论文大纲 | `docs/小论文大纲.md` | ✅ 大纲就绪（4.1–4.4 待真实数据） |
| 五赛事材料裁剪矩阵 | `docs/参赛材料裁剪矩阵.md` | ✅ 五赛事叙事裁剪 |

## 数据伦理

家人声纹录音、红队方言合成样本一律签署知情同意书（模板见 `docs/知情同意书模板.md`），仅用于研究与评测，不外传。
