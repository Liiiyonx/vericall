# 谛听 VeriCall

[![CI](https://github.com/Liiiyonx/vericall/actions/workflows/ci.yml/badge.svg)](https://github.com/Liiiyonx/vericall/actions/workflows/ci.yml)
![AASIST dev EER](https://img.shields.io/badge/AASIST_dev_EER-0.745%25-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

面向老年群体的 **AI 拟声电话诈骗拦截系统**。

当诈骗分子用 AI 伪造家人声音来电时，系统在通话中实时给出「声学伪造检测 + 家庭声纹比对 + 话术风险」三重证据判定，帮助老人识别风险并及时联系子女。

> **口径说明**：系统内的「拦截」指**旁路预警 + 人工决策**（整屏警示 / 语音播报 / 子女卡片 / 拉子女进通话），不做通话阻断。
>
> **本仓库为源代码仓**：不含数据集、模型权重、评测产出与部署配置。运行所需的数据与密钥由使用者自行准备。

## 已实现能力

| 通道 | 技术 | 状态 |
|---|---|---|
| ① 声学伪造检测 | XLS-R 中文域（默认）/ AASIST 英文基线回退 | XLS-R 红队克隆拦截、真语音放行；AASIST dev EER 0.745% / eval EER 3.494% |
| ② 家庭声纹 | CAMPPlus（modelscope） | 真实推理，同人 ≈ 1.0 / 异人 ≈ 0.03 |
| ③ 话术语义 | SenseVoice ASR + 云端 DeepSeek 风险评分 | 真实推理，无密钥时回退规则评分器 |
| ⓪ 涉诈号码先验 | 号码黑名单查表 | 命中即先验抬升风险 |
| 融合决策 | 三通道加权 + 纵深防御 + 单通道中危升级 | 可解释，绿 / 黄 / 红三级 |
| 适老交互 | 大字模式 + 红黄绿整屏色块 + 语音播报 | 纯前端，localStorage 记忆偏好 |
| 子女端联动 | 拦截分享卡片（来电时间 / 三通道证据 / 建议话术） | 可保存转发 |
| 流式实时引擎 | VAD + WebSocket + 增量融合状态机 | 3s 滑窗 / 1s 步进，边说边判 |

## 架构

三通道并行推理 → 融合编排 → 三级裁决。完整架构图（mermaid）见 [`assets/architecture.md`](assets/architecture.md)。

```
src/
├── paths.py                  # 统一路径配置（环境变量 > .env > 默认值）
├── api/
│   ├── server.py             # FastAPI 演示服务 + 前端接口
│   ├── family_api.py         # 子女守护端（账号 / 告警 / 三方通话 WS）
│   └── history_store.py      # 检测历史分片存储
├── server/                   # 流式实时引擎
│   ├── stream_pipeline.py    # 流式管线
│   ├── incremental_fusion.py # 增量融合
│   ├── vad.py                # 语音活动检测
│   ├── window_buffer.py      # 滑窗缓冲
│   ├── scheduler.py          # 调度
│   ├── ws_api.py             # WebSocket 接口
│   └── audio_util.py
└── fusion/
    ├── acoustic_channel.py   # 通道① AASIST 声学伪造
    ├── xlsr_cn_channel.py    # 通道① 中文域鲁棒通道（XLSR-CN）
    ├── number_channel.py     # 通道⓪ 涉诈号码先验查表
    ├── voiceprint_channel.py # 通道② CAMPPlus 声纹（落盘持久化）
    ├── semantic_channel.py   # 通道③ SenseVoice + LLM 话术
    ├── rule_scorer.py        # 离线规则评分器（LLM 兜底）
    ├── transcript_cache.py   # 转写缓存（md5 keyed，断网可复用）
    ├── challenge_response.py # 挑战应答（回放防护）
    ├── fusion_orchestrator.py# 三通道融合决策
    └── pipeline.py           # 端到端管线编排（含离线降级）
```

## 目录结构

```
vericall/
├── src/          # 核心源码（三通道 + 融合 + 流式引擎 + API）
├── tests/        # pytest 单测（融合决策 / EER / 流式 / 家庭隔离 / 安全加固）
├── configs/      # AASIST 训练配置
├── scripts/      # 数据下载 / 预处理 / 训练 / 评测 / 监控
├── web/          # 单页演示前端（适老端 / 子女端 / 实时监测 / 演示展厅）
├── assets/       # 架构图等文档素材
└── .github/      # CI 工作流
```

## 快速开始

### 1. 装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置路径与密钥

所有绝对路径集中在 `src/paths.py`，按 **环境变量 > 项目根 `.env` > 代码默认值** 三级解析：

```bash
cp .env.example .env
# 然后按实际情况编辑 VERICALL_DATA_ROOT 等
python -c "from src.paths import report; print(report())"   # 打印当前生效配置与存在性
```

关键项：`VERICALL_DATA_ROOT`（数据集 / 模型 / 日志根）、`VERICALL_ASVSPOOF_LA`、`VERICALL_SENSEVOICE_DIR`、`VERICALL_AASIST_DIR`、`VERICALL_HISTORY_DIR`、`VERICALL_DEVICE`、`VERICALL_PORT`。

语义通道默认走云端 LLM，需提供 `SCAM_LLM_BASE` / `SCAM_LLM_KEY` / `SCAM_LLM_MODEL`；未提供时自动回退到 `rule_scorer.py` 规则评分。

### 3. 基座模型（可选，缺省走降级）

```bash
git clone https://github.com/clovaai/aasist external/aasist   # AASIST 基线需自行 clone
python scripts/launch_training.py --config configs/AASIST_5060.conf
```

### 4. 启动

```bash
# Windows
start.bat

# Linux / macOS
bash start.sh
```

启动器会自动：① 检查并安装缺失依赖；② 运行 `scripts/preflight.py` 检查云端 LLM、CUDA、SenseVoice、AASIST 与 XLS-R 状态；③ 拉起服务并打开 `http://localhost:8000`。

预检只报告状态，不会因端点不可达而改写运行模式；缺云密钥时语义通道走规则兜底。

## 离线降级模式

设置 `VERICALL_OFFLINE=1` 开启三级降级，每一级都保住核心链路：

| 环节 | 正常 | 降级策略 |
|---|---|---|
| 通道① AASIST | GPU 推理 | 无权重 → stub；有权重则 CPU 推理 |
| 通道② CAMPPlus | GPU | CPU 推理（约 30MB，毫秒级） |
| 通道③ ASR | SenseVoice | 转写缓存复用历史转写 |
| 通道③ LLM | 云端 deepseek-chat | 规则评分器 `rule_scorer.py` 兜底 |

## 适老一体机形态（老人端常驻 + 子女端 PWA）

- `start_elder.bat`：一键启动后端 + 全屏 kiosk 打开适老端（Edge → Chrome → 默认浏览器回退）；
- 适老端「常开守护」：麦克风常开走 `/ws/stream` 边说边测，红黄绿大环 + 语音提醒 + 红牌一键叫子女（黄色提醒 20s 冷却）；
- 开机自启：`scripts/install_elder_autostart.bat`（启动文件夹快捷方式，删除即取消）；
- 子女端 PWA：`web/manifest.json` + `sw.js`（离线壳缓存，`/api` `/ws` 不缓存）；手机浏览器打开 `/child.html` 添加到主屏幕即装成 App。

## 家庭联防闭环

- **号码举报**：子女端告警卡「举报此号码」→ `POST /api/family/numbers/report` → 写入家庭私有黑名单，只对当前 `family_id` 生效，不污染公开库；
- **来电号码链路**：检测页可选填来电号码，随检测历史落盘并出现在子女端告警卡；
- **举报材料包**：一键下载「时间 · 号码 · 三通道证据 · 建议」文本（12321 / 96110 参考格式）。

数据按 `family_id` 隔离账号、告警、ACK、待接呼叫、通话 WebSocket、检测历史与声纹列表。远端访问 `/api/history` 与 `/api/voices` 必须携带 Bearer token。

## 安全设计

- **登录与 token**：密码使用 PBKDF2-SHA256 加盐哈希；登录按 IP + 用户名双键限流（5 分钟 / 5 次）；失败身份缓存有容量上限；token 仅保存 SHA-256 摘要，30 天过期。
- **代理与本机兼容**：`VERICALL_TRUST_PROXY_HEADERS` 默认关闭；本机免登录路径同时校验回环地址、`Origin` 与 `Host`，防伪造转发头越权。
- **WebSocket**：`/ws/stream` 与 `/ws/call` 使用 30 秒一次性票据，票据绑定用途、房间与家庭且只能消费一次；连接数、音频帧与控制帧均有限额。
- **输入与数据边界**：上传限制大小、后缀与音频魔数；检测历史按家庭分片写入；声纹、挑战应答、家庭举报均按 `family_id` 隔离；声纹名与 household 防路径穿越。
- **本地敏感数据**：账号、邀请码、声纹向量、上传音频、转写缓存与家庭历史分片在 POSIX 上按目录 `0700` / 文件 `0600` 收口；Windows 依赖专用服务账号与宿主 ACL。
- **接口暴露面**：API 文档默认关闭；响应 `Cache-Control: no-store`；统一附加 `X-Content-Type-Options`、`X-Frame-Options`、`Referrer-Policy`、`Permissions-Policy`。

> 已知限制：登录限流与 WS 票据为单进程内存状态，多 worker 下不共享；前端 token 存于 `localStorage`，依赖 HTTPS 与严格 XSS 防护；生产上线仍需可信反向代理、密钥管理、集中审计与分布式限流。

## 测试

```bash
pip install pytest numpy
pytest tests/ -q     # 176 项：无需 GPU、无需外部服务
```

26 个测试模块，176 项用例，覆盖融合决策、EER 指标、流式管线、家庭隔离与安全加固。

## 许可

MIT，见 [LICENSE](LICENSE)。
