# 红队方言评测集

通道③（话术语义风险）的评测样本库：用真实方言 / 红队诈骗剧本录制的
音频，用来检验"话术风险评分"在口语化、方言化场景下的鲁棒性。

## 组织方式

每个场景一个子目录，目录下含 `manifest.json` 与若干 `.wav`：

```
data/redteam/
├── dial_henan/                 # 河南方言场景
│   ├── manifest.json
│   ├── scam_001.wav            # 冒充熟人要钱（方言）
│   └── normal_001.wav          # 家人正常闲聊（方言）
└── dial_cantonese/
    └── ...
```

### manifest.json 格式

```json
[
  {"wav": "scam_001.wav",  "label": "scam",   "note": "冒充子女要验证码"},
  {"wav": "normal_001.wav","label": "normal", "note": "家人报平安"}
]
```

- `label` 仅用于统计报表，取值 `scam` / `normal`。
- `note` 给人看，说明该样本意图。

## 运行

```bash
# 先盘点有哪些样本（不需模型）
python evaluation/redteam_eval.py --list

# 跑全量评测（需本地 Ollama + SenseVoice 权重，见 .env.example）
python evaluation/redteam_eval.py --out reports/redteam_<时间戳>.json
```

## 数据伦理

所有方言 / 红队样本**必须**签署知情同意书（模板见 `docs/知情同意书模板.md`），
仅用于研究与评测，严禁外传。合成样本须在 manifest 注明合成方式。
