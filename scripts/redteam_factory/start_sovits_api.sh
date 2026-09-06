#!/usr/bin/env bash
# start_sovits_api.sh — 启动 GPT-SoVITS api_v2 服务（2026-09-06 修复后固化）
# 修复要点（详见 docs/开发进度存档.md 09-06 段）：
#   1. 依赖全部放 torch_site2（conda env 内包会被清理代理删）→ PYTHONPATH 前置
#   2. torch 用 2.8.0+cu128（2.11 需 torchcodec 且 DLL 缺依赖）
#   3. 中文 BERT/文本链需 NLTK_DATA（cmudict + averaged_perceptron_tagger_eng 已手动下载）
#   4. fast_langdetect 模型缓存目录需先建
set -a; source D:/VeriCall_data/secrets/vericall_secrets.env 2>/dev/null; set +a

ROOT="C:/Users/Liii/Desktop/谛听VeriCall/external/GPT-SoVITS"
PY="D:/VeriCall_data/conda_envs/gpt-sovits/python.exe"
export PYTHONPATH="D:/VeriCall_data/pyenv/torch_site2"
export NLTK_DATA="D:/VeriCall_data/pyenv/torch_site2/nltk_data"

# fast_langdetect 模型缓存目录
mkdir -p "$ROOT/GPT_SoVITS/pretrained_models/fast_langdetect"

cd "$ROOT" || exit 1
exec "$PY" -u api_v2.py -a 127.0.0.1 -p 9880
