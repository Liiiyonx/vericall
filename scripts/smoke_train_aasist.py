#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
谛听 VeriCall — AASIST 训练管线冒烟测试。

用合成音频(正弦+噪声)构造一个极小的 ASVspoof2019-LA 布局数据集,
跑 1 个 epoch 的 AASIST 训练 + dev 评估, 验证整条管线
(数据加载 -> 模型前向 -> 损失 -> EER 计算) 无报错。
不需要 16GB 真实数据, 也不需要联网。

用法:
    python scripts/smoke_train_aasist.py
完成后会在 <root>/smoke_exp 生成权重与 metric_log。
"""
import os
import json
import sys
import shutil
import subprocess
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")))
from paths import SMOKE_LA, SMOKE_ROOT, AASIST_DIR, TRAIN_FLAC  # noqa: E402

ROOT = str(SMOKE_LA)
AASIST_DIR = str(AASIST_DIR)


def _gen_flac(path, sr=16000, dur=4.0, spoof=False, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * dur)) / sr
    if spoof:
        # 伪造: 多个高频叠加 + 更强谐波, 模拟 TTS/VC 伪影
        x = (np.sin(2 * np.pi * 220 * t) * 0.4
             + np.sin(2 * np.pi * 440 * t) * 0.3
             + np.sin(2 * np.pi * 880 * t) * 0.2)
        x = x * (1 + 0.3 * np.sin(2 * np.pi * 5 * t))
    else:
        # 真实: 基频 + 轻噪
        x = np.sin(2 * np.pi * 150 * t) * 0.6 + rng.normal(0, 0.05, len(t))
    x = x / np.max(np.abs(x) + 1e-9)
    sf.write(path, x.astype(np.float32), sr, subtype="PCM_16")


def _write_protocol(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for spk, utt, sysid, key in rows:
            f.write(f"{spk} {utt} - {sysid} {key}\n")


def build_synthetic():
    splits = {
        "train": ("train", 60, 40),   # (split_out, n_bonafide, n_spoof)
        "dev": ("dev", 20, 20),
        "eval": ("eval", 20, 20),
    }
    proto_dir = os.path.join(ROOT, "ASVspoof2019_LA_cm_protocols")
    os.makedirs(proto_dir, exist_ok=True)
    for split_name, (out, n_b, n_s) in splits.items():
        flac_dir = os.path.join(ROOT, f"ASVspoof2019_LA_{out}", "flac")
        os.makedirs(flac_dir, exist_ok=True)
        rows = []
        prefix = {"train": "LA_T", "dev": "LA_D", "eval": "LA_E"}[split_name]
        for i in range(n_b):
            utt = f"{prefix}_{1000000+i:07d}"
            _gen_flac(os.path.join(flac_dir, utt + ".flac"), spoof=False, seed=i)
            rows.append((f"{1000000+i}", utt, "NA", "bonafide"))
        for i in range(n_s):
            utt = f"{prefix}_{2000000+i:07d}"
            _gen_flac(os.path.join(flac_dir, utt + ".flac"), spoof=True, seed=1000 + i)
            rows.append((f"{2000000+i}", utt, f"A{(i % 13) + 7:02d}", "spoof"))
        ext = "trn" if out == "train" else "trl"
        _write_protocol(os.path.join(proto_dir, f"ASVspoof2019.LA.cm.{out}.{ext}.txt"), rows)
    print(f"[smoke] 合成数据已生成 @ {ROOT}")


def write_smoke_config():
    cfg = {
        "database_path": ROOT,
        "asv_score_path": str(SMOKE_ROOT / "ASVspoof2019_LA_asv_scores" / "none.scores"),
        "model_path": "./models/weights/AASIST_5060.pth",
        "batch_size": 8,
        "num_epochs": 1,
        "loss": "CCE",
        "track": "LA",
        "eval_all_best": "True",
        "eval_output": "eval_scores_using_best_dev_model.txt",
        "cudnn_deterministic_toggle": "False",
        "cudnn_benchmark_toggle": "True",
        "model_config": {
            "architecture": "AASIST",
            "nb_samp": 64600,
            "first_conv": 128,
            "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
            "gat_dims": [64, 32],
            "pool_ratios": [0.5, 0.7, 0.5, 0.5],
            "temperatures": [2.0, 2.0, 100.0, 100.0]
        },
        "optim_config": {
            "optimizer": "adam",
            "amsgrad": "False",
            "base_lr": 0.0001,
            "lr_min": 0.000005,
            "betas": [0.9, 0.999],
            "weight_decay": 0.0001,
            "scheduler": "cosine"
        }
    }
    cfg_path = os.path.join(AASIST_DIR, "smoke_5060.conf")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)
    return cfg_path


def main():
    build_synthetic()
    cfg_path = write_smoke_config()
    out_dir = os.path.join(AASIST_DIR, "smoke_exp")
    if os.path.exists(out_dir):
        try:
            shutil.rmtree(out_dir)
        except OSError:
            # 沙箱环境无回收站时 rmtree 可能被拦截, 忽略即可
            pass
    cmd = [
        sys.executable, "main.py",
        "--config", cfg_path,
        "--output_dir", out_dir,
        "--seed", "1234",
    ]
    print(f"[smoke] 运行: {' '.join(cmd)} (cwd={AASIST_DIR})")
    rc = subprocess.call(cmd, cwd=AASIST_DIR)
    print(f"[smoke] AASIST main 退出码 = {rc}")
    if rc == 0:
        print("[smoke] ✅ 管线端到端跑通 (数据加载/前向/损失/EER 均正常)")
    else:
        print("[smoke] ❌ 退出码非0, 需排查")
    return rc


if __name__ == "__main__":
    main()
