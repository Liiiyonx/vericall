#!/usr/bin/env python
"""export_aasist_onnx.py — 5.2 通道① AASIST 导出 ONNX（2026-09-07）

复用 acoustic_channel 的权重解析与构建口径（Model(model_config) + best.pth），
导出 CPU/GPU 通用 ONNX（batch 动态），并做 torch-vs-onnxruntime logits 一致性校验。
用法：python -u scripts/eval/export_aasist_onnx.py [--out data/models/aasist.onnx]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from paths import AASIST_DIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/models/aasist.onnx"))
    ap.add_argument("--nb-samp", type=int, default=48000)
    args = ap.parse_args()

    sys.path.insert(0, str(AASIST_DIR))
    from fusion.acoustic_channel import _resolve_weight, _load_exp_config  # noqa: E402
    import numpy as np
    import torch

    wpath, mc = _resolve_weight()
    assert wpath and mc, "未找到 AASIST 权重/config"
    print(f"[权重] {wpath}")
    print(f"[config] {mc}")

    from models.AASIST import Model
    model = Model(mc).eval()
    sd = torch.load(wpath, map_location="cpu")
    model.load_state_dict(sd)
    print("[模型] 构建+加载完成")

    # AASIST forward 返回 (last_hidden, out)；包一层只留 logits（同 acoustic_channel 用法 `_, out = model(x)`）
    import torch.nn as nn

    class _Logits(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            _, out = self.m(x)
            return out
    wrapped = _Logits(model).eval()

    dummy = torch.randn(1, args.nb_samp)
    t0 = time.time()
    with torch.no_grad():
        ref = wrapped(dummy).detach().numpy()
    print(f"[torch] 前向 ok ({time.time()-t0:.2f}s), logits={ref.shape}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped, dummy, str(out),
        input_names=["wave"], output_names=["logits"],
        dynamic_axes={"wave": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=15)
    print(f"[onnx] 已导出 -> {out} ({out.stat().st_size/1e6:.1f}MB)")

    # 一致性校验
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        r = sess.run(["logits"], {"wave": dummy.numpy()})[0]
        max_d = float(np.abs(r - ref).max())
        print(f"[校验] onnxruntime logits 与 torch 最大差 {max_d:.6f} "
              f"({'PASS' if max_d < 1e-3 else 'FAIL'})")
    except ImportError:
        print("[校验] 未装 onnxruntime，跳过（仅导出）")


if __name__ == "__main__":
    main()
