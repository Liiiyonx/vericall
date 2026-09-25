# -*- coding: utf-8 -*-
"""任务书 A 层工具链的单测（纯逻辑，无 torch/soundfile 依赖，CI 可跑）。

覆盖：
  - scripts/convert_fmfcc_protocol.py：协议 5 列格式、G00/Axx 前缀解析、扁平布局回退
  - scripts/degrade_audio.py：_ulaw 纯 numpy 实现（Python 3.13 已移除 audioop）
  - evaluation/cross_domain_eval.py：build_matrix 已知格 + TODO 兜底
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "evaluation"))


# 注意：本文件不再模块顶层导入任何会触发 `import paths` 的模块
# （convert_fmfcc_protocol 仅依赖标准库，可安全顶层导入；
#  degrade_audio / cross_domain_eval 会经 audio_util / paths 间接导入 paths，
#  其模块级 `OFFLINE` 常量在首次 import 时冻结。为避免干扰
#  tests/test_offline_pipeline.py（它靠设置 VERICALL_OFFLINE=1 后再 import paths），
#  这两个模块改为在函数内延迟导入。）
import convert_fmfcc_protocol as cf  # noqa: E402  (仅标准库，无 paths 依赖)


# --------------------------------------------------------------------------- #
# convert_fmfcc_protocol
# --------------------------------------------------------------------------- #
def _make_tree(root: Path, with_subdirs: bool = True):
    """造一棵合成 FMFCC-A 目录：Development/{bonafide,fake} 若干 G00/Axx 文件。"""
    if with_subdirs:
        bon = root / "Development" / "bonafide"
        fak = root / "Development" / "fake"
    else:
        bon = fak = root  # 扁平：文件直接放 root
    bon.mkdir(parents=True, exist_ok=True)
    fak.mkdir(parents=True, exist_ok=True)
    for i in range(2):
        (bon / f"G00_{i:04d}.wav").write_text("x")
    for i in range(3):
        (fak / f"A0{i+1}_{i:04d}.wav").write_text("x")
    return root / ("Development" if with_subdirs else "")


def test_system_from_name():
    assert cf._system_from_name("G00_0001") == ("bonafide", None)
    assert cf._system_from_name("A03_0007") == ("spoof", "A03")
    assert cf._system_from_name("readme.txt") is None


def test_convert_split_standard_layout(tmp_path):
    split_dir = _make_tree(tmp_path, with_subdirs=True)
    out = tmp_path / "out"
    r = cf.convert_split(split_dir, "dev", out, copy_audio=False, sr=16000)
    assert r["n_bonafide"] == 2
    assert r["n_spoof"] == 3
    proto = Path(r["protocol"]).read_text(encoding="utf-8").splitlines()
    assert len(proto) == 5
    for line in proto:
        parts = line.split()
        assert len(parts) == 5, line
        assert parts[2] == "-"
        assert parts[4] in ("bonafide", "spoof")
        if parts[4] == "bonafide":
            assert parts[3] == "-"
        else:
            assert re.match(r"^A\d{2}$", parts[3])


def test_convert_split_flat_layout(tmp_path):
    """扁平布局：文件直接放 root，靠 G00/Axx 前缀判定。"""
    split_dir = _make_tree(tmp_path, with_subdirs=False)
    out = tmp_path / "out"
    r = cf.convert_split(split_dir, "dev", out, copy_audio=False, sr=16000)
    assert r["n_bonafide"] == 2
    assert r["n_spoof"] == 3


# --------------------------------------------------------------------------- #
# degrade_audio._ulaw（Py3.13 兼容）
# --------------------------------------------------------------------------- #
def test_ulaw_no_audioop_and_bounded():
    import degrade_audio  # 延迟导入，避免冻结 paths.OFFLINE
    x = np.linspace(-1, 1, 1000, dtype=np.float32)
    y = degrade_audio._ulaw(x)
    assert y.shape == x.shape
    assert y.dtype == np.float32
    assert float(y.max()) <= 1.0 + 1e-6 and float(y.min()) >= -1.0 - 1e-6
    # 不依赖已移除的 audioop（检查方法字节码里是否出现过该名字）
    assert "audioop" not in degrade_audio._ulaw.__code__.co_names


def test_ulaw_monotonic_companding():
    """μ-law 对大幅度的量化更粗：|x|=0.9 与 |x|=1.0 解码差应小于 0.1 与 0.2 的差。"""
    import degrade_audio
    near = degrade_audio._ulaw(np.array([0.9], dtype=np.float32))[0]
    far = degrade_audio._ulaw(np.array([1.0], dtype=np.float32))[0]
    mid = degrade_audio._ulaw(np.array([0.1], dtype=np.float32))[0]
    assert (far - near) < (near - mid)  # 幅度越大，步长越粗


# --------------------------------------------------------------------------- #
# cross_domain_eval.build_matrix
# --------------------------------------------------------------------------- #
def test_build_matrix_known_cell_and_todo():
    import cross_domain_eval as cde  # 延迟导入
    avail = {
        "2019LA-eval": (True, "eval 协议"),
        "FMFCC-A": (False, "缺 dev 协议"),
        "CFAD": (False, "缺 dev 协议"),
        "redteam-clean": (False, "缺红队音频"),
        "redteam-phone": (False, "缺红队音频"),
        "in-the-wild": (False, "缺数据"),
    }
    md = cde.build_matrix(avail, computed={})
    assert "3.49%" in md
    assert "| 2019LA（现模型） | 3.49%" in md
    # 无数据域应为 TODO
    assert md.count("TODO") >= 4


def test_build_matrix_fills_computed_cell():
    import cross_domain_eval as cde  # 延迟导入
    avail = {c: (True, "ok") for c in
             ["2019LA-eval", "FMFCC-A", "CFAD", "redteam-clean",
              "redteam-phone", "in-the-wild"]}
    computed = {"FMFCC-A": {"n_bonafide": 1000, "n_spoof": 1000,
                            "eer_pct": 12.345, "threshold": 0.5,
                            "far_pct": 1.0, "frr_pct": 2.0}}
    md = cde.build_matrix(avail, computed)
    assert "12.345%" in md
