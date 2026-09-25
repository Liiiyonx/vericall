# -*- coding: utf-8 -*-
"""AASIST 权重选择与可审计元数据回归（不加载 torch）。"""
from __future__ import annotations

import hashlib
import json

from fusion.acoustic_channel import _resolve_weight_details
import paths


def _make_exp(root, name: str, quality: dict | None = None,
              weights: tuple[str, ...] = ()) -> None:
    exp = root / name
    wdir = exp / "weights"
    wdir.mkdir(parents=True)
    for filename in weights:
        (wdir / filename).write_bytes(filename.encode("utf-8"))
    (exp / "config.conf").write_text(json.dumps({
        "model_config": {"nb_samp": 48000},
    }), encoding="utf-8")
    if quality is not None:
        (exp / "model_quality.json").write_text(
            json.dumps(quality), encoding="utf-8")


def test_model_quality_best_ckpt_wins_and_sha_is_reported(tmp_path):
    _make_exp(
        tmp_path,
        "target",
        {
            "best_ckpt": "epoch_23_0.745.pth",
            "dev_eer": 0.745,
            "swa_dev_eer": 1.178,
        },
        ("epoch_23_0.745.pth", "best.pth", "swa.pth"),
    )
    _make_exp(
        tmp_path,
        "decoy",
        {"best_ckpt": "best.pth", "dev_eer": 99.0},
        ("best.pth",),
    )

    weight, config, meta = _resolve_weight_details(tmp_path / "target")
    assert weight.endswith("epoch_23_0.745.pth")
    assert config["nb_samp"] == 48000
    assert meta["experiment"] == "target"
    assert meta["selection_reason"] == "model_quality.json.best_ckpt"
    assert meta["quality_source"] == "model_quality.json:best_ckpt"
    assert meta["dev_eer"] == 0.745
    expected = hashlib.sha256(b"epoch_23_0.745.pth").hexdigest()
    assert meta["weight_sha256"] == expected
    assert "decoy" not in weight


def test_selection_never_crosses_experiment_and_falls_back_locally(tmp_path):
    _make_exp(tmp_path, "target", None, ("swa.pth", "epoch_7_2.5.pth"))
    _make_exp(tmp_path, "other", {"best_ckpt": "best.pth", "dev_eer": 0.1},
              ("best.pth",))

    weight, _, meta = _resolve_weight_details(tmp_path / "target")
    assert weight.endswith("swa.pth")
    assert meta["selection_reason"] == "experiment_swa"
    assert meta["experiment"] == "target"
    assert meta["weight_sha256"]


def test_server_status_reuses_selected_weight_metadata(tmp_path, monkeypatch):
    import api.server as server

    exp = tmp_path / "exp"
    exp.mkdir()
    (exp / "model_quality.json").write_text(json.dumps({
        "eval_eer_best_verified": 3.494,
    }), encoding="utf-8")
    meta = {
        "weight_path": str(exp / "weights" / "epoch_23_0.745.pth"),
        "weight_sha256": "a" * 64,
        "experiment": "target",
        "experiment_dir": str(exp),
        "quality_source": "model_quality.json:best_ckpt",
        "selection_reason": "model_quality.json.best_ckpt",
        "dev_eer": 0.745,
    }
    monkeypatch.setattr(server, "EXP_DIR", exp)
    monkeypatch.setattr(
        "fusion.acoustic_channel._resolve_weight_details",
        lambda: (meta["weight_path"], {"nb_samp": 48000}, meta),
    )

    status = server._model_status()
    assert status["aasist_dev_eer"] == 0.745
    assert status["eval_eer"] == 3.494
    assert status["aasist_weight"]["experiment"] == "target"
    assert status["aasist_weight"]["selection_reason"] == \
        "model_quality.json.best_ckpt"
    assert status["aasist_weight"]["weight_sha256"] == "a" * 12


def test_acoustic_backend_defaults_to_chinese_wide(monkeypatch):
    monkeypatch.delenv("VERICALL_ACOUSTIC", raising=False)
    monkeypatch.delitem(paths._DOTENV, "VERICALL_ACOUSTIC", raising=False)
    monkeypatch.delitem(paths._SECRETS, "VERICALL_ACOUSTIC", raising=False)
    assert paths.acoustic_backend() == "xlsr_cn:wide"


def test_acoustic_selection_uses_xlsr_and_records_active(monkeypatch):
    import api.server as server

    class FakeAasist:
        def __init__(self, device=None):
            self.device = device

    class FakeXlsr:
        def __init__(self, scorer="wide", device=None):
            self.scorer = scorer
            self.device = device

        def load(self):
            return True

    monkeypatch.setattr("fusion.acoustic_channel.AcousticChannel", FakeAasist)
    monkeypatch.setattr("fusion.xlsr_cn_channel.XlsrCnChannel", FakeXlsr)
    monkeypatch.setattr(server, "_acoustic_state", {
        "requested": None, "active": None, "fallback_reason": None})

    channel = server._load_acoustic_channel("xlsr_cn:v4")
    assert isinstance(channel, FakeXlsr)
    assert channel.scorer == "v4"
    assert server._acoustic_state == {
        "requested": "xlsr_cn:v4",
        "active": "xlsr_cn:v4",
        "fallback_reason": None,
    }


def test_acoustic_selection_falls_back_to_aasist_with_reason(monkeypatch):
    import api.server as server

    class FakeAasist:
        def __init__(self, device=None):
            self.device = device

    class FailingXlsr:
        def __init__(self, scorer="wide", device=None):
            self.scorer = scorer

        def load(self):
            return False

    monkeypatch.setattr("fusion.acoustic_channel.AcousticChannel", FakeAasist)
    monkeypatch.setattr("fusion.xlsr_cn_channel.XlsrCnChannel", FailingXlsr)
    monkeypatch.setattr(server, "_acoustic_state", {
        "requested": None, "active": None, "fallback_reason": None})

    channel = server._load_acoustic_channel("xlsr_cn:wide")
    assert isinstance(channel, FakeAasist)
    assert server._acoustic_state == {
        "requested": "xlsr_cn:wide",
        "active": "aasist",
        "fallback_reason": "xlsr_load_failed",
    }
