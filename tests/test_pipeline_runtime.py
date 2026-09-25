# -*- coding: utf-8 -*-
"""端到端编排中的 ASR 显存策略回归。"""
from __future__ import annotations

from types import SimpleNamespace

import fusion.pipeline as pipeline


class _FakeAsr:
    def __init__(self):
        self._model = object()

    def transcribe(self, audio_path, language="auto"):
        return "妈，我手机坏了，快转五万到这个卡"


def _fake_sem(backend: str):
    def analyze(transcript):
        return SimpleNamespace(
            risk=0.95,
            category="money_request",
            reason="要求紧急转账",
            transcript=transcript,
            latency_s=0.01,
        )

    return SimpleNamespace(
        backend=backend,
        asr=_FakeAsr(),
        analyze=analyze,
        _fallback=lambda reason, t0: SimpleNamespace(
            risk=0.5,
            category="unknown",
            reason=reason,
            transcript="",
            latency_s=0.0,
        ),
    )


def _run_semantic(monkeypatch, backend: str, free_bytes: float,
                  force: bool = False) -> object:
    called = {"empty_cache": False}
    monkeypatch.setattr(
        "fusion.transcript_cache.get", lambda audio_path: "")
    monkeypatch.setattr(
        "fusion.transcript_cache.put", lambda audio_path, text: None)
    monkeypatch.setattr(pipeline, "_FORCE_UNLOAD_ASR", force)
    monkeypatch.setattr(
        pipeline, "_free_vram_bytes", lambda: free_bytes)
    monkeypatch.setattr(
        pipeline, "_empty_cache", lambda: called.__setitem__(
            "empty_cache", True))

    pipe = pipeline.VeriCallPipeline.__new__(pipeline.VeriCallPipeline)
    pipe.sem = _fake_sem(backend)
    verdict = pipe._semantic_verdict("call.wav")
    return pipe.sem.asr._model, called["empty_cache"], verdict


def test_cloud_backend_keeps_asr_resident_even_when_vram_is_low(monkeypatch):
    model, empty_cache, verdict = _run_semantic(
        monkeypatch, "cloud", free_bytes=0.0)
    assert model is not None
    assert empty_cache is False
    assert verdict.label == "money_request"


def test_ollama_backend_unloads_asr_when_vram_is_low(monkeypatch):
    model, empty_cache, _ = _run_semantic(
        monkeypatch, "ollama", free_bytes=0.0)
    assert model is None
    assert empty_cache is True


def test_force_unload_overrides_cloud_residency(monkeypatch):
    model, empty_cache, _ = _run_semantic(
        monkeypatch, "cloud", free_bytes=float("inf"), force=True)
    assert model is None
    assert empty_cache is True
