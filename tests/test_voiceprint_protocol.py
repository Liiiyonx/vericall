# -*- coding: utf-8 -*-
"""test_voiceprint_protocol.py — B4 协议单测：双信道注册 + challenge-response（2026-09-07）

纯逻辑（无 funasr/GPU）：嵌入函数与落盘用 monkeypatch 桩。
"""
from __future__ import annotations

import numpy as np
import pytest

from fusion.challenge_response import VOICE_THR, assess, gen_challenge
from fusion.voiceprint_channel import VoiceprintChannel, MATCH_THRESHOLD


class TestChallengeResponse:
    def test_gen_len(self):
        c = gen_challenge()
        assert len(c) == 4 and c.isdigit()

    def test_pass(self):
        c = "1357"
        r = assess(c, c, 0.61)
        assert r["pass_ok"] and r["verdict"] == "ok"

    def test_digits_mismatch(self):
        r = assess("1357", "2468", 0.61)
        assert not r["pass_ok"]

    def test_voice_mismatch(self):
        r = assess("1357", "1357", 0.30)
        assert not r["pass_ok"]
        assert "音色" in r["reasons"][1]

    def test_thr_source(self):
        assert VOICE_THR == MATCH_THRESHOLD  # 与声纹登记阈值同源


class TestEnrollDualChannel:
    @staticmethod
    def _channel(monkeypatch, embeddings: dict, save_hook=None):
        vc = VoiceprintChannel.__new__(VoiceprintChannel)
        vc.profiles = {}

        def fake_embed(path):
            return np.asarray(embeddings[path], dtype=np.float32) / np.linalg.norm(
                embeddings[path])

        calls = {}

        def fake_save(name, vec, source_file=""):
            calls["vec"] = vec.copy()
            calls["src"] = source_file

        monkeypatch.setattr(vc, "_embed", fake_embed)
        monkeypatch.setattr(vc, "_save_profile", fake_save)
        return vc, calls

    def test_pass_saves(self, monkeypatch):
        # 同一人两条信道，夹角很小 → 两样本对平均模板均高分
        q = np.array([1.0, 0.0, 0.0])
        s = np.array([0.95, np.sqrt(1 - 0.95 ** 2), 0.0])
        vc, calls = self._channel(monkeypatch, {"q.wav": q, "s.wav": s})
        vc.enroll_dual_channel("妈", "q.wav", "s.wav")
        assert calls.get("src") == "dual_channel"
        assert calls["vec"] is not None

    def test_bad_sample_rejected(self, monkeypatch):
        # 第二条信道完全不同人 → 对模板相似度必然低于阈值 → 抛错且不落盘
        q = np.array([1.0, 0.0, 0.0])
        other = np.array([0.0, 1.0, 0.0])
        vc, calls = self._channel(monkeypatch, {"q.wav": q, "s.wav": other})
        with pytest.raises(ValueError, match="质量不足"):
            vc.enroll_dual_channel("妈", "q.wav", "s.wav")
        assert "vec" not in calls  # 未落盘
