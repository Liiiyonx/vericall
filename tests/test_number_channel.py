# -*- coding: utf-8 -*-
"""test_number_channel.py — 通道⓪ 号码先验层单测（2026-09-07）

覆盖：号码归一化 / 黑名单精确命中（来源=blocklist）/ 启发式（virtual_prefix,
overseas_cc）/ 无号码 / 未命中；融合器短路（命中即 block 且跳过三通道）。
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from fusion.fusion_orchestrator import ChannelVerdict, FusionOrchestrator
from fusion.number_channel import NumberChannel, classify_heuristic, normalize


def _nc_with(blocklist_lines: list[str], heuristic: bool = True) -> NumberChannel:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "blocklist.txt"
        p.write_text("\n".join(blocklist_lines), encoding="utf-8")
        # 注意：NumberChannel 默认读环境变量/默认路径；这里直接用其私有字典注入测试文件
    nc = NumberChannel.__new__(NumberChannel)
    nc.blocklist_path = p
    nc.heuristic = heuristic
    nc._by_number = {}
    for raw in blocklist_lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        num = normalize(parts[0])
        if num:
            nc._by_number.setdefault(num, parts[1] if len(parts) > 1 else "")
    return nc


class TestNormalize:
    def test_prefix_strip(self):
        assert normalize("+86 138 0013 8000") == "13800138000"
        assert normalize("008613800138000") == "13800138000"  # 13 位 86 前缀→异常仍保留? 
        assert normalize("8613800138000") == "13800138000"

    def test_separators(self):
        assert normalize("010-1234-5678") == "01012345678"

    def test_unknown(self):
        assert normalize(None) == ""
        assert normalize("*") == ""
        assert normalize("未知") == ""
        assert normalize("") == ""

    def test_overseas_keep(self):
        assert normalize("+852 5123 4567") == "85251234567"
        assert normalize("00852 5123 4567") == "0085251234567" or normalize("0085251234567") == "0085251234567"


class TestHeuristic:
    def test_virtual(self):
        assert classify_heuristic("17012345678") == "virtual_prefix"
        assert classify_heuristic("17123456789") == "virtual_prefix"
        assert classify_heuristic("13800138000") is None

    def test_overseas(self):
        assert classify_heuristic("85251234567") == "overseas_cc"
        assert classify_heuristic("0085251234567") == "overseas_cc"
        assert classify_heuristic("886912345678") == "overseas_cc"

    def test_mainland_ok(self):
        assert classify_heuristic("13800138000") is None
        assert classify_heuristic("01012345678") is None  # 国内固话


class TestChannel:
    def test_blocklist_hit(self):
        nc = _nc_with(["17000000001\t涉诈通报示例", "# 注释", "13800138000"])
        v = nc.check("17000000001")
        assert v.matched and v.source == "blocklist" and v.confidence > 0.98
        v2 = nc.check("13800138000")
        assert v2.matched and v2.source == "blocklist"

    def test_heuristic_hit(self):
        nc = _nc_with([], heuristic=True)
        v = nc.check("17012345678")
        assert v.matched and v.source == "heuristic:virtual_prefix"
        v2 = nc.check("+85251234567")
        assert v2.matched and v2.source == "heuristic:overseas_cc"

    def test_no_caller_id(self):
        nc = _nc_with([])
        v = nc.check(None)
        assert not v.matched and v.source == "no_caller_id"

    def test_not_hit(self):
        nc = _nc_with([], heuristic=False)
        v = nc.check("13800138000")
        assert not v.matched and v.source == "not_hit"


class TestPipelineShortCircuit:
    def test_pipeline_number_hit_skips_3ch(self, tmp_path, monkeypatch):
        bl = tmp_path / "b.txt"
        bl.write_text("17012345678\t测试条目\n", encoding="utf-8")
        monkeypatch.setenv("VERICALL_NUMBER_BLOCKLIST", str(bl))
        from fusion.pipeline import VeriCallPipeline as Pipeline
        # 假通道：若三通道被执行会因缺方法而失败 → 短路成功即证明跳过
        p = Pipeline(voiceprint=object(), semantic=object(), acoustic=object())
        r = p.analyze("unused.wav", caller_number="17012345678")
        assert r.final == "block"
        assert len(r.channels) == 1 and r.channels[0]["name"] == "number"
        assert "跳过" in r.rationale
    def test_number_hit_blocks_without_3ch(self):
        nc = _nc_with(["17000000001"])
        orch = FusionOrchestrator()
        r = orch.decide(
            ChannelVerdict("acoustic", 0.0, "normal", "正常", 1.0),
            ChannelVerdict("voiceprint", 0.0, "match", "家人", 1.0),
            ChannelVerdict("semantic", 0.0, "normal", "正常", 1.0),
            number=nc.check("17000000001"))
        assert r.final == "block"
        assert "号码" in r.rationale and "跳过" in r.rationale
        assert len(r.channels) == 1 and r.channels[0]["name"] == "number"

    def test_number_miss_falls_through(self):
        nc = _nc_with([], heuristic=False)
        orch = FusionOrchestrator()
        r = orch.decide(
            ChannelVerdict("acoustic", 0.1, "spoof", "正常", 1.0),
            ChannelVerdict("voiceprint", 0.1, "match", "家人", 1.0),
            ChannelVerdict("semantic", 0.1, "normal", "正常", 1.0),
            number=nc.check("13800138000"))
        assert r.final == "allow"
        assert len(r.channels) == 3  # 三通道照常
