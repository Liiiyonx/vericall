# -*- coding: utf-8 -*-
"""test_number_channel.py — 通道⓪ 号码先验层单测（2026-09-07）

覆盖：号码归一化 / 黑名单精确命中（block 短路）/ 启发式（caution 继续三通道）
/ 无号码 / 未命中；融合器不会因号段提示产生偏见式硬拦。
"""
from __future__ import annotations

import tempfile
import threading
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
    nc.family_blocklist_path = Path(td) / "family_blocklist.jsonl"
    nc._family_by_number = {}
    nc._lock = threading.Lock()
    nc._mtime_key = None
    nc._family_mtime_key = None
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
        assert v.action == "block" and v.score == 1.0
        v2 = nc.check("13800138000")
        assert v2.matched and v2.source == "blocklist"

    def test_heuristic_hit(self):
        nc = _nc_with([], heuristic=True)
        v = nc.check("17012345678")
        assert v.matched and v.source == "heuristic:virtual_prefix"
        assert v.action == "caution" and v.score == 0.6
        v2 = nc.check("+85251234567")
        assert v2.matched and v2.source == "heuristic:overseas_cc"
        assert v2.action == "caution"

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

    def test_heuristic_caution_continues_and_is_preserved(self):
        nc = _nc_with([], heuristic=True)
        number = nc.check("17012345678")
        orch = FusionOrchestrator()
        r = orch.decide(
            ChannelVerdict("acoustic", 0.05, "bonafide", "真声", 1.0),
            ChannelVerdict("voiceprint", 0.1, "match", "家人", 1.0),
            ChannelVerdict("semantic", 0.1, "normal", "正常", 1.0),
            number=number)
        assert r.final == "caution"
        assert r.channels[0]["name"] == "number"
        assert r.channels[0]["action"] == "caution"
        assert len(r.channels) == 4
        assert "继续三通道" in r.rationale


class TestReport:
    """众包上报（2026-09-11）：report() 写库并即时生效 / 重载仍然命中。"""

    def test_report_writes_and_hits(self, tmp_path):
        p = tmp_path / "bl.txt"
        p.write_text("# 注释行\n", encoding="utf-8")
        nc = NumberChannel(blocklist_path=p, heuristic=False)
        assert nc.report("17012345678", remark="子女端举报")
        v = nc.check("17012345678")
        assert v.matched and v.source == "blocklist"
        # 模拟服务重启：新实例重新 load 文件，追加行仍命中
        nc2 = NumberChannel(blocklist_path=p, heuristic=False)
        assert nc2.check("17012345678").matched

    def test_report_invalid_number(self, tmp_path):
        nc = NumberChannel(blocklist_path=tmp_path / "x.txt", heuristic=False)
        assert not nc.report(None)
        assert not nc.report("*")
        assert not nc.report("未知")

    def test_external_file_change_is_hot_reloaded(self, tmp_path):
        p = tmp_path / "bl.txt"
        p.write_text("17000000001\t旧条目\n", encoding="utf-8")
        nc = NumberChannel(blocklist_path=p, heuristic=False)
        assert not nc.check("17000000002").matched

        p.write_text(
            "17000000001\t旧条目\n17000000002\t另一实例举报\n",
            encoding="utf-8",
        )
        v = nc.check("17000000002")
        assert v.matched and v.source == "blocklist"
        assert "另一实例举报" in v.detail

    def test_report_updates_mtime_without_redundant_reload(
            self, tmp_path, monkeypatch):
        p = tmp_path / "bl.txt"
        p.write_text("# empty\n", encoding="utf-8")
        nc = NumberChannel(blocklist_path=p, heuristic=False)
        reads = 0
        original = nc._read_blocklist

        def counting_read():
            nonlocal reads
            reads += 1
            return original()

        monkeypatch.setattr(nc, "_read_blocklist", counting_read)
        assert nc.report("17012345678", remark="举报")
        assert reads == 1
        assert nc.check("17012345678").matched
        assert reads == 1


class TestFamilyReportIsolation:
    def test_family_report_is_visible_only_to_own_family(
            self, tmp_path, monkeypatch):
        public = tmp_path / "public.txt"
        public.write_text("# 公开库为空\n", encoding="utf-8")
        private = tmp_path / "family_blocklist.jsonl"
        monkeypatch.setenv("VERICALL_FAMILY_NUMBER_BLOCKLIST", str(private))
        nc = NumberChannel(blocklist_path=public, heuristic=False)

        assert nc.report(
            "17012345678", remark="家庭 A 举报", family_id="family-a")
        hit = nc.check("17012345678", family_id="family-a")
        assert hit.matched and hit.source == "family_blocklist"
        assert hit.action == "block" and "家庭 A 举报" in hit.detail
        assert not nc.check("17012345678", family_id="family-b").matched
        assert "17012345678" not in public.read_text(encoding="utf-8")

        restarted = NumberChannel(blocklist_path=public, heuristic=False)
        assert restarted.check(
            "17012345678", family_id="family-a").source == "family_blocklist"
        assert not restarted.check(
            "17012345678", family_id="family-b").matched
