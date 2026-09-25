# -*- coding: utf-8 -*-
"""端到端基准工具的纯函数回归。"""
from __future__ import annotations

from scripts.bench_end_to_end import (
    _multipart_body,
    percentile,
    summarize_latencies,
)


def test_percentile_and_summary_are_deterministic():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    summary = summarize_latencies([1.0, 2.0, 3.0, 4.0])
    assert summary["n"] == 4
    assert summary["p50_s"] == 2.5
    assert summary["mean_s"] == 2.5
    assert summary["max_s"] == 4.0


def test_multipart_demo_and_upload_fields(tmp_path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFFdemo")
    body, boundary = _multipart_body(audio, "A", "10086")
    assert f"--{boundary}".encode("ascii") in body
    assert b'name="file"; filename="sample.wav"' in body
    assert b'name="demo"' in body
    assert b'name="caller_number"' in body
    assert b"RIFFdemo" in body


def test_multipart_includes_household():
    body, _ = _multipart_body(None, "B", "", "pilot-07")
    assert b'name="household"' in body
    assert b"pilot-07" in body
