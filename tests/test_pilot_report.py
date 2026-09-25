# -*- coding: utf-8 -*-
"""试点报告统计口径回归。"""
from __future__ import annotations

from datetime import datetime

from api import history_store
from scripts.pilot_report import (
    load_records,
    load_ack_events,
    normalize_feedback,
    render,
    summarize,
)


def test_pilot_report_computes_feedback_fp_fn_and_ack_latency(tmp_path):
    records = [
        {
            "id": "r1", "household": "pilot-01",
            "time": "2026-09-01 10:00:00", "final": "block",
            "score": 0.9, "name": "可疑来电", "caller_number": "10086",
        },
        {
            "id": "r2", "household": "pilot-01",
            "time": "2026-09-01 10:10:00", "final": "allow",
            "score": 0.1, "name": "家人",
        },
        {
            "id": "r3", "household": "pilot-02",
            "time": "2026-09-02 09:00:00", "final": "caution",
            "score": 0.6, "name": "营销电话",
        },
    ]
    feedback = normalize_feedback({
        "records": [
            {"id": "r1", "human_final": "allow"},
            {"id": "r2", "human_final": "block"},
            {"id": "r3", "human_final": "caution"},
        ],
    })
    ack_path = tmp_path / "acks.jsonl"
    ack_at = datetime.strptime(
        "2026-09-01 10:00:10", "%Y-%m-%d %H:%M:%S").timestamp()
    ack_path.write_text(
        f'{{"alert_id":"r1","ack_at":{ack_at}}}\n', encoding="utf-8")
    ack_events = load_ack_events(ack_path)

    summary = summarize(records, feedback, ack_events)
    assert summary["overall"] == {
        "total": 3, "block": 1, "caution": 1, "allow": 1,
        "risk_allow": 0, "labeled": 3, "fp": 1, "fn": 1,
        "alert_created": 2, "acked": 1,
    }
    assert summary["ack_latency"]["median_s"] == 10.0
    assert summary["ack_latency"]["p95_s"] == 10.0

    report = render(summary, {"pilot-01": "第一周"})
    assert "误报：1 条；漏报：1 条" in report
    assert "中位数 10.00 s" in report
    assert "第一周" in report


def test_empty_pilot_report_is_explicitly_zero_not_synthetic():
    summary = summarize([])
    report = render(summary)
    assert summary["overall"]["total"] == 0
    assert "未开展的真实试点不会由脚本自动补齐" in report


def test_pilot_report_merges_legacy_and_family_shards(
        tmp_path, monkeypatch):
    monkeypatch.delenv("VERICALL_HISTORY_DIR", raising=False)
    legacy = tmp_path / "history.jsonl"
    legacy.write_text(
        '{"id":"legacy-1","household":"pilot-01","final":"allow"}\n',
        encoding="utf-8",
    )
    history_store.append_history({
        "id": "legacy-1",
        "family_id": "pilot-01",
        "household": "pilot-01",
        "final": "block",
    }, legacy)
    history_store.append_history({
        "id": "family-2",
        "family_id": "pilot-02",
        "household": "pilot-02",
        "final": "caution",
    }, legacy)

    records = load_records(legacy)

    assert [record["id"] for record in records] == ["legacy-1", "family-2"]
    assert records[0]["final"] == "allow"
