# -*- coding: utf-8 -*-
"""家庭历史分片与旧文件迁移回归。"""
from __future__ import annotations

import json

from api import history_store


def test_history_is_physically_split_by_family(tmp_path, monkeypatch):
    monkeypatch.delenv("VERICALL_HISTORY_DIR", raising=False)
    legacy = tmp_path / "history.jsonl"
    legacy.write_text(json.dumps({
        "id": "legacy-demo",
        "final": "caution",
        "name": "旧记录",
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    family_a_path = history_store.append_history({
        "id": "a1",
        "family_id": "family-a",
        "final": "block",
    }, legacy)
    family_b_path = history_store.append_history({
        "id": "b1",
        "family_id": "family-b",
        "final": "allow",
    }, legacy)

    assert family_a_path != family_b_path
    assert family_a_path.parent == (tmp_path / "history").resolve()
    assert family_b_path.parent == family_a_path.parent
    assert legacy.read_text(encoding="utf-8").count("\n") == 1

    a_records = list(history_store.iter_history(
        legacy, family_id="family-a"))
    b_records = list(history_store.iter_history(
        legacy, family_id="family-b"))
    demo_records = list(history_store.iter_history(
        legacy, family_id="demo"))

    assert [record["id"] for record in a_records] == ["a1"]
    assert [record["id"] for record in b_records] == ["b1"]
    assert [record["id"] for record in demo_records] == ["legacy-demo"]


def test_history_shard_filename_cannot_escape_root(tmp_path, monkeypatch):
    monkeypatch.delenv("VERICALL_HISTORY_DIR", raising=False)
    shard = history_store.history_path_for(
        "../../outside", tmp_path / "history.jsonl")

    assert shard.parent == (tmp_path / "history").resolve()
    assert ".." not in shard.parts
    assert shard.name.endswith(".jsonl")


def test_legacy_history_migration_is_idempotent_and_archives(
        tmp_path, monkeypatch):
    monkeypatch.delenv("VERICALL_HISTORY_DIR", raising=False)
    legacy = tmp_path / "history.jsonl"
    history_store.append_history({
        "id": "a1",
        "family_id": "family-a",
        "final": "block",
    }, legacy)
    legacy.write_text("\n".join([
        json.dumps({"id": "a1", "family_id": "family-a"},
                   ensure_ascii=False),
        json.dumps({"id": "b1", "family_id": "family-b"},
                   ensure_ascii=False),
        json.dumps({"id": "demo1", "final": "caution"},
                   ensure_ascii=False),
        "{not-json}",
    ]) + "\n", encoding="utf-8")

    result = history_store.migrate_legacy_history(legacy)

    assert result["migrated"] == 2
    assert result["skipped"] == 1
    assert result["invalid"] == 1
    assert result["archived_to"]
    assert not legacy.exists()
    assert [record["id"] for record in history_store.iter_history(
        legacy, family_id="family-a")] == ["a1"]
    assert [record["id"] for record in history_store.iter_history(
        legacy, family_id="family-b")] == ["b1"]
    assert [record["id"] for record in history_store.iter_history(
        legacy, family_id="demo")] == ["demo1"]
