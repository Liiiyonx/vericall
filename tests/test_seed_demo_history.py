from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_seed_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "seed_demo_history.py"
    spec = importlib.util.spec_from_file_location("seed_demo_history", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _read_shard(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_seed_demo_history_is_complete_and_idempotent(tmp_path):
    seed = _load_seed_module()
    first = seed.seed_demo_history(
        family_id="demo", history_dir=tmp_path / "history")
    second = seed.seed_demo_history(
        family_id="demo", history_dir=tmp_path / "history")
    records = _read_shard(Path(first["history_path"]))

    assert first["inserted"] == 8
    assert first["total"] == 8
    assert second["inserted"] == 0
    assert second["total"] == 8
    assert len(records) == 8
    assert {record["source"] for record in records} == {"ai-demo"}
    assert sum(record["final"] == "block" for record in records) == 3
    assert sum(record["final"] == "caution" for record in records) == 2
    assert sum(record["final"] == "allow" for record in records) == 3
    assert all(
        {channel["name"] for channel in record["channels"]}
        == {"acoustic", "voiceprint", "semantic"}
        for record in records
    )


def test_seed_demo_history_reset_replaces_only_seed_records(tmp_path):
    seed = _load_seed_module()
    history_dir = tmp_path / "history"
    first = seed.seed_demo_history(
        family_id="demo", history_dir=history_dir)
    shard = Path(first["history_path"])
    with shard.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "id": "real-record",
            "source": "upload",
            "family_id": "demo",
            "final": "allow",
        }, ensure_ascii=False) + "\n")

    result = seed.seed_demo_history(
        family_id="demo", history_dir=history_dir, reset=True)
    records = _read_shard(shard)

    assert result["removed"] == 8
    assert result["inserted"] == 8
    assert result["total"] == 8
    assert {record["id"] for record in records} == set(seed.SEED_IDS) | {
        "real-record"}
