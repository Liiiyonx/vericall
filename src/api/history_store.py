# -*- coding: utf-8 -*-
"""检测历史的家庭分片存储。

新记录写入 ``DATA_DIR/history/<family>-<hash>.jsonl``，避免多个家庭
共用同一物理文件；旧 ``history.jsonl`` 继续只读兼容，便于平滑迁移。
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import uuid
from pathlib import Path

from paths import DEFAULT_FAMILY_ID, HIST_FILE, ROOT, env
from private_fs import append_private_text

_history_lock = threading.Lock()


def next_id() -> str:
    """返回与现有检测记录一致的短 ID。"""
    return uuid.uuid4().hex[:10]


def _history_root(path: str | Path) -> Path:
    """返回家庭分片目录；生产环境可通过环境变量显式覆盖。"""
    configured = (env("VERICALL_HISTORY_DIR", "") or "").strip()
    if configured:
        configured_path = Path(configured).expanduser()
        if not configured_path.is_absolute():
            configured_path = ROOT / configured_path
        return configured_path.resolve()
    return Path(path).expanduser().resolve().parent / "history"


def history_path_for(family_id: str,
                     path: str | Path | None = None) -> Path:
    """返回家庭专属历史文件路径，文件名不直接包含可穿越的家庭字符串。"""
    base = Path(path) if path is not None else HIST_FILE
    family = str(family_id or DEFAULT_FAMILY_ID).strip() or DEFAULT_FAMILY_ID
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", family).strip("._-")[:32]
    slug = slug or "family"
    digest = hashlib.sha256(family.encode("utf-8")).hexdigest()[:12]
    return _history_root(base) / f"{slug}-{digest}.jsonl"


def _record_matches_family(rec: dict, family_id: str,
                           default_family_id: str) -> bool:
    record_family = str(rec.get("family_id") or "").strip()
    if family_id == default_family_id:
        return not record_family or record_family == family_id
    return record_family == family_id


def iter_history(path: str | Path | None = None,
                 family_id: str | None = None,
                 default_family_id: str = DEFAULT_FAMILY_ID):
    """按家庭读取旧文件与分片，并对重复 ID 去重。"""
    base = Path(path) if path is not None else HIST_FILE
    sources = [base]
    requested_family = str(family_id or "").strip()
    if requested_family:
        shard = history_path_for(requested_family, base)
        if shard != base:
            sources.append(shard)

    seen_ids: set[str] = set()
    for source in sources:
        if not source.is_file():
            continue
        try:
            handle = source.open(encoding="utf-8")
        except OSError:
            continue
        with handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                if requested_family:
                    if not _record_matches_family(
                            rec, requested_family, default_family_id):
                        continue
                    rec["family_id"] = requested_family
                record_id = str(rec.get("id") or "").strip()
                if record_id and record_id in seen_ids:
                    continue
                if record_id:
                    seen_ids.add(record_id)
                yield rec


def iter_all_history(path: str | Path | None = None):
    """读取旧文件和全部家庭分片，供试点报告、审计与迁移工具消费。"""
    base = Path(path) if path is not None else HIST_FILE
    base = base.expanduser().resolve()
    sources = [base]
    root = _history_root(base)
    if root.is_dir():
        sources.extend(
            shard for shard in sorted(root.glob("*.jsonl"))
            if shard.resolve() != base
        )

    seen_ids: set[str] = set()
    for source in sources:
        if not source.is_file():
            continue
        try:
            handle = source.open(encoding="utf-8")
        except OSError:
            continue
        with handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                record_id = str(rec.get("id") or "").strip()
                if record_id and record_id in seen_ids:
                    continue
                if record_id:
                    seen_ids.add(record_id)
                yield rec


def append_history(rec: dict, path: str | Path | None = None,
                   family_id: str | None = None) -> Path:
    """原子地追加一条历史记录。

    ``path`` 是旧 ``history.jsonl`` 的基准路径，用于兼容现有测试和部署。
    当记录带 ``family_id`` 时自动写入家庭分片，并返回实际落盘路径；
    无家庭 ID 的旧调用仍写基准文件。``O_APPEND`` 与进程内锁共同避免
    HTTP / WebSocket 并发记录互相穿插。
    """
    if not isinstance(rec, dict):
        raise TypeError("history record must be a dict")
    base = Path(path) if path is not None else HIST_FILE
    record = dict(rec)
    effective_family = str(
        family_id or record.get("family_id") or "").strip()
    if effective_family:
        record["family_id"] = effective_family
        target = history_path_for(effective_family, base)
    else:
        target = base

    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _history_lock:
        append_private_text(target, line)
    return target


def migrate_legacy_history(
        path: str | Path | None = None,
        *,
        default_family_id: str = DEFAULT_FAMILY_ID,
        archive: bool = True,
        dry_run: bool = False) -> dict:
    """把旧 ``history.jsonl`` 迁入家庭分片，并可选归档旧文件。

    迁移按记录 ID 去重，已存在的分片记录不会被重复写入。建议在服务
    停止时执行；函数不会删除数据，归档失败时保留旧文件供重试。
    """
    base = Path(path) if path is not None else HIST_FILE
    result = {
        "legacy_path": str(base),
        "migrated": 0,
        "skipped": 0,
        "invalid": 0,
        "archived_to": None,
        "dry_run": bool(dry_run),
    }
    if not base.is_file():
        return result

    shard_dir = _history_root(base)
    existing_ids: set[str] = set()
    if shard_dir.is_dir():
        for shard in shard_dir.glob("*.jsonl"):
            try:
                for rec in iter_history(shard, family_id=None):
                    record_id = str(rec.get("id") or "").strip()
                    if record_id:
                        existing_ids.add(record_id)
            except OSError:
                continue

    pending: list[dict] = []
    seen_ids: set[str] = set()
    with base.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                result["invalid"] += 1
                continue
            if not isinstance(rec, dict):
                result["invalid"] += 1
                continue
            record_id = str(rec.get("id") or "").strip()
            if record_id and (
                    record_id in existing_ids or record_id in seen_ids):
                result["skipped"] += 1
                continue
            if record_id:
                seen_ids.add(record_id)
            rec = dict(rec)
            rec["family_id"] = str(
                rec.get("family_id") or default_family_id).strip()
            pending.append(rec)

    result["migrated"] = len(pending)
    if dry_run or not pending:
        return result

    for rec in pending:
        append_history(rec, base)

    if archive:
        suffix = time.strftime("%Y%m%d_%H%M%S")
        archive_path = base.with_name(f"{base.name}.migrated-{suffix}")
        counter = 1
        while archive_path.exists():
            archive_path = base.with_name(
                f"{base.name}.migrated-{suffix}-{counter}")
            counter += 1
        base.replace(archive_path)
        result["archived_to"] = str(archive_path)
    return result
