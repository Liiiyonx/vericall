# -*- coding: utf-8 -*-
"""Sensitive local-file permission and durability regressions."""
from __future__ import annotations

import os
import stat

import pytest

from private_fs import append_private_text, write_private_text


def test_private_text_write_is_atomic_and_keeps_content(tmp_path):
    target = tmp_path / "private" / "accounts.json"
    write_private_text(target, '{"first": true}')
    write_private_text(target, '{"second": true}')

    assert target.read_text(encoding="utf-8") == '{"second": true}'
    assert not list(target.parent.glob("*.tmp"))
    assert not list(target.parent.glob(".*.tmp"))


def test_private_text_append_creates_file(tmp_path):
    target = tmp_path / "private" / "events.jsonl"
    append_private_text(target, "one\n")
    append_private_text(target, "two\n")

    assert target.read_text(encoding="utf-8") == "one\ntwo\n"


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are not ACLs")
def test_private_files_and_directories_use_owner_only_modes(tmp_path):
    target = tmp_path / "private" / "events.jsonl"
    write_private_text(target, "one\n")
    append_private_text(target, "two\n")

    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700
