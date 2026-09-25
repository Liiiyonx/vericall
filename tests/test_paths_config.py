# -*- coding: utf-8 -*-
"""运行期配置回归：外部 secrets、进程优先级和动态离线切换。"""
from __future__ import annotations

import paths


def test_dotenv_supports_export_and_utf8_bom(tmp_path):
    p = tmp_path / ".env"
    p.write_text("\ufeffexport A=1\nB='two words'\n# ignored\n",
                 encoding="utf-8")
    parsed = paths._load_dotenv(p)
    assert parsed == {"A": "1", "B": "two words"}


def test_process_env_beats_project_and_secret_values(monkeypatch):
    monkeypatch.setitem(paths._DOTENV, "ORDER_TEST", "project")
    monkeypatch.setitem(paths._SECRETS, "ORDER_TEST", "secret")
    monkeypatch.setenv("ORDER_TEST", "process")
    assert paths.env("ORDER_TEST") == "process"
    monkeypatch.delenv("ORDER_TEST")
    assert paths.env("ORDER_TEST") == "project"
    monkeypatch.delitem(paths._DOTENV, "ORDER_TEST")
    assert paths.env("ORDER_TEST") == "secret"


def test_offline_mode_is_read_at_call_time(monkeypatch):
    monkeypatch.setenv("VERICALL_OFFLINE", "0")
    assert paths.is_offline() is False
    monkeypatch.setenv("VERICALL_OFFLINE", "1")
    assert paths.is_offline() is True
