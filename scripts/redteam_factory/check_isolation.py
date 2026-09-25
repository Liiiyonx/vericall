# -*- coding: utf-8 -*-
"""check_isolation.py — 红队集隔离检查（详册 §7.3）

每次评估/训练前执行：确认 redteam/、in_the_wild/、partialspoof/ 目录
未被任何训练配置或训练脚本引用，防止评估集污染训练集导致指标虚高。

扫描范围：configs/、src/、scripts/（排除工厂自身与评测目录）。
命中即 exit 1 并列出 file:line，供 CI 周回归挂接。

用法：python scripts/redteam_factory/check_isolation.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

FORBIDDEN = re.compile(r"redteam|in_the_wild|partialspoof", re.I)
SCAN_DIRS = ["configs", "src", "scripts"]
# 这些路径里的命中是合法的（评测/工厂/路径常量定义本身）
ALLOW = re.compile(
    r"(evaluation[\\/]|scripts[\\/]redteam_factory[\\/]|scripts[\\/]scam_corpus[\\/]"
    r"|scripts[\\/]eval[\\/]|check_isolation\.py|src[\\/]paths\.py|src[\\/]fusion[\\/]xlsr_cn_channel\.py|__pycache__)", re.I)
SCAN_EXTS = {".py", ".conf", ".json", ".yaml", ".yml", ".toml", ".sh", ".bat"}
# 训练语义：仅在含训练关键词的文件中命中才算违规（README/注释类说明不算）
TRAIN_HINT = re.compile(r"train|launch|protocol|dataset|data_dir|train_config", re.I)


def main():
    violations = []
    for d in SCAN_DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.suffix not in SCAN_EXTS or ALLOW.search(str(p)):
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if not TRAIN_HINT.search(p.name) and "train" not in text[:2000].lower():
                # 非训练相关文件跳过（降低噪音）
                if p.suffix == ".py" and "train" not in text.lower():
                    continue
            for ln, line in enumerate(text.splitlines(), 1):
                if FORBIDDEN.search(line) and not line.strip().startswith(("#", "//")):
                    violations.append(f"{p.relative_to(ROOT)}:{ln}: {line.strip()[:100]}")

    if violations:
        print("❌ 隔离违规：训练链路引用了隔离目录")
        for v in violations:
            print("  " + v)
        sys.exit(1)
    print("✅ 隔离检查通过：训练链路未引用 redteam/ in_the_wild/ partialspoof/")
    sys.exit(0)


if __name__ == "__main__":
    main()
