# -*- coding: utf-8 -*-
"""Migrate legacy history.jsonl into per-family shards.

Run while the VeriCall service is stopped:
    python scripts/migrate_history_shards.py --dry-run
    python scripts/migrate_history_shards.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from api import history_store  # noqa: E402
from paths import HIST_FILE  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history",
        type=Path,
        default=HIST_FILE,
        help=f"legacy history file (default: {HIST_FILE})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and count records without writing or archiving",
    )
    parser.add_argument(
        "--keep-legacy",
        action="store_true",
        help="leave the legacy file in place after migration",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = history_store.migrate_legacy_history(
        args.history,
        archive=not args.keep_legacy,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
