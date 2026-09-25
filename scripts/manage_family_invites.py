# -*- coding: utf-8 -*-
"""Manage pilot family invitation codes.

Examples:
    python scripts/manage_family_invites.py create --family-id pilot-01
    python scripts/manage_family_invites.py create --family-id pilot-02 --uses 8
    python scripts/manage_family_invites.py list
    python scripts/manage_family_invites.py disable FAMILY_CODE
"""
from __future__ import annotations

import argparse
import json
import secrets
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import api.family_api as family  # noqa: E402


def create(args: argparse.Namespace) -> int:
    if args.uses < 1:
        raise SystemExit("--uses must be >= 1")
    if args.days < 1:
        raise SystemExit("--days must be >= 1")
    code = (args.code or f"{args.family_id}-{secrets.token_hex(3)}").strip()
    if not code or len(code) < 6:
        raise SystemExit("--code must contain at least 6 characters")
    data = family._load_invites()
    if code in data["invites"]:
        raise SystemExit(f"invitation already exists: {code}")
    data["invites"][code] = {
        "family_id": args.family_id,
        "max_uses": args.uses,
        "used": 0,
        "expires_at": time.time() + args.days * 86400,
        "active": True,
        "note": args.note,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    family._save_invites(data)
    print(json.dumps({"invitation": code, **data["invites"][code]},
                     ensure_ascii=False, indent=2))
    return 0


def list_invites(_args: argparse.Namespace) -> int:
    data = family._load_invites()
    if not data["invites"]:
        print("No invitations.")
        return 0
    for code, item in sorted(data["invites"].items()):
        expired = (
            item.get("expires_at") is not None
            and time.time() >= float(item["expires_at"])
        )
        status = "disabled" if item.get("active", True) is not True else (
            "expired" if expired else "active")
        print(
            f"{code}\tfamily={item.get('family_id')}\t"
            f"used={item.get('used', 0)}/{item.get('max_uses', 1)}\t"
            f"status={status}")
    return 0


def disable(args: argparse.Namespace) -> int:
    data = family._load_invites()
    item = data["invites"].get(args.code)
    if not isinstance(item, dict):
        raise SystemExit(f"invitation not found: {args.code}")
    item["active"] = False
    item["disabled_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    family._save_invites(data)
    print(f"disabled: {args.code}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    create_parser = sub.add_parser("create")
    create_parser.add_argument("--family-id", required=True)
    create_parser.add_argument("--code", default="")
    create_parser.add_argument("--uses", type=int, default=5)
    create_parser.add_argument("--days", type=int, default=30)
    create_parser.add_argument("--note", default="")
    create_parser.set_defaults(func=create)

    list_parser = sub.add_parser("list")
    list_parser.set_defaults(func=list_invites)

    disable_parser = sub.add_parser("disable")
    disable_parser.add_argument("code")
    disable_parser.set_defaults(func=disable)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
