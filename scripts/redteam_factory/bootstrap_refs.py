#!/usr/bin/env python
"""bootstrap_refs.py — 为 GPT-SoVITS 等克隆引擎自举说话人参考音（2026-09-05）

背景：志愿者参考音未到；用现有 Edge-TTS 合成清音（真·不同音色）充当克隆参考，
生成 data/redteam/<dialect>/ref_boot_*.wav + data/redteam/refs.json（prompt_text 映射）。
这样 GPT-SoVITS 克隆线无需等待志愿者即可开跑（音色=合成音色，作攻击者身份即可）。

用法：python bootstrap_refs.py [--per-dialect N]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "data" / "redteam" / "factory" / "meta.csv"
CORPUS = ROOT / "data" / "scam_corpus" / "corpus_v0.1.jsonl"
REF_ROOT = ROOT / "data" / "redteam"
REFS_JSON = REF_ROOT / "refs.json"
DIALECTS = ["mandarin", "minnan", "sichuan", "cantonese", "dongbei", "henan"]


def load_corpus_text() -> dict:
    d = {}
    if CORPUS.exists():
        for line in open(CORPUS, encoding="utf-8"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                d[r["id"]] = r["text"]
            except Exception:
                pass
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-dialect", type=int, default=3)
    args = ap.parse_args()

    texts = load_corpus_text()
    cands: dict[str, list[tuple[str, Path, str, str]]] = {d: [] for d in DIALECTS}
    if META.exists():
        import csv
        with open(META, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("engine") != "edgetts" or row.get("channel") != "clean":
                    continue
                rel = row["path"]
                dialect = row.get("dialect", "")
                if dialect not in cands:
                    continue
                fpath = ROOT / rel
                if not fpath.exists():
                    continue
                sid = row.get("script_id", "")
                cands[dialect].append((fpath.name, fpath, sid, texts.get(sid, "")))

    REF_ROOT.mkdir(parents=True, exist_ok=True)
    refs = {}
    made = 0
    for dialect, items in cands.items():
        dref = REF_ROOT / dialect
        dref.mkdir(parents=True, exist_ok=True)
        # 多音色去重取前 N
        seen_voices = set()
        picked = 0
        for name, src, sid, text in items:
            voice = name.split("_")[2] if "_" in name else name  # edgetts_<dialect>_<voice>_<sid>
            if voice in seen_voices:
                continue
            seen_voices.add(voice)
            if not (2 <= len(text) <= 200):
                continue
            out = dref / f"ref_boot_{name}"
            shutil.copy2(src, out)
            refs[str(out).replace("\\", "/")] = {
                "prompt_text": text, "script_id": sid, "dialect": dialect,
                "source": "edgetts-bootstrap", "voice": voice,
            }
            made += 1
            picked += 1
            if picked >= args.per_dialect:
                break

    REFS_JSON.write_text(json.dumps(refs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"自举参考音 {made} 个 -> {REF_ROOT}/<dialect>/ref_boot_*.wav")
    print(f"映射表 -> {REFS_JSON} ({len(refs)} 条)")
    if not made:
        print("⚠ 无可用 edgetts clean 产物（先跑 edgetts 线再自举）")


if __name__ == "__main__":
    main()
