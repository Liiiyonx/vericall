#!/usr/bin/env python
"""weekly_gate.py — 5.5 周回归一键门禁（2026-09-07，CPU-only，可进 CI）

每周五门禁的本地一键版：纯逻辑单测 + 隔离检查 + 关键产物校验 + 证据数字存在性。
全过输出 weekly_gate_report.md（PASS/FAIL 清单）。GPU/真机项不在此列（另测）。
用法：python -u scripts/ci/weekly_gate.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
R = []  # (项, 通过?, 备注)


def run(cmd: list[str], timeout: int = 600) -> tuple[int, str]:
    p = subprocess.run([PY, *cmd], cwd=ROOT, capture_output=True, text=True,
                       timeout=timeout, encoding="utf-8", errors="replace")
    return p.returncode, (p.stdout + p.stderr)[-600:]


def check(name: str, ok: bool, note: str = ""):
    R.append((name, ok, note))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {note}")


def main():
    t0 = time.time()
    print("== 周回归门禁 ==")
    # 1) 纯逻辑单测
    rc, out = run(["-m", "pytest", "tests/", "-q", "-p", "no:cacheprovider"], timeout=900)
    ok_p = ("passed" in out and "failed" not in out and "error" not in out.lower()) or rc == 0
    check("pytest 全绿", ok_p, out.strip().splitlines()[-1][:80] if out.strip() else f"rc={rc}")
    # 2) 隔离检查
    rc2, _ = run(["scripts/redteam_factory/check_isolation.py"], timeout=300)
    check("红队隔离检查", rc2 == 0)
    # 3) 关键产物存在且可解析
    jsons = {
        "声纹标定": "evaluation/voiceprint_calibration.json",
        "语义F1": "evaluation/semantic_f1.json",
        "语义校准": "evaluation/semantic_calibration.json",
    }
    for k, rel in jsons.items():
        p = ROOT / rel
        ok = p.is_file()
        if ok:
            try:
                json.loads(p.read_text(encoding="utf-8")); ok = True
            except Exception:
                ok = False
        check(f"产物 {k}", ok, rel)
    # 4) 证据数字
    vp = json.loads((ROOT / jsons["声纹标定"]).read_text(encoding="utf-8"))
    check("EER<2%", vp.get("eer", 1) < 0.02, f"EER={vp.get('eer')}")
    f1 = json.loads((ROOT / jsons["语义F1"]).read_text(encoding="utf-8"))
    check("语义二分类F1>=0.9", f1["binary_scam_vs_normal"]["f1"] >= 0.9,
          f"F1={f1['binary_scam_vs_normal']['f1']}")
    # 5) meta 规模
    import csv
    try:
        rows = list(csv.DictReader(open(ROOT / "data/redteam/factory/meta.csv", encoding="utf-8-sig")))
        check("红队 meta>=1.6万", len(rows) >= 16000, f"rows={len(rows)}")
    except Exception as e:
        check("红队 meta", False, str(e)[:60])

    fails = [r for r in R if not r[1]]
    lines = ["# 周回归门禁报告（5.5，2026-09-07）", "",
             f"- 执行耗时 {time.time()-t0:.0f}s · 结论 **{'PASS' if not fails else f'{len(fails)} 项 FAIL'}**", "",
             "| 项 | 结果 | 备注 |", "|---|---|---|"]
    for name, ok, note in R:
        lines.append(f"| {name} | {'✅' if ok else '❌'} | {note} |")
    lines += ["", "> GPU/真机项（流式 P90、回放实测）见 streaming/voiceprint 报告，本门禁为纯逻辑+资产门禁。"]
    out_p = ROOT / "evaluation" / "weekly_gate_report.md"
    out_p.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告 -> {out_p.relative_to(ROOT)}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
