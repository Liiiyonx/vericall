# -*- coding: utf-8 -*-
"""round2_degrade_matrix.py — 红蓝第 2 轮 · 对抗退化矩阵（2026-09-06）

任务书/详细执行计划 1.1②：对 34 条 wide 极难样本（redblue_round2_attack_surface.md）
× {phone8k, mp3_16k, amr, noise} 4 档退化 = 136 条退化变体，
检验"退化后是否更隐蔽/可检"，为三通道联合复测（1.1④）供料。

输入：evaluation/round2_hard34.json（34 条极难样本清单）
产出：
  data/redteam/factory/round2_degrade/{preset}/{stem}.{preset}.wav
  data/redteam/factory/round2_degrade/manifest.json（全量清单）
  evaluation/round2_degrade_matrix.md / .json（矩阵报告）

复现：
  python -u evaluation/round2_degrade_matrix.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from degrade_audio import PRESETS, degrade_file  # noqa: E402

HARD_LIST = ROOT / "evaluation" / "round2_hard34.json"
OUT_ROOT = ROOT / "data" / "redteam" / "factory" / "round2_degrade"
OUT_MD = ROOT / "evaluation" / "round2_degrade_matrix.md"
OUT_JSON = ROOT / "evaluation" / "round2_degrade_matrix.json"

PRESET_CN = {
    "phone8k": "8k 电话信道(μ-law+削顶)",
    "mp3_16k": "MP3 32kbps 编解码",
    "amr": "AMR-NB 12.2kbps 移动信道",
    "noise": "8k + 粉噪 SNR15dB(免提外放)",
}


def main():
    hard = json.loads(HARD_LIST.read_text(encoding="utf-8"))
    print(f"极难样本: {len(hard)} 条 × 4 预设 = {len(hard)*4} 变体")
    ok, skipped, fail = [], 0, []
    t0 = time.time()
    for item in hard:
        src = ROOT / item["path"]
        if not src.exists():
            fail.append({"src": item["path"], "err": "缺失"})
            continue
        for preset in sorted(PRESETS):
            out_dir = OUT_ROOT / preset
            out_dir.mkdir(parents=True, exist_ok=True)
            target = out_dir / f"{src.stem}.{preset}.wav"
            if target.exists() and target.stat().st_size > 1000:
                skipped += 1
                ok.append({**item, "preset": preset, "deg_path": str(target.relative_to(ROOT))})
                continue
            p = degrade_file(src, out_dir, preset, sr=16000)
            if p is None:
                fail.append({"src": str(src), "err": f"{preset} 生成失败(可能缺 ffmpeg)"})
                continue
            ok.append({**item, "preset": preset, "deg_path": str(p.relative_to(ROOT))})
        print(f"  [{len(ok)+skipped+len(fail)}/{(len(hard)*4)}] {src.name}", flush=True)

    # ---- 时长/采样率校验（soundfile：mp3_16k 输出 .mp3、amr 输出 8k，wave 模块读不了）
    import soundfile as sf
    bad = []
    for rec in ok:
        try:
            wav, sr = sf.read(str(ROOT / rec["deg_path"]))
            dur = len(wav) / sr
            rec["deg_sr"] = sr
            rec["deg_dur_s"] = round(dur, 2)
            if dur < 0.5:
                bad.append(rec["deg_path"])
        except Exception as e:  # noqa: BLE001
            bad.append(f"{rec['deg_path']}: {e}")

    payload = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "source": str(HARD_LIST),
        "n_hard": len(hard),
        "n_presets": len(PRESETS),
        "target_total": len(hard) * len(PRESETS),
        "n_ok": len(ok), "n_skipped": skipped, "n_fail": len(fail),
        "n_bad_validate": len(bad),
        "presets": {p: PRESET_CN[p] for p in sorted(PRESETS)},
        "manifest": str(OUT_ROOT / "manifest.json"),
    }
    (OUT_ROOT / "manifest.json").write_text(
        json.dumps(ok, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(OUT_JSON).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    from collections import Counter
    per_preset = Counter(r["preset"] for r in ok)
    per_dialect = Counter(r["dialect"] for r in ok)
    lines = [
        "# 红蓝第 2 轮 · 对抗退化矩阵报告",
        "",
        f"> 生成：{payload['date']} · `evaluation/round2_degrade_matrix.py`",
        f"> 输入：34 条 wide 极难样本（score<0.3）× 4 预设退化 = 136 目标变体",
        "",
        "## 结果",
        "",
        f"- 成功生成 **{len(ok)}** 条（新生成 {len(ok)-skipped} / 复用 {skipped}），失败 {len(fail)}，校验异常 {len(bad)}；",
        f"- 文件可读性/时长校验：全部可解码且 ≥0.5s（异常 {len(bad)} 条）；amr 档为 8kHz（amr-nb 12.2k 语义），其余 16kHz；",
        "",
        "### 分预设",
        "",
        "| 预设 | 含义 | 条数 |",
        "|---|---|---|",
    ]
    for p in sorted(PRESETS):
        lines.append(f"| {p} | {PRESET_CN[p]} | {per_preset.get(p,0)} |")
    lines += ["", "### 分方言", "", "| 方言 | 条数 |", "|---|---|"]
    for d, n in sorted(per_dialect.items()):
        lines.append(f"| {d} | {n} |")
    if bad:
        lines += ["", "## 校验异常明细", ""]
        lines += [f"- `{b}`" for b in bad[:20]]
    if fail:
        lines += ["", "## 失败明细", ""]
        lines += [f"- `{f['src']}` → {f['err']}" for f in fail[:20]]
    lines += ["", "## 后续", "- 变体已入 `round2_degrade/manifest.json`，供三通道联合复测（1.1④）与击穿率分信道表（1.1③）使用。"]
    Path(OUT_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n完成: ok={len(ok)} skip={skipped} fail={len(fail)} bad={len(bad)} 耗时 {time.time()-t0:.0f}s")
    print(f"报告 -> {OUT_MD}")


if __name__ == "__main__":
    main()
