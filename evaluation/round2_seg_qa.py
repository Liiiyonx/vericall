# -*- coding: utf-8 -*-
"""
round2_seg_qa.py — 任务 1.2 红队切段质量分层复核（机器预筛部分）
============================================================
对 round2_hard34.json 中 28 条极难切段（kind=seg）做音频结构健康筛查：
时长 / 响度 / 静音占比 / 削波 / 首尾空隙，输出逐条 verdict（pass_struct / suspect_*），
flagged 项留给人耳终审（切坏/静音/过短判别）。6 条母本仅作 sanity 参照。

口径升级目标：极难集 = wide_score<0.3 且人工确认音频有效（本脚本输出机器证据）。

产出：evaluation/round2_seg_qa.md / .json
纯 CPU，无 GPU 依赖。
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
HARD34 = REPO / "evaluation" / "round2_hard34.json"
OUT_MD = REPO / "evaluation" / "round2_seg_qa.md"
OUT_JSON = REPO / "evaluation" / "round2_seg_qa.json"

# 筛查阈值（对 Edge-TTS/SoVITS 合成 clean 段适用）
DUR_MIN_S = 1.0          # 过短：整段 <1s 信息量可疑
VOICED_MIN_S = 0.7       # 有效语音总量下限
VOICED_RATIO_MIN = 0.35  # 语音帧占比下限（静音段过长 → 疑切坏/内容稀）
RMS_QUIET_DB = -42.0     # 整段过静
PEAK_QUIET_DB = -36.0    # 峰值过低（音量异常）
CLIP_RATIO_MAX = 5e-4    # 硬削波占比上限
LEAD_SIL_MAX = 0.6       # 首部静音过长（切点含大段空白）
TAIL_SIL_MAX = 0.9       # 尾部静音过长


def analyze(path: Path):
    data, sr = sf.read(str(path), always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    dur = n / sr
    fl = max(1, int(0.02 * sr))          # 20ms 帧
    nf = n // fl
    frames = data[: nf * fl].reshape(nf, fl) if nf else np.zeros((1, fl))
    rms = frames.std(axis=1)
    eps = 1e-12
    voiced = rms > 10 ** (-45 / 20)      # 帧 RMS > -45 dBFS 视为有声
    v_ratio = float(voiced.mean()) if nf else 0.0
    v_s = float(voiced.sum()) * (fl / sr) if nf else 0.0
    if nf and voiced.any():
        vi = np.where(voiced)[0]
        onset_s = float(vi[0]) * (fl / sr)
        offset_s = float((vi[-1] + 1)) * (fl / sr)
    else:
        onset_s, offset_s = dur, 0.0
    peak = float(np.abs(data).max()) if n else 0.0
    peak_db = 20 * math.log10(peak + eps)
    rms_all = float(np.sqrt((data ** 2).mean())) if n else 0.0
    rms_db = 20 * math.log10(rms_all + eps)
    clip_ratio = float((np.abs(data) >= 0.999).mean()) if n else 0.0
    return {
        "sr": int(sr), "dur_s": round(dur, 3),
        "rms_db": round(rms_db, 2), "peak_db": round(peak_db, 2),
        "voiced_ratio": round(v_ratio, 3), "voiced_s": round(v_s, 3),
        "leading_sil_s": round(onset_s, 3), "trailing_sil_s": round(dur - offset_s, 3),
        "clip_ratio": clip_ratio,
    }


def verdict(m: dict, kind: str):
    if kind != "seg":
        return "master-ref", "母本参照，不参与切段判定"
    flags = []
    if m["voiced_s"] < VOICED_MIN_S:
        flags.append(f"有效语音仅{m['voiced_s']:.2f}s(<{VOICED_MIN_S}s)")
    if m["dur_s"] < DUR_MIN_S:
        flags.append(f"整段仅{m['dur_s']:.2f}s(<{DUR_MIN_S}s)")
    if m["voiced_ratio"] < VOICED_RATIO_MIN:
        flags.append(f"语音占比{m['voiced_ratio']:.0%}(<{VOICED_RATIO_MIN:.0%})")
    if m["rms_db"] < RMS_QUIET_DB:
        flags.append(f"整段过静 {m['rms_db']}dBFS")
    if m["peak_db"] < PEAK_QUIET_DB:
        flags.append(f"峰值过低 {m['peak_db']}dBFS")
    if m["clip_ratio"] > CLIP_RATIO_MAX:
        flags.append(f"硬削波 {m['clip_ratio']:.1e}")
    if m["leading_sil_s"] > LEAD_SIL_MAX:
        flags.append(f"首部静音 {m['leading_sil_s']:.2f}s")
    if m["trailing_sil_s"] > TAIL_SIL_MAX:
        flags.append(f"尾部静音 {m['trailing_sil_s']:.2f}s")
    if flags:
        return "suspect", "；".join(flags)
    return "pass_struct", "时长/响度/语音占比/削波/首尾空隙结构健康，音频有效（待耳听抽查）"


def main():
    hard = json.loads(HARD34.read_text(encoding="utf-8"))
    rows = []
    for it in hard:
        p = REPO / it["path"]
        base = {
            "path": it["path"], "engine": it["engine"], "dialect": it["dialect"],
            "script_id": it["script_id"], "speaker_ref": it["speaker_ref"],
            "wide_score": it["wide_score"], "kind": it["kind"],
        }
        if not p.exists():
            rows.append({**base, "status": "missing", "verdict": "missing", "note": "文件缺失"})
            continue
        try:
            m = analyze(p)
        except Exception as e:  # noqa: BLE001
            rows.append({**base, "status": "error", "verdict": "error", "note": str(e)[:120]})
            continue
        v, note = verdict(m, it["kind"])
        rows.append({**base, "status": "ok", **m, "verdict": v, "note": note})

    segs = [r for r in rows if r["kind"] == "seg"]
    masters = [r for r in rows if r["kind"] == "master"]
    assert len(segs) == 28 and len(masters) == 6, f"清单数量异常 seg={len(segs)} master={len(masters)}"
    n_pass = sum(1 for r in segs if r["verdict"] == "pass_struct")
    n_sus = sum(1 for r in segs if r["verdict"] == "suspect")
    n_miss = sum(1 for r in segs if r["status"] in ("missing", "error"))

    # ---- markdown ----
    L = []
    L.append("# 红队切段质量分层复核（任务 1.2 · 机器预筛）\n")
    L.append(f"> 日期：2026-09-06 · 脚本：`evaluation/round2_seg_qa.py`（纯 CPU）· 输入：`round2_hard34.json`（6 母本 + 28 切段 = 34）\n")
    L.append("## 一、方法\n")
    L.append("对每条样本做音频结构健康筛查（soundfile 读 wav，20ms 帧 RMS 语音检测）：\n")
    L.append("| 指标 | 阈值 | 含义 |")
    L.append("|---|---|---|")
    L.append(f"| 整段时长 | ≥{DUR_MIN_S}s | 过短段信息量可疑 |")
    L.append(f"| 有效语音量 | ≥{VOICED_MIN_S}s 且占比 ≥{VOICED_RATIO_MIN:.0%} | 静音过多疑切坏/内容稀 |")
    L.append(f"| 整段 RMS / 峰值 | ≥{RMS_QUIET_DB} / ≥{PEAK_QUIET_DB} dBFS | 音量异常 |")
    L.append(f"| 硬削波占比 | ≤{CLIP_RATIO_MAX:.0e} | 重编码/限幅伪影 |")
    L.append(f"| 首/尾静音 | ≤{LEAD_SIL_MAX}s / ≤{TAIL_SIL_MAX}s | 切点含大段空白 |")
    L.append("\n> 说明：本报告为**机器预筛**，verdict=pass_struct 仅代表结构健康，仍需耳听抽查；verdict=suspect 的段建议人耳终审（听判切坏/静音/过短），必要时 drop。\n")
    L.append(f"## 二、结果统计（28 切段）\n")
    L.append(f"- **pass_struct（结构有效）**：{n_pass} 段；**suspect（需人耳终审）**：{n_sus} 段；缺失/读取错误：{n_miss} 段\n")
    L.append(f"- 母本 6 条仅作参照（master-ref，不判定）。\n")
    L.append("\n## 三、逐段明细（28 切段）\n")
    L.append("| script_id | speaker | 方言 | wide | 时长s | RMS dB | 语音占比 | 削波 | 首部静音s | 尾部静音s | verdict | 说明 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(segs, key=lambda x: (x["verdict"] == "pass_struct", x["script_id"])):
        if r["status"] != "ok":
            L.append(f"| {r['script_id']} | {r['speaker_ref']} | {r['dialect']} | {r['wide_score']} | - | - | - | - | - | - | {r['verdict']} | {r['note']} |")
            continue
        L.append(
            f"| {r['script_id']} | {r['speaker_ref']} | {r['dialect']} | {r['wide_score']:.3f} | "
            f"{r['dur_s']:.2f} | {r['rms_db']:.1f} | {r['voiced_ratio']:.0%} | "
            f"{('%.0e' % r['clip_ratio']) if r['clip_ratio'] > 0 else '0'} | {r['leading_sil_s']:.2f} | "
            f"{r['trailing_sil_s']:.2f} | {r['verdict']} | {r['note']} |"
        )
    L.append("\n## 四、待人工耳检清单\n")
    sus = [r for r in segs if r["verdict"] == "suspect"]
    if sus:
        for r in sus:
            L.append(f"- `{r['script_id']}` {r['speaker_ref']}（{r['dialect']}，wide {r['wide_score']:.3f}）：{r['note']}")
    else:
        L.append("- 无。全部 28 段结构健康，进入耳听抽查阶段。\n")
    L.append("\n## 五、口径升级建议\n")
    L.append("- 极难集口径由「wide<0.3」升级为「wide<0.3 **且** 音频结构有效（本报告 pass_struct）」，与计划任务 1.2 验收一致；")
    L.append("  suspect 段待耳检结论（pass→保留 / drop→移出极难集）后再定稿清单。\n")
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    json.dump({"n_seg": len(segs), "n_pass": n_pass, "n_suspect": n_sus, "n_missing_error": n_miss,
               "rows": rows}, OUT_JSON.open("w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[round2_seg_qa] seg={len(segs)} pass={n_pass} suspect={n_sus} miss/err={n_miss}")
    print(f"[round2_seg_qa] wrote {OUT_MD.name} / {OUT_JSON.name}")


if __name__ == "__main__":
    sys.exit(main())
