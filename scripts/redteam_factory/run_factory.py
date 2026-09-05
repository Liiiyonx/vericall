# -*- coding: utf-8 -*-
"""run_factory.py — 红队伪造语料工厂编排器（极致化计划书 §四）

网格：引擎(GPT-SoVITS/CosyVoice/Seed-VC) × 方言参考音 × 话术 × 信道
（clean/phone8k/mp3_16k/amr/noise），批量产出 16kHz wav + meta.csv。

meta.csv 前 6 列与详册 §7.2 约定一致：
  path, label(bonafide/spoof), attack_id, channel, source, license
扩展列：engine, dialect, script_id, speaker_ref, duration_s

设计原则：
- 引擎适配器用 cmd 模板（factory_config.json），未安装的引擎探测后整列跳过，
  工厂不因单个引擎缺失而停摆；
- 断点续跑：目标 wav 已存在即跳过；
- seedvc 是 voice conversion，需要 source_wav（先用任一已合成/真人 wav）。

用法：
  python run_factory.py --dry-run                # 打印网格计划与引擎可用性
  python run_factory.py --engine sovits --dialect minnan --per-cell 5
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from degrade_audio import PRESETS, degrade_file  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parent / "factory_config.json"
META_NAME = "meta.csv"
META_FIELDS = ["path", "label", "attack_id", "channel", "source", "license",
               "engine", "dialect", "script_id", "speaker_ref", "duration_s"]


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def probe_engines(cfg: dict) -> dict:
    """返回 {engine_name: (available, reason)}"""
    status = {}
    for eng in cfg["engines"]:
        if eng.get("kind") == "edgetts":
            try:
                import edge_tts  # noqa: F401
                ff = shutil.which("ffmpeg")
                status[eng["name"]] = (bool(ff), "ok" if ff else "缺 ffmpeg（mp3 转 wav 需要）")
            except ImportError:
                status[eng["name"]] = (False, "pip install edge-tts")
            continue
        probe = ROOT / eng["probe_path"]
        status[eng["name"]] = (probe.exists(),
                               "ok" if probe.exists() else f"未找到 {eng['probe_path']}")
    return status


def load_scripts(cfg: dict, dialect: str | None, limit_per_dialect: int) -> list:
    """从话术库取文本；dialect_style 命中方言目录的优先。"""
    recs = []
    for src in cfg["script_sources"]:
        p = ROOT / src
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    recs.append({"id": r.get("id", "?"),
                                 "text": r["text"],
                                 "dialect_style": r.get("dialect_style", r.get("style", ""))})
    if dialect:
        hit = [r for r in recs if r["dialect_style"].startswith(dialect)]
        rest = [r for r in recs if not r["dialect_style"].startswith(dialect)]
        recs = hit + rest  # 方言匹配的排前面，不足用普通话补齐
    return recs[:limit_per_dialect]


def find_refs(cfg: dict, dialect: str, limit: int, eng: dict) -> list:
    """说话人参考：CLI 引擎为方言目录下的参考音 wav；edgetts 引擎为音色名。"""
    if eng.get("kind") == "edgetts":
        voices = eng.get("voices", {}).get(dialect, eng.get("voices", {}).get("mandarin", []))
        return voices[:limit]
    d = ROOT / cfg["speaker_ref_root"] / dialect
    if not d.exists():
        return []
    return sorted(d.glob("*.wav"))[:limit]


def _synth_edgetts(voice: str, text: str, out_path: Path, retries: int = 3) -> tuple:
    """edge-tts 合成：内存收集 mp3 流 → ffmpeg 管道转 16kHz 单声道 wav。

    不落中间 mp3 文件（避免批量临时文件删除触发安全钩子）；
    服务端偶发 NoAudioReceived，重试 3 次。
    """
    import asyncio

    import edge_tts

    async def _fetch() -> bytes:
        chunks = []
        async for chunk in edge_tts.Communicate(text, voice).stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])
        return b"".join(chunks)

    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            # asyncio.wait_for 兜底：半开连接永久挂起时 45s 强断（台账坑位速查）
            mp3_bytes = asyncio.run(asyncio.wait_for(_fetch(), timeout=45))
            if not mp3_bytes:
                raise RuntimeError("empty audio stream")
            ff = subprocess.run(
                ["ffmpeg", "-y", "-i", "pipe:0", "-ar", "16000", "-ac", "1",
                 str(out_path)],
                input=mp3_bytes, capture_output=True, timeout=60)
            if ff.returncode != 0 or not out_path.exists():
                raise RuntimeError("ffmpeg: " + ff.stderr.decode(errors="ignore")[-200:])
            return True, "ok"
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            if attempt < retries:
                time.sleep(2 * attempt)
    return False, last_err


def _synth_sovits_api(url: str, ref_audio: Path, prompt_text: str, text: str,
                      out_path: Path, retries: int = 2) -> tuple:
    """GPT-SoVITS api_v2（GPU 常驻服务）批处理适配。GET /tts 返回音频流落盘。"""
    import urllib.parse
    import urllib.request

    qs = urllib.parse.urlencode({
        "text": text[:400], "text_lang": "zh",
        "ref_audio_path": str(ref_audio),
        "prompt_text": prompt_text[:200], "prompt_lang": "zh",
        "text_split_method": "cut5", "batch_size": 1,
        "media_type": "wav", "streaming_mode": "false",
    })
    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(f"{url}?{qs}", timeout=120) as resp:
                data = resp.read()
            if not data or len(data) < 1024:
                raise RuntimeError(f"空/异常响应 {len(data) if data else 0}B")
            out_path.write_bytes(data)
            return True, "ok"
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            if attempt < retries:
                time.sleep(2 * attempt)
    return False, last_err


def _prompt_for(ref_audio: Path) -> str:
    """查自举参考音的 prompt_text（refs.json 中登记）。"""
    refs_json = ROOT / "data" / "redteam" / "refs.json"
    if refs_json.exists():
        import json as _json
        try:
            return _json.loads(refs_json.read_text(encoding="utf-8")).get(
                str(ref_audio), "").get("prompt_text", "")
        except Exception:
            return ""
    return ""


def synth_one(eng: dict, ref, text: str, out_path: Path,
              source_wav: Path | None = None) -> tuple:
    """调引擎合成一条。ref：CLI/API 引擎为参考音 Path，edgetts 为音色名。返回 (ok, msg)。"""
    if eng.get("kind") == "edgetts":
        return _synth_edgetts(str(ref), text, out_path)
    if eng.get("kind") == "sovits_api":
        url = eng.get("url", "http://127.0.0.1:9880/tts")
        return _synth_sovits_api(url, Path(ref), _prompt_for(Path(ref)), text, out_path)
    cmd = eng["cmd"].format(ref=str(ref), ref_text=_prompt_for(Path(ref)), text=text,
                            out=str(out_path), source_wav=str(source_wav or ref))
    try:
        proc = subprocess.run(cmd, shell=True, cwd=ROOT, timeout=300,
                              capture_output=True, text=True)
        if proc.returncode == 0 and out_path.exists():
            return True, "ok"
        return False, (proc.stderr or proc.stdout or "no output")[-300:]
    except subprocess.TimeoutExpired:
        return False, "timeout(300s)"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", help="只跑指定引擎（默认全部可用引擎）")
    ap.add_argument("--dialect", help="只跑指定方言目录（默认全部）")
    ap.add_argument("--per-cell", type=int, default=0,
                    help="覆盖每方言话术条数上限（默认取配置值）")
    ap.add_argument("--channels", help="逗号分隔，默认全部预设")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--shard-idx", type=int, default=0, help="分片序号(0 起)")
    ap.add_argument("--shard-total", type=int, default=1, help="分片总数")
    ap.add_argument("--meta-file", help="meta 输出路径（默认 out_root/meta.csv，分片用独立文件）")
    args = ap.parse_args()

    if not (0 <= args.shard_idx < args.shard_total):
        sys.exit("--shard-idx 越界")

    cfg = load_config()
    status = probe_engines(cfg)
    out_root = ROOT / cfg["output_root"]
    channels = args.channels.split(",") if args.channels else cfg["channels"]
    bad = set(channels) - (PRESETS | {"clean"})
    if bad:
        sys.exit(f"未知信道预设: {bad}（可选 clean/{'/'.join(sorted(PRESETS))}）")

    engines = [e for e in cfg["engines"] if not args.engine or e["name"] == args.engine]
    dialects = [args.dialect] if args.dialect else cfg["dialects"]

    print("== 引擎探测 ==")
    for name, (ok, reason) in status.items():
        print(f"  {name}: {'可用' if ok else '不可用 — ' + reason}")

    plan = []
    for eng in engines:
        if not status.get(eng["name"], (False,))[0]:
            continue
        for dialect in dialects:
            refs = find_refs(cfg, dialect, cfg["cell_limits"]["refs_per_dialect"], eng)
            scripts = load_scripts(cfg, dialect,
                                   args.per_cell or cfg["cell_limits"]["scripts_per_dialect"])
            if not refs:
                print(f"  [warn] {dialect}: 无可用说话人参考（参考音/音色），跳过")
                continue
            for ref in refs:
                for sc in scripts:
                    plan.append((eng, dialect, ref, sc))

    print(f"\n== 网格计划 ==  {len(plan)} 条合成 × {len(channels)} 信道 = {len(plan) * len(channels)} 个 wav")
    plan = plan[args.shard_idx::args.shard_total]  # 分片：round-robin 均匀拆分
    print(f"== 本分片 ==  {args.shard_idx+1}/{args.shard_total} → {len(plan)} 条合成")
    if args.dry_run:
        from collections import Counter
        c = Counter((e['name'], d) for e, d, _, _ in plan)
        for (en, d), n in sorted(c.items()):
            print(f"  {en} × {d}: {n}")
        return

    out_root.mkdir(parents=True, exist_ok=True)
    meta_path = Path(args.meta_file) if args.meta_file else out_root / META_NAME
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_exists = meta_path.exists()
    n_ok, n_skip, n_fail = 0, 0, 0
    t0 = time.time()

    with open(meta_path, "a", encoding="utf-8", newline="") as mf:
        w = csv.writer(mf)
        if not meta_exists:
            w.writerow(META_FIELDS)
        for eng, dialect, ref, sc in plan:
            ref_id = Path(str(ref)).stem
            stem = f"{eng['name']}_{dialect}_{ref_id}_{sc['id']}"
            clean_path = out_root / eng["name"] / dialect / f"{stem}.wav"
            if clean_path.exists():
                n_skip += 1
                continue
            clean_path.parent.mkdir(parents=True, exist_ok=True)
            ok, msg = synth_one(eng, ref, sc["text"], clean_path)
            if not ok:
                n_fail += 1
                print(f"  [fail] {stem}: {msg}")
                continue
            n_ok += 1
            attack_id = f"{eng['attack_prefix']}-{dialect.upper()}"
            for ch in channels:
                if ch == "clean":
                    final = clean_path
                else:
                    final = degrade_file(clean_path, clean_path.parent, ch)
                    if final is None:
                        continue
                rel = final.relative_to(ROOT).as_posix()
                w.writerow([rel, "spoof", attack_id, ch,
                            f"redteam_factory/{eng['display']}", eng["license"],
                            eng["name"], dialect, sc["id"], ref_id, ""])
            mf.flush()
            print(f"  [ok] {stem} ×{len(channels)}")

    dt = time.time() - t0
    print(f"\n完成：合成 {n_ok}，跳过(已存在) {n_skip}，失败 {n_fail}，耗时 {dt:.0f}s")
    print(f"meta -> {meta_path}")
    print("下一步：python scripts/redteam_factory/quality_gate.py")


if __name__ == "__main__":
    main()
