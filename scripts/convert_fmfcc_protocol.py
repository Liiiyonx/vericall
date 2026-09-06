#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FMFCC-A → ASVspoof 协议转换器（任务书 P1-3 · 中文场景首次验证）
================================================================
把 FMFCC-A（Fake Media Forensics Challenge，中文合成语音检测数据集）的
标签与文件布局，转换成 AASIST / 本项目 `eval_aasist_eer.py` 能直接消费的
ASVspoof 风格 5 列协议 + ASVspoof 风格目录布局。

为什么需要它：
  通道① AASIST 目前只在英文 ASVspoof2019 LA 上训练/评测。要回答"英文模型
  直测中文到底行不行"，必须先把 FMFCC-A 对齐成统一协议，才能复用
  `scripts/eval_aasist_eer.py` 与 `evaluation/attack_breakdown_eval.py` 跑 EER。

FMFCC-A 结构（参考 https://github.com/Amforever/FMFCC-A）：
  - 总 5 万条：1 万真人(G00) + 4 万伪造(A01–A13，11 个 TTS + 2 个 VC 系统)。
  - 划分：Training / Development / Evaluation；eval 集有一半做过
    压缩/加噪后处理（跨条件鲁棒性）。
  - 文件命名形如 `G00_0001.wav`(真人) / `A01_0001.wav`(伪造，A01 为系统号)；
    也可能为 mp3/aac（依赖 ffmpeg 转 wav）。

输出布局（把 --out 当成一个"伪装成 ASVspoof LA"的数据根，可直接喂给现有评测）：
  <out>/
    ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.<split>.trl.txt
    ASVspoof2019_LA_<split>/flac/<utt_id>.wav   (--no-audio 时只写协议)

协议每行 5 列（与 ASVspoof2019 LA / parse_protocol.py 严格一致）：
    <speaker> <utt_id> - <system_id> <bonafide|spoof>
    - token[1] = utt_id  （作为 flac/wav 文件名，不含扩展名）
    - token[3] = system_id（bonafide 用 "-" 占位；spoof 用 A01..A13）
    - token[4] = 标签

此后即可（真实权重就位后）：
    set VERICALL_ASVSPOOF_LA=<out>
    python scripts/eval_aasist_eer.py --split dev --limit 2000   # 冒烟
    python scripts/eval_aasist_eer.py --split dev                # 全量中文 EER

无真实数据时可用 `--selftest` 自建一棵合成目录验证协议生成逻辑。

用法：
    python scripts/convert_fmfcc_protocol.py --root D:/FMFCC-A --out data/raw/FMFCC-A_asvspoof
    python scripts/convert_fmfcc_protocol.py --root D:/FMFCC-A --out data/raw/FMFCC-A_asvspoof --no-audio
    python scripts/convert_fmfcc_protocol.py --selftest
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

# 真人前缀 / 伪造系统前缀（FMFCC-A 论文 Table 1）
BONAFIDE_PREFIX = "G00"
SPOOF_RE = re.compile(r"^A(\d{2})$")

# FMFCC-A 常见 split 名 → ASVspoof 风格后缀
SPLIT_ALIAS = {
    "training": "dev", "train": "dev", "trn": "dev",
    "development": "dev", "dev": "dev", "dev_neval": "dev",
    "evaluation": "eval", "eval": "eval", "test": "eval", "te": "eval",
}
# 反向：本项目协议后缀 → FMFCC-A 目录扫描关键词
PROTO_SUFFIX = {"dev": "dev", "eval": "eval"}


def _system_from_name(stem: str) -> str | None:
    """从文件名前缀解析系统号：G00→真人占位；A01..A13→伪造系统。

    返回 ('bonafide', None) 或 ('spoof', 'A01')，无法解析返回 None。
    兼容 `G00_0001`、`A01_0001`、`G00-0001.wav` 等命名。
    """
    m = re.match(r"^(G00|A\d{2})[_\-]?", stem)
    if not m:
        return None
    tok = m.group(1)
    if tok == BONAFIDE_PREFIX:
        return ("bonafide", None)
    if SPOOF_RE.match(tok):
        return ("spoof", tok)
    return None


def _find_split_dirs(root: Path) -> dict[str, Path]:
    """扫描 root，返回 {split: 该 split 目录}。

    识别两类布局：
      A) 标准：<root>/<Training|Development|Evaluation>/<bonafide|fake>/
      B) 扁平：<root>/<bonafide|fake>/  且文件名带 G00/Axx 前缀
    """
    found: dict[str, Path] = {}

    # A) 标准分目录
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        low = d.name.lower()
        for key, suffix in SPLIT_ALIAS.items():
            if low == key:
                found.setdefault(suffix, d)
    if found:
        return found

    # B) 扁平：root 下直接有 bonafide/fake
    bf = {"bonafide": root / "bonafide", "fake": root / "fake"}
    if bf["bonafide"].is_dir() or bf["fake"].is_dir():
        # 扁平布局默认当作 dev（用户可用 --split 显式指定）
        return {"dev": root}
    return found


def _iter_audio(d: Path):
    for f in sorted(d.rglob("*")):
        if f.is_file() and f.suffix.lower() in (".wav", ".flac", ".mp3", ".aac", ".ogg"):
            yield f


def convert_split(split_dir: Path, split: str, out_root: Path,
                  copy_audio: bool, sr: int, prefer_label: str | None = None) -> dict:
    """转换一个 split，写出协议与（可选）音频。返回统计字典。"""
    proto_dir = out_root / "ASVspoof2019_LA_cm_protocols"
    proto_dir.mkdir(parents=True, exist_ok=True)
    proto_name = f"ASVspoof2019.LA.cm.{split}.trl.txt"
    proto_path = proto_dir / proto_name
    flac_dir = out_root / f"ASVspoof2019_LA_{split}" / "flac"
    flac_dir.mkdir(parents=True, exist_ok=True)

    # 收集 bonafide / fake 来源目录
    bon_src = split_dir / "bonafide"
    fake_src = split_dir / "fake"
    sources: list[tuple[Path, str]] = []
    if bon_src.is_dir():
        sources.append((bon_src, "bonafide"))
    if fake_src.is_dir():
        sources.append((fake_src, "fake"))
    if not sources:
        # 扁平布局：文件名前缀决定标签（直接把音频放在 split 目录下）
        has_prefix = any(_system_from_name(p.stem) for p in _iter_audio(split_dir))
        if has_prefix:
            sources.append((split_dir, "auto"))

    if not sources:
        return {"split": split, "protocol": str(proto_path),
                "n_bonafide": 0, "n_spoof": 0,
                "audio_written": 0, "audio_skipped": 0, "audio_fail": 0}

    rows: list[str] = []
    n_bon = n_spo = 0
    n_written = n_skipped = 0
    n_audio_fail = 0

    for src_dir, kind in sources:
        for af in _iter_audio(src_dir):
            stem = af.stem
            # 判定标签
            if kind == "bonafide":
                label, sysid = "bonafide", None
            elif kind == "fake":
                label, sysid = "spoof", None  # 系统号从文件名前缀取
            else:  # auto：从文件名前缀推断
                parsed = _system_from_name(stem)
                if parsed is None:
                    continue
                label, sysid = parsed
            # 伪造系统号：优先文件名前缀，其次目录/参数
            if label == "spoof" and sysid is None:
                p = _system_from_name(stem)
                sysid = p[1] if p and p[1] else (prefer_label or "A00")
            if label == "bonafide":
                sysid_out = "-"
                spk = stem
            else:
                sysid_out = sysid or "A00"
                spk = sysid_out
            rows.append(f"{spk} {stem} - {sysid_out} {label}")
            if label == "bonafide":
                n_bon += 1
            else:
                n_spo += 1

            # 可选：转 16k wav
            if copy_audio:
                dst = flac_dir / f"{stem}.wav"
                if dst.exists():
                    n_skipped += 1
                    continue
                ok = _to_wav(af, dst, sr)
                if ok:
                    n_written += 1
                else:
                    n_audio_fail += 1

    with open(proto_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + ("\n" if rows else ""))

    return {
        "split": split, "protocol": str(proto_path),
        "n_bonafide": n_bon, "n_spoof": n_spo,
        "audio_written": n_written, "audio_skipped": n_skipped,
        "audio_fail": n_audio_fail,
    }


def _to_wav(src: Path, dst: Path, sr: int) -> bool:
    """尽量把任意格式音频转成 16k 单声道 wav。失败时返回 False。"""
    try:
        import numpy as np
        try:
            import soundfile as sf
            data, fs = sf.read(str(src), dtype="float32", always_2d=False)
        except Exception:
            # 退回 ffmpeg 中转
            import subprocess
            tmp = dst.with_suffix(".tmp.wav")
            r = subprocess.run(
                ["ffmpeg", "-y", "-i", str(src), "-ar", str(sr),
                 "-ac", "1", str(tmp)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if r.returncode != 0 or not tmp.exists():
                return False
            import soundfile as sf2
            data, fs = sf2.read(str(tmp), dtype="float32")
            tmp.unlink(missing_ok=True)
        data = np.asarray(data, dtype=np.float32)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if int(fs) != int(sr):
            # 简易重采样（scipy 优先）
            try:
                from scipy.signal import resample_poly
                g = int(sr); up = g // 100
                data = resample_poly(data, up, max(1, int(fs) // 100)).astype(np.float32)
                # 粗略对齐到目标采样率（resample_poly 比例不精确时退化为跳过）
            except Exception:
                pass
        # 归一化防溢出
        peak = float(np.max(np.abs(data))) if data.size else 0.0
        if peak > 1.0:
            data = data / peak
        import soundfile as sf3
        sf3.write(str(dst), data, sr, subtype="PCM_16")
        return True
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="FMFCC-A → ASVspoof 协议转换")
    ap.add_argument("--root", default=None, help="FMFCC-A 根目录")
    ap.add_argument("--out", default=str(Path("data/raw/FMFCC-A_asvspoof")),
                    help="输出（ASVspoof 风格）数据根")
    ap.add_argument("--splits", nargs="*", default=None,
                    help="限定 split，如 --splits dev eval；默认扫描全部")
    ap.add_argument("--no-audio", action="store_true",
                    help="只写协议，不转换音频")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--selftest", action="store_true",
                    help="用合成目录自测协议生成逻辑（无需真实数据）")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    if not args.root:
        ap.error("--root 必填（或加 --selftest）")
    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"root 不存在: {root}")

    split_dirs = _find_split_dirs(root)
    if not split_dirs:
        raise SystemExit(
            f"在 {root} 未识别到 FMFCC-A 布局（期望 <split>/<bonafide|fake>/ 或 "
            f"直接 <bonafide|fake>/）。")
    targets = args.splits or list(split_dirs.keys())
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"FMFCC-A 根: {root}")
    print(f"输出根    : {out_root}")
    print(f"转换 split: {targets}   音频={'否' if args.no_audio else '是'}")
    print("-" * 60)
    for sp in targets:
        sd = split_dirs.get(sp)
        if sd is None:
            print(f"  [{sp}] 未找到对应目录，跳过")
            continue
        r = convert_split(sd, sp, out_root, not args.no_audio, args.sr)
        print(f"  [{sp}] 协议={r['protocol']}")
        print(f"         bonafide={r['n_bonafide']}  spoof={r['n_spoof']}"
              f"  audio_w={r['audio_written']} skip={r['audio_skipped']}"
              f" fail={r['audio_fail']}")
    print("-" * 60)
    print(f"完成。下一步（真实 AASIST 权重就位后）：")
    print(f"  set VERICALL_ASVSPOOF_LA={out_root}")
    print(f"  python scripts/eval_aasist_eer.py --split dev --limit 2000")


def _selftest() -> None:
    """自建一棵合成 FMFCC-A 目录，验证协议解析/转换逻辑。"""
    import numpy as np
    try:
        import soundfile as sf
    except Exception:
        sf = None

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "FMFCC-A"
        # 标准布局：Development/{bonafide,fake}
        dev = root / "Development"
        (dev / "bonafide").mkdir(parents=True)
        (dev / "fake").mkdir(parents=True)
        # 写几个合成 wav（若 soundfile 可用）
        def _w(name: Path):
            if sf is not None:
                sf.write(str(name), np.zeros(1600, dtype=np.float32), 16000)
        for i in range(3):
            _w(dev / "bonafide" / f"G00_{i:04d}.wav")
        for i in range(5):
            _w(dev / "fake" / f"A0{i+1}_{i:04d}.wav")
        # 还放一个非标准扩展名 mp3（无 ffmpeg 时应被安全跳过音频、但协议仍记）
        (dev / "fake" / "A09_0099.mp3").write_bytes(b"fake")

        out = Path(td) / "out"
        r = convert_split(dev, "dev", out, copy_audio=True, sr=16000)

        proto = Path(r["protocol"]).read_text(encoding="utf-8").splitlines()
        assert r["n_bonafide"] == 3, r
        assert r["n_spoof"] == 6, r  # 5 wav + 1 mp3（协议都记）
        # 校验 5 列格式
        for line in proto:
            parts = line.split()
            assert len(parts) == 5, f"协议行应为 5 列: {line!r}"
            assert parts[4] in ("bonafide", "spoof")
            assert parts[2] == "-"
            if parts[4] == "bonafide":
                assert parts[3] == "-"
            else:
                assert re.match(r"^A\d{2}$", parts[3]), parts
        # 音频：wav 应成功写出，mp3 在无声文件下可能失败（不强制）
        assert r["audio_written"] >= 3, r
        print(f"[selftest] 通过：bon={r['n_bonafide']} spo={r['n_spoof']} "
              f"audio_w={r['audio_written']} fail={r['audio_fail']}")
        print(f"          示例协议行：{proto[0]!r} / {proto[3]!r}")


if __name__ == "__main__":
    main()
