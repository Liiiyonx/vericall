#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 HuggingFace 镜像 (Bisher/ASVspoof_2019_LA) 的 parquet 数据集
转换为 AASIST (clovaai/aasist) 训练所需的目录与 protocol 文件。

输出布局 (database_path = out_root):
    out_root/
        ASVspoof2019_LA_train/flac/<utt_id>.flac
        ASVspoof2019_LA_dev/flac/<utt_id>.flac
        ASVspoof2019_LA_eval/flac/<utt_id>.flac
        ASVspoof2019_LA_cm_protocols/
            ASVspoof2019.LA.cm.train.trn.txt
            ASVspoof2019.LA.cm.dev.trl.txt
            ASVspoof2019.LA.cm.eval.trl.txt

protocol 每行格式 (与 data_utils.genSpoof_list 严格对齐):
    <speaker_id> <utt_id> - <system_id> <bonafide|spoof>
    - token[1] = utt_id  (作为 flac 文件名, 不含扩展名)
    - token[4] = 标签     (bonafide / spoof)  -> 训练/验证标签
    - token[3] = system_id (eval 打分时作为 src 写入分数文件)

parquet 字段: speaker_id(str) audio_file_name(str) audio(struct:array/path/sampling_rate)
             system_id(str) key(class_label 0=bonafide,1=spoof)
"""
import os
import sys
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from paths import PARQUET_DIR, ASVSPOOF_LA  # noqa: E402

# parquet 文件名中的 split -> AASIST 目录/协议后缀
SPLIT_MAP = {
    "train": "train",
    "validation": "dev",
    "test": "eval",
}
PROTO_EXT = {"train": "trn", "dev": "trl", "eval": "trl"}


def _norm(x: np.ndarray) -> np.ndarray:
    """确保音频在 [-1, 1] 范围 (HF audio 特征通常已是 float32 归一化)。"""
    x = np.asarray(x, dtype=np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1.5:  # 看起来是 int16 量纲, 归一化
        x = x / 32768.0
    return x


def convert_one(pq_path: str, out_root: str, split_out: str, batch_size: int = 500):
    flac_dir = os.path.join(out_root, f"ASVspoof2019_LA_{split_out}", "flac")
    proto_dir = os.path.join(out_root, "ASVspoof2019_LA_cm_protocols")
    os.makedirs(flac_dir, exist_ok=True)
    os.makedirs(proto_dir, exist_ok=True)
    proto_name = f"ASVspoof2019.LA.cm.{split_out}.{PROTO_EXT[split_out]}.txt"
    proto_path = os.path.join(proto_dir, proto_name)

    pf = pq.ParquetFile(pq_path)
    n_written = 0
    n_skipped = 0
    with open(proto_path, "w", encoding="utf-8") as fproto:
        for batch in pf.iter_batches(batch_size=batch_size):
            audio_list = batch.column("audio").field("array").to_pylist()
            fnames = batch.column("audio_file_name").to_pylist()
            spks = batch.column("speaker_id").to_pylist()
            sysids = batch.column("system_id").to_pylist()
            keys = batch.column("key").to_pylist()
            for arr, fn, spk, sysid, k in zip(audio_list, fnames, spks, sysids, keys):
                utt = os.path.splitext(str(fn))[0]
                flac_path = os.path.join(flac_dir, utt + ".flac")
                if os.path.exists(flac_path):
                    n_skipped += 1
                else:
                    x = _norm(np.asarray(arr, dtype=np.float32))
                    sf.write(flac_path, x, 16000, subtype="PCM_16")
                    n_written += 1
                key_str = "bonafide" if int(k) == 0 else "spoof"
                sysid_out = "NA" if key_str == "bonafide" else (str(sysid) if sysid is not None else "NA")
                fproto.write(f"{spk} {utt} - {sysid_out} {key_str}\n")
    print(f"[{split_out}] parquet={os.path.basename(pq_path)} "
          f"flac_written={n_written} skipped={n_skipped} "
          f"protocol={proto_path}")
    return n_written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet_dir", default=str(PARQUET_DIR))
    ap.add_argument("--out_root", default=str(ASVSPOOF_LA))
    ap.add_argument("--batch_size", type=int, default=500)
    ap.add_argument("--splits", nargs="*", default=None,
                    help="限定转换的 split, 如 --splits train validation")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    file_map = {
        "train": os.path.join(args.parquet_dir, "train-00000-of-00001.parquet"),
        "validation": os.path.join(args.parquet_dir, "validation-00000-of-00001.parquet"),
        "test": os.path.join(args.parquet_dir, "test-00000-of-00001.parquet"),
    }
    targets = args.splits or list(file_map.keys())
    for src_split in targets:
        pq_path = file_map[src_split]
        if not os.path.exists(pq_path):
            print(f"[skip] {src_split}: parquet 不存在 -> {pq_path}")
            continue
        convert_one(pq_path, args.out_root, SPLIT_MAP[src_split], args.batch_size)
    print("全部转换完成。")


if __name__ == "__main__":
    main()
