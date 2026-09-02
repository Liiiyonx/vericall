#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用 huggingface_hub + hf_xet 从 HF CDN 拉取 ASVspoof 2019 LA 的 parquet
(Bisher/ASVspoof_2019_LA)。Xet 分块文件经 CDN 重建, 通常比 datashare 直连快得多。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from paths import PARQUET_DIR  # noqa: E402

os.environ["HF_HUB_ENABLE_XET"] = "1"
from huggingface_hub import hf_hub_download

REPO = "Bisher/ASVspoof_2019_LA"
LOCAL = str(PARQUET_DIR)
FILES = [
    "data/train-00000-of-00001.parquet",
    "data/validation-00000-of-00001.parquet",
    # 评测集较大, 先拉训练/验证用于基线训练; 如需评测再补 test
    # "data/test-00000-of-00001.parquet",
]

if __name__ == "__main__":
    os.makedirs(LOCAL, exist_ok=True)
    for fn in FILES:
        print(f"[xet] downloading {fn} ...", flush=True)
        try:
            p = hf_hub_download(repo_id=REPO, filename=fn,
                                repo_type="dataset", local_dir=LOCAL)
            print(f"[xet] DONE -> {p}", flush=True)
        except Exception as e:
            print(f"[xet] FAILED {fn}: {e!r}", flush=True)
            sys.exit(1)
    print("[xet] all parquet downloaded.", flush=True)
