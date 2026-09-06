"""
AASIST 微基准 v2: 每阶段完成立即落盘(flush), 分离 数据加载 vs GPU 计算 耗时。
用法: vericall/python scripts/bench_aasist.py
"""
import os, sys, time, argparse
import pathlib
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from paths import AASIST_DIR, ASVSPOOF_LA, BENCH_LOG, CONFIG_DIR  # noqa: E402

sys.path.insert(0, str(AASIST_DIR))
import main as aasist_main
get_loader = aasist_main.get_loader
get_model = aasist_main.get_model

CONF = str(CONFIG_DIR / "AASIST_5060.conf")
DB = str(ASVSPOOF_LA)
N = 10
LOG = str(BENCH_LOG)

import json
config = json.load(open(CONF))


def log(msg):
    with open(LOG, "a") as f:
        f.write(msg + "\n")
        f.flush()
    print(msg, flush=True)


def main():
    open(LOG, "w").close()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device}  N={N}")

    model = get_model(config["model_config"], torch.device(device))
    trn_loader, dev_loader, _ = get_loader(pathlib.Path(DB), 1234, config)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 0.9]).to(device))
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)

    # 1) 仅数据加载
    t0 = time.time()
    batches = []
    it = iter(trn_loader)
    for i in range(N):
        try:
            bx, by = next(it)
        except StopIteration:
            break
        batches.append((bx, by))
    t_data = time.time() - t0
    log(f"[1] 仅数据加载 {len(batches)} batch: {t_data:.2f}s  ({t_data/max(1,len(batches))*1000:.0f} ms/batch)")

    # 2) 仅 GPU 计算 (预取好的 batch 搬到 GPU)
    t0 = time.time()
    for bx, by in batches:
        bx = bx.to(device)
        by = by.view(-1).type(torch.int64).to(device)
        _, out = model(bx)
        loss = criterion(out, by)
        optim.zero_grad(); loss.backward(); optim.step()
    torch.cuda.synchronize()
    t_compute = time.time() - t0
    log(f"[2] 仅 GPU 计算 {len(batches)} batch: {t_compute:.2f}s  ({t_compute/max(1,len(batches))*1000:.0f} ms/batch)")

    # 3) 端到端
    t0 = time.time()
    it2 = iter(trn_loader)
    done = 0
    while done < N:
        try:
            bx, by = next(it2)
        except StopIteration:
            break
        bx = bx.to(device)
        by = by.view(-1).type(torch.int64).to(device)
        _, out = model(bx)
        loss = criterion(out, by)
        optim.zero_grad(); loss.backward(); optim.step()
        done += 1
    torch.cuda.synchronize()
    t_e2e = time.time() - t0
    log(f"[3] 端到端 {done} batch: {t_e2e:.2f}s  ({t_e2e/max(1,done)*1000:.0f} ms/batch)")

    per_epoch_min = t_e2e / max(1, done) * len(trn_loader) / 60
    log(f"[估算] 单 epoch≈{per_epoch_min:.1f} 分钟 (train_loader={len(trn_loader)} batches)")
    if t_e2e > 0:
        log(f"[结论] 数据加载占比≈{t_data/t_e2e*100:.0f}%  纯GPU占比≈{t_compute/t_e2e*100:.0f}%")
    log("DONE")


if __name__ == "__main__":
    main()
