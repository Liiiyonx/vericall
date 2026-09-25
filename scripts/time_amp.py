"""精确测 AMP+cut48000+batch28 的每 batch 耗时, 推算单 epoch 时长。"""
import sys, json, pathlib, time

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
from paths import AASIST_DIR, ASVSPOOF_LA, CONFIG_DIR, DEVICE  # noqa: E402

sys.path.insert(0, str(AASIST_DIR))
import torch
import main as M


def _run():
    config = json.load(open(str(CONFIG_DIR / "AASIST_5060.conf")))
    DB = pathlib.Path(ASVSPOOF_LA)
    device = DEVICE
    M.USE_AMP = True
    model = M.get_model(config["model_config"], torch.device(device))
    trn, dev, ev = M.get_loader(DB, 1234, config)
    crit = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 0.9]).to(device))
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    it = iter(trn)
    N = 15
    t0 = time.time()
    for i in range(N):
        bx, by = next(it)
        bx = bx.to(device); by = by.view(-1).type(torch.int64).to(device)
        with torch.cuda.amp.autocast(enabled=True):
            _, out = model(bx)
            loss = crit(out, by)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    torch.cuda.synchronize()
    dt = time.time() - t0
    per = dt / N * 1000
    print(f"[time] {N} batch 用时 {dt:.1f}s  ({per:.0f} ms/batch)", flush=True)
    per_epoch_min = per / 1000 * len(trn) / 60
    print(f"[time] 单 epoch≈{per_epoch_min:.1f} 分钟 ({len(trn)} batches, {config['num_epochs']} epochs ≈ {per_epoch_min*config['num_epochs']:.1f}h)", flush=True)


if __name__ == "__main__":
    _run()
