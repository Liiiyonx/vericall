"""极速 AMP 冒烟: 5 batch 确认混合精度训练无形状/运行时错误。
注意: 整段逻辑包在 __main__ 守卫内, 否则 Windows spawn 的 DataLoader worker
会重跑本文件导致递归。"""
import sys, json, pathlib

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
    print(f"[smoke] train_loader batches={len(trn)}, cut={config['model_config']['nb_samp']}", flush=True)
    it = iter(trn)
    for i in range(5):
        bx, by = next(it)
        bx = bx.to(device)
        by = by.view(-1).type(torch.int64).to(device)
        with torch.cuda.amp.autocast(enabled=True):
            _, out = model(bx)
            loss = crit(out, by)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        print(f"[smoke] batch {i} loss={loss.item():.4f} out_shape={tuple(out.shape)} OK", flush=True)
    print("[smoke] AMP 训练 5 batch 通过, 无错误", flush=True)


if __name__ == "__main__":
    _run()
