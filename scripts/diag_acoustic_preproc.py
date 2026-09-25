# -*- coding: utf-8 -*-
"""诊断: fusion 预处理 vs 训练链路预处理 在同权重下的分数差异"""
import sys, os, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from paths import AASIST_DIR, EXP_DIR, DEV_FLAC  # noqa: E402

sys.path.insert(0, str(AASIST_DIR))
import numpy as np, torch, torch.nn.functional as F
import soundfile as sf

exp = str(EXP_DIR)
cfg = json.load(open(os.path.join(exp, "config.conf"), encoding="utf-8"))
mc = cfg["model_config"]
from models.AASIST import Model
model = Model(mc)
sd = torch.load(os.path.join(exp, "weights", "best.pth"), map_location="cpu")
model.load_state_dict(sd)
model.eval()

def score(wav):
    x = torch.from_numpy(np.float32(wav)).unsqueeze(0)
    with torch.no_grad():
        _, out = model(x)
    p = F.softmax(out, dim=-1)[0]
    return float(p[1]), float(p[0])

def train_style(path, cut=64600):
    wav = np.trim_zeros(np.float32(sf.read(path)[0]), "b")  # 训练链路: 去尾零
    if len(wav) > cut: wav = wav[:cut]
    if len(wav) < cut: wav = np.pad(wav, (0, cut-len(wav)))
    return wav

def fusion_style(path, ns=48000):
    wav = np.float32(sf.read(path)[0])
    if len(wav) >= ns:
        start = (len(wav)-ns)//2
        wav = wav[start:start+ns]
    else:
        wav = np.tile(wav, ns//len(wav)+1)[:ns]
    return wav

dev = str(DEV_FLAC)
files = [
    ("LA_D_1105538.flac", "真声(家人LA_0069)"),
    ("LA_D_1002910.flac", "伪造(冒充家人)"),
    ("LA_D_1090286.flac", "真声(陌生人LA_0070)"),
]
for fn, tag in files:
    p = os.path.join(dev, fn)
    s1 = score(train_style(p)); s2 = score(fusion_style(p))
    print(f"{tag} {fn}")
    print(f"  训练链路(64600,trim): spoof={s1[0]:.4f} bona={s1[1]:.4f}")
    print(f"  fusion链路(中段48000): spoof={s2[0]:.4f} bona={s2[1]:.4f}")
