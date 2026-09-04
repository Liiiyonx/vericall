# -*- coding: utf-8 -*-
"""
谛听 VeriCall — 通道① 声学伪造检测 (AASIST) 真实集成
=====================================================
加载训练好的 AASIST 模型，对一段 16kHz 单声道语音做"合成/克隆"概率判定，
输出融合器约定的 ChannelVerdict(score=伪造概率 0~1, label, detail, confidence)。

模型要点（已核对 AASIST.py / data_utils.py）：
  * 输入是「原始波形」raw waveform，形状 [1, nb_samp]，采样率 16kHz。
    AASIST 的 conv_time 是 SincConv（可学习带通滤波器），直接吃波形，
    不用手工特征（非 LFCC）。故推理前只需：读音频→重采样16k→单声道→裁剪/补齐到 nb_samp。
  * 输出是 2 类 logits。注意：原版 AASIST 的 data_utils 标签映射是
    bonafide=1、spoof=0（与直觉相反！）——训练落盘 dev_score.txt 的
    out[:,1] 分数越高越"真"。故 softmax 第 0 维才是"伪造概率"，
    与 main.py produce_evaluation_file 的 batch_out[:,1]=bonafide 分数口径一致。

与融合器解耦：
  * 正常：score = 伪造概率（越高越疑似 AIGC 合成/克隆语音）。
  * 权重未就绪 / 推理异常：返回 confidence=0 的占位判决，绝不误拦。

GPU 显存（RTX 5060 8GB，关键！）：
  与 pipeline 的串行策略一致——analyze 内部 load→infer→unload，
  任意时刻 GPU 上只此一个大模型，避免与通道②③叠加 OOM。
"""
from __future__ import annotations

import os
import re
import sys
import glob
import json
from typing import Optional

import numpy as np
import warnings

# 抑制推理时的无害告警（torchaudio 2.9 后端变更提示 / autocast 2.x 弃用提示）
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")
warnings.filterwarnings("ignore", message=".*under the hood.*")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from paths import AASIST_DIR, SV_EXAMPLE, DEVICE  # noqa: E402


def _resolve_weight() -> tuple[Optional[str], Optional[dict]]:
    """在 exp_result/*/weights 下找最佳权重及其 model_config。

    优先级：
      1) best.pth（取最新修改的，训练完成产物）
      2) swa.pth（最终 SWA 模型）
      3) epoch_{N}_{EER}.pth 中 EER 最小者
    找到权重后，读取其实验目录下的 config.conf 得到 model_config（架构/nb_samp 等），
    这样通道①完全由训练产物驱动，无需硬编码。
    """
    if not os.path.isdir(AASIST_DIR):
        return None, None
    exp_root = os.path.join(AASIST_DIR, "exp_result")
    if not os.path.isdir(exp_root):
        return None, None

    def cfg_of(weight_path: str) -> Optional[dict]:
        exp_dir = os.path.dirname(os.path.dirname(weight_path))
        return _load_exp_config(exp_dir)

    # 1) best.pth（最新修改）
    best = []
    for p in glob.glob(os.path.join(exp_root, "*", "weights", "best.pth")):
        best.append((os.path.getmtime(p), p))
    if best:
        best.sort(reverse=True)
        w = best[0][1]
        return w, cfg_of(w)

    # 2) swa.pth（最新修改）
    swa = []
    for p in glob.glob(os.path.join(exp_root, "*", "weights", "swa.pth")):
        swa.append((os.path.getmtime(p), p))
    if swa:
        swa.sort(reverse=True)
        w = swa[0][1]
        return w, cfg_of(w)

    # 3) epoch_{N}_{EER}.pth（EER 最小，历史遗留产物）
    ep = []
    for p in glob.glob(os.path.join(exp_root, "*", "weights", "epoch_*.pth")):
        m = re.search(r"epoch_\d+_(\d+\.\d+)\.pth$", p)
        if m:
            ep.append((float(m.group(1)), p))
    if ep:
        ep.sort()  # EER 升序
        w = ep[0][1]
        return w, cfg_of(w)

    # 4) epoch_{N}.pth（skip 模式周期性兜底权重，无 EER；取编号最大、mtime 最新者）
    epn = []
    for p in glob.glob(os.path.join(exp_root, "*", "weights", "epoch_*.pth")):
        m = re.match(r".*epoch_(\d+)\.pth$", p)
        if m:
            epn.append((int(m.group(1)), os.path.getmtime(p), p))
    if epn:
        epn.sort(key=lambda t: (t[0], t[1]), reverse=True)
        w = epn[0][2]
        return w, cfg_of(w)

    return None, None


def _load_exp_config(exp_dir: str) -> Optional[dict]:
    cfg_path = os.path.join(exp_dir, "config.conf")
    if not os.path.isfile(cfg_path):
        return None
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("model_config")
    except Exception:
        return None


def _model_quality(exp_dir: str, ckpt_path: str) -> tuple[Optional[float], str]:
    """估计当前权重的模型质量（dev EER 百分比），用于成熟度门控。

    优先级:
      1) 实验目录 model_quality.json（训练结束时 main.py 写入，最准）
      2) 权重文件名里的 EER（epoch_{N}_{EER}.pth，训练中刷新最佳时产物）
      3) 同 weights 目录里 epoch_{N}_{EER}.pth 的最小 EER（近似）
      4) 未知 -> None
    """
    q = os.path.join(exp_dir, "model_quality.json")
    if os.path.isfile(q):
        try:
            with open(q, "r", encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d.get("dev_eer"), (int, float)):
                return float(d["dev_eer"]), "model_quality.json"
        except Exception:
            pass
    m = re.search(r"epoch_\d+_(\d+\.\d+)\.pth$", os.path.basename(ckpt_path))
    if m:
        return float(m.group(1)), "checkpoint文件名"
    best = None
    wdir = os.path.dirname(ckpt_path)
    for p in glob.glob(os.path.join(wdir, "epoch_*_*.pth")):
        mm = re.search(r"epoch_\d+_(\d+\.\d+)\.pth$", os.path.basename(p))
        if mm:
            v = float(mm.group(1))
            best = v if best is None else min(best, v)
    if best is not None:
        return best, "同目录dev最佳近似"
    return None, "未知"


def _confidence_cap(eer: Optional[float]) -> tuple[float, str]:
    """按模型质量给置信度封顶 —— 成熟度门控。

    动机: 融合器的硬拦截要求单通道 score>=0.85 且 confidence>=0.6。
    一个 dev EER 还有 16% 的半成品模型完全可能把真声判到 0.9+，
    若不限流就会把家人的正常来电拦掉（实测发生过）。
    封顶后: 硬拦截(需 >=0.6)与单通道升级(需 >=0.6)都不会被半成品触发，
    但分数仍进入加权汇总，双通道互相印证时照常拦截。
    """
    if eer is None:
        return 0.75, "模型质量未知"
    if eer <= 2.0:
        return 1.00, "成熟"
    if eer <= 5.0:
        return 0.75, "接近成熟"
    return 0.55, "未成熟"


class AcousticChannel:
    """通道① AASIST 声学伪造检测。load→infer→unload 串行释放 GPU。"""

    def __init__(self, weight_path: Optional[str] = None,
                 model_config: Optional[dict] = None,
                 device: Optional[str] = None):
        self.weight_path = weight_path
        self.model_config = model_config
        if device:
            self.device = device
        elif DEVICE == "cpu":
            self.device = "cpu"
        else:
            # torch 懒加载：未安装时（离线演示机）不报错，回退 cpu
            self.device = "cpu"
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
            except Exception:
                self.device = "cpu"
        self._model = None
        self._resolved = False
        self.eer: Optional[float] = None       # 模型 dev EER (百分比), 未知为 None
        self.eer_source: str = "未知"
        self.cap: float = 0.75                 # 置信度封顶
        self.cap_reason: str = "模型质量未知"

    # ------------------------------------------------------------------ #
    def load(self) -> bool:
        """加载 AASIST 权重到 GPU。返回是否成功。"""
        if self._model is not None:
            return True
        wpath, mc = self.weight_path, self.model_config
        if (wpath is None or mc is None) and not self._resolved:
            wpath, mc = _resolve_weight()
            self.weight_path, self.model_config = wpath, mc
            self._resolved = True
        if wpath is None or mc is None:
            return False

        # 成熟度门控: 依据模型质量决定本通道置信度上限
        exp_dir = os.path.dirname(os.path.dirname(wpath))
        self.eer, self.eer_source = _model_quality(exp_dir, wpath)
        self.cap, self.cap_reason = _confidence_cap(self.eer)

        _aasist_str = str(AASIST_DIR)
        if _aasist_str not in sys.path:
            sys.path.insert(0, _aasist_str)
        from models.AASIST import Model
        import torch
        model = Model(mc).to(self.device)
        sd = torch.load(wpath, map_location=self.device)
        model.load_state_dict(sd)
        model.eval()
        self._model = model
        return True

    def unload(self):
        """释放模型与 GPU 显存（配合 pipeline 串行策略）。"""
        self._model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_waveform(path: str, target_sr: int = 16000,
                       nb_samp: int = 48000) -> np.ndarray:
        """读任意音频→16k 单声道→裁剪/补齐到 nb_samp 的 float32 数组。

        优先 torchaudio（mp3/wav/flac 通吃），失败回退 soundfile，
        再回退 scipy/原始；重采样用 torchaudio，缺失则线性插值兜底。
        """
        wav = None
        sr = None
        # 1) torchaudio（最通用）
        try:
            import torchaudio
            wav_t, sr = torchaudio.load(path)  # (channels, n)
            wav = wav_t.numpy()
        except Exception:
            wav = None
        # 2) soundfile（wav/flac，部分环境支持 mp3）
        if wav is None:
            try:
                import soundfile as sf
                wav, sr = sf.read(path, always_2d=True)  # (n, channels)
                wav = wav.T  # (channels, n)
            except Exception:
                wav = None
        if wav is None:
            raise RuntimeError(f"无法读取音频（缺 torchaudio/soundfile）: {path}")

        # 转单声道
        if wav.ndim == 2 and wav.shape[0] > 1:
            wav = wav.mean(axis=0)
        else:
            wav = wav.reshape(-1)
        wav = wav.astype(np.float32)

        # 重采样到 target_sr
        if int(sr) != target_sr:
            try:
                import torchaudio
                from torchaudio.functional import resample as ta_resample
                wav = ta_resample(torch.from_numpy(wav).float(),
                                  int(sr), target_sr).numpy()
            except Exception:
                n = int(round(len(wav) * target_sr / int(sr)))
                wav = np.interp(np.linspace(0, len(wav) - 1, max(n, 1)),
                                np.arange(len(wav)), wav)

        # 裁剪/补齐到 nb_samp
        if len(wav) >= nb_samp:
            start = (len(wav) - nb_samp) // 2  # 取中段，避开首尾静音偏置
            wav = wav[start:start + nb_samp]
        else:
            num_repeats = int(nb_samp / len(wav)) + 1
            wav = np.tile(wav, num_repeats)[:nb_samp]
        return wav.astype(np.float32)

    # ------------------------------------------------------------------ #
    def analyze(self, audio_path: str,
                nb_samp: Optional[int] = None,
                unload_after: bool = True) -> "ChannelVerdict":
        """分析一个音频文件。

        unload_after=False 时不卸载模型（常驻模式）：AASIST 显存仅 ~0.3GB，
        常驻可把"每次分析/流式每窗"的重复加载开销归零。批量离线脚本保持
        默认 True 不变；pipeline 与流式调度传 False。
        """
        from fusion.fusion_orchestrator import ChannelVerdict

        if not self.load():
            return ChannelVerdict(
                name="acoustic", score=0.0, label="no_model",
                detail="AASIST 权重未就绪（训练中/未训练）", confidence=0.0)
        try:
            import torch
            import torch.nn.functional as F
            ns = nb_samp or int(self.model_config.get("nb_samp", 48000))
            wav = self._load_waveform(audio_path, 16000, ns)
            x = torch.from_numpy(wav).float().unsqueeze(0).to(self.device)  # [1, ns]
            use_amp = (self.device == "cuda")
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=use_amp):
                    _, out = self._model(x)
            probs = F.softmax(out, dim=-1)[0]
            # 原版 AASIST 标签约定: bonafide=1, spoof=0（经 dev_score.txt 逐条核对）
            spoof_prob = float(probs[0].item())
            bonafide_prob = float(probs[1].item())

            # 置信度：用 |spoof_prob-0.5| 衡量模型区分度（越偏离 0.5 越自信），
            # 再按模型成熟度封顶 —— 未成熟的模型不允许单独触发硬拦截。
            margin = abs(spoof_prob - 0.5)
            confidence = min(1.0, 0.4 + margin * 1.2, self.cap)
            label = "spoof" if spoof_prob >= 0.5 else "bonafide"
            eer_txt = ("devEER={:.1f}%".format(self.eer)
                       if self.eer is not None else "devEER未知")
            detail = (f"AASIST 合成概率={spoof_prob:.3f} "
                      f"(真实概率={bonafide_prob:.3f}) | {eer_txt} "
                      f"[{self.cap_reason}, 置信度上限{self.cap:.2f}]")
            return ChannelVerdict(
                name="acoustic",
                score=round(spoof_prob, 3),
                label=label,
                detail=detail,
                confidence=round(confidence, 2),
            )
        except Exception as e:  # noqa: BLE001
            return ChannelVerdict(
                name="acoustic", score=0.0, label="error",
                detail=f"AASIST 推理异常: {e}", confidence=0.0)
        finally:
            if unload_after:
                self.unload()


# ---------------------------------------------------------------------- #
def _demo():
    ac = AcousticChannel()
    zh = str(SV_EXAMPLE / "zh.mp3")
    if not os.path.isfile(zh):
        print("缺少示例音频")
        return
    v = ac.analyze(zh)
    print(f"[通道① 声学] score={v.score} label={v.label} "
          f"({v.confidence}) {v.detail}")


if __name__ == "__main__":
    _demo()
