# -*- coding: utf-8 -*-
"""
通道①-X：XLS-R + 中文域 LR 声学伪造检测（2026-09-06）

背景：AASIST（英文 19LA 训练）对中文域失效（CFAD 真/伪均判真 → 红队击穿 95.5%）。
本通道用 XLS-R 自监督特征 + 中文域训练 LR（aishell 100人真 34,715 + FMFCC 伪 17,636）
替代 AASIST，对中文 TTS/克隆攻击达到红队检出 97.9% / 击穿 0.3%（redblue_round1）。

接口与 AcousticChannel 对齐（analyze→ChannelVerdict），可无缝接入 FusionOrchestrator。
注意：XLS-R 300m + 特征提取耗时 ~0.1-0.5s/条(短音频)，比 AASIST 慢；按需启用。

用法：
  from fusion.xlsr_cn_channel import XlsrCnChannel
  ch = XlsrCnChannel(); ch.load()
  v = ch.analyze("call.wav", unload_after=False)
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import Optional

from fusion.fusion_orchestrator import ChannelVerdict

ROOT = Path(__file__).resolve().parents[2]
SCORER_PKL = ROOT / "data" / "redteam" / "factory" / "cn_lr_scorer_full.pkl"
SSL_LOCAL = "D:/VeriCall_data/models/wav2vec2-xls-r-300m"


class XlsrCnChannel:
    """通道①-X：XLS-R + 中文域 LR 声学伪造检测。

    scorer 选择：'v4'（默认，aishell+FMFCC，专注 TTS/克隆，红队检出 97.9%）
              或 'wide'（aishell+FMFCC+CFAD伪，全谱，红队 ~96% 且 CFAD 47→28%）。
    """

    def __init__(self, scorer: str = "v4",
                 device: Optional[str] = None):
        self.scorer = scorer
        self.scorer_pkl = SCORER_PKL if scorer == "v4" else SCORER_PKL.with_name("cn_lr_scorer_wide.pkl")
        self._ssl = None
        self._fe = None
        self._clf = None
        self._device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                self._device = "cuda"
        except Exception:
            self._device = "cpu"
        if device:
            self._device = device
        if scorer == "wide":
            self.eer = 28.2            # CFAD 严谨 3 折均值
            self.eer_source = "宽覆盖：CFAD 排除锚点 3 折 ~28% / 红队检出 ~96%"
        else:
            self.eer = 0.0             # 同域说话人外 0.00%
            self.eer_source = "aishell 90/10 说话人外同域评测"
        self.cap: float = 0.95         # 高置信
        self.cap_reason = f"中文域训练（{scorer}），同域/红队双验通过"

    def load(self) -> bool:
        """加载 XLS-R + 中文域 LR。依赖缺时返回 False（离线演示机回退）。"""
        try:
            import torch  # noqa: F401
            from transformers import AutoFeatureExtractor, AutoModel
            if not self.scorer_pkl.exists():
                return False
            self._fe = AutoFeatureExtractor.from_pretrained(SSL_LOCAL)
            self._ssl = AutoModel.from_pretrained(SSL_LOCAL).to(self._device).eval()
            with open(self.scorer_pkl, "rb") as f:
                self._clf = pickle.load(f)
            return True
        except Exception:
            self._ssl = self._fe = self._clf = None
            return False

    def _load_waveform(self, path: str, target_sr: int = 16000):
        import librosa
        import soundfile as sf
        wav, sr = sf.read(path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != target_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        if len(wav) > target_sr * 30:
            wav = wav[: target_sr * 30]
        return wav

    def analyze(self, audio_path: str,
                unload_after: bool = True) -> "ChannelVerdict":
        """分析音频，返回 ChannelVerdict（score=spoof 概率，高=疑似 AIGC）。"""
        import torch
        if self._ssl is None or self._clf is None:
            return ChannelVerdict(
                name="xlsr_cn", score=0.0, label="no_model",
                detail="XLS-R/中文域LR 未加载", confidence=0.0)
        try:
            wav = self._load_waveform(audio_path)
            inp = self._fe([wav], sampling_rate=16000, return_tensors="pt").to(self._device)
            with torch.no_grad():
                h = self._ssl(**inp).last_hidden_state.mean(dim=1).cpu().numpy()
            spoof_prob = float((1.0 - self._clf.predict_proba(h)[:, 1])[0])
            label = "spoof" if spoof_prob >= 0.5 else "bonafide"
            margin = abs(spoof_prob - 0.5)
            if unload_after:
                self.unload()
            return ChannelVerdict(
                name="xlsr_cn", score=round(spoof_prob, 3), label=label,
                detail=f"XLS-R+中文域LR spoof概率={spoof_prob:.3f} (同域EER 0.00% 参考)",
                confidence=round(min(margin * 2, 0.99), 3))
        except Exception as e:  # noqa: BLE001
            if unload_after:
                self.unload()
            return ChannelVerdict(
                name="xlsr_cn", score=0.0, label="error",
                detail=f"{type(e).__name__}: {e}", confidence=0.0)

    def unload(self):
        """释放 GPU。"""
        try:
            if self._ssl is not None:
                import torch
                self._ssl = self._fe = None
                if self._device == "cuda":
                    torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    import sys
    ch = XlsrCnChannel()
    print("load:", ch.load())
    if len(sys.argv) > 1:
        v = ch.analyze(sys.argv[1], unload_after=True)
        print(f"verdict: {v}")
