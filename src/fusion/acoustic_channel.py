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
import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import warnings

# 抑制推理时的无害告警（torchaudio 2.9 后端变更提示 / autocast 2.x 弃用提示）
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")
warnings.filterwarnings("ignore", message=".*under the hood.*")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from paths import AASIST_DIR, AASIST_EXP, EXP_DIR, SV_EXAMPLE, DEVICE  # noqa: E402


def _sha256_file(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def _load_quality(exp_dir: str) -> Optional[dict]:
    path = os.path.join(exp_dir, "model_quality.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _quality_for(exp_dir: str, ckpt_path: str,
                 quality: Optional[dict] = None) -> tuple[Optional[float], str]:
    """返回所选权重对应的 dev EER 与来源，优先使用 model_quality.json。"""
    quality = quality if quality is not None else _load_quality(exp_dir)
    name = os.path.basename(ckpt_path)
    if quality:
        if name == str(quality.get("best_ckpt") or "") \
                and isinstance(quality.get("dev_eer"), (int, float)):
            return float(quality["dev_eer"]), "model_quality.json:best_ckpt"
        if name == "swa.pth" and isinstance(quality.get("swa_dev_eer"), (int, float)):
            return float(quality["swa_dev_eer"]), "model_quality.json:swa"
        if name == "best.pth" and isinstance(quality.get("best_ckpt"), str):
            candidate = os.path.join(os.path.dirname(ckpt_path),
                                     os.path.basename(quality["best_ckpt"]))
            if os.path.isfile(candidate) and ckpt_path == candidate \
                    and isinstance(quality.get("dev_eer"), (int, float)):
                return float(quality["dev_eer"]), "model_quality.json:best_ckpt"

    m = re.search(r"epoch_\d+_(\d+\.\d+)\.pth$", name)
    if m:
        return float(m.group(1)), "checkpoint文件名"
    best = None
    wdir = os.path.dirname(ckpt_path)
    for p in glob.glob(os.path.join(wdir, "epoch_*_*.pth")):
        mm = re.search(r"epoch_\d+_(\d+\.\d+)\.pth$", os.path.basename(p))
        if mm:
            value = float(mm.group(1))
            best = value if best is None else min(best, value)
    if best is not None:
        return best, "同目录dev最佳近似"
    return None, "未知"


def _resolve_weight_details(
        exp_dir: str | os.PathLike | None = None
        ) -> tuple[Optional[str], Optional[dict], dict]:
    """只扫描指定实验目录，返回权重、model_config 与可审计元数据。

    选择优先级固定为：
      1) ``model_quality.json.best_ckpt``（训练脚本明确记录的最佳权重）
      2) 同实验 ``best.pth``
      3) 同实验 ``swa.pth``
      4) 同实验 ``epoch_{N}_{EER}.pth`` 中 EER 最小者
      5) 同实验最新编号的 ``epoch_{N}.pth``
    绝不跨实验按 mtime 选权。
    """
    selected_exp = Path(exp_dir or EXP_DIR).resolve()
    meta = {
        "weight_path": None,
        "weight_sha256": None,
        "experiment": selected_exp.name or AASIST_EXP,
        "experiment_dir": str(selected_exp),
        "quality_source": "未知",
        "selection_reason": "no_weight",
        "dev_eer": None,
    }
    weights_dir = selected_exp / "weights"
    if not weights_dir.is_dir():
        return None, None, meta

    quality = _load_quality(str(selected_exp))
    candidates: list[tuple[str, Path]] = []
    if quality and isinstance(quality.get("best_ckpt"), str):
        p = weights_dir / os.path.basename(quality["best_ckpt"])
        if p.is_file():
            candidates.append(("model_quality.json.best_ckpt", p))
    for reason, name in (("experiment_best", "best.pth"),
                         ("experiment_swa", "swa.pth")):
        p = weights_dir / name
        if p.is_file():
            candidates.append((reason, p))

    epoch_scored: list[tuple[float, Path]] = []
    for p in weights_dir.glob("epoch_*_*.pth"):
        m = re.search(r"epoch_\d+_(\d+\.\d+)\.pth$", p.name)
        if m:
            epoch_scored.append((float(m.group(1)), p))
    if epoch_scored:
        epoch_scored.sort(key=lambda item: (item[0], item[1].name))
        candidates.append(("lowest_eer_checkpoint", epoch_scored[0][1]))

    epoch_plain: list[tuple[int, float, Path]] = []
    for p in weights_dir.glob("epoch_*.pth"):
        m = re.fullmatch(r"epoch_(\d+)\.pth", p.name)
        if m:
            epoch_plain.append((int(m.group(1)), p.stat().st_mtime, p))
    if epoch_plain:
        epoch_plain.sort(key=lambda item: (item[0], item[1]), reverse=True)
        candidates.append(("latest_periodic_checkpoint", epoch_plain[0][2]))

    if not candidates:
        return None, None, meta

    reason, weight_path = candidates[0]
    eer, eer_source = _quality_for(str(selected_exp), str(weight_path), quality)
    meta.update({
        "weight_path": str(weight_path),
        "weight_sha256": _sha256_file(str(weight_path)),
        "quality_source": eer_source,
        "selection_reason": reason,
        "dev_eer": eer,
    })
    return str(weight_path), _load_exp_config(str(selected_exp)), meta


def _resolve_weight(exp_dir: str | os.PathLike | None = None
                    ) -> tuple[Optional[str], Optional[dict]]:
    """兼容旧调用：仅返回权重与 model_config。"""
    weight_path, model_config, _ = _resolve_weight_details(exp_dir)
    return weight_path, model_config


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
    return _quality_for(exp_dir, ckpt_path)


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
        self.weight_metadata: dict = {
            "weight_path": weight_path,
            "weight_sha256": None,
            "experiment": "",
            "experiment_dir": "",
            "quality_source": "未知",
            "selection_reason": "explicit_weight" if weight_path else "unresolved",
            "dev_eer": None,
        }
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
            wpath, mc, self.weight_metadata = _resolve_weight_details()
            self.weight_path, self.model_config = wpath, mc
            self._resolved = True
        elif wpath is not None:
            exp_dir = os.path.dirname(os.path.dirname(os.path.abspath(wpath)))
            _, _, metadata = _resolve_weight_details(exp_dir)
            if metadata.get("weight_path") == os.path.abspath(wpath):
                self.weight_metadata = metadata
            else:
                self.weight_metadata.update({
                    "weight_path": os.path.abspath(wpath),
                    "weight_sha256": _sha256_file(wpath),
                    "experiment": os.path.basename(exp_dir),
                    "experiment_dir": exp_dir,
                })
        if wpath is None or mc is None:
            return False

        # 成熟度门控: 依据模型质量决定本通道置信度上限
        exp_dir = os.path.dirname(os.path.dirname(wpath))
        meta_eer = self.weight_metadata.get("dev_eer")
        if isinstance(meta_eer, (int, float)):
            self.eer = float(meta_eer)
            self.eer_source = str(
                self.weight_metadata.get("quality_source") or "未知")
        else:
            self.eer, self.eer_source = _model_quality(exp_dir, wpath)
        self.weight_metadata["dev_eer"] = self.eer
        self.weight_metadata["quality_source"] = self.eer_source
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
