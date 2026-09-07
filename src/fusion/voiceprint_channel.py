# -*- coding: utf-8 -*-
"""
通道② 家人声纹确认（真实说话人验证版）
=========================================
职责：判断"正在通话的人是不是登记的家人本人"。

做法（全本地、零 API）：
- 用 modelscope 的轻量说话人验证模型 CAMPPlus（speech_campplus_sv_zh-cn_16k，
  ~30MB，首次自动下载缓存）提取 192 维声纹向量。
- enroll(家人名, 音频) 登记声纹；verify(音频) 与已登记声纹做余弦相似度。
- 输出 ChannelVerdict：score = 可疑度（越高越不像本人），供融合编排器消费。

与通道①③的关系：
    通道① 声学伪造(AASIST) -> 是不是合成语音
    通道② 本模块           -> 是不是登记的家人本人   ← 这里
    通道③ 话术语义(SenseVoice+Ollama) -> 内容是不是诈骗话术

设计要点：
- 懒加载：import 不依赖 funasr/模型；首次 enroll/verify 才加载（自动下载 CAMPPlus）。
- 声纹相似度 -> 可疑度映射：相似度>=MATCH_T(0.520，2026-09-07 标定) 视为本人(score≈0)，
  低于阈值则可疑度随差距上升；低于 REJECT_T(0.443) 视为"完全不像本人"(score→1)。
  出处：evaluation/voiceprint_calibration.md（200 人合库 EER 0.51%）。
- 未登记任何家人时 verify 返回低置信 stub（融合器不会误拦）。
- 关于"克隆语音"：CAMPPlus 对高保真声纹克隆有一定区分力但非绝对；
  这正是为何本系统要三通道融合——声纹被攻破时，通道①③仍能兜底。

用法（库）：
    from fusion.voiceprint_channel import VoiceprintChannel
    vp = VoiceprintChannel()
    vp.enroll("女儿", "enroll_daughter.wav")
    v = vp.verify("incoming_call.wav")      # -> ChannelVerdict(score,label,detail,confidence)
    print(v.score, v.label)                 # 越低越像本人

命令行自测：
    python voiceprint_channel.py --enroll 女儿=zh.mp3 --verify en.mp3
    （用 SenseVoice 示例音频演示：同文件自比应为 match；不同语种/说话人应为 mismatch）
"""
from __future__ import annotations

import os
import sys
import time
import json
import numpy as np
from typing import Dict, Optional, Tuple

# CAMPPlus 说话人验证模型（modelscope，首次自动下载 ~28MB）
# 注意：模型 ID 必须带 -common 后缀，否则 modelscope 返回 404（record not found）。
# 该模型是 funasr 内置 cam++ 模型类（仓库无 model.py），不要用 trust_remote_code。
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from paths import DEVICE, SV_EXAMPLE, VOICEPRINT_DIR  # noqa: E402

SV_MODEL_ID = "iic/speech_campplus_sv_zh-cn_16k-common"
DEVICE = "cuda:0" if DEVICE == "cuda" else "cpu"  # paths.DEVICE 由 VERICALL_DEVICE 控制

# 相似度阈值（余弦，CAMPPlus 192 维；同人通常 0.6~0.9，异人 0.1~0.4）
# 【科学标定 2026-09-07 · 出处 evaluation/voiceprint_calibration.md】
#   200 人合库（AISHELL-1 100 + AISHELL-3 100）K=12 抽话语两两比对：
#   EER 0.51% @0.472；工作点 κ∈{1:1, 3:1误拒贵, 1:3误受贵} → 阈值 0.471/0.443/0.520。
#   映射：MATCH = 0.520（1:3 误受贵点：高置信才认"本人"，少把陌生人放成家人）；
#         REJECT = 0.443（3:1 误拒贵点之下判"非家人"高可疑，宁可疑不误放）；
#         中间 [0.443, 0.520) 为"不确定"→ 线性映射 + unknown 标签（融合器 caution）。
MATCH_THRESHOLD = 0.520   # 高于此视为"本人"（= 误受3x贵工作点，FAR 0.17%）
REJECT_THRESHOLD = 0.443  # 低于此视为"完全不像本人"（= 误拒3x贵工作点，FRR 0.26%）


class VoiceprintChannel:
    """通道②：家人声纹确认。线程不安全（模型与声纹表有状态）。"""

    def __init__(self, model_id: str = SV_MODEL_ID, device: str = DEVICE):
        self.model_id = model_id
        self.device = device
        self._model = None
        self.profiles: Dict[str, np.ndarray] = {}  # 家人名 -> 归一化声纹向量
        self.load_profiles()                       # 启动时自动加载落盘声纹

    # ------------------------------------------------------------------ #
    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from funasr import AutoModel
        except ImportError as e:
            raise RuntimeError(
                "funasr 未安装，请先: pip install funasr modelscope "
                f"(-i https://pypi.tuna.tsinghua.edu.cn/simple) -> {e}")
        print(f"[声纹] 加载 CAMPPlus 说话人模型 @ {self.model_id} ...")
        t0 = time.time()
        self._model = AutoModel(
            model=self.model_id,
            device=self.device,
            disable_update=True,
        )
        print(f"[声纹] 加载完成 ({time.time()-t0:.1f}s)")
        return self._model

    def _embed(self, audio_path: str) -> np.ndarray:
        """提取归一化声纹向量（L2 norm=1）。"""
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(audio_path)
        model = self._load()
        res = model.generate(input=audio_path, batch_size_s=60)
        raw = None
        for key in ("spk_embedding", "embedding", "spk_feature"):
            if res and isinstance(res[0], dict) and res[0].get(key) is not None:
                raw = res[0][key]
                break
        if raw is None:
            raise RuntimeError(f"模型未返回声纹向量, 实际键: {list(res[0].keys()) if res else None}")
        # 模型在 GPU 上推理，返回的张量在 cuda 设备，需先 .cpu() 再转 numpy
        if hasattr(raw, "cpu"):
            raw = raw.detach().cpu().numpy()
        elif hasattr(raw, "detach"):
            raw = raw.detach().numpy()
        vec = np.asarray(raw, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(vec)
        if norm < 1e-8:
            raise RuntimeError("声纹向量为零向量（音频可能无声/过短）")
        return vec / norm

    # ------------------------------------------------------------------ #
    def enroll(self, name: str, audio_path: str) -> None:
        """登记一位家人的声纹（可重复覆盖；多样本自动平均后重归一化更稳）。"""
        vec = self._embed(audio_path)
        self._save_profile(name, vec, source_file=os.path.basename(audio_path))
        print(f"[声纹] 已登记家人「{name}」（当前共 {len(self.profiles)} 位）")

    # ------------------------------------------------------------------ #
    # 落盘 / 加载（P0-6：声纹库持久化，重启不丢）
    def _profile_dir(self, name: str) -> Path:
        return VOICEPRINT_DIR / name

    def _profile_vec_path(self, name: str) -> Path:
        return self._profile_dir(name) / "embedding.npy"

    def _meta_path(self, name: str) -> Path:
        return self._profile_dir(name) / "meta.json"

    def _load_profile_vec(self, name: str) -> Optional[np.ndarray]:
        p = self._profile_vec_path(name)
        if p.is_file():
            try:
                return np.load(str(p)).astype(np.float32)
            except Exception:  # noqa: BLE001
                return None
        return None

    def _load_meta(self, name: str) -> Optional[dict]:
        p = self._meta_path(name)
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                return None
        return None

    def _save_profile(self, name: str, vec: np.ndarray, source_file: str = "") -> None:
        """写盘一位家人的声纹向量 + meta；已有则取平均后重归一化（抗信道/感冒波动）。"""
        d = self._profile_dir(name)
        d.mkdir(parents=True, exist_ok=True)
        existing = self._load_profile_vec(name)
        samples = 1
        if existing is not None:
            combined = (existing + vec) / 2.0
            n = float(np.linalg.norm(combined))
            combined = combined / n if n > 1e-8 else combined
            vec = combined
            prev = self._load_meta(name) or {}
            samples = int(prev.get("samples", 1)) + 1
        np.save(str(self._profile_vec_path(name)), vec.astype(np.float32))
        self._save_meta(name, {
            "source_file": source_file,
            "enroll_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "samples": samples,
        })
        self.profiles[name] = vec

    def _save_meta(self, name: str, obj: dict) -> None:
        try:
            self._meta_path(name).write_text(
                json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass

    def load_profiles(self) -> None:
        """启动时自动加载 data/voiceprints/ 下所有落盘声纹（不依赖 funasr）。"""
        if not VOICEPRINT_DIR.is_dir():
            return
        for d in sorted(VOICEPRINT_DIR.iterdir()):
            if not d.is_dir():
                continue
            vec = self._load_profile_vec(d.name)
            if vec is not None:
                self.profiles[d.name] = vec

    def verify(self, audio_path: str) -> "ChannelVerdict":
        """验证来话者是否为已登记家人。返回 ChannelVerdict（score=可疑度）。"""
        from fusion.fusion_orchestrator import ChannelVerdict
        t0 = time.time()
        if not self.profiles:
            return ChannelVerdict(
                name="voiceprint", score=0.0, label="stub",
                detail="未登记任何家人声纹", confidence=0.0,
            )
        try:
            vec = self._embed(audio_path)
        except Exception as e:  # noqa: BLE001
            return ChannelVerdict(
                name="voiceprint", score=0.5, label="unknown",
                detail=f"声纹提取失败: {e}", confidence=0.3)

        best_name, best_sim = self._best_match(vec)
        # 可疑度：相似度高 -> 低可疑；低于拒绝阈值 -> 高可疑
        if best_sim >= MATCH_THRESHOLD:
            score = max(0.0, (MATCH_THRESHOLD - best_sim) / MATCH_THRESHOLD)
            label = "match"
        elif best_sim <= REJECT_THRESHOLD:
            score = 1.0
            label = "mismatch"
        else:
            # 线性映射到 (REJECT, MATCH) 区间
            score = (MATCH_THRESHOLD - best_sim) / (MATCH_THRESHOLD - REJECT_THRESHOLD)
            label = "mismatch"
        score = float(np.clip(score, 0.0, 1.0))
        return ChannelVerdict(
            name="voiceprint",
            score=round(score, 2),
            label=label,
            detail=f"最似「{best_name}」相似度={best_sim:.2f}",
            confidence=0.85 if label != "unknown" else 0.4,
        )

    def _best_match(self, vec: np.ndarray) -> Tuple[str, float]:
        best_name, best_sim = "", -1.0
        for name, p in self.profiles.items():
            sim = float(np.dot(vec, p))  # 均已 L2 归一化 -> 余弦=点积
            if sim > best_sim:
                best_sim, best_name = sim, name
        return best_name, best_sim


# ---------------------------------------------------------------------- #
def _demo():
    if len(sys.argv) > 1 and sys.argv[1] == "--enroll":
        # 用法: --enroll 女儿=zh.mp3 --verify en.mp3
        vp = VoiceprintChannel()
        pending_verify = None
        i = 2
        while i < len(sys.argv):
            a = sys.argv[i]
            if a.startswith("--enroll") and "=" in a:
                name, path = a.split("=", 1)
                vp.enroll(name, path)
            elif a == "--verify" and i + 1 < len(sys.argv):
                pending_verify = sys.argv[i + 1]
                i += 1
            i += 1
        if pending_verify:
            v = vp.verify(pending_verify)
            print(f"验证结果: score={v.score} label={v.label} ({v.latency_s}s) {v.detail}")
    else:
        # 默认演示：用 SenseVoice 示例音频自比/跨比
        zh = str(SV_EXAMPLE / "zh.mp3")
        en = str(SV_EXAMPLE / "en.mp3")
        if not (os.path.isfile(zh) and os.path.isfile(en)):
            print("缺少示例音频，无法演示")
            return
        vp = VoiceprintChannel()
        vp.enroll("示例说话人", zh)
        v_self = vp.verify(zh)     # 同文件自比 -> 应 match
        v_other = vp.verify(en)     # 不同语种/说话人 -> 应 mismatch
        print(f"[自比 zh]  score={v_self.score} label={v_self.label} {v_self.detail}")
        print(f"[跨比 en]  score={v_other.score} label={v_other.label} {v_other.detail}")


if __name__ == "__main__":
    _demo()
