# -*- coding: utf-8 -*-
"""
谛听 VeriCall — 端到端融合管线（三通道真实集成）
===============================================================
把三个通道接到一起跑一条真实通话音频，输出最终"放行/警惕/拦截"裁决。

当前状态：
    通道① 声学伪造(AASIST)   -> 真实（加载训练权重做推理；权重未就绪时安全占位不误拦）
    通道② 家人声纹(CAMPPlus) -> 真实（已验证：同人相似度≈1.0 / 异人≈0.03）
    通道③ 话术语义(SenseVoice+Ollama) -> 真实（ASR 转写 + deepseek-r1 风险评分）

GPU 显存管理（RTX 5060 8GB，关键！）：
    模型常驻策略（2026-09 重构）：AASIST 与 CAMPPlus 合计仅 ~0.6GB，常驻不卸载；
    SenseVoice 跑完 ASR 后默认也常驻，仅在调 LLM 前检测到剩余显存不足
    （按 deepseek-r1:8b ≈5GB 估，VERICALL_LLM_VRAM_GB 可调）时才卸载腾位。
    旧行为（每轮全量卸载，单次分析 22-32s）可用 VERICALL_UNLOAD_ASR=1 恢复。

用法：
    python -m fusion.pipeline
    （默认用 SenseVoice 示例音频演示：同说话人->放行，异说话人->拦截）

作为库：
    from fusion.pipeline import VeriCallPipeline
    pipe = VeriCallPipeline()
    pipe.enroll("女儿", "daughter.wav")
    r = pipe.analyze("incoming_call.wav")     # -> FusionResult
    print(r.final, r.rationale)
"""
from __future__ import annotations

import sys
import time
import os

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import SV_EXAMPLE, DEVICE, OFFLINE, DEMO_CACHE  # noqa: E402

from fusion.fusion_orchestrator import FusionOrchestrator, ChannelVerdict
from fusion.voiceprint_channel import VoiceprintChannel
from fusion.semantic_channel import SemanticChannel
from fusion.acoustic_channel import AcousticChannel


def _empty_cache():
    """尽力释放 GPU 显存（模型对象置 None 后调用）。"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# LLM 冷加载所需显存下限（字节）。剩余低于此值时先卸载 ASR 再调 LLM。
# 2026-09-02 实测：8GB 卡上 SenseVoice 常驻后剩 ~5.5GB，r1:8b 冷加载仍 500，
# 阈值从 5.0 上调至 6.0（宁可多卸一次 ASR，也别让 LLM 反复 500 重试拖慢整体）。
_LLM_VRAM_BYTES = int(float(os.environ.get("VERICALL_LLM_VRAM_GB", "6.0")) * 1024 ** 3)
_FORCE_UNLOAD_ASR = os.environ.get("VERICALL_UNLOAD_ASR", "0") == "1"


def _free_vram_bytes() -> float:
    """当前 GPU 剩余可用显存（字节）；无 CUDA 返回 +inf（表示无需腾位）。"""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.mem_get_info()[0]
    except Exception:
        pass
    return float("inf")


class VeriCallPipeline:
    """谛听 VeriCall 端到端管线：声纹 + 话术 + (声学占位) -> 融合裁决。"""

    def __init__(self,
                 voiceprint: VoiceprintChannel | None = None,
                 semantic: SemanticChannel | None = None,
                 acoustic: AcousticChannel | None = None,
                 orchestrator: FusionOrchestrator | None = None,
                 device: str | None = None):
        device = device or DEVICE
        self.vp = voiceprint or VoiceprintChannel()
        self.sem = semantic or SemanticChannel()
        self.ac = acoustic or AcousticChannel(device=device)
        self.orch = orchestrator or FusionOrchestrator()
        self._number_ch = None  # 通道⓪ 号码先验（懒加载）

    # ------------------------------------------------------------------ #
    def enroll(self, name: str, audio_path: str) -> None:
        """登记一位家人的声纹（通道②）。"""
        self.vp.enroll(name, audio_path)

    # ------------------------------------------------------------------ #
    def _voiceprint_verdict(self, audio_path: str) -> ChannelVerdict:
        """通道② 真实核验（CAMPPlus ~0.3GB，常驻不卸载）。"""
        return self.vp.verify(audio_path)

    def _acoustic_verdict(self, audio_path: str) -> ChannelVerdict:
        """通道① 真实 AASIST 声学伪造检测（常驻不卸载）。"""
        return self.ac.analyze(audio_path, unload_after=False)

    def _semantic_verdict(self, audio_path: str) -> ChannelVerdict:
        """通道③ 真实话术分析：ASR 转写 ->（按需腾位）-> LLM 评分。

        调 LLM 前仅当剩余显存不足以容纳大模型（默认按 r1:8b ≈5GB 估）
        才卸载 SenseVoice——显存充裕时 ASR 常驻，省去每次 ~10s 的重载。
        VERICALL_UNLOAD_ASR=1 可强制恢复"每次卸载"旧行为（OOM 兜底）。
        """
        from fusion.transcript_cache import get as cache_get, put as cache_put

        t0 = time.time()
        try:
            transcript = self.sem.asr.transcribe(audio_path)
            if transcript:
                cache_put(audio_path, transcript)   # 跑通即落盘，断网可复用
        except Exception as e:  # noqa: BLE001
            transcript = cache_get(audio_path) or ""
            reason = f"ASR 失败({'缓存命中' if transcript else '缓存缺失'}): {e}"
        else:
            reason = ""
        if self.sem.asr._model is not None and (
                _FORCE_UNLOAD_ASR or _free_vram_bytes() < _LLM_VRAM_BYTES):
            self.sem.asr._model = None
            _empty_cache()

        if not transcript:
            res = self.sem._fallback(reason or "ASR 转写为空", t0)
        else:
            res = self.sem.analyze(transcript)
            res.transcript = transcript
            res.latency_s = round(time.time() - t0, 2)

        conf = 0.9 if res.category != "unknown" else 0.3
        return ChannelVerdict(
            name="semantic",
            score=res.risk,
            label=res.category,
            detail=f"{res.reason} | 转写: {res.transcript[:40]}",
            confidence=conf,
        )

    # ------------------------------------------------------------------ #
    def _number_verdict(self, caller_number: str | None):
        """⓪ 号码先验（懒加载 NumberChannel；无号码返回 None 走三通道）。"""
        if caller_number is None or not str(caller_number).strip():
            return None
        if self._number_ch is None:
            from fusion.number_channel import NumberChannel
            self._number_ch = NumberChannel()
        v = self._number_ch.check(str(caller_number))
        print(f"[通道⓪ 号码] {caller_number} -> {v.source} (score={v.score} "
              f"conf={v.confidence}) {v.detail}")
        return v

    def analyze(self, audio_path: str, scenario: str | None = None,
                caller_number: str | None = None) -> "FusionResult":
        """对一段来电音频跑完整三通道融合，返回最终裁决。

        scenario: 演示场景编号（A/B/C）。离线模式（VERICALL_OFFLINE=1）下
        直接走缓存通道 + 实时规则话术，跳过 Ollama/SenseVoice/torch。
        caller_number: 来电号码（可选）——先走 ⓪ 号码先验，命中即短路拦截，
        跳过三通道推理（省算力）；未命中照常三通道。
        """
        # 通道⓪ 号码先验（本地查表 0ms，命中即短路，最省算力）
        number = self._number_verdict(caller_number)
        if number is not None and number.matched:
            return self.orch.decide(
                self.orch.safe_stub("acoustic", "跳过(号码先验命中)"),
                self.orch.safe_stub("voiceprint", "跳过(号码先验命中)"),
                self.orch.safe_stub("semantic", "跳过(号码先验命中)"),
                number=number)
        if OFFLINE:
            return self._offline_result(scenario)
        print(f"\n=== 分析音频: {audio_path} ===")

        # 通道① 声学伪造：真实 AASIST 推理（load→infer→释放 GPU）
        acoustic = self._acoustic_verdict(audio_path)
        print(f"[通道① 声学] score={acoustic.score} label={acoustic.label} "
              f"({acoustic.confidence}) {acoustic.detail}")

        # 通道② 声纹：真实
        voiceprint = self._voiceprint_verdict(audio_path)
        print(f"[通道② 声纹] score={voiceprint.score} label={voiceprint.label} "
              f"({voiceprint.confidence}) {voiceprint.detail}")

        # 通道③ 话术：真实 ASR + LLM
        semantic = self._semantic_verdict(audio_path)
        print(f"[通道③ 话术] score={semantic.score} label={semantic.label} "
              f"({semantic.confidence}) {semantic.detail}")

        # 融合裁决
        result = self.orch.decide(acoustic, voiceprint, semantic)
        print(f"→ 最终【{result.final}】 汇总可疑度={result.score} "
              f"置信度={result.confidence}")
        print(f"   理由: {result.rationale}")
        return result

    # ------------------------------------------------------------------ #
    def _offline_result(self, scenario: str | None) -> "FusionResult":
        """离线降级：用 demo_cache.json 的缓存通道判决 + 实时规则话术，跑真实融合。

        无 GPU/Ollama/SenseVoice 也能演示 A/B/C 三场景；话术通道仍走零依赖规则引擎，
        融合裁决逻辑完全真实，只是声学/声纹通道用预存缓存。
        """
        import json

        scen = (scenario or "A").upper()
        if not DEMO_CACHE.is_file():
            return FusionResult(
                final="allow", score=0.0, confidence=0.0,
                rationale="【离线演示模式】未找到 demo_cache.json，返回中性占位",
                channels=[], offline=True)
        try:
            cache = json.loads(DEMO_CACHE.read_text(encoding="utf-8"))
            entry = cache.get("scenarios", {}).get(scen)
        except Exception as e:  # noqa: BLE001
            return FusionResult(
                final="allow", score=0.0, confidence=0.0,
                rationale=f"【离线演示模式】缓存读取失败: {e}", channels=[], offline=True)
        if entry is None:
            return FusionResult(
                final="allow", score=0.0, confidence=0.0,
                rationale=f"【离线演示模式】无场景 {scen} 缓存，返回中性占位",
                channels=[], offline=True)

        # 话术通道：实时跑规则引擎（用缓存转写文本）
        transcript = entry.get("transcript", "")
        sem = self.sem._rule_score(transcript)
        semantic = ChannelVerdict(
            name="semantic", score=sem.risk, label=sem.category,
            detail=f"{sem.reason} | 转写: {transcript[:40]}", confidence=0.9)

        # 声学 / 声纹：用预存缓存判决（标注离线来源，不误用为真实推理）
        ac_raw = entry.get("acoustic", {})
        vp_raw = entry.get("voiceprint", {})
        acoustic = ChannelVerdict(
            name=ac_raw.get("name", "acoustic"),
            score=float(ac_raw.get("score", 0.0)),
            label=ac_raw.get("label", "stub"),
            detail=ac_raw.get("detail", "离线缓存"),
            confidence=float(ac_raw.get("confidence", 0.0)))
        voiceprint = ChannelVerdict(
            name=vp_raw.get("name", "voiceprint"),
            score=float(vp_raw.get("score", 0.0)),
            label=vp_raw.get("label", "stub"),
            detail=vp_raw.get("detail", "离线缓存"),
            confidence=float(vp_raw.get("confidence", 0.0)))

        result = self.orch.decide(acoustic, voiceprint, semantic)
        result.offline = True
        result.rationale = "【离线演示模式·缓存通道】" + result.rationale
        return result


# ---------------------------------------------------------------------- #
def _demo():
    zh = str(SV_EXAMPLE / "zh.mp3")
    en = str(SV_EXAMPLE / "en.mp3")
    if not (os.path.isfile(zh) and os.path.isfile(en)):
        print("缺少示例音频，无法演示")
        return

    pipe = VeriCallPipeline()
    # 登记"家人"声纹（用中文示例音频作为家人样本）
    pipe.enroll("家人", zh)

    print("\n########## 场景一：家人本人来电（同说话人）##########")
    r1 = pipe.analyze(zh)

    print("\n########## 场景二：疑似冒用（不同说话人）##########")
    r2 = pipe.analyze(en)

    print("\n=== 汇总 ===")
    print(f"场景一(同人): {r1.final} | 场景二(异人): {r2.final}")


if __name__ == "__main__":
    _demo()
