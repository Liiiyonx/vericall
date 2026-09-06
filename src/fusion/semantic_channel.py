# -*- coding: utf-8 -*-
"""
通道③ 话术语义风险分析（通道③ + 真实 ASR 版）
=============================================
职责：
  1. [ASR]   接收通话音频 -> 本地 SenseVoice 转写为文本（中文/方言/英文多语种）。
  2. [LLM]   转写文本 -> 诈骗话术风险分与类别（**默认云端 DeepSeek deepseek-chat**，
             仅显式 VERICALL_SEMANTIC_LLM=ollama 时才走本地 Ollama）。

架构位置：
    通道① 声学伪造检测（AASIST）     -> "是不是合成语音"
    通道② 声纹确认（家人声纹比对）   -> "是不是本人"
    通道③ 本模块                     -> "内容是不是诈骗话术"  ← 这里

设计要点：
- LLM 后端（默认 cloud，铁律：不用本地 Ollama，太慢）：
    cloud → DeepSeek OpenAI 兼容 API（SCAM_LLM_BASE/SCAM_LLM_KEY/SCAM_LLM_MODEL，
            密钥由 D:/VeriCall_data/secrets/vericall_secrets.env 提供；无密钥不静默回退）
    ollama → 仅显式 VERICALL_SEMANTIC_LLM=ollama 时启用（本地 paths.OLLAMA_HOST）
- ASR 懒加载：import 本模块不依赖 funasr；只有调用 transcribe/analyze_audio 时才加载模型。
- ⚠️ 显存约束仅存在于 ollama 逃生口（RTX 5060 8GB）：SenseVoice（ASR）与 r1:8b 不能同占 GPU。
  cloud 模式 LLM 在云端，无此约束。
- 输出统一为 SemanticResult（risk/category/reason/latency/model/backend），供三通道决策融合。
- 任意环节失败都返回带 reason 的兜底结果，融合层按"低置信"处理，绝不静默吞错。

用法（作为库）：
    from fusion.semantic_channel import SemanticChannel
    ch = SemanticChannel()                       # 默认云端 DeepSeek（无密钥→规则兜底）
    r = ch.analyze("妈，是我，手机摔了，急用五万块...")        # 纯文本
    r = ch.analyze_audio("call_20260831.wav")                  # 音频 -> ASR -> 风险

命令行自测：
    python semantic_channel.py "任意待测文本"
    python semantic_channel.py --audio <SenseVoiceSmall>/example/zh.mp3
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

# 让本模块既能被 `from fusion.semantic_channel import` 导入，
# 也能 `python semantic_channel.py` 直接跑。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import (SENSEVOICE_DIR, OLLAMA_HOST, OLLAMA_MODEL, DEVICE, OFFLINE,  # noqa: E402
                   SEMANTIC_LLM_BACKEND)

OLLAMA_URL = f"{OLLAMA_HOST.rstrip('/')}/api/chat"
DEFAULT_MODEL = OLLAMA_MODEL
TIMEOUT_SECONDS = 120  # 给足时间（本地 r1 思考型慢；云端一般数秒内返回）

# ---- 云端 DeepSeek（默认后端；OpenAI 兼容 API，密钥来自 secrets env）----
CLOUD_BASE = (os.environ.get("SCAM_LLM_BASE") or "https://api.deepseek.com/v1").rstrip("/")
CLOUD_KEY = os.environ.get("SCAM_LLM_KEY") or ""
CLOUD_MODEL = os.environ.get("SCAM_LLM_MODEL") or "deepseek-chat"


def resolve_backend(explicit: Optional[str] = None) -> str:
    """解析 LLM 后端。铁律：默认 cloud，绝不静默回退 ollama。

    cloud  → DeepSeek 云端（需要 SCAM_LLM_KEY；无密钥由调用方/analyze 走规则兜底）
    ollama → 仅当显式 env VERICALL_SEMANTIC_LLM=ollama 或显式传参才启用
    """
    raw = (explicit or os.environ.get("VERICALL_SEMANTIC_LLM") or SEMANTIC_LLM_BACKEND or "cloud")
    b = str(raw).strip().lower()
    return "ollama" if b == "ollama" else "cloud"

# ASR 推理设备：cuda:0 / cpu，由 VERICALL_DEVICE 控制（见 src/paths.py）
ASR_DEVICE = "cuda:0" if DEVICE == "cuda" else "cpu"

# 四类高发电话诈骗模式（对齐公安部反诈宣传口径）
CATEGORIES = {
    "impersonation":  "冒充身份（自称子女/公检法/客服/领导熟人）",
    "urgency_threat": "紧急恐吓（制造恐慌：涉案/冻结/家人出事）",
    "money_request":  "转账汇款（索要钱财、要求垫付、安全账户）",
    "credential":     "骗取凭据（验证码/密码/银行卡号/人脸识别）",
    "normal":         "正常对话",
    "unknown":        "无法判断",
}

SYSTEM_PROMPT = """你是电话诈骗话术分析引擎。分析用户给出的通话转写文本，判断是否属于诈骗话术。

诈骗类别定义：
- impersonation: 冒充身份（自称子女/公检法/客服/领导熟人，常配合"新号码"话术）
- urgency_threat: 紧急恐吓（涉案、账户冻结、家人出事等制造恐慌）
- money_request: 转账汇款（要钱、垫付、"安全账户"、二维码收款）
- credential: 骗取凭据（索要验证码/密码/卡号/诱导人脸识别）
- normal: 正常对话（家人闲聊、快递通知、正常业务办理）

只输出一个 JSON 对象，不要输出其他任何内容：
{"risk": <0到1的小数，诈骗风险分>, "category": <类别英文标识>, "reason": "<不超过40字的中文理由>"}

注意：
- 文本可能不完整（实时流式转写），信息不足时 risk 给 0.2-0.4 并选 normal 或 unknown
- 多类并存时选最严重的一类，risk 取高
- 正常家人对话即使提到钱（如"我给你转了生活费"）也应判 normal"""


@dataclass
class SemanticResult:
    risk: float            # 0~1 诈骗风险分
    category: str          # CATEGORIES 键
    reason: str            # 中文理由（面向适老提示/日志）
    latency_s: float       # 端到端时延（秒）
    model: str             # 使用的风险模型名
    transcript: str = ""   # ASR 转写文本（纯文本分析时为空）
    backend: str = ""      # llm 后端：cloud | ollama | rule | rule-offline

    def to_dict(self) -> dict:
        return asdict(self)


class SenseVoiceASR:
    """本地 SenseVoice 语音转写（funasr 驱动，懒加载）。"""

    def __init__(self, model_dir: str = str(SENSEVOICE_DIR), device: str = ASR_DEVICE):
        self.model_dir = model_dir
        self.device = device
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from funasr import AutoModel
        except ImportError as e:
            raise RuntimeError(
                "funasr 未安装，请先: pip install funasr modelscope "
                f"(-i https://pypi.tuna.tsinghua.edu.cn/simple) -> {e}")
        if not os.path.isdir(self.model_dir):
            raise RuntimeError(f"SenseVoice 目录不存在: {self.model_dir}")
        print(f"[ASR] 加载 SenseVoice @ {self.model_dir} (device={self.device}) ...")
        t0 = time.time()
        self._model = AutoModel(
            model=self.model_dir,
            trust_remote_code=True,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device=self.device,
            disable_update=True,
        )
        print(f"[ASR] 加载完成 ({time.time()-t0:.1f}s)")
        return self._model

    @staticmethod
    def _clean(text: str) -> str:
        """剥离 SenseVoice 输出的 <|zh|> <|NEUTRAL|> <|Speech|> <|endoftext|> 等标签。"""
        text = re.sub(r"<\|[^|]+\|>", "", text)
        return text.strip()

    def transcribe(self, audio_path: str, language: str = "auto") -> str:
        """音频 -> 文本。language: auto/zh/en/yue/ja/ko。"""
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(audio_path)
        model = self._load()
        res = model.generate(
            input=audio_path,
            language=language,
            use_itn=True,           # 数字转写为中文（"五万"而非"50000"）
            batch_size_s=60,
        )
        raw = res[0]["text"] if res and "text" in res[0] else ""
        return self._clean(raw)


class SemanticChannel:
    """通道③：话术语义风险分析。可纯文本，也可音频->ASR->风险。"""

    def __init__(self, model: Optional[str] = None, url: Optional[str] = None,
                 timeout: int = TIMEOUT_SECONDS, asr: Optional[SenseVoiceASR] = None,
                 backend: Optional[str] = None):
        # 铁律（2026-09-06）：默认 cloud（DeepSeek 云端），ollama 仅显式开启
        self.backend = resolve_backend(backend)
        if self.backend == "cloud":
            self.model = model or CLOUD_MODEL
            self.url = f"{CLOUD_BASE}/chat/completions"
        else:
            self.model = model or DEFAULT_MODEL
            self.url = url or OLLAMA_URL
        self.timeout = timeout
        self.asr = asr or SenseVoiceASR()
        self._llm_cache: Optional[bool] = None   # LLM 可达性缓存（cloud/ollama 共用）

    # ------------------------------------------------------------------ #
    def analyze(self, transcript: str) -> SemanticResult:
        """分析一段转写文本。transcript 需非空。"""
        if not transcript or not transcript.strip():
            return self._fallback("空文本")
        if OFFLINE or not self.llm_available():
            # 离线 / LLM 不可达（cloud 缺密钥 或 ollama 未起）：规则评分器兜底（零依赖）
            return self._rule_score(transcript)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript.strip()[:2000]},
        ]
        t0 = time.time()
        content = self._call_llm(messages)   # 返回模型原始文本；失败为 None
        if content is None:
            return self._fallback("LLM 调用失败(含重试)", t0)
        content = self._strip_think(content)
        parsed = self._parse_json(content)
        if parsed is None:
            return self._fallback(f"LLM 输出无法解析: {content[:80]!r}", t0)
        risk = max(0.0, min(1.0, float(parsed.get("risk", 0.5))))
        category = parsed.get("category", "unknown")
        if category not in CATEGORIES:
            category = "unknown"
        return SemanticResult(
            risk=round(risk, 2),
            category=category,
            reason=str(parsed.get("reason", ""))[:60],
            latency_s=round(time.time() - t0, 2),
            model=self.model,
            transcript=transcript.strip(),
            backend=self.backend,
        )

    def analyze_audio(self, audio_path: str, language: str = "auto") -> SemanticResult:
        """音频 -> SenseVoice 转写 -> 风险评分。端到端时延含 ASR。"""
        if OFFLINE:
            # 离线降级：无 SenseVoice，无法转写；返回中性占位（演示走缓存转写文本）
            return SemanticResult(
                risk=0.5, category="unknown",
                reason="离线模式无法转写(无 SenseVoice)", latency_s=0.0,
                model="rule-offline", transcript="", backend="rule-offline")
        t0 = time.time()
        try:
            transcript = self.asr.transcribe(audio_path, language=language)
        except Exception as e:  # noqa: BLE001 - 转写失败要兜底而非崩溃
            return SemanticResult(
                risk=0.5, category="unknown",
                reason=f"ASR 失败: {e}", latency_s=round(time.time() - t0, 2),
                model=self.model, transcript="")
        if not transcript:
            return SemanticResult(
                risk=0.5, category="unknown",
                reason="ASR 转写为空", latency_s=round(time.time() - t0, 2),
                model=self.model, transcript="")
        r = self.analyze(transcript)
        r.transcript = transcript
        r.latency_s = round(time.time() - t0, 2)
        return r

    # ------------------------------------------------------------------ #
    def _call_llm(self, messages: list, temperature: float = 0.2) -> Optional[str]:
        """调 LLM 并返回模型原始文本（cloud=DeepSeek chat/completions，
        ollama=/api/chat）。网络抖动 / HTTP 429/500/503 有限退避重试；失败返回 None。

        cloud 模式分类任务 temperature 直接传顶层；ollama 走 options 包装。
        """
        if self.backend == "cloud":
            body = {"model": self.model, "stream": False,
                    "messages": messages, "temperature": temperature}
            headers = {"Content-Type": "application/json",
                       "Authorization": f"Bearer {CLOUD_KEY}"}

            def pick(d):
                return d["choices"][0]["message"]["content"]
        else:
            body = {"model": self.model, "stream": False,
                    "messages": messages, "options": {"temperature": temperature}}
            headers = {"Content-Type": "application/json"}

            def pick(d):
                return d.get("message", {}).get("content", "")

        last_err = None
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    self.url, data=json.dumps(body).encode("utf-8"), headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return pick(json.loads(resp.read().decode("utf-8")))
            except urllib.error.HTTPError as e:
                last_err = f"HTTP {e.code}"
                if e.code in (429, 500, 503):
                    time.sleep(2 + attempt * 3)   # 退避后重试
                    continue
                return None
            except (urllib.error.URLError, TimeoutError,
                    json.JSONDecodeError, KeyError) as e:
                last_err = str(e)
                time.sleep(1 + attempt * 2)
                continue
        print(f"[话术] {self.backend} 重试后仍失败: {last_err}")
        return None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _strip_think(text: str) -> str:
        """剥离 deepseek-r1 的 <think>...</think> 思考段。"""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()

    @staticmethod
    def _parse_json(text: str) -> Optional[dict]:
        """容错解析：模型偶尔会包 ```json 代码块或夹带前后缀。"""
        m = re.search(r"\{[^{}]*\}", text, flags=re.S)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None

    def _rule_score(self, text: str) -> "SemanticResult":
        """离线/降级时的话术评分：委托给 rule_scorer 模块（零依赖、可解释）。

        rule_scorer 同时也是论文里 "规则基线 vs LLM" 的消融实验实现。
        """
        from fusion.rule_scorer import rule_score
        res = rule_score(text)
        return SemanticResult(
            risk=res.risk, category=res.category, reason=res.reason,
            latency_s=0.0, model="rule-offline", backend="rule",
            transcript=(text or "").strip())

    def llm_available(self) -> bool:
        """LLM 可达性探测（结果缓存，避免每次分析都发请求）。

        cloud：有 SCAM_LLM_KEY 即视为可用（真实连通性由调用重试兜底）；
        ollama：探测 /api/tags。
        """
        if OFFLINE:
            return False
        if self._llm_cache is not None:
            return self._llm_cache
        ok = False
        if self.backend == "cloud":
            ok = bool(CLOUD_KEY)
        else:
            try:
                with urllib.request.urlopen(
                        f"{OLLAMA_HOST.rstrip('/')}/api/tags", timeout=1.5) as resp:
                    ok = resp.status == 200
            except Exception:  # noqa: BLE001
                ok = False
        self._llm_cache = ok
        return ok

    def _fallback(self, why: str, t0: Optional[float] = None,
                  backend: Optional[str] = None) -> SemanticResult:
        lat = round(time.time() - t0, 2) if t0 else 0.0
        return SemanticResult(risk=0.5, category="unknown",
                              reason=f"低置信: {why}", latency_s=lat,
                              model=self.model, backend=backend or self.backend)


# ---------------------------------------------------------------------- #
def _demo():
    samples = [
        ("诈骗-冒充+要钱", "喂妈，是我啊，我手机摔坏了这是同学的号。我在学校出点事急用钱，"
                          "你先转五万到这个卡号，别告诉我爸，快点啊要不来不及了。"),
        ("诈骗-冒充公检法", "这里是市公安局，你名下的银行卡涉嫌洗钱案件，需要把资金转入"
                           "安全账户配合清查，否则今天下午就会冻结你全部账户并逮捕你。"),
        ("正常-家人", "妈，我周末回来吃饭，你别买菜了我在路上买。对了爸的降压药吃完没？"),
        ("正常-业务", "您好，您的快递已放到小区丰巢柜，取件码是 3372，请及时领取。"),
    ]
    ch = SemanticChannel()
    print(f"风险模型: {ch.model} @ {ch.url} (backend={ch.backend})\n")
    for name, text in samples:
        r = ch.analyze(text)
        flag = "⚠️" if r.risk >= 0.7 else "✓"
        print(f"[{flag}] {name}: risk={r.risk} cat={r.category} "
              f"({r.latency_s}s) {r.reason}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--audio" and len(sys.argv) > 2:
        ch = SemanticChannel()
        r = ch.analyze_audio(sys.argv[2])
        print(json.dumps(r.to_dict(), ensure_ascii=False, indent=2))
    elif len(sys.argv) > 1:
        ch = SemanticChannel()
        r = ch.analyze(" ".join(sys.argv[1:]))
        print(json.dumps(r.to_dict(), ensure_ascii=False, indent=2))
    else:
        _demo()
