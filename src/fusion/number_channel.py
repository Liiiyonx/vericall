# -*- coding: utf-8 -*-
"""通道⓪ 号码先验层（number_channel）——轻量先验通道（2026-09-07）

定位：三通道（声学/声纹/语义）之前的**零成本先验闸门**。
- 来电号码精确命中公开涉诈号码库（12321 举报 / 运营商标记 / GitHub 开源黑名单）
  → 直接 block，跳过三通道推理，省算力（官方"已知号码"联防）；
- 虚商号段 / 境外来显等启发式命中 → 只升级为 caution，继续跑三通道，避免号段偏见误拦；
- 未命中（新号/改号/AI 拟声）→ 才进音频三通道（我们"未知号码"内容级检测）。
两者互补：官方黑名单联防 + 我们内容级实时检测。

设计原则：**不重造轮子**——本模块只做"接入 + 融合"：
  1. 号码库：外部公开源文件（每行一个，可带注释/来源），本模块只负责加载与查表；
  2. 启发式：内置可解释规则（虚拟号段/境外改号等公开通报常见涉诈特征），开关可配；
  3. 不碰腾讯/运营商付费私有库，不联网实时查号（隐私/合规：本地库，不把家人号码外发）。

数据文件：data/numbers/blocklist.txt（用户自维护公开源黑名单，逐行号码，`#` 注释，
支持 "号码<TAB>备注<TAB>来源" 便于审计）。
环境变量：
  VERICALL_NUMBER_BLOCKLIST = 黑名单文件路径（默认 data/numbers/blocklist.txt）
  VERICALL_NUMBER_HEURISTIC = 1|0 是否启用启发式（默认 1）
号码归一化：去空格/横杠/括号，剥离 +86/0086/86 前缀，保留 E.164 风格 11 位。
启发式规则（可解释，特征来自公开反诈通报，规则命中 ≠ 实锤，仅先验加速）：
  - virtual_prefix  : 虚拟运营商号段 170/171/162/165/167（虚商实名弱，诈骗高发段）
  - overseas_cc     : 境外/改号特征 00/+/008x/009x 开头（国际来显，冒充客服高发）
  - no_caller_id    : 无号码/未知（*）——只提示不拦（隐私号合法使用普遍）
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from private_fs import append_private_text

ROOT = Path(__file__).resolve().parents[2]

# 默认黑名单路径（用户可经环境变量覆盖；文件由公开源维护，本模块不内置实号）
DEFAULT_BLOCKLIST = ROOT / "data" / "numbers" / "blocklist.txt"
# 家庭自建举报单独落盘，避免某一家庭的私享举报污染其他家庭。
DEFAULT_FAMILY_BLOCKLIST = ROOT / "data" / "numbers" / "family_blocklist.jsonl"

# 虚拟运营商号段（公开通报高发段）
VIRTUAL_PREFIXES = ("170", "171", "162", "165", "167")
# 国内手机号 1[3-9]xxxxxxxxx（11 位）
MAINLAND_MOBILE = re.compile(r"^1[3-9]\d{9}$")
# 国内服务短号（400/800/95xxx，如 95588）
MAINLAND_SERVICE = re.compile(r"^(?:400|800|95)\d{0,10}$")
# 国内固话 0 开头（010/021/0755...）
MAINLAND_LANDLINE = re.compile(r"^0\d{9,11}$")
# 境外/国际改号：00 开头 或 中国港澳台区号 852/853/886
OVERSEAS_LEAD = ("852", "853", "886")


def normalize(number: str) -> str:
    """去分隔符 → 去国际前缀 → 返回可比较串（非数字开头返回空串）。"""
    if number is None:
        return ""
    s = re.sub(r"[\s\-()（）.]", "", str(number).strip())
    if s in ("", "*", "未知", "unknown", "anonymous"):
        return ""
    if "*" in s or not re.fullmatch(r"\+?\d{3,15}", s):
        return ""  # 匿名/异常字符视为无号码
    if s.startswith("+"):
        s = s[1:]
    if s.startswith("0086"):
        s = s[4:]
    elif s.startswith("86") and len(s) == 13:  # 8613xxxxxxxxx
        s = s[2:]
    return s


def classify_heuristic(num: str) -> str | None:
    """返回命中规则名或 None（num 为已归一化串；空/未知返回 None）。
    先排除国内形态（手机/固话/400·800·95 服务号），再判虚商/境外。
    规则均为"公开通报特征"，不是实锤：命中仅作先验加速/谨慎提示。"""
    if not num:
        return None
    if len(num) == 11 and num.startswith(VIRTUAL_PREFIXES):
        return "virtual_prefix"
    if MAINLAND_MOBILE.match(num) or MAINLAND_SERVICE.match(num) \
            or MAINLAND_LANDLINE.match(num):
        return None
    if len(num) >= 7:
        if num.startswith("00") or num[:3] in OVERSEAS_LEAD:
            return "overseas_cc"
        # 非国内首位（含美/英等 +1/+44 国际号；国内形态已在上方排除）
        if num[0] in "23456789" or (num[0] == "1" and len(num) >= 10):
            return "overseas_cc"
    return None


@dataclass
class NumberVerdict:
    """号码先验判定（并入 FusionOrchestrator 的 ChannelVerdict 同构字段）。"""
    number: str           # 归一化号码（空=无来电显示）
    matched: bool         # 是否命中（黑名单精确命中 或 启发式命中）
    source: str           # blocklist / heuristic:<rule> / not_hit / no_caller_id
    detail: str           # 给人看的说明（含黑名单备注）
    action: str = "none"  # block=精确黑名单短路 / caution=启发式继续三通道 / none=无先验
    score: float = 0.0    # 并入融合器：命中=1.0
    confidence: float = 1.0
    latency_ms: float = 0.0

    def to_channel_dict(self) -> dict:
        return {
            "name": "number",
            "score": self.score,
            "label": self.source,
            "detail": self.detail,
            "confidence": self.confidence,
            "matched": self.matched,
            "source": self.source,
            "action": self.action,
            "number": self.number,
            "latency_ms": round(self.latency_ms, 2),
        }


class NumberChannel:
    """通道⓪ 号码先验。线程安全查表，并按文件 mtime 自动热重载。"""

    def __init__(self, blocklist_path: str | os.PathLike | None = None,
                 heuristic: bool | None = None):
        env_bl = os.environ.get("VERICALL_NUMBER_BLOCKLIST")
        self.blocklist_path = Path(blocklist_path or env_bl or DEFAULT_BLOCKLIST)
        env_family_bl = os.environ.get("VERICALL_FAMILY_NUMBER_BLOCKLIST")
        self.family_blocklist_path = Path(
            env_family_bl or DEFAULT_FAMILY_BLOCKLIST)
        env_heur = os.environ.get("VERICALL_NUMBER_HEURISTIC", "1")
        self.heuristic = bool(int(env_heur)) if heuristic is None else heuristic
        self._by_number: dict[str, str] = {}   # number -> remark
        self._family_by_number: dict[str, dict[str, str]] = {}
        self._lock = threading.Lock()
        self._mtime_key: tuple[int, int] | None = None
        self._family_mtime_key: tuple[int, int] | None = None
        self.load()

    # ------------------------------------------------------------------ #
    def _file_key(self) -> tuple[int, int] | None:
        try:
            st = self.blocklist_path.stat()
            return st.st_mtime_ns, st.st_size
        except OSError:
            return None

    def _read_blocklist(self) -> tuple[dict[str, str], tuple[int, int] | None]:
        loaded: dict[str, str] = {}
        key = self._file_key()
        if key is None:
            return loaded, None
        try:
            lines = self.blocklist_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return loaded, None
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            num = normalize(parts[0])
            if num:
                remark = parts[1] if len(parts) > 1 else ""
                # 同号多次出现保留首个备注（后到者补充到 detail 由调用方拼）
                loaded.setdefault(num, remark)
        return loaded, key

    def _family_file_key(self) -> tuple[int, int] | None:
        try:
            st = self.family_blocklist_path.stat()
            return st.st_mtime_ns, st.st_size
        except OSError:
            return None

    def _read_family_blocklist(
            self) -> tuple[dict[str, dict[str, str]],
                           tuple[int, int] | None]:
        """读取家庭私有举报 JSONL；损坏行跳过，不影响公开库和其他家庭。"""
        loaded: dict[str, dict[str, str]] = {}
        key = self._family_file_key()
        if key is None:
            return loaded, None
        try:
            lines = self.family_blocklist_path.read_text(
                encoding="utf-8").splitlines()
        except OSError:
            return loaded, None
        for raw in lines:
            try:
                item = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(item, dict):
                continue
            family_id = str(item.get("family_id") or "").strip()
            if not re.fullmatch(r"[\w.-]{1,64}", family_id):
                continue
            num = normalize(str(item.get("number") or ""))
            if not num:
                continue
            remark = str(item.get("remark") or "")[:80]
            loaded.setdefault(family_id, {}).setdefault(num, remark)
        return loaded, key

    def load(self) -> None:
        """加载公开库与家庭私有举报库。"""
        loaded, key = self._read_blocklist()
        family_loaded, family_key = self._read_family_blocklist()
        with self._lock:
            self._by_number = loaded
            self._mtime_key = key
            self._family_by_number = family_loaded
            self._family_mtime_key = family_key
        # 失败静默原则：库为空时 detail 提示

    def _maybe_reload(self) -> None:
        key = self._file_key()
        family_key = self._family_file_key()
        if key == self._mtime_key and family_key == self._family_mtime_key:
            return
        with self._lock:
            if key != self._mtime_key:
                loaded, current_key = self._read_blocklist()
                self._by_number = loaded
                self._mtime_key = current_key
            if family_key != self._family_mtime_key:
                family_loaded, current_family_key = self._read_family_blocklist()
                self._family_by_number = family_loaded
                self._family_mtime_key = current_family_key

    def check(self, caller_number: str | None,
              family_id: str | None = None) -> NumberVerdict:
        t0 = time.time()
        self._maybe_reload()
        num = normalize(caller_number)
        if not num:
            return NumberVerdict(number="", matched=False, source="no_caller_id",
                                 action="none",
                                 detail="无来电显示/未知号码，跳过号码先验，走三通道",
                                 latency_ms=(time.time() - t0) * 1000)
        with self._lock:
            by_number = self._by_number
            family_by_number = self._family_by_number
        family_key = str(family_id or "").strip()
        if family_key:
            family_remark = family_by_number.get(family_key, {}).get(num)
            if num in family_by_number.get(family_key, {}):
                return NumberVerdict(
                    number=num, matched=True, source="family_blocklist",
                    action="block",
                    detail=f"命中本家庭举报库（备注：{family_remark}）",
                    score=1.0, confidence=0.99,
                    latency_ms=(time.time() - t0) * 1000)
        remark = by_number.get(num)
        if remark is not None or num in by_number:
            src = "blocklist"
            detail = f"命中公开涉诈号码库（{self.blocklist_path.name}"
            detail += f"，备注：{remark}）" if remark else "（公开库无备注）"
            return NumberVerdict(number=num, matched=True, source=src,
                                 action="block",
                                 detail=detail, score=1.0, confidence=0.99,
                                 latency_ms=(time.time() - t0) * 1000)
        if self.heuristic:
            rule = classify_heuristic(num)
            if rule:
                tip = {"virtual_prefix": "虚拟运营商号段（诈骗高发段，建议谨慎接听）",
                       "overseas_cc": "境外/改号来显（冒充客服高发，注意核实）"}[rule]
                return NumberVerdict(number=num, matched=True,
                                     source=f"heuristic:{rule}",
                                     action="caution",
                                     detail=f"{tip}；仅先验提示，继续三通道内容检测",
                                     score=0.6, confidence=0.85,
                                     latency_ms=(time.time() - t0) * 1000)
        return NumberVerdict(number=num, matched=False, source="not_hit",
                             action="none",
                             detail="号码未命中先验库，进入三通道内容检测",
                             latency_ms=(time.time() - t0) * 1000)

    def report(self, caller_number: str, remark: str = "",
               source: str = "community",
               family_id: str | None = None) -> bool:
        """上报来电号码并立即生效。

        ``family_id`` 为空时写入公开库，兼容管理员手工维护；传入家庭 ID 时
        写入家庭私有 JSONL，只有同一家庭的后续检测会命中，防止跨家庭污染。
        """
        num = normalize(caller_number)
        if not num:
            return False
        family_key = str(family_id or "").strip()
        if family_key and not re.fullmatch(r"[\w.-]{1,64}", family_key):
            return False
        remark = str(remark or "")[:80]
        source = str(source or "community")[:32]
        if family_key:
            item = {
                "family_id": family_key,
                "number": num,
                "remark": remark,
                "source": source,
                "reported_at": time.time(),
            }
            p = self.family_blocklist_path
            with self._lock:
                try:
                    append_private_text(
                        p,
                        json.dumps(
                            item, ensure_ascii=False,
                            separators=(",", ":")) + "\n",
                    )
                except OSError:
                    return False
                loaded, key = self._read_family_blocklist()
                self._family_mtime_key = key
                if key is None:
                    self._family_by_number.setdefault(
                        family_key, {}).setdefault(num, remark)
                else:
                    self._family_by_number = loaded
            return True

        p = self.blocklist_path
        with self._lock:
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                exists = p.is_file() and p.stat().st_size > 0
                with open(p, "a", encoding="utf-8") as f:
                    if exists:
                        f.write("\n")
                    f.write(f"{num}\t{remark}\t{source}\n")
            except OSError:
                return False
            # 重新读取可同时吸收其他实例刚刚追加的条目；随后同步 mtime，
            # 避免本实例下一次 check() 因自身写入而再次 load。
            loaded, key = self._read_blocklist()
            self._mtime_key = key
            if key is None:
                self._by_number.setdefault(num, remark)
            else:
                self._by_number = loaded
        return True


# ---------------------------------------------------------------------- #
def demo() -> None:
    nc = NumberChannel()
    for n in ["17012345678", "0085212345678", "13800138000",
              "+86 138 0013 8000", None, "*", "010-12345678"]:
        v = nc.check(n)
        print(f"{str(n):<18} -> {v.source:<16} score={v.score} conf={v.confidence} | {v.detail}")


if __name__ == "__main__":
    demo()
