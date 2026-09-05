# -*- coding: utf-8 -*-
"""build_corpus.py — 万级中文诈骗话术库构建管线（极致化计划书 §三，v0.1 目标 2k 条）

与 scripts/generate_scam_scripts.py（红队 TTS 用 80 条小样本）的关系：
  本管线面向「语义通道训练/评测语料」，规模目标 10,000+，三轴标签
  （诈骗类型 × 话术阶段 × 风险要素），支持云端旗舰 LLM + 本地 Ollama 双后端。

流水线：
  seeds 网格（8 类 × 4 阶段）→ LLM 扩写 → 清洗 → 去重（精确 + 近重复）
  → 风险要素自动标注 → 通道③自洽过滤（可选）→ jsonl + 抽检 CSV + 统计

LLM 后端（优先级从高到低）：
  云端 OpenAI 兼容 API（不计算力主力）：
      set SCAM_LLM_BASE=https://api.deepseek.com/v1
      set SCAM_LLM_KEY=sk-...
      set SCAM_LLM_MODEL=deepseek-chat
  本地 Ollama 回退：默认 http://localhost:11434，模型 deepseek-r1:8b

用法：
  python build_corpus.py --per-cell 5                 # 8类×4阶段×5 = 160 条/轮
  python build_corpus.py --per-cell 60 --append       # 追加扩库（自动跨文件去重）
  python build_corpus.py --dry-run --backend ollama
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import random
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from seeds import (CATEGORIES, CATEGORY_RISK_MAP, DIALECT_STYLES,  # noqa: E402
                   RISK_ELEMENTS, SEEDS, STAGES, STYLE_AXES)

OUT_DIR = ROOT / "data" / "scam_corpus"
CORPUS_PATH = OUT_DIR / "corpus_v0.1.jsonl"
REVIEW_PATH = OUT_DIR / "review_sample.csv"
STATS_PATH = OUT_DIR / "stats.json"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "deepseek-r1:8b")

GEN_PROMPT = """你是电话诈骗案例库构建助手（用于反诈系统的检测训练与对抗评测）。把下面的"话术骨架"扩写成一段真实自然的通话口语文本。

要求：
1. 保留骨架里的关键要素（身份、事由、金额/凭证、紧迫感），可自然改述
2. 口语化，像真人打电话，可有停顿、语气词（喂、啊、那个、你听我说）
3. {length_hint}
4. 只输出扩写后的正文，不要任何解释、引号或序号

话术阶段：{stage_desc}
话术骨架：{seed}"""

STAGE_LENGTH = {
    "opening": "15-40字，一两句话",
    "buildup": "30-80字，一段话",
    "pressure": "20-60字，情绪激烈",
    "ask": "30-80字，把索要动作说得自然",
}

# 风险要素关键词（自动标注用；命中即标，未命中回落到类型默认）
RISK_KEYWORDS = {
    "money_transfer": ["转", "汇", "打钱", "垫", "充值", "定金", "保证金", "押金", "税费", "解冻费"],
    "account_info": ["卡号", "密码", "身份证", "账户信息"],
    "verify_code": ["验证码", "校验码"],
    "screen_share": ["屏幕共享", "远程", "会议软件"],
    "face_auth": ["人脸", "眨眼", "摇头", "活体", "念数字"],
    "secrecy": ["别告诉", "保密", "别声张", "不许说", "瞒着"],
    "time_pressure": ["马上", "立刻", "半小时", "两小时", "截止", "逾期", "到期", "今晚"],
    "authority_threat": ["逮捕", "冻结", "公安", "检察", "法院", "办案", "强制"],
    "safe_account": ["安全账户", "资金清查", "核验账户"],
    "no_police": ["别报警", "不能报警", "不许报警"],
}


# ---------------- LLM 后端 ----------------

def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def call_ollama(prompt: str, timeout: int = 120) -> str:
    payload = json.dumps({
        "model": OLLAMA_MODEL, "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0.9},
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return _strip_think(data.get("message", {}).get("content", ""))


def call_cloud(prompt: str, timeout: int = 120) -> str:
    base = os.environ["SCAM_LLM_BASE"].rstrip("/")
    key = os.environ["SCAM_LLM_KEY"]
    model = os.environ.get("SCAM_LLM_MODEL", "deepseek-chat")
    payload = json.dumps({
        "model": model, "stream": False, "temperature": 0.9,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/chat/completions", data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


def pick_backend(name: str):
    if name == "cloud" or (name == "auto" and os.environ.get("SCAM_LLM_KEY")):
        return "cloud", call_cloud
    return "ollama", call_ollama


# ---------------- 去重 ----------------

def _norm(text: str) -> str:
    return re.sub(r"[\s，。！？、…—·,.!?~\-\"'“”]", "", text)


def _ngrams(text: str, n: int = 3) -> set:
    t = _norm(text)
    return {t[i:i + n] for i in range(max(1, len(t) - n + 1))}


class Deduper:
    """精确去重 + 3-gram Jaccard ≥0.55 判近重复（零依赖，2k-10k 规模够用）。"""

    def __init__(self, threshold: float = 0.55):
        self.threshold = threshold
        self.exact: set = set()
        self.grams: list = []

    def is_dup(self, text: str) -> bool:
        n = _norm(text)
        if n in self.exact:
            return True
        g = _ngrams(text)
        for g0 in self.grams:
            inter = len(g & g0)
            if inter and inter / len(g | g0) >= self.threshold:
                return True
        return False

    def add(self, text: str):
        self.exact.add(_norm(text))
        self.grams.append(_ngrams(text))


# ---------------- 标注 ----------------

def annotate_risk(text: str, category: str) -> list:
    hits = {k for k, kws in RISK_KEYWORDS.items() if any(w in text for w in kws)}
    hits |= set(CATEGORY_RISK_MAP.get(category, []))  # 类型默认要素兜底
    order = list(RISK_ELEMENTS)
    return sorted(hits, key=order.index)


def make_seed(category: str, stage: str, rng: random.Random) -> str:
    seed = rng.choice(SEEDS[(category, stage)])
    for m in set(re.findall(r"\{(\w+)\}", seed)):
        if m in STYLE_AXES:
            seed = seed.replace("{" + m + "}", rng.choice(STYLE_AXES[m]))
    return seed


def clean_text(text: str) -> str:
    text = text.strip().strip('"').strip()
    # 剥离模型带出的引导语（可能多层嵌套，循环剥离至稳定）：
    # 冒号式前缀、【】包裹前缀、"好的，请看："式应答残留
    pats = [
        r"^(扩写后文本|正文|话术|扩写结果|通话文本)\s*[:：]\s*",
        r"^【[^】]{1,12}】\s*",
        r"^(好的|好|请看|如下|以下是)[，,：:。\s]*",
    ]
    for _ in range(4):
        before = text
        for p in pats:
            text = re.sub(p, "", text)
        if text == before:
            break
    return text.strip().strip('"').strip()


def load_existing(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    # 2026-09-05 加Ff1a全局 socket 超时Ff0c防止 urllib 半开连接无限挂起
    import socket as _socket
    _socket.setdefaulttimeout(180)
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=5, help="每(类型×阶段)格生成条数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backend", choices=["auto", "cloud", "ollama"], default="auto")
    ap.add_argument("--append", action="store_true", help="追加到现有语料并跨文件去重")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-filter", action="store_true", help="跳过通道③自洽过滤")
    ap.add_argument("--review-ratio", type=float, default=0.2, help="人工抽检比例")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    backend_name, call_llm = pick_backend(args.backend)
    cells = [(c, s) for c in CATEGORIES for s in STAGES]
    total = args.per_cell * len(cells)
    print(f"后端={backend_name}，网格={len(cells)} 格 × {args.per_cell} = 计划 {total} 条")

    existing = load_existing(CORPUS_PATH) if args.append else []
    dedup = Deduper()
    for rec in existing:
        dedup.add(rec["text"])

    ch = None
    if not args.no_filter:
        from fusion.semantic_channel import SemanticChannel
        ch = SemanticChannel()

    out, dropped_dup, dropped_len, dropped_risk, failed = [], 0, 0, 0, 0
    today = dt.date.today().isoformat()

    # 增量落盘：每条接受即写入（长批次中断不丢已产数据），非 append 模式先清空
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.append and not args.dry_run:
        CORPUS_PATH.write_text("", encoding="utf-8")
    inc_f = None if args.dry_run else open(CORPUS_PATH, "a", encoding="utf-8")

    try:
        for i in range(args.per_cell):
            for cat, stage in cells:
                seed = make_seed(cat, stage, rng)
                prompt = GEN_PROMPT.format(
                    seed=seed, stage_desc=f"{stage}（{STAGES[stage]}）",
                    length_hint=STAGE_LENGTH[stage])
                try:
                    text = clean_text(call_llm(prompt))
                except Exception as e:
                    failed += 1
                    print(f"  [fail] {type(e).__name__}: {e}")
                    continue
                if not (10 <= len(text) <= 300):
                    dropped_len += 1
                    continue
                if dedup.is_dup(text):
                    dropped_dup += 1
                    continue

                rec = {
                    "id": f"SC-{len(existing) + len(out) + 1:06d}",
                    "category": cat,
                    "stage": stage,
                    "risk_elements": annotate_risk(text, cat),
                    "label": "scam",
                    "text": text,
                    "dialect_style": rng.choice(DIALECT_STYLES),
                    "source": f"{backend_name}:{os.environ.get('SCAM_LLM_MODEL', OLLAMA_MODEL)}",
                    "seed_skeleton": seed,
                    "created": today,
                    "review": {"sampled": False, "verdict": None},
                }
                # 自洽过滤：ask/pressure 阶段要求高置信；opening 允许低分（本就隐晦）
                if ch is not None and stage in ("pressure", "ask"):
                    r = ch.analyze(text)
                    if r.risk < 0.5:
                        dropped_risk += 1
                        continue
                    rec["self_check_risk"] = r.risk

                dedup.add(text)
                out.append(rec)
                if inc_f is not None:
                    inc_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    inc_f.flush()
                print(f"  [ok {rec['id']}] {cat}/{stage} {text[:36]}...")
    finally:
        if inc_f is not None:
            inc_f.close()

    print(f"\n生成 {len(out)} | 近重复丢 {dropped_dup} | 长度丢 {dropped_len} "
          f"| 自检丢 {dropped_risk} | 调用失败 {failed}")

    if args.dry_run:
        return

    all_recs = existing + out
    # 人工抽检：本轮新样本按 20% 抽样，标注后回写全量文件
    for rec in rng.sample(out, k=int(len(out) * args.review_ratio)) if out else []:
        rec["review"]["sampled"] = True
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        for rec in all_recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(REVIEW_PATH, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "category", "stage", "risk_elements", "text", "verdict(填 pass/fix/drop)"])
        for rec in all_recs:
            if rec["review"]["sampled"]:
                w.writerow([rec["id"], rec["category"], rec["stage"],
                            "|".join(rec["risk_elements"]), rec["text"], ""])

    stats = {
        "created": today,
        "backend": backend_name,
        "total": len(all_recs),
        "by_category": {c: sum(1 for r in all_recs if r["category"] == c) for c in CATEGORIES},
        "by_stage": {s: sum(1 for r in all_recs if r["stage"] == s) for s in STAGES},
        "review_sampled": sum(1 for r in all_recs if r["review"]["sampled"]),
    }
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"语料 -> {CORPUS_PATH}（累计 {len(all_recs)}）\n抽检 -> {REVIEW_PATH}\n统计 -> {STATS_PATH}")


if __name__ == "__main__":
    main()
