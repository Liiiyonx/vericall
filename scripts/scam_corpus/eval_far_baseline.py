#!/usr/bin/env python
"""通道③ FAR 基线评估（2026-09-05；09-05 21:30 增云端后端）

SemanticChannel（本地 Ollama r1:8b）或云端 DeepSeek（OpenAI 兼容）对正负样本打分：
  - 负样本（529，含近边界 legit_money/legit_service）应低风险  -> FAR = 负样本被判诈骗占比
  - 正样本（抽 N 条诈骗话术）应高风险                        -> 召回 = 正样本被判诈骗占比
两后端共用同一 SYSTEM_PROMPT（取 fusion/semantic_channel.py），保证语义可比。

用法：
  python eval_far_baseline.py --backend cloud [--scam-n 529] [--seed 42] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import socket as _socket
_socket.setdefaulttimeout(90)

DATA = ROOT / "data" / "scam_corpus"
BENIGN_PATH = DATA / "benign_corpus.jsonl"
CORPUS_PATH = DATA / "corpus_v0.1.jsonl"
OUT_MD = ROOT / "evaluation" / "far_baseline.md"
OUT_JSON = ROOT / "evaluation" / "far_baseline.json"

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


def load(path: Path):
    recs = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except Exception:
            continue
    return recs


def _strip_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return s


class CloudScorer:
    """云端 DeepSeek（OpenAI 兼容 /chat/completions），同 SYSTEM_PROMPT。"""

    def __init__(self, timeout: int = 60):
        self.base = os.environ["SCAM_LLM_BASE"].rstrip("/")
        self.key = os.environ["SCAM_LLM_KEY"]
        self.model = os.environ.get("SCAM_LLM_MODEL", "deepseek-chat")
        self.timeout = timeout

    def analyze(self, text: str):
        payload = json.dumps({
            "model": self.model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text.strip()[:2000]},
            ],
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base}/chat/completions", data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(_strip_fence(content))
        return {"risk": max(0.0, min(1.0, float(parsed.get("risk", 0.5)))),
                "category": parsed.get("category", "unknown"),
                "reason": str(parsed.get("reason", ""))[:60]}


def make_scorer(backend: str):
    if backend == "cloud":
        print(f"后端=cloud ({os.environ.get('SCAM_LLM_BASE','')} / {os.environ.get('SCAM_LLM_MODEL','deepseek-chat')})")
        return CloudScorer()
    print("后端=ollama (SemanticChannel / deepseek-r1:8b)")
    from fusion.semantic_channel import SemanticChannel
    return SemanticChannel(timeout=40, asr=None)


def is_near(r: dict) -> bool:
    return bool(r.get("near_boundary")) or "近边界" in str(r.get("category", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scam-n", type=int, default=529, help="正样本抽样数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help=">0 时仅跑前 N 条（冒烟）")
    ap.add_argument("--backend", choices=["ollama", "cloud"], default="cloud")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    scorer = make_scorer(args.backend)

    benign = load(BENIGN_PATH)
    scam_all = load(CORPUS_PATH)
    scam = rng.sample(scam_all, min(args.scam_n, len(scam_all)))
    near = [r for r in benign if is_near(r)]

    if args.limit:
        benign = benign[: args.limit]
        scam = scam[: args.limit]
        near = [r for r in benign if is_near(r)]

    print(f"样本: benign={len(benign)} (近边界 {len(near)}) scam={len(scam)} limit={args.limit}")

    def score_one(rec, tag):
        try:
            t0 = time.time()
            r = scorer.analyze(rec.get("text", ""))
            rec["_risk"] = float(r["risk"])
            rec["_latency"] = round(time.time() - t0, 2)
            print(f"  [{tag} {rec.get('id','?')}] risk={r['risk']:.2f} lat={rec['_latency']}s", flush=True)
            return True
        except Exception as e:  # noqa: BLE001
            rec["_risk"] = None
            print(f"  [fail {tag} {rec.get('id','?')}] {type(e).__name__}: {e}", flush=True)
            return False

    n_ok = 0
    for rec in benign:
        n_ok += 1 if score_one(rec, "B") else 0
    for rec in scam:
        n_ok += 1 if score_one(rec, "S") else 0
    print(f"打分完成: {n_ok}/{len(benign)+len(scam)}")

    def risks(recs):
        return [r["_risk"] for r in recs if r.get("_risk") is not None]

    b_risks, s_risks = risks(benign), risks(scam)
    nb_risks = [r["_risk"] for r in benign
                if r.get("_risk") is not None and "近边界" in str(r.get("category", ""))]
    if not b_risks:
        print("无有效打分，退出")
        return

    rows = []
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        far = sum(1 for x in b_risks if x >= thr) / max(1, len(b_risks))
        far_nb = sum(1 for x in nb_risks if x >= thr) / max(1, len(nb_risks))
        rec = sum(1 for x in s_risks if x >= thr) / max(1, len(s_risks))
        rows.append({"threshold": thr, "FAR_all": round(far, 4),
                     "FAR_near_boundary": round(far_nb, 4), "recall_scam": round(rec, 4)})
        print(f"thr={thr} FAR={far:.3f} FAR_nb={far_nb:.3f} recall={rec:.3f}", flush=True)

    row05 = next(r for r in rows if r["threshold"] == 0.5)
    avg_lat = sum(r.get("_latency", 0) for r in benign) / max(1, len(benign))

    md = [
        "# 通道③ FAR 基线报告",
        "",
        f"> 生成：{time.strftime('%Y-%m-%d %H:%M')} · 后端={args.backend} · seed={args.seed}",
        f"> 样本：负样本 {len(b_risks)} 条（近边界 {len(nb_risks)}）+ 正样本抽样 {len(s_risks)} 条 · 单条均延迟 {avg_lat:.1f}s",
        "",
        "## FAR / 召回扫描（阈值 = 风险分卡点）",
        "",
        "| 阈值 | FAR(全负样本) | FAR(近边界) | 召回(正样本) |",
        "|---|---|---|---|",
    ]
    for r in rows:
        md.append(f"| {r['threshold']} | {r['FAR_all']:.1%} | {r['FAR_near_boundary']:.1%} | {r['recall_scam']:.1%} |")
    md += [
        "",
        f"## 0.5 工作点：FAR={row05['FAR_all']:.2%}（近边界 {row05['FAR_near_boundary']:.2%}），召回={row05['recall_scam']:.2%}",
        "",
        "> 注：家庭场景「误拒代价>>误受」→ 正式部署建议取 FAR 最低且召回可接受的阈值，",
        "> 数值以上表为准；本报告同时服务《参赛优化任务书》评测证据链。",
        "",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    summary = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
               "backend": args.backend,
               "n_benign": len(b_risks), "n_near": len(nb_risks), "n_scam": len(s_risks),
               "avg_latency_s": round(avg_lat, 2), "rows": rows,
               "wp_0_5": row05}
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告 -> {OUT_MD}")


if __name__ == "__main__":
    main()
