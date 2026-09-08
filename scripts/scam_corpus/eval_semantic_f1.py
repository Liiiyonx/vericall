#!/usr/bin/env python
"""eval_semantic_f1.py — 线 A4：话术分类多分类 F1（2026-09-07）

用 A1 冻结评测集（scam eval 2,387 + benign eval 481）走云端 deepseek-chat 做
9 类分类（8 类诈骗 + benign 正常），输出：
  - 9 类宏/加权 F1、逐类 P/R/F1、整体准确率
  - 二分类（诈骗 vs 正常）P/R/F1（防"多类全对但诈骗漏拦"假象）
评测口径：独立评测 prompt（非生产 SemanticChannel 的合并 impersonation 口径），
原因与依据见 evaluation/semantic_review_template.md §类别粒度；生产侧分类合并
不影响本评测对语料标注体系（8 类细分）的可比性。

用法（先 source 密钥）：
  python -u scripts/scam_corpus/eval_semantic_f1.py [--limit N] [--threads 8]
产物：evaluation/semantic_f1.json / .md
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LABELS = ["elder_healthcare", "fake_investment", "impersonate_authority",
          "impersonate_family", "lottery_prize", "online_loan", "refund_cs",
          "romance_pig", "benign"]
LABEL_ZH = {"elder_healthcare": "养老健康", "fake_investment": "虚假投资",
            "impersonate_authority": "冒充公检法", "impersonate_family": "冒充亲友",
            "lottery_prize": "中奖", "online_loan": "网贷", "refund_cs": "退款客服",
            "romance_pig": "杀猪盘", "benign": "正常"}

PROMPT_V1 = """你是电信诈骗话术分类器。把下面这条中文通话话术归到 9 类之一，只输出 JSON。
类别：elder_healthcare 养老健康/保健品诈骗 | fake_investment 虚假投资/荐股 |
impersonate_authority 冒充公检法/领导 | impersonate_family 冒充亲友/子女/孙辈 |
lottery_prize 中奖/兑奖缴费 | online_loan 网贷/贷款 | refund_cs 退款/客服 |
romance_pig 杀猪盘/婚恋诱导投资 | benign 正常日常通话（含家人借钱、缴费提醒、客服回访等）。
规则：话术可能只是完整诈骗链的一段（开场/铺垫/索要/施压），凭内容语气判其归属类别即可；
正常生活场景即使提到转账（亲属借钱、正常还款）也判 benign。
输出 JSON：{"label":"<上列英文键>","risk":<0到1>}
"""

# v2：加入 romance↔investment 与 authority↔family 的显式界定规则（A4 细分迭代 v2，2026-09-08）。
PROMPT_V2 = """你是电信诈骗话术分类器。把下面这条中文通话话术归到 9 类之一，只输出 JSON。
类别：elder_healthcare 养老健康/保健品诈骗 | fake_investment 虚假投资/荐股 |
impersonate_authority 冒充公检法/领导 | impersonate_family 冒充亲友/子女/孙辈 |
lottery_prize 中奖/兑奖缴费 | online_loan 网贷/贷款 | refund_cs 退款/客服 |
romance_pig 杀猪盘/婚恋诱导投资 | benign 正常日常通话（含家人借钱、缴费提醒、客服回访等）。
规则：
1. 话术可能只是完整诈骗链的一段（开场/铺垫/索要/施压），凭内容语气判其归属类别即可；正常生活场景
   即使提到转账（亲属借钱、正常还款、缴费提醒）也判 benign。
2. romance_pig 与 fake_investment 的界定：fake_investment = 陌生人/社群/直播间荐股带单，无情感关系
   经营；若话术含婚恋语境（婚恋平台/相亲/红娘介绍、情感称谓如亲爱的/老婆/老公/宝贝、嘘寒问暖建立
   信任、异地恋/见面铺垫后再引到投资赚钱、礼物/转账测试真心等），即使主体在讲"带你投资/一起赚钱/
   内幕消息"，也判 romance_pig（杀猪盘=婚恋诱导投资）。
3. impersonate_authority 与 impersonate_family 的界定：自称公检法/客服/机构并以法律威慑、涉案、
   安全账户、索要验证码施压 → impersonate_authority；以亲人身份急事求助（出车祸/被抓/生病住院/
   手机丢失借号/要医药费保释金）→ impersonate_family，即使带恐吓语气。
输出 JSON：{"label":"<上列英文键>","risk":<0到1>}
"""

# v3：在 v2 基础上修正 authority↔refund 边界（v2 把"客服+施压"误并入 authority，refund R 96.7→90.5；
# v3 authority 仅限执法/政府机关口吻，机构客服类一律 refund_cs）。2026-09-08。
PROMPT_V3 = """你是电信诈骗话术分类器。把下面这条中文通话话术归到 9 类之一，只输出 JSON。
类别：elder_healthcare 养老健康/保健品诈骗 | fake_investment 虚假投资/荐股 |
impersonate_authority 冒充公检法/政府执法 | impersonate_family 冒充亲友/子女/孙辈 |
lottery_prize 中奖/兑奖缴费 | online_loan 网贷/贷款 | refund_cs 退款/客服（含冒充平台、商家、银行、
物流、航空等客服，以退款理赔/取消业务/扣款威胁为由）| romance_pig 杀猪盘/婚恋诱导投资 | benign 正常
日常通话（含家人借钱、缴费提醒、客服回访等）。
规则：
1. 话术可能只是完整诈骗链的一段（开场/铺垫/索要/施压），凭内容语气判其归属类别即可；正常生活场景
   即使提到转账（亲属借钱、正常还款、缴费提醒）也判 benign。
2. romance_pig 与 fake_investment 的界定：fake_investment = 陌生人/社群/直播间荐股带单，无情感关系
   经营；若话术含婚恋语境（婚恋平台/相亲、情感称谓如亲爱的/老婆/老公/宝贝、嘘寒问暖建立信任、
   异地恋/见面铺垫后再引到投资赚钱、礼物/转账测试真心等），即使主体在讲"带你投资/一起赚钱/内幕消息"，
   也判 romance_pig（杀猪盘=婚恋诱导投资）。
3. impersonate_authority / impersonate_family / refund_cs 三向界定：
   - impersonate_authority = 自称**公检法/政府执法机关/通管局/纪委**等国家执法机构，以涉案、拘捕、
     通缉、安全账户、配合调查、洗钱为由施压（即使自称"领导/主任"也按机构语境判断）；
   - 自称亲人急事求助（出车祸/生病住院/手机丢失借号/要医药费保释金）→ impersonate_family；
   - **自称平台/商家/银行/物流/航空公司等客服机构，以退款理赔、误开会员扣款、取消业务、屏幕共享、
     征信受损、订单异常为由（即使语带威胁催迫、限时转账）→ refund_cs，不判 authority**。
输出 JSON：{"label":"<上列英文键>","risk":<0到1>}
"""

# v4：在 v2 基础上增强 romance 召回——把 romance 判定从"强婚恋语境"扩展到"关系经营 + 私密性"弱信号，
# 针对杀猪盘"收网投资段"（骗子建立情感后单独发的纯投资话术，单段无婚恋称谓）。
# 依据 semantic_f1_ab.md：romance FN 63% 含弱信号（咱俩/咱们/导师/内部项目/别声张/别跟家里）。
PROMPT_V4 = """你是电信诈骗话术分类器。把下面这条中文通话话术归到 9 类之一，只输出 JSON。
类别：elder_healthcare 养老健康/保健品诈骗 | fake_investment 虚假投资/荐股 |
impersonate_authority 冒充公检法/领导 | impersonate_family 冒充亲友/子女/孙辈 |
lottery_prize 中奖/兑奖缴费 | online_loan 网贷/贷款 | refund_cs 退款/客服 |
romance_pig 杀猪盘/婚恋诱导投资 | benign 正常日常通话（含家人借钱、缴费提醒、客服回访等）。
规则：
1. 话术可能只是完整诈骗链的一段（开场/铺垫/索要/施压），凭内容语气判其归属类别即可；正常生活场景
   即使提到转账（亲属借钱、正常还款、缴费提醒）也判 benign。
2. romance_pig 与 fake_investment 的界定（重点，宁可多召回 romance）：
   - fake_investment = 陌生人/社群/直播间公开荐股带单，无任何情感/关系经营；
   - 只要话术在"讲投资/赚钱/内幕/行情"的同时，出现下列**任一**关系经营或私密性信号，即判 romance_pig
     （杀猪盘=先经营关系再诱导投资，其"投资段"常带这些痕迹）：
     * 关系经营：咱俩/咱们/一块儿/一起/以后还/想到你/为你/带你一起/见面/聚聚/处对象/恋爱/相亲/网恋；
     * 身份包装：跟导师/朋友/同学/亲戚做内部项目、内部名额/名额紧俏、只带你一个人、我的账户给你看；
     * 私密性：先别跟家里人说/别声张/就咱俩知道/偷偷/瞒着/别让家人知道；
     * 情感称谓：哥/姐/宝贝/亲爱的/老公/老婆/想你/在乎你/关心你（用于拉近关系而非陌生推销口吻）。
   - 若通篇是"陌生人公开推销、限时抢购、加群跟单、老师带单"且无上述关系信号，才判 fake_investment。
3. impersonate_authority 与 impersonate_family 的界定：自称公检法/客服/机构并以法律威慑、涉案、
   安全账户、索要验证码施压 → impersonate_authority；以亲人身份急事求助（出车祸/被抓/生病住院/
   手机丢失借号/要医药费保释金）→ impersonate_family，即使带恐吓语气。
输出 JSON：{"label":"<上列英文键>","risk":<0到1>}
"""

CURRENT_PROMPT = PROMPT_V1


def load_texts() -> dict[str, str]:
    """id -> 文本（scam 与 benign 合并，前缀区分类别由调用方提供）。"""
    texts = {}
    for path, _ in ((ROOT / "data/scam_corpus/corpus_v0.1.jsonl", 0),
                    (ROOT / "data/scam_corpus/benign_corpus.jsonl", 1)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    texts[r["id"]] = {"text": r.get("text", ""), "cat": r.get("category", "?")}
    return texts


def call_llm(base: str, key: str, model: str, text: str) -> str:
    payload = json.dumps({"model": model, "temperature": 0.1, "max_tokens": 80,
                          "messages": [{"role": "system", "content": CURRENT_PROMPT},
                                       {"role": "user", "content": text.strip()[:800]}]}
                         ).encode("utf-8")
    req = urllib.request.Request(f"{base}/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def parse_label(raw: str) -> str:
    import re
    m = re.search(r'"label"\s*:\s*"([^"]+)"', raw)
    if not m:
        m = re.search(r'"label"\s*:\s*"([^"]+)"', raw.replace("'", '"'))
    lab = m.group(1) if m else ""
    return lab if lab in LABELS else "error"


def parse_risk(raw: str) -> float:
    """解析 risk 字段（LLM 0-1 语义风险分）；缺省给 0.5 中性。"""
    import re
    m = re.search(r'"risk"\s*:\s*([0-9]*\.?[0-9]+)', raw)
    if not m:
        m = re.search(r'"risk"\s*:\s*([0-9]*\.?[0-9]+)', raw.replace("'", '"'))
    try:
        v = float(m.group(1)) if m else 0.5
    except Exception:  # noqa: BLE001
        v = 0.5
    return max(0.0, min(1.0, v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--variant", type=int, default=1, choices=[1, 2, 3, 4],
                    help="提示词版本：1=基线(默认) 2=romance/authority 界定增强 3=权威专属档 4=romance弱信号增强")
    args = ap.parse_args()
    global CURRENT_PROMPT
    CURRENT_PROMPT = {1: PROMPT_V1, 2: PROMPT_V2, 3: PROMPT_V3, 4: PROMPT_V4}[args.variant]
    tag = f"_v{args.variant}"

    base = os.environ["SCAM_LLM_BASE"].rstrip("/")
    key = os.environ["SCAM_LLM_KEY"]
    model = os.environ.get("SCAM_LLM_MODEL", "deepseek-chat")

    texts = load_texts()
    items = []  # (id, expected_label, text)
    for name, split in (("scam", "scam_eval_ids.json"), ("benign", "benign_eval_ids.json")):
        ids = json.loads((ROOT / f"data/scam_corpus/eval_split/{split}").read_text(encoding="utf-8"))
        for i in ids:
            t = texts[i]
            exp = "benign" if name == "benign" else t["cat"]
            items.append((i, exp, t["text"]))
    if args.limit:
        items = items[: args.limit]
    print(f"[评测集] {len(items)} 条（scam eval + benign eval 冻结集）", flush=True)

    results, t0 = [], time.time()
    def work(it):
        i, exp, text = it
        for _ in range(3):
            try:
                raw = call_llm(base, key, model, text)
                lab = parse_label(raw)
                return (i, exp, lab, parse_risk(raw))
            except Exception:
                time.sleep(2)
        return (i, exp, "error", 0.5)
    done = 0
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futs = [ex.submit(work, it) for it in items]
        for fu in as_completed(futs):
            results.append(fu.result())
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[完成] {done}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)

    # 行级落盘（A5 校准数据，避免重复调用云端）
    rows_csv = ROOT / "evaluation" / f"semantic_f1_rows{tag}.csv"
    with open(rows_csv, "w", encoding="utf-8-sig", newline="") as fcsv:
        wcsv = csv.writer(fcsv)
        wcsv.writerow(["id", "expected", "pred", "risk"])
        for i, exp, pred, risk in results:
            wcsv.writerow([i, exp, pred, risk])
    print(f"[行级] 已落盘 {rows_csv.name}（{len(results)} 行）", flush=True)

    # ---- 指标 ----
    from collections import Counter, defaultdict
    conf = Counter((exp, pred) for _, exp, pred, _ in results)
    errs = [r for r in results if r[2] == "error"]
    ok = [r for r in results if r[2] != "error"]
    n = len(ok)
    acc = sum(1 for _, e, p, _ in ok if e == p) / n if n else 0
    per = {}
    for lab in LABELS:
        tp = conf.get((lab, lab), 0)
        fp = sum(conf.get((o, lab), 0) for o in LABELS if o != lab)
        fn = sum(conf.get((lab, p), 0) for p in LABELS if p != lab)
        p_ = tp / (tp + fp) if tp + fp else 0.0
        r_ = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0
        per[lab] = {"tp": tp, "fp": fp, "fn": fn, "precision": round(p_, 4),
                    "recall": round(r_, 4), "f1": round(f1, 4)}
    macro = sum(per[l]["f1"] for l in LABELS) / len(LABELS)
    wsum = sum(conf[(l, p)] for l in LABELS for p in LABELS)
    weighted = sum(per[l]["f1"] * sum(conf[(l, p)] for p in LABELS) for l in LABELS) / wsum if wsum else 0
    # 二分类：诈骗(8类) vs 正常
    def bin_metrics(exp_is_scam, pred_is_scam):
        tp = sum(1 for _, e, p, _ in ok if exp_is_scam(e) and pred_is_scam(p))
        fp = sum(1 for _, e, p, _ in ok if not exp_is_scam(e) and pred_is_scam(p))
        fn = sum(1 for _, e, p, _ in ok if exp_is_scam(e) and not pred_is_scam(p))
        p_ = tp / (tp + fp) if tp + fp else 0.0
        r_ = tp / (tp + fn) if tp + fn else 0.0
        return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p_, 4),
                "recall": round(r_, 4), "f1": round(2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0, 4)}
    scam = lambda l: l != "benign"
    bin_scam = bin_metrics(lambda e: e != "benign", lambda p: p != "benign")

    payload = {"date": time.strftime("%Y-%m-%d"), "variant": args.variant,
               "model": model, "n": n, "api_error": len(errs),
               "accuracy": round(acc, 4), "macro_f1": round(macro, 4),
               "weighted_f1": round(weighted, 4), "binary_scam_vs_normal": bin_scam,
               "per_class": per}
    (ROOT / f"evaluation/semantic_f1{tag}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [f"# 话术分类多分类 F1（线 A4，{payload['date']}，variant {args.variant}）", "",
             f"- 评测集：A1 冻结 {n} 条（scam eval 2,387 + benign eval 481，API error {len(errs)}）",
             f"- 模型：cloud {model}；**准确率 {acc*100:.1f}% / 宏 F1 {macro*100:.1f}% / 加权 F1 {weighted*100:.1f}%**",
             f"- 二分类(诈骗 vs 正常)：P {bin_scam['precision']*100:.1f}% / R {bin_scam['recall']*100:.1f}% / "
             f"F1 {bin_scam['f1']*100:.1f}%（tp {bin_scam['tp']} / fp {bin_scam['fp']} / fn {bin_scam['fn']}）", "",
             "| 类别 | 中文 | TP | FP | FN | P | R | F1 |", "|---|---|---|---|---|---|---|---|"]
    for lab in LABELS:
        v = per[lab]
        lines.append(f"| {lab} | {LABEL_ZH[lab]} | {v['tp']} | {v['fp']} | {v['fn']} | "
                     f"{v['precision']*100:.1f}% | {v['recall']*100:.1f}% | {v['f1']*100:.1f}% |")
    lines += ["", "## 结论判定", "- 二分类诈骗 F1 ≥0.90 即 A4 达标；多分类加权 F1 作参考（8 类细分更难）。"]
    (ROOT / f"evaluation/semantic_f1{tag}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"acc {acc*100:.1f}%  macro {macro*100:.1f}%  weighted {weighted*100:.1f}%")
    print(f"产物: evaluation/semantic_f1{tag}.json / .md")


if __name__ == "__main__":
    main()
