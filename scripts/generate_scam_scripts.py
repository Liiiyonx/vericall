# -*- coding: utf-8 -*-
"""
红队诈骗话术生成器
==================
职责：用本地 deepseek-r1 批量扩写诈骗话术变体，产出带标签的 jsonl，
      供后续 GPT-SoVITS 声音克隆 → 通道③评测 / 检测模型负样本训练。

产出格式（data/redteam/scam_scripts.jsonl，UTF-8，一行一条）：
    {"id": "MT-0001", "category": "impersonation", "label": "scam",
     "text": "...", "style": "口语/方言语感标记", "source": "r1-generated"}

设计要点：
- 每类诈骗由「种子模板 + 风格轴（称呼/事由/金额/紧迫度/方言口音）」组合扩写，
  保证多样性可控、可复现（同 seed 同输出）。
- 生成后用通道③自检：r1 自己生成 → 通道③再判一遍，风险分 <0.6 的样本丢弃，
  避免低质量样本污染评测集（自洽过滤）。
- 方言层：先生成普通话口语稿，GPT-SoVITS 阶段再做方言化（福州话/闽南话/普通话带腔），
  本脚本负责产出「方言语感标记」供 TTS 提示词使用。

用法：
    python generate_scam_scripts.py --per-category 20   # 每类20条，共~80条
    python generate_scam_scripts.py --dry-run           # 只打印不写文件
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fusion.semantic_channel import SemanticChannel, SemanticResult  # noqa: E402

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "deepseek-r1:8b"
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "redteam" / "scam_scripts.jsonl"

# 四类诈骗种子模板（{var} 由风格轴填充）
SEEDS = {
    "impersonation": [
        "喂{relation}，是我啊，{phone_excuse}。{matter}，你先转{money}到{pay_target}，{keep_secret}。",
        "{relation}吗？我是你{kin}的{role}，他{incident}了，现在急需{money}处理，{urgency}。",
    ],
    "urgency_threat": [
        "这里是{authority}，你{doc}涉嫌{crime}，需将资金转入{pay_target}配合清查，{deadline}否则{consequence}。",
        "{relation}，你{kin}在我手上……不对，你{kin}出了{incident}，马上转{money}过来救人，{no_police}！",
    ],
    "money_request": [
        "{relation}，我在开{benefit}，有个{opportunity}，垫付{money}就能{gain}，{urgency}。",
        "妈，我朋友急用{money}，你先转到{pay_target}，我下周还你，{keep_secret}。",
    ],
    "credential": [
        "您好，我是{service}客服，您的账户存在异常，需要验证。手机马上会收到{code_name}，念给我就能解除冻结。",
        "我是{authority}的，做一下身份核实：报一下{card_info}，再对着手机{face_action}……好，验证通过。",
    ],
}

# 风格轴（扩写维度）
STYLE_AXES = {
    "relation": ["妈", "爸", "奶奶", "爷爷", "阿婆", "老李", "王阿姨"],
    "kin": ["儿子", "女儿", "孙子", "外孙", "老伙计"],
    "role": ["班主任", "同事", "辅导员", "主治医生"],
    "phone_excuse": ["手机摔坏了这是同学的号", "手机被收了我偷偷用别人手机打的", "我在外地手机丢了"],
    "matter": ["我出车祸撞了人私了要赔钱", "我被骗了要还平台的钱", "我赌钱输急了"],
    "incident": ["车祸", "突发疾病住院", "被人打了"],
    "money": ["三万", "五万", "八万", "两万六", "一万八"],
    "pay_target": ["这个卡号", "安全账户", "这个二维码", "对公账户"],
    "keep_secret": ["先别告诉我爸", "别打电话给我老师", "这事别声张"],
    "urgency": ["马上就要", "半小时内必须到账", "晚一步就完了"],
    "authority": ["市公安局", "检察院", "通讯管理局", "医保局"],
    "doc": ["名下的银行卡", "医保账户", "身份证登记信息"],
    "crime": ["洗钱案件", "医保诈骗", "身份盗用"],
    "deadline": ["今天下午五点前", "两小时内"],
    "consequence": ["冻结你全部账户并逮捕你", "案件移交法院强制执行"],
    "service": ["银行", "快递公司", "电商平台"],
    "code_name": ["验证码", "短信校验码"],
    "card_info": ["银行卡号和取款密码", "身份证号和短信验证码"],
    "face_action": ["眨眨眼摇摇头做人脸识别"],
    "benefit": ["投资说明会", "养生讲座", "免费体检"],
    "opportunity": ["内部理财名额", "原始股认购", "高息存款"],
    "gain": ["本金翻倍", "月息百分之十", "稳赚不赔"],
    "no_police": ["别报警", "不许告诉别人"],
}

# 方言语感标记（GPT-SoVITS 阶段的 TTS 提示）
DIALECT_STYLES = ["普通话-年轻男声", "普通话-中年女声", "普通话-带福州腔", "闽南话口音-男", "普通话-老年男声"]

GEN_PROMPT = """你是电话诈骗案例库构建助手（用于反诈系统的对抗评测）。把下面的"话术骨架"扩写成一段真实自然的通话口语文本。

要求：
1. 保留骨架里的关键要素（身份、事由、金额、紧迫感、支付方式），可自然改述
2. 口语化，像真人打电话，可以有停顿、语气词（喂、啊、那个、你听我说）
3. 30-80字，一段话
4. 只输出扩写后的正文，不要任何解释或引号

话术骨架：{seed}"""


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def _call_llm(prompt: str, timeout: int = 120) -> str:
    payload = json.dumps({
        "model": MODEL,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0.9},  # 生成任务要多样性
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return _strip_think(data.get("message", {}).get("content", ""))


def make_seed(category: str, rng: random.Random) -> str:
    seed = rng.choice(SEEDS[category])
    for m in set(re.findall(r"\{(\w+)\}", seed)):
        if m in STYLE_AXES:
            seed = seed.replace("{" + m + "}", rng.choice(STYLE_AXES[m]))
    return seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=20, help="每类生成条数")
    ap.add_argument("--seed", type=int, default=42, help="随机种子（可复现）")
    ap.add_argument("--dry-run", action="store_true", help="只打印不写文件")
    ap.add_argument("--no-filter", action="store_true", help="跳过通道③自检过滤")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    ch = SemanticChannel()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    out, dropped = [], 0
    cat_names = list(SEEDS)
    total = args.per_category * len(cat_names)
    print(f"计划生成 {total} 条（每类 {args.per_category}），模型 {MODEL}\n")

    for i in range(args.per_category):
        for cat in cat_names:
            seed = make_seed(cat, rng)
            try:
                text = _call_llm(GEN_PROMPT.format(seed=seed))
            except Exception as e:
                print(f"  [skip] LLM 调用失败: {e}")
                continue
            text = text.strip().strip('"').strip()
            # 清洗模型偶尔带出的前缀（冒烟测试实测：会输出"扩写后文本："等引导语）
            text = re.sub(r"^(扩写后文本|正文|话术|扩写结果)\s*[:：]\s*", "", text).strip()
            if not (10 <= len(text) <= 300):
                dropped += 1
                continue

            rec = {
                "id": f"{cat[:2].upper()}-{len(out)+1:04d}",
                "category": cat,
                "label": "scam",
                "text": text,
                "style": rng.choice(DIALECT_STYLES),
                "source": "r1-generated",
                "seed_skeleton": seed,
            }

            # 通道③自检：低风险分的生成样本视为质量差，丢弃
            if not args.no_filter:
                r: SemanticResult = ch.analyze(text)
                if r.risk < 0.6:
                    dropped += 1
                    print(f"  [drop {rec['id']}] risk={r.risk} {text[:30]}...")
                    continue
                rec["self_check_risk"] = r.risk

            out.append(rec)
            print(f"  [ok {rec['id']}] ({rec['style']}) {text[:40]}...")

    if args.dry_run:
        print(f"\n(dry-run) 共 {len(out)} 条，丢弃 {dropped} 条")
        return

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n完成：{len(out)} 条 -> {OUT_PATH}（丢弃 {dropped} 条低质量/失败）")


if __name__ == "__main__":
    main()
