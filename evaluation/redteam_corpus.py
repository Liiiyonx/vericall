# -*- coding: utf-8 -*-
"""
谛听 VeriCall · 通道③ 红队方言语料（离线合成版）
================================================================
为「红队方言鲁棒性评测」提供**合成**文本语料：覆盖多方言的诈骗话术与正常闲聊。

设计意图（为何要合成方言语料）：
  通道③目前靠关键词命中估算话术风险（见 fusion/rule_scorer.py）。
  方言诈骗话术常用方言词/繁体/谐音，往往**不含**普通话关键词 → 规则基线漏拦。
  本语料刻意让 粤语/闽南 等方言诈骗样本用方言词绕过关键词，从而在评测里
  暴露「方言漏拦盲区」；而 普通话/河南/四川/东北 诈骗样本保留关键词 → 命中。
  正常样本横跨各方言，用于统计误拦率(FA)。

每条样本 = {text, dialect, label, note}，label ∈ {scam, normal}。
语料为**合成文本**（note 标注「合成」），不涉及真实个人录音，符合伦理要求。

用法：
    from evaluation.redteam_corpus import build_corpus, load_text_corpus
    paths = build_corpus(REDTEAM_DIR)          # 写 data/redteam/<dialect>/manifest.json
    samples = load_text_corpus(REDTEAM_DIR)    # 读回 [(dialect, text, label, note)]
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from paths import REDTEAM_DIR  # noqa: E402


@dataclass
class Sample:
    dialect: str
    text: str
    label: str          # "scam" / "normal"
    note: str


# 各方言：(诈骗话术列表, 正常闲聊列表)
# 诈骗样本里：普通话系保留关键词（命中）；粤语/闽南用方言词/繁体绕过（漏拦）。
_DIALECT_SCENARIOS: dict[str, tuple[list[str], list[str]]] = {
    "mandarin": (
        [
            "妈我是同学号我手机摔了急用五万块你转卡号别告诉我爸",
            "这里是市公安局你银行卡涉嫌洗钱要把资金转安全账户否则冻结逮捕",
            "爸我领导急用钱你先垫付两万到这个二维码别声张",
        ],
        [
            "妈我周末回来吃饭你别买菜了路上买爸降压药吃完没",
            "喂您好您的快递放丰巢柜了取件码3372请领取",
            "爸今天降温记得加衣服我下班早点回",
        ],
    ),
    "henan": (
        [
            "妈恁孩儿手机掉厕所了这是同学号急用五万你转卡号别给俺爸说",
            "哥我是领导微信垫付两万块急用明天还你别给嫂子说",
            "婶儿公安局说你卡涉嫌洗黑钱赶紧转安全账户要不就抓人",
        ],
        [
            "妈今黑儿咱吃烩面吧我下班捎回去爸那药吃了没",
            "喂快递搁门卫了取件码6621记着拿",
            "爸天冷多穿点我寻思着早回",
        ],
    ),
    "cantonese": (
        [
            "阿媽係我啊我電話跌落水呢個係同學個號我喺學校出咗事急住使錢你快啲過啲錢去個戶口唔好講畀老豆聽",
            "呢度係公安局你張卡疑涉洗黑錢要將筆錢過去監管戶口先至唔會拉人",
            "阿爸我老細急住使錢你過兩萬去個二維碼唔好同阿媽講",
        ],
        [
            "阿媽我聽日返嚟食飯唔使買餸我順便帶爸啲降壓藥食完未",
            "喂你個快遞放咗喺豐巢櫃取件碼8890記得拎",
            "阿爸天氣凍多件衫我收工早啲返",
        ],
    ),
    "sichuan": (
        [
            "妈我是别个号的我手机棒老火了急用五万你转到卡号莫跟我爸说",
            "哥我是领导你先垫付两万到这个码上急用莫跟嫂子讲",
            "姐公安局说你卡遭洗钱咯快转安全账户不然要抓人",
        ],
        [
            "妈今晚吃火锅嘛我下班带起爸的药吃了没得",
            "喂你快递搁门卫取件码5530记到拿",
            "爸降温咯多穿点我早点回",
        ],
    ),
    "dongbei": (
        [
            "妈我是同学号我手机秃噜了急用五万你转卡号别跟我爸说",
            "哥我是领导你先垫付两万到这码上急用别跟嫂子说",
            "大姨公安局说你卡涉嫌洗钱赶紧转安全账户要不逮人",
        ],
        [
            "妈今晚整点啥吃我下班捎回去爸那药吃了没",
            "喂快递搁门卫了取件码7741记着取",
            "爸天冷穿厚点我早点回",
        ],
    ),
    "minnan": (
        [
            "阿母是我啦我手機挵歹去這是同学的號我佇學校出代誌急欲用錢你緊來提錢去這个戶頭莫共阿爸講",
            "這是警察局你彼張卡疑似洗钱愛提錢過去監管戶頭才毋會捉人",
            "阿爸我頭家急欲用錢你提兩萬去這个碼莫共阿母講",
        ],
        [
            "阿母我暗仔轉來食飯免買菜我順紲帶爸的降壓藥食完無",
            "喂你個快遞放佇豐巢櫃取件碼2109記得提",
            "阿爸天冷加衫我收工早轉來",
        ],
    ),
}


def build_corpus(dest: Path | None = None) -> list[Path]:
    """把内置方言场景写成 data/redteam/<dialect>/manifest.json。

    返回写入的 manifest 路径列表。dest 默认 REDTEAM_DIR。
    每个 manifest 项为 {wav:null, text, dialect, label, note(标注合成)}。
    """
    dest = Path(dest or REDTEAM_DIR)
    written: list[Path] = []
    for dialect, (scams, normals) in _DIALECT_SCENARIOS.items():
        d = dest / dialect
        d.mkdir(parents=True, exist_ok=True)
        rows = []
        for t in scams:
            rows.append({"wav": None, "text": t, "dialect": dialect,
                         "label": "scam", "note": f"合成-诈骗话术({dialect})"})
        for t in normals:
            rows.append({"wav": None, "text": t, "dialect": dialect,
                         "label": "normal", "note": f"合成-正常闲聊({dialect})"})
        man = d / "manifest.json"
        man.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        written.append(man)
    return written


def load_text_corpus(redteam_dir: Path | None = None) -> list[Sample]:
    """读回所有带 text 字段的样本，返回 Sample 列表。"""
    redteam_dir = Path(redteam_dir or REDTEAM_DIR)
    out: list[Sample] = []
    if not redteam_dir.is_dir():
        return out
    for man in sorted(redteam_dir.rglob("manifest.json")):
        rows = json.loads(man.read_text(encoding="utf-8"))
        for r in rows:
            if not r.get("text"):
                continue
            out.append(Sample(
                dialect=r.get("dialect", man.parent.name),
                text=r["text"],
                label=r.get("label", "normal"),
                note=r.get("note", ""),
            ))
    return out


if __name__ == "__main__":
    ps = build_corpus()
    print(f"已生成 {len(ps)} 个方言 manifest：")
    for p in ps:
        print(f"  {p}")
    n = len(load_text_corpus())
    print(f"读回样本总数：{n}")
