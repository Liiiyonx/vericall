# -*- coding: utf-8 -*-
"""P2 红队：方言鲁棒性 + 对抗扰动（离线，纯 numpy/stdlib）。

验证：
  - 扰动模块确定性，且能破坏关键词子串匹配（space 插空格 / homophone 形近替换）；
  - 合成语料生成与读回正确（6 方言 × 6 样本，scam/normal 各半）；
  - 离线评测能复现核心结论：粤语/闽南 100% 漏拦、普通话系 0% 漏拦、
    误拦率 0%、对抗逃避率(space) 100%。
"""
import sys
from pathlib import Path

import pytest

# conftest 已将 evaluation / src 加入 sys.path
import redteam_perturb as rp          # noqa: E402
import redteam_corpus as corpus_mod   # noqa: E402
import redteam_eval                   # noqa: E402
from fusion.rule_scorer import rule_score  # noqa: E402


# ------------------------------------------------------------------ #
# 扰动模块
# ------------------------------------------------------------------ #
def test_perturb_space_breaks_substring():
    assert rp.perturb("转账", "space") == "转 账"
    # 插空格后原关键词不再作为子串出现
    p = rp.perturb("急用五万块你转卡号", "space")
    assert "五万" not in p and "卡号" not in p


def test_perturb_homophone_substitutes():
    assert rp.perturb("转账", "homophone") == "专帐"   # 转->专, 账->帐


def test_perturb_deterministic():
    t = "妈我是同学号急用五万块"
    assert rp.perturb(t, "space") == rp.perturb(t, "space")
    assert rp.perturb(t, "homophone") == rp.perturb(t, "homophone")


def test_perturb_unknown_strategy_raises():
    with pytest.raises(ValueError):
        rp.perturb("x", "bogus")


# ------------------------------------------------------------------ #
# 语料生成
# ------------------------------------------------------------------ #
def test_corpus_build_and_load(tmp_path):
    ps = corpus_mod.build_corpus(tmp_path)
    assert len(ps) == 6
    samples = corpus_mod.load_text_corpus(tmp_path)
    assert len(samples) == 36
    # 每方言 scam/normal 各 3
    by_dialect = {}
    for s in samples:
        by_dialect.setdefault(s.dialect, {"scam": 0, "normal": 0})
        by_dialect[s.dialect][s.label] += 1
    for d, c in by_dialect.items():
        assert c["scam"] == 3 and c["normal"] == 3, d


# ------------------------------------------------------------------ #
# 离线评测复现核心结论
# ------------------------------------------------------------------ #
def _run(tmp_path):
    corpus_mod.build_corpus(tmp_path)
    samples = corpus_mod.load_text_corpus(tmp_path)
    return redteam_eval.evaluate_text(samples, threshold=0.5)


def test_overall_false_alarm_is_zero(tmp_path):
    res = _run(tmp_path)
    assert res["overall"]["fa_pct"] == 0.0     # 正常样本零误拦


def test_dialect_gap_cantonese_minnan_fully_missed(tmp_path):
    res = _run(tmp_path)
    by = {r["dialect"]: r for r in res["per_dialect"]}
    assert by["cantonese"]["miss_pct"] == 100.0
    assert by["minnan"]["miss_pct"] == 100.0
    # 普通话系应被命中
    assert by["mandarin"]["miss_pct"] == 0.0
    assert by["henan"]["miss_pct"] == 0.0


def test_adversarial_space_evasion_full(tmp_path):
    res = _run(tmp_path)
    # 在已检出的普通话系诈骗上，插空格扰动应 100% 绕过
    assert res["adversarial_evasion_pct"]["space"] == 100.0
    # 形近替换也应造成一定逃避（>0）
    assert res["adversarial_evasion_pct"]["homophone"] > 0.0


def test_scored_scam_detected_but_space_evades(tmp_path):
    """单条端到端：命中诈骗被检出，插空格后跌到阈值下。"""
    corpus_mod.build_corpus(tmp_path)
    scam = "妈我是同学号我手机摔了急用五万块你转卡号别告诉我爸"
    assert rule_score(scam).risk >= 0.5
    assert rule_score(rp.perturb(scam, "space")).risk < 0.5
