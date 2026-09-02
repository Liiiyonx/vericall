# -*- coding: utf-8 -*-
"""规则评分器单测（任务书 P2-6）：四类话术命中 + 正常文本零误报。

不依赖 torch / funasr / Ollama，可在纯 CI 环境运行。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fusion.rule_scorer import rule_score


def test_normal_text_zero_hit():
    r = rule_score("妈，我周末回来吃饭，你别买菜了我在路上买。对了爸的降压药吃完没？")
    assert r.category == "normal"
    assert r.risk <= 0.05


def test_normal_business_zero_hit():
    r = rule_score("您好，您的快递已放到小区丰巢柜，取件码是 3372，请及时领取。")
    assert r.category == "normal"
    assert r.risk <= 0.05


def test_impersonation_plus_money_request_high_risk():
    # 场景 B 话术：冒充 + 要钱 + 恐吓，应触发高分类别命中，风险封顶 1.0
    r = rule_score("喂妈，是我啊，我手机摔坏了这是同学的号。我在学校出点事急用钱，"
                   "你先转五万到这个卡号，别告诉我爸，快点啊要不来不及了。")
    assert r.category in ("impersonation", "money_request", "urgency_threat")
    assert r.hits.get("impersonation", 0) >= 1
    assert r.hits.get("money_request", 0) >= 1
    assert r.risk >= 0.7


def test_credential_category():
    r = rule_score("把你的验证码发给我，还有银行卡密码，我要验证一下身份。")
    assert r.hits.get("credential", 0) >= 1
    assert r.risk > 0.05


def test_risk_formula_cap():
    # 多类别多词命中，风险应封顶 1.0，不溢出
    txt = ("我是你领导，公安说你涉嫌洗钱，账户要冻结，马上转账到安全账户，"
           "把验证码和密码发来，银行卡号也报一下。")
    r = rule_score(txt)
    assert r.risk <= 1.0


def test_empty_text_safe():
    r = rule_score("")
    assert r.category == "normal"
    assert r.risk <= 0.05
