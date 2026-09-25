#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Seed realistic, idempotent AI analysis records for judge demos.

The generated records use the normal family history store, so the child
dashboard, alert list, CSV export and weekly metrics all exercise the
same production data path as real detections.

Examples:
    python scripts/seed_demo_history.py
    python scripts/seed_demo_history.py --dry-run
    python scripts/seed_demo_history.py --reset
    python scripts/seed_demo_history.py --family-id pilot-01
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from api import history_store  # noqa: E402
from paths import DEFAULT_FAMILY_ID, HIST_FILE, ROOT as PROJECT_ROOT  # noqa: E402
from private_fs import write_private_text  # noqa: E402

DEMO_SOURCE = "ai-demo"
SEED_IDS = tuple(f"aidemo{index:02d}" for index in range(1, 9))


def _channel(name: str, score: float, label: str, detail: str,
             confidence: float) -> dict:
    return {
        "name": name,
        "score": round(score, 3),
        "label": label,
        "detail": detail,
        "confidence": round(confidence, 3),
    }


def _sample(*, seed_id: str, days_ago: int, hour: int, minute: int,
            name: str, scene: str, caller_number: str, final: str,
            score: float, confidence: float, elapsed_s: float,
            rationale: str, transcript: str, acoustic: tuple,
            voiceprint: tuple, semantic: tuple) -> dict:
    """Build one record without a timestamp; build_records applies dates."""
    return {
        "id": seed_id,
        "days_ago": days_ago,
        "hour": hour,
        "minute": minute,
        "name": name,
        "scene": scene,
        "source": DEMO_SOURCE,
        "household": "",
        "caller_number": caller_number,
        "elapsed_s": elapsed_s,
        "final": final,
        "score": round(score, 3),
        "confidence": round(confidence, 3),
        "rationale": rationale,
        "channels": [
            _channel("acoustic", *acoustic),
            _channel("voiceprint", *voiceprint),
            _channel("semantic", *semantic),
        ],
        "offline": False,
        "demo": True,
        "transcript": transcript,
    }


def _scenarios() -> list[dict]:
    return [
        _sample(
            seed_id="aidemo01",
            days_ago=6,
            hour=9,
            minute=16,
            name="AI 克隆女儿紧急借钱",
            scene="克隆亲人",
            caller_number="+86 10 5550 0186",
            final="block",
            score=0.94,
            confidence=0.96,
            elapsed_s=3.8,
            rationale=(
                "声学信号存在明显合成断续，声纹与已登记女儿不匹配；"
                "话术同时命中紧急借钱、手机损坏和禁止联系家人，"
                "综合判断为高风险克隆诈骗。"
            ),
            transcript=(
                "妈，我手机摔坏了，这是同学的号码。学校急用五万块，"
                "你先转给我，千万别告诉我爸，晚点我再解释。"
            ),
            acoustic=(
                0.94,
                "spoof",
                "高频细节与呼吸段不连续，疑似语音合成",
                0.95,
            ),
            voiceprint=(
                0.91,
                "mismatch",
                "与已登记女儿声纹相似度约 0.31，低于家庭核验阈值",
                0.95,
            ),
            semantic=(
                0.96,
                "scam",
                "命中紧急借钱、身份替换、隐瞒家人和催促转账话术",
                0.97,
            ),
        ),
        _sample(
            seed_id="aidemo02",
            days_ago=5,
            hour=10,
            minute=42,
            name="冒充公检法要求转入安全账户",
            scene="冒充公检法",
            caller_number="021-5550 2711",
            final="block",
            score=0.97,
            confidence=0.98,
            elapsed_s=2.9,
            rationale=(
                "声纹未匹配任何家庭成员，来电话术完整复现公检法诈骗链："
                "制造涉案恐慌、要求保密并转入所谓安全账户，风险极高。"
            ),
            transcript=(
                "这里是市公安局，你名下银行卡涉嫌洗钱。现在必须配合清查，"
                "把存款转到安全账户，案件结束会自动退还，不能告诉别人。"
            ),
            acoustic=(
                0.46,
                "uncertain",
                "电话压缩明显，声学证据不足以单独定性",
                0.62,
            ),
            voiceprint=(
                0.92,
                "mismatch",
                "与家庭声纹库全部成员不匹配",
                0.94,
            ),
            semantic=(
                0.99,
                "scam",
                "命中冒充公检法、涉案恐吓、安全账户和保密要求",
                0.99,
            ),
        ),
        _sample(
            seed_id="aidemo03",
            days_ago=4,
            hour=16,
            minute=8,
            name="虚假投资导师诱导追加资金",
            scene="虚假投资",
            caller_number="+86 21 5550 9032",
            final="block",
            score=0.91,
            confidence=0.94,
            elapsed_s=4.6,
            rationale=(
                "声学未发现明显合成痕迹，但通话方为陌生声纹；"
                "话术以内部渠道、稳赚不赔和限时名额持续催促转账，"
                "符合虚假投资诈骗的典型模式。"
            ),
            transcript=(
                "今晚是最后一批内部名额，老师会带着操作，稳赚不赔。"
                "你先追加十万，盈利后连本带利一起提现。"
            ),
            acoustic=(
                0.28,
                "bonafide",
                "未发现明显合成痕迹，通话压缩在正常范围",
                0.78,
            ),
            voiceprint=(
                0.87,
                "mismatch",
                "陌生来电声纹，与家庭成员均不匹配",
                0.91,
            ),
            semantic=(
                0.97,
                "scam",
                "命中内部渠道、稳赚不赔、限时名额和追加转账",
                0.98,
            ),
        ),
        _sample(
            seed_id="aidemo04",
            days_ago=3,
            hour=20,
            minute=14,
            name="情感陪伴后引导转账",
            scene="杀猪盘苗头",
            caller_number="+86 13 5550 6408",
            final="caution",
            score=0.79,
            confidence=0.83,
            elapsed_s=5.1,
            rationale=(
                "当前没有直接要求转账，暂未达到强制拦截阈值；"
                "但陌生声纹、长期陪伴话术与投资平台线索叠加，"
                "存在杀猪盘关系培养风险，建议持续关注。"
            ),
            transcript=(
                "最近和你聊天特别安心，我想带你一起做点副业。"
                "等我们熟悉了，我再教你在这个平台慢慢存钱。"
            ),
            acoustic=(
                0.21,
                "bonafide",
                "暂未发现声学合成痕迹",
                0.76,
            ),
            voiceprint=(
                0.84,
                "mismatch",
                "陌生声纹，身份尚未确认",
                0.89,
            ),
            semantic=(
                0.91,
                "suspicious",
                "命中情感拉近、共同投资、平台存钱和延迟转账引导",
                0.92,
            ),
        ),
        _sample(
            seed_id="aidemo05",
            days_ago=2,
            hour=11,
            minute=27,
            name="冒充平台客服办理退款",
            scene="退款客服",
            caller_number="400 555 3126",
            final="caution",
            score=0.67,
            confidence=0.79,
            elapsed_s=3.4,
            rationale=(
                "声学通道出现轻度合成嫌疑，声纹与家庭成员不匹配；"
                "话术涉及退款、屏幕共享和验证码，建议先挂断并通过官方渠道核实。"
            ),
            transcript=(
                "您购买的商品有质量问题，我们给您三倍退款。"
                "请打开屏幕共享，我教您完成验证，验证码也发给我确认一下。"
            ),
            acoustic=(
                0.68,
                "spoof",
                "轻微合成痕迹，背景噪声衔接不够自然",
                0.74,
            ),
            voiceprint=(
                0.73,
                "mismatch",
                "陌生客服声纹，未在家庭声纹库登记",
                0.86,
            ),
            semantic=(
                0.78,
                "suspicious",
                "命中主动退款、屏幕共享和索取验证码",
                0.88,
            ),
        ),
        _sample(
            seed_id="aidemo06",
            days_ago=1,
            hour=14,
            minute=5,
            name="银行正常业务回访",
            scene="正常回访",
            caller_number="95588",
            final="allow",
            score=0.09,
            confidence=0.91,
            elapsed_s=2.4,
            rationale=(
                "声学与语义均未发现风险特征；来电仅确认业务办理意向，"
                "未索要密码、验证码或要求转账，判定为正常服务回访。"
            ),
            transcript=(
                "您好，这里是银行业务回访，想确认您上午办理的储蓄卡"
                "是否需要开通短信提醒。您不方便的话可以稍后到网点办理。"
            ),
            acoustic=(
                0.06,
                "bonafide",
                "自然真人语音，未发现合成痕迹",
                0.94,
            ),
            voiceprint=(
                0.32,
                "unknown",
                "客服声纹未登记，不作为风险加分项",
                0.72,
            ),
            semantic=(
                0.05,
                "benign",
                "正常业务回访，无转账、验证码或隐私索取",
                0.92,
            ),
        ),
        _sample(
            seed_id="aidemo07",
            days_ago=0,
            hour=8,
            minute=41,
            name="陌生快递通知（已核验）",
            scene="快递通知",
            caller_number="+86 10 5550 4419",
            final="allow",
            score=0.19,
            confidence=0.84,
            elapsed_s=2.1,
            rationale=(
                "来电只说明快递位置，没有链接、收费或验证码要求；"
                "结合近期物流记录核实后未发现异常，判定为低风险。"
            ),
            transcript=(
                "您的快递已经放到东门快递柜，取件码我会发短信。"
                "如果柜门打不开，可以明天再取，不需要支付额外费用。"
            ),
            acoustic=(
                0.08,
                "bonafide",
                "自然真人语音，压缩质量正常",
                0.92,
            ),
            voiceprint=(
                0.48,
                "unknown",
                "配送员声纹未登记，不参与风险加分",
                0.68,
            ),
            semantic=(
                0.14,
                "benign",
                "通知型话术，索要验证码和转账风险词均为零",
                0.89,
            ),
        ),
        _sample(
            seed_id="aidemo08",
            days_ago=0,
            hour=10,
            minute=32,
            name="家人本人来电（周末问候）",
            scene="家人真声",
            caller_number="",
            final="allow",
            score=0.06,
            confidence=0.95,
            elapsed_s=1.9,
            rationale=(
                "声学为自然真人语音，声纹与已登记女儿高度匹配；"
                "通话内容为日常问候，三路均无风险信号。"
            ),
            transcript=(
                "妈，我周末回来吃饭，你别买菜了我在路上买。"
                "爸的降压药吃完没有？记得按时量血压。"
            ),
            acoustic=(
                0.03,
                "bonafide",
                "自然真人语音，声学质量正常",
                0.97,
            ),
            voiceprint=(
                0.02,
                "match",
                "与已登记女儿声纹高度匹配，相似度约 0.82",
                0.96,
            ),
            semantic=(
                0.06,
                "benign",
                "日常家庭问候，无转账、恐吓或隐私索取",
                0.96,
            ),
        ),
    ]


def build_records(now: datetime | None = None) -> list[dict]:
    """Return chronological seed records with timestamps relative to now."""
    now = now or datetime.now()
    records = []
    for item in _scenarios():
        target = now.replace(
            hour=int(item.pop("hour")),
            minute=int(item.pop("minute")),
            second=0,
            microsecond=0,
        ) - timedelta(days=int(item.pop("days_ago")))
        if target > now:
            target -= timedelta(days=1)
        record = dict(item)
        record["time"] = target.strftime("%Y-%m-%d %H:%M:%S")
        records.append(record)
    records.sort(key=lambda record: record["time"])
    return records


def _history_path(family_id: str) -> Path:
    return history_store.history_path_for(family_id, HIST_FILE)


def _existing_seed_ids(family_id: str) -> set[str]:
    return {
        str(record.get("id") or "")
        for record in history_store.iter_history(
            HIST_FILE,
            family_id=family_id,
            default_family_id=DEFAULT_FAMILY_ID,
        )
        if str(record.get("id") or "") in SEED_IDS
    }


def _remove_seed_records(family_id: str, *, dry_run: bool = False) -> int:
    """Remove only records generated by this script from the family shard."""
    path = _history_path(family_id)
    if not path.is_file():
        return 0

    kept_lines: list[str] = []
    removed = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            kept_lines.append(line)
            continue
        record_id = str(record.get("id") or "") if isinstance(record, dict) else ""
        is_seed = (
            isinstance(record, dict)
            and (
                record_id in SEED_IDS
                or str(record.get("source") or "").strip().lower() == DEMO_SOURCE
            )
        )
        if is_seed:
            removed += 1
        else:
            kept_lines.append(line)

    if removed and not dry_run:
        payload = "\n".join(kept_lines)
        if payload:
            payload += "\n"
        with history_store._history_lock:
            write_private_text(path, payload)
    return removed


def seed_demo_history(*, family_id: str = DEFAULT_FAMILY_ID,
                      history_dir: str | Path | None = None,
                      reset: bool = False,
                      dry_run: bool = False) -> dict:
    """Insert the fixed demo set and return a machine-readable summary."""
    previous = os.environ.get("VERICALL_HISTORY_DIR")
    if history_dir is not None:
        configured = Path(history_dir).expanduser()
        if not configured.is_absolute():
            configured = PROJECT_ROOT / configured
        os.environ["VERICALL_HISTORY_DIR"] = str(configured.resolve())

    try:
        before = _existing_seed_ids(family_id)
        removed = _remove_seed_records(family_id, dry_run=dry_run) if reset else 0
        records = build_records()
        pending = [record for record in records if record["id"] not in before]
        if reset:
            pending = records
        if not dry_run:
            for record in pending:
                history_store.append_history(
                    record, HIST_FILE, family_id=family_id)
        after = (
            set(SEED_IDS)
            if reset or not dry_run
            else before | {record["id"] for record in pending}
        )
        return {
            "ok": True,
            "family_id": family_id,
            "history_path": str(_history_path(family_id)),
            "source": DEMO_SOURCE,
            "inserted": len(pending),
            "removed": removed,
            "total": len(after),
            "dry_run": bool(dry_run),
        }
    finally:
        if history_dir is not None:
            if previous is None:
                os.environ.pop("VERICALL_HISTORY_DIR", None)
            else:
                os.environ["VERICALL_HISTORY_DIR"] = previous


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family-id",
        default=DEFAULT_FAMILY_ID,
        help=f"target family history shard (default: {DEFAULT_FAMILY_ID})",
    )
    parser.add_argument(
        "--history-dir",
        default="",
        help="override VERICALL_HISTORY_DIR, useful for staging and tests",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="remove only previous ai-demo records, then seed the fixed set",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report the operation without writing any records",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = seed_demo_history(
        family_id=args.family_id,
        history_dir=args.history_dir or None,
        reset=args.reset,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
