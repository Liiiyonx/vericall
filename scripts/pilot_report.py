# -*- coding: utf-8 -*-
"""社区试点统计报告生成器。

数据源：
  - ``data/history.jsonl`` + ``data/history/*.jsonl``：脚本自动合并旧文件
    与家庭分片，每条检测记录中的 ``household`` 用于按户聚合；
  - ``data/family_alert_events.jsonl``：子女确认告警时写入的时间戳；
  - 可选人工反馈 JSON：按记录 ID 标注真实结论。

报告只陈述输入数据中可计算的指标。空数据会生成 0 值报告，不填充
虚构用户、客户或误报率。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from api.history_store import iter_all_history  # noqa: E402

VALID_FINAL = {"allow", "caution", "block"}
VALID_HUMAN = {"allow", "caution", "block"}


def load_records(history: Path) -> list[dict]:
    """兼容旧路径，并自动合并同目录下的家庭历史分片。"""
    return list(iter_all_history(history))


def load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default


def normalize_feedback(raw: Any) -> dict[str, str]:
    """兼容 ``{id: final}`` 与 ``{"records": [{"id", "human_final"}]}``。"""
    if not isinstance(raw, dict):
        return {}
    source = raw.get("records") if isinstance(raw.get("records"), list) else raw
    if isinstance(source, list):
        result = {}
        for item in source:
            if not isinstance(item, dict):
                continue
            record_id = str(item.get("id") or "").strip()
            human_final = str(item.get("human_final") or "").strip()
            if record_id and human_final in VALID_HUMAN:
                result[record_id] = human_final
        return result
    return {
        str(record_id): str(human_final)
        for record_id, human_final in source.items()
        if str(human_final) in VALID_HUMAN
    }


def load_ack_events(path: Path) -> dict[str, float]:
    events: dict[str, float] = {}
    if not path.is_file():
        return events
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                alert_id = str(item.get("alert_id") or item.get("id") or "")
                ack_at = float(item.get("ack_at"))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if alert_id:
                events[alert_id] = ack_at
    return events


def parse_time(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).timestamp()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(records: list[dict],
              feedback: dict[str, str] | None = None,
              ack_events: dict[str, float] | None = None) -> dict:
    """按户聚合，并计算人工复核后的 FP/FN 与 ACK 时延。"""
    feedback = feedback or {}
    ack_events = ack_events or {}
    households: dict[str, dict] = defaultdict(
        lambda: {
            "total": 0, "block": 0, "caution": 0, "allow": 0,
            "risk_allow": 0, "labeled": 0, "fp": 0, "fn": 0,
            "daily": defaultdict(int), "examples": [],
        })
    overall = {
        "total": 0, "block": 0, "caution": 0, "allow": 0,
        "risk_allow": 0, "labeled": 0, "fp": 0, "fn": 0,
        "alert_created": 0, "acked": 0,
    }
    ack_latencies: list[float] = []

    for record in records:
        household = str(record.get("household") or "未标注")
        item = households[household]
        final = str(record.get("final") or "allow")
        if final not in VALID_FINAL:
            final = "allow"
        item["total"] += 1
        overall["total"] += 1
        item[final] += 1
        overall[final] += 1
        day = str(record.get("time") or "")[:10] or "未知"
        item["daily"][day] += 1

        score = record.get("score") or 0
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0
        if final == "allow" and score >= 0.4:
            item["risk_allow"] += 1
            overall["risk_allow"] += 1

        record_id = str(record.get("id") or "")
        human_final = feedback.get(record_id) or record.get("human_final")
        human_final = str(human_final or "")
        if human_final in VALID_HUMAN:
            item["labeled"] += 1
            overall["labeled"] += 1
            if final in ("block", "caution") and human_final == "allow":
                item["fp"] += 1
                overall["fp"] += 1
            if final == "allow" and human_final in ("block", "caution"):
                item["fn"] += 1
                overall["fn"] += 1

        if final in ("block", "caution"):
            overall["alert_created"] += 1
            if record_id in ack_events:
                created = parse_time(record.get("time"))
                if created is not None:
                    latency = ack_events[record_id] - created
                    if latency >= 0:
                        ack_latencies.append(latency)
                        overall["acked"] += 1

        if len(item["examples"]) < 5 and final != "allow":
            item["examples"].append({
                "time": record.get("time"),
                "name": record.get("name"),
                "final": final,
                "human_final": human_final,
                "score": score,
                "caller": record.get("caller_number") or "",
                "rationale": str(record.get("rationale") or "")[:100],
            })

    for item in households.values():
        item["daily"] = dict(item["daily"])

    latency = {
        "count": len(ack_latencies),
        "median_s": percentile(ack_latencies, 0.5),
        "p95_s": percentile(ack_latencies, 0.95),
        "max_s": max(ack_latencies) if ack_latencies else None,
    }
    return {
        "households": dict(households),
        "overall": overall,
        "ack_latency": latency,
    }


def _seconds(value: float | None) -> str:
    if value is None:
        return "—"
    if value >= 3600:
        return f"{value / 3600:.2f} h"
    if value >= 60:
        return f"{value / 60:.2f} min"
    return f"{value:.2f} s"


def render(summary: dict,
           notes: dict | None = None,
           source_label: str = "data/history.jsonl") -> str:
    notes = notes or {}
    households = summary["households"]
    overall = summary["overall"]
    ack = summary["ack_latency"]
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# 谛听 VeriCall · 社区试点周报",
        "",
        f"> 生成时间：{generated}",
        f"> 数据源：`{source_label}`",
        f"> 覆盖家庭：{len(households)} 户；样本：{overall['total']} 次判定",
        "> 本报告仅汇总输入数据；未开展的真实试点不会由脚本自动补齐。",
        "",
        "## 总览",
        "",
        "| 总判定 | 拦截 | 警惕 | 放行 | 可疑放行 | 已人工复核 | 误报 | 漏报 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {overall['total']} | {overall['block']} | "
            f"{overall['caution']} | {overall['allow']} | "
            f"{overall['risk_allow']} | {overall['labeled']} | "
            f"{overall['fp']} | {overall['fn']} |"
        ),
        "",
        "## 各户汇总",
        "",
        "| 家庭 | 总判定 | 拦截 | 警惕 | 放行 | 可疑放行 | 已复核 | 误报 | 漏报 | 备注 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for household, item in sorted(households.items()):
        lines.append(
            f"| {household} | {item['total']} | {item['block']} | "
            f"{item['caution']} | {item['allow']} | {item['risk_allow']} | "
            f"{item['labeled']} | {item['fp']} | {item['fn']} | "
            f"{notes.get(household, '')} |")

    lines += [
        "",
        "## 人工反馈指标",
        "",
        "- 误报：系统判定为 `block`/`caution`，人工复核确认真实结论为 `allow`。",
        "- 漏报：系统判定为 `allow`，人工复核确认真实结论为 `block`/`caution`。",
        f"- 已复核样本：{overall['labeled']} 条；误报：{overall['fp']} 条；"
        f"漏报：{overall['fn']} 条。",
        "",
        "## 告警响应时延",
        "",
        (
            f"- 告警：{overall['alert_created']} 条；已记录子女确认："
            f"{overall['acked']} 条。"
        ),
        (
            f"- 确认时延：中位数 {_seconds(ack['median_s'])}，"
            f"P95 {_seconds(ack['p95_s'])}，最大值 {_seconds(ack['max_s'])}。"
        ),
        "- 时延从历史记录 `time` 计算到 `family_alert_events.jsonl` 的 `ack_at`。",
        "",
        "## 按日分布",
        "",
        "| 日期 | 次数 |",
        "|---|---:|",
    ]
    days: dict[str, int] = defaultdict(int)
    for item in households.values():
        for day, count in item["daily"].items():
            days[day] += count
    for day in sorted(days):
        lines.append(f"| {day} | {days[day]} |")

    lines += ["", "## 示例记录（拦截/警惕）", ""]
    for household, item in sorted(households.items()):
        if not item["examples"]:
            continue
        lines.append(f"### {household}")
        for example in item["examples"]:
            human = example["human_final"] or "未复核"
            lines.append(
                f"- {example['time']} [{example['final']} / {human}] "
                f"{example['name']} score={example['score']} "
                f"号码={example['caller'] or '—'} — {example['rationale']}")
    lines += [
        "",
        "## 数据口径与伦理边界",
        "",
        "- `household` 只能使用试点编号，不写姓名、手机号、详细住址等直接标识。",
        "- 人工反馈应由获得本人授权的复核人填写；没有反馈时 FP/FN 均不推断。",
        "- ACK 时间只证明子女端查看了本机告警，不等价于真实风险处置完成。",
        "- 本机 HTTP 结果不能替代运营商线路、真实手机弱网和真实诈骗团伙测试。",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="生成社区试点周报")
    parser.add_argument(
        "--history", default=str(ROOT / "data" / "history.jsonl"))
    parser.add_argument(
        "--out",
        default=str(ROOT / "docs"
                    / f"试点报告_{datetime.now().strftime('%Y-%m-%d')}.md"))
    parser.add_argument("--households", type=int, default=0,
                        help="只保留前 N 户（按家庭标识排序）")
    parser.add_argument("--notes", default="", help="按家庭备注 JSON 文件")
    parser.add_argument("--feedback", default="",
                        help="人工反馈 JSON；支持 id->结论或 records 列表")
    parser.add_argument(
        "--ack-events",
        default=str(ROOT / "data" / "family_alert_events.jsonl"),
        help="告警确认事件 JSONL")
    args = parser.parse_args()

    history_path = Path(args.history)
    records = load_records(history_path)
    feedback = normalize_feedback(
        load_json(Path(args.feedback), {}) if args.feedback else {})
    ack_events = load_ack_events(Path(args.ack_events))
    summary = summarize(records, feedback, ack_events)
    households = summary["households"]
    if args.households > 0:
        keep = set(sorted(households)[:args.households])
        # 过滤后重新计算总览与时延，避免报表首行和明细口径不一致。
        summary = summarize(
            [record for record in records
             if str(record.get("household") or "未标注") in keep],
            feedback, ack_events)
    notes = load_json(Path(args.notes), {}) if args.notes else {}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        render(
            summary,
            notes,
            source_label=(
                f"{history_path} + {history_path.parent / 'history'}/*.jsonl"
            ),
        ),
        encoding="utf-8")
    overall = summary["overall"]
    print(f"已生成试点周报: {out}")
    print(f"  家庭数={len(summary['households'])}  "
          f"总判定={overall['total']}  已复核={overall['labeled']}")
    if overall["total"] == 0:
        print("  提示：输入没有检测记录；报告中没有填充任何模拟试点数据。")


if __name__ == "__main__":
    main()
