# -*- coding: utf-8 -*-
"""谛听 VeriCall 本地 HTTP 端到端基准。

对运行中的 ``/api/analyze`` 发真实 multipart 请求，记录冷启动和热启动
总响应时延，并单独保留服务端 ``elapsed_s``。结果只代表当前主机、当前
模型和当前网络；脚本不会把本机结果外推为真实手机弱网指标。
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import platform
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_latencies(values: list[float]) -> dict:
    if not values:
        return {
            "n": 0, "min_s": None, "mean_s": None, "p50_s": None,
            "p90_s": None, "p95_s": None, "p99_s": None, "max_s": None,
        }
    return {
        "n": len(values),
        "min_s": round(min(values), 3),
        "mean_s": round(statistics.mean(values), 3),
        "p50_s": round(percentile(values, 0.50) or 0.0, 3),
        "p90_s": round(percentile(values, 0.90) or 0.0, 3),
        "p95_s": round(percentile(values, 0.95) or 0.0, 3),
        "p99_s": round(percentile(values, 0.99) or 0.0, 3),
        "max_s": round(max(values), 3),
    }


def _json_request(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _multipart_body(file_path: Path | None, demo: str | None,
                    caller_number: str,
                    household: str = "") -> tuple[bytes, str]:
    boundary = f"----vericall-{uuid.uuid4().hex}"
    chunks: list[bytes] = []

    def field(name: str, value: str) -> None:
        chunks.extend([
            f"--{boundary}\r\n".encode("ascii"),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(
                "ascii"),
            str(value).encode("utf-8"),
            b"\r\n",
        ])

    if file_path is not None:
        content_type = (mimetypes.guess_type(file_path.name)[0]
                        or "application/octet-stream")
        chunks.extend([
            f"--{boundary}\r\n".encode("ascii"),
            (
                f'Content-Disposition: form-data; name="file"; '
                f'filename="{file_path.name}"\r\n'
            ).encode("utf-8"),
            f"Content-Type: {content_type}\r\n\r\n".encode("ascii"),
            file_path.read_bytes(),
            b"\r\n",
        ])
    if demo:
        field("demo", demo)
    if caller_number:
        field("caller_number", caller_number)
    if household:
        field("household", household)
    chunks.append(f"--{boundary}--\r\n".encode("ascii"))
    return b"".join(chunks), boundary


def post_analyze(base_url: str, file_path: Path | None, demo: str | None,
                 caller_number: str, timeout: float,
                 bearer_token: str = "",
                 household: str = "") -> tuple[float, dict]:
    body, boundary = _multipart_body(
        file_path, demo, caller_number, household)
    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
    }
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/analyze",
        data=body,
        method="POST",
        headers=headers,
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    latency = time.perf_counter() - started
    return latency, payload


def audio_info(file_path: Path | None) -> dict:
    if file_path is None:
        return {"path": None, "bytes": None, "duration_s": None}
    info = {
        "path": str(file_path),
        "bytes": file_path.stat().st_size,
        "duration_s": None,
    }
    try:
        import soundfile as sf
        meta = sf.info(str(file_path))
        info["duration_s"] = round(meta.duration, 3)
    except Exception:  # noqa: BLE001
        pass
    return info


def render_markdown(report: dict) -> str:
    lat = report["latency"]
    server = report["server_elapsed"]
    lines = [
        "# 谛听 VeriCall · 本地 HTTP 端到端基准",
        "",
        f"> 生成时间：{report['generated_at']}",
        f"> 目标：`{report['base_url']}`",
        f"> 主机：`{report['host']}` / Python `{report['python']}`",
        "",
        "## 模型状态",
        "",
        f"- 请求声学通道：`{report['status'].get('acoustic_requested')}`",
        f"- 实际声学通道：`{report['status'].get('acoustic_active')}`",
        f"- 回退原因：`{report['status'].get('acoustic_fallback_reason')}`",
        f"- 语义后端：`{report['status'].get('semantic_backend')}`",
        f"- 设备：`{report['status'].get('device')}`",
        "",
        "## 输入",
        "",
        f"- 音频：`{report['input']['path']}`",
        f"- 文件大小：{report['input']['bytes']} bytes",
        f"- 音频时长：{report['input']['duration_s']} s",
        f"- Demo：`{report['demo']}`",
        f"- 家庭编号：`{report.get('household') or '未指定'}`",
        f"- Bearer 鉴权：`{'已启用' if report.get('authenticated') else '未启用'}`",
        f"- 冷启动样本：{report['cold_s']} s",
        f"- 热启动样本：{lat['n']} 次（预热 {report['warmup']} 次不计入）",
        "",
        "## 总响应时延",
        "",
        "| 指标 | P50 | P90 | P95 | P99 | Mean | Min | Max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| 秒 | {lat['p50_s']} | {lat['p90_s']} | {lat['p95_s']} | "
            f"{lat['p99_s']} | {lat['mean_s']} | {lat['min_s']} | "
            f"{lat['max_s']} |"
        ),
        "",
        "## 服务端 `elapsed_s`",
        "",
        "| 指标 | P50 | P90 | P95 | P99 | Mean | Min | Max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| 秒 | {server['p50_s']} | {server['p90_s']} | "
            f"{server['p95_s']} | {server['p99_s']} | {server['mean_s']} | "
            f"{server['min_s']} | {server['max_s']} |"
        ),
        "",
        "## 口径边界",
        "",
        "- 总响应包含本机 multipart 上传、服务端排队、模型分析和 JSON 返回。",
        "- 本机 loopback 不代表手机弱网、运营商线路、跨地域公网或并发压力。",
        "- 冷启动只测一次，受操作系统缓存、GPU 初始化和模型文件位置影响。",
        "- 若需比赛证据，应同时记录硬件型号、模型哈希、音频集和运行命令。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="测量运行中谛听服务的 /api/analyze 端到端时延")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--audio", default="", help="待上传音频路径")
    parser.add_argument("--demo", default="",
                        help="使用内置 demo A/B/C（与 --audio 二选一）")
    parser.add_argument("--caller-number", default="")
    parser.add_argument(
        "--household", default="",
        help="试点家庭编号；通过 multipart household 字段写入检测历史")
    parser.add_argument(
        "--bearer-token", default="",
        help="远端鉴权服务使用；不会写入报告")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--out-dir", default=str(ROOT / "evaluation"))
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs 必须 >= 1")
    if args.warmup < 0:
        parser.error("--warmup 必须 >= 0")
    file_path = Path(args.audio).expanduser().resolve() if args.audio else None
    if file_path is not None and not file_path.is_file():
        parser.error(f"音频不存在: {file_path}")
    demo = args.demo.strip().upper() or None
    household = args.household.strip()
    bearer_token = args.bearer_token.strip()
    if file_path is not None and demo:
        parser.error("--audio 与 --demo 只能选择一个")
    if file_path is None and demo is None:
        demo = "A"

    base_url = args.base_url.rstrip("/")
    try:
        health = _json_request(f"{base_url}/healthz", args.timeout)
        status = _json_request(f"{base_url}/api/status", args.timeout)
    except Exception as exc:  # noqa: BLE001
        print(f"[基准] 服务不可达，未生成任何时延数字: {exc}",
              file=sys.stderr)
        return 2

    if not health.get("ok"):
        print(f"[基准] /healthz 未返回 ok: {health}", file=sys.stderr)
        return 2

    cold_s = None
    try:
        cold_s, cold_payload = post_analyze(
            base_url, file_path, demo, args.caller_number, args.timeout,
            bearer_token, household)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[基准] 首次分析失败，未生成时延表: {exc}",
            file=sys.stderr)
        print("提示：检查音频格式、模型文件、GPU/CPU 后端及 /api/status。",
              file=sys.stderr)
        return 2

    for _ in range(args.warmup):
        post_analyze(
            base_url, file_path, demo, args.caller_number, args.timeout,
            bearer_token, household)

    latencies: list[float] = []
    server_elapsed: list[float] = []
    verdicts: dict[str, int] = {}
    for _ in range(args.runs):
        latency, payload = post_analyze(
            base_url, file_path, demo, args.caller_number, args.timeout,
            bearer_token, household)
        latencies.append(latency)
        try:
            server_elapsed.append(float(payload.get("elapsed_s")))
        except (TypeError, ValueError):
            pass
        final = str(payload.get("final") or "unknown")
        verdicts[final] = verdicts.get(final, 0) + 1

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = {
        "generated_at": generated_at,
        "base_url": base_url,
        "host": platform.platform(),
        "python": platform.python_version(),
        "status": status,
        "input": audio_info(file_path),
        "demo": demo,
        "household": household or None,
        "authenticated": bool(bearer_token),
        "cold_s": round(cold_s, 3),
        "cold_final": cold_payload.get("final"),
        "warmup": args.warmup,
        "runs": args.runs,
        "latency": summarize_latencies(latencies),
        "server_elapsed": summarize_latencies(server_elapsed),
        "verdicts": verdicts,
        "scope": "local_http",
        "limits": [
            "loopback only",
            "no mobile radio network",
            "no carrier channel",
            "single-request sequential load",
        ],
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"e2e_latency_{stamp}.json"
    md_path = out_dir / f"e2e_latency_{stamp}.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"冷启动: {report['cold_s']} s")
    print(
        "热启动总响应: "
        f"P50={report['latency']['p50_s']}s "
        f"P95={report['latency']['p95_s']}s "
        f"P99={report['latency']['p99_s']}s"
    )
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
