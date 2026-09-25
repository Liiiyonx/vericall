# -*- coding: utf-8 -*-
"""谛听 VeriCall 启动预检：只报告状态，不改写用户的运行模式。"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paths import (  # noqa: E402
    AASIST_EXP,
    DEVICE,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    SENSEVOICE_DIR,
    SEMANTIC_LLM_BACKEND,
    XLSR_MODEL_DIR,
    is_offline,
    semantic_cloud_config,
)


def _endpoint_reachable(url: str, timeout: float = 1.5) -> bool:
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        if not host:
            return False
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _ollama_reachable(timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(
                f"{OLLAMA_HOST.rstrip('/')}/api/tags", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:  # noqa: BLE001
        return False


def _device_info() -> dict:
    info = {"configured": DEVICE, "torch": False, "cuda": False, "name": ""}
    try:
        import torch
        info["torch"] = True
        info["cuda"] = bool(torch.cuda.is_available())
        if info["cuda"]:
            info["name"] = str(torch.cuda.get_device_name(0))
    except Exception:  # noqa: BLE001
        pass
    return info


def _aasist_info() -> dict:
    try:
        from fusion.acoustic_channel import _resolve_weight_details
        _, _, meta = _resolve_weight_details()
        return {
            "ok": bool(meta.get("weight_path")),
            "experiment": meta.get("experiment") or AASIST_EXP,
            "weight": Path(meta["weight_path"]).name
            if meta.get("weight_path") else "",
            "selection_reason": meta.get("selection_reason"),
            "dev_eer": meta.get("dev_eer"),
            "sha256_prefix": str(meta.get("weight_sha256") or "")[:12] or None,
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": type(exc).__name__}


def _xlsr_info() -> dict:
    scorer = ROOT / "data" / "redteam" / "factory" / "cn_lr_scorer_wide.pkl"
    model_files = ("config.json", "pytorch_model.bin")
    missing = [name for name in model_files
               if not (XLSR_MODEL_DIR / name).is_file()]
    return {
        "ok": XLSR_MODEL_DIR.is_dir() and not missing and scorer.is_file(),
        "model_dir": str(XLSR_MODEL_DIR),
        "missing": missing,
        "scorer": scorer.is_file(),
    }


def collect() -> dict:
    base, key, model = semantic_cloud_config()
    offline = is_offline()
    ollama_ok = _ollama_reachable()
    cloud_reachable = _endpoint_reachable(base)
    if offline:
        mode = "offline"
    elif SEMANTIC_LLM_BACKEND == "ollama" and ollama_ok:
        mode = "online_ollama"
    elif SEMANTIC_LLM_BACKEND != "ollama" and key:
        mode = "online_cloud" if cloud_reachable else "online_cloud_unverified"
    else:
        mode = "online_rule_fallback"

    sensevoice = {
        "ok": SENSEVOICE_DIR.is_dir(),
        "path": str(SENSEVOICE_DIR),
    }
    return {
        "mode": mode,
        "offline_override": offline,
        "semantic": {
            "backend": SEMANTIC_LLM_BACKEND,
            "cloud_configured": bool(key),
            "cloud_endpoint_reachable": cloud_reachable,
            "cloud_host": urlsplit(base).hostname or "",
            "cloud_model": model,
            "ollama_configured": f"{OLLAMA_HOST} @ {OLLAMA_MODEL}",
            "ollama_reachable": ollama_ok,
        },
        "device": _device_info(),
        "sensevoice": sensevoice,
        "aasist": _aasist_info(),
        "xlsr": _xlsr_info(),
    }


def _print_human(info: dict) -> None:
    labels = {
        "offline": "离线演示（显式 VERICALL_OFFLINE=1）",
        "online_cloud": "在线（云端语义，端点可达）",
        "online_cloud_unverified": "在线（云端语义，端点暂不可达）",
        "online_ollama": "在线（Ollama 语义）",
        "online_rule_fallback": "在线（语义规则兜底）",
    }
    print("=" * 52)
    print("谛听 VeriCall · 启动预检")
    print(f"运行模式 : {labels.get(info['mode'], info['mode'])}")
    sem = info["semantic"]
    print(f"语义后端 : {sem['backend']} | 云密钥 {'已配置' if sem['cloud_configured'] else '未配置'}"
          f" | 云端 {'可达' if sem['cloud_endpoint_reachable'] else '不可达'}")
    print(f"Ollama   : {'可达' if sem['ollama_reachable'] else '不可达'} | {sem['ollama_configured']}")
    dev = info["device"]
    print(f"推理设备 : {dev['configured']} | torch={'有' if dev['torch'] else '无'}"
          f" | CUDA={'可用' if dev['cuda'] else '不可用'}"
          + (f" | {dev['name']}" if dev["name"] else ""))
    sv = info["sensevoice"]
    print(f"SenseVoice: {'就绪' if sv['ok'] else '缺失'} | {sv['path']}")
    aa = info["aasist"]
    if aa.get("ok"):
        print(f"AASIST   : 就绪 | {aa['experiment']} | {aa['weight']}"
              f" | devEER={aa['dev_eer']}% | sha256={aa['sha256_prefix']}")
    else:
        print(f"AASIST   : 未就绪 | {aa.get('error', '未找到权重')}")
    xl = info["xlsr"]
    print(f"XLS-R    : {'就绪' if xl['ok'] else '缺失/不完整'} | {xl['model_dir']}")
    if info["mode"] == "online_rule_fallback":
        print("提示     : 云端密钥和 Ollama 均不可用，将通过规则评分兜底；"
              "如需离线缓存演示，请手动设置 VERICALL_OFFLINE=1。")
    print("=" * 52)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    info = collect()
    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
    else:
        _print_human(info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
