# -*- coding: utf-8 -*-
"""
谛听 VeriCall — 演示 Web API（FastAPI）
====================================
复用 fusion.VeriCallPipeline，为前端单页应用（web/）提供接口。

接口：
    GET  /api/status    系统状态（AASIST devEER / 声纹库 / Ollama）
    POST /api/analyze   上传音频 或 demo 场景(A/B/C) -> 三通道裁决
    POST /api/enroll    家人声纹登记（上传 wav + 称谓）
    GET  /api/voices    已登记家人声纹列表
    GET  /api/history   历史检测记录
    /                   静态托管 web/index.html

运行：
    python src/api/server.py            # 默认 8000 端口
    VERICALL_PORT=9000 python src/api/server.py
    VERICALL_DEVICE=cpu  强制 CPU 推理（默认 cuda）
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent          # src/api
ROOT = HERE.parent.parent                        # 项目根
sys.path.insert(0, str(ROOT / "src"))

# 所有路径与设备配置统一在 src/paths.py（环境变量 / .env 覆盖）
from paths import (  # noqa: E402
    WEB_DIR, VOICE_DIR, UPLOAD_DIR, HIST_FILE,
    DEV_FLAC, EXP_DIR, OLLAMA_HOST, DEVICE, API_PORT, OFFLINE, ensure_data_dirs,
)

ensure_data_dirs()

# 内置演示场景（ASVspoof2019 LA dev，三场景实战复测同款）
DEMO_SCENARIOS = {
    "A": (str(DEV_FLAC / "LA_D_1105538.flac"), "家人本人来电（真声，音色匹配）"),
    "B": (str(DEV_FLAC / "LA_D_1002910.flac"), "AIGC 冒充家人（同音色伪造声）"),
    "C": (str(DEV_FLAC / "LA_D_1090286.flac"), "陌生人来电（真声，音色不匹配）"),
}
DEMO_FAMILY_ENROLL = str(DEV_FLAC / "LA_D_1047731.flac")  # 演示用「家人」登记声纹
DEMO_FAMILY_NAME = "家人"

from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="谛听 VeriCall API")
_lock = threading.Lock()
_pipe = None
_enrolled: set[str] = set()


def get_pipe():
    """懒加载管线；启动时优先用落盘声纹，缺失时回退扫 family_voice 重算。"""
    global _pipe
    if _pipe is None:
        from fusion.pipeline import VeriCallPipeline
        from fusion.acoustic_channel import AcousticChannel
        # 声学通道选择：VERICALL_ACOUSTIC=xlsr_cn 用中文域 XLS-R+LR（红队检出 97.9%），
        # xlsr_cn:wide 用宽覆盖（含 CFAD 声码器伪，CFAD 47→28%）；默认保持 AASIST（英文基线，向后兼容）。
        if os.environ.get("VERICALL_ACOUSTIC", "aasist").startswith("xlsr_cn"):
            from fusion.xlsr_cn_channel import XlsrCnChannel
            _scorer = os.environ.get("VERICALL_ACOUSTIC", "").split(":")[1] if ":" in os.environ.get("VERICALL_ACOUSTIC", "") else "v4"
            ac = XlsrCnChannel(scorer=_scorer, device=DEVICE)
            if not ac.load():
                print("[API] XLS-R 中文域声学通道加载失败，回退 AASIST")
                ac = AcousticChannel(device=DEVICE)
        else:
            ac = AcousticChannel(device=DEVICE)
        _pipe = VeriCallPipeline(acoustic=ac)
        if OFFLINE:
            # 离线模式：声纹通道用缓存判决，不依赖 funasr 重算
            print("[API] 离线降级模式：跳过声纹重算，使用落盘/缓存声纹")
            for name in _pipe.vp.profiles:
                _enrolled.add(name)
            return _pipe
        if VOICE_DIR.is_dir():
            for d in sorted(VOICE_DIR.iterdir()):
                if not d.is_dir():
                    continue
                if d.name in _pipe.vp.profiles:   # 已落盘，直接用
                    _enrolled.add(d.name)
                    continue
                wavs = sorted(d.glob("*.wav"), key=lambda p: p.stat().st_mtime)
                if wavs:
                    try:
                        _pipe.enroll(d.name, str(wavs[-1]))
                        _enrolled.add(d.name)
                        print(f"[API] 预登记声纹: {d.name} <- {wavs[-1].name}")
                    except Exception as e:  # noqa: BLE001
                        print(f"[API] 预登记失败 {d.name}: {e}")
    return _pipe


def _append_history(rec: dict):
    HIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HIST_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _model_status() -> dict:
    eer = None
    q = EXP_DIR / "model_quality.json"
    if q.is_file():
        try:
            eer = json.loads(q.read_text(encoding="utf-8")).get("dev_eer")
        except Exception:  # noqa: BLE001
            pass
    voices = []
    if VOICE_DIR.is_dir():
        for d in sorted(VOICE_DIR.iterdir()):
            if d.is_dir():
                voices.append({"name": d.name,
                               "samples": len(list(d.glob("*.wav")))})
    ollama = False
    try:
        import urllib.request
        with urllib.request.urlopen(f"{OLLAMA_HOST.rstrip('/')}/api/tags",
                                    timeout=1.5) as resp:
            ollama = resp.status == 200
    except Exception:  # noqa: BLE001
        pass
    return {
        "aasist_dev_eer": eer,
        "eval_eer": (json.loads(q.read_text(encoding="utf-8"))
                     .get("eval_eer_best_verified") if q.is_file() else None),
        "voices": voices,
        "ollama": ollama,
        "device": DEVICE,
    }


@app.get("/api/status")
def status():
    return _model_status()


@app.post("/api/analyze")
async def analyze(file: UploadFile | None = File(default=None),
                  demo: str | None = Form(default=None)):
    if not _lock.acquire(blocking=False):
        return JSONResponse({"error": "busy", "message": "上一次分析尚未结束，请稍候"},
                            status_code=503)
    try:
        pipe = get_pipe()
        source = "upload"
        if OFFLINE and not demo:
            return JSONResponse(
                {"error": "offline_demo_only",
                 "message": "离线降级模式仅支持 demo 场景 A/B/C，请用 demo 参数调用"},
                status_code=400)
        if demo:
            demo = demo.upper()
            if demo not in DEMO_SCENARIOS:
                return JSONResponse({"error": "bad_demo"}, status_code=400)
            if OFFLINE:
                # 离线降级：不依赖演示音频文件与声纹登记，直接走缓存通道
                audio_path = ""
                source = f"demo:{demo}"
                name = DEMO_SCENARIOS[demo][1]
            else:
                audio_path, title = DEMO_SCENARIOS[demo]
                if not Path(audio_path).is_file():
                    return JSONResponse(
                        {"error": "demo_missing",
                         "message": f"演示音频不存在: {audio_path}"}, status_code=404)
                # 演示场景需要「家人」声纹已登记（懒登记）
                if DEMO_FAMILY_NAME not in _enrolled and \
                        Path(DEMO_FAMILY_ENROLL).is_file():
                    pipe.enroll(DEMO_FAMILY_NAME, DEMO_FAMILY_ENROLL)
                    _enrolled.add(DEMO_FAMILY_NAME)
                source = f"demo:{demo}"
                name = title
        else:
            if file is None or not file.filename:
                return JSONResponse({"error": "no_file"}, status_code=400)
            suffix = Path(file.filename).suffix.lower() or ".wav"
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            audio_path = str(UPLOAD_DIR / f"{uuid.uuid4().hex[:10]}{suffix}")
            with open(audio_path, "wb") as f:
                f.write(await file.read())
            name = file.filename

        t0 = time.time()
        result = pipe.analyze(audio_path, scenario=demo if demo else None)
        elapsed = round(time.time() - t0, 1)

        rec = {
            "id": uuid.uuid4().hex[:10],
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "name": name,
            "source": source,
            "elapsed_s": elapsed,
            **result.to_dict(),
        }
        _append_history(rec)
        return rec
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": "analyze_failed", "message": str(e)},
                            status_code=500)
    finally:
        _lock.release()


@app.post("/api/enroll")
async def enroll(name: str = Form(...),
                 session: str = Form(default="putonghua"),
                 file: UploadFile = File(...)):
    name = name.strip()
    if not name:
        return JSONResponse({"error": "no_name"}, status_code=400)
    # 安全：家人称谓白名单（中文/字母/数字/下划线/连字符，1-20 位），
    # 防 `../../` 路径穿越写出 VOICE_DIR（安全竞赛作品自身不能有注入面）
    if not re.fullmatch(r"[\w\u4e00-\u9fff-]{1,20}", name):
        return JSONResponse(
            {"error": "bad_name",
             "message": "称谓仅支持中文/字母/数字/下划线/连字符，且 1-20 个字符"},
            status_code=400)
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,30}", session):
        session = "session"
    dest_dir = VOICE_DIR / name
    # 双保险：断言目标仍在 VOICE_DIR 内
    dest_dir = dest_dir.resolve()
    voice_root = VOICE_DIR.resolve()
    if not str(dest_dir).startswith(str(voice_root)):
        return JSONResponse({"error": "bad_path"}, status_code=400)
    dest_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = Path(file.filename or "rec.wav").suffix.lower() or ".wav"
    out = dest_dir / f"{session}_{ts}{suffix}"
    with open(out, "wb") as f:
        f.write(await file.read())

    if not _lock.acquire(blocking=False):
        return JSONResponse({"error": "busy", "message": "管线占用中，请稍候再登记"},
                            status_code=503)
    try:
        pipe = get_pipe()
        pipe.enroll(name, str(out))
        _enrolled.add(name)
        return {"ok": True, "name": name, "file": str(out),
                "voices": _model_status()["voices"]}
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": "enroll_failed", "message": str(e)},
                            status_code=500)
    finally:
        _lock.release()


@app.get("/api/voices")
def voices():
    return {"voices": _model_status()["voices"],
            "enrolled_in_memory": sorted(_enrolled)}


@app.get("/api/history")
def history(limit: int = Query(default=50, le=200)):
    recs = []
    if HIST_FILE.is_file():
        with open(HIST_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        recs.append(json.loads(line))
                    except Exception:  # noqa: BLE001
                        pass
    return {"records": list(reversed(recs))[:limit]}


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

# ------------------------------------------------------------------ #
# P1-1 流式实时接口（/ws/stream）：导入失败不阻断其余 REST 接口
# ------------------------------------------------------------------ #
try:
    from server.ws_api import register_stream_ws
    from server.scheduler import make_real_backends
    from server.stream_pipeline import StreamProcessor
    from fusion.fusion_orchestrator import FusionOrchestrator

    def _make_stream_processor():
        pipe = get_pipe()
        backends = make_real_backends(pipe, offline=OFFLINE)
        return StreamProcessor(backends=backends,
                               orchestrator=FusionOrchestrator(),
                               family_name=DEMO_FAMILY_NAME)

    register_stream_ws(app, _make_stream_processor)
    print("[API] 流式接口 /ws/stream 已启用")
except Exception as e:  # noqa: BLE001
    print(f"[API] 流式接口未启用: {e}")


if __name__ == "__main__":
    import uvicorn
    print(f"[API] 谛听 VeriCall 演示服务: http://localhost:{API_PORT}")
    uvicorn.run(app, host="127.0.0.1", port=API_PORT, log_level="warning")
