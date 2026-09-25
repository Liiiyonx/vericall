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
    /                   统一入口 web/login.html
    /index.html         鉴真演示台

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
    DEV_FLAC, EXP_DIR, OLLAMA_HOST, DEVICE, API_PORT, API_HOST,
    DEFAULT_FAMILY_ID, acoustic_backend, ensure_data_dirs, is_offline,
    semantic_cloud_config,
)
from api.history_store import append_history, iter_history, next_id  # noqa: E402
from private_fs import ensure_private_dir  # noqa: E402

ensure_data_dirs()

# 内置演示场景（ASVspoof2019 LA dev，三场景实战复测同款）
DEMO_SCENARIOS = {
    "A": (str(DEV_FLAC / "LA_D_1105538.flac"), "家人本人来电（真声，音色匹配）"),
    "B": (str(DEV_FLAC / "LA_D_1002910.flac"), "AIGC 冒充家人（同音色伪造声）"),
    "C": (str(DEV_FLAC / "LA_D_1090286.flac"), "陌生人来电（真声，音色不匹配）"),
}
DEMO_FAMILY_ENROLL = str(DEV_FLAC / "LA_D_1047731.flac")  # 演示用「家人」登记声纹
DEMO_FAMILY_NAME = "家人"

from fastapi import (
    FastAPI, File, Form, Header, Query, Request, UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

_ENABLE_DOCS = (os.environ.get("VERICALL_ENABLE_DOCS", "0") or "0").strip().lower() in (
    "1", "true", "yes", "on")
app = FastAPI(
    title="谛听 VeriCall API",
    docs_url="/docs" if _ENABLE_DOCS else None,
    redoc_url="/redoc" if _ENABLE_DOCS else None,
    openapi_url="/openapi.json" if _ENABLE_DOCS else None,
)
_lock = threading.Lock()
_pipe = None
_enrolled: set[str] = set()
_ALLOWED_UPLOAD_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".webm"}
_UPLOAD_CHUNK_BYTES = 1024 * 1024
_acoustic_state = {
    "requested": acoustic_backend(),
    "active": None,
    "fallback_reason": None,
}


@app.middleware("http")
async def _security_headers(request: Request, call_next):
    if (request.url.path.startswith("/api/")
            and _content_length_exceeds_upload_limit(request)):
        response = JSONResponse(
            {"error": "request_too_large",
             "message": "请求体超过服务器允许的上限"},
            status_code=413,
        )
    else:
        response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; base-uri 'self'; object-src 'none'; "
        "frame-ancestors 'none'; form-action 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; media-src 'self' data: blob:; "
        "connect-src 'self'; font-src 'self' data:")
    response.headers.setdefault(
        "Permissions-Policy",
        "camera=(), geolocation=(), microphone=(self)")
    response.headers.setdefault(
        "Cross-Origin-Resource-Policy", "same-origin")
    response.headers.setdefault(
        "Cross-Origin-Opener-Policy", "same-origin")
    response.headers.setdefault(
        "X-Permitted-Cross-Domain-Policies", "none")
    if request.url.path.startswith("/api/"):
        response.headers.setdefault("Cache-Control", "no-store")
    return response


class UploadValidationError(Exception):
    """上传文件校验失败，由接口转换成稳定的 JSON 错误。"""

    def __init__(self, status_code: int, error: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.error = error
        self.message = message


def _max_upload_bytes() -> int:
    """读取上传上限；非法值回退到 25MB，避免配置错误放开无上限上传。"""
    raw = os.environ.get("VERICALL_MAX_UPLOAD_MB", "25")
    try:
        mb = int(raw)
    except (TypeError, ValueError):
        mb = 25
    if mb <= 0:
        mb = 25
    return mb * 1024 * 1024


def _content_length_exceeds_upload_limit(request: Request) -> bool:
    """在 multipart 解析前拒绝声明为超大体积的 API 请求。

    路由内仍会按实际读取字节再次校验，此处只负责尽早挡住常见
    ``Content-Length`` 攻击，避免先落盘再拒绝。
    """
    raw = request.headers.get("content-length")
    if not raw:
        return False
    try:
        declared = int(raw)
    except (TypeError, ValueError):
        return False
    # multipart 边界、字段名和少量表单字段需要额外空间。
    return declared > _max_upload_bytes() + 2 * 1024 * 1024


def _valid_audio_signature(header: bytes, suffix: str) -> bool:
    """轻量魔数校验，拒绝仅伪造扩展名的非音频内容。"""
    if suffix == ".wav":
        return len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WAVE"
    if suffix == ".flac":
        return header.startswith(b"fLaC")
    if suffix == ".mp3":
        return header.startswith(b"ID3") or (
            len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0)
    if suffix == ".m4a":
        return len(header) >= 12 and header[4:8] == b"ftyp"
    if suffix == ".ogg":
        return header.startswith(b"OggS")
    if suffix == ".webm":
        return header.startswith(b"\x1a\x45\xdf\xa3")
    return False


async def _save_upload(file: UploadFile, *, prefix: str,
                       dest_dir: Path | None = None) -> Path:
    """分块保存上传文件，拒绝未知后缀和超限内容，失败时清理残留。"""
    default_name = "rec.wav"
    filename = file.filename or default_name
    suffix = Path(filename).suffix.lower() or Path(default_name).suffix.lower()
    if suffix not in _ALLOWED_UPLOAD_SUFFIXES:
        allowed = "、".join(sorted(_ALLOWED_UPLOAD_SUFFIXES))
        raise UploadValidationError(
            400, "unsupported_file_type",
            f"仅支持 {allowed} 格式的音频文件")

    target_dir = dest_dir or UPLOAD_DIR
    ensure_private_dir(target_dir)
    path = target_dir / f"{prefix}_{uuid.uuid4().hex[:10]}{suffix}"
    total = 0
    header = b""
    try:
        fd = os.open(
            path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            while True:
                chunk = await file.read(_UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > _max_upload_bytes():
                    raise UploadValidationError(
                        413, "file_too_large",
                        f"音频文件不能超过 {_max_upload_bytes() // (1024 * 1024)}MB")
                f.write(chunk)
                if len(header) < 64:
                    header = (header + chunk)[:64]
        if total == 0:
            raise UploadValidationError(400, "empty_file", "上传的音频文件为空")
        if not _valid_audio_signature(header, suffix):
            raise UploadValidationError(
                400, "invalid_audio_content",
                "文件内容不是有效的音频格式，请重新录音或转换后再上传")
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def _parse_acoustic_backend(requested: str) -> tuple[str, str]:
    """返回 ``(backend, scorer)``；未知后端交给 AASIST 安全回退。"""
    value = (requested or "").strip()
    if value == "aasist":
        return "aasist", ""
    if value.startswith("xlsr_cn"):
        parts = value.split(":", 1)
        scorer = parts[1] if len(parts) == 2 and parts[1] else "wide"
        if scorer not in ("wide", "v4"):
            raise ValueError(f"unsupported XLS-R scorer: {scorer}")
        return "xlsr_cn", scorer
    raise ValueError(f"unsupported acoustic backend: {value}")


def _load_acoustic_channel(requested: str | None = None):
    """按配置创建声学通道，并记录实际后端与回退原因。"""
    from fusion.acoustic_channel import AcousticChannel

    requested = (requested or acoustic_backend()).strip()
    _acoustic_state.update(
        requested=requested, active=None, fallback_reason=None)
    try:
        backend, scorer = _parse_acoustic_backend(requested)
    except ValueError as exc:
        _acoustic_state.update(active="aasist",
                               fallback_reason=str(exc))
        return AcousticChannel(device=DEVICE)

    if backend == "aasist":
        _acoustic_state["active"] = "aasist"
        return AcousticChannel(device=DEVICE)

    try:
        from fusion.xlsr_cn_channel import XlsrCnChannel
        ac = XlsrCnChannel(scorer=scorer, device=DEVICE)
        if ac.load():
            _acoustic_state["active"] = f"xlsr_cn:{scorer}"
            return ac
        reason = "xlsr_load_failed"
    except Exception as exc:  # noqa: BLE001
        reason = f"xlsr_init_failed:{type(exc).__name__}"
    print(f"[API] XLS-R 中文域声学通道加载失败（{reason}），回退 AASIST")
    _acoustic_state.update(active="aasist", fallback_reason=reason)
    return AcousticChannel(device=DEVICE)


def get_pipe():
    """懒加载管线；启动时优先用落盘声纹，缺失时回退扫 family_voice 重算。"""
    global _pipe
    if _pipe is None:
        from fusion.pipeline import VeriCallPipeline
        ac = _load_acoustic_channel()
        _pipe = VeriCallPipeline(acoustic=ac)
        if is_offline():
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


def _allowed_voice_names(pipe, family_id: str) -> list[str]:
    """仅返回当前家庭拥有的声纹名称，供融合管线做家庭内比对。"""
    from api.family_api import voice_owner

    profiles = getattr(getattr(pipe, "vp", None), "profiles", {}) or {}
    return sorted(
        str(name) for name in profiles
        if voice_owner(str(name)) == family_id)


def _model_status(family_id: str | None = None) -> dict:
    from fusion.acoustic_channel import _resolve_weight_details

    _, _, weight_meta = _resolve_weight_details()
    eer = weight_meta.get("dev_eer")
    eval_eer = None
    q = EXP_DIR / "model_quality.json"
    if q.is_file():
        try:
            eval_eer = json.loads(q.read_text(encoding="utf-8")).get(
                "eval_eer_best_verified")
        except Exception:  # noqa: BLE001
            pass
    from api.family_api import voice_owner

    voices = []
    if VOICE_DIR.is_dir():
        for d in sorted(VOICE_DIR.iterdir()):
            if d.is_dir() and (family_id is None
                               or voice_owner(d.name) == family_id):
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
    # 2026-09-06：语义通道 LLM 后端（cloud=DeepSeek 默认 / ollama 显式 / rule 兜底）
    from paths import SEMANTIC_LLM_BACKEND
    _sem = SEMANTIC_LLM_BACKEND
    _, cloud_key, _ = semantic_cloud_config()
    if is_offline():
        sem_backend = "rule"
    elif _sem == "ollama":
        sem_backend = "ollama"
    else:
        sem_backend = "cloud" if cloud_key else "rule"
    active = _acoustic_state.get("active") or acoustic_backend()
    return {
        "aasist_dev_eer": eer,
        "eval_eer": eval_eer,
        "aasist_weight": {
            **weight_meta,
            "weight_sha256": (
                str(weight_meta.get("weight_sha256") or "")[:12] or None),
        },
        "voices": voices,
        "voice_count": len(voices),
        "ollama": ollama,
        "semantic_backend": sem_backend,
        "device": DEVICE,
        "acoustic": active,
        "acoustic_requested": _acoustic_state.get("requested")
                              or acoustic_backend(),
        "acoustic_active": active,
        "acoustic_fallback_reason": _acoustic_state.get("fallback_reason"),
    }


@app.get("/api/status")
def status(request: Request,
           authorization: str | None = Header(default=None)):
    from api.family_api import request_family_scope

    family_id, access_mode = request_family_scope(
        request, authorization, required=False, allow_elder_key=True)
    payload = _model_status(family_id)
    if family_id is None:
        payload["voices"] = []
        payload["voice_count"] = 0
        payload["family_scope"] = "public"
    else:
        payload["family_scope"] = family_id
    payload["access_mode"] = access_mode
    return payload


@app.get("/healthz")
def healthz():
    """仅检查进程存活，不触发模型加载，供容器与反向代理探活。"""
    return {"ok": True, "service": "vericall", "time": int(time.time())}


@app.post("/api/analyze")
async def analyze(file: UploadFile | None = File(default=None),
                  demo: str | None = Form(default=None),
                  caller_number: str | None = Form(default=None),
                  household: str | None = Form(default=None),
                  request: Request = None,
                  authorization: str | None = Header(default=None)):
    from api.family_api import request_family_scope

    family_id, _ = request_family_scope(
        request, authorization, required=True, allow_elder_key=True)
    household_value = (household or "").strip()
    if household_value and not re.fullmatch(
            r"[\w.-]{1,40}", household_value, flags=re.UNICODE):
        return JSONResponse(
            {"error": "bad_household",
             "message": "household 仅支持 1-40 位字母、数字、下划线、点或连字符"},
            status_code=400)
    audio_path = ""
    uploaded_path: Path | None = None
    if not _lock.acquire(blocking=False):
        return JSONResponse({"error": "busy", "message": "上一次分析尚未结束，请稍候"},
                            status_code=503)
    pipe = None
    try:
        source = "upload"
        if is_offline() and not demo:
            return JSONResponse(
                {"error": "offline_demo_only",
                 "message": "离线降级模式仅支持 demo 场景 A/B/C，请用 demo 参数调用"},
                status_code=400)
        if demo:
            demo = demo.upper()
            if demo not in DEMO_SCENARIOS:
                return JSONResponse({"error": "bad_demo"}, status_code=400)
            if is_offline():
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
                pipe = get_pipe()
                if DEMO_FAMILY_NAME not in _enrolled and \
                        Path(DEMO_FAMILY_ENROLL).is_file():
                    pipe.enroll(DEMO_FAMILY_NAME, DEMO_FAMILY_ENROLL)
                    _enrolled.add(DEMO_FAMILY_NAME)
                source = f"demo:{demo}"
                name = title
        else:
            if file is None or not file.filename:
                return JSONResponse({"error": "no_file"}, status_code=400)
            try:
                uploaded_path = await _save_upload(file, prefix="audio")
            except UploadValidationError as e:
                return JSONResponse(
                    {"error": e.error, "message": e.message},
                    status_code=e.status_code)
            audio_path = str(uploaded_path)
            name = file.filename

        t0 = time.time()
        if pipe is None:
            pipe = get_pipe()
        allowed_names = _allowed_voice_names(pipe, family_id)
        result = pipe.analyze(audio_path, scenario=demo if demo else None,
                              caller_number=caller_number,
                              allowed_names=allowed_names,
                              family_id=family_id)
        elapsed = round(time.time() - t0, 1)

        rec = {
            "id": next_id(),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "name": name,
            "source": source,
            "family_id": family_id,
            "household": household_value,
            "caller_number": str(caller_number or ""),
            "elapsed_s": elapsed,
            **result.to_dict(),
        }
        append_history(rec, HIST_FILE)
        return rec
    except Exception as e:  # noqa: BLE001
        print(f"[API] analyze failed: {type(e).__name__}: {e}")
        return JSONResponse(
            {"error": "analyze_failed",
             "message": "分析失败，请稍后重试"},
                            status_code=500)
    finally:
        if uploaded_path is not None:
            uploaded_path.unlink(missing_ok=True)
        _lock.release()


@app.post("/api/enroll")
async def enroll(name: str = Form(...),
                 session: str = Form(default="putonghua"),
                 file: UploadFile = File(...),
                 request: Request = None,
                 authorization: str | None = Header(default=None)):
    from api.family_api import claim_voice_owner, request_family_scope

    family_id, _ = request_family_scope(request, authorization, required=True)
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
    if dest_dir != voice_root and voice_root not in dest_dir.parents:
        return JSONResponse({"error": "bad_path"}, status_code=400)
    if not claim_voice_owner(name, family_id):
        return JSONResponse(
            {"error": "voice_name_conflict",
             "message": "该家人称谓已被其他家庭使用，请更换名称"},
            status_code=409)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        out = await _save_upload(
            file, prefix=f"{session}_{ts}", dest_dir=dest_dir)
    except UploadValidationError as e:
        return JSONResponse(
            {"error": e.error, "message": e.message},
            status_code=e.status_code)

    if not _lock.acquire(blocking=False):
        out.unlink(missing_ok=True)
        return JSONResponse({"error": "busy", "message": "管线占用中，请稍候再登记"},
                            status_code=503)
    keep_upload = False
    try:
        pipe = get_pipe()
        pipe.enroll(name, str(out))
        _enrolled.add(name)
        keep_upload = True
        return {"ok": True, "name": name, "file": out.name,
                "family_id": family_id,
                "voices": _model_status(family_id)["voices"]}
    except Exception as e:  # noqa: BLE001
        print(f"[API] enroll failed: {type(e).__name__}: {e}")
        return JSONResponse(
            {"error": "enroll_failed",
             "message": "声纹登记失败，请检查录音质量后重试"},
                            status_code=500)
    finally:
        if not keep_upload:
            out.unlink(missing_ok=True)
        _lock.release()


@app.get("/api/voices")
def voices(request: Request,
           authorization: str | None = Header(default=None)):
    from api.family_api import request_family_scope, voice_owner

    family_id, access_mode = request_family_scope(
        request, authorization, required=True)
    visible = _model_status(family_id)["voices"]
    return {"family_id": family_id, "access_mode": access_mode,
            "voices": visible,
            "enrolled_in_memory": sorted(
                name for name in _enrolled if voice_owner(name) == family_id)}


@app.get("/api/history")
def history(request: Request,
            authorization: str | None = Header(default=None),
            limit: int = Query(default=50, ge=1, le=200)):
    from api.family_api import request_family_scope

    family_id, access_mode = request_family_scope(
        request, authorization, required=True)
    recs = list(iter_history(
        HIST_FILE, family_id=family_id,
        default_family_id=DEFAULT_FAMILY_ID))
    return {"family_id": family_id, "access_mode": access_mode,
            "records": list(reversed(recs))[:limit]}


# ------------------------------------------------------------------ #
# 挑战应答（B4-① 防回放）：数字复核端到端闭环（2026-09-11）
#   begin  生成 4 位随机数并显示/播报给对方 -> 对方跟读录音 -> verify
#   verify 双校验：ASR 跟读数字 == 挑战串，且跟读段与已登记家人声纹相似度 >= VOICE_THR
# 挑战码仅存内存、60s 过期、一次性使用；离线降级模式（无 ASR/声纹模型）返回明确错误。
# ------------------------------------------------------------------ #
_challenges: dict[str, dict] = {}   # code -> {digits, expires, family_id}
_CHALLENGE_TTL_S = 60
_CHALLENGE_MAX_ACTIVE = 512
_CHALLENGE_MAX_PER_FAMILY = 32
_challenges_lock = threading.Lock()


@app.post("/api/challenge/begin")
def challenge_begin(request: Request,
                    authorization: str | None = Header(default=None)):
    from api.family_api import request_family_scope
    from fusion.challenge_response import gen_challenge

    family_id, _ = request_family_scope(
        request, authorization, required=True)
    code = uuid.uuid4().hex[:8]
    digits = gen_challenge()
    with _challenges_lock:
        now = time.time()
        for key, item in list(_challenges.items()):
            if float(item.get("expires", 0)) <= now:
                _challenges.pop(key, None)
        family_active = sum(
            1 for item in _challenges.values()
            if item.get("family_id") == family_id)
        if (len(_challenges) >= _CHALLENGE_MAX_ACTIVE
                or family_active >= _CHALLENGE_MAX_PER_FAMILY):
            return JSONResponse(
                {"error": "challenge_limit",
                 "message": "未完成的数字复核过多，请稍后再试"},
                status_code=429,
                headers={"Retry-After": "1"})
        _challenges[code] = {
            "digits": digits,
            "expires": now + _CHALLENGE_TTL_S,
            "family_id": family_id,
        }
    return {"ok": True, "code": code, "digits": digits, "ttl_s": _CHALLENGE_TTL_S}


def _claim_challenge(code: str, family_id: str) -> str | None:
    """原子领取挑战；跨家庭请求不能消费或读取其他家庭的挑战。"""
    now = time.time()
    with _challenges_lock:
        for key, item in list(_challenges.items()):
            if float(item.get("expires", 0)) <= now:
                _challenges.pop(key, None)
        entry = _challenges.get(code)
        if not entry or entry.get("family_id") != family_id:
            return None
        _challenges.pop(code, None)
        return str(entry.get("digits") or "")


@app.post("/api/challenge/verify")
async def challenge_verify(request: Request,
                           code: str = Form(...),
                           file: UploadFile = File(...),
                           authorization: str | None = Header(default=None)):
    from api.family_api import request_family_scope
    from fusion.challenge_response import assess

    family_id, _ = request_family_scope(
        request, authorization, required=True)
    if not re.fullmatch(r"[0-9a-f]{8}", code or ""):
        return JSONResponse({"error": "expired",
                             "message": "挑战已过期，请重新发起"}, status_code=400)
    digits = _claim_challenge(code, family_id)
    if digits is None:
        return JSONResponse({"error": "expired",
                             "message": "挑战已过期，请重新发起"}, status_code=400)
    if is_offline():
        return JSONResponse({"error": "offline",
                             "message": "离线降级模式无 ASR/声纹模型，数字复核不可用（可切在线模式）"},
                            status_code=400)
    try:
        upload_path = await _save_upload(file, prefix="chal")
    except UploadValidationError as e:
        return JSONResponse(
            {"error": e.error, "message": e.message},
            status_code=e.status_code)
    path = str(upload_path)
    try:
        try:
            pipe = get_pipe()
            transcript = pipe.sem.asr.transcribe(path)
        except Exception as e:  # noqa: BLE001
            print(f"[API] challenge ASR failed: {type(e).__name__}: {e}")
            return JSONResponse({"error": "asr_failed",
                                 "message": "转写失败，请确认在线语音模型已就绪"},
                                status_code=500)
        allowed_names = _allowed_voice_names(pipe, family_id)
        best_name, sim, vp_err = pipe.vp.similarity(
            path, allowed_names=allowed_names)
        verdict = assess(digits, transcript, sim if vp_err is None else 0.0)
        return {"ok": True,
                "verdict": verdict,
                "transcript": transcript[:200],
                "best_name": best_name or "",
                "similarity": round(sim, 3),
                "voiceprint_error": bool(vp_err)}
    finally:
        upload_path.unlink(missing_ok=True)


# ------------------------------------------------------------------ #
# 家庭守护模块（子女端登录 / 风险告警 / 三方通话），挂载须在 StaticFiles 之前
# ------------------------------------------------------------------ #
try:
    from api.family_api import router as family_router
    app.include_router(family_router)
    print("[API] 家庭守护模块已启用（/api/family/* /api/call/* /ws/call/*）")
except Exception as e:  # noqa: BLE001
    print(f"[API] 家庭守护模块未启用: {e}")


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "login.html")

# ------------------------------------------------------------------ #
# P1-1 流式实时接口（/ws/stream）：导入失败不阻断其余 REST 接口
# ------------------------------------------------------------------ #
try:
    from server.ws_api import register_stream_ws
    from server.scheduler import make_real_backends
    from server.stream_pipeline import StreamProcessor
    from fusion.fusion_orchestrator import FusionOrchestrator

    def _make_stream_processor(sample_rate: int = 16000,
                               family_id: str = DEFAULT_FAMILY_ID):
        pipe = get_pipe()
        allowed_names = _allowed_voice_names(pipe, family_id)
        backends = make_real_backends(
            pipe, offline=is_offline(), sr=sample_rate,
            allowed_names=allowed_names)
        family_name = allowed_names[0] if len(allowed_names) == 1 else DEMO_FAMILY_NAME
        try:
            return StreamProcessor(backends=backends,
                                   orchestrator=FusionOrchestrator(),
                                   sample_rate=sample_rate,
                                   family_name=family_name)
        except Exception:
            cleanup = backends.get("_cleanup")
            if callable(cleanup):
                cleanup()
            raise

    register_stream_ws(app, _make_stream_processor)
    print("[API] 流式接口 /ws/stream 已启用")
except Exception as e:  # noqa: BLE001
    print(f"[API] 流式接口未启用: {e}")

# StaticFiles 也匹配 websocket scope，必须最后注册，避免吞掉 /ws/stream。
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    print(f"[API] 谛听 VeriCall 演示服务: http://{API_HOST}:{API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT, log_level="warning")
