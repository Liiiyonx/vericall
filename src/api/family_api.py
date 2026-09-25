# -*- coding: utf-8 -*-
"""
谛听 VeriCall — 家庭守护模块（子女端账号 / 风险告警 / 三方通话）
================================================================
新增能力（2026-09-08）：

1. 子女端登录
   POST /api/family/register   注册子女账号（PBKDF2 加盐哈希，仅存本机）
   POST /api/family/login      登录 -> 返回 token（Bearer）
   GET  /api/family/me         当前登录用户信息

2. 风险告警推送（识别到风险 -> 子女端可见）
   GET  /api/family/alerts            未确认 + 历史告警（源自 history.jsonl 中非放行记录）
   POST /api/family/alerts/{id}/ack   子女确认"已知晓"

3. 拉子女进通话（三方通话，房间制）
   POST /api/call/invite       父母端发起守护呼叫 -> {room_id}
   GET  /api/call/pending      子女端轮询来电（响铃中的房间）
   WS   /ws/call/{room_id}?role=elder|child
        二进制帧 = 16kHz Int16 PCM 音频，双向转发；
        文本帧 = JSON 控制消息（join/state/transcript/leave），透传给对端。

设计说明：
- 账号与 token 落盘 data/family_accounts.json（密钥不进 C 盘以外的约束不适用于
  本机演示数据；口令加盐哈希，不存明文）。
- 呼叫房间仅存内存，服务重启即清空（演示语义正确：重启后双方重连即可）。
- 父母端（elder）走本机页面，无需登录；子女端接口全部需要 Bearer token。
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import secrets
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from paths import DATA_DIR, DEFAULT_FAMILY_ID, HIST_FILE, VOICE_DIR
from api.history_store import iter_history
from private_fs import append_private_text, write_private_text

ACCOUNTS_FILE = DATA_DIR / "family_accounts.json"   # {"users": {...}, "tokens": {...}}
ACK_FILE = DATA_DIR / "family_alert_ack.json"        # {family_id: [alert_id, ...]}
ACK_EVENT_FILE = DATA_DIR / "family_alert_events.jsonl"
VOICE_OWNERS_FILE = DATA_DIR / "voice_owners.json"   # {voice_name: family_id}
INVITES_FILE = DATA_DIR / "family_invites.json"

router = APIRouter(tags=["family"])
_store_lock = threading.Lock()
TOKEN_TTL_S = 30 * 86400
WS_TICKET_TTL_S = 30
LOGIN_WINDOW_S = 5 * 60
LOGIN_MAX_FAILURES = 5
LOGIN_MAX_TRACKED_IDENTITIES = 4096
PASSWORD_MIN_CHARS = 8
PASSWORD_MAX_CHARS = 128
ELDER_ACCESS_ENV = "VERICALL_ELDER_ACCESS_KEY"
ELDER_ACCESS_HEADER = "X-Elder-Access-Key"
ELDER_ACCESS_MIN_CHARS = 16
EXPOSE_DEMO_CREDENTIALS_ENV = "VERICALL_EXPOSE_DEMO_CREDENTIALS"
DEMO_FAMILY_ID = DEFAULT_FAMILY_ID
_ws_tickets: dict[str, dict] = {}
_ws_tickets_lock = threading.Lock()
_login_failures: dict[str, list[float]] = {}
_login_lock = threading.Lock()


def _env_enabled(name: str, default: str = "1") -> bool:
    return (os.environ.get(name, default) or default).strip().lower() in (
        "1", "true", "yes", "on")


def _env_positive_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)) or default)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _header_value(headers, name: str) -> str:
    """读取请求头，兼容 FastAPI Headers 与测试中的普通字典。"""
    if not headers:
        return ""
    try:
        value = headers.get(name)
    except AttributeError:
        value = None
    if value is None:
        lowered = name.casefold()
        try:
            value = next(
                (item for key, item in headers.items()
                 if str(key).casefold() == lowered),
                None,
            )
        except (AttributeError, TypeError):
            value = None
    return str(value or "")


def _elder_access_key() -> str:
    return (os.environ.get(ELDER_ACCESS_ENV, "") or "").strip()


def _has_valid_elder_access(request: Request | None) -> bool:
    configured = _elder_access_key()
    supplied = _header_value(
        getattr(request, "headers", None), ELDER_ACCESS_HEADER)
    return bool(
        len(configured) >= ELDER_ACCESS_MIN_CHARS
        and supplied
        and hmac.compare_digest(configured, supplied)
    )


def _public_config(request: Request | None = None) -> dict:
    raw_limit = os.environ.get("VERICALL_MAX_UPLOAD_MB", "25")
    try:
        max_upload_mb = int(raw_limit)
    except (TypeError, ValueError):
        max_upload_mb = 25
    if max_upload_mb <= 0:
        max_upload_mb = 25
    demo_account = _env_enabled("VERICALL_DEMO_ACCOUNT", "1")
    config = {
        "allow_registration": _env_enabled("VERICALL_ALLOW_REGISTRATION", "1"),
        "demo_account": demo_account,
        "invite_required": _env_enabled(
            "VERICALL_REQUIRE_FAMILY_INVITE", "0"),
        "max_upload_mb": max_upload_mb,
        "challenge_ttl_s": 60,
        "call_rooms_in_memory": True,
    }
    expose_demo_credentials = _env_enabled(
        EXPOSE_DEMO_CREDENTIALS_ENV, "0")
    local_request = bool(request is not None and is_local_request(request))
    demo_username = os.environ.get("VERICALL_DEMO_USERNAME", "DemoChild").strip()
    demo_password = os.environ.get("VERICALL_DEMO_PASSWORD", "").strip()
    if (
        demo_account
        and (expose_demo_credentials or local_request)
        and demo_username
        and demo_password
    ):
        credentials = {
            "child_username": demo_username,
            "child_password": demo_password,
        }
        elder_key = _elder_access_key()
        if len(elder_key) >= ELDER_ACCESS_MIN_CHARS:
            credentials["elder_key"] = elder_key
        config["demo_credentials"] = credentials
    return config


@router.get("/api/config")
def public_config(request: Request = None):
    """公开前端所需的能力开关，不包含密钥或家庭数据。"""
    return _public_config(request)

# ---------------------------------------------------------------- 账号存储

def _token_key(token: str) -> str:
    """Token 落盘只保存 SHA-256 摘要，避免账号文件泄露后可直接冒用。"""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _normalize_store(d: dict) -> bool:
    """补齐家庭字段并迁移旧明文 token；返回是否发生改动。"""
    changed = False
    users = d.setdefault("users", {})
    for user in users.values():
        if not isinstance(user, dict):
            continue
        if not user.get("family_id"):
            user["family_id"] = DEMO_FAMILY_ID
            changed = True

    tokens = d.setdefault("tokens", {})
    migrated = {}
    for key, value in tokens.items():
        if not isinstance(value, dict):
            changed = True
            continue
        hashed = key if re.fullmatch(r"[0-9a-f]{64}", str(key)) else _token_key(str(key))
        if hashed != key:
            changed = True
        item = dict(value)
        username = str(item.get("username") or "")
        user = users.get(username) or {}
        family_id = str(item.get("family_id") or user.get("family_id") or
                        DEMO_FAMILY_ID)
        if item.get("family_id") != family_id:
            item["family_id"] = family_id
            changed = True
        migrated[hashed] = item
    if migrated != tokens:
        d["tokens"] = migrated
    return changed


def _load_store() -> dict:
    if ACCOUNTS_FILE.is_file():
        try:
            d = json.loads(ACCOUNTS_FILE.read_text(encoding="utf-8"))
            d.setdefault("users", {})
            d.setdefault("tokens", {})
            _normalize_store(d)
            return d
        except Exception:  # noqa: BLE001
            pass
    return {"users": {}, "tokens": {}}


def _save_store(d: dict):
    write_private_text(
        ACCOUNTS_FILE, json.dumps(d, ensure_ascii=False, indent=1))


def _hash_pw(pw: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"),
                               bytes.fromhex(salt), 60_000).hex()


class _Auth(BaseModel):
    username: str
    password: str
    display_name: str | None = None
    invitation: str | None = None


def _load_invites() -> dict:
    if INVITES_FILE.is_file():
        try:
            data = json.loads(INVITES_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("invites", {})
                return data
        except Exception:  # noqa: BLE001
            pass
    return {"invites": {}}


def _save_invites(data: dict) -> None:
    write_private_text(
        INVITES_FILE, json.dumps(data, ensure_ascii=False, indent=1))


def _resolve_invitation(code: str, invites: dict) -> tuple[str | None, str | None]:
    """校验邀请码并返回 ``(family_id, error)``。不在此处消耗次数。"""
    code = str(code or "").strip()
    if not re.fullmatch(r"[\w.-]{6,80}", code):
        return None, "invalid_invitation"
    item = invites.get("invites", {}).get(code)
    if not isinstance(item, dict):
        return None, "invalid_invitation"
    if item.get("active", True) is not True:
        return None, "invitation_disabled"
    expires_at = item.get("expires_at")
    if expires_at not in (None, "") and time.time() >= float(expires_at):
        return None, "invitation_expired"
    family_id = str(item.get("family_id") or "").strip()
    if not re.fullmatch(r"[\w.-]{1,64}", family_id):
        return None, "invalid_invitation_family"
    used = int(item.get("used", 0) or 0)
    max_uses = int(item.get("max_uses", 1) or 1)
    if used >= max_uses:
        return None, "invitation_exhausted"
    return family_id, None


@router.post("/api/family/register")
def family_register(body: _Auth):
    config = _public_config()
    if not config["allow_registration"]:
        return JSONResponse(
            {"error": "registration_disabled",
             "message": "当前部署已关闭自助注册，请联系管理员开通账号"},
            status_code=403)
    username = body.username.strip()
    if not re.fullmatch(r"[\w-]{3,20}", username):
        return JSONResponse({"error": "bad_username",
                             "message": "用户名需 3-20 位字母/数字/下划线"},
                            status_code=400)
    if not PASSWORD_MIN_CHARS <= len(body.password) <= PASSWORD_MAX_CHARS:
        return JSONResponse({"error": "weak_password",
                             "message": (
                                 f"密码需 {PASSWORD_MIN_CHARS}-"
                                 f"{PASSWORD_MAX_CHARS} 位")},
                            status_code=400)
    invitation = (body.invitation or "").strip()
    if config["invite_required"] and not invitation:
        return JSONResponse(
            {"error": "invitation_required",
             "message": "试点部署需要家庭邀请码，请联系项目管理员"},
            status_code=400)
    with _store_lock:
        store = _load_store()
        if username in store["users"]:
            return JSONResponse({"error": "exists",
                                 "message": "该用户名已被注册"}, status_code=409)
        family_id = DEMO_FAMILY_ID
        invites = _load_invites()
        if invitation:
            family_id, invite_error = _resolve_invitation(invitation, invites)
            if invite_error:
                messages = {
                    "invalid_invitation": "邀请码无效，请联系项目管理员",
                    "invitation_disabled": "邀请码已停用",
                    "invitation_expired": "邀请码已过期",
                    "invalid_invitation_family": "邀请码家庭配置无效",
                    "invitation_exhausted": "邀请码使用次数已达上限",
                }
                return JSONResponse(
                    {"error": invite_error,
                     "message": messages[invite_error]},
                    status_code=403)
            item = invites["invites"][invitation]
            item["used"] = int(item.get("used", 0) or 0) + 1
            item["last_used_at"] = time.time()
            _save_invites(invites)
        salt = secrets.token_hex(16)
        store["users"][username] = {
            "salt": salt,
            "hash": _hash_pw(body.password, salt),
            "display_name": (body.display_name or username).strip()[:20],
            "family_id": family_id,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        _save_store(store)
    return {"ok": True, "message": "注册成功，请登录"}


@router.post("/api/family/login")
def family_login(body: _Auth, request: Request = None):
    username = body.username.strip()
    login_keys = _login_identity_keys(request, username)
    retry_after = _login_retry_after(login_keys)
    if retry_after:
        return JSONResponse(
            {"error": "too_many_attempts",
             "message": "登录尝试过于频繁，请稍后再试"},
            status_code=429,
            headers={"Retry-After": str(retry_after)},
        )
    with _store_lock:
        store = _load_store()
        u = store["users"].get(username)
        if not u:
            # 无账号也执行一次等价哈希，避免通过响应时间枚举用户名。
            _hash_pw(body.password[:PASSWORD_MAX_CHARS], "00" * 16)
            _record_login_failure(login_keys)
            return JSONResponse({"error": "bad_credentials",
                                 "message": "用户名或密码不正确"}, status_code=401)
        if not body.password or len(body.password) > PASSWORD_MAX_CHARS:
            _record_login_failure(login_keys)
            return JSONResponse({"error": "bad_credentials",
                                 "message": "用户名或密码不正确"}, status_code=401)
        try:
            password_hash = _hash_pw(body.password, u["salt"])
        except (KeyError, TypeError, ValueError):
            password_hash = ""
        if not secrets.compare_digest(password_hash, str(u.get("hash", ""))):
            _record_login_failure(login_keys)
            return JSONResponse({"error": "bad_credentials",
                                 "message": "用户名或密码不正确"}, status_code=401)
        _clear_login_failures(login_keys)
        token = secrets.token_hex(24)
        store["tokens"][_token_key(token)] = {
            "username": username,
            "family_id": u.get("family_id") or DEMO_FAMILY_ID,
            "created": time.time(),
        }
        # 清理 30 天前的旧 token
        now = time.time()
        store["tokens"] = {t: v for t, v in store["tokens"].items()
                           if now - v.get("created", 0) < TOKEN_TTL_S}
        _save_store(store)
    return {"ok": True, "token": token,
            "user": {"username": username,
                     "display_name": u["display_name"],
                     "family_id": u.get("family_id") or DEMO_FAMILY_ID}}


def _user_from_token(token: str | None) -> dict:
    if not token:
        raise HTTPException(status_code=401, detail="not_logged_in")
    with _store_lock:
        store = _load_store()
        key = _token_key(token)
        t = store["tokens"].get(key)
        if not t:
            raise HTTPException(status_code=401, detail="token_expired")
        try:
            expired = time.time() - float(t.get("created", 0)) >= TOKEN_TTL_S
        except (TypeError, ValueError):
            expired = True
        if expired:
            store["tokens"].pop(key, None)
            _save_store(store)
            raise HTTPException(status_code=401, detail="token_expired")
        u = store["users"].get(t["username"])
        if not u:
            raise HTTPException(status_code=401, detail="user_gone")
        # 旧账户/明文 token 首次成功鉴权后即完成持久迁移。
        _save_store(store)
        return {
            "username": t["username"],
            "display_name": u["display_name"],
            "family_id": t.get("family_id") or u.get("family_id") or
                         DEMO_FAMILY_ID,
        }


def current_user(authorization: str | None = Header(default=None)) -> dict:
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
    return _user_from_token(token)


def _bearer_token(authorization: str | None) -> str | None:
    if authorization and authorization.startswith("Bearer "):
        return authorization[7:].strip()
    return None


def _trust_proxy_headers() -> bool:
    return _env_enabled("VERICALL_TRUST_PROXY_HEADERS", "0")


_LOCAL_CLIENT_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}
_LOCAL_BROWSER_HOSTS = {"127.0.0.1", "::1", "localhost"}


def _normalized_host(host: str | None) -> str:
    value = str(host or "").strip().lower()
    if value.startswith("[") and "]" in value:
        value = value[1:value.index("]")]
    if value.startswith("::ffff:"):
        value = value[7:]
    return value


def _is_loopback_host(host: str | None) -> bool:
    return _normalized_host(host) in _LOCAL_CLIENT_HOSTS


def _forwarded_client_ip(headers) -> str | None:
    if headers is None:
        return None
    for name in ("x-forwarded-for", "x-real-ip"):
        try:
            value = headers.get(name) or ""
        except Exception:  # noqa: BLE001
            continue
        if value:
            return value.split(",", 1)[0].strip()[:64] or None
    return None


def _origin_matches_loopback_host(headers, scheme: str) -> bool:
    """浏览器本机兼容路径只接受指向同一回环主机的同源页面。"""
    if headers is None:
        return True
    try:
        origin = str(headers.get("origin") or "").strip()
    except Exception:  # noqa: BLE001
        return False
    if not origin:
        return True
    if origin == "null":
        return False
    try:
        parsed_origin = urlsplit(origin)
        target = urlsplit("//" + str(headers.get("host") or ""))
    except Exception:  # noqa: BLE001
        return False
    origin_host = _normalized_host(parsed_origin.hostname)
    target_host = _normalized_host(target.hostname)
    if (parsed_origin.scheme not in ("http", "https")
            or origin_host not in _LOCAL_BROWSER_HOSTS
            or target_host not in _LOCAL_BROWSER_HOSTS
            or origin_host != target_host):
        return False
    default_ports = {"http": 80, "https": 443, "ws": 80, "wss": 443}
    origin_port = parsed_origin.port or default_ports.get(
        parsed_origin.scheme, 0)
    target_port = target.port or default_ports.get(str(scheme).lower(), 0)
    return origin_port == target_port


def _has_forwarding_headers(headers) -> bool:
    if headers is None:
        return False
    for name in ("forwarded", "x-forwarded-for", "x-real-ip",
                 "x-forwarded-host", "x-forwarded-proto"):
        try:
            if headers.get(name):
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


def _client_ip(request: Request | None) -> str:
    if request is None:
        return "unknown"
    if _trust_proxy_headers():
        forwarded = _forwarded_client_ip(getattr(request, "headers", None))
        if forwarded:
            return forwarded
    return str((request.client.host if request.client else "") or "unknown")


def is_local_request(request: Request | None) -> bool:
    """父母端与谛听同机；仅本机地址可走无账号兼容路径。

    默认不信任转发头：反向代理与谛听同机时，远端请求可能以 127.0.0.1
    到达。只有显式设置 ``VERICALL_TRUST_PROXY_HEADERS=1`` 才接受代理头。
    """
    if request is None:
        return False
    headers = getattr(request, "headers", None)
    if _trust_proxy_headers():
        host = _client_ip(request)
    else:
        if _has_forwarding_headers(headers):
            return False
        host = (request.client.host if request.client else "") or ""
    scheme = str(getattr(getattr(request, "url", None), "scheme", "http"))
    return (_is_loopback_host(host)
            and _origin_matches_loopback_host(headers, scheme))


def is_local_websocket(websocket: WebSocket | None) -> bool:
    """WebSocket 版本的本机判断，拒绝携带代理转发头的伪本机请求。"""
    if websocket is None:
        return False
    headers = getattr(websocket, "headers", None)
    if _trust_proxy_headers():
        host = (_forwarded_client_ip(headers)
                or (websocket.client.host if websocket.client else "") or "")
    else:
        if _has_forwarding_headers(headers):
            return False
        host = (websocket.client.host if websocket.client else "") or ""
    scheme = str(getattr(getattr(websocket, "url", None), "scheme", "ws"))
    return (_is_loopback_host(host)
            and _origin_matches_loopback_host(headers, scheme))


def _login_identity_keys(request: Request | None, username: str) -> tuple[str, str]:
    return (
        f"ip:{_client_ip(request)}",
        f"user:{username.casefold()[:80]}",
    )


def _prune_login_failures_locked(
        now: float, max_entries: int | None = None) -> None:
    cutoff = now - LOGIN_WINDOW_S
    for key, attempts in list(_login_failures.items()):
        active = [ts for ts in attempts if ts > cutoff]
        if active:
            _login_failures[key] = active
        else:
            _login_failures.pop(key, None)
    limit = max_entries or _env_positive_int(
        "VERICALL_LOGIN_MAX_TRACKED_IDENTITIES",
        LOGIN_MAX_TRACKED_IDENTITIES,
    )
    while len(_login_failures) > limit:
        _login_failures.pop(next(iter(_login_failures)), None)


def _login_retry_after(keys: tuple[str, str]) -> int:
    now = time.time()
    with _login_lock:
        _prune_login_failures_locked(now)
        waits = []
        for key in keys:
            attempts = _login_failures.get(key, [])
            if len(attempts) >= LOGIN_MAX_FAILURES:
                waits.append(LOGIN_WINDOW_S - (now - attempts[0]))
        return max(0, math.ceil(max(waits))) if waits else 0


def _record_login_failure(keys: tuple[str, str]) -> None:
    now = time.time()
    with _login_lock:
        _prune_login_failures_locked(now)
        for key in keys:
            _login_failures.setdefault(key, []).append(now)
        _prune_login_failures_locked(now)


def _clear_login_failures(keys: tuple[str, str]) -> None:
    with _login_lock:
        for key in keys:
            _login_failures.pop(key, None)


def request_family_scope(request: Request | None,
                         authorization: str | None,
                         *, required: bool = True,
                         allow_elder_key: bool = False
                         ) -> tuple[str | None, str]:
    """解析请求家庭范围。

    返回 ``(family_id, access_mode)``。指定接口可接受适老设备访问口令；
    Bearer token 次之；无凭据时仅本机父母端映射到默认家庭。远端匿名可选
    场景返回 ``(None, "public")``，必需场景直接 401。
    """
    if allow_elder_key and _has_valid_elder_access(request):
        return DEMO_FAMILY_ID, "elder_key"
    token = _bearer_token(authorization)
    if token:
        user = _user_from_token(token)
        return _family_of(user), "token"
    if authorization:
        raise HTTPException(status_code=401, detail="invalid_authorization")
    if is_local_request(request):
        return DEMO_FAMILY_ID, "local"
    if required:
        raise HTTPException(status_code=401, detail="not_logged_in")
    return None, "public"


# ---------------------------------------------------------------- 声纹归属

def _load_voice_owners() -> dict[str, str]:
    if VOICE_OWNERS_FILE.is_file():
        try:
            raw = json.loads(VOICE_OWNERS_FILE.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return {str(k): str(v) for k, v in raw.items() if v}
        except Exception:  # noqa: BLE001
            pass
    return {}


def _save_voice_owners(owners: dict[str, str]) -> None:
    write_private_text(
        VOICE_OWNERS_FILE,
        json.dumps(owners, ensure_ascii=False, indent=1),
    )


def voice_owner(name: str) -> str:
    """返回声纹归属家庭；旧声纹没有元数据时归入默认演示家庭。"""
    return _load_voice_owners().get(str(name), DEMO_FAMILY_ID)


def claim_voice_owner(name: str, family_id: str) -> bool:
    """占用声纹名，避免两个家庭在共享管线中使用同名但不同身份。"""
    name = str(name).strip()
    family_id = str(family_id or DEMO_FAMILY_ID)
    with _store_lock:
        owners = _load_voice_owners()
        current = owners.get(name)
        if current is None and (VOICE_DIR / name).is_dir():
            # 旧声纹目录默认属于本机演示家庭。
            current = DEMO_FAMILY_ID
        if current != family_id:
            if current is not None:
                return False
        if owners.get(name) != family_id:
            owners[name] = family_id
            _save_voice_owners(owners)
    return True


def _prune_ws_tickets(now: float | None = None) -> None:
    """清理过期的一次性 WebSocket 票据；调用方需持有票据锁。"""
    now = time.time() if now is None else now
    for ticket, item in list(_ws_tickets.items()):
        if float(item.get("expires", 0)) <= now:
            _ws_tickets.pop(ticket, None)


def _issue_ws_ticket(user: dict, room_id: str) -> str:
    return _issue_ws_ticket_for(user, purpose="call", room_id=room_id)


def _issue_ws_ticket_for(user: dict, *, purpose: str,
                         room_id: str | None = None) -> str:
    ticket = secrets.token_urlsafe(24)
    now = time.time()
    with _ws_tickets_lock:
        _prune_ws_tickets(now)
        _ws_tickets[ticket] = {
            "purpose": purpose,
            "family_id": _family_of(user),
            "expires": now + WS_TICKET_TTL_S,
        }
        if room_id is not None:
            _ws_tickets[ticket]["room_id"] = room_id
    return ticket


def _consume_ws_ticket(ticket: str | None, *, purpose: str,
                       room_id: str | None = None) -> str | None:
    """原子消费一次性票据，返回绑定家庭；用途或资源不匹配时返回 None。"""
    if not ticket:
        return None
    now = time.time()
    with _ws_tickets_lock:
        item = _ws_tickets.pop(ticket, None)
        _prune_ws_tickets(now)
    if not item or float(item.get("expires", 0)) <= now:
        return None
    if item.get("purpose", "call") != purpose:
        return None
    if room_id is not None and item.get("room_id") != room_id:
        return None
    return str(item.get("family_id") or DEMO_FAMILY_ID)


def authorize_websocket(websocket: WebSocket, *, purpose: str,
                        room_id: str | None = None) -> tuple[str, str]:
    """解析 WebSocket 家庭范围，支持 Bearer、一次性票据和本机兼容路径。"""
    authorization = websocket.headers.get("authorization") or ""
    token = _bearer_token(authorization)
    if token:
        return _family_of(_user_from_token(token)), "token"
    if authorization:
        raise HTTPException(status_code=401, detail="invalid_authorization")
    ticket_family = _consume_ws_ticket(
        websocket.query_params.get("ticket"), purpose=purpose,
        room_id=room_id)
    if ticket_family:
        return ticket_family, "ticket"
    if websocket.query_params.get("ticket"):
        raise HTTPException(status_code=401, detail="invalid_ticket")
    if is_local_websocket(websocket):
        return DEMO_FAMILY_ID, "local"
    raise HTTPException(status_code=401, detail="not_logged_in")


@router.get("/api/family/me")
def family_me(user: dict = Depends(current_user)):
    return {"ok": True, "user": user}


# ---------------------------------------------------------------- 风险告警

def _family_of(user: dict | None) -> str:
    return str((user or {}).get("family_id") or DEMO_FAMILY_ID)


def _load_acks(family_id: str = DEMO_FAMILY_ID) -> set:
    if ACK_FILE.is_file():
        try:
            raw = json.loads(ACK_FILE.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return set(raw.get(family_id, []))
            if isinstance(raw, list) and family_id == DEMO_FAMILY_ID:
                return set(raw)  # 兼容旧版全局列表，仅归入默认家庭
        except Exception:  # noqa: BLE001
            pass
    return set()


def _save_acks(family_id: str, acked: set) -> None:
    data = {}
    if ACK_FILE.is_file():
        try:
            raw = json.loads(ACK_FILE.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                data = {str(k): list(v) for k, v in raw.items()
                        if isinstance(v, list)}
            elif isinstance(raw, list):
                data[DEMO_FAMILY_ID] = raw
        except Exception:  # noqa: BLE001
            pass
    data[family_id] = sorted(acked)
    write_private_text(ACK_FILE, json.dumps(data, ensure_ascii=False))


@router.get("/api/family/alerts")
def family_alerts(user: dict = Depends(current_user), limit: int = 100):
    """从检测历史中筛出非放行记录，作为子女端告警事件。"""
    family_id = _family_of(user)
    acked = _load_acks(family_id)
    alerts = []
    for rec in iter_history(
            HIST_FILE, family_id=family_id,
            default_family_id=DEMO_FAMILY_ID):
        if rec.get("final") not in ("block", "caution"):
            continue
        rec["ack"] = rec.get("id") in acked
        alerts.append(rec)
    alerts.reverse()
    return {"family_id": family_id,
            "alerts": alerts[:limit],
            "unacked": sum(1 for a in alerts if not a.get("ack"))}


@router.post("/api/family/alerts/{alert_id}/ack")
def family_alert_ack(alert_id: str, user: dict = Depends(current_user)):
    family_id = _family_of(user)
    with _store_lock:
        acked = _load_acks(family_id)
        acked.add(alert_id)
        _save_acks(family_id, acked)
        event = {
            "alert_id": alert_id,
            "family_id": family_id,
            "ack_at": time.time(),
            "ack_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        append_private_text(
            ACK_EVENT_FILE, json.dumps(event, ensure_ascii=False) + "\n")
    return {"ok": True}


# ---------------------------------------------------------------- 号码联防举报

@router.post("/api/family/numbers/report")
def family_number_report(body: dict, user: dict = Depends(current_user)):
    """子女端一键举报：写入当前家庭私有举报库，不污染其他家庭。

    公开号码库仍由管理员/公开源维护；家庭举报只对所属家庭的检测生效。
    """
    from fusion.number_channel import NumberChannel
    family_id = _family_of(user)
    num = str(body.get("number") or "").strip()
    if not num:
        return JSONResponse({"error": "no_number"}, status_code=400)
    nc = NumberChannel()
    if not nc.report(
            num, remark=str(body.get("remark") or "子女端举报")[:40],
            family_id=family_id):
        return JSONResponse({"error": "invalid_number",
                             "message": "号码格式无效，未写入黑名单"}, status_code=400)
    return {"ok": True,
            "message": "已加入本家庭黑名单，仅对当前家庭的来电分析生效"}


# ---------------------------------------------------------------- 三方通话

class _CallRoom:
    """一次守护呼叫：elder 发起，child 接听；音频与控制消息双向转发。"""

    def __init__(self, room_id: str, reason: str, level: str,
                 transcript: str = "", family_id: str = DEMO_FAMILY_ID):
        self.id = room_id
        self.family_id = family_id
        self.reason = reason
        self.level = level            # red | yellow
        self.transcript = transcript
        self.created = time.time()
        self.state = "ringing"        # ringing -> active -> ended
        self.elder: WebSocket | None = None
        self.child: WebSocket | None = None


_rooms: dict[str, _CallRoom] = {}
_rooms_lock = threading.Lock()
_call_connections_lock = threading.Lock()
_active_call_connections = 0


def _acquire_call_connection() -> bool:
    global _active_call_connections
    limit = _env_positive_int("VERICALL_MAX_CALL_CONNECTIONS", 8)
    with _call_connections_lock:
        if _active_call_connections >= limit:
            return False
        _active_call_connections += 1
        return True


def _release_call_connection() -> None:
    global _active_call_connections
    with _call_connections_lock:
        _active_call_connections = max(0, _active_call_connections - 1)


@router.post("/api/call/invite")
async def call_invite(body: dict, request: Request):
    """父母端发起「拉子女进通话」。body: {reason, level, transcript?}

    加固（2026-09-11）：该接口不要求子女账号，默认仅允许本机调用。
    公网适老端必须携带 VERICALL_ELDER_ACCESS_KEY；需放开其他远程调用时
    可显式设置 VERICALL_ALLOW_REMOTE_INVITE=1。
    """
    if os.environ.get("VERICALL_ALLOW_REMOTE_INVITE", "0") != "1":
        if (not _has_valid_elder_access(request)
                and not is_local_request(request)):
            return JSONResponse({"error": "forbidden",
                                 "message": "守护呼叫仅允许本机发起"}, status_code=403)
    reason = str(body.get("reason") or "谛听检测到可疑来电")[:200]
    level = str(body.get("level") or "red")
    if level not in ("red", "yellow"):
        level = "red"
    transcript = str(body.get("transcript") or "")[:500]
    room_id = uuid.uuid4().hex[:8]
    with _rooms_lock:
        # 同一时刻只保留一个响铃房间（父母端单线场景）
        for r in list(_rooms.values()):
            if r.state == "ringing":
                r.state = "ended"
                del _rooms[r.id]
        _rooms[room_id] = _CallRoom(
            room_id, reason, level, transcript,
            family_id=DEMO_FAMILY_ID)
    elder_ticket = _issue_ws_ticket_for(
        {"family_id": DEMO_FAMILY_ID},
        purpose="call_elder",
        room_id=room_id,
    )
    return {
        "ok": True,
        "room_id": room_id,
        "state": "ringing",
        "ws_ticket": elder_ticket,
        "expires_in": WS_TICKET_TTL_S,
    }


@router.get("/api/call/pending")
def call_pending(user: dict = Depends(current_user)):
    """子女端轮询：是否有响铃中的守护呼叫。"""
    family_id = _family_of(user)
    with _rooms_lock:
        for r in _rooms.values():
            if r.state == "ringing" and r.family_id == family_id:
                return {"ringing": True, "room_id": r.id, "reason": r.reason,
                        "level": r.level, "transcript": r.transcript,
                        "age_s": round(time.time() - r.created, 1)}
    return {"ringing": False}


class _WsTicketRequest(BaseModel):
    room_id: str


@router.post("/api/call/ws-ticket")
def call_ws_ticket(body: _WsTicketRequest,
                   user: dict = Depends(current_user)):
    """签发短期一次性票据，避免把长期登录 token 放进 WebSocket URL。"""
    room_id = body.room_id.strip()
    family_id = _family_of(user)
    with _rooms_lock:
        room = _rooms.get(room_id)
        valid = (room is not None and room.state != "ended"
                 and room.family_id == family_id)
    if not valid:
        raise HTTPException(status_code=404, detail="call_not_found")
    return {
        "ok": True,
        "ticket": _issue_ws_ticket(user, room_id),
        "expires_in": WS_TICKET_TTL_S,
    }


@router.post("/api/stream/ws-ticket")
def stream_ws_ticket(request: Request):
    """为子女账号或持有效访问口令的适老设备签发流式 WebSocket 票据。"""
    if _has_valid_elder_access(request):
        user = {"family_id": DEMO_FAMILY_ID}
    else:
        user = current_user(
            getattr(request, "headers", {}).get("authorization"))
    return {
        "ok": True,
        "ticket": _issue_ws_ticket_for(user, purpose="stream"),
        "expires_in": WS_TICKET_TTL_S,
    }


@router.websocket("/ws/call/{room_id}")
async def call_ws(ws: WebSocket, room_id: str):
    role = ws.query_params.get("role", "elder")
    if role not in ("elder", "child"):
        await ws.close(code=4400)
        return
    user = None
    if role == "child":
        ticket = ws.query_params.get("ticket")
        if ticket:
            ticket_family = _consume_ws_ticket(
                ticket, purpose="call", room_id=room_id)
            if not ticket_family:
                await ws.close(code=4401)
                return
            user = {"family_id": ticket_family}
        else:
            # 保留 Authorization 头作为同源客户端的兼容鉴权方式。
            # 长期 token 不再允许放进 URL query，避免被日志或历史记录泄露。
            auth = ws.headers.get("authorization") or ""
            token = auth[7:] if auth.startswith("Bearer ") else ""
            try:
                user = _user_from_token(token)
            except HTTPException:
                await ws.close(code=4401)
                return
    else:
        ticket = ws.query_params.get("ticket")
        if ticket:
            ticket_family = _consume_ws_ticket(
                ticket, purpose="call_elder", room_id=room_id)
            if not ticket_family:
                await ws.close(code=4401)
                return
            user = {"family_id": ticket_family}
        elif (os.environ.get("VERICALL_ALLOW_REMOTE_INVITE", "0") != "1"
                and not is_local_websocket(ws)):
            await ws.close(code=4403)
            return
    with _rooms_lock:
        room = _rooms.get(room_id)
    if room is None or room.state == "ended":
        await ws.close(code=4404)
        return
    if user is not None and _family_of(user) != room.family_id:
        await ws.close(code=4403)
        return

    if not _acquire_call_connection():
        await ws.close(code=4429)
        return

    accepted = False
    registered = False
    try:
        await ws.accept()
        accepted = True
        replaced = False
        with _rooms_lock:
            if room.state == "ended" or _rooms.get(room.id) is not room:
                replaced = True
            else:
                slot = "elder" if role == "elder" else "child"
                previous = getattr(room, slot)
                setattr(room, slot, ws)
                if role == "child" and room.state == "ringing":
                    room.state = "active"
                peer = room.child if role == "elder" else room.elder
                registered = True
        if replaced:
            await ws.close(code=4404)
            return

        async def _notify(target, msg):
            if target is None:
                return
            try:
                await target.send_text(json.dumps(msg, ensure_ascii=False))
            except Exception:  # noqa: BLE001
                pass

        if previous is not None and previous is not ws:
            # 新连接替换同一角色的陈旧连接，避免偶发掉线后无法重连。
            try:
                await previous.close(code=4410)
            except Exception:  # noqa: BLE001
                pass

        # 告知双方房间状态
        await _notify(ws, {"type": "joined", "role": role, "state": room.state,
                           "reason": room.reason, "level": room.level,
                           "transcript": room.transcript})
        await _notify(peer, {"type": "peer_joined", "role": role})

        max_audio_frame = _env_positive_int(
            "VERICALL_MAX_CALL_FRAME_MB", 2) * 1024 * 1024
        max_control_frame = _env_positive_int(
            "VERICALL_MAX_CALL_CONTROL_KB", 16) * 1024
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            data = msg.get("bytes")
            text = msg.get("text")
            if data is not None:
                if not data or len(data) > max_audio_frame or len(data) % 2:
                    await ws.send_json(
                        {"type": "error", "code": "invalid_audio_frame",
                         "message": "音频分片为空、过大或字节长度非法"})
                    await ws.close(code=1009)
                    break
            elif text is not None:
                if len(text.encode("utf-8")) > max_control_frame:
                    await ws.send_json(
                        {"type": "error", "code": "control_frame_too_large"})
                    await ws.close(code=1009)
                    break
                try:
                    obj = json.loads(text)
                except (TypeError, json.JSONDecodeError):
                    await ws.send_json(
                        {"type": "error", "code": "invalid_control_frame"})
                    continue
                if not isinstance(obj, dict):
                    await ws.send_json(
                        {"type": "error", "code": "invalid_control_frame"})
                    continue
                if obj.get("type") == "transcript":
                    room.transcript = str(obj.get("text", ""))[:500]
            target = room.child if role == "elder" else room.elder
            if target is None:
                continue
            try:
                if data is not None:
                    await target.send_bytes(data)          # PCM 音频转发
                elif text is not None:
                    # 控制消息透传（state / transcript / hangup …）
                    await target.send_text(text)
            except Exception:  # noqa: BLE001
                break
    except WebSocketDisconnect:
        pass
    finally:
        if registered:
            with _rooms_lock:
                slot = "elder" if role == "elder" else "child"
                owns_slot = getattr(room, slot) is ws
                if owns_slot:
                    setattr(room, slot, None)
                other = room.child if role == "elder" else room.elder
                empty = room.elder is None and room.child is None
                if empty:
                    room.state = "ended"
                    _rooms.pop(room.id, None)
            if owns_slot:
                await _notify(other, {"type": "peer_left", "role": role})
        if accepted:
            try:
                await ws.close(code=1000)
            except Exception:  # noqa: BLE001
                pass
        _release_call_connection()
