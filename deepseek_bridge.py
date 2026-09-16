#!/usr/bin/env python3
"""
DeepSeek Web → 9Router Bridge (v4.0 — session-invalid recovery)

What's new in v4.0
------------------
* Tool-mode requests now reuse the cached DeepSeek chat by default and send
  only messages added since the last assistant response. This avoids creating
  one upstream chat per tool call and replaying the full transcript each turn.
  Set BRIDGE_TOOL_FRESH_SESSION=1 to restore the old isolation behavior.
* Hard upstream errors are no longer retried by both the web client and bridge;
  429 responses also activate a shared cooldown across concurrent sessions.
* Live Conversation objects are retained, full recovery prompts are generated
  lazily, duplicate attachments are collapsed, raw tool logging defaults off,
  and synthetic SSE output uses larger batches.
* Claude/system context now provides stable session affinity across context
  recaps, tool calls may be batched in one response, and rapid short requests
  are paced more conservatively without delaying already-long generations.
* Fix for DeepSeek `biz_code 26 / "invalid message id"`: when the upstream
  rejects the continuation anchor (parent_message_id) of a cached session,
  the bridge now invalidates that session in the local cache, evicts its
  client from the pool, creates a brand-new session, and retries the same
  prompt on the fresh session — instead of re-sending the poisoned request
  three times in a row.
* Detection is centralised in _is_session_invalid_error(), which matches
  any of: "invalid message id", biz_code 26 (in any quoting style), etc.
* A tombstone sentinel ("__bridge_invalid__") is written to the session
  cache on invalidation, so the entry is not reused even if cache.set("")
  is rejected by the storage layer.
* Separate MAX_SESSION_RECREATIONS budget, so session recreation cannot
  run away if BRIDGE_MAX_TOOL_ATTEMPTS is raised by the user.
* Non-tool conv.send() path now also recovers from session-invalid via
  _send_sync_collect().
* _make_fresh_session() factored out of _resolve_session so both first
  creation and recovery share one code path.

All v3.9 features retained:
* Task-framed TOOL_SYSTEM_PROMPT (no "the runtime ONLY understands X").
* _is_stall() rejects "done"/"ok"/single tokens.
* Unified retry loop, empty-nudge, near-miss-nudge.
* Near-miss detection (marker present, parse failed).
* Separate truncation budget: history capped first, tool block prepended
  unclipped. Parse-failure logs bumped to raw[:2000].
* Windows session-cache OSError tolerance, global request pacing,
  MAX_FLATTENED_CHARS.
* v3.7 brace-aware tool-call extraction, DSML / Anthropic XML fallback,
  bare-JSON fallback, multi-block, missing close tag.
* Anthropic /v1/messages streaming + non-streaming.
"""

from __future__ import annotations

import ast
import asyncio
import base64
import hashlib
import json
import logging
import mimetypes
import os
import re
import tempfile
import threading
import time
import urllib.parse
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import anyio
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

import deepseek_web_api_client as dsw
from deepseek_web_api_client import (
    DeepSeekClient,
    DeepSeekError,
    ConfigurationError,
    DeepSeekWasmSolver,
    ORIGIN,
    Reply,
    StreamEvent,
)

from curl_cffi import requests as ccr

log = logging.getLogger("deepseek_bridge")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


# ── Configuration ───────────────────────────────────────────────────────────
def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).lower() in ("1", "true", "yes", "on")


BRIDGE_HOST = os.environ.get("BRIDGE_HOST", "127.0.0.1")
BRIDGE_PORT = int(os.environ.get("BRIDGE_PORT", "8123"))
DEEPSEEK_TOKEN = os.environ.get("DEEPSEEK_WEB_TOKEN", "")
DEEPSEEK_COOKIES = os.environ.get("DEEPSEEK_COOKIES", "")

BRIDGE_ACCOUNTS_RAW = os.environ.get("BRIDGE_ACCOUNTS", "").strip()
BRIDGE_ACCOUNTS: Dict[str, Dict[str, str]] = {}
if BRIDGE_ACCOUNTS_RAW:
    try:
        BRIDGE_ACCOUNTS = json.loads(BRIDGE_ACCOUNTS_RAW)
    except json.JSONDecodeError as e:
        log.error("Failed to parse BRIDGE_ACCOUNTS JSON: %s", e)

SESSION_TTL_S = float(os.environ.get("BRIDGE_SESSION_TTL_S", str(30 * 24 * 3600)))
FILE_CACHE_TTL_S = float(os.environ.get("BRIDGE_FILE_TTL_S", str(30 * 24 * 3600)))
MAX_FILE_BYTES = int(os.environ.get("BRIDGE_MAX_FILE_BYTES", str(200 * 1024 * 1024)))
EMIT_REASONING = _env_bool("BRIDGE_EMIT_REASONING", True)
EMIT_TOOLS = _env_bool("BRIDGE_EMIT_TOOLS", False)
USE_USER_AS_SESSION = _env_bool("BRIDGE_USE_USER_AS_SESSION", False)
SESSION_KEY_MODE = os.environ.get("BRIDGE_SESSION_KEY_MODE", "system").strip().lower()
IDLE_CLIENT_TTL_S = float(os.environ.get("BRIDGE_IDLE_CLIENT_TTL_S", str(6 * 3600)))
IMPERSONATE = os.environ.get("BRIDGE_IMPERSONATE", "chrome131")

TOOL_MODE_ALWAYS_NEW_SESSION = _env_bool("BRIDGE_TOOL_FRESH_SESSION", False)
LOG_RAW_TOOL_OUTPUT = _env_bool("BRIDGE_LOG_RAW_TOOL_OUTPUT", False)

MAX_FLATTENED_CHARS = int(os.environ.get("BRIDGE_MAX_FLATTENED_CHARS", "60000"))
MAX_INCREMENTAL_CHARS = max(
    4000, int(os.environ.get("BRIDGE_MAX_INCREMENTAL_CHARS", "24000"))
)
MIN_REQUEST_GAP_S = float(os.environ.get("BRIDGE_MIN_REQUEST_GAP_S", "1.5"))
RATE_LIMIT_COOLDOWN_S = float(
    os.environ.get("BRIDGE_RATE_LIMIT_COOLDOWN_S", "15")
)
OUTPUT_CHUNK_CHARS = max(
    1, int(os.environ.get("BRIDGE_OUTPUT_CHUNK_CHARS", "512"))
)
MAX_TOOL_ATTEMPTS = int(os.environ.get("BRIDGE_MAX_TOOL_ATTEMPTS", "3"))
MAX_SESSION_RECREATIONS = int(
    os.environ.get("BRIDGE_MAX_SESSION_RECREATIONS", "2")
)

# Sentinel written to the session cache when a session is invalidated, so
# it will not be resumed even if cache.set(key, "") is rejected.
INVALID_SESSION_SENTINEL = os.environ.get(
    "BRIDGE_SESSION_INVALID_SENTINEL", "__bridge_invalid__"
)

EMPTY_RETRY_NUDGE = (
    "\n\n[system] Your previous reply was empty or too short. "
    "Please answer the last USER message above. If a tool is needed, emit one "
    "<tool_call> block. Otherwise write a normal reply of at least one full "
    "sentence. Do not reply with only 'done' or a single word."
)

NEAR_MISS_RETRY_NUDGE = (
    "\n\n[system] Your previous reply contained a tool-call-shaped block but "
    "it was malformed and could not be executed. Re-emit each intended call "
    "as a separate block in this form, with valid JSON inside:\n"
    "<tool_call>\n"
    '{"name": "ToolName", "arguments": {...}}\n'
    "</tool_call>\n"
    "Do not add commentary before or after."
)

EMPTY_VISIBLE_FALLBACK = (
    "[bridge: DeepSeek returned no usable output after retries. "
    "This usually means upstream rate-limited us, the session was invalidated, "
    "or the prompt was too long. Wait briefly, then retry in this conversation.]"
)

TOOL_DESC_MAX = int(os.environ.get("BRIDGE_TOOL_DESC_MAX", "200"))
TOOL_PARAM_DESC_MAX = int(os.environ.get("BRIDGE_TOOL_PARAM_DESC_MAX", "100"))

_TOOL_ALLOWLIST_RAW = os.environ.get(
    "BRIDGE_TOOL_ALLOWLIST",
    "Bash,Read,Write,Edit,Glob,Grep,LS,TodoRead,TodoWrite,MultiEdit",
)
TOOL_ALLOWLIST: Optional[Set[str]] = None
if _TOOL_ALLOWLIST_RAW.strip() != "*":
    TOOL_ALLOWLIST = {n.strip() for n in _TOOL_ALLOWLIST_RAW.split(",") if n.strip()}

if not DEEPSEEK_TOKEN and not BRIDGE_ACCOUNTS:
    raise ConfigurationError("Either DEEPSEEK_WEB_TOKEN or BRIDGE_ACCOUNTS must be configured.")


# ── Share the WASM module across all DeepSeekClient instances ───────────────
_solver_cache: Dict[str, DeepSeekWasmSolver] = {}
_solver_cache_lock = threading.Lock()
_orig_solver_init = dsw.DeepSeekWasmSolver.__init__


def _cached_solver_init(self, wasm_path: str = dsw.WASM_FILE):
    with _solver_cache_lock:
        cached = _solver_cache.get(wasm_path)
        if cached is not None:
            self.wasm_path = cached.wasm_path
            self.engine = cached.engine
            self.module = cached.module
            return
    _orig_solver_init(self, wasm_path)
    with _solver_cache_lock:
        _solver_cache.setdefault(wasm_path, self)


dsw.DeepSeekWasmSolver.__init__ = _cached_solver_init


# ── Global request pacing ───────────────────────────────────────────────────
_pacing_lock = asyncio.Lock()
_last_request_at: float = 0.0
_rate_limit_until: float = 0.0


def _is_rate_limit_error(err: Optional[str]) -> bool:
    if not err:
        return False
    low = err.lower()
    return (
        "429" in low
        or "rate limit" in low
        or "rate_limit" in low
        or "too many requests" in low
    )


def _activate_rate_limit_cooldown() -> None:
    global _rate_limit_until
    _rate_limit_until = max(
        _rate_limit_until,
        time.monotonic() + max(RATE_LIMIT_COOLDOWN_S, 0.0),
    )


async def _pace():
    global _last_request_at
    async with _pacing_lock:
        now = time.monotonic()
        gap_wait = MIN_REQUEST_GAP_S - (now - _last_request_at)
        cooldown_wait = _rate_limit_until - now
        wait = max(gap_wait, cooldown_wait)
        if wait > 0:
            await anyio.sleep(wait)
        _last_request_at = time.monotonic()


# ── Client pool ─────────────────────────────────────────────────────────────
class ClientPool:
    def __init__(self, token: str, cookies: Optional[str]):
        self._token = token
        self._cookies = cookies
        self._clients: Dict[str, DeepSeekClient] = {}
        self._conversations: Dict[str, Any] = {}
        self._last_used: Dict[str, float] = {}
        self._control: Optional[DeepSeekClient] = None
        self._lock = threading.RLock()
        self._session_locks: Dict[str, asyncio.Lock] = {}
        self._locks_lock = threading.Lock()
        self._warned_resume_fail: Set[str] = set()

    def control(self) -> DeepSeekClient:
        ctl = self._control
        if ctl is not None and getattr(ctl, "_bootstrapped", False):
            return ctl
        with self._lock:
            if self._control is None:
                self._control = DeepSeekClient(
                    token=self._token, cookies=self._cookies or None
                )
            ctl = self._control
            already_bootstrapped = getattr(ctl, "_bootstrapped", False)
        if not already_bootstrapped:
            ctl.bootstrap()
            log.info("Control client bootstrapped")
        return ctl

    def _make_client(self) -> DeepSeekClient:
        ctl = self.control()
        c = DeepSeekClient(token=self._token, cookies=self._cookies or None)
        c.cache = ctl.cache
        c.store = ctl.store
        return c

    def for_session(self, session_id: str) -> DeepSeekClient:
        with self._lock:
            c = self._clients.get(session_id)
            if c is not None:
                self._last_used[session_id] = time.time()
                return c
        c = self._make_client()
        c.session.headers["Referer"] = f"{ORIGIN}/a/chat/s/{session_id}"
        with self._lock:
            existing = self._clients.get(session_id)
            if existing is not None:
                try:
                    c.close()
                except Exception:
                    pass
                self._last_used[session_id] = time.time()
                return existing
            self._clients[session_id] = c
            self._last_used[session_id] = time.time()
            return c

    def new_client_for_new_session(self) -> DeepSeekClient:
        return self._make_client()

    def register(
        self,
        session_id: str,
        client: DeepSeekClient,
        conversation: Optional[Any] = None,
    ) -> None:
        with self._lock:
            client.session.headers["Referer"] = f"{ORIGIN}/a/chat/s/{session_id}"
            self._clients[session_id] = client
            if conversation is not None:
                self._conversations[session_id] = conversation
            self._last_used[session_id] = time.time()

    def conversation_for(self, session_id: str) -> Optional[Any]:
        with self._lock:
            return self._conversations.get(session_id)

    def remember_conversation(self, session_id: str, conversation: Any) -> None:
        with self._lock:
            self._conversations[session_id] = conversation

    def touch(self, session_id: str) -> None:
        with self._lock:
            self._last_used[session_id] = time.time()

    def evict(self, session_id: str) -> None:
        """Remove a session's client from the pool without touching the cache."""
        with self._lock:
            c = self._clients.pop(session_id, None)
            self._conversations.pop(session_id, None)
            self._last_used.pop(session_id, None)
            self._session_locks.pop(session_id, None)
            self._warned_resume_fail.discard(session_id)
        if c is not None:
            try:
                c.close()
            except Exception:
                pass

    def lock_for(self, key: str) -> asyncio.Lock:
        with self._locks_lock:
            lk = self._session_locks.get(key)
            if lk is None:
                lk = asyncio.Lock()
                self._session_locks[key] = lk
            return lk

    def reap_idle(self) -> int:
        cutoff = time.time() - IDLE_CLIENT_TTL_S
        removed = 0
        with self._lock:
            for sid in list(self._clients.keys()):
                if self._last_used.get(sid, 0.0) < cutoff:
                    c = self._clients.pop(sid, None)
                    self._conversations.pop(sid, None)
                    self._last_used.pop(sid, None)
                    self._session_locks.pop(sid, None)
                    if c is not None:
                        try:
                            c.close()
                        except Exception:
                            pass
                    removed += 1
        return removed

    def close_all(self) -> None:
        with self._lock:
            for c in self._clients.values():
                try:
                    c.close()
                except Exception:
                    pass
            self._clients.clear()
            self._conversations.clear()
            self._last_used.clear()
            if self._control is not None:
                try:
                    self._control.close()
                except Exception:
                    pass
                self._control = None


_pools: Dict[Tuple[str, str], ClientPool] = {}
_pools_lock = threading.Lock()


def get_pool(headers: Optional[Dict[str, str]] = None) -> ClientPool:
    token = DEEPSEEK_TOKEN
    cookies = DEEPSEEK_COOKIES
    
    if headers and "authorization" in headers:
        parts = headers["authorization"].split(" ")
        if len(parts) == 2 and parts[0].lower() == "bearer":
            api_key = parts[1]
            if api_key in BRIDGE_ACCOUNTS:
                acc = BRIDGE_ACCOUNTS[api_key]
                token = acc.get("token", "")
                cookies = acc.get("cookies", "")

    if not token:
        raise HTTPException(status_code=401, detail="No DeepSeek token available for this request")
        
    pool_key = (token, cookies)
    with _pools_lock:
        if pool_key not in _pools:
            _pools[pool_key] = ClientPool(token=token, cookies=cookies or None)
        return _pools[pool_key]


def get_all_pools() -> List[ClientPool]:
    with _pools_lock:
        return list(_pools.values())


# ── v4.0: session-invalid detection & recovery helpers ──────────────────────
#
# DeepSeek's web backend returns HTTP 200 with a JSON body of the form
#     {"code":0,"msg":"","data":{"biz_code":26,"biz_msg":"invalid message id",
#                                "biz_data":null}}
# when it is handed a parent_message_id (the anchor for "continue this
# conversation") that it does not recognise for the session. This happens
# most often when we resume a cached session whose server-side last message
# was never persisted (e.g. the previous turn produced an empty stream).
#
# The retry loop below treats this as a session problem, not a prompt
# problem, and rebuilds the session instead of re-sending into the void.

_SESSION_INVALID_MARKERS: Tuple[str, ...] = (
    "invalid message id",
    "biz_code\":26",
    "biz_code':26",
    "biz_code: 26",
    "biz_code:26",
    "biz_code=26",
    "'biz_code': 26",
    '"biz_code": 26',
)


def _is_session_invalid_error(err: Optional[str]) -> bool:
    """True if the upstream error string mentions an invalid message id /
    biz_code 26. Accepts stream_error:..., exception:..., or raw text."""
    if not err:
        return False
    low = err.lower()
    for marker in _SESSION_INVALID_MARKERS:
        if marker.lower() in low:
            return True
    return False


def _invalidate_cached_session(
    pool: ClientPool,
    lookup_key: str,
    session_id: Optional[str] = None,
) -> None:
    """Tombstone the cached session id and evict its client from the pool.

    We write a sentinel rather than an empty string, so that even if the
    cache layer refuses to persist empty values, the entry will not be
    reused by _resolve_session.
    """
    try:
        ctl = pool.control()
    except Exception as e:
        log.warning("invalidate: control client unavailable: %s", e)
        ctl = None

    if ctl is not None:
        try:
            ctl.cache.set(lookup_key, INVALID_SESSION_SENTINEL)
            try:
                ctl.cache.flush()
            except OSError as e:
                log.warning("cache.flush() failed (non-fatal): %s", e)
        except Exception as e:
            log.warning(
                "Failed to tombstone cache for key=%s: %s",
                lookup_key[:24], e,
            )

    if session_id:
        pool.evict(session_id)


async def _make_fresh_session(
    pool: ClientPool,
    lookup_key: str,
) -> Tuple[DeepSeekClient, str, Any]:
    """Create a brand-new DeepSeek session, register it, and cache the id.

    Returns (client, session_id, conv).
    """
    client = pool.new_client_for_new_session()
    conv = await anyio.to_thread.run_sync(client.new_conversation)
    session_id = conv.session_id
    pool.register(session_id, client, conv)
    try:
        pool.control().cache.set(lookup_key, session_id)
        try:
            pool.control().cache.flush()
        except OSError as e:
            log.warning("cache.flush() failed (non-fatal): %s", e)
    except Exception as e:
        log.warning("Failed to cache session_id %s: %s", session_id, e)
    log.info("New session %s (key=%s)", session_id, lookup_key[:24])
    return client, session_id, conv


# ── Request / response models ───────────────────────────────────────────────
ContentPart = Dict[str, Any]
MessageContent = Union[str, List[ContentPart]]


class ChatMessage(BaseModel):
    role: str
    content: Optional[MessageContent] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = "deepseek-chat"
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    user: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None


class AnthropicMessagesRequest(BaseModel):
    model: str
    max_tokens: Optional[int] = None
    messages: List[Dict[str, Any]]
    system: Optional[Union[str, List[Dict[str, Any]]]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


# ── Message normalization ───────────────────────────────────────────────────
def _extract_parts(content: Optional[MessageContent]) -> Tuple[str, List[str]]:
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []

    text_parts: List[str] = []
    file_urls: List[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "text":
            text_parts.append(part.get("text") or "")
        elif ptype == "image_url":
            url = (part.get("image_url") or {}).get("url")
            if url:
                file_urls.append(url)
        elif ptype == "file":
            fobj = part.get("file") or {}
            url = fobj.get("url") or part.get("url")
            if url:
                file_urls.append(url)
        elif ptype == "document":
            src = part.get("source") or {}
            if src.get("type") == "base64":
                mime = src.get("media_type", "application/octet-stream")
                data = src.get("data", "")
                file_urls.append(f"data:{mime};base64,{data}")
            elif src.get("type") == "url":
                url = src.get("url")
                if url:
                    file_urls.append(url)
        elif ptype == "input_audio":
            log.warning("input_audio not supported by DeepSeek Web; ignoring")
    return "\n".join(text_parts), file_urls


class NormalizedMessage:
    __slots__ = ("role", "text", "files")

    def __init__(self, role: str, text: str, files: List[str]):
        self.role = role
        self.text = text
        self.files = files


def _normalize(messages: List[ChatMessage]) -> List[NormalizedMessage]:
    return [
        NormalizedMessage(m.role, *_extract_parts(m.content)) for m in messages
    ]


# ── Session identity ────────────────────────────────────────────────────────
def _hash_texts(parts: List[str]) -> str:
    payload = json.dumps(
        [p.strip() for p in parts], sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _compute_lookup_key(
    req: ChatCompletionRequest,
    headers: Dict[str, str],
) -> str:
    explicit = headers.get("x-session-id") or headers.get("x-conversation-id")
    if not explicit and USE_USER_AS_SESSION:
        explicit = req.user
    if explicit:
        return f"sid:{explicit}"

    if SESSION_KEY_MODE == "system":
        system_parts: List[str] = []
        for m in req.messages:
            if m.role == "system":
                text, _files = _extract_parts(m.content)
                if text.strip():
                    system_parts.append(text)
        if system_parts:
            return "system:" + _hash_texts(system_parts)

    for m in req.messages:
        if m.role == "user":
            text, files = _extract_parts(m.content)
            return "conv:" + _hash_texts([text] + list(files))

    return "conv:" + _hash_texts([""])


# ── File fetching / uploading ───────────────────────────────────────────────
_DATA_URL_RE = re.compile(r"^data:([^;,]+)?(;base64)?,(.*)$", re.DOTALL)
_CD_FILENAME_RE = re.compile(r'filename\*?="?([^";]+)"?', re.IGNORECASE)

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)


def _decode_data_url(url: str) -> Tuple[bytes, str, str]:
    m = _DATA_URL_RE.match(url)
    assert m is not None
    mime = m.group(1) or "application/octet-stream"
    is_b64 = bool(m.group(2))
    raw = m.group(3)
    data = base64.b64decode(raw) if is_b64 else urllib.parse.unquote_to_bytes(raw)
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(413, f"file too large: {len(data)} bytes")
    ext = mimetypes.guess_extension(mime) or ".bin"
    return data, f"upload{ext}", mime


def _fetch_url_sync(url: str) -> Tuple[bytes, str, str]:
    r = ccr.get(
        url,
        impersonate=IMPERSONATE,
        timeout=120,
        allow_redirects=True,
        headers={
            "User-Agent": _BROWSER_UA,
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Sec-Fetch-Dest": "image",
            "Sec-Fetch-Mode": "no-cors",
            "Sec-Fetch-Site": "cross-site",
        },
    )
    if r.status_code != 200:
        raise HTTPException(502, f"upstream returned {r.status_code} for {url[:80]}")
    content = r.content
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(413, f"file too large: {len(content)} bytes")
    mime = (r.headers.get("content-type") or "application/octet-stream").split(";")[0].strip()

    filename = ""
    cd = r.headers.get("content-disposition", "") or ""
    m_cd = _CD_FILENAME_RE.search(cd)
    if m_cd:
        filename = m_cd.group(1)
    if not filename:
        parsed = urllib.parse.urlparse(url)
        filename = os.path.basename(parsed.path) or "upload"
        if not os.path.splitext(filename)[1]:
            filename += mimetypes.guess_extension(mime) or ""
    return content, filename, mime


async def _fetch_url(url: str) -> Tuple[bytes, str, str]:
    if _DATA_URL_RE.match(url):
        return _decode_data_url(url)
    if not url.lower().startswith(("http://", "https://")):
        raise HTTPException(400, f"unsupported URL scheme: {url[:60]}")
    return await anyio.to_thread.run_sync(_fetch_url_sync, url)


async def _resolve_file_ids(urls: List[str], pool: ClientPool) -> List[str]:
    if not urls:
        return []
    ctl = pool.control()
    ids: List[str] = []
    seen_digests: Set[str] = set()
    for url in dict.fromkeys(urls):
        try:
            content, filename, _mime = await _fetch_url(url)
        except HTTPException:
            raise
        except Exception as e:
            log.warning("Failed to fetch %s: %s", url[:80], e)
            continue
        digest = hashlib.sha256(content).hexdigest()
        if digest in seen_digests:
            continue
        seen_digests.add(digest)
        cache_key = f"file:{digest}"
        cached, fresh = ctl.cache.get(cache_key, FILE_CACHE_TTL_S)
        if fresh and cached:
            log.info("File cache hit (%s) -> %s", filename, cached)
            ids.append(cached)
            continue
        suffix = Path(filename).suffix or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            uploaded = await anyio.to_thread.run_sync(
                lambda: ctl.upload_file(tmp_path, poll=True)
            )
            log.info("Uploaded %s -> %s (status=%s)",
                     filename, uploaded.id, uploaded.status)
            ctl.cache.set(cache_key, uploaded.id)
            ids.append(uploaded.id)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return ids


# ── Model flags ─────────────────────────────────────────────────────────────
def _normalize_model_name(model: str) -> str:
    if "/" in model:
        return model.rsplit("/", 1)[-1]
    return model


def _model_flags(model: str) -> Dict[str, bool]:
    name = _normalize_model_name(model).lower()
    return {
        "thinking": "reasoner" in name or "think" in name,
        "search": "search" in name,
    }


# ── Prompt truncation ───────────────────────────────────────────────────────
def _truncate_to(prompt: str, budget: int) -> str:
    """Truncate prompt to fit within `budget` chars, keeping head + tail."""
    if budget <= 0:
        return prompt[-max(budget, 200):]
    if len(prompt) <= budget:
        return prompt
    head_len = budget // 4
    tail_len = budget - head_len - 80
    head = prompt[:head_len]
    tail = prompt[-tail_len:]
    trunc = (
        head
        + f"\n\n[... {len(prompt) - head_len - tail_len} chars of "
          f"middle history truncated by bridge ...]\n\n"
        + tail
    )
    log.info("Truncated prompt: %d -> %d chars", len(prompt), len(trunc))
    return trunc


# ── Tool-call emulation ─────────────────────────────────────────────────────
TOOL_SYSTEM_PROMPT = """\
You have access to tools. To use them, output one or more blocks in this form:

<tool_call>
{"name": "ToolName", "arguments": {"param": "value"}}
</tool_call>

Notes:
- You may emit multiple tool-call blocks in one reply. Batch independent reads,
  searches, and other independent operations to minimize round trips.
- Keep dependent operations sequential: wait for a result when a later call
  needs information from an earlier call.
- The "name" field must exactly match a tool name listed below.
- The "arguments" field is a JSON object matching that tool's parameters.
- After you receive the tool result, continue the task.
- For a requested file edit, inspect only what is necessary, then perform the
  Write or Edit promptly. Prefer one comprehensive write over many tiny edits.
- Do not narrate a plan instead of acting when the required tool is available.
- If no tool is needed, just reply in plain text.
- Every reply must contain content. Never send an empty reply.
"""


def _shorten(text: str, limit: int) -> str:
    if limit <= 0:
        return text
    text = text.strip().replace("\n", " ")
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def _filter_tools(
    tools: List[Dict[str, Any]],
    allowlist: Optional[Set[str]],
) -> List[Dict[str, Any]]:
    if allowlist is None:
        return tools
    out = []
    for t in tools:
        fn = t.get("function") or t
        if str(fn.get("name", "")) in allowlist:
            out.append(t)
    return out


def _format_tools_block(tools: List[Dict[str, Any]]) -> str:
    lines: List[str] = [TOOL_SYSTEM_PROMPT, "\nAvailable tools:"]
    for t in tools:
        fn = t.get("function") or t
        name = fn.get("name", "?")
        desc = _shorten(fn.get("description") or "", TOOL_DESC_MAX)
        params = fn.get("parameters") or {}
        props = params.get("properties") or {}
        required = set(params.get("required") or [])
        lines.append(f"\n- {name}: {desc}")
        if props:
            for pname, pspec in props.items():
                ptype = pspec.get("type", "any")
                req = " (required)" if pname in required else ""
                pdesc = _shorten(pspec.get("description") or "", TOOL_PARAM_DESC_MAX)
                lines.append(f"    {pname}: {ptype}{req} — {pdesc}")
    return "\n".join(lines) + "\n"


# ── Tool-call parsing ───────────────────────────────────────────────────────
_TOOL_CALL_OPEN_RE = re.compile(r"<tool_call\s*>", re.IGNORECASE)
_TOOL_CALL_CLOSE_RE = re.compile(r"</tool_call\s*>", re.IGNORECASE)
_TOOL_CALL_FULL_RE = re.compile(
    r"<tool_call\s*>.*?</tool_call\s*>",
    re.DOTALL | re.IGNORECASE,
)

_XML_INVOKE_RE = re.compile(
    r"<[^<>]*?invoke\s+name=[\"']([^\"']+)[\"']\s*>(.*?)</[^<>]*?invoke\s*>",
    re.DOTALL | re.IGNORECASE,
)
_XML_PARAM_RE = re.compile(
    r"<[^<>]*?parameter\s+name=[\"']([^\"']+)[\"']\s*>(.*?)</[^<>]*?parameter\s*>",
    re.DOTALL | re.IGNORECASE,
)
_XML_WRAPPER_RE = re.compile(
    r"<[^<>]*?(?:function_calls|tool_calls|tool_use)\b[^<>]*>.*?"
    r"</[^<>]*?(?:function_calls|tool_calls|tool_use)\s*>",
    re.DOTALL | re.IGNORECASE,
)
_XML_ORPHAN_PARAM_RE = re.compile(
    r"</?[^<>]*?parameter\b[^<>]*>",
    re.IGNORECASE,
)
_LEAK_RE = re.compile(
    r"<[^<>]*?(?:invoke|parameter|function_calls|tool_calls|tool_use|DSML)"
    r"[^<>]*>",
    re.IGNORECASE,
)

# Stall vocabulary: short replies that mean the model is closing the turn
# without actually answering.
_STALL_WORDS = {
    "done", "ok", "okay", "k", "yes", "no", "y", "n", "sure",
    "…", "...", "——", "—", "-", ":", ".", "ready", "standby",
}


def _is_stall(text: str) -> bool:
    """True if text is effectively empty for our purposes."""
    if text is None:
        return True
    s = text.strip()
    if len(s) < 2:
        return True
    low = s.lower()
    if low in _STALL_WORDS:
        return True
    if len(s) < 8 and " " not in s and not any(c in s for c in ".,!?:;"):
        return True
    return False


def _has_tool_marker(text: str) -> bool:
    """True if text looks like the model tried to call a tool, however
    malformed. Used to distinguish 'plain text reply' from 'near-miss'."""
    if not text:
        return False
    low = text.lower()
    if "<tool_call" in low or "<invoke" in low:
        return True
    if '"arguments"' in text or "'arguments'" in text:
        return True
    return False


def _find_balanced_json(
    text: str, start: int = 0
) -> Optional[Tuple[str, int, int]]:
    n = len(text)
    i = start
    while i < n:
        if text[i] == "{":
            depth = 0
            in_str = False
            escape = False
            j = i
            while j < n:
                c = text[j]
                if in_str:
                    if escape:
                        escape = False
                    elif c == "\\":
                        escape = True
                    elif c == '"':
                        in_str = False
                else:
                    if c == '"':
                        in_str = True
                    elif c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            return text[i:j + 1], i, j + 1
                j += 1
            return None
        i += 1
    return None


def _find_tool_call_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []

    for m in _TOOL_CALL_FULL_RE.finditer(text):
        spans.append((m.start(), m.end()))

    for m in _TOOL_CALL_OPEN_RE.finditer(text):
        if any(s <= m.start() < e for s, e in spans):
            continue
        close = _TOOL_CALL_CLOSE_RE.search(text, m.end())
        if close:
            spans.append((m.start(), close.end()))
            continue
        found = _find_balanced_json(text, m.end())
        if found:
            spans.append((m.start(), found[2]))
            continue
        nxt = _TOOL_CALL_OPEN_RE.search(text, m.end())
        end = nxt.start() if nxt else len(text)
        spans.append((m.start(), end))

    spans.sort()
    dedup: List[Tuple[int, int]] = []
    for s, e in spans:
        if dedup and s < dedup[-1][1]:
            ps, pe = dedup[-1]
            dedup[-1] = (ps, max(pe, e))
        else:
            dedup.append((s, e))
    return dedup


def _extract_tool_call_jsons(text: str) -> List[str]:
    out: List[str] = []
    for s, e in _find_tool_call_spans(text):
        block = text[s:e]
        block = _TOOL_CALL_OPEN_RE.sub("", block, count=1)
        block = _TOOL_CALL_CLOSE_RE.sub("", block, count=1)
        block = re.sub(r"^\s*```(?:json)?\s*", "", block)
        block = re.sub(r"\s*```\s*$", "", block)
        found = _find_balanced_json(block)
        if found:
            out.append(found[0])
        else:
            stripped = block.strip().strip("`").strip()
            if stripped:
                out.append(stripped)
    return out


def _try_json_or_salvage(raw: str) -> Optional[Dict[str, Any]]:
    raw = raw.strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    fixed = re.sub(r",(\s*[}\]])", r"\1", raw)
    try:
        obj = json.loads(fixed)
        if isinstance(obj, dict):
            log.info("Salvaged tool_call JSON by stripping trailing commas")
            return obj
    except json.JSONDecodeError:
        pass
    try:
        obj = ast.literal_eval(raw)
        if isinstance(obj, dict):
            log.info("Salvaged tool_call JSON via ast.literal_eval")
            return obj
    except Exception:
        pass
    return None


def _parse_xml_tool_calls(
    text: str,
    valid_names: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for name, body in _XML_INVOKE_RE.findall(text):
        if valid_names is not None and name not in valid_names:
            log.warning(
                "REJECTED XML tool call %r — not in valid set: %s",
                name, ", ".join(sorted(valid_names)),
            )
            continue
        args: Dict[str, Any] = {}
        for pname, pval in _XML_PARAM_RE.findall(body):
            raw = pval.strip()
            try:
                args[pname] = json.loads(raw)
            except Exception:
                args[pname] = raw
        out.append({
            "id": f"call_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(
                    args, separators=(",", ":"), ensure_ascii=False
                ),
            },
        })
    return out


def _coerce_tool_call(
    obj: Dict[str, Any],
    valid_names: Optional[Set[str]],
) -> Optional[Dict[str, Any]]:
    name = obj.get("name") or obj.get("tool") or obj.get("function")
    if not name:
        return None
    name = str(name)
    if valid_names is not None and name not in valid_names:
        log.warning(
            "REJECTED tool call %r — not in valid set: %s",
            name, ", ".join(sorted(valid_names)),
        )
        return None
    args = obj.get("arguments") or obj.get("args") or obj.get("input") or {}
    if not isinstance(args, (dict, list)):
        args = {"value": args}
    return {
        "id": f"call_{uuid.uuid4().hex[:12]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args, separators=(",", ":"), ensure_ascii=False),
        },
    }


def _parse_tool_calls(
    text: str,
    valid_names: Optional[Set[str]] = None,
) -> Optional[List[Dict[str, Any]]]:
    if not text:
        return None

    out: List[Dict[str, Any]] = []

    for raw in _extract_tool_call_jsons(text):
        obj = _try_json_or_salvage(raw)
        if obj is None:
            log.warning("tool_call JSON parse failed after salvage: %s", raw[:2000])
            continue
        tc = _coerce_tool_call(obj, valid_names)
        if tc:
            out.append(tc)

    if not out:
        xml_out = _parse_xml_tool_calls(text, valid_names)
        if xml_out:
            log.info("Parsed %d tool call(s) from XML fallback", len(xml_out))
            out.extend(xml_out)

    if not out and "<tool_call" not in text.lower() and "<invoke" not in text.lower():
        found = _find_balanced_json(text)
        if found:
            obj = _try_json_or_salvage(found[0])
            if obj is not None:
                tc = _coerce_tool_call(obj, valid_names)
                if tc:
                    log.info("Parsed tool call from bare JSON fallback")
                    out.append(tc)

    return out or None


def _strip_tool_xml(text: str) -> str:
    if not text:
        return text

    spans = _find_tool_call_spans(text)
    if spans:
        parts: List[str] = []
        cursor = 0
        for s, e in spans:
            parts.append(text[cursor:s])
            cursor = e
        parts.append(text[cursor:])
        text = "".join(parts)

    text = _XML_WRAPPER_RE.sub("", text)
    text = _XML_INVOKE_RE.sub("", text)
    text = _XML_ORPHAN_PARAM_RE.sub("", text)

    m = _LEAK_RE.search(text)
    if m:
        log.warning(
            "Leaked tool-call XML after primary strip (fragment): %r",
            m.group(0),
        )
        text = _LEAK_RE.sub("", text)
    return text.strip()


def _valid_tool_names(req: "ChatCompletionRequest") -> Optional[Set[str]]:
    if not req.tools:
        return None
    filtered = _filter_tools(req.tools, TOOL_ALLOWLIST)
    if not filtered:
        return None
    names: Set[str] = set()
    for t in filtered:
        fn = t.get("function") or t
        n = fn.get("name")
        if n:
            names.add(str(n))
    return names or None


def _flatten_history_for_tools(messages: List[ChatMessage]) -> str:
    chunks: List[str] = []
    for m in messages:
        role = m.role
        text, _files = _extract_parts(m.content)

        if role == "system":
            continue
        if role == "user":
            if text:
                chunks.append(f"USER:\n{text}")
        elif role == "assistant":
            if m.tool_calls:
                blocks = []
                for tc in m.tool_calls:
                    fn = tc.get("function") or {}
                    name = fn.get("name") or ""
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except Exception:
                        args = {"_raw": fn.get("arguments")}
                    blocks.append(
                        "<tool_call>\n" +
                        json.dumps({"name": name, "arguments": args},
                                   separators=(",", ":"), ensure_ascii=False) +
                        "\n</tool_call>"
                    )
                if text:
                    chunks.append(f"ASSISTANT:\n{text}\n" + "\n".join(blocks))
                else:
                    chunks.append("ASSISTANT:\n" + "\n".join(blocks))
            elif text:
                chunks.append(f"ASSISTANT:\n{text}")
        elif role == "tool":
            tid = m.tool_call_id or "?"
            body = text or "(empty)"
            chunks.append(f"TOOL_RESULT id={tid}:\n{body}")

    return "\n\n".join(chunks)


def _tool_prompt_for_request(
    req: ChatCompletionRequest,
    *,
    resumed: bool,
) -> str:
    """Build a tool prompt without replaying context already in DeepSeek.

    A fresh upstream chat needs the tool definitions and complete request
    history. A resumed chat already contains that material, so only messages
    added after the latest assistant response are sent (normally one or more
    tool results, or the next user message).
    """
    messages = req.messages
    tool_block = ""

    if resumed:
        last_assistant = next(
            (i for i in range(len(messages) - 1, -1, -1)
             if messages[i].role == "assistant"),
            -1,
        )
        if last_assistant >= 0 and last_assistant + 1 < len(messages):
            messages = messages[last_assistant + 1:]
        elif messages:
            messages = messages[-1:]
    else:
        filtered = _filter_tools(req.tools or [], TOOL_ALLOWLIST)
        tool_block = _format_tools_block(filtered) if filtered else ""

    history = _flatten_history_for_tools(messages)
    if resumed:
        budget = min(MAX_INCREMENTAL_CHARS, MAX_FLATTENED_CHARS)
    else:
        budget = MAX_FLATTENED_CHARS - len(tool_block) - 200
    history = _truncate_to(history, max(budget, 4000))
    return tool_block + ("\n---\n" if tool_block else "") + history


def _describe_codepoints(s: str, limit: int = 400) -> str:
    out: List[str] = []
    for ch in s[:limit]:
        o = ord(ch)
        if o > 127:
            out.append(f"<U+{o:04X}>")
        else:
            out.append(ch)
    if len(s) > limit:
        out.append(f"... (+{len(s) - limit} more)")
    return "".join(out)


# ── DeepSeek send + unified retry loop ──────────────────────────────────────
async def _send_once_collect(
    conv, prompt: str, *, flags: Dict[str, bool], ref_ids: List[str],
) -> Tuple[str, str]:
    """One stream, collect all response text. Returns (text, err)."""
    vision_model = "vision" if ref_ids else None
    buf: List[str] = []
    try:
        gen = conv.stream(
            prompt,
            thinking=flags["thinking"],
            search=flags["search"],
            prefetch=False,
            ref_file_ids=ref_ids,
            model_type=vision_model,
        )
        while True:
            ev: Optional[StreamEvent] = await anyio.to_thread.run_sync(
                _next_or_none, gen
            )
            if ev is None:
                break
            if ev.type == "response":
                buf.append(ev.content)
            elif ev.type == "error":
                return "".join(buf), f"stream_error:{ev.data}"
            elif ev.type == "done":
                break
    except Exception as e:
        return "".join(buf), f"exception:{e}"
    return "".join(buf), ""


async def _send_sync_collect(
    conv,
    prompt: str,
    *,
    flags: Dict[str, bool],
    ref_ids: List[str],
    pool: ClientPool,
    lookup_key: str,
    session_id: str,
) -> Tuple[Reply, str, Any]:
    """Non-streaming send with one-shot session-invalid recovery.

    Returns (reply, session_id, conv). The session_id / conv may differ
    from the inputs if the original session was invalidated.
    """
    vision_model = "vision" if ref_ids else None

    def _do_send(c):
        return c.send(
            prompt,
            thinking=flags["thinking"],
            search=flags["search"],
            prefetch=False,
            ref_file_ids=ref_ids,
            model_type=vision_model,
        )

    try:
        reply: Reply = await anyio.to_thread.run_sync(_do_send, conv)
        return reply, session_id, conv
    except Exception as e:
        if _is_rate_limit_error(str(e)):
            _activate_rate_limit_cooldown()
        if not _is_session_invalid_error(str(e)):
            raise
        log.warning(
            "Non-tool send hit invalid message id (session=%s); recreating session",
            session_id,
        )
        _invalidate_cached_session(pool, lookup_key, session_id)
        _client, new_sid, new_conv = await _make_fresh_session(pool, lookup_key)
        reply = await anyio.to_thread.run_sync(_do_send, new_conv)
        return reply, new_sid, new_conv


async def _call_deepseek_with_retries(
    conv,
    prompt: str,
    *,
    flags: Dict[str, bool],
    ref_ids: List[str],
    session_id: str,
    valid_names: Optional[Set[str]],
    pool: Optional[ClientPool] = None,
    lookup_key: Optional[str] = None,
    recreate_session: Optional[Callable[[], Awaitable[Tuple[Any, str, Any]]]] = None,
    recovery_prompt: Optional[str] = None,
    recovery_prompt_factory: Optional[Callable[[], str]] = None,
) -> Tuple[str, Optional[List[Dict[str, Any]]], bool]:
    """Send prompt with up to MAX_TOOL_ATTEMPTS attempts. Retries on:
      - session-invalid (biz_code 26) → invalidate + recreate session
      - empty response                    → escalate to EMPTY_RETRY_NUDGE
      - stall response (done / ok / …)    → escalate to EMPTY_RETRY_NUDGE
      - near-miss (marker present, parse failed) → escalate to NEAR_MISS_NUDGE

    Returns (full_text, tool_calls_or_None, ok). ok=False means every
    attempt failed and the caller should surface a visible fallback."""
    last_text = ""
    current_conv = conv
    current_session_id = session_id
    nudge_level = 0
    session_recreations = 0
    base_prompt = prompt
    attempts_used = 0

    for attempt in range(1, MAX_TOOL_ATTEMPTS + 1):
        attempts_used = attempt
        current_prompt = base_prompt
        if nudge_level == 1:
            current_prompt = base_prompt + EMPTY_RETRY_NUDGE
        elif nudge_level >= 2:
            current_prompt = base_prompt + NEAR_MISS_RETRY_NUDGE
        current_prompt = _truncate_to(current_prompt, MAX_FLATTENED_CHARS)

        await _pace()
        text, err = await _send_once_collect(
            current_conv, current_prompt, flags=flags, ref_ids=ref_ids,
        )
        last_text = text

        if err:
            if text.strip() and not _is_stall(text):
                partial_calls = _parse_tool_calls(text, valid_names)
                if partial_calls:
                    log.info(
                        "Using %d complete tool call(s) received before a "
                        "late stream error",
                        len(partial_calls),
                    )
                    return text, partial_calls, True
                if (
                    not _has_tool_marker(text)
                    and text.rstrip().endswith((".", "!", "?", ")", "]", "}"))
                ):
                    log.info(
                        "Using complete plain-text answer received before a "
                        "late stream error (len=%d)",
                        len(text),
                    )
                    return text, None, True

            # v4.0 — session-invalid handling comes first: this is a session
            # problem, not a prompt problem, so we must not burn nudge levels
            # on it and we must not re-send into the same poisoned conv.
            if (
                _is_session_invalid_error(err)
                and recreate_session is not None
            ):
                session_recreations += 1
                if session_recreations > MAX_SESSION_RECREATIONS:
                    log.error(
                        "Giving up: exceeded MAX_SESSION_RECREATIONS=%d "
                        "(session=%s)",
                        MAX_SESSION_RECREATIONS, current_session_id,
                    )
                    break
                log.warning(
                    "attempt %d/%d: session invalid (biz_code 26) "
                    "session=%s; recreating (%d/%d)",
                    attempt, MAX_TOOL_ATTEMPTS, current_session_id,
                    session_recreations, MAX_SESSION_RECREATIONS,
                )
                if pool is not None and lookup_key:
                    _invalidate_cached_session(
                        pool, lookup_key, current_session_id
                    )
                try:
                    _c, current_session_id, current_conv = await recreate_session()
                    if recovery_prompt_factory is not None:
                        base_prompt = recovery_prompt_factory()
                    else:
                        base_prompt = recovery_prompt or prompt
                except Exception as e:
                    log.exception(
                        "Failed to recreate session after invalid id: %s", e
                    )
                    break
                # Nudge level deliberately unchanged: the model never
                # produced an answer, so a nudge would be meaningless.
                continue

            if _is_rate_limit_error(err):
                _activate_rate_limit_cooldown()
            log.warning(
                "DeepSeek upstream error; bridge will not amplify client retries: "
                "%s (session=%s prompt_len=%d)",
                err, current_session_id, len(current_prompt),
            )
            break

        if not text.strip() or _is_stall(text):
            log.warning(
                "attempt %d/%d: empty/stall (len=%d, session=%s prompt_len=%d)",
                attempt, MAX_TOOL_ATTEMPTS, len(text), current_session_id,
                len(current_prompt),
            )
            if LOG_RAW_TOOL_OUTPUT and attempt == 1:
                log.warning(
                    "Sent prompt on empty attempt (first 2000 chars):\n%s",
                    _describe_codepoints(current_prompt[:2000]),
                )
            nudge_level = min(nudge_level + 1, 2)
            continue

        tool_calls = _parse_tool_calls(text, valid_names)
        if tool_calls:
            if attempt > 1:
                log.info("Retry succeeded with %d tool call(s)", len(tool_calls))
            return text, tool_calls, True

        if _has_tool_marker(text):
            log.warning(
                "attempt %d/%d: near-miss (marker present, parse failed) "
                "session=%s text_head=%r",
                attempt, MAX_TOOL_ATTEMPTS, current_session_id, text[:300],
            )
            nudge_level = min(nudge_level + 1, 2)
            continue

        # Plain text answer — acceptable.
        if attempt > 1:
            log.info("Retry succeeded with plain text (len=%d)", len(text))
        return text, None, True

    log.warning(
        "DeepSeek call failed after %d bridge attempt(s) for session=%s "
        "(last_text_len=%d)",
        attempts_used, current_session_id, len(last_text),
    )
    return last_text, None, False


# ── Anthropic <-> OpenAI conversion ─────────────────────────────────────────
def _anthropic_system_to_text(system: Optional[Union[str, List[Dict[str, Any]]]]) -> str:
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    parts: List[str] = []
    for blk in system:
        if isinstance(blk, dict) and blk.get("type") == "text":
            parts.append(blk.get("text") or "")
    return "\n".join(parts)


def _anthropic_content_to_parts(
    content: Any,
) -> Tuple[str, List[str], Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]]]:
    if isinstance(content, str):
        return content, [], None, None
    if not isinstance(content, list):
        return "", [], None, None

    text_parts: List[str] = []
    file_urls: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    tool_results: List[Dict[str, Any]] = []

    for blk in content:
        if not isinstance(blk, dict):
            continue
        t = blk.get("type")

        if t == "text":
            text_parts.append(blk.get("text") or "")

        elif t == "image":
            src = blk.get("source") or {}
            if src.get("type") == "base64":
                mime = src.get("media_type", "image/png")
                data = src.get("data", "")
                file_urls.append(f"data:{mime};base64,{data}")
            elif src.get("type") == "url":
                url = src.get("url")
                if url:
                    file_urls.append(url)

        elif t == "tool_use":
            tool_calls.append({
                "id": blk.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    "name": blk.get("name") or "",
                    "arguments": json.dumps(
                        blk.get("input") or {},
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ),
                },
            })

        elif t == "tool_result":
            tr_id = blk.get("tool_use_id") or "?"
            tr_content = blk.get("content")
            if isinstance(tr_content, list):
                sub: List[str] = []
                for sb in tr_content:
                    if isinstance(sb, dict) and sb.get("type") == "text":
                        sub.append(sb.get("text") or "")
                    elif isinstance(sb, str):
                        sub.append(sb)
                tr_content = "\n".join(sub)
            elif not isinstance(tr_content, str):
                tr_content = (
                    json.dumps(tr_content, ensure_ascii=False)
                    if tr_content is not None else ""
                )
            tool_results.append({"tool_call_id": tr_id, "content": tr_content})

    return (
        "\n".join(text_parts),
        file_urls,
        tool_calls or None,
        tool_results or None,
    )


def _anthropic_session_hint(req: AnthropicMessagesRequest) -> Optional[str]:
    """Return a caller-provided stable conversation identity, when present."""
    metadata = req.metadata or {}
    for key in ("session_id", "conversation_id", "user_id"):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _anthropic_to_openai(req: AnthropicMessagesRequest) -> ChatCompletionRequest:
    msgs: List[ChatMessage] = []

    sys_text = _anthropic_system_to_text(req.system)
    if sys_text:
        msgs.append(ChatMessage(role="system", content=sys_text))

    for m in req.messages:
        role = m.get("role")
        content = m.get("content")
        text, files, tool_calls, tool_results = _anthropic_content_to_parts(content)

        if role == "user":
            if tool_results:
                for tr in tool_results:
                    msgs.append(ChatMessage(
                        role="tool",
                        content=tr["content"],
                        tool_call_id=tr["tool_call_id"],
                    ))
            if text or files:
                parts: List[Dict[str, Any]] = []
                if text:
                    parts.append({"type": "text", "text": text})
                for u in files:
                    parts.append({"type": "image_url", "image_url": {"url": u}})
                c: MessageContent = parts if parts else text
                msgs.append(ChatMessage(role="user", content=c))

        elif role == "assistant":
            msgs.append(ChatMessage(
                role="assistant",
                content=text or None,
                tool_calls=tool_calls,
            ))

    tools_oai: Optional[List[Dict[str, Any]]] = None
    if req.tools:
        tools_oai = []
        for t in req.tools:
            tools_oai.append({
                "type": "function",
                "function": {
                    "name": t.get("name") or "",
                    "description": t.get("description") or "",
                    "parameters": t.get("input_schema") or {},
                },
            })

    return ChatCompletionRequest(
        model=req.model,
        messages=msgs,
        stream=False,
        tools=tools_oai,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )


def _openai_to_anthropic_response(
    openai_resp: Dict[str, Any],
    model: str,
) -> Dict[str, Any]:
    choice = openai_resp["choices"][0]
    msg = choice["message"]
    finish = choice.get("finish_reason") or "stop"

    content_blocks: List[Dict[str, Any]] = []
    text = msg.get("content")
    if text:
        content_blocks.append({"type": "text", "text": text})

    tool_calls = msg.get("tool_calls") or []
    for tc in tool_calls:
        fn = tc.get("function") or {}
        try:
            inp = json.loads(fn.get("arguments") or "{}")
        except Exception:
            inp = {"_raw": fn.get("arguments")}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:16]}",
            "name": fn.get("name") or "",
            "input": inp,
        })

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    stop_reason = {
        "tool_calls": "tool_use",
        "length": "max_tokens",
        "stop": "end_turn",
    }.get(finish, "end_turn")

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def _anth_event(event_type: str, data: Dict[str, Any]) -> str:
    return (
        f"event: {event_type}\n"
        f"data: {json.dumps(data, separators=(',', ':'), ensure_ascii=False)}\n\n"
    )


# ── SSE helpers ─────────────────────────────────────────────────────────────
def _sse_chunk(
    *,
    chunk_id: str,
    model: str,
    delta: Dict[str, Any],
    finish_reason: Optional[str] = None,
) -> str:
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {"index": 0, "delta": delta, "finish_reason": finish_reason}
        ],
    }
    return f"data: {json.dumps(payload, separators=(',', ':'), ensure_ascii=False)}\n\n"


def _sse_done() -> str:
    return "data: [DONE]\n\n"


def _next_or_none(gen):
    try:
        return next(gen)
    except StopIteration:
        return None


# ── Request planning ────────────────────────────────────────────────────────
def _is_tool_mode(req: ChatCompletionRequest) -> bool:
    if req.tools:
        return True
    return any(m.role == "tool" for m in req.messages)


def _plan_request(req: ChatCompletionRequest):
    normalized = _normalize(req.messages)
    if not normalized:
        raise HTTPException(400, "messages must not be empty")

    flags = _model_flags(req.model)
    tool_mode = _is_tool_mode(req)

    if tool_mode:
        prompt = _tool_prompt_for_request(req, resumed=False)
        if not prompt.strip():
            raise HTTPException(400, "no content to send")
        current = NormalizedMessage("user", prompt, [])
        return normalized, [], current, flags, tool_mode, prompt

    last_user_idx = None
    for i in range(len(normalized) - 1, -1, -1):
        if normalized[i].role == "user":
            last_user_idx = i
            break
    if last_user_idx is None:
        raise HTTPException(400, "no user message found")

    prior_msgs = normalized[:last_user_idx]
    current = normalized[last_user_idx]
    return normalized, prior_msgs, current, flags, tool_mode, current.text


# ── Session resolution ──────────────────────────────────────────────────────
async def _resolve_session(
    req: ChatCompletionRequest,
    headers: Dict[str, str],
    prior_msgs: List[NormalizedMessage],
    current: NormalizedMessage,
    flags: Dict[str, bool],
    pool: ClientPool,
    lookup_key: str,
    tool_mode: bool,
):
    prompt = current.text

    if tool_mode and TOOL_MODE_ALWAYS_NEW_SESSION:
        client, session_id, conv = await _make_fresh_session(pool, lookup_key)
        log.info("Tool-mode fresh session %s (key=%s)",
                 session_id, lookup_key[:24])
        return client, session_id, conv, prompt, flags

    session_id: Optional[str] = None
    cached, fresh = pool.control().cache.get(lookup_key, SESSION_TTL_S)
    # v4.0: treat the invalidation tombstone as "no cached session".
    if fresh and cached and cached != INVALID_SESSION_SENTINEL:
        session_id = cached

    if session_id:
        try:
            client = pool.for_session(session_id)
            conv = pool.conversation_for(session_id)
            if conv is None:
                conv = await anyio.to_thread.run_sync(
                    lambda: client.resume_conversation(
                        session_id, reconstruct=False
                    )
                )
                pool.remember_conversation(session_id, conv)
            if tool_mode:
                prompt = _tool_prompt_for_request(req, resumed=True)
            log.info("Resumed session %s (key=%s)", session_id, lookup_key[:24])
            return client, session_id, conv, prompt, flags
        except OSError as e:
            if session_id not in pool._warned_resume_fail:
                pool._warned_resume_fail.add(session_id)
                log.warning(
                    "Resume failed for %s (%s); creating new session",
                    session_id, e,
                )
        except Exception as e:
            log.warning("Resume failed for %s: %s; creating new session",
                        session_id, e)

    client, session_id, conv = await _make_fresh_session(pool, lookup_key)
    return client, session_id, conv, prompt, flags


# ── Streaming (OpenAI format) ───────────────────────────────────────────────
async def _stream_deepseek(
    req: ChatCompletionRequest,
    headers: Dict[str, str],
) -> AsyncIterator[str]:
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    model = req.model
    pool = get_pool(headers)

    try:
        _normalized, prior_msgs, current, flags, tool_mode, prompt = _plan_request(req)
    except HTTPException as e:
        yield _sse_chunk(chunk_id=chunk_id, model=model,
                         delta={"content": f"error: {e.detail}"},
                         finish_reason="error")
        yield _sse_done()
        return

    lookup_key = _compute_lookup_key(req, headers)
    lock = pool.lock_for(lookup_key)

    async with lock:
        try:
            client, session_id, conv, prompt, flags = await _resolve_session(
                req, headers, prior_msgs, current, flags, pool, lookup_key, tool_mode,
            )
        except Exception as e:
            log.exception("session resolution failed")
            yield _sse_chunk(chunk_id=chunk_id, model=model,
                             delta={"content": f"error: {e}"},
                             finish_reason="error")
            yield _sse_done()
            return

        pool.touch(session_id)

        try:
            ref_ids = await _resolve_file_ids(current.files, pool)
        except HTTPException as e:
            yield _sse_chunk(chunk_id=chunk_id, model=model,
                             delta={"content": f"error: {e.detail}"},
                             finish_reason="error")
            yield _sse_done()
            return

        if tool_mode:
            valid_names = _valid_tool_names(req)

            # v4.0: closure that recreates the session on demand.
            async def _recreate():
                return await _make_fresh_session(pool, lookup_key)

            full_text, tool_calls, ok = await _call_deepseek_with_retries(
                conv, prompt, flags=flags, ref_ids=ref_ids,
                session_id=session_id, valid_names=valid_names,
                pool=pool, lookup_key=lookup_key,
                recreate_session=_recreate,
                recovery_prompt_factory=lambda: _tool_prompt_for_request(
                    req, resumed=False
                ),
            )
            if LOG_RAW_TOOL_OUTPUT and full_text:
                log.info("RAW tool-mode output (first 800 chars):\n%s",
                         _describe_codepoints(full_text[:800]))

            if tool_calls:
                log.info("Parsed %d tool call(s): %s",
                         len(tool_calls),
                         ", ".join(tc["function"]["name"] for tc in tool_calls))
                yield _sse_chunk(
                    chunk_id=chunk_id, model=model,
                    delta={
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {"index": i, **tc} for i, tc in enumerate(tool_calls)
                        ],
                    },
                )
                yield _sse_chunk(chunk_id=chunk_id, model=model,
                                 delta={}, finish_reason="tool_calls")
            else:
                if not ok:
                    visible = EMPTY_VISIBLE_FALLBACK
                else:
                    visible = _strip_tool_xml(full_text) or full_text
                yield _sse_chunk(chunk_id=chunk_id, model=model,
                                 delta={"role": "assistant"})
                for i in range(0, len(visible), OUTPUT_CHUNK_CHARS):
                    yield _sse_chunk(
                        chunk_id=chunk_id, model=model,
                        delta={"content": visible[i:i + OUTPUT_CHUNK_CHARS]},
                    )
                yield _sse_chunk(chunk_id=chunk_id, model=model,
                                 delta={}, finish_reason="stop")

            yield _sse_done()
            return

        # Non-tool streaming
        yield _sse_chunk(chunk_id=chunk_id, model=model,
                         delta={"role": "assistant"})

        await _pace()
        vision_model = "vision" if ref_ids else None
        gen = conv.stream(
            prompt,
            thinking=flags["thinking"],
            search=flags["search"],
            prefetch=False,
            ref_file_ids=ref_ids,
            model_type=vision_model,
        )

        try:
            while True:
                ev: Optional[StreamEvent] = await anyio.to_thread.run_sync(
                    _next_or_none, gen
                )
                if ev is None:
                    break
                if ev.type == "thinking":
                    if EMIT_REASONING:
                        yield _sse_chunk(
                            chunk_id=chunk_id, model=model,
                            delta={"reasoning_content": ev.content},
                        )
                elif ev.type == "response":
                    yield _sse_chunk(
                        chunk_id=chunk_id, model=model,
                        delta={"content": ev.content},
                    )
                elif ev.type == "search" and EMIT_TOOLS:
                    yield _sse_chunk(chunk_id=chunk_id, model=model,
                                     delta={"search": ev.data})
                elif ev.type == "open" and EMIT_TOOLS:
                    yield _sse_chunk(chunk_id=chunk_id, model=model,
                                     delta={"open": ev.data})
                elif ev.type == "file" and EMIT_TOOLS:
                    yield _sse_chunk(chunk_id=chunk_id, model=model,
                                     delta={"file": ev.data})
                elif ev.type == "error":
                    if _is_rate_limit_error(str(ev.data)):
                        _activate_rate_limit_cooldown()
                    log.error("DeepSeek stream error: %s", ev.data)
                    yield _sse_chunk(chunk_id=chunk_id, model=model,
                                     delta={}, finish_reason="error")
                    yield _sse_done()
                    return
                elif ev.type == "done":
                    break
        except Exception as e:
            if _is_rate_limit_error(str(e)):
                _activate_rate_limit_cooldown()
            log.exception("stream failed")
            yield _sse_chunk(chunk_id=chunk_id, model=model,
                             delta={}, finish_reason="error")
            yield _sse_done()
            return

        yield _sse_chunk(chunk_id=chunk_id, model=model,
                         delta={}, finish_reason="stop")
        yield _sse_done()


# ── Streaming (Anthropic format) ────────────────────────────────────────────
async def _stream_anthropic(
    req: AnthropicMessagesRequest,
    headers: Dict[str, str],
) -> AsyncIterator[str]:
    msg_id = f"msg_{uuid.uuid4().hex}"
    model = req.model
    pool = get_pool(headers)

    def _err(etype: str, message: str) -> str:
        return _anth_event("error", {
            "type": "error",
            "error": {"type": etype, "message": message},
        })

    yield _anth_event("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })

    oai_req = _anthropic_to_openai(req)

    try:
        _normalized, prior_msgs, current, flags, tool_mode, prompt = _plan_request(oai_req)
    except HTTPException as e:
        yield _err("invalid_request_error", str(e.detail))
        return

    lookup_key = _compute_lookup_key(oai_req, headers)
    lock = pool.lock_for(lookup_key)

    async with lock:
        try:
            client, session_id, conv, prompt, flags = await _resolve_session(
                oai_req, headers, prior_msgs, current, flags, pool, lookup_key, tool_mode,
            )
        except Exception as e:
            log.exception("anthropic session resolution failed")
            yield _err("api_error", str(e))
            return

        pool.touch(session_id)

        try:
            ref_ids = await _resolve_file_ids(current.files, pool)
        except HTTPException as e:
            yield _err("invalid_request_error", str(e.detail))
            return

        if tool_mode:
            valid_names = _valid_tool_names(oai_req)

            async def _recreate():
                return await _make_fresh_session(pool, lookup_key)

            full_text, tool_calls, ok = await _call_deepseek_with_retries(
                conv, prompt, flags=flags, ref_ids=ref_ids,
                session_id=session_id, valid_names=valid_names,
                pool=pool, lookup_key=lookup_key,
                recreate_session=_recreate,
                recovery_prompt_factory=lambda: _tool_prompt_for_request(
                    oai_req, resumed=False
                ),
            )
            if LOG_RAW_TOOL_OUTPUT and full_text:
                log.info("RAW tool-mode output (first 800 chars):\n%s",
                         _describe_codepoints(full_text[:800]))

            if tool_calls:
                for idx, tc in enumerate(tool_calls):
                    fn = tc["function"]
                    try:
                        inp = json.loads(fn["arguments"])
                    except Exception:
                        inp = {"_raw": fn["arguments"]}
                    yield _anth_event("content_block_start", {
                        "type": "content_block_start",
                        "index": idx,
                        "content_block": {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": fn["name"],
                            "input": {},
                        },
                    })
                    yield _anth_event("content_block_delta", {
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(
                                inp, separators=(",", ":"), ensure_ascii=False
                            ),
                        },
                    })
                    yield _anth_event("content_block_stop", {
                        "type": "content_block_stop",
                        "index": idx,
                    })
                yield _anth_event("message_delta", {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                    "usage": {"output_tokens": 0},
                })
            else:
                if not ok:
                    visible = EMPTY_VISIBLE_FALLBACK
                else:
                    visible = _strip_tool_xml(full_text) or full_text
                yield _anth_event("content_block_start", {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                })
                if visible:
                    yield _anth_event("content_block_delta", {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": visible},
                    })
                yield _anth_event("content_block_stop", {
                    "type": "content_block_stop",
                    "index": 0,
                })
                yield _anth_event("message_delta", {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 0},
                })

            yield _anth_event("message_stop", {"type": "message_stop"})
            return

        # Non-tool streaming
        yield _anth_event("content_block_start", {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        })

        await _pace()
        vision_model = "vision" if ref_ids else None
        gen = conv.stream(
            prompt,
            thinking=flags["thinking"],
            search=flags["search"],
            prefetch=False,
            ref_file_ids=ref_ids,
            model_type=vision_model,
        )

        got_any = False
        try:
            while True:
                ev: Optional[StreamEvent] = await anyio.to_thread.run_sync(
                    _next_or_none, gen
                )
                if ev is None:
                    break
                if ev.type == "response":
                    if ev.content:
                        got_any = True
                        yield _anth_event("content_block_delta", {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": ev.content},
                        })
                elif ev.type == "thinking":
                    pass
                elif ev.type == "error":
                    if _is_rate_limit_error(str(ev.data)):
                        _activate_rate_limit_cooldown()
                    log.error("DeepSeek stream error: %s", ev.data)
                    yield _err("api_error", str(ev.data))
                    return
                elif ev.type == "done":
                    break
        except Exception as e:
            if _is_rate_limit_error(str(e)):
                _activate_rate_limit_cooldown()
            log.exception("anthropic stream failed")
            yield _err("api_error", str(e))
            return

        if not got_any:
            yield _anth_event("content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": EMPTY_VISIBLE_FALLBACK},
            })

        yield _anth_event("content_block_stop", {
            "type": "content_block_stop",
            "index": 0,
        })
        yield _anth_event("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 0},
        })
        yield _anth_event("message_stop", {"type": "message_stop"})


# ── Non-streaming (OpenAI) ──────────────────────────────────────────────────
async def _collect_deepseek(
    req: ChatCompletionRequest,
    headers: Dict[str, str],
) -> Dict[str, Any]:
    model = req.model
    pool = get_pool(headers)

    _normalized, prior_msgs, current, flags, tool_mode, prompt = _plan_request(req)
    lookup_key = _compute_lookup_key(req, headers)
    lock = pool.lock_for(lookup_key)

    async with lock:
        client, session_id, conv, prompt, flags = await _resolve_session(
            req, headers, prior_msgs, current, flags, pool, lookup_key, tool_mode,
        )
        pool.touch(session_id)

        ref_ids = await _resolve_file_ids(current.files, pool)

        if tool_mode:
            valid_names = _valid_tool_names(req)

            async def _recreate():
                return await _make_fresh_session(pool, lookup_key)

            full_text, tool_calls, ok = await _call_deepseek_with_retries(
                conv, prompt, flags=flags, ref_ids=ref_ids,
                session_id=session_id, valid_names=valid_names,
                pool=pool, lookup_key=lookup_key,
                recreate_session=_recreate,
                recovery_prompt_factory=lambda: _tool_prompt_for_request(
                    req, resumed=False
                ),
            )
            if LOG_RAW_TOOL_OUTPUT and full_text:
                log.info("RAW tool-mode output (first 800 chars):\n%s",
                         _describe_codepoints(full_text[:800]))
            if tool_calls:
                log.info("Parsed %d tool call(s): %s",
                         len(tool_calls),
                         ", ".join(tc["function"]["name"] for tc in tool_calls))
                message: Dict[str, Any] = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                }
                finish = "tool_calls"
            else:
                if not ok:
                    message = {"role": "assistant", "content": EMPTY_VISIBLE_FALLBACK}
                else:
                    visible = _strip_tool_xml(full_text) or full_text
                    message = {"role": "assistant", "content": visible}
                finish = "stop"
        else:
            await _pace()
            # v4.0: session-invalid recovery in the non-tool path.
            reply, _new_sid, _new_conv = await _send_sync_collect(
                conv, prompt,
                flags=flags, ref_ids=ref_ids,
                pool=pool, lookup_key=lookup_key, session_id=session_id,
            )
            content = (
                reply.text if reply.text and reply.text.strip()
                else EMPTY_VISIBLE_FALLBACK
            )
            message = {"role": "assistant", "content": content}
            if EMIT_REASONING and reply.thinking:
                message["reasoning_content"] = reply.thinking
            finish = "stop"

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {"index": 0, "message": message, "finish_reason": finish}
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }


# ── Background reaper ───────────────────────────────────────────────────────
async def _reaper():
    while True:
        try:
            await anyio.sleep(600.0)
            removed = 0
            for pool in get_all_pools():
                removed += await anyio.to_thread.run_sync(lambda p=pool: p.reap_idle())
            if removed:
                log.info("Reaped %d idle session clients", removed)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("reaper error: %s", e)


# ── FastAPI app ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        if DEEPSEEK_TOKEN:
            await anyio.to_thread.run_sync(lambda: get_pool().control())
    except Exception as e:
        log.warning("Pre-warm failed: %s", e)
    reaper_task = asyncio.create_task(_reaper())
    try:
        yield
    finally:
        reaper_task.cancel()
        try:
            await reaper_task
        except (asyncio.CancelledError, Exception):
            pass
        for pool in get_all_pools():
            pool.close_all()


app = FastAPI(title="DeepSeek Web → 9Router Bridge (v4.0)", lifespan=lifespan)


@app.get("/health")
async def health():
    active_sessions = sum(len(p._clients) for p in get_all_pools())
    return {
        "status": "ok",
        "service": "deepseek-bridge",
        "version": "4.0",
        "active_sessions": active_sessions,
        "active_pools": len(get_all_pools()),
        "tool_allowlist": sorted(TOOL_ALLOWLIST) if TOOL_ALLOWLIST else "*",
        "tool_mode_fresh_session": TOOL_MODE_ALWAYS_NEW_SESSION,
        "session_key_mode": SESSION_KEY_MODE,
        "log_raw_tool_output": LOG_RAW_TOOL_OUTPUT,
        "max_flattened_chars": MAX_FLATTENED_CHARS,
        "max_incremental_chars": MAX_INCREMENTAL_CHARS,
        "min_request_gap_s": MIN_REQUEST_GAP_S,
        "max_tool_attempts": MAX_TOOL_ATTEMPTS,
        "max_session_recreations": MAX_SESSION_RECREATIONS,
        "rate_limit_cooldown_s": RATE_LIMIT_COOLDOWN_S,
        "rate_limit_cooldown_remaining_s": max(
            0.0, _rate_limit_until - time.monotonic()
        ),
        "output_chunk_chars": OUTPUT_CHUNK_CHARS,
        "endpoints": ["/v1/chat/completions", "/v1/messages", "/v1/messages/count_tokens"],
    }


@app.get("/v1/models")
async def list_models():
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": "deepseek-chat", "object": "model", "created": now, "owned_by": "deepseek-web"},
            {"id": "deepseek-reasoner", "object": "model", "created": now, "owned_by": "deepseek-web"},
            {"id": "deepseek-search", "object": "model", "created": now, "owned_by": "deepseek-web"},
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    headers = {k.lower(): v for k, v in request.headers.items()}
    if req.stream:
        return StreamingResponse(
            _stream_deepseek(req, headers),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    return JSONResponse(content=await _collect_deepseek(req, headers))


@app.post("/v1/messages")
async def anthropic_messages(req: AnthropicMessagesRequest, request: Request):
    headers = {k.lower(): v for k, v in request.headers.items()}
    session_hint = _anthropic_session_hint(req)
    if session_hint:
        headers.setdefault("x-session-id", session_hint)
    if req.stream:
        return StreamingResponse(
            _stream_anthropic(req, headers),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    try:
        oai_req = _anthropic_to_openai(req)
        oai_resp = await _collect_deepseek(oai_req, headers)
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": str(e.detail)},
            },
        )
    except Exception as e:
        log.exception("anthropic non-stream failed")
        return JSONResponse(
            status_code=500,
            content={
                "type": "error",
                "error": {"type": "api_error", "message": str(e)},
            },
        )
    return JSONResponse(content=_openai_to_anthropic_response(oai_resp, req.model))


@app.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(req: AnthropicMessagesRequest):
    total = 0
    total += len(_anthropic_system_to_text(req.system)) // 4
    for m in req.messages:
        text, files, _tc, _tr = _anthropic_content_to_parts(m.get("content"))
        total += len(text) // 4
        total += len(files) * 1000
    return {"input_tokens": max(total, 1)}


# ── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("Starting DeepSeek bridge on %s:%d", BRIDGE_HOST, BRIDGE_PORT)
    log.info(
        "config: session_ttl=%ss file_ttl=%ss max_file=%sMB "
        "reasoning=%s tools=%s user_as_session=%s session_key_mode=%s "
        "idle_ttl=%ss "
        "impersonate=%s tool_fresh_session=%s raw_log=%s allowlist=%s "
        "max_flat_chars=%d max_incremental_chars=%d min_gap_s=%.2f "
        "max_attempts=%d max_recreations=%d "
        "rate_cooldown_s=%.1f output_chunk_chars=%d",
        SESSION_TTL_S, FILE_CACHE_TTL_S, MAX_FILE_BYTES // (1024 * 1024),
        EMIT_REASONING, EMIT_TOOLS, USE_USER_AS_SESSION, SESSION_KEY_MODE,
        IDLE_CLIENT_TTL_S,
        IMPERSONATE, TOOL_MODE_ALWAYS_NEW_SESSION, LOG_RAW_TOOL_OUTPUT,
        ",".join(sorted(TOOL_ALLOWLIST)) if TOOL_ALLOWLIST else "*",
        MAX_FLATTENED_CHARS, MAX_INCREMENTAL_CHARS, MIN_REQUEST_GAP_S,
        MAX_TOOL_ATTEMPTS,
        MAX_SESSION_RECREATIONS, RATE_LIMIT_COOLDOWN_S, OUTPUT_CHUNK_CHARS,
    )
    uvicorn.run(app, host=BRIDGE_HOST, port=BRIDGE_PORT,
                log_level="info", access_log=False)
