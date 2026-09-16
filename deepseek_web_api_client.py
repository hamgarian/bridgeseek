"""DeepSeek Web API Client.

Fixes in this revision
----------------------
* SSE parser no longer leaks metadata values (e.g. "DEEP_SEARCH",
  "FINISHED") into the THINK text stream. Metadata SET ops are
  filtered before the bare-string fallback.
* BATCH ops are now unwrapped: TOOL_SEARCH and TOOL_OPEN fragments
  arriving inside a `{"o":"BATCH"}` envelope are dispatched to their
  stream events (this is how DeepSeek ships search results).
* `last_fragment_type` tracking: bare content deltas are only applied
  when the last emitted fragment was THINK or RESPONSE, never after a
  TOOL_* or FILE fragment.
* `conversation_mode` ("DEFAULT" | "DEEP_SEARCH" | ...) is now surfaced
  on the Reply and in the `done` event metadata.
"""


from __future__ import annotations

import uuid
import argparse
import base64
import json
import logging
import mimetypes
import os
import struct
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Tuple

import anyio
from curl_cffi import requests as ccr, CurlMime
import wasmtime

from dotenv import load_dotenv

try:
    import httpx
except ImportError:
    httpx = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ORIGIN = "https://chat.deepseek.com"
API_BASE = f"{ORIGIN}/api/v0"
HIF_URL = "https://hif-leim.deepseek.com/query"
WASM_CDN_URL = "https://fe-static.deepseek.com/chat/static/sha3_wasm_bg.7b9ca65ddd.wasm"
WASM_FILE = "sha3_wasm_bg.7b9ca65ddd.wasm"
COOKIE_FILE = "cookies.txt"
SESSION_FILE = ".deepseek_sessions.json"
CACHE_FILE = ".deepseek_cache.json"

DEFAULT_DEVICE_ID = os.environ.get("DEEPSEEK_DEVICE_ID", uuid.uuid4().hex)
DEFAULT_MODEL_TYPE = "default"
DEFAULT_TIMEOUT_MS = 60_000
POW_EXPIRE_BUFFER_S = 15
MAX_RETRIES = 3
AUTO_RESUME_MAX = 2

AUTH_TTL_S = 300
DEVICE_TTL_S = 3600
SETTINGS_TTL_S = 300
HIF_TTL_S = 600

FILE_POLL_INTERVAL_S = 0.8
FILE_POLL_TIMEOUT_S = 180.0

SESSION_FETCH_MAX_PAGES = 8

_HISTORY_ENDPOINTS: Tuple[str, ...] = (
    "/chat/history_messages",
    "/chat_session/messages",
    "/chat/history_page",
)

load_dotenv()
log = logging.getLogger("deepseek")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DeepSeekError(Exception): pass
class ConfigurationError(DeepSeekError): pass
class AuthError(DeepSeekError): pass
class PowError(DeepSeekError): pass


class RateLimitError(DeepSeekError):
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class StreamError(DeepSeekError):
    """Unexpected end or malformed data in the SSE stream."""


class ResumableStreamError(StreamError):
    """Stream dropped after yielding content; safe to resume."""


class UploadError(DeepSeekError):
    """File upload or file-status polling failed."""


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class StreamEvent:
    type: str
    content: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchHit:
    status: Optional[str] = None
    summary: Optional[str] = None
    queries: List[Dict[str, Any]] = field(default_factory=list)
    results: List[Dict[str, Any]] = field(default_factory=list)
    stage_id: Optional[int] = None


@dataclass
class OpenedUrl:
    url: Optional[str] = None
    title: Optional[str] = None
    site_name: Optional[str] = None
    snippet: Optional[str] = None
    stage_id: Optional[int] = None


@dataclass
class Reply:
    text: str
    thinking: str
    request_message_id: Optional[int]
    response_message_id: Optional[int]
    model_type: str
    search_triggered: bool = False
    elapsed_secs: Optional[float] = None
    token_usage: Optional[int] = None
    ref_file_ids: List[str] = field(default_factory=list)
    search_results: List[SearchHit] = field(default_factory=list)
    opened_urls: List[OpenedUrl] = field(default_factory=list)
    conversation_mode: str = "DEFAULT"

    def __str__(self) -> str:
        return self.text


@dataclass
class HistoryMessage:
    id: int
    role: str
    content: str
    parent_id: Optional[int] = None
    inserted_at: Optional[float] = None
    file_ids: List[str] = field(default_factory=list)


@dataclass
class SessionRecord:
    session_id: str
    title: Optional[str] = None
    model_type: str = DEFAULT_MODEL_TYPE
    last_parent_id: Optional[int] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class RemoteSession:
    id: str
    title: Optional[str] = None
    title_type: Optional[str] = None
    model_type: str = DEFAULT_MODEL_TYPE
    pinned: bool = False
    updated_at: float = 0.0


@dataclass
class UploadedFile:
    id: str
    file_name: str
    file_size: int
    model_kind: str
    status: str
    is_image: bool
    token_usage: Optional[int] = None
    error_code: Optional[str] = None
    audit_result: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    inserted_at: Optional[float] = None
    updated_at: Optional[float] = None

    def __str__(self) -> str:
        return f"{self.file_name} ({self.id}) status={self.status}"


def _file_from_json(node: Dict[str, Any]) -> UploadedFile:
    return UploadedFile(
        id=node["id"],
        file_name=node.get("file_name", ""),
        file_size=int(node.get("file_size") or 0),
        model_kind=node.get("model_kind") or "",
        status=node.get("status") or "",
        is_image=bool(node.get("is_image")),
        token_usage=node.get("token_usage"),
        error_code=node.get("error_code"),
        audit_result=node.get("audit_result"),
        width=node.get("width"),
        height=node.get("height"),
        inserted_at=node.get("inserted_at"),
        updated_at=node.get("updated_at"),
    )


# ---------------------------------------------------------------------------
# Persistent TTL cache
# ---------------------------------------------------------------------------

class PersistentCache:
    def __init__(self, path: str = CACHE_FILE):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, Any]] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception as e:
                log.warning("Cache load failed: %s", e)

    def _save(self) -> None:
        if not self._dirty:
            return
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._data), encoding="utf-8")
        tmp.replace(self.path)
        self._dirty = False

    def get(self, key: str, ttl: float) -> Tuple[Any, bool]:
        entry = self._data.get(key)
        if not entry:
            return None, False
        return entry["v"], (time.time() - entry["t"]) < ttl

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = {"v": value, "t": time.time()}
            self._dirty = True
            self._save()

    def flush(self) -> None:
        with self._lock:
            self._save()


# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------

class SessionStore:
    def __init__(self, path: str = SESSION_FILE):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {"version": 1, "current": None, "sessions": {}}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception as e:
                log.warning("Session store load failed: %s", e)

    def _save(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def get(self, session_id: str) -> Optional[SessionRecord]:
        raw = self._data["sessions"].get(session_id)
        return SessionRecord(**raw) if raw else None

    def put(self, rec: SessionRecord) -> None:
        with self._lock:
            rec.updated_at = time.time()
            self._data["sessions"][rec.session_id] = asdict(rec)
            self._data["current"] = rec.session_id
            self._save()

    def put_bulk(self, records: List[SessionRecord]) -> None:
        with self._lock:
            for rec in records:
                if not rec.updated_at:
                    rec.updated_at = time.time()
                self._data["sessions"][rec.session_id] = asdict(rec)
            self._save()

    def current(self) -> Optional[SessionRecord]:
        sid = self._data.get("current")
        return self.get(sid) if sid else None

    def all(self) -> List[SessionRecord]:
        return [SessionRecord(**v) for v in self._data["sessions"].values()]


# ---------------------------------------------------------------------------
# Cookies
# ---------------------------------------------------------------------------

def load_cookies(path: str = COOKIE_FILE) -> str:
    p = Path(path)
    if p.exists():
        text = p.read_text(encoding="utf-8", errors="replace")
        netscape, simple = [], []
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                parts = line.split("\t")
                if len(parts) >= 7 and "deepseek.com" in parts[0].lower():
                    name, value = parts[5], "\t".join(parts[6:])
                    if value:
                        netscape.append(f"{name}={value}")
                    continue
            if "=" in line:
                simple.append(line)
        if netscape:
            return "; ".join(netscape)
        if simple:
            return "; ".join(simple)
        stripped = " ".join(text.split())
        if stripped:
            return stripped
    return os.environ.get("DEEPSEEK_COOKIES", "").strip()


# ---------------------------------------------------------------------------
# WASM PoW solver
# ---------------------------------------------------------------------------

class DeepSeekWasmSolver:
    def __init__(self, wasm_path: str = WASM_FILE):
        self.wasm_path = wasm_path
        self._ensure_module()
        self.engine = wasmtime.Engine()
        with open(self.wasm_path, "rb") as f:
            self.module = wasmtime.Module(self.engine, f.read())

    def _ensure_module(self):
        if not os.path.exists(self.wasm_path):
            log.info("Downloading WASM…")
            r = ccr.get(WASM_CDN_URL, headers={"User-Agent": "Mozilla/5.0"},
                        impersonate="chrome131", timeout=30)
            r.raise_for_status()
            Path(self.wasm_path).write_bytes(r.content)

    def _write_str(self, store, instance, text: str):
        raw = text.encode("utf-8")
        alloc = instance.exports(store)["__wbindgen_export_0"]
        ptr = alloc(store, len(raw), 1)
        mem_ptr = instance.exports(store)["memory"].data_ptr(store)
        for i, b in enumerate(raw):
            mem_ptr[ptr + i] = b
        return ptr, len(raw)

    def solve(self, challenge: str, salt: str, difficulty: int, expire_at: int) -> int:
        prefix = f"{salt}_{expire_at}_"
        store = wasmtime.Store(self.engine)
        linker = wasmtime.Linker(self.engine)
        linker.define_wasi()
        instance = linker.instantiate(store, self.module)
        stack_alloc = instance.exports(store)["__wbindgen_add_to_stack_pointer"]
        retptr = stack_alloc(store, -16)
        try:
            c_ptr, c_len = self._write_str(store, instance, challenge)
            p_ptr, p_len = self._write_str(store, instance, prefix)
            instance.exports(store)["wasm_solve"](
                store, retptr, c_ptr, c_len, p_ptr, p_len, float(difficulty))
            mem_ptr = instance.exports(store)["memory"].data_ptr(store)
            status = int.from_bytes(bytes(mem_ptr[retptr:retptr+4]), "little", signed=True)
            if status == 0:
                raise PowError("WASM solver found no valid nonce")
            return int(struct.unpack("<d", bytes(mem_ptr[retptr+8:retptr+16]))[0])
        finally:
            stack_alloc(store, 16)


# ---------------------------------------------------------------------------
# PoW cache
# ---------------------------------------------------------------------------

class PowCache:
    def __init__(self, fetcher: Callable[[str], Tuple[str, float]]):
        self._fetcher = fetcher
        self._cache: Dict[str, Tuple[str, float]] = {}
        self._pending: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    def get(self, path: str, *, prefetch: bool = True) -> str:
        with self._lock:
            entry = self._cache.pop(path, None)
        if entry is not None:
            header, expire_at = entry
            if expire_at - POW_EXPIRE_BUFFER_S > time.time():
                if prefetch:
                    self.prefetch(path)
                return header
        header, _ = self._fetcher(path)
        if prefetch:
            self.prefetch(path)
        return header

    def prefetch(self, path: str) -> None:
        with self._lock:
            existing = self._pending.get(path)
            if existing and existing.is_alive():
                return
            t = threading.Thread(target=self._do, args=(path,), daemon=True)
            self._pending[path] = t
            t.start()

    def _do(self, path: str) -> None:
        try:
            result = self._fetcher(path)
            with self._lock:
                self._cache[path] = result
        except Exception as e:
            log.debug("PoW prefetch failed: %s", e)
        finally:
            with self._lock:
                self._pending.pop(path, None)


# ---------------------------------------------------------------------------
# Frame logger (JSONL)
# ---------------------------------------------------------------------------

class FrameLogger:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.path, "a", encoding="utf-8")
        self._lock = threading.Lock()

    def _write(self, record: Dict[str, Any]) -> None:
        with self._lock:
            self._fp.write(json.dumps(record, separators=(",", ":"), default=str) + "\n")
            self._fp.flush()

    def raw(self, direction: str, line: str, event_name: Optional[str] = None) -> None:
        self._write({"t": time.time(), "dir": direction, "kind": "raw",
                     "event": event_name, "line": line})

    def frame(self, payload: Dict[str, Any]) -> None:
        self._write({"t": time.time(), "dir": "in", "kind": "frame", "frame": payload})

    def note(self, message: str, **extra: Any) -> None:
        rec = {"t": time.time(), "dir": "sys", "kind": "note", "message": message}
        rec.update(extra)
        self._write(rec)

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:
            pass

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()


# ---------------------------------------------------------------------------
# SSE parser
# ---------------------------------------------------------------------------

@dataclass
class SSEState:
    current_stage: str = "THINK"
    request_message_id: Optional[int] = None
    response_message_id: Optional[int] = None
    search_triggered: bool = False
    token_usage: Optional[int] = None
    elapsed_secs: Optional[float] = None
    conversation_mode: str = "DEFAULT"
    last_fragment_type: str = ""     # "THINK" | "RESPONSE" | "TOOL_SEARCH" | ...

    seen_thinking: str = ""
    seen_response: str = ""
    attempt_thinking: str = ""
    attempt_response: str = ""


class SSEParser:
    def __init__(
        self,
        state: Optional[SSEState] = None,
        *,
        thinking: bool = True,
        logger: Optional[FrameLogger] = None,
    ):
        self.state = state or SSEState(current_stage="THINK" if thinking else "RESPONSE")
        self.logger = logger
        self._buf = b""
        self._event_name: Optional[str] = None
        self._closed = False
        self._server_auto_resume = False

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def server_auto_resume(self) -> bool:
        return self._server_auto_resume

    def feed(self, chunk: bytes) -> List[StreamEvent]:
        out: List[StreamEvent] = []
        if self._closed:
            return out
        self._buf += chunk
        while b"\n" in self._buf:
            raw_line, self._buf = self._buf.split(b"\n", 1)
            raw_line = raw_line.rstrip(b"\r")
            out.extend(self._handle_line(raw_line))
            if self._closed:
                return out
        return out

    def finish(self) -> List[StreamEvent]:
        if self._buf:
            out = self._handle_line(self._buf.rstrip(b"\r"))
            self._buf = b""
            return out
        return []

    def _handle_line(self, raw_line: bytes) -> List[StreamEvent]:
        if not raw_line:
            self._event_name = None
            return []
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError:
            return []
        if self.logger:
            self.logger.raw("in", line, self._event_name)
        if line.startswith("event:"):
            name = line[6:].strip()
            self._event_name = name
            if name == "close":
                self._closed = True
            return []
        if not line.startswith("data:"):
            return []
        raw_json = line[5:].strip()
        if not raw_json or raw_json == "[DONE]":
            return []
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError:
            return []
        if self.logger:
            self.logger.frame(payload)
        return self._handle_payload(payload)

    # ---- payload dispatch --------------------------------------------

    def _handle_payload(self, payload: Dict[str, Any]) -> List[StreamEvent]:
        out: List[StreamEvent] = []

        # Named auxiliary SSE events
        if self._event_name == "update_file":
            out.append(StreamEvent(type="file_update", data=dict(payload)))
            return out
        if self._event_name == "ready" and "response_message_id" in payload:
            self.state.request_message_id = payload.get("request_message_id")
            self.state.response_message_id = payload["response_message_id"]
            out.append(StreamEvent(type="meta", data={
                "request_message_id": self.state.request_message_id,
                "response_message_id": self.state.response_message_id,
            }))
            return out
        if self._event_name == "close" or (
            "auto_resume" in payload and "click_behavior" in payload
        ):
            self._server_auto_resume = bool(payload.get("auto_resume"))
            return out

        # ready-style meta carried as a plain frame
        if "response_message_id" in payload:
            self.state.request_message_id = payload.get("request_message_id")
            self.state.response_message_id = payload["response_message_id"]
            out.append(StreamEvent(type="meta", data={
                "request_message_id": self.state.request_message_id,
                "response_message_id": self.state.response_message_id,
            }))
            return out

        code = payload.get("code")
        if code is not None and code != 0:
            out.append(StreamEvent(type="error",
                                   data={"code": code, "message": payload.get("msg", "")}))
            return out

        v = payload.get("v")
        p = payload.get("p", "") or ""
        o = payload.get("o")

        # ---- BATCH envelope: unwrap and dispatch each sub-op ----------
        if o == "BATCH" and isinstance(v, list):
            for item in v:
                if not isinstance(item, dict):
                    continue
                sub_p = item.get("p") or ""
                sub_o = item.get("o")
                sub_v = item.get("v")

                # fragment array appended
                if sub_p in ("fragments", "response/fragments") and isinstance(sub_v, list):
                    for frag in sub_v:
                        self._emit_fragment(frag, out, initial=True)
                    continue

                # text content delta for the last fragment
                if sub_p.endswith("/content") and isinstance(sub_v, str):
                    self._append_token(sub_v, out)
                    continue

                # status updates, elapsed_secs, etc → metadata, ignore
                if sub_p.endswith("/status"):
                    continue
                if sub_p.endswith("/elapsed_secs") and isinstance(sub_v, (int, float)):
                    self.state.elapsed_secs = float(sub_v)
                    continue
                if sub_p == "accumulated_token_usage":
                    self.state.token_usage = sub_v
                    continue
                if sub_p == "conversation_mode":
                    self.state.conversation_mode = str(sub_v)
                    continue
                if sub_p == "search_triggered":
                    self.state.search_triggered = bool(sub_v)
                    continue

                # unknown sub-op — drop
                if self.logger:
                    self.logger.note("unhandled BATCH sub-op",
                                     p=sub_p, o=sub_o, v=str(sub_v)[:80])
            return out

        # ---- initial snapshot / fragment append ----------------------
        if isinstance(v, dict) and "response" in v:
            obj = v["response"]
            if "message_id" in obj:
                self.state.response_message_id = obj["message_id"]
            if obj.get("search_triggered"):
                self.state.search_triggered = True
            if obj.get("conversation_mode"):
                self.state.conversation_mode = obj["conversation_mode"]
            for frag in obj.get("fragments") or []:
                self._emit_fragment(frag, out, initial=True)
            return out

        if p in ("response/fragments", "fragments") and isinstance(v, list):
            for item in v:
                self._emit_fragment(item, out, initial=True)
            return out

        # ---- metadata SET ops — MUST be filtered before text deltas ---
        if p in ("response/conversation_mode", "conversation_mode"):
            if isinstance(v, str):
                self.state.conversation_mode = v
            return out
        if p.endswith("/status"):
            return out
        if p.endswith("/elapsed_secs") and isinstance(v, (int, float)):
            self.state.elapsed_secs = float(v)
            return out
        if p.endswith("/search_triggered"):
            self.state.search_triggered = bool(v)
            return out
        if p in ("accumulated_token_usage", "response/accumulated_token_usage"):
            self.state.token_usage = v
            return out

        # ---- string deltas -------------------------------------------
        if isinstance(v, str):
            if p.endswith("/content"):
                self._append_token(v, out)
                return out
            if p.startswith("response/status"):
                return out
            if o == "APPEND":
                # explicit append of text
                if self.state.last_fragment_type in ("THINK", "RESPONSE"):
                    self._append_token(v, out)
                else:
                    if self.logger:
                        self.logger.note("dropping APPEND after non-text frag",
                                         last=self.state.last_fragment_type,
                                         v=v[:60])
                return out
            if not p and o is None:
                # bare delta — applies to most recent TEXT fragment only
                if self.state.last_fragment_type in ("THINK", "RESPONSE"):
                    self._append_token(v, out)
                else:
                    if self.logger:
                        self.logger.note("dropping bare delta after non-text frag",
                                         last=self.state.last_fragment_type,
                                         v=v[:60])
                return out
            return out

        # numeric/other bare values are metadata
        return out

    # ---- fragment dispatch -------------------------------------------

    def _emit_fragment(self, frag: Dict[str, Any], out: List[StreamEvent],
                       *, initial: bool) -> None:
        ft = frag.get("type")

        if ft == "FILE":
            self.state.last_fragment_type = "FILE"
            out.append(StreamEvent(type="file", data={"files": frag.get("files") or []}))
            return
        if ft == "TOOL_SEARCH":
            self.state.last_fragment_type = "TOOL_SEARCH"
            out.append(StreamEvent(type="search", data={
                "status": frag.get("status"),
                "content": frag.get("content"),
                "queries": frag.get("queries") or [],
                "results": frag.get("results") or [],
                "stage_id": frag.get("stage_id"),
            }))
            return
        if ft == "TOOL_OPEN":
            self.state.last_fragment_type = "TOOL_OPEN"
            out.append(StreamEvent(type="open", data={
                "status": frag.get("status"),
                "result": frag.get("result") or {},
                "reference": frag.get("reference") or {},
                "stage_id": frag.get("stage_id"),
            }))
            return

        if ft not in ("THINK", "RESPONSE"):
            return

        self.state.current_stage = ft
        self.state.last_fragment_type = ft
        content = frag.get("content") or ""
        if not content:
            return

        if initial:
            if ft == "THINK":
                self.state.attempt_thinking = content
            else:
                self.state.attempt_response = content
        else:
            if ft == "THINK":
                self.state.attempt_thinking += content
            else:
                self.state.attempt_response += content

        self._flush_delta(ft, out)

    def _append_token(self, token: str, out: List[StreamEvent]) -> None:
        stage = self.state.current_stage
        if stage == "THINK":
            self.state.attempt_thinking += token
        else:
            self.state.attempt_response += token
        self._flush_delta(stage, out)

    def _flush_delta(self, stage: str, out: List[StreamEvent]) -> None:
        if stage == "THINK":
            attempt = self.state.attempt_thinking
            seen = self.state.seen_thinking
            ev_type = "thinking"
        else:
            attempt = self.state.attempt_response
            seen = self.state.seen_response
            ev_type = "response"

        if attempt == seen:
            return

        if attempt.startswith(seen):
            delta = attempt[len(seen):]
            if stage == "THINK":
                self.state.seen_thinking = attempt
            else:
                self.state.seen_response = attempt
            out.append(StreamEvent(type=ev_type, content=delta))
            return

        if seen.startswith(attempt):
            return

        n = 0
        maxn = min(len(seen), len(attempt))
        while n < maxn and seen[n] == attempt[n]:
            n += 1
        log.debug(
            "SSE dedup resync on %s (prefix=%d, seen=%d, attempt=%d)",
            stage, n, len(seen), len(attempt),
        )
        delta = attempt[n:]
        if stage == "THINK":
            self.state.seen_thinking = attempt
        else:
            self.state.seen_response = attempt
        if delta:
            out.append(StreamEvent(type=ev_type, content=delta))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _parse_pow_challenge(ch: Dict[str, Any], target_path: str, answer: int) -> Tuple[str, float]:
    payload = {
        "algorithm": ch["algorithm"],
        "challenge": ch["challenge"],
        "salt": ch["salt"],
        "answer": answer,
        "signature": ch["signature"],
        "target_path": target_path,
    }
    header = base64.b64encode(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).decode("utf-8")
    return header, ch["expire_at"] / 1000.0


def _build_completion_body(
    *, session_id: str, parent_id: Optional[int],
    model_type: str, prompt: str,
    thinking: bool, search: bool,
    ref_file_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "chat_session_id": session_id,
        "parent_message_id": parent_id,
        "model_type": model_type if parent_id is None else None,
        "prompt": prompt,
        "ref_file_ids": list(ref_file_ids or []),
        "thinking_enabled": thinking,
        "search_enabled": search,
        "action": None,
        "preempt": False,
    }


def _flatten_settings(raw: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, entry in raw.items():
        out[name] = entry.get("value") if isinstance(entry, dict) and "value" in entry else entry
    return out


def _extract_messages(payload: Dict[str, Any]) -> List[HistoryMessage]:
    found: Dict[int, HistoryMessage] = {}

    def _content_for(role: str, node: Dict[str, Any]) -> str:
        frags = node.get("fragments")
        if not isinstance(frags, list):
            return ""
        wanted = "REQUEST" if role == "USER" else "RESPONSE"
        parts = []
        for f in frags:
            if not isinstance(f, dict):
                continue
            if f.get("type") == wanted and f.get("content"):
                parts.append(str(f["content"]))
        if parts:
            return "".join(parts)
        for f in frags:
            if isinstance(f, dict) and f.get("content"):
                parts.append(str(f["content"]))
        return "".join(parts)

    def _file_ids_for(node: Dict[str, Any]) -> List[str]:
        out: List[str] = []
        frags = node.get("fragments")
        if not isinstance(frags, list):
            return out
        for f in frags:
            if isinstance(f, dict) and f.get("type") == "FILE":
                for fobj in f.get("files") or []:
                    if isinstance(fobj, dict) and fobj.get("id"):
                        out.append(str(fobj["id"]))
        return out

    def _add(node: Dict[str, Any]) -> bool:
        mid = node.get("message_id")
        role = node.get("role")
        if not isinstance(mid, int) or not isinstance(role, str):
            return False
        role = role.upper()
        if role not in ("USER", "ASSISTANT"):
            return False
        found[mid] = HistoryMessage(
            id=mid,
            role=role,
            content=_content_for(role, node),
            parent_id=node.get("parent_id"),
            inserted_at=node.get("inserted_at"),
            file_ids=_file_ids_for(node),
        )
        return True

    biz = (payload.get("data") or {}).get("biz_data") or {}
    chat_messages = biz.get("chat_messages")
    if isinstance(chat_messages, list):
        for m in chat_messages:
            if isinstance(m, dict):
                _add(m)

    if not found:
        def visit(node: Any):
            if isinstance(node, dict):
                _add(node)
                for v in node.values():
                    visit(v)
            elif isinstance(node, list):
                for v in node:
                    visit(v)
        visit(payload)

    return sorted(found.values(), key=lambda m: m.id)


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _guess_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def _search_hit_from_event(data: Dict[str, Any]) -> SearchHit:
    return SearchHit(
        status=data.get("status"),
        summary=data.get("content"),
        queries=data.get("queries") or [],
        results=data.get("results") or [],
        stage_id=data.get("stage_id"),
    )


def _opened_from_event(data: Dict[str, Any]) -> OpenedUrl:
    result = data.get("result") or {}
    return OpenedUrl(
        url=result.get("url"),
        title=result.get("title"),
        site_name=result.get("site_name"),
        snippet=result.get("snippet"),
        stage_id=data.get("stage_id"),
    )


# ---------------------------------------------------------------------------
# ============================================================================
#  SYNC CLIENT
# ============================================================================
# ---------------------------------------------------------------------------

class DeepSeekClient:
    def __init__(
        self,
        token: str,
        *,
        cookies: Optional[str] = None,
        device_id: str = DEFAULT_DEVICE_ID,
        session_file: str = SESSION_FILE,
        cache_file: str = CACHE_FILE,
        frame_logger: Optional[FrameLogger] = None,
        prefetch_pow: bool = True,
    ):
        if not token:
            raise ConfigurationError("Bearer token required")
        self.device_id = device_id
        self.token = token
        self._bootstrapped = False
        self.settings: Dict[str, Any] = {}
        self._settings_fetched_at = 0.0
        self._hif_lock = threading.Lock()
        self._prefetch_pow = prefetch_pow
        self.frame_logger = frame_logger

        self.solver = DeepSeekWasmSolver()
        self.store = SessionStore(session_file)
        self.cache = PersistentCache(cache_file)

        self.session = ccr.Session(impersonate="chrome131")
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Origin": ORIGIN,
            "Referer": f"{ORIGIN}/",
            "x-client-platform": "web",
            "x-client-version": "2.5.0",
            "x-client-locale": "en_US",
            "x-client-bundle-id": "com.deepseek.chat",
            "x-client-timezone-offset": "25200",
            "x-device-id": self.device_id,
            "x-device-model": "",
        })

        cookie_header = cookies if cookies is not None else load_cookies()
        if cookie_header:
            self.session.headers["Cookie"] = cookie_header
            names = [c.split("=", 1)[0] for c in cookie_header.split("; ")]
            log.info("Loaded %d cookies: %s", len(names), ", ".join(names))
            if not any(n.startswith("CF_VERIFIED_DEVICE_") for n in names):
                log.warning("No CF_VERIFIED_DEVICE_* cookie — WAF may reject.")
        else:
            log.warning("No cookies loaded.")

        self.pow = PowCache(self._solve_pow_sync)

    @classmethod
    def from_env(cls, **kwargs) -> "DeepSeekClient":
        token = os.environ.get("DEEPSEEK_WEB_TOKEN")
        if not token:
            raise ConfigurationError("DEEPSEEK_WEB_TOKEN not set")
        return cls(token=token, **kwargs)

    # ----- bootstrap ---------------------------------------------------

    def bootstrap(self, *, force: bool = False) -> "DeepSeekClient":
        if self._bootstrapped and not force:
            return self
        t0 = time.perf_counter()

        _, fresh = self.cache.get("auth_ok_at", AUTH_TTL_S)
        if force or not fresh:
            r = self.session.get(f"{API_BASE}/users/current", timeout=15)
            r.raise_for_status()
            data = r.json()
            if data.get("code") != 0:
                raise AuthError(f"Auth failed: {data.get('msg')}")
            self.cache.set("auth_ok_at", time.time())
            name = data.get("data", {}).get("biz_data", {}).get("id_profile", {}).get("name", "?")
            log.info("Auth OK as %s", name)
        else:
            log.info("Auth cache hit")

        _, fresh = self.cache.get("device_check_at", DEVICE_TTL_S)
        if force or not fresh:
            r = self.session.post(
                f"{API_BASE}/users/auth_token/check_device",
                json={"device_id": self.device_id, "device_model": ""},
                timeout=15,
            )
            r.raise_for_status()
            self.cache.set("device_check_at", time.time())
            log.info("Device check OK")
        else:
            log.info("Device cache hit")

        settings, fresh = self.cache.get("settings", SETTINGS_TTL_S)
        if force or not fresh:
            self._fetch_settings(force=True)
            self.cache.set("settings", self.settings)
            log.info("Settings fetched (%d keys)", len(self.settings))
        else:
            self.settings = settings or {}
            self._settings_fetched_at = time.time()
            log.info("Settings cache hit (%d keys)", len(self.settings))

        self._bootstrapped = True
        log.info("Bootstrap complete in %.0fms",
                 (time.perf_counter() - t0) * 1000)
        return self

    def close(self) -> None:
        self.cache.flush()
        try:
            self.session.close()
        except Exception:
            pass

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()

    def _fetch_settings(self, scope: str = "main", *, force: bool = False) -> Dict[str, Any]:
        r = self.session.get(
            f"{API_BASE}/client/settings",
            params={"did": self.device_id, "scope": scope},
            timeout=15,
        )
        r.raise_for_status()
        raw = r.json().get("data", {}).get("biz_data", {}).get("settings", {})
        self.settings = _flatten_settings(raw)
        self._settings_fetched_at = time.time()
        return self.settings

    def _completion_timeout(self) -> float:
        ms = self.settings.get("completion_request_timeout_ms", DEFAULT_TIMEOUT_MS)
        try:
            return float(ms) / 1000.0 + 30.0
        except Exception:
            return (DEFAULT_TIMEOUT_MS / 1000.0) + 30.0

    # ----- PoW ---------------------------------------------------------

    def _solve_pow_sync(self, target_path: str) -> Tuple[str, float]:
        r = self.session.post(
            f"{API_BASE}/chat/create_pow_challenge",
            json={"target_path": target_path}, timeout=15,
        )
        r.raise_for_status()
        ch = r.json()["data"]["biz_data"]["challenge"]
        t0 = time.perf_counter()
        answer = self.solver.solve(
            ch["challenge"], ch["salt"], ch["difficulty"], ch["expire_at"])
        log.info("PoW solved in %.1fms (nonce=%s, path=%s)",
                 (time.perf_counter() - t0) * 1000, answer, target_path)
        return _parse_pow_challenge(ch, target_path, answer)

    # ----- HIF ---------------------------------------------------------

    def _fetch_hif_token(self) -> Optional[str]:
        cached, fresh = self.cache.get("hif", HIF_TTL_S)
        if fresh and cached:
            return cached
        try:
            r = self.session.get(HIF_URL, timeout=10)
            if r.status_code != 200:
                return None
            token = r.json().get("data", {}).get("biz_data", {}).get("value")
            if token:
                self.cache.set("hif", token)
            return token
        except Exception as e:
            log.warning("HIF fetch failed: %s", e)
            return None

    # ----- files -------------------------------------------------------

    def upload_file(
        self,
        path: "str | Path",
        *,
        poll: bool = True,
        timeout_s: float = FILE_POLL_TIMEOUT_S,
    ) -> UploadedFile:
        self.bootstrap()
        p = Path(path)
        if not p.exists():
            raise UploadError(f"file not found: {p}")

        pow_header = self.pow.get("/api/v0/file/upload_file")
        mime = _guess_mime(p)
        file_bytes = p.read_bytes()
        log.info("Uploading %s (%d bytes, %s)", p.name, len(file_bytes), mime)

        mp = CurlMime()
        mp.addpart(
            name="file",
            content_type=mime,
            filename=p.name,
            data=file_bytes,
        )

        saved_ct = self.session.headers.pop("Content-Type", None)
        try:
            r = self.session.post(
                f"{API_BASE}/file/upload_file",
                multipart=mp,
                headers={
                    "x-ds-pow-response": pow_header,
                    "Accept": "*/*",
                },
                timeout=300,
            )
        finally:
            mp.close()
            if saved_ct is not None:
                self.session.headers["Content-Type"] = saved_ct

        if r.status_code != 200:
            body = r.content[:2000]
            try:
                body_text = body.decode("utf-8", "replace")
            except Exception:
                body_text = repr(body)
            log.error("upload HTTP %s: %s", r.status_code, body_text)
            raise UploadError(
                f"upload HTTP {r.status_code}: {body_text[:400]}"
            )

        payload = r.json()
        if payload.get("code") != 0:
            raise UploadError(
                f"upload code={payload.get('code')} msg={payload.get('msg')}"
            )
        biz = payload.get("data", {}).get("biz_data") or {}
        f = _file_from_json(biz)
        log.info("Uploaded %s -> %s (status=%s)", p.name, f.id, f.status)

        if poll and f.status not in ("SUCCESS", "FAILED"):
            f = self.wait_for_file(f.id, timeout_s=timeout_s)
        return f

    def fetch_files(self, file_ids: List[str]) -> List[UploadedFile]:
        if not file_ids:
            return []
        self.bootstrap()
        r = self.session.get(
            f"{API_BASE}/file/fetch_files",
            params={"file_ids": ",".join(file_ids)},
            timeout=15,
        )
        r.raise_for_status()
        payload = r.json()
        if payload.get("code") != 0:
            raise UploadError(
                f"fetch_files code={payload.get('code')} msg={payload.get('msg')}"
            )
        files = payload.get("data", {}).get("biz_data", {}).get("files") or []
        return [_file_from_json(f) for f in files]

    def wait_for_file(
        self,
        file_id: str,
        *,
        timeout_s: float = FILE_POLL_TIMEOUT_S,
        interval_s: float = FILE_POLL_INTERVAL_S,
    ) -> UploadedFile:
        deadline = time.time() + timeout_s
        last: Optional[UploadedFile] = None
        while time.time() < deadline:
            files = self.fetch_files([file_id])
            if not files:
                raise UploadError(f"file {file_id} vanished while polling")
            last = files[0]
            if last.status in ("SUCCESS", "FAILED"):
                if last.status == "FAILED":
                    raise UploadError(
                        f"file {file_id} failed: error_code={last.error_code}"
                    )
                log.info("File %s ready (audit=%s, tokens=%s)",
                         file_id, last.audit_result, last.token_usage)
                return last
            time.sleep(interval_s)
        raise UploadError(
            f"file {file_id} still {last.status if last else '?'} after {timeout_s}s"
        )

    # ----- history -----------------------------------------------------

    def fetch_history(self, session_id: str, *, max_pages: int = 3) -> List[HistoryMessage]:
        self.bootstrap()
        cached_path, _ = self.cache.get("history_path", 10 ** 9)
        candidates = ([cached_path] if cached_path else []) + [
            p for p in _HISTORY_ENDPOINTS if p != cached_path
        ]
        last_err: Optional[Exception] = None
        for path in candidates:
            try:
                messages = self._fetch_history_via(path, session_id, max_pages)
                if path != cached_path:
                    self.cache.set("history_path", path)
                    log.info("History endpoint working: %s", path)
                return messages
            except Exception as e:
                log.debug("History via %s failed: %s", path, e)
                last_err = e
        log.warning("All history endpoints failed; last error: %s", last_err)
        return []

    def _history_request(
        self, path: str, params: Dict[str, Any], *, force_post: bool = False
    ) -> Optional[Dict[str, Any]]:
        url = f"{API_BASE}{path}"
        if force_post:
            r = self.session.post(url, json=params, timeout=15)
        else:
            r = self.session.get(url, params=params, timeout=15)
        if r.status_code != 200:
            return None
        try:
            return r.json()
        except Exception:
            return None

    def _fetch_history_via(
        self, path: str, session_id: str, max_pages: int
    ) -> List[HistoryMessage]:
        messages: List[HistoryMessage] = []
        cursor: Optional[str] = None

        for _ in range(max_pages):
            params: Dict[str, Any] = {"chat_session_id": session_id}
            if cursor:
                params["lte_cursor.id"] = cursor

            payload = self._history_request(path, params)
            if payload is None:
                raise StreamError(f"history {path} returned no JSON")
            if payload.get("code") != 0:
                raise StreamError(
                    f"history code={payload.get('code')}: {payload.get('msg')}"
                )
            page = _extract_messages(payload)
            if not page:
                break
            messages = page + messages

            biz = payload.get("data", {}).get("biz_data", {}) or {}
            next_cursor = (
                biz.get("cursor")
                or biz.get("next_cursor")
                or (biz.get("page") or {}).get("cursor")
            )
            if not next_cursor:
                break
            cursor = str(next_cursor)

        return messages

    def reconstruct_parent_id(self, session_id: str) -> Optional[int]:
        try:
            r = self.session.get(
                f"{API_BASE}/chat/history_messages",
                params={"chat_session_id": session_id},
                timeout=15,
            )
            if r.status_code == 200:
                biz = (r.json().get("data") or {}).get("biz_data") or {}
                cmid = (biz.get("chat_session") or {}).get("current_message_id")
                if isinstance(cmid, int):
                    log.info("parent_id from current_message_id = %s", cmid)
                    return cmid
        except Exception as e:
            log.debug("current_message_id fast path failed: %s", e)

        messages = self.fetch_history(session_id)
        for m in reversed(messages):
            if m.role == "ASSISTANT":
                return m.id
        return None

    # ----- remote session listing -------------------------------------

    def fetch_remote_sessions(
        self,
        *,
        max_pages: int = SESSION_FETCH_MAX_PAGES,
    ) -> List[RemoteSession]:
        self.bootstrap()
        out: List[RemoteSession] = []
        seen_ids: set = set()

        for pinned_flag in (False, True):
            cursor_updated_at: Optional[float] = None
            for page_idx in range(max_pages):
                params: Dict[str, Any] = {
                    "lte_cursor.pinned": "true" if pinned_flag else "false",
                }
                if cursor_updated_at is not None:
                    params["lte_cursor.updated_at"] = str(cursor_updated_at)

                r = self.session.get(
                    f"{API_BASE}/chat_session/fetch_page",
                    params=params,
                    timeout=15,
                )
                if r.status_code != 200:
                    log.warning("fetch_page HTTP %s (pinned=%s page=%d)",
                                r.status_code, pinned_flag, page_idx)
                    break
                payload = r.json()
                if payload.get("code") != 0:
                    log.warning("fetch_page code=%s msg=%s",
                                payload.get("code"), payload.get("msg"))
                    break

                biz = payload.get("data", {}).get("biz_data") or {}
                sessions = biz.get("chat_sessions") or []
                if not sessions:
                    break

                for s in sessions:
                    sid = s.get("id")
                    if not sid or sid in seen_ids:
                        continue
                    seen_ids.add(sid)
                    out.append(RemoteSession(
                        id=sid,
                        title=s.get("title"),
                        title_type=s.get("title_type"),
                        model_type=s.get("model_type") or DEFAULT_MODEL_TYPE,
                        pinned=bool(s.get("pinned", pinned_flag)),
                        updated_at=float(s.get("updated_at") or 0.0),
                    ))

                if not biz.get("has_more"):
                    break
                last_updated = sessions[-1].get("updated_at")
                if not last_updated:
                    break
                cursor_updated_at = float(last_updated)

        log.info("Fetched %d remote sessions", len(out))
        return out

    def sync_sessions(self) -> Dict[str, int]:
        remote = self.fetch_remote_sessions()
        added = 0
        updated = 0
        new_records: List[SessionRecord] = []

        for s in remote:
            local = self.store.get(s.id)
            if local is None:
                new_records.append(SessionRecord(
                    session_id=s.id,
                    title=s.title,
                    model_type=s.model_type,
                    last_parent_id=None,
                    created_at=s.updated_at or time.time(),
                    updated_at=s.updated_at or time.time(),
                ))
                added += 1
            else:
                if local.title != s.title or local.model_type != s.model_type:
                    updated += 1
                local.title = s.title
                local.model_type = s.model_type
                new_records.append(local)

        self.store.put_bulk(new_records)
        log.info("Session sync: added=%d updated=%d total_remote=%d",
                 added, updated, len(remote))
        return {
            "added": added,
            "updated": updated,
            "total_remote": len(remote),
        }

    # ----- sessions ----------------------------------------------------

    def new_conversation(self, *, model_type: str = DEFAULT_MODEL_TYPE) -> "Conversation":
        self.bootstrap()
        r = self.session.post(f"{API_BASE}/chat_session/create", json={}, timeout=15)
        r.raise_for_status()
        sid = r.json()["data"]["biz_data"]["chat_session"]["id"]
        self.session.headers["Referer"] = f"{ORIGIN}/a/chat/s/{sid}"
        rec = SessionRecord(session_id=sid, model_type=model_type)
        self.store.put(rec)
        log.info("Created session %s", sid)
        return Conversation(self, rec)

    def resume_conversation(
        self, session_id: str, *, reconstruct: bool = True
    ) -> "Conversation":
        self.bootstrap()
        rec = self.store.get(session_id) or SessionRecord(session_id=session_id)
        if rec.last_parent_id is None and reconstruct:
            log.info("parent_id unknown; reconstructing from history…")
            pid = self.reconstruct_parent_id(session_id)
            if pid is not None:
                rec.last_parent_id = pid
                log.info("Reconstructed parent_id = %s", pid)
        self.store.put(rec)
        self.session.headers["Referer"] = f"{ORIGIN}/a/chat/s/{session_id}"
        return Conversation(self, rec)

    def conversation(self, session_id: Optional[str] = None, *, create: bool = True) -> "Conversation":
        if session_id:
            return self.resume_conversation(session_id)
        cur = self.store.current()
        if cur is not None:
            return self.resume_conversation(cur.session_id)
        if create:
            return self.new_conversation()
        raise ConfigurationError("No session and create=False")

    def sessions(self) -> List[SessionRecord]:
        return sorted(self.store.all(), key=lambda r: r.updated_at, reverse=True)

    def ask(self, prompt: str, *, thinking: bool = True, search: bool = True,
            new_session: bool = False,
            files: Optional[List[str]] = None) -> str:
        conv = self.new_conversation() if new_session else self.conversation()
        ref_ids = self._prepare_files(files) if files else []
        return conv.send(prompt, thinking=thinking, search=search,
                         prefetch=False, ref_file_ids=ref_ids).text

    def _prepare_files(self, paths: List[str]) -> List[str]:
        ids: List[str] = []
        for p in paths:
            f = self.upload_file(p, poll=True)
            ids.append(f.id)
        return ids


class Conversation:
    def __init__(self, client: DeepSeekClient, record: SessionRecord):
        self.client = client
        self.record = record
        self.pending_file_ids: List[str] = []

    @property
    def session_id(self) -> str: return self.record.session_id
    @property
    def model_type(self) -> str: return self.record.model_type

    def send(self, prompt: str, *, thinking: bool = True, search: bool = True,
             model_type: Optional[str] = None, prefetch: bool = True,
             ref_file_ids: Optional[List[str]] = None) -> Reply:
        thinking_parts, text_parts = [], []
        meta: Dict[str, Any] = {}
        searches: List[SearchHit] = []
        opened: List[OpenedUrl] = []
        used_ids: List[str] = list(ref_file_ids) if ref_file_ids else list(self.pending_file_ids)

        for ev in self.stream(prompt, thinking=thinking, search=search,
                              model_type=model_type, prefetch=prefetch,
                              ref_file_ids=used_ids):
            if ev.type == "thinking":
                thinking_parts.append(ev.content)
            elif ev.type == "response":
                text_parts.append(ev.content)
            elif ev.type == "meta":
                meta.update(ev.data)
            elif ev.type == "search":
                searches.append(_search_hit_from_event(ev.data))
            elif ev.type == "open":
                opened.append(_opened_from_event(ev.data))
            elif ev.type == "error":
                raise StreamError(ev.data.get("message", "stream error"))

        self.pending_file_ids = []
        return Reply(
            text="".join(text_parts).strip(),
            thinking="".join(thinking_parts).strip(),
            request_message_id=meta.get("request_message_id"),
            response_message_id=meta.get("response_message_id"),
            model_type=self.record.model_type,
            search_triggered=meta.get("search_triggered", False) or bool(searches),
            elapsed_secs=meta.get("elapsed_secs"),
            token_usage=meta.get("token_usage"),
            ref_file_ids=used_ids,
            search_results=searches,
            opened_urls=opened,
            conversation_mode=meta.get("conversation_mode", "DEFAULT"),
        )

    def stream(self, prompt: str, *, thinking: bool = True, search: bool = True,
               model_type: Optional[str] = None, prefetch: bool = True,
               ref_file_ids: Optional[List[str]] = None) -> Iterator[StreamEvent]:
        if model_type:
            self.record.model_type = model_type
        state = SSEState(current_stage="THINK" if thinking else "RESPONSE")
        last_err: Optional[Exception] = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                yield from self._stream_once(
                    prompt, thinking=thinking, search=search,
                    prefetch=prefetch, state=state, attempt=attempt,
                    ref_file_ids=ref_file_ids,
                )
                return
            except ResumableStreamError as e:
                last_err = e
                if not state.seen_thinking and not state.seen_response:
                    time.sleep(1.0)
                    continue
                if attempt > AUTO_RESUME_MAX:
                    break
                log.warning("Stream dropped after partial output; auto-resuming (attempt %d)",
                            attempt)
                time.sleep(0.5)
                continue
            except RateLimitError as e:
                last_err = e
                wait = e.retry_after or (2 ** attempt)
                log.warning("Rate limited, sleeping %.1fs", wait)
                time.sleep(wait)
            except StreamError as e:
                last_err = e
                log.warning("Stream error: %s", e)
                time.sleep(1.0)

        yield StreamEvent(type="error", data={"message": f"retries exhausted: {last_err}"})

    def _stream_once(self, prompt, *, thinking, search, prefetch, state: SSEState,
                     attempt: int = 1, ref_file_ids: Optional[List[str]] = None):
        target_path = "/api/v0/chat/completion"
        pow_header = self.client.pow.get(target_path, prefetch=prefetch)
        hif_token = self.client._fetch_hif_token()

        headers = {"x-ds-pow-response": pow_header}
        if hif_token:
            headers["x-hif-leim"] = hif_token

        body = _build_completion_body(
            session_id=self.record.session_id,
            parent_id=self.record.last_parent_id,
            model_type=self.record.model_type,
            prompt=prompt, thinking=thinking, search=search,
            ref_file_ids=ref_file_ids,
        )
        url = f"{ORIGIN}{target_path}"

        state.attempt_thinking = ""
        state.attempt_response = ""
        state.current_stage = "THINK" if thinking else "RESPONSE"
        # NOTE: last_fragment_type is intentionally NOT reset across resume
        # attempts — it carries the knowledge of what the last emitted
        # fragment was, which is useful for correct bare-delta attribution
        # after a resume.

        if self.client.frame_logger:
            self.client.frame_logger.note(
                "completion-request", url=url, parent_id=self.record.last_parent_id,
                thinking=thinking, search=search, attempt=attempt,
                ref_file_ids=list(ref_file_ids or []),
                resumed=bool(state.seen_thinking or state.seen_response),
            )

        resp = self.client.session.post(
            url, json=body, headers=headers, stream=True,
            timeout=self.client._completion_timeout(), allow_redirects=False,
        )
        try:
            resp.raw.decode_content = True
        except Exception:
            pass

        if resp.status_code == 429:
            retry = resp.headers.get("Retry-After")
            raise RateLimitError("429", retry_after=_parse_retry_after(retry))
        if resp.status_code >= 500:
            _ = resp.content
            raise StreamError(f"server {resp.status_code}")

        ct = resp.headers.get("Content-Type", "")
        if "text/event-stream" not in ct:
            body_bytes = b""
            try:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        body_bytes += chunk
            except Exception:
                pass
            snippet = body_bytes[:400].decode("utf-8", "replace")
            log.error("Non-SSE (%s, %d bytes): %s", ct, len(body_bytes), snippet)
            yield StreamEvent(type="error", data={
                "status": resp.status_code, "content_type": ct, "body": snippet})
            return

        resp.raise_for_status()

        parser = SSEParser(state=state, thinking=thinking,
                           logger=self.client.frame_logger)
        partial = False
        try:
            for chunk in resp.iter_content(chunk_size=4096):
                if not chunk:
                    continue
                for ev in parser.feed(chunk):
                    if ev.type in ("thinking", "response"):
                        partial = True
                    yield ev
                if parser.closed:
                    break
        except Exception as e:
            if partial or state.seen_thinking or state.seen_response:
                raise ResumableStreamError(f"stream interrupted: {e}") from e
            raise StreamError(f"stream failed: {e}") from e

        for ev in parser.finish():
            yield ev

        if state.response_message_id is not None:
            self.record.last_parent_id = state.response_message_id
            self.client.store.put(self.record)

        if parser.server_auto_resume and not partial:
            raise ResumableStreamError("server signalled auto_resume")

        yield StreamEvent(type="done", data={
            "request_message_id": state.request_message_id,
            "response_message_id": state.response_message_id,
            "search_triggered": state.search_triggered,
            "elapsed_secs": state.elapsed_secs,
            "token_usage": state.token_usage,
            "resume_attempts": attempt,
            "conversation_mode": state.conversation_mode,
        })


# ---------------------------------------------------------------------------
# ============================================================================
#  ASYNC CLIENT
# ============================================================================
# ---------------------------------------------------------------------------

class AsyncDeepSeekClient:
    def __init__(
        self,
        token: str,
        *,
        cookies: Optional[str] = None,
        device_id: str = DEFAULT_DEVICE_ID,
        session_file: str = SESSION_FILE,
        cache_file: str = CACHE_FILE,
        frame_logger: Optional[FrameLogger] = None,
        prefetch_pow: bool = True,
    ):
        if httpx is None:
            raise ConfigurationError("httpx not installed: pip install 'httpx[http2]'")
        if not token:
            raise ConfigurationError("Bearer token required")
        self.device_id = device_id
        self.token = token
        self._bootstrapped = False
        self.settings: Dict[str, Any] = {}
        self._settings_fetched_at = 0.0
        self._prefetch_pow = prefetch_pow
        self.frame_logger = frame_logger

        self.solver = DeepSeekWasmSolver()
        self.store = SessionStore(session_file)
        self.cache = PersistentCache(cache_file)

        cookie_header = cookies if cookies is not None else load_cookies()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Origin": ORIGIN,
            "Referer": f"{ORIGIN}/",
            "x-client-platform": "web",
            "x-client-version": "2.5.0",
            "x-client-locale": "en_US",
            "x-client-bundle-id": "com.deepseek.chat",
            "x-client-timezone-offset": "25200",
            "x-device-id": self.device_id,
            "x-device-model": "",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
        }
        if cookie_header:
            headers["Cookie"] = cookie_header
            log.info("Loaded %d cookies", len(cookie_header.split("; ")))

        self._client = httpx.AsyncClient(
            http2=True, timeout=60.0, headers=headers, follow_redirects=False,
        )

        self._sync_session = ccr.Session(impersonate="chrome131")
        self._sync_session.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Origin": ORIGIN,
            "Referer": f"{ORIGIN}/",
            "x-client-platform": "web",
            "x-client-version": "2.5.0",
            "x-client-locale": "en_US",
            "x-client-bundle-id": "com.deepseek.chat",
            "x-client-timezone-offset": "25200",
            "x-device-id": self.device_id,
            "x-device-model": "",
        })
        if cookie_header:
            self._sync_session.headers["Cookie"] = cookie_header

        self.pow = PowCache(self._solve_pow_sync)

    @classmethod
    def from_env(cls, **kwargs) -> "AsyncDeepSeekClient":
        token = os.environ.get("DEEPSEEK_WEB_TOKEN")
        if not token:
            raise ConfigurationError("DEEPSEEK_WEB_TOKEN not set")
        return cls(token=token, **kwargs)

    async def __aenter__(self):
        await self.bootstrap()
        return self

    async def __aexit__(self, *exc):
        await self.close()

    async def close(self) -> None:
        self.cache.flush()
        try:
            await self._client.aclose()
        except Exception:
            pass
        try:
            self._sync_session.close()
        except Exception:
            pass

    async def bootstrap(self, *, force: bool = False) -> "AsyncDeepSeekClient":
        if self._bootstrapped and not force:
            return self
        t0 = time.perf_counter()

        _, fresh = self.cache.get("auth_ok_at", AUTH_TTL_S)
        if force or not fresh:
            r = await self._client.get(f"{API_BASE}/users/current")
            r.raise_for_status()
            data = r.json()
            if data.get("code") != 0:
                raise AuthError(f"Auth failed: {data.get('msg')}")
            self.cache.set("auth_ok_at", time.time())
            name = data.get("data", {}).get("biz_data", {}).get("id_profile", {}).get("name", "?")
            log.info("Auth OK as %s", name)
        else:
            log.info("Auth cache hit")

        _, fresh = self.cache.get("device_check_at", DEVICE_TTL_S)
        if force or not fresh:
            r = await self._client.post(
                f"{API_BASE}/users/auth_token/check_device",
                json={"device_id": self.device_id, "device_model": ""},
            )
            r.raise_for_status()
            self.cache.set("device_check_at", time.time())
            log.info("Device check OK")
        else:
            log.info("Device cache hit")

        settings, fresh = self.cache.get("settings", SETTINGS_TTL_S)
        if force or not fresh:
            await self._fetch_settings(force=True)
            self.cache.set("settings", self.settings)
            log.info("Settings fetched (%d keys)", len(self.settings))
        else:
            self.settings = settings or {}
            self._settings_fetched_at = time.time()
            log.info("Settings cache hit (%d keys)", len(self.settings))

        self._bootstrapped = True
        log.info("Bootstrap complete in %.0fms",
                 (time.perf_counter() - t0) * 1000)
        return self

    async def _fetch_settings(self, scope: str = "main", *, force: bool = False) -> Dict[str, Any]:
        r = await self._client.get(
            f"{API_BASE}/client/settings",
            params={"did": self.device_id, "scope": scope},
        )
        r.raise_for_status()
        raw = r.json().get("data", {}).get("biz_data", {}).get("settings", {})
        self.settings = _flatten_settings(raw)
        self._settings_fetched_at = time.time()
        return self.settings

    def _completion_timeout(self) -> float:
        ms = self.settings.get("completion_request_timeout_ms", DEFAULT_TIMEOUT_MS)
        try:
            return float(ms) / 1000.0 + 30.0
        except Exception:
            return (DEFAULT_TIMEOUT_MS / 1000.0) + 30.0

    def _solve_pow_sync(self, target_path: str) -> Tuple[str, float]:
        r = self._sync_session.post(
            f"{API_BASE}/chat/create_pow_challenge",
            json={"target_path": target_path}, timeout=15,
        )
        r.raise_for_status()
        ch = r.json()["data"]["biz_data"]["challenge"]
        t0 = time.perf_counter()
        answer = self.solver.solve(
            ch["challenge"], ch["salt"], ch["difficulty"], ch["expire_at"])
        log.info("PoW solved in %.1fms (nonce=%s, path=%s)",
                 (time.perf_counter() - t0) * 1000, answer, target_path)
        return _parse_pow_challenge(ch, target_path, answer)

    async def _fetch_hif_token(self) -> Optional[str]:
        cached, fresh = self.cache.get("hif", HIF_TTL_S)
        if fresh and cached:
            return cached
        try:
            r = await self._client.get(HIF_URL)
            if r.status_code != 200:
                return None
            token = r.json().get("data", {}).get("biz_data", {}).get("value")
            if token:
                self.cache.set("hif", token)
            return token
        except Exception as e:
            log.warning("HIF fetch failed: %s", e)
            return None

    # ----- files (async) ----------------------------------------------

    async def upload_file(
        self,
        path: "str | Path",
        *,
        poll: bool = True,
        timeout_s: float = FILE_POLL_TIMEOUT_S,
    ) -> UploadedFile:
        await self.bootstrap()
        p = Path(path)
        if not p.exists():
            raise UploadError(f"file not found: {p}")
        pow_header = await anyio.to_thread.run_sync(
            lambda: self.pow.get("/api/v0/file/upload_file")
        )
        mime = _guess_mime(p)
        content = p.read_bytes()

        saved_ct = self._client.headers.pop("content-type", None)
        try:
            r = await self._client.post(
                f"{API_BASE}/file/upload_file",
                files={"file": (p.name, content, mime)},
                headers={"x-ds-pow-response": pow_header},
                timeout=300.0,
            )
        finally:
            if saved_ct is not None:
                self._client.headers["Content-Type"] = saved_ct

        if r.status_code != 200:
            body_text = r.content[:2000].decode("utf-8", "replace")
            raise UploadError(f"upload HTTP {r.status_code}: {body_text[:400]}")

        payload = r.json()
        if payload.get("code") != 0:
            raise UploadError(
                f"upload code={payload.get('code')} msg={payload.get('msg')}"
            )
        f = _file_from_json(payload.get("data", {}).get("biz_data") or {})
        if poll and f.status not in ("SUCCESS", "FAILED"):
            f = await self.wait_for_file(f.id, timeout_s=timeout_s)
        return f

    async def fetch_files(self, file_ids: List[str]) -> List[UploadedFile]:
        if not file_ids:
            return []
        await self.bootstrap()
        r = await self._client.get(
            f"{API_BASE}/file/fetch_files",
            params={"file_ids": ",".join(file_ids)},
            timeout=15.0,
        )
        r.raise_for_status()
        payload = r.json()
        if payload.get("code") != 0:
            raise UploadError(f"fetch_files code={payload.get('code')}")
        files = payload.get("data", {}).get("biz_data", {}).get("files") or []
        return [_file_from_json(f) for f in files]

    async def wait_for_file(
        self,
        file_id: str,
        *,
        timeout_s: float = FILE_POLL_TIMEOUT_S,
        interval_s: float = FILE_POLL_INTERVAL_S,
    ) -> UploadedFile:
        deadline = time.time() + timeout_s
        last: Optional[UploadedFile] = None
        while time.time() < deadline:
            files = await self.fetch_files([file_id])
            if not files:
                raise UploadError(f"file {file_id} vanished while polling")
            last = files[0]
            if last.status in ("SUCCESS", "FAILED"):
                if last.status == "FAILED":
                    raise UploadError(
                        f"file {file_id} failed: error_code={last.error_code}"
                    )
                return last
            await anyio.sleep(interval_s)
        raise UploadError(
            f"file {file_id} still {last.status if last else '?'} after {timeout_s}s"
        )

    # ----- history -----------------------------------------------------

    async def fetch_history(self, session_id: str, *, max_pages: int = 3) -> List[HistoryMessage]:
        await self.bootstrap()
        cached_path, _ = self.cache.get("history_path", 10 ** 9)
        candidates = ([cached_path] if cached_path else []) + [
            p for p in _HISTORY_ENDPOINTS if p != cached_path
        ]
        last_err: Optional[Exception] = None
        for path in candidates:
            try:
                messages = await self._fetch_history_via(path, session_id, max_pages)
                if path != cached_path:
                    self.cache.set("history_path", path)
                    log.info("History endpoint working: %s", path)
                return messages
            except Exception as e:
                log.debug("History via %s failed: %s", path, e)
                last_err = e
        log.warning("All history endpoints failed; last error: %s", last_err)
        return []

    async def _history_request(self, path: str, params: Dict[str, Any], *,
                                force_post: bool = False) -> Optional[Dict[str, Any]]:
        url = f"{API_BASE}{path}"
        if force_post:
            r = await self._client.post(url, json=params)
        else:
            r = await self._client.get(url, params=params)
        if r.status_code != 200:
            return None
        try:
            return r.json()
        except Exception:
            return None

    async def _fetch_history_via(
        self, path: str, session_id: str, max_pages: int
    ) -> List[HistoryMessage]:
        messages: List[HistoryMessage] = []
        cursor: Optional[str] = None
        for _ in range(max_pages):
            params: Dict[str, Any] = {"chat_session_id": session_id}
            if cursor:
                params["lte_cursor.id"] = cursor
            payload = await self._history_request(path, params)
            if payload is None:
                raise StreamError(f"history {path} returned no JSON")
            if payload.get("code") != 0:
                raise StreamError(f"history code={payload.get('code')}")
            page = _extract_messages(payload)
            if not page:
                break
            messages = page + messages
            biz = payload.get("data", {}).get("biz_data", {}) or {}
            next_cursor = (
                biz.get("cursor")
                or biz.get("next_cursor")
                or (biz.get("page") or {}).get("cursor")
            )
            if not next_cursor:
                break
            cursor = str(next_cursor)
        return messages

    async def reconstruct_parent_id(self, session_id: str) -> Optional[int]:
        try:
            r = await self._client.get(
                f"{API_BASE}/chat/history_messages",
                params={"chat_session_id": session_id},
                timeout=15,
            )
            if r.status_code == 200:
                biz = (r.json().get("data") or {}).get("biz_data") or {}
                cmid = (biz.get("chat_session") or {}).get("current_message_id")
                if isinstance(cmid, int):
                    log.info("parent_id from current_message_id = %s", cmid)
                    return cmid
        except Exception as e:
            log.debug("current_message_id fast path failed: %s", e)
        messages = await self.fetch_history(session_id)
        for m in reversed(messages):
            if m.role == "ASSISTANT":
                return m.id
        return None

    # ----- remote session listing (async) -----------------------------

    async def fetch_remote_sessions(
        self,
        *,
        max_pages: int = SESSION_FETCH_MAX_PAGES,
    ) -> List[RemoteSession]:
        await self.bootstrap()
        out: List[RemoteSession] = []
        seen_ids: set = set()

        for pinned_flag in (False, True):
            cursor_updated_at: Optional[float] = None
            for page_idx in range(max_pages):
                params: Dict[str, Any] = {
                    "lte_cursor.pinned": "true" if pinned_flag else "false",
                }
                if cursor_updated_at is not None:
                    params["lte_cursor.updated_at"] = str(cursor_updated_at)

                r = await self._client.get(
                    f"{API_BASE}/chat_session/fetch_page",
                    params=params,
                    timeout=15.0,
                )
                if r.status_code != 200:
                    break
                payload = r.json()
                if payload.get("code") != 0:
                    break

                biz = payload.get("data", {}).get("biz_data") or {}
                sessions = biz.get("chat_sessions") or []
                if not sessions:
                    break

                for s in sessions:
                    sid = s.get("id")
                    if not sid or sid in seen_ids:
                        continue
                    seen_ids.add(sid)
                    out.append(RemoteSession(
                        id=sid,
                        title=s.get("title"),
                        title_type=s.get("title_type"),
                        model_type=s.get("model_type") or DEFAULT_MODEL_TYPE,
                        pinned=bool(s.get("pinned", pinned_flag)),
                        updated_at=float(s.get("updated_at") or 0.0),
                    ))

                if not biz.get("has_more"):
                    break
                last_updated = sessions[-1].get("updated_at")
                if not last_updated:
                    break
                cursor_updated_at = float(last_updated)

        return out

    async def sync_sessions(self) -> Dict[str, int]:
        remote = await self.fetch_remote_sessions()
        added = 0
        updated = 0
        new_records: List[SessionRecord] = []

        for s in remote:
            local = self.store.get(s.id)
            if local is None:
                new_records.append(SessionRecord(
                    session_id=s.id,
                    title=s.title,
                    model_type=s.model_type,
                    last_parent_id=None,
                    created_at=s.updated_at or time.time(),
                    updated_at=s.updated_at or time.time(),
                ))
                added += 1
            else:
                if local.title != s.title or local.model_type != s.model_type:
                    updated += 1
                local.title = s.title
                local.model_type = s.model_type
                new_records.append(local)

        self.store.put_bulk(new_records)
        return {
            "added": added,
            "updated": updated,
            "total_remote": len(remote),
        }

    # ----- sessions ----------------------------------------------------

    async def new_conversation(self, *, model_type: str = DEFAULT_MODEL_TYPE) -> "AsyncConversation":
        await self.bootstrap()
        r = await self._client.post(f"{API_BASE}/chat_session/create", json={})
        r.raise_for_status()
        sid = r.json()["data"]["biz_data"]["chat_session"]["id"]
        self._client.headers["Referer"] = f"{ORIGIN}/a/chat/s/{sid}"
        rec = SessionRecord(session_id=sid, model_type=model_type)
        self.store.put(rec)
        log.info("Created session %s", sid)
        return AsyncConversation(self, rec)

    async def resume_conversation(self, session_id: str, *, reconstruct: bool = True) -> "AsyncConversation":
        await self.bootstrap()
        rec = self.store.get(session_id) or SessionRecord(session_id=session_id)
        if rec.last_parent_id is None and reconstruct:
            pid = await self.reconstruct_parent_id(session_id)
            if pid is not None:
                rec.last_parent_id = pid
                log.info("Reconstructed parent_id = %s", pid)
        self.store.put(rec)
        self._client.headers["Referer"] = f"{ORIGIN}/a/chat/s/{session_id}"
        return AsyncConversation(self, rec)

    async def conversation(self, session_id: Optional[str] = None, *, create: bool = True) -> "AsyncConversation":
        if session_id:
            return await self.resume_conversation(session_id)
        cur = self.store.current()
        if cur is not None:
            return await self.resume_conversation(cur.session_id)
        if create:
            return await self.new_conversation()
        raise ConfigurationError("No session and create=False")

    async def ask(self, prompt: str, *, thinking: bool = True, search: bool = True,
                  new_session: bool = False,
                  files: Optional[List[str]] = None) -> str:
        conv = await (self.new_conversation() if new_session else self.conversation())
        ref_ids: List[str] = []
        if files:
            for p in files:
                f = await self.upload_file(p, poll=True)
                ref_ids.append(f.id)
        reply = await conv.send(prompt, thinking=thinking, search=search,
                                prefetch=False, ref_file_ids=ref_ids)
        return reply.text


class AsyncConversation:
    def __init__(self, client: AsyncDeepSeekClient, record: SessionRecord):
        self.client = client
        self.record = record
        self.pending_file_ids: List[str] = []

    @property
    def session_id(self) -> str: return self.record.session_id
    @property
    def model_type(self) -> str: return self.record.model_type

    async def send(self, prompt: str, *, thinking: bool = True, search: bool = True,
                   model_type: Optional[str] = None, prefetch: bool = True,
                   ref_file_ids: Optional[List[str]] = None) -> Reply:
        thinking_parts, text_parts = [], []
        meta: Dict[str, Any] = {}
        searches: List[SearchHit] = []
        opened: List[OpenedUrl] = []
        used_ids: List[str] = list(ref_file_ids) if ref_file_ids else list(self.pending_file_ids)

        async for ev in self.stream(prompt, thinking=thinking, search=search,
                                    model_type=model_type, prefetch=prefetch,
                                    ref_file_ids=used_ids):
            if ev.type == "thinking":
                thinking_parts.append(ev.content)
            elif ev.type == "response":
                text_parts.append(ev.content)
            elif ev.type == "meta":
                meta.update(ev.data)
            elif ev.type == "search":
                searches.append(_search_hit_from_event(ev.data))
            elif ev.type == "open":
                opened.append(_opened_from_event(ev.data))
            elif ev.type == "error":
                raise StreamError(ev.data.get("message", "stream error"))

        self.pending_file_ids = []
        return Reply(
            text="".join(text_parts).strip(),
            thinking="".join(thinking_parts).strip(),
            request_message_id=meta.get("request_message_id"),
            response_message_id=meta.get("response_message_id"),
            model_type=self.record.model_type,
            search_triggered=meta.get("search_triggered", False) or bool(searches),
            elapsed_secs=meta.get("elapsed_secs"),
            token_usage=meta.get("token_usage"),
            ref_file_ids=used_ids,
            search_results=searches,
            opened_urls=opened,
            conversation_mode=meta.get("conversation_mode", "DEFAULT"),
        )

    async def stream(self, prompt: str, *, thinking: bool = True, search: bool = True,
                     model_type: Optional[str] = None, prefetch: bool = True,
                     ref_file_ids: Optional[List[str]] = None) -> AsyncIterator[StreamEvent]:
        if model_type:
            self.record.model_type = model_type
        state = SSEState(current_stage="THINK" if thinking else "RESPONSE")
        last_err: Optional[Exception] = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async for ev in self._stream_once(
                    prompt, thinking=thinking, search=search,
                    prefetch=prefetch, state=state, attempt=attempt,
                    ref_file_ids=ref_file_ids,
                ):
                    yield ev
                return
            except ResumableStreamError as e:
                last_err = e
                if not state.seen_thinking and not state.seen_response:
                    await anyio.sleep(1.0)
                    continue
                if attempt > AUTO_RESUME_MAX:
                    break
                log.warning("Stream dropped; auto-resuming (attempt %d)", attempt)
                await anyio.sleep(0.5)
                continue
            except RateLimitError as e:
                last_err = e
                await anyio.sleep(e.retry_after or (2 ** attempt))
            except StreamError as e:
                last_err = e
                log.warning("Stream error: %s", e)
                await anyio.sleep(1.0)

        yield StreamEvent(type="error", data={"message": f"retries exhausted: {last_err}"})

    async def _stream_once(self, prompt, *, thinking, search, prefetch,
                           state: SSEState, attempt: int = 1,
                           ref_file_ids: Optional[List[str]] = None):
        target_path = "/api/v0/chat/completion"
        pow_header = await anyio.to_thread.run_sync(
            lambda: self.client.pow.get(target_path, prefetch=prefetch))
        hif_token = await self.client._fetch_hif_token()

        headers = {"x-ds-pow-response": pow_header, "Accept": "text/event-stream"}
        if hif_token:
            headers["x-hif-leim"] = hif_token

        body = _build_completion_body(
            session_id=self.record.session_id,
            parent_id=self.record.last_parent_id,
            model_type=self.record.model_type,
            prompt=prompt, thinking=thinking, search=search,
            ref_file_ids=ref_file_ids,
        )
        url = f"{ORIGIN}{target_path}"

        state.attempt_thinking = ""
        state.attempt_response = ""
        state.current_stage = "THINK" if thinking else "RESPONSE"

        if self.client.frame_logger:
            self.client.frame_logger.note(
                "completion-request", url=url, parent_id=self.record.last_parent_id,
                attempt=attempt, ref_file_ids=list(ref_file_ids or []),
                resumed=bool(state.seen_thinking or state.seen_response),
            )

        timeout = self.client._completion_timeout()
        partial = False
        async with self.client._client.stream(
            "POST", url, json=body, headers=headers, timeout=timeout,
        ) as resp:
            if resp.status_code == 429:
                retry = resp.headers.get("Retry-After")
                raise RateLimitError("429", retry_after=_parse_retry_after(retry))
            if resp.status_code >= 500:
                await resp.aread()
                raise StreamError(f"server {resp.status_code}")

            ct = resp.headers.get("Content-Type", "")
            if "text/event-stream" not in ct:
                body_bytes = await resp.aread()
                snippet = body_bytes[:400].decode("utf-8", "replace")
                log.error("Non-SSE (%s): %s", ct, snippet)
                yield StreamEvent(type="error", data={
                    "status": resp.status_code, "content_type": ct, "body": snippet})
                return

            parser = SSEParser(state=state, thinking=thinking,
                               logger=self.client.frame_logger)
            try:
                async for chunk in resp.aiter_bytes(chunk_size=4096):
                    if not chunk:
                        continue
                    for ev in parser.feed(chunk):
                        if ev.type in ("thinking", "response"):
                            partial = True
                        yield ev
                    if parser.closed:
                        break
            except Exception as e:
                if partial or state.seen_thinking or state.seen_response:
                    raise ResumableStreamError(f"stream interrupted: {e}") from e
                raise StreamError(f"stream failed: {e}") from e

            for ev in parser.finish():
                yield ev

            if state.response_message_id is not None:
                self.record.last_parent_id = state.response_message_id
                self.client.store.put(self.record)

            if parser.server_auto_resume and not partial:
                raise ResumableStreamError("server signalled auto_resume")

        yield StreamEvent(type="done", data={
            "request_message_id": state.request_message_id,
            "response_message_id": state.response_message_id,
            "search_triggered": state.search_triggered,
            "elapsed_secs": state.elapsed_secs,
            "token_usage": state.token_usage,
            "resume_attempts": attempt,
            "conversation_mode": state.conversation_mode,
        })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _setup_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _print_stream(conv: Conversation, prompt: str, *, thinking: bool, search: bool,
                  ref_file_ids: Optional[List[str]] = None) -> None:
    in_thinking = False
    reply_header_done = False

    for ev in conv.stream(prompt, thinking=thinking, search=search,
                          prefetch=True, ref_file_ids=ref_file_ids):
        if ev.type == "thinking":
            if not in_thinking:
                print("\n[thinking] ", end="", flush=True)
                in_thinking = True
            print(ev.content, end="", flush=True)

        elif ev.type == "response":
            if in_thinking:
                print()
                in_thinking = False
            if not reply_header_done:
                print("\n[reply] ", end="", flush=True)
                reply_header_done = True
            print(ev.content, end="", flush=True)

        elif ev.type == "search":
            if in_thinking:
                print()
                in_thinking = False
            status = ev.data.get("status") or "?"
            summary = ev.data.get("content") or ""
            results = ev.data.get("results") or []
            queries = ev.data.get("queries") or []
            qstr = ", ".join(q.get("query", "?") for q in queries) if queries else ""
            print(f"\n[search:{status}] {summary}" + (f"  ({qstr})" if qstr else ""))
            for r in results[:5]:
                print(f"    · {r.get('title') or '?'}  —  {r.get('url') or ''}")
            if len(results) > 5:
                print(f"    … and {len(results) - 5} more")

        elif ev.type == "open":
            if in_thinking:
                print()
                in_thinking = False
            result = ev.data.get("result") or {}
            print(f"[open] {result.get('title') or '?'}  —  {result.get('url') or ''}")

        elif ev.type == "file":
            if in_thinking:
                print()
                in_thinking = False
            for f in ev.data.get("files") or []:
                if isinstance(f, dict):
                    print(f"[file] {f.get('file_name')}  ({f.get('status')})")

        elif ev.type == "file_update":
            if in_thinking:
                print()
                in_thinking = False
            print(f"[file_update] {ev.data}")

        elif ev.type == "error":
            if in_thinking:
                print()
                in_thinking = False
            print(f"\n[error] {ev.data}", file=sys.stderr)
            return

    if in_thinking:
        print()
    print()


def _cmd_chat(args) -> int:
    frame_logger = FrameLogger(args.dump_frames) if args.dump_frames else None
    client = DeepSeekClient.from_env(frame_logger=frame_logger).bootstrap()
    conv = client.new_conversation(model_type=args.model) if args.new else client.conversation()
    print(f"[session {conv.session_id}]  model={conv.model_type}  parent={conv.record.last_parent_id}")
    print("Commands: /file <path>  /files  /clear  /sync  exit")
    try:
        while True:
            try:
                prompt = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print(); break
            if not prompt:
                continue
            if prompt.lower() in ("exit", "quit", ":q"):
                break
            if prompt.startswith("/file "):
                path = prompt[6:].strip()
                try:
                    f = client.upload_file(path, poll=True)
                    conv.pending_file_ids.append(f.id)
                    print(f"  attached {f.file_name} ({f.id}) status={f.status}")
                except UploadError as e:
                    print(f"  upload failed: {e}", file=sys.stderr)
                continue
            if prompt == "/files":
                if not conv.pending_file_ids:
                    print("  (no files attached)")
                else:
                    for f in client.fetch_files(conv.pending_file_ids):
                        print(f"  {f.id}  {f.file_name}  status={f.status}")
                continue
            if prompt == "/clear":
                conv.pending_file_ids = []
                print("  cleared")
                continue
            if prompt == "/sync":
                try:
                    stats = client.sync_sessions()
                    print(f"  {stats}")
                except DeepSeekError as e:
                    print(f"  sync failed: {e}", file=sys.stderr)
                continue
            print("ai> ", end="", flush=True)
            _print_stream(conv, prompt,
                          thinking=not args.no_think,
                          search=not args.no_search)
    finally:
        client.close()
        if frame_logger:
            frame_logger.close()
    return 0


def _cmd_ask(args) -> int:
    frame_logger = FrameLogger(args.dump_frames) if args.dump_frames else None
    client = DeepSeekClient.from_env(frame_logger=frame_logger).bootstrap()
    conv = client.new_conversation() if args.new else client.conversation()
    try:
        ref_ids: List[str] = []
        for p in (args.file or []):
            f = client.upload_file(p, poll=True)
            ref_ids.append(f.id)

        if args.no_stream:
            reply = conv.send(args.prompt,
                              thinking=not args.no_think,
                              search=not args.no_search,
                              prefetch=False,
                              ref_file_ids=ref_ids)
            print(reply.text)
            if reply.search_results:
                print(f"\n[search: {len(reply.search_results)} batch(es), "
                      f"{len(reply.opened_urls)} page(s) opened, "
                      f"mode={reply.conversation_mode}]")
                for hit in reply.search_results:
                    if hit.summary:
                        print(f"  {hit.summary}")
                    for r in hit.results[:5]:
                        print(f"    · {r.get('title')} — {r.get('url')}")
        else:
            _print_stream(conv, args.prompt,
                          thinking=not args.no_think,
                          search=not args.no_search,
                          ref_file_ids=ref_ids)
    finally:
        client.close()
        if frame_logger:
            frame_logger.close()
    return 0


def _cmd_new(args) -> int:
    client = DeepSeekClient.from_env().bootstrap()
    try:
        conv = client.new_conversation(model_type=args.model)
        print(conv.session_id)
    finally:
        client.close()
    return 0


def _cmd_resume(args) -> int:
    client = DeepSeekClient.from_env().bootstrap()
    try:
        conv = client.resume_conversation(args.session_id, reconstruct=not args.no_reconstruct)
        print(f"session={conv.session_id}  parent={conv.record.last_parent_id}")
    finally:
        client.close()
    return 0


def _cmd_history(args) -> int:
    client = DeepSeekClient.from_env().bootstrap()
    try:
        msgs = client.fetch_history(args.session_id)
        if not msgs:
            print("(no messages or all endpoints failed)")
            return 0
        for m in msgs:
            preview = m.content.replace("\n", " ")[:80]
            extra = f"  files={len(m.file_ids)}" if m.file_ids else ""
            print(f"{m.id:>6}  {m.role:<9}  {preview}{extra}")
    finally:
        client.close()
    return 0


def _cmd_sessions(args) -> int:
    client = DeepSeekClient.from_env()

    if args.sync:
        try:
            stats = client.sync_sessions()
            print(f"[sync] added={stats['added']} updated={stats['updated']} "
                  f"total_remote={stats['total_remote']}")
        except DeepSeekError as e:
            print(f"[sync failed] {e}", file=sys.stderr)

    if args.remote:
        try:
            remote = client.fetch_remote_sessions()
        except DeepSeekError as e:
            print(f"[remote failed] {e}", file=sys.stderr)
            return 1
        local_ids = {r.session_id for r in client.store.all()}
        current = client.store.current()
        for s in sorted(remote, key=lambda r: r.updated_at, reverse=True):
            marker = "*" if current and s.id == current.session_id else " "
            local_flag = "L" if s.id in local_ids else " "
            pin = "P" if s.pinned else " "
            title = (s.title or "")[:60]
            print(f"{marker}{local_flag}{pin} {s.id}  "
                  f"model={s.model_type:8s}  updated={int(s.updated_at)}  {title}")
        return 0

    cur = client.store.current()
    for r in client.sessions():
        marker = "*" if cur and r.session_id == cur.session_id else " "
        title = (r.title or "")[:60]
        print(f"{marker} {r.session_id}  model={r.model_type:8s}  "
              f"parent={r.last_parent_id}  {title}")
    return 0


def _cmd_info(args) -> int:
    client = DeepSeekClient.from_env().bootstrap()
    try:
        print("device_id:", client.device_id)
        print(f"settings ({len(client.settings)} keys):")
        for k in sorted(client.settings):
            s = str(client.settings[k])
            print(f"  {k} = {s[:77]}{'...' if len(s) > 80 else ''}")
    finally:
        client.close()
    return 0


def _cmd_clear_cache(args) -> int:
    path = Path(CACHE_FILE)
    if path.exists():
        path.unlink()
        print(f"removed {CACHE_FILE}")
    return 0


def _cmd_upload(args) -> int:
    client = DeepSeekClient.from_env().bootstrap()
    try:
        f = client.upload_file(args.path, poll=not args.no_poll)
        print(json.dumps(asdict(f), indent=2))
    finally:
        client.close()
    return 0


def _cmd_fetch_files(args) -> int:
    client = DeepSeekClient.from_env().bootstrap()
    try:
        files = client.fetch_files(args.file_ids)
        for f in files:
            print(json.dumps(asdict(f), indent=2))
    finally:
        client.close()
    return 0


def _cmd_list_remote(args) -> int:
    client = DeepSeekClient.from_env().bootstrap()
    try:
        remote = client.fetch_remote_sessions()
        if args.json:
            for s in remote:
                print(json.dumps(asdict(s)))
            return 0
        for s in sorted(remote, key=lambda r: r.updated_at, reverse=True):
            pin = "📌" if s.pinned else "  "
            print(f"{pin} {s.id}  model={s.model_type:8s}  "
                  f"updated={int(s.updated_at)}  {s.title or ''}")
        print(f"\nTotal: {len(remote)}")
    finally:
        client.close()
    return 0


def _cmd_async_demo(args) -> int:
    async def _run():
        logger = FrameLogger(args.dump_frames) if args.dump_frames else None
        client = AsyncDeepSeekClient.from_env(frame_logger=logger)
        try:
            await client.bootstrap()
            conv = await (client.new_conversation() if args.new else client.conversation())
            ref_ids: List[str] = []
            for p in (args.file or []):
                f = await client.upload_file(p, poll=True)
                ref_ids.append(f.id)
            reply = await conv.send(args.prompt,
                                    thinking=not args.no_think,
                                    search=not args.no_search,
                                    ref_file_ids=ref_ids)
            print(reply.text)
            if reply.search_results:
                print(f"\n[search: {len(reply.search_results)} batch(es), "
                      f"{len(reply.opened_urls)} page(s) opened, "
                      f"mode={reply.conversation_mode}]")
        finally:
            await client.close()
            if logger:
                logger.close()
    anyio.run(_run)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="deepseek", description="DeepSeek Web API client")
    p.add_argument("--debug", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--model", default=DEFAULT_MODEL_TYPE,
                        choices=["default", "expert", "vision"])
        sp.add_argument("--no-think", action="store_true")
        sp.add_argument("--no-search", action="store_true")
        sp.add_argument("--dump-frames", metavar="PATH")

    sp = sub.add_parser("chat", help="interactive REPL (sync)")
    sp.add_argument("--new", action="store_true")
    add_common(sp); sp.set_defaults(func=_cmd_chat)

    sp = sub.add_parser("ask", help="one-shot prompt (sync)")
    sp.add_argument("prompt")
    sp.add_argument("--new", action="store_true")
    sp.add_argument("--file", action="append", metavar="PATH")
    sp.add_argument("--no-stream", action="store_true")
    add_common(sp); sp.set_defaults(func=_cmd_ask)

    sp = sub.add_parser("ask-async", help="one-shot prompt (async demo)")
    sp.add_argument("prompt")
    sp.add_argument("--new", action="store_true")
    sp.add_argument("--file", action="append", metavar="PATH")
    add_common(sp); sp.set_defaults(func=_cmd_async_demo)

    sp = sub.add_parser("new", help="create a new session")
    sp.add_argument("--model", default=DEFAULT_MODEL_TYPE,
                    choices=["default", "expert", "vision"])
    sp.set_defaults(func=_cmd_new)

    sp = sub.add_parser("resume", help="resume a session (reconstructs parent_id)")
    sp.add_argument("session_id")
    sp.add_argument("--no-reconstruct", action="store_true")
    sp.set_defaults(func=_cmd_resume)

    sp = sub.add_parser("history", help="dump history messages for a session")
    sp.add_argument("session_id")
    sp.set_defaults(func=_cmd_history)

    sp = sub.add_parser("sessions", help="list sessions (local by default)")
    sp.add_argument("--remote", action="store_true")
    sp.add_argument("--sync", action="store_true")
    sp.set_defaults(func=_cmd_sessions)

    sp = sub.add_parser("list-remote", help="list sessions from the server")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=_cmd_list_remote)

    sp = sub.add_parser("upload", help="upload a file and print its metadata")
    sp.add_argument("path")
    sp.add_argument("--no-poll", action="store_true")
    sp.set_defaults(func=_cmd_upload)

    sp = sub.add_parser("fetch-files", help="fetch file metadata by id")
    sp.add_argument("file_ids", nargs="+")
    sp.set_defaults(func=_cmd_fetch_files)

    sp = sub.add_parser("info");     sp.set_defaults(func=_cmd_info)

    sp = sub.add_parser("clear-cache", help="delete the persistent cache")
    sp.set_defaults(func=_cmd_clear_cache)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    _setup_logging(args.debug)
    try:
        return args.func(args)
    except DeepSeekError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())