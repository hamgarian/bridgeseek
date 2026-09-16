# BridgeSeek: Free DeepSeek API

![BridgeSeek Banner](banner.png)

A local FastAPI bridge that exposes the **chat.deepseek.com** web backend as
OpenAI- and Anthropic-compatible HTTP APIs, giving you a **free DeepSeek API** 
experience using your own authenticated browser session. Any client that speaks the
OpenAI `chat/completions` or Anthropic `messages` protocol can use DeepSeek
Chat, DeepSeek Reasoner, and DeepSeek Search seamlessly.

It is designed to drop into tool-calling agents (e.g. 9Router / Claude Code
style runtimes): it translates tool definitions into DeepSeek's prompt format,
parses tool-call blocks back out of the model output, and maintains stable
upstream chat sessions across multi-turn tool use.

> [!WARNING]
> **Disclaimer:** Using this bridge automates the DeepSeek web interface, which violates their Terms of Service. Your DeepSeek account **may be banned or suspended** if you generate excessive traffic or if they detect automation. Use this tool responsibly and entirely at your own risk.

---

## Features

- **OpenAI-compatible API** — `POST /v1/chat/completions` (streaming & non-streaming).
- **Anthropic-compatible API** — `POST /v1/messages` plus `POST /v1/messages/count_tokens`.
- **Tool / function calling** — converts OpenAI `tools` and Anthropic `tools` into a
  task-framed prompt, then parses `<tool_call>` blocks (with brace-aware, DSML /
  Anthropic-XML, and bare-JSON fallbacks) back into structured tool calls.
- **Incremental Tool Turns** — tool turns reuse existing upstream chats by default and send only incremental messages, saving context budget (`BRIDGE_TOOL_FRESH_SESSION`).
- **Reasoning support** — emits DeepSeek "thinking" output as reasoning deltas.
- **Web search** — `deepseek-search` model surfaces search / open-page fragments (unwraps SSE `BATCH` envelopes).
- **File uploads** — images and documents are uploaded to DeepSeek, polled until
  ready, and referenced as attachments in the conversation.
- **Session reuse & recovery** — caches upstream chats, reuses them across tool
  turns, and automatically recreates a session when DeepSeek rejects a stale
  continuation anchor (`biz_code 26` / "invalid message id").
- **Rate-limit awareness** — shared cooldown across concurrent sessions on `429`.
- **PoW solving** — resolves DeepSeek's SHA3 proof-of-work challenge via WASM
  (`sha3_wasm_bg.7b9ca65ddd.wasm`).

---

## Architecture

```
┌──────────────┐   OpenAI / Anthropic    ┌────────────────────┐   HTTPS    ┌──────────────────┐
│   Client     │ ──────────────────────► │  deepseek_bridge   │ ─────────► │ chat.deepseek.com│
│ (9Router etc)│ ◄────────────────────── │  (FastAPI, :8123)  │ ◄───────── │   /api/v0/*      │
└──────────────┘      SSE / JSON         └─────────┬──────────┘    SSE     └──────────────────┘
                                                   │
                                        ┌──────────▼───────────┐
                                        │ deepseek_web_api_client │
                                        │  auth · PoW · uploads   │
                                        │  SSE parsing · sessions │
                                        └─────────────────────────┘
```

| File | Purpose |
| --- | --- |
| `deepseek_bridge.py` | FastAPI server exposing the OpenAI/Anthropic endpoints, tool-call translation, and session invalidation recovery (v4.0). |
| `deepseek_web_api_client.py` | Core DeepSeek web client: auth, proof-of-work, file uploads, SSE stream parsing, session history. |
| `sha3_wasm_bg.7b9ca65ddd.wasm` | WASM module used to solve DeepSeek's proof-of-work challenge. |
| `requirements.txt` | Python dependency specification. |
| `.env.example` | Environment variable template file. |

---

## Requirements

- Python 3.10+
- Install dependencies:

```bash
pip install -r requirements.txt
```

or manually:

```bash
pip install fastapi uvicorn pydantic anyio curl_cffi wasmtime python-dotenv requests httpx
```

> `curl_cffi` is required for TLS/HTTP2 browser impersonation, and `wasmtime` is
> required to run the proof-of-work WASM module.

---

## Getting credentials

The bridge authenticates as *you* against the DeepSeek web app, so it needs your
browser session credentials. Open `https://chat.deepseek.com` while logged in,
open DevTools → **Network**, click any `api/v0` request, and copy:

1. **`DEEPSEEK_WEB_TOKEN`** — the `Authorization` header value, with the leading
   `Bearer ` stripped.
2. **`DEEPSEEK_COOKIES`** — the entire `Cookie` request header (one line),
   including the session cookie and Cloudflare device cookies.

---

## Configuration

Copy the example file and fill in the required values:

```bash
cp .env.example .env
```

Minimum required:

```env
DEEPSEEK_WEB_TOKEN=...
DEEPSEEK_COOKIES=...
```

### Bridge settings

| Variable | Default | Description |
| --- | --- | --- |
| `BRIDGE_HOST` | `127.0.0.1` | Bind address. |
| `BRIDGE_PORT` | `8123` | Listen port. |
| `BRIDGE_IMPERSONATE` | `chrome131` | curl_cffi browser impersonation target. |
| `BRIDGE_SESSION_TTL_S` | `2592000` | Upstream session cache TTL (seconds). |
| `BRIDGE_FILE_TTL_S` | `2592000` | Uploaded-file reference TTL (seconds). |
| `BRIDGE_MAX_FILE_BYTES` | `209715200` | Max upload size (bytes, 200 MB). |
| `BRIDGE_IDLE_CLIENT_TTL_S` | `21600` | Evict idle upstream clients after this. |
| `BRIDGE_EMIT_REASONING` | `1` | Emit reasoning / thinking deltas. |
| `BRIDGE_EMIT_TOOLS` | `0` | Emit native tool-call stream events. |
| `BRIDGE_USE_USER_AS_SESSION` | `0` | Treat the `user` field as a session key. |
| `BRIDGE_SESSION_KEY_MODE` | `system` | `system` (stable Claude/system affinity) or `first-user`. |
| `BRIDGE_TOOL_FRESH_SESSION` | `0` | `0` reuses session with incremental turns; `1` forces a new session per turn. |
| `BRIDGE_LOG_RAW_TOOL_OUTPUT` | `0` | Log raw tool output (verbose). |
| `BRIDGE_RATE_LIMIT_COOLDOWN_S` | `15` | Shared cooldown after a `429`. |
| `BRIDGE_OUTPUT_CHUNK_CHARS` | `512` | Synthetic SSE chunk size. |
| `BRIDGE_MIN_REQUEST_GAP_S` | `1.5` | Minimum gap between rapid short requests. |
| `BRIDGE_MAX_TOOL_ATTEMPTS` | `3` | Retry budget for tool turns. |
| `BRIDGE_MAX_SESSION_RECREATIONS` | `2` | Max session recreations on `biz_code 26` invalidation. |
| `BRIDGE_MAX_FLATTENED_CHARS` | `60000` | History truncation budget. |
| `BRIDGE_MAX_INCREMENTAL_CHARS` | `24000` | Incremental tool-turn truncation budget. |
| `BRIDGE_TOOL_ALLOWLIST` | `Bash,Read,Write,Edit,Glob,Grep,LS,TodoRead,TodoWrite,MultiEdit` | Tools exposed to the model (`*` = all). |

---

## Running

```bash
# 1. Activate your virtualenv
source venv/Scripts/activate      # Git Bash on Windows
# or: source venv/bin/activate    # Linux / macOS

# 2. Start the bridge
python deepseek_bridge.py
```

The server listens on `http://127.0.0.1:8123` by default.

Health check:

```bash
curl http://127.0.0.1:8123/health
```

---

## Connecting to 9Router

You can use BridgeSeek as an upstream provider for [9Router](https://github.com/9-router/9router) (or Claude Code via 9Router). 

To configure 9Router to use this bridge, update your 9Router configuration (usually `config.toml` or environment variables) to point to the local bridge:

```toml
[providers.deepseek]
api_base = "http://127.0.0.1:8123/v1"
api_key = "dummy-key" # The bridge handles the real web token
model = "deepseek-chat" # or "deepseek-reasoner" / "deepseek-search"
```

If you're using environment variables:

```bash
export OPENAI_API_BASE="http://127.0.0.1:8123/v1"
export OPENAI_API_KEY="dummy-key"
export OPENAI_MODEL_NAME="deepseek-chat"
```

Once running, 9Router will seamlessly forward `chat/completions` or `messages` requests to BridgeSeek, allowing you to use DeepSeek's web backend for tool-calling agents.

---

## API

### `GET /health`

Returns service status, active sessions, and the effective configuration.

### `GET /v1/models`

Lists available models:

- `deepseek-chat`
- `deepseek-reasoner`
- `deepseek-search`

### `POST /v1/chat/completions`

OpenAI-compatible chat endpoint. Supports `stream: true` (SSE) and non-streaming.

```bash
curl http://127.0.0.1:8123/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

Session affinity is taken from (in order): `x-session-id`, `x-conversation-id`,
`metadata.session_id`, the `user` field (if `BRIDGE_USE_USER_AS_SESSION=1`), or a
hash of the `system` messages.

### `POST /v1/messages`

Anthropic-compatible messages endpoint (streaming and non-streaming), with a
translation layer that maps Anthropic tools/content blocks to the internal
OpenAI-shaped request and back.

### `POST /v1/messages/count_tokens`

Returns `{"input_tokens": N}` — a rough character-based estimate.

---

## Testing

```bash
python -m unittest
```

---

## Troubleshooting

- **`ConfigurationError: DEEPSEEK_WEB_TOKEN environment variable is required`** —
  set `DEEPSEEK_WEB_TOKEN` in `.env` and restart.
- **Frequent `429` / rate limits** — increase `BRIDGE_MIN_REQUEST_GAP_S` and/or
  `BRIDGE_RATE_LIMIT_COOLDOWN_S`.
- **`biz_code 26` / "invalid message id"** — automatically recovered in v4.0! The bridge invalidates the poisoned session in `.deepseek_cache.json` using a tombstone sentinel (`__bridge_invalid__`), evicts the client from the pool, creates a fresh session, and retries up to `BRIDGE_MAX_SESSION_RECREATIONS` times.
- **Upload failures** — verify `BRIDGE_MAX_FILE_BYTES` and that your account
  allows file attachments.

---

## Security notes

- The bridge holds your live DeepSeek session credentials. Run it on
  `127.0.0.1` only, and never commit `.env` or the session/cache JSON files.
- Use `BRIDGE_API_KEY` (see `.env.example`) if you must expose it beyond localhost.
- All `.env`, `.deepseek_sessions.json`, and `.deepseek_cache.json` files are
  gitignored — keep it that way.

---

## License

See `LICENSE` if present; otherwise treat as private/unlicensed until specified.


 
 # #   L i c e n s e 
 
 T h i s   p r o j e c t   i s   l i c e n s e d   u n d e r   t h e   M I T   L i c e n s e   -   s e e   t h e   [ L I C E N S E ] ( L I C E N S E )   f i l e   f o r   d e t a i l s .  
 