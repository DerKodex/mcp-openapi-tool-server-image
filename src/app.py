#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP OpenAPI Bridge — Generic, Self-Discovering (HTTP-first, hardened stdio)
---------------------------------------------------------------------------
- Generic only (no domain assumptions)
- HTTP RPC (MCP_RPC_URL) preferred; stdio optional
- Stdio uses the *descriptor-only* signature expected by newer MCP SDKs
- Stdio connector tries multiple argument permutations automatically

Run:
  uvicorn app:app --host 0.0.0.0 --port 8080

Env:
  MCP_RPC_URL='http://mcp-server:8080/mcp'
  MCP_STDIO_ENABLED=0            # 0/1 (default 0). If 1, stdio allowed.
  MCP_FORCE_STDIO=0              # 0/1. If 1, prefer stdio for discovery/calls.
  MCP_STDIO_INIT_TIMEOUT=45      # seconds for session.initialize()
  MCP_STDIO_PREFLIGHT=1          # 0/1 run preflights
  MCP_STDIO_EXTRA_ARGS=''        # optional space-separated extra args appended for stdio attempts
  MCP_SERVERS='[{"alias":"mcp","mode":"stdio","cmd":["/path/to/mcp-server","--transport","stdio","--access-level","readonly"],"env":{"K":"V"},"cwd":"/work"}]'
  MCP_DISCOVERY_WAIT=2
  MCP_FORWARD_URL='http://mcp-upstream:8080'
"""

import asyncio
import inspect
import json
import os
import stat
import sys
import textwrap
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote

import httpx
from fastapi import Body, FastAPI, HTTPException, Path, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# stdio toggles
# -----------------------------------------------------------------------------
MCP_STDIO_ENABLED = os.getenv("MCP_STDIO_ENABLED", "0").strip().lower() in ("1", "true", "yes")
MCP_FORCE_STDIO = os.getenv("MCP_FORCE_STDIO", "0").strip().lower() in ("1", "true", "yes")
MCP_STDIO_INIT_TIMEOUT = int(os.getenv("MCP_STDIO_INIT_TIMEOUT", "45"))
MCP_STDIO_PREFLIGHT = os.getenv("MCP_STDIO_PREFLIGHT", "1").strip().lower() in ("1", "true", "yes")
MCP_STDIO_EXTRA_ARGS = os.getenv("MCP_STDIO_EXTRA_ARGS", "").strip()

MCP_AVAILABLE = False
MCPClientSession = None
stdio_client = None
StdioParamsType = None

if MCP_STDIO_ENABLED or MCP_FORCE_STDIO:
    try:
        from mcp.client.stdio import stdio_client as _stdio_client  # type: ignore
        stdio_client = _stdio_client
        MCP_AVAILABLE = True
    except Exception:
        MCP_AVAILABLE = False
    try:
        from mcp.client.session import ClientSession as MCPClientSession  # type: ignore
    except Exception:
        try:
            from mcp.client.session import Session as MCPClientSession  # type: ignore
        except Exception:
            MCPClientSession = None
    try:
        from mcp.client.stdio import StdioServerParameters as StdioParamsType  # type: ignore
    except Exception:
        StdioParamsType = None  # we’ll use SimpleNamespace


# =============================================================================
# Data structures
# =============================================================================

class ServerConfig(BaseModel):
    alias: str
    mode: str = Field("stdio", description="stdio | custom")
    cmd: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    cwd: Optional[str] = None


class ToolDescriptor(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None
    examples: List[Dict[str, Any]] = Field(default_factory=list)


class ServerState:
    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg
        self.connected = False
        self.client = None  # MCP session when stdio connected
        self.tools: Dict[str, ToolDescriptor] = {}


class DiscoveryState:
    def __init__(self):
        self.servers: Dict[str, ServerState] = {}

    def list_servers(self) -> List[Dict[str, Any]]:
        return [{
            "alias": alias,
            "mode": st.cfg.mode,
            "connected": st.connected,
            "tools": list(st.tools.keys()),
        } for alias, st in self.servers.items()]


DISCOVERY = DiscoveryState()
MCP_FORWARD_URL = os.getenv("MCP_FORWARD_URL")
MCP_RPC_URL = os.getenv("MCP_RPC_URL")


# =============================================================================
# Utilities
# =============================================================================

def getenv_json(name: str, default: Any) -> Any:
    val = os.getenv(name)
    if not val:
        return default
    try:
        return json.loads(val)
    except Exception:
        return default


def _schema_enums(schema: Dict[str, Any]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    try:
        props = (schema or {}).get("properties", {})
        for k, v in (props or {}).items():
            if isinstance(v, dict) and "enum" in v and isinstance(v["enum"], list):
                out[k] = [str(x) for x in v["enum"]]
    except Exception:
        pass
    return out


def _schema_fields(schema: Dict[str, Any]) -> List[str]:
    try:
        props = (schema or {}).get("properties", {})
        return list(props.keys())
    except Exception:
        return []


def _build_examples_from_schema(tool_name: str, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    props = _schema_fields(schema)
    enums = _schema_enums(schema)
    examples: List[Dict[str, Any]] = []

    def sample_body() -> Dict[str, Any]:
        body: Dict[str, Any] = {}
        for p in props:
            if p in enums and enums[p]:
                body[p] = enums[p][0]
            else:
                body[p] = f"<{p}>"
        return body

    body = sample_body()
    examples.append({
        "intent": f"Call '{tool_name}' with a JSON body",
        "call": {"POST": f"/mcp/tool/{tool_name}", "body": body},
        "notes": ["Body is forwarded as-is to the MCP tool."]
    })
    examples.append({
        "intent": f"GET invoke '{tool_name}' with JSON-encoded args",
        "call": {"GET": f"/mcp/tool/{tool_name}?args=" + json.dumps(body)},
        "notes": ["The 'args' query param must be JSON-encoded."]
    })
    interesting = [k for k in ("operation", "resource", "action", "kind", "type") if k in props]
    if interesting:
        body2 = sample_body()
        examples.append({
            "intent": f"Explicitly set {', '.join(interesting)} for '{tool_name}'",
            "call": {"POST": f"/mcp/tool/{tool_name}", "body": body2},
            "notes": ["Values shown use the first enum choices when available."]
        })
    return examples


# =============================================================================
# MCP integration — HTTP RPC (primary)
# =============================================================================

async def rpc_try_methods(client: httpx.AsyncClient, url: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    last_exc: Optional[Exception] = None
    for payload in candidates:
        try:
            r = await client.post(url, json=payload, timeout=60)
            if r.status_code in (200, 207):
                return r.json()
        except Exception as e:
            last_exc = e
    if last_exc:
        raise last_exc
    raise RuntimeError("No RPC method variant succeeded")

async def mcp_http_list_tools() -> List[Dict[str, Any]]:
    if not MCP_RPC_URL:
        return []
    rpc_url = MCP_RPC_URL.rstrip("/")
    async with httpx.AsyncClient() as client:
        data = await rpc_try_methods(client, rpc_url, [
            {"method": "tools/list", "params": {}},
            {"method": "list_tools", "params": {}},
            {"method": "tool/list", "params": {}},
            {"method": "tools.list", "params": {}},
        ])
    tools = []
    container = data.get("result", data)
    for t in container.get("tools", []):
        name = t.get("name")
        if not name:
            continue
        schema = t.get("inputSchema") or t.get("input_schema") or {}
        if hasattr(schema, "model_dump"):
            schema = schema.model_dump()
        tools.append({
            "name": name,
            "description": t.get("description") or "",
            "input_schema": schema,
        })
    return tools

async def mcp_http_call_tool(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if not MCP_RPC_URL:
        raise RuntimeError("MCP_RPC_URL not set")
    rpc_url = MCP_RPC_URL.rstrip("/")
    async with httpx.AsyncClient() as client:
        data = await rpc_try_methods(client, rpc_url, [
            {"method": "tools/call", "params": {"name": tool_name, "arguments": args}},
            {"method": "call_tool", "params": {"name": tool_name, "arguments": args}},
            {"method": "tool/call", "params": {"name": tool_name, "arguments": args}},
            {"method": "tools.call", "params": {"name": tool_name, "arguments": args}},
        ])
    container = data.get("result", data)
    content = container.get("content") or container.get("contents") or []
    normalized = {"type": "mcp_result", "content": []}
    for item in content:
        if isinstance(item, dict):
            if item.get("type") == "text" and "text" in item:
                normalized["content"].append({"type": "text", "text": item["text"]})
            else:
                normalized["content"].append(item)
        else:
            normalized["content"].append({"type": "text", "text": str(item)})
    return normalized


# =============================================================================
# MCP integration — stdio (descriptor-only, multi-variant attempts)
# =============================================================================

async def _enter_ctx(cm):
    """Safely enter an async context manager, tolerant of non-awaitable returns."""
    if cm is None:
        return None
    aenter = getattr(cm, "__aenter__", None)
    if not callable(aenter):
        return cm
    res = aenter()
    return await res if inspect.isawaitable(res) else res

async def _exit_ctx(cm):
    """Safely exit an async context manager, ignoring errors."""
    if cm is None:
        return
    aexit = getattr(cm, "__aexit__", None)
    if callable(aexit):
        try:
            res = aexit(None, None, None)
            if inspect.isawaitable(res):
                await res
        except Exception:
            pass

def _is_executable(path: str) -> bool:
    try:
        st = os.stat(path)
        if not stat.S_ISREG(st.st_mode):
            return False
        return bool(st.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
    except FileNotFoundError:
        return False
    except Exception:
        return False

def _split_cmd(cfg_cmd: Optional[List[str] | str]) -> Tuple[str, List[str]]:
    if isinstance(cfg_cmd, list) and cfg_cmd:
        return str(cfg_cmd[0]), [str(x) for x in cfg_cmd[1:]]
    if isinstance(cfg_cmd, str) and cfg_cmd:
        return cfg_cmd, []
    raise RuntimeError("cmd must be a non-empty list[str] or str")

def _env_for(cfg: ServerConfig) -> Dict[str, str]:
    env = dict(os.environ)
    if cfg.env:
        env.update(cfg.env)
    return env

def _make_desc(command: str, args: List[str], env: Dict[str, str], cwd: str):
    if StdioParamsType is not None:
        return StdioParamsType(
            command=command, args=args, env=env, cwd=cwd,
            encoding="utf-8", stderr_encoding="utf-8",
            encoding_error_handler="replace", stderr_encoding_error_handler="replace",
        )
    return SimpleNamespace(
        command=command, args=args, env=env, cwd=cwd,
        encoding="utf-8", stderr_encoding="utf-8",
        encoding_error_handler="replace", stderr_encoding_error_handler="replace",
    )

def _with_extra_args(args: List[str]) -> List[str]:
    if not MCP_STDIO_EXTRA_ARGS:
        return args
    extra = [a for a in MCP_STDIO_EXTRA_ARGS.split() if a]
    return args + extra

def _arg_permutations(base_args: List[str]) -> List[List[str]]:
    """Generate sensible permutations around transport flag styles."""
    perms: List[List[str]] = []

    def has_pair(k: str) -> bool:
        try:
            i = base_args.index(k)
            return i < len(base_args) - 1
        except ValueError:
            return False

    def has_equals(k: str) -> bool:
        return any(s.startswith(k + "=") for s in base_args)

    # 0) as-is
    perms.append(list(base_args))

    # 1) '--transport','stdio' -> '--transport=stdio'
    if has_pair("--transport"):
        i = base_args.index("--transport")
        if i < len(base_args) - 1:
            v = base_args[i + 1]
            perm = base_args[:i] + [f"--transport={v}"] + base_args[i + 2:]
            perms.append(perm)

    # 2) '--transport=stdio' -> split pair
    if has_equals("--transport"):
        for s in base_args:
            if s.startswith("--transport="):
                v = s.split("=", 1)[1]
                perm = [x for x in base_args if x != s]
                perm = ["--transport", v] + perm
                perms.append(perm)
                break

    # 3) add equals if missing
    if not has_pair("--transport") and not has_equals("--transport"):
        perms.append(base_args + ["--transport=stdio"])

    # 4) add pair if missing
    if not has_pair("--transport") and not has_equals("--transport"):
        perms.append(base_args + ["--transport", "stdio"])

    # 5) bare switch
    if "--stdio" not in base_args:
        perms.append(base_args + ["--stdio"])

    # Deduplicate
    seen = set()
    uniq: List[List[str]] = []
    for p in perms:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            uniq.append(_with_extra_args(p))
    return uniq

async def _run_short(command: str, args: List[str], env: Dict[str, str], cwd: str, tag: str, timeout_s: float = 6.0):
    try:
        proc = await asyncio.create_subprocess_exec(
            command, *args, cwd=cwd, env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            outs, errs = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass
            print(f"[stdio-preflight] {tag} timeout: {command} {' '.join(args)}", file=sys.stderr)
            return
        out_s = (outs or b"")[:400].decode("utf-8", "replace")
        err_s = (errs or b"")[:400].decode("utf-8", "replace")
        code = proc.returncode
        print(f"[stdio-preflight] {tag} rc={code}; stdout[:400]=\n{out_s}\n--- stderr[:400]=\n{err_s}", file=sys.stderr)
    except FileNotFoundError:
        print(f"[stdio-preflight] {tag} NOT FOUND: {command}", file=sys.stderr)
    except Exception as e:
        print(f"[stdio-preflight] {tag} failed: {type(e).__name__}: {e}", file=sys.stderr)

async def _preflight_all(command: str, args: List[str], env: Dict[str, str], cwd: str):
    if not MCP_STDIO_PREFLIGHT:
        return
    await _run_short(command, ["--version"], env, cwd, "version")
    await _run_short(command, ["--help"], env, cwd, "help")
    await _run_short(command, args, env, cwd, "configured-args", timeout_s=4.0)

async def mcp_connect_stdio(cfg: ServerConfig) -> Any:
    """Descriptor-only stdio connector with multi-variant arg attempts."""
    if not (MCP_STDIO_ENABLED or MCP_FORCE_STDIO):
        raise RuntimeError("MCP stdio disabled (set MCP_STDIO_ENABLED=1 or MCP_FORCE_STDIO=1).")
    if not MCP_AVAILABLE or MCPClientSession is None or stdio_client is None:
        raise RuntimeError("MCP stdio client/session not available.")
    if not cfg.cmd:
        raise RuntimeError(f"Server {cfg.alias}: stdio mode requires 'cmd'.")

    command, base_args = _split_cmd(cfg.cmd)
    if os.path.isabs(command) and not _is_executable(command):
        raise RuntimeError(f"Executable not found or not executable: '{command}'")
    env = _env_for(cfg)
    cwd = cfg.cwd or os.getcwd()

    await _preflight_all(command, base_args, env, cwd)

    attempts = _arg_permutations(base_args)
    errors: List[str] = []

    for idx, args in enumerate(attempts):
        desc = _make_desc(command, args, env, cwd)
        stdio_cm = None
        try:
            stdio_cm = stdio_client(desc)  # descriptor-only signature
            rw = await _enter_ctx(stdio_cm)
            if not isinstance(rw, (tuple, list)) or len(rw) != 2:
                raise TypeError("stdio_client did not yield (read_stream, write_stream)")
            read_stream, write_stream = rw[0], rw[1]
            session = MCPClientSession(read_stream, write_stream)

            init = getattr(session, "initialize", None)
            if callable(init):
                maybe = init()
                if inspect.isawaitable(maybe):
                    await asyncio.wait_for(maybe, timeout=MCP_STDIO_INIT_TIMEOUT)

            # success
            setattr(session, "__stdio_ctx__", stdio_cm)
            setattr(session, "__stdio_args__", list(args))
            return session
        except Exception as e:
            try:
                await _exit_ctx(stdio_cm)
            except Exception:
                pass
            kind = type(e).__name__
            errors.append(f"#{idx+1} args={args} -> {kind}: {e}")

    raise RuntimeError(
        "Failed to open stdio_client for "
        f"'{cfg.alias}': " + " | ".join(errors)
    )

async def mcp_list_tools_stdio(session) -> List[Dict[str, Any]]:
    result = await session.list_tools()
    raw = getattr(result, "tools", result)
    out: List[Dict[str, Any]] = []
    for t in raw:
        name = getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else None)
        if not name:
            continue
        desc = getattr(t, "description", None)
        if desc is None and isinstance(t, dict):
            desc = t.get("description", "")
        schema = getattr(t, "inputSchema", None)
        if schema is None and isinstance(t, dict):
            schema = t.get("inputSchema") or t.get("input_schema") or {}
        if hasattr(schema, "model_dump"):
            schema = schema.model_dump()
        out.append({"name": name, "description": desc or "", "input_schema": schema or {}})
    return out

async def mcp_call_tool_stdio(session, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    call = await session.call_tool(tool_name, args)
    content_seq = getattr(call, "content", None) or call
    normalized = {"type": "mcp_result", "content": []}
    try:
        from mcp.types import TextContent as _TC  # type: ignore
    except Exception:
        _TC = None  # type: ignore
    for item in content_seq:
        if _TC and isinstance(item, _TC):
            normalized["content"].append({"type": "text", "text": item.text})
        else:
            payload = item.model_dump() if hasattr(item, "model_dump") else (item if isinstance(item, dict) else {})
            ctype = payload.get("type") or "unknown"
            normalized["content"].append(payload if ctype != "unknown" else {"type": "unknown", "data": payload})
    return normalized


# =============================================================================
# FastAPI App + Discovery
# =============================================================================

app = FastAPI(
    title="MCP OpenAPI Bridge (Generic, Self-Discovering)",
    version="6.3.1",
    description=textwrap.dedent(
        """\
        Generic OpenAPI façade for MCP servers.
        - HTTP RPC preferred (MCP_RPC_URL).
        - Stdio optional (enable with MCP_STDIO_ENABLED=1). Descriptor-only signature.
        - Multi-variant stdio arg attempts (pair/equal/bare switch).
        - No domain-specific logic; arguments are forwarded as provided.
        - Per-tool examples auto-generated from MCP schema.
        """
    ),
    servers=[{"url": "http://localhost:8080"}],
    openapi_url="/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.middleware("http")
async def normalize_odd_paths(request: Request, call_next):
    raw_path = request.scope.get("path") or ""
    decoded = unquote(raw_path)
    for bad in ("<server-alias>", "<server>", "server-alias", "server"):
        prefix = f"/{bad}/"
        if decoded.startswith(prefix):
            decoded = "/mcp/" + decoded[len(prefix):]
            break
    marker = "/tool/"
    if marker in decoded:
        head, tail = decoded.split(marker, 1)
        tail = tail.replace("  ", " ").strip()
        if " " in tail:
            tail = "/".join([p for p in tail.split(" ") if p])
        decoded = head + marker + tail
    if decoded.startswith("/tool/"):
        decoded = "/mcp" + decoded
    if decoded.startswith("/") and "/tool/" in decoded:
        first = decoded.split("/", 2)[1]
        if first and first not in DISCOVERY.servers:
            decoded = "/mcp/" + decoded.split("/", 2)[2]
    if decoded != raw_path:
        request.scope["path"] = decoded
    return await call_next(request)


@app.on_event("startup")
async def on_startup():
    servers_cfg = getenv_json("MCP_SERVERS", None) or []
    for cfg in servers_cfg:
        cfg_obj = ServerConfig(**cfg)
        DISCOVERY.servers[cfg_obj.alias] = ServerState(cfg_obj)
    if "mcp" not in DISCOVERY.servers:
        DISCOVERY.servers["mcp"] = ServerState(ServerConfig(alias="mcp"))
    try:
        await do_discover(wait_seconds=int(os.getenv("MCP_DISCOVERY_WAIT", "0")))
    except Exception as e:
        print(f"Validation failed:\n{e}", file=sys.stderr)

@app.on_event("shutdown")
async def on_shutdown():
    # Attempt to cleanly close any open stdio context
    for _, st in list(DISCOVERY.servers.items()):
        try:
            client = st.client
            if client is not None:
                await _exit_ctx(getattr(client, "__stdio_ctx__", None))
        except Exception:
            pass


def refresh_servers_from_env() -> bool:
    updated = False
    servers_cfg = getenv_json("MCP_SERVERS", None) or []
    for cfg in servers_cfg:
        cfg_obj = ServerConfig(**cfg)
        st = DISCOVERY.servers.get(cfg_obj.alias)
        if not st:
            DISCOVERY.servers[cfg_obj.alias] = ServerState(cfg_obj)
            updated = True
        else:
            if (st.cfg.cmd != cfg_obj.cmd) or (st.cfg.mode != cfg_obj.mode) or (st.cfg.env != cfg_obj.env) or (st.cfg.cwd != cfg_obj.cwd):
                DISCOVERY.servers[cfg_obj.alias] = ServerState(cfg_obj)
                updated = True
    return updated


async def do_discover(wait_seconds: int = 0) -> Dict[str, Any]:
    for alias, st in list(DISCOVERY.servers.items()):
        tools_raw: List[Dict[str, Any]] = []
        prefer_stdio = MCP_FORCE_STDIO

        if not prefer_stdio and MCP_RPC_URL:
            try:
                tools_raw = await mcp_http_list_tools()
                if tools_raw:
                    st.connected = True
            except Exception as e:
                print(f"[discover:http-rpc] list_tools failed via {MCP_RPC_URL}: {e}", file=sys.stderr)

        if (prefer_stdio or not tools_raw) and (MCP_STDIO_ENABLED or MCP_FORCE_STDIO) and st.cfg.mode == "stdio" and st.cfg.cmd:
            try:
                if not st.connected or st.client is None:
                    st.client = await mcp_connect_stdio(st.cfg)
                    st.connected = True
                tools_raw = await mcp_list_tools_stdio(st.client)
            except Exception as e:
                print(f"[discover] stdio connect/list failed for '{alias}': {e}", file=sys.stderr)
                st.connected = False
                st.client = None

        st.tools.clear()
        for tr in tools_raw:
            name = tr["name"]
            desc = tr.get("description") or ""
            schema = tr.get("input_schema") or {}
            examples = _build_examples_from_schema(name, schema or {})
            st.tools[name] = ToolDescriptor(name=name, description=desc, input_schema=schema, examples=examples)

    if wait_seconds > 0:
        await asyncio.sleep(min(wait_seconds, 10))

    return {
        "servers": [{
            "alias": alias,
            "connected": st.connected,
            "tools": [t for t in st.tools.keys()]
        } for alias, st in DISCOVERY.servers.items()]
    }


# =============================================================================
# Health / Info
# =============================================================================

@app.get("/livez", tags=["health"], summary="Livez")
async def livez():
    return {"status": "ok"}

@app.get("/readyz", tags=["health"], summary="Readyz")
async def readyz():
    return {"status": "ok", "servers": DISCOVERY.list_servers()}

@app.get("/healthz", tags=["health"], summary="Healthz")
async def healthz():
    return {"status": "ok"}

@app.get("/servers", tags=["info"], summary="Servers Info")
async def servers_info():
    return {"servers": DISCOVERY.list_servers()}


# =============================================================================
# Discovery control
# =============================================================================

@app.post("/discover", tags=["discovery"], summary="Discover Endpoint")
async def discover_endpoint(wait: Optional[int] = Query(0, description="Seconds to wait (max 10)")):
    wait = max(0, min(int(wait or 0), 10))
    refresh_servers_from_env()
    return await do_discover(wait_seconds=wait)

@app.get("/discovery/status", tags=["discovery"], summary="Discovery Status")
async def discovery_status():
    return {"servers": DISCOVERY.list_servers()}


# =============================================================================
# OpenAPI enrichment
# =============================================================================

def openapi_extra_blocks() -> Dict[str, Any]:
    x_model_instructions = {
        "usage": [
            "Use GET with ?args={...} (JSON-encoded) or POST with a JSON body.",
            "If the tool expects no arguments, send POST {}.",
            "Arguments are forwarded exactly as provided to the MCP tool.",
            "See /mcp/tool/{tool}/schema and /mcp/tool/{tool}/example for schema-derived guidance."
        ],
        "discovery": [
            "List tools: GET /{SERVER}/tools/list",
            "Per-tool schema: GET /{SERVER}/tool/{TOOL}/schema",
            "Per-tool examples: GET /{SERVER}/tool/{TOOL}/example",
            "Zero-argument test: GET /{SERVER}/tool/{TOOL}/try"
        ]
    }
    x_mcp_tool_catalog: List[Dict[str, Any]] = []
    for alias, st in DISCOVERY.servers.items():
        for tname, td in st.tools.items():
            x_mcp_tool_catalog.append({
                "server": alias,
                "tool": tname,
                "description": td.description,
                "schema": td.input_schema or {"type": "object"},
                "requiredFields": (td.input_schema or {}).get("required", []),
                "examples": td.examples,
            })
    return {
        "x-model-instructions": x_model_instructions,
        "x-mcp-tool-catalog": x_mcp_tool_catalog,
        "x-mcp-prompts": {},
        "x-mcp-resources": {}
    }

@app.get("/{server}/tools/list", tags=["discovery"], summary="Tools List")
async def tools_list(server: str = Path(..., description="Server alias")):
    st = DISCOVERY.servers.get(server)
    if not st:
        raise HTTPException(404, f"Unknown server '{server}'")
    tools = []
    for tname, td in st.tools.items():
        tools.append({"name": tname, "description": td.description, "input_schema": td.input_schema})
    return {"server": server, "tools": tools}


# =============================================================================
# HTTP fallback
# =============================================================================

async def forward_via_http(server: str, tool_path: str, method: str, params: Dict[str, Any], body: Optional[Dict[str, Any]]):
    if not MCP_FORWARD_URL:
        raise HTTPException(503, "No MCP server connected and MCP_FORWARD_URL not set")
    url = MCP_FORWARD_URL.rstrip("/") + f"/{server}/tool/{tool_path}"
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            if method.upper() == "GET":
                r = await client.get(url, params=params)
            else:
                r = await client.post(url, params=params, json=body or {})
        if r.headers.get("content-type", "").startswith("application/json"):
            return JSONResponse(status_code=r.status_code, content=r.json())
        return JSONResponse(status_code=r.status_code, content={"upstream_text": r.text})
    except Exception as e:
        raise HTTPException(503, f"HTTP fallback to {url} failed: {e}")


# =============================================================================
# Generic Tool Dispatch
# =============================================================================

async def do_tool_call(
    server: str,
    tool_name: str,
    body: Optional[Dict[str, Any]],
    qargs: Optional[str],
    dryrun: bool,
) -> Any:
    if body is None:
        if qargs is not None:
            try:
                parsed = json.loads(qargs)
                args = parsed if isinstance(parsed, dict) else {"args": parsed}
            except Exception:
                args = {"args": qargs}
        else:
            args = {}
    else:
        args = body

    if dryrun:
        return {"dryrun": True, "server": server, "tool": tool_name, "args": args}

    prefer_stdio = MCP_FORCE_STDIO

    if not prefer_stdio and MCP_RPC_URL:
        try:
            return await mcp_http_call_tool(tool_name, args)
        except Exception as e:
            if MCP_FORWARD_URL:
                return await forward_via_http(server, tool_name, "POST", {}, args)
            raise HTTPException(502, f"HTTP RPC invocation failed: {e}")

    if MCP_STDIO_ENABLED or MCP_FORCE_STDIO:
        st = DISCOVERY.servers.get(server)
        if not st or not st.cfg.cmd:
            raise HTTPException(503, "stdio requested but not configured (missing server/cmd)")
        try:
            if not st.connected or st.client is None:
                st.client = await mcp_connect_stdio(st.cfg)
                st.connected = True
                try:
                    await do_discover(0)
                except Exception:
                    pass
            return await mcp_call_tool_stdio(st.client, tool_name, args)
        except Exception as e:
            raise HTTPException(502, f"Stdio invocation failed: {e}")

    raise HTTPException(503, "No usable transport: set MCP_RPC_URL or enable MCP_STDIO_ENABLED/MCP_FORCE_STDIO")


@app.post(
    "/{server}/tool/{tool_name:path}",
    tags=["tools"],
    summary="Tool Dispatch (POST)",
    responses={200: {"description": "Successful Response", "content": {"application/json": {}}}},
)
async def tool_dispatch_post(
    server: str = Path(..., description="Server alias (e.g., 'mcp')"),
    tool_name: str = Path(..., description="Exact tool name as exposed by MCP"),
    dryrun: Optional[bool] = Query(False, description="If true, returns the would-be MCP call without executing it"),
    body: Optional[Dict[str, Any]] = Body(None),
):
    try:
        result = await do_tool_call(server, tool_name, body, None, bool(dryrun))
        return JSONResponse(result)
    except HTTPException as e:
        if e.status_code == 503 and MCP_FORWARD_URL:
            params: Dict[str, Any] = {}
            if dryrun:
                params["dryrun"] = dryrun
            return await forward_via_http(server, tool_name, "POST", params, body or {})
        raise


@app.get(
    "/{server}/tool/{tool_name:path}",
    tags=["tools"],
    summary="Tool Dispatch (GET)",
    responses={200: {"description": "Successful Response", "content": {"application/json": {}}}},
)
async def tool_dispatch_get(
    server: str = Path(..., description="Server alias (e.g., 'mcp')"),
    tool_name: str = Path(..., description="Exact tool name as exposed by MCP"),
    dryrun: Optional[bool] = Query(False, description="If true, returns the would-be MCP call without executing it"),
    args: Optional[str] = Query(None, description="JSON-encoded arguments; sent as {'args': '<raw>'} if not JSON"),
):
    body = None
    try:
        result = await do_tool_call(server, tool_name, body, args, bool(dryrun))
        return JSONResponse(result)
    except HTTPException as e:
        if e.status_code == 503 and MCP_FORWARD_URL:
            params: Dict[str, Any] = {}
            if dryrun:
                params["dryrun"] = dryrun
            if args is not None:
                params["args"] = args
            return await forward_via_http(server, tool_name, "GET", params, None)
        raise


# -------------------- helper endpoints (generic) --------------------

@app.get("/mcp/tool/{tool}/schema", tags=["tools", "schema"], summary="Tool schema")
async def tool_schema(tool: str):
    st = DISCOVERY.servers.get("mcp")
    if not st or tool not in st.tools:
        return {}
    return st.tools[tool].input_schema or {}

@app.get("/mcp/tool/{tool}/help", tags=["tools", "help"], summary="Tool help (generic)")
async def tool_help(tool: str):
    st = DISCOVERY.servers.get("mcp")
    if not st or tool not in st.tools:
        return {
            "name": tool,
            "description": "",
            "notes": [
                "Arguments are forwarded to the MCP tool exactly as you send them.",
                "Use /mcp/tool/{tool}/schema and /mcp/tool/{tool}/example for guidance."
            ],
        }
    td = st.tools[tool]
    return {
        "name": tool,
        "description": td.description,
        "notes": [
            "Arguments are forwarded to the MCP tool exactly as you send them.",
            "Use /mcp/tool/{tool}/schema and /mcp/tool/{tool}/example for guidance."
        ],
    }

@app.get("/mcp/tool/{tool}/example", tags=["tools", "example"], summary="Tool examples (schema-derived)")
async def tool_example(tool: str):
    st = DISCOVERY.servers.get("mcp")
    if not st or tool not in st.tools:
        return {"examples": []}
    return {"examples": st.tools[tool].examples}

@app.get("/mcp/tool/{tool}/try", tags=["tools", "try"], summary="Tool zero-arg try",
         description="Calls this tool with `{}` (no arguments).")
async def tool_try(tool: str, dryrun: Optional[bool] = Query(False)):
    return await tool_dispatch_get(server="mcp", tool_name=tool, dryrun=dryrun)


# =============================================================================
# OpenAPI Post-processor
# =============================================================================

_original_openapi = app.openapi
def custom_openapi():
    openapi_schema = _original_openapi()
    openapi_schema.update({ **openapi_extra_blocks() })
    app.openapi_schema = openapi_schema
    return app.openapi_schema
app.openapi = custom_openapi
