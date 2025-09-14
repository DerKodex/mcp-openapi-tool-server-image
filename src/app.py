#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP OpenAPI Bridge — Generic, Self-Discovering (HTTP-first)
-----------------------------------------------------------
- Purely generic: no domain-specific logic or assumptions
- Discovers MCP tools at runtime and exposes generic HTTP endpoints
- HTTP RPC (MCP_RPC_URL) is the default and recommended path
- Optional stdio support is DISABLED by default (set MCP_STDIO_ENABLED=1 to opt in)

Run:
  uvicorn app:app --host 0.0.0.0 --port 8080

Env:
  MCP_RPC_URL='http://mcp-server:8080/mcp'     # Raw MCP HTTP RPC endpoint (recommended)
  MCP_STDIO_ENABLED=0                          # 0/1 (default 0). If 1, tries stdio *after* HTTP.
  MCP_SERVERS='[{"alias":"mcp","mode":"stdio","cmd":["/path/to/mcp-server","--flag"]}]'
  MCP_DISCOVERY_WAIT=2
  MCP_FORWARD_URL='http://mcp-upstream:8080'   # OPTIONAL: REST bridge fallback when stdio/HTTP both not available
"""

import asyncio
import inspect
import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

import httpx
from fastapi import FastAPI, Body, Query, Path, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# Optional MCP stdio client (DISABLED by default; see MCP_STDIO_ENABLED)
# -----------------------------------------------------------------------------
MCP_STDIO_ENABLED = os.getenv("MCP_STDIO_ENABLED", "0").strip().lower() in ("1", "true", "yes")

MCP_AVAILABLE = False
MCPClientSession = None
stdio_client = None
if MCP_STDIO_ENABLED:
    try:
        from mcp.client.stdio import stdio_client as _stdio_client  # type: ignore
        stdio_client = _stdio_client
        MCP_AVAILABLE = True
    except Exception:
        MCP_AVAILABLE = False
    try:
        # Session type name varies by version
        from mcp.client.session import ClientSession as MCPClientSession  # type: ignore
    except Exception:
        try:
            from mcp.client.session import Session as MCPClientSession  # type: ignore
        except Exception:
            MCPClientSession = None


# =============================================================================
# Data structures (generic)
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
        self.client = None  # MCP session (when stdio enabled)
        self.tools: Dict[str, ToolDescriptor] = {}
        self._stdio_ctx = None
        self._session_ctx = None


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
    """Return {field: [enum values]} for fields that define enum arrays."""
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
    """Return list of top-level property names from schema (best-effort)."""
    try:
        props = (schema or {}).get("properties", {})
        return list(props.keys())
    except Exception:
        return []


def _build_examples_from_schema(tool_name: str, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    GENERIC example generator:
    - If enum fields exist, create a couple of GET/POST examples using their first values.
    - Include other fields with placeholder values.
    - No domain assumptions. Everything is derived from names/enums only.
    """
    props = _schema_fields(schema)
    enums = _schema_enums(schema)
    required = (schema or {}).get("required", [])
    examples: List[Dict[str, Any]] = []

    # Helper: build a sample body using first enum values + placeholders
    def sample_body() -> Dict[str, Any]:
        body: Dict[str, Any] = {}
        for p in props:
            if p in enums and enums[p]:
                body[p] = enums[p][0]
            else:
                # simple placeholder; keep strings for most fields
                body[p] = f"<{p}>"
        return body

    # If we have two common enum fields, show combo examples; otherwise generic
    body = sample_body()

    # POST example
    examples.append({
        "intent": f"Call '{tool_name}' with a JSON body matching its schema",
        "call": {
            "POST": f"/mcp/tool/{tool_name}",
            "body": body
        },
        "notes": [
            "Send exactly the fields your MCP tool expects. The body is forwarded as-is."
        ]
    })

    # GET example using ?args=
    examples.append({
        "intent": f"Call '{tool_name}' via GET with JSON-encoded args",
        "call": {
            "GET": f"/mcp/tool/{tool_name}?args=" + json.dumps(body)
        },
        "notes": [
            "The 'args' query param must be JSON-encoded. The server forwards it as-is."
        ]
    })

    # If there are obvious enums like operation/resource, add one more explicit sample
    interesting = [k for k in ("operation", "resource", "action", "kind", "type") if k in props]
    if interesting:
        body2 = sample_body()
        examples.append({
            "intent": f"Explicitly set {', '.join(interesting)} for '{tool_name}'",
            "call": {
                "POST": f"/mcp/tool/{tool_name}",
                "body": body2
            },
            "notes": ["Values shown are the first enum choices when available."]
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
# MCP integration — stdio (optional / off by default)
# =============================================================================

async def _enter_ctx(cm):
    aenter = getattr(cm, "__aenter__", None)
    if not callable(aenter):
        raise TypeError("Expected an async context manager")
    entered = aenter()
    return await entered if inspect.isawaitable(entered) else entered

async def _exit_ctx(cm):
    aexit = getattr(cm, "__aexit__", None)
    if callable(aexit):
        try:
            maybe = aexit(None, None, None)
            if inspect.isawaitable(maybe):
                await maybe
        except Exception:
            pass

async def mcp_connect_stdio(cfg: ServerConfig):
    """
    Extremely conservative stdio connector:
    - Only attempts the simplest signatures observed across SDKs
    - No keyword variants; pass list[str] or str positionally
    - env/cwd aren't passed (older SDKs differ); wrap externally if needed
    """
    if not MCP_STDIO_ENABLED:
        raise RuntimeError("MCP stdio disabled (set MCP_STDIO_ENABLED=1 to enable).")
    if not MCP_AVAILABLE or MCPClientSession is None or stdio_client is None:
        raise RuntimeError("MCP stdio client/session not available in this environment.")
    if not cfg.cmd:
        raise RuntimeError(f"Server {cfg.alias}: stdio mode requires 'cmd'.")

    # Normalize
    if isinstance(cfg.cmd, list):
        cmd_list = [str(x) for x in cfg.cmd]
        cmd_str = cmd_list[0]
    elif isinstance(cfg.cmd, str):
        cmd_list = [cfg.cmd]
        cmd_str = cfg.cmd
    else:
        raise RuntimeError(f"Server {cfg.alias}: cmd must be list[str] or str.")

    last_exc: Optional[BaseException] = None

    # Try passing a list as a single positional argument
    try:
        stdio_ctx = stdio_client(cmd_list)
        rw = await _enter_ctx(stdio_ctx)
        if not isinstance(rw, (tuple, list)) or len(rw) != 2:
            await _exit_ctx(stdio_ctx)
            raise TypeError("stdio_client did not yield (read_stream, write_stream)")
        read_stream, write_stream = rw[0], rw[1]
        session_ctx = MCPClientSession(read_stream, write_stream)
        session = await _enter_ctx(session_ctx)

        init = getattr(session, "initialize", None)
        if callable(init):
            maybe = init()
            if inspect.isawaitable(maybe):
                await maybe

        setattr(session, "__mcp_stdio_ctx__", stdio_ctx)
        setattr(session, "__mcp_session_ctx__", session_ctx)
        return session
    except Exception as e:
        last_exc = e

    # Try passing a string as a single positional argument
    try:
        stdio_ctx = stdio_client(cmd_str)
        rw = await _enter_ctx(stdio_ctx)
        if not isinstance(rw, (tuple, list)) or len(rw) != 2:
            await _exit_ctx(stdio_ctx)
            raise TypeError("stdio_client did not yield (read_stream, write_stream)")
        read_stream, write_stream = rw[0], rw[1]
        session_ctx = MCPClientSession(read_stream, write_stream)
        session = await _enter_ctx(session_ctx)

        init = getattr(session, "initialize", None)
        if callable(init):
            maybe = init()
            if inspect.isawaitable(maybe):
                await maybe

        setattr(session, "__mcp_stdio_ctx__", stdio_ctx)
        setattr(session, "__mcp_session_ctx__", session_ctx)
        return session
    except Exception as e:
        last_exc = e

    raise RuntimeError(f"Failed to open stdio_client for '{cfg.alias}': {last_exc}")

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
    version="5.1.0",
    description=textwrap.dedent(
        """\
        A generic, self-discovering OpenAPI façade for MCP servers.
        - HTTP RPC is the primary path (MCP_RPC_URL).
        - Stdio is optional and OFF by default (set MCP_STDIO_ENABLED=1 to enable).
        - No domain-specific logic; arguments are forwarded exactly as provided.
        - Per-tool examples are auto-generated from each tool's schema.
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

# Normalize legacy odd paths into /mcp/tool/{...}
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
    # Seed server list from env (for stdio mode only)
    servers_cfg = getenv_json("MCP_SERVERS", None) or []
    for cfg in servers_cfg:
        cfg_obj = ServerConfig(**cfg)
        DISCOVERY.servers[cfg_obj.alias] = ServerState(cfg_obj)
    if "mcp" not in DISCOVERY.servers:
        DISCOVERY.servers["mcp"] = ServerState(ServerConfig(alias="mcp"))

    await do_discover(wait_seconds=int(os.getenv("MCP_DISCOVERY_WAIT", "0")))

@app.on_event("shutdown")
async def on_shutdown():
    # Close any active session/stdio contexts
    for _, st in list(DISCOVERY.servers.items()):
        try:
            if st.client is not None:
                await _exit_ctx(getattr(st.client, "__mcp_session_ctx__", None))
                await _exit_ctx(getattr(st.client, "__mcp_stdio_ctx__", None))
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

        # ---- Prefer HTTP RPC discovery first (more stable)
        if MCP_RPC_URL:
            try:
                tools_raw = await mcp_http_list_tools()
                if tools_raw:
                    st.connected = True
            except Exception as e:
                print(f"[discover:http-rpc] list_tools failed via {MCP_RPC_URL}: {e}", file=sys.stderr)

        # ---- Optional stdio discovery (only if HTTP yielded nothing and stdio is enabled)
        if not tools_raw and MCP_STDIO_ENABLED and st.cfg.mode == "stdio" and st.cfg.cmd:
            try:
                if not st.connected:
                    st.client = await mcp_connect_stdio(st.cfg)
                    st.connected = True
                try:
                    tools_raw = await mcp_list_tools_stdio(st.client)
                except Exception as e:
                    print(f"[discover] list_tools via stdio failed for '{alias}': {e}", file=sys.stderr)
                    tools_raw = []
            except Exception as e:
                print(f"[discover] stdio connect failed for '{alias}': {e}", file=sys.stderr)
                st.connected = False
                st.client = None

        # ---- Update local tool cache (with schema-driven examples)
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
async def discover_endpoint(wait: Optional[int] = Query(0, description="Seconds to wait (max 10) for discovery")):
    wait = max(0, min(int(wait or 0), 10))
    return await do_discover(wait_seconds=wait)

@app.get("/discovery/status", tags=["discovery"], summary="Discovery Status")
async def discovery_status():
    return {"servers": DISCOVERY.list_servers()}


# =============================================================================
# OpenAPI enrichment (generic, with examples)
# =============================================================================

def openapi_extra_blocks() -> Dict[str, Any]:
    x_model_instructions = {
        "usage": [
            "Use **GET** with `?args={...}` (JSON-encoded) or **POST** with a JSON body.",
            "If the tool expects no arguments, send POST `{}`.",
            "This bridge forwards arguments exactly as provided to the MCP tool.",
            "Inspect `/mcp/tool/{tool}/schema` and `/mcp/tool/{tool}/example` for guidance derived from the MCP tool schema."
        ],
        "discovery": [
            "List tools: `GET /{SERVER}/tools/list`.",
            "Per-tool schema: `GET /{SERVER}/tool/{TOOL}/schema`.",
            "Per-tool examples: `GET /{SERVER}/tool/{TOOL}/example`.",
            "Zero-argument test: `GET /{SERVER}/tool/{TOOL}/try`."
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
        tools.append({
            "name": tname,
            "description": td.description,
            "input_schema": td.input_schema
        })
    return {"server": server, "tools": tools}


# =============================================================================
# HTTP Fallback helper
# =============================================================================

async def forward_via_http(server: str, tool_path: str, method: str, params: Dict[str, Any], body: Optional[Dict[str, Any]]):
    if not MCP_FORWARD_URL:
        raise HTTPException(503, "No MCP server connected and MCP_FORWARD_URL not set for HTTP fallback")
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
# Generic Tool Dispatch (HTTP-first; exact pass-through)
# =============================================================================

async def do_tool_call(
    server: str,
    tool_name: str,
    body: Optional[Dict[str, Any]],
    qargs: Optional[str],
    dryrun: bool,
) -> Any:
    # Prepare arguments exactly as provided
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

    # HTTP path (preferred)
    if MCP_RPC_URL:
        try:
            return await mcp_http_call_tool(tool_name, args)
        except Exception as e:
            if MCP_FORWARD_URL:
                params: Dict[str, Any] = {}
                return await forward_via_http(server, tool_name, "POST", params, args)
            raise HTTPException(502, f"HTTP RPC invocation failed: {e}")

    # Optional stdio path (only if enabled and 'mcp' server has cmd)
    if MCP_STDIO_ENABLED:
        st = DISCOVERY.servers.get(server)
        if not st or not st.cfg.cmd:
            raise HTTPException(503, "No MCP_RPC_URL and stdio not configured")
        try:
            if not st.connected:
                st.client = await mcp_connect_stdio(st.cfg)
                st.connected = True
                # Refresh discovery after connect
                await do_discover(0)
            return await mcp_call_tool_stdio(st.client, tool_name, args)
        except Exception as e:
            raise HTTPException(502, f"Stdio invocation failed: {e}")

    # Nothing usable
    raise HTTPException(503, "No MCP_RPC_URL configured and stdio disabled. Set MCP_RPC_URL or enable MCP_STDIO_ENABLED=1.")


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
    args: Optional[str] = Query(None, description="JSON-encoded arguments; if not JSON, will be sent as {'args': '<raw>'}"),
):
    body = None  # GET has no body
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
