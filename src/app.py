#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP OpenAPI Bridge — Generic, Self-Discovering (resilient)
----------------------------------------------------------
- Discovers MCP servers + tools at runtime
- Exposes generic HTTP endpoints per tool
- Calls succeed even if discovery hasn't populated a tool yet
- OpenAPI is populated purely from MCP discovery (no domain-specific logic)

Run:
  uvicorn app:app --host 0.0.0.0 --port 8080

Env:
  MCP_SERVERS='[{"alias":"mcp","mode":"stdio","cmd":["/path/to/mcp-server","--flag"]}]'
  MCP_DISCOVERY_WAIT=2
  MCP_FORWARD_URL='http://mcp-upstream:8080'   # OPTIONAL: REST bridge fallback
  MCP_RPC_URL='http://mcp-server:8080/mcp'     # OPTIONAL: raw MCP HTTP RPC
"""

import asyncio
import inspect
import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional
from urllib.parse import unquote
from types import SimpleNamespace

import httpx
from fastapi import FastAPI, Body, Query, Path, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ---- Optional MCP stdio client ---------------------------------------------
try:
    from mcp.client.stdio import stdio_client
    from mcp.types import TextContent
    MCP_AVAILABLE = True
except Exception:
    MCP_AVAILABLE = False


# =============================================================================
# Data structures
# =============================================================================

class ServerConfig(BaseModel):
    alias: str
    mode: str = Field("stdio", description="stdio | custom")
    cmd: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None


class ToolDescriptor(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None


class ServerState:
    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg
        self.connected = False
        self.client = None
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
MCP_FORWARD_URL = os.getenv("MCP_FORWARD_URL")  # Optional REST-forward target
MCP_RPC_URL = os.getenv("MCP_RPC_URL")          # Optional raw MCP HTTP RPC


# =============================================================================
# Utilities
# =============================================================================

def getenv_json(name: str, default: Any) -> Any:
    """Parse an env var as JSON or return default."""
    val = os.getenv(name)
    if not val:
        return default
    try:
        return json.loads(val)
    except Exception:
        return default


# =============================================================================
# MCP integration (stdio)
# =============================================================================

async def _normalize_stdio_result(res):
    """
    Normalize stdio_client result:
    - If it is an async context manager: await __aenter__().
    - Else if it's awaitable: await it.
    - Else return the instance.
    """
    if hasattr(res, "__aenter__"):
        entered = res.__aenter__()
        client = await entered if inspect.isawaitable(entered) else entered
        try:
            setattr(client, "__mcp_ctx__", res)  # keep ctx for graceful shutdown
        except Exception:
            pass
        return client

    if inspect.isawaitable(res):
        return await res

    return res


async def mcp_connect_stdio(cfg: ServerConfig):
    """
    Create an MCP stdio client across SDK variants while avoiding keyword-only
    signatures and avoiding argv/list forms entirely.

    We call stdio_client with a single positional "spec"-like object that exposes:
      - .command: str
      - .args: list[str]
      - .env: Optional[dict[str,str]]

    Order:
      A) stdio_client(SimpleNamespace(command, args, env))
      B) stdio_client(SimpleNamespace(command, args)) with env injected into os.environ

    We do NOT try list/string variants to prevent "'list' object has no attribute 'command'".
    """
    if not MCP_AVAILABLE:
        raise RuntimeError("MCP python SDK not installed. pip install mcp[stdio]")
    if not cfg.cmd:
        raise RuntimeError(f"Server {cfg.alias}: stdio mode requires 'cmd'.")

    # Normalize command + args
    if isinstance(cfg.cmd, list):
        if not cfg.cmd:
            raise RuntimeError(f"Server {cfg.alias}: empty cmd list.")
        _command, _args = str(cfg.cmd[0]), [str(x) for x in cfg.cmd[1:]]
    elif isinstance(cfg.cmd, str):
        _command, _args = cfg.cmd, []
    else:
        raise RuntimeError(f"Server {cfg.alias}: cmd must be list[str] or str, got {type(cfg.cmd)}")

    last_exc: Optional[BaseException] = None

    # Variant A: pass a 'spec'-like object as single positional arg (with env attr)
    try:
        spec = SimpleNamespace(command=_command, args=_args, env=(cfg.env or None))
        res = stdio_client(spec)  # one positional argument
        client = await _normalize_stdio_result(res)
        init = getattr(client, "initialize", None)
        if callable(init):
            maybe = init()
            if inspect.isawaitable(maybe):
                await maybe
        return client
    except Exception as e:
        last_exc = e

    # Variant B: temporarily inject env into process and pass spec without env attribute
    orig_env = None
    try:
        if cfg.env:
            orig_env = os.environ.copy()
            os.environ.update(cfg.env)

        spec2 = SimpleNamespace(command=_command, args=_args)
        res = stdio_client(spec2)  # still one positional argument
        client = await _normalize_stdio_result(res)
        init = getattr(client, "initialize", None)
        if callable(init):
            maybe = init()
            if inspect.isawaitable(maybe):
                await maybe
        return client
    except Exception as e2:
        last_exc = e2
        raise RuntimeError(f"Failed to create stdio MCP client for '{cfg.alias}': {last_exc}")
    finally:
        if orig_env is not None:
            os.environ.clear()
            os.environ.update(orig_env)


async def mcp_list_tools(session) -> List[Dict[str, Any]]:
    result = await session.list_tools()
    # Accept either an object with .tools or a plain list
    raw_tools = getattr(result, "tools", result)
    tools = []
    for t in raw_tools:
        name = getattr(t, "name", None) or t.get("name")
        if not name:
            continue
        description = getattr(t, "description", None)
        if description is None and isinstance(t, dict):
            description = t.get("description", "")
        schema_obj = getattr(t, "inputSchema", None)
        if schema_obj is None and isinstance(t, dict):
            schema_obj = t.get("inputSchema") or t.get("input_schema") or {}
        if hasattr(schema_obj, "model_dump"):
            schema_obj = schema_obj.model_dump()
        tools.append({
            "name": name,
            "description": description or "",
            "input_schema": schema_obj or {}
        })
    return tools


async def mcp_call_tool(session, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    call = await session.call_tool(tool_name, args)
    normalized = {"type": "mcp_result", "content": []}
    content_seq = getattr(call, "content", None) or call
    for item in content_seq:
        if isinstance(item, TextContent):
            normalized["content"].append({"type": "text", "text": item.text})
        else:
            payload = item.model_dump() if hasattr(item, "model_dump") else (item if isinstance(item, dict) else {})
            ctype = payload.get("type") or "unknown"
            normalized["content"].append(payload if ctype != "unknown" else {"type": "unknown", "data": payload})
    return normalized


# =============================================================================
# MCP integration (HTTP RPC shim for streamable-http)
# =============================================================================

async def rpc_try_methods(client: httpx.AsyncClient, url: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Try a set of method payloads until one returns 200 with a plausible shape.
    Returns parsed JSON or raises.
    """
    last_exc: Optional[Exception] = None
    for payload in candidates:
        try:
            r = await client.post(url, json=payload, timeout=60)
            if r.status_code in (200, 207):
                data = r.json()
                return data
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
# FastAPI App + Discovery
# =============================================================================

app = FastAPI(
    title="MCP OpenAPI Bridge (Generic, Self-Discovering)",
    version="4.1.1",
    description=textwrap.dedent(
        """\
        A generic, self-discovering OpenAPI façade for MCP servers.
        It discovers tools (and their JSON Schemas) from MCP and exposes a generic
        GET/POST endpoint per tool. Requests are passed through without any
        domain-specific transformations or assumptions.
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

# --- Path normalizer (generic legacy compatibility)
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
        DISCOVERY.servers["mcp"] = ServerState(ServerConfig(alias="mcp"))  # placeholder
    await do_discover(wait_seconds=int(os.getenv("MCP_DISCOVERY_WAIT", "0")))

@app.on_event("shutdown")
async def on_shutdown():
    # Gracefully close any stdio sessions opened via async context manager
    for alias, st in list(DISCOVERY.servers.items()):
        try:
            if st.client is not None:
                ctx = getattr(st.client, "__mcp_ctx__", None)
                if ctx and hasattr(ctx, "__aexit__"):
                    await ctx.__aexit__(None, None, None)
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
            if (not st.cfg.cmd and cfg_obj.cmd) or (st.cfg.cmd != cfg_obj.cmd) or (st.cfg.mode != cfg_obj.mode) or (st.cfg.env != cfg_obj.env):
                DISCOVERY.servers[cfg_obj.alias] = ServerState(cfg_obj)
                updated = True
    return updated


async def do_discover(wait_seconds: int = 0) -> Dict[str, Any]:
    for alias, st in list(DISCOVERY.servers.items()):
        tools_raw: List[Dict[str, Any]] = []

        # ---- Try stdio path
        if st.cfg.mode == "stdio" and st.cfg.cmd:
            try:
                if not st.connected:
                    st.client = await mcp_connect_stdio(st.cfg)
                    st.connected = True
                try:
                    tools_raw = await mcp_list_tools(st.client)
                except Exception as e:
                    print(f"[discover] list_tools via stdio failed for '{alias}': {e}", file=sys.stderr)
                    tools_raw = []
            except Exception as e:
                print(f"[discover] stdio connect failed for '{alias}': {e}", file=sys.stderr)
                st.connected = False
                st.client = None

        # ---- HTTP RPC fallback discovery (streamable-http) if no tools yet
        if not tools_raw and MCP_RPC_URL:
            try:
                tools_raw = await mcp_http_list_tools()
                if tools_raw:
                    st.connected = st.connected or True  # virtually connected
            except Exception as e:
                print(f"[discover:http-rpc] list_tools failed via {MCP_RPC_URL}: {e}", file=sys.stderr)

        # ---- Update local tool cache (generic only)
        st.tools.clear()
        for tr in tools_raw:
            name = tr["name"]
            desc = tr.get("description") or ""
            schema = tr.get("input_schema") or {}
            st.tools[name] = ToolDescriptor(name=name, description=desc, input_schema=schema)

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
# OpenAPI enrichment (generic)
# =============================================================================

def openapi_extra_blocks() -> Dict[str, Any]:
    x_model_instructions = {
        "usage": [
            "Use **GET** with `?args={...}` (JSON-encoded) or **POST** with a JSON body.",
            "If the tool expects no arguments, send POST `{}`.",
            "This bridge does not modify arguments; it forwards them as-is to MCP.",
            "Use `/mcp/tool/{tool}/schema` to inspect the input schema discovered from MCP."
        ],
        "discovery": [
            "List tools: `GET /{SERVER}/tools/list`.",
            "Per-tool schema: `GET /{SERVER}/tool/{TOOL}/schema`.",
            "Per-tool help (generic): `GET /{SERVER}/tool/{TOOL}/help`.",
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
# Generic Tool Dispatch (no domain logic, no inference)
# =============================================================================

async def ensure_connected(server: str) -> ServerState:
    st = DISCOVERY.servers.get(server)
    if not st and DISCOVERY.servers:
        first_alias, st = next(iter(DISCOVERY.servers.items()))
        print(f"[compat] Alias '{server}' not found; falling back to '{first_alias}'", file=sys.stderr)
    if not st:
        raise HTTPException(503, "No MCP server configured. Set MCP_SERVERS env to a valid stdio MCP server.")
    if not st.connected or not st.client:
        try:
            if st.cfg.mode == "stdio" and st.cfg.cmd:
                st.client = await mcp_connect_stdio(st.cfg)
                st.connected = True
                await do_discover(0)
            else:
                raise RuntimeError("Missing stdio cmd for MCP server")
        except Exception as e:
            raise HTTPException(503, f"Server '{st.cfg.alias}' is not connected: {e}")
    return st


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

    # Try stdio first; on 503, caller may forward via HTTP if configured
    try:
        st = await ensure_connected(server)
        return await mcp_call_tool(st.client, tool_name, args)
    except HTTPException as e:
        if e.status_code == 503 and MCP_RPC_URL:
            # Fallback to HTTP RPC
            print(f"[invoke-http-rpc] url={MCP_RPC_URL} tool={tool_name} args={args}", file=sys.stderr)
            return await mcp_http_call_tool(tool_name, args)
        raise


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
    args: Optional[str] = Query(None, description="JSON-encoded arguments; if not JSON, will be passed as {'args': '<raw>'}"),
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
                "Use /mcp/tool/{tool}/schema to inspect expected fields."
            ],
        }
    td = st.tools[tool]
    return {
        "name": tool,
        "description": td.description,
        "notes": [
            "Arguments are forwarded to the MCP tool exactly as you send them.",
            "Use /mcp/tool/{tool}/schema to inspect expected fields."
        ],
    }

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
