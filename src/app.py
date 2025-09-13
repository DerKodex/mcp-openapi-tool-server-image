#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP OpenAPI Bridge — Generic, Self-Discovering
----------------------------------------------
- 100% generic: no kubernetes/cli-specific logic or flags
- Discovers MCP servers + tools (and prompts/resources if available)
- Exposes:
    • Generic GET/POST:   /{server}/tool/{tool_path:path}
    • Granular GET/POST:  /{server}/tool/{tool}/{action}
                           /{server}/tool/{tool}/{action}/{kind}
- Body is optional. GET can pass arguments via query params.
- Unknown query params (except reserved) are forwarded to the tool.

Run:
  uvicorn app:app --host 0.0.0.0 --port 8080

Env:
  MCP_SERVERS='[{"alias":"mcp","mode":"stdio","cmd":["/path/to/mcp-server"]}]'
  MCP_DISCOVERY_WAIT=2
"""

import asyncio
import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Body, Query, Path, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pydantic import BaseModel, Field

# ---- Optional MCP stdio client (kept generic) --------------------------------
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
        self.prompts: List[Dict[str, Any]] = []
        self.resources: List[Dict[str, Any]] = []


class DiscoveryState:
    def __init__(self):
        self.servers: Dict[str, ServerState] = {}

    def list_servers(self) -> List[Dict[str, Any]]:
        return [{
            "alias": alias,
            "mode": st.cfg.mode,
            "connected": st.connected,
            "tools": list(st.tools.keys()),
            "prompts": len(st.prompts),
            "resources": len(st.resources),
        } for alias, st in self.servers.items()]


DISCOVERY = DiscoveryState()


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


# =============================================================================
# MCP integration (generic)
# =============================================================================

async def mcp_connect_stdio(cfg: ServerConfig):
    if not MCP_AVAILABLE:
        raise RuntimeError("MCP python SDK not installed. pip install mcp[stdio]")
    if not cfg.cmd:
        raise RuntimeError(f"Server {cfg.alias}: stdio mode requires 'cmd'.")
    client = await stdio_client(cfg.cmd, env=cfg.env or {})
    await client.initialize()
    return client


async def mcp_list_tools(session) -> List[Dict[str, Any]]:
    """Return a list of generic tool dicts."""
    out = []
    try:
        result = await session.list_tools()
        for t in result.tools:
            out.append({
                "name": t.name,
                "description": getattr(t, "description", "") or "",
                "input_schema": t.inputSchema.model_dump() if getattr(t, "inputSchema", None) else {}
            })
    except Exception:
        pass
    return out


async def mcp_list_prompts(session) -> List[Dict[str, Any]]:
    """Best-effort generic prompt discovery (optional in MCP)."""
    items = []
    try:
        if hasattr(session, "list_prompts"):
            pres = await session.list_prompts()
            for p in getattr(pres, "prompts", []):
                items.append({
                    "name": getattr(p, "name", None),
                    "description": getattr(p, "description", None),
                })
    except Exception:
        pass
    return items


async def mcp_list_resources(session) -> List[Dict[str, Any]]:
    """Best-effort generic resource discovery (optional in MCP)."""
    items = []
    try:
        if hasattr(session, "list_resources"):
            rres = await session.list_resources()
            for r in getattr(rres, "resources", []):
                items.append({
                    "uri": getattr(r, "uri", None),
                    "name": getattr(r, "name", None),
                    "description": getattr(r, "description", None),
                    "mimeType": getattr(r, "mimeType", None),
                })
    except Exception:
        pass
    return items


async def mcp_call_tool(session, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    call = await session.call_tool(tool_name, args)
    normalized = {"type": "mcp_result", "content": []}
    for item in call.content:
        if isinstance(item, TextContent):
            normalized["content"].append({"type": "text", "text": item.text})
        else:
            payload = item.model_dump() if hasattr(item, "model_dump") else {}
            ctype = payload.get("type") or "unknown"
            normalized["content"].append(payload if ctype != "unknown" else {"type": "unknown", "data": payload})
    return normalized


# =============================================================================
# FastAPI App + Discovery
# =============================================================================

app = FastAPI(
    title="MCP OpenAPI Bridge (Generic, Self-Discovering)",
    version="4.0.0",
    description=textwrap.dedent(
        """\
        A generic, self-discovering OpenAPI façade for MCP servers.
        It discovers tools, prompts, and resources from MCP and exposes:
        • One generic GET/POST endpoint per tool
        • Auto-generated granular endpoints for each `action`/`kind` path segment
        All endpoints accept GET (query) and POST (JSON); request bodies are optional.
        Use `{}` for empty POST bodies. If you cannot send a body, use GET with `?args={...}` or `?key=value`.
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


@app.on_event("startup")
async def on_startup():
    servers_cfg = getenv_json("MCP_SERVERS", None) or []
    for cfg in servers_cfg:
        cfg_obj = ServerConfig(**cfg)
        DISCOVERY.servers[cfg_obj.alias] = ServerState(cfg_obj)
    await do_discover(wait_seconds=int(os.getenv("MCP_DISCOVERY_WAIT", "0")))


async def do_discover(wait_seconds: int = 0) -> Dict[str, Any]:
    for alias, st in DISCOVERY.servers.items():
        try:
            if st.cfg.mode == "stdio":
                if not st.connected:
                    st.client = await mcp_connect_stdio(st.cfg)
                    st.connected = True
                tools_raw = await mcp_list_tools(st.client)
                st.prompts = await mcp_list_prompts(st.client)
                st.resources = await mcp_list_resources(st.client)
            else:
                tools_raw = []
                st.prompts = []
                st.resources = []

            st.tools.clear()
            for tr in tools_raw:
                name = tr["name"]
                td = ToolDescriptor(
                    name=name,
                    description=tr.get("description") or "",
                    input_schema=tr.get("input_schema") or {}
                )
                st.tools[name] = td

        except Exception as e:
            st.connected = False
            st.client = None
            st.tools.clear()
            st.prompts = []
            st.resources = []
            print(f"[discover] Server '{alias}' discovery failed: {e}", file=sys.stderr)

    if wait_seconds > 0:
        await asyncio.sleep(min(wait_seconds, 10))

    return {
        "servers": [{
            "alias": alias,
            "connected": st.connected,
            "tools": [t for t in st.tools.keys()],
            "prompts": len(st.prompts),
            "resources": len(st.resources),
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
# OpenAPI enrichment (x-* blocks) — generic!
# =============================================================================

def openapi_extra_blocks() -> Dict[str, Any]:
    x_model_instructions = {
        "callDiscipline": [
            "Use GET with query or POST with JSON. If no args, POST `{}`.",
            "Granular endpoints `/{SERVER}/tool/{TOOL}/{action}[/{kind}]` map path segments to fields.",
            "The bridge forwards unknown query params to the tool call payload.",
            "Use `dryrun=true` to preview the composed MCP call (no execution)."
        ],
        "bodyShapes": [
            "Direct body: `{ ... }`",
            "Wrapped body: `{ \"args\": { ... } }`"
        ],
        "discovery": [
            "List tools: `GET /{SERVER}/tools/list`.",
            "Per-tool schema: `GET /{SERVER}/tool/{TOOL}/schema`.",
            "Per-tool example: `GET /{SERVER}/tool/{TOOL}/example`.",
            "Per-tool help: `GET /{SERVER}/tool/{TOOL}/help`.",
            "Zero-argument test: `GET /{SERVER}/tool/{TOOL}/try`."
        ],
        "typicalFlow": [
            "1) Read `/openapi.json` and `/discovery/status`.",
            "2) Choose a tool whose schema/description fits the task.",
            "3) Use a granular path when you know the action/kind.",
            "4) Send minimal arguments; tools decide semantics."
        ],
        "errorFix": [
            "If you see 'expected a request body', try POST `{}`.",
            "If schema validation fails, check `/schema` or `/help`.",
            "Use `dryrun=true` to inspect the payload before calling."
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
            })

    return {
        "x-model-instructions": x_model_instructions,
        "x-mcp-tool-catalog": x_mcp_tool_catalog,
        "x-mcp-prompts": {
            alias: st.prompts for alias, st in DISCOVERY.servers.items()
        },
        "x-mcp-resources": {
            alias: st.resources for alias, st in DISCOVERY.servers.items()
        }
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
# Generic + Granular Tool Dispatch (fully generic)
# =============================================================================

RESERVED_QUERY_KEYS = {
    "args", "dryrun"
}

def _collect_query_payload(request: Request) -> Dict[str, Any]:
    """
    Turn *unknown* query params into a dict (forwarded to the tool).
    Reserved keys are kept for the bridge.
    """
    payload: Dict[str, Any] = {}
    for k, v in request.query_params.multi_items():
        if k in RESERVED_QUERY_KEYS:
            continue
        # Keep last occurrence; callers can set dict-y payload via 'args' JSON if needed.
        payload[k] = v
    return payload


async def ensure_connected(server: str) -> ServerState:
    st = DISCOVERY.servers.get(server)
    if not st:
        raise HTTPException(404, f"Unknown server '{server}'")
    if not st.connected or not st.client:
        raise HTTPException(503, f"Server '{server}' is not connected")
    return st


async def do_tool_call(
    server: str,
    tool_path: str,
    body: Optional[Dict[str, Any]],
    args_qs_json: Optional[str],
    dryrun: bool,
) -> Any:
    st = await ensure_connected(server)

    components = [c for c in tool_path.split("/") if c]
    if not components:
        raise HTTPException(400, "Tool path missing")
    tool = components[0]
    suffix = "/".join(components[1:]) if len(components) > 1 else ""

    # Try to locate descriptor; if missing, refresh once, but still attempt call
    td = st.tools.get(tool)
    if not td:
        print(f"[dispatch] Tool '{tool}' not in cache for server '{server}'. Refreshing discovery...", file=sys.stderr)
        await do_discover(wait_seconds=0)
        st2 = DISCOVERY.servers.get(server)
        td = st2.tools.get(tool) if st2 else None

    # Convenience helper endpoints (schema/example/help/try) — generic
    if suffix in ("schema", "example", "help"):
        if td:
            if suffix == "schema":
                return td.input_schema or {}
            if suffix == "example":
                # no generic examples — return empty list; servers may implement their own
                return {"examples": []}
            if suffix == "help":
                return {"name": tool, "description": td.description, "schema": td.input_schema or {}}
        # minimal fallback
        if suffix == "schema":
            return {}
        if suffix == "example":
            return {"examples": []}
        if suffix == "help":
            return {"name": tool, "description": "", "schema": {}}

    if suffix == "try":
        if dryrun:
            return {"dryrun": True, "tool": tool, "args": {}}
        return await mcp_call_tool(st.client, tool, {})

    # Build final args: start from body (or {}), apply args from query string JSON if provided,
    # and inject operation/resource from the path if not present (granular behavior).
    final_args: Dict[str, Any] = {}
    if body:
        final_args.update(body)

    if args_qs_json:
        try:
            parsed = json.loads(args_qs_json)
            if isinstance(parsed, dict):
                final_args.update(parsed)
        except Exception:
            # ignore parse error; 'args' query is optional convenience
            pass

    # Inject from suffix for granular paths (action/kind)
    if suffix and suffix not in ("invoke",):
        parts = suffix.split("/")
        action = parts[0] if len(parts) >= 1 else ""
        kind = parts[1] if len(parts) >= 2 else ""
        if action and "operation" not in final_args:
            final_args["operation"] = action
        if kind and "resource" not in final_args:
            final_args["resource"] = kind

    if dryrun:
        print(f"[dryrun] server={server} tool={tool} suffix='{suffix}' final_args={final_args}", file=sys.stderr)
        return {
            "dryrun": True,
            "server": server,
            "tool": tool,
            "final_args": final_args
        }

    try:
        print(f"[invoke] server={server} tool={tool} suffix='{suffix}' final_args={final_args}", file=sys.stderr)
        return await mcp_call_tool(st.client, tool, final_args)
    except Exception as e:
        raise HTTPException(502, f"Tool invocation failed for '{tool_path}': {e}")


# ---------- Catch-all generic dispatcher ----------

@app.post(
    "/{server}/tool/{tool_path:path}",
    tags=["tools"],
    summary="Tool Dispatch",
    responses={200: {"description": "Successful Response", "content": {"application/json": {}}}},
)
async def tool_dispatch_post(
    request: Request,
    server: str = Path(..., description="Server alias (e.g., 'mcp')"),
    tool_path: str = Path(..., description="Tool or tool path like 'some_tool/do/thing'"),
    args: Optional[str] = Query(None, description="JSON-encoded args fallback"),
    dryrun: Optional[bool] = Query(False, description="If true, returns the would-be MCP call without executing it"),
    body: Optional[Dict[str, Any]] = Body(None),
):
    # Merge unknown query params into body (forwarding), but let explicit JSON body win.
    forwarded = _collect_query_payload(request)
    merged_body = dict(forwarded)
    if body:
        merged_body.update(body)

    result = await do_tool_call(server, tool_path, merged_body, args, bool(dryrun))
    return JSONResponse(result)


@app.get(
    "/{server}/tool/{tool_path:path}",
    tags=["tools"],
    summary="Tool Dispatch",
    responses={200: {"description": "Successful Response", "content": {"application/json": {}}}},
)
async def tool_dispatch_get(
    request: Request,
    server: str = Path(..., description="Server alias (e.g., 'mcp')"),
    tool_path: str = Path(..., description="Tool or tool path like 'some_tool/do/thing'"),
    args: Optional[str] = Query(None, description="JSON-encoded args fallback"),
    dryrun: Optional[bool] = Query(False, description="If true, returns the would-be MCP call without executing it"),
):
    forwarded = _collect_query_payload(request)
    result = await do_tool_call(server, tool_path, forwarded, args, bool(dryrun))
    return JSONResponse(result)


# ---------- Explicit granular routes (avoid 404s) ----------

# POST /{server}/tool/{tool}/{action}/{kind}
@app.post("/{server}/tool/{tool}/{action}/{kind}", tags=["tools"], summary="Granular Tool Dispatch (POST)")
async def granular_post_kind(
    request: Request,
    server: str,
    tool: str,
    action: str,
    kind: str,
    args: Optional[str] = Query(None),
    dryrun: Optional[bool] = Query(False),
    body: Optional[Dict[str, Any]] = Body(None),
):
    forwarded = _collect_query_payload(request)
    merged_body = dict(forwarded)
    if body:
        merged_body.update(body)
    return await tool_dispatch_post(
        request=request,
        server=server,
        tool_path=f"{tool}/{action}/{kind}",
        args=args, dryrun=dryrun, body=merged_body
    )

# GET /{server}/tool/{tool}/{action}/{kind}
@app.get("/{server}/tool/{tool}/{action}/{kind}", tags=["tools"], summary="Granular Tool Dispatch (GET)")
async def granular_get_kind(
    request: Request,
    server: str,
    tool: str,
    action: str,
    kind: str,
    args: Optional[str] = Query(None),
    dryrun: Optional[bool] = Query(False),
):
    return await tool_dispatch_get(
        request=request,
        server=server,
        tool_path=f"{tool}/{action}/{kind}",
        args=args, dryrun=dryrun
    )

# POST /{server}/tool/{tool}/{action}
@app.post("/{server}/tool/{tool}/{action}", tags=["tools"], summary="Granular Tool Dispatch (POST)")
async def granular_post_action(
    request: Request,
    server: str,
    tool: str,
    action: str,
    args: Optional[str] = Query(None),
    dryrun: Optional[bool] = Query(False),
    body: Optional[Dict[str, Any]] = Body(None),
):
    forwarded = _collect_query_payload(request)
    merged_body = dict(forwarded)
    if body:
        merged_body.update(body)
    return await tool_dispatch_post(
        request=request,
        server=server,
        tool_path=f"{tool}/{action}",
        args=args, dryrun=dryrun, body=merged_body
    )

# GET /{server}/tool/{tool}/{action}
@app.get("/{server}/tool/{tool}/{action}", tags=["tools"], summary="Granular Tool Dispatch (GET)")
async def granular_get_action(
    request: Request,
    server: str,
    tool: str,
    action: str,
    args: Optional[str] = Query(None),
    dryrun: Optional[bool] = Query(False),
):
    return await tool_dispatch_get(
        request=request,
        server=server,
        tool_path=f"{tool}/{action}",
        args=args, dryrun=dryrun
    )


# =============================================================================
# Per-tool helper endpoints for ANY server (generic)
# =============================================================================

@app.get("/{server}/tool/{tool}/invoke", tags=["tools", "invoke"], summary="Invoke (GET /invoke)")
async def tool_invoke_get(server: str, tool: str, dryrun: Optional[bool] = Query(False)):
    return await tool_dispatch_get(server=server, tool_path=f"{tool}/try", args=None, dryrun=dryrun, request=None)  # alias to /try


@app.get("/{server}/tool/{tool}/schema", tags=["tools", "schema"], summary="Tool schema")
async def tool_schema(server: str, tool: str):
    st = DISCOVERY.servers.get(server)
    if not st or tool not in st.tools:
        return {}
    return st.tools[tool].input_schema or {}


@app.get("/{server}/tool/{tool}/example", tags=["tools", "example"], summary="Tool example")
async def tool_example(server: str, tool: str):
    # Generic server-agnostic; examples are server-specific so we return none.
    return {"examples": []}


@app.get("/{server}/tool/{tool}/help", tags=["tools", "help"], summary="Tool help")
async def tool_help(server: str, tool: str):
    st = DISCOVERY.servers.get(server)
    if not st or tool not in st.tools:
        return {"name": tool, "description": "", "schema": {}}
    td = st.tools[tool]
    return {"name": tool, "description": td.description, "schema": td.input_schema or {}}


@app.get("/{server}/tool/{tool}/try", tags=["tools", "try"], summary="Tool zero-arg try",
         description="Calls this tool with `{}` (no arguments).")
async def tool_try(server: str, tool: str, dryrun: Optional[bool] = Query(False)):
    if dryrun:
        return {"dryrun": True, "tool": tool, "args": {}}
    st = await ensure_connected(server)
    return await mcp_call_tool(st.client, tool, {})


# =============================================================================
# OpenAPI Post-processor: insert x-* fields
# =============================================================================

_original_openapi = app.openapi

def custom_openapi():
    openapi_schema = _original_openapi()
    openapi_schema.update(openapi_extra_blocks())
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
