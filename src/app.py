#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP OpenAPI Bridge — Generic, Self-Discovering
----------------------------------------------
- Completely generic (no domain-specific logic or flags).
- Discovers MCP servers/tools/prompts/resources.
- Exposes:
    • Generic GET/POST:   /{server}/tool/{tool_path:path}
    • Granular GET/POST:  /{server}/tool/{tool}/{action}
                           /{server}/tool/{tool}/{action}/{kind}
- Request body is optional. GET can pass arguments via query params.
- Unknown query params (except 'args' and 'dryrun') are forwarded to the tool.
- For granular routes, 'action' → payload.operation and 'kind' → payload.resource (if not already provided).

Run:
  uvicorn app:app --host 0.0.0.0 --port 8080

Environment:
  MCP_SERVERS='[{"alias":"mcp","mode":"stdio","cmd":["/path/to/mcp-server"],"env":{"FOO":"bar"}}]'
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

# ---- Optional MCP stdio client (generic) -------------------------------------
try:
    from mcp.client.stdio import stdio_client
    from mcp.types import TextContent
    MCP_AVAILABLE = True
except Exception:
    MCP_AVAILABLE = False


# =============================================================================
# Models / State (simple & generic)
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
# Helpers
# =============================================================================

def getenv_json(name: str, default: Any) -> Any:
    val = os.getenv(name)
    if not val:
        return default
    try:
        return json.loads(val)
    except Exception:
        return default


def _safe_get(obj: Any, name: str, default=None):
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


# =============================================================================
# MCP Integration (generic)
# =============================================================================

async def mcp_connect_stdio(cfg: ServerConfig):
    if not MCP_AVAILABLE:
        raise RuntimeError("MCP python SDK not installed. pip install 'mcp[stdio]'")
    if not cfg.cmd:
        raise RuntimeError(f"Server {cfg.alias}: stdio mode requires 'cmd'.")
    client = await stdio_client(cfg.cmd, env=cfg.env or {})
    await client.initialize()
    return client


async def mcp_list_tools(session) -> List[Dict[str, Any]]:
    """Return a generic list of tool dicts."""
    out = []
    try:
        result = await session.list_tools()
        for t in result.tools:
            out.append({
                "name": t.name,
                "description": _safe_get(t, "description", "") or "",
                "input_schema": t.inputSchema.model_dump() if _safe_get(t, "inputSchema") else {}
            })
    except Exception:
        pass
    return out


async def mcp_list_prompts(session) -> List[Dict[str, Any]]:
    items = []
    try:
        if hasattr(session, "list_prompts"):
            pres = await session.list_prompts()
            for p in getattr(pres, "prompts", []):
                items.append({
                    "name": _safe_get(p, "name"),
                    "description": _safe_get(p, "description"),
                })
    except Exception:
        pass
    return items


async def mcp_list_resources(session) -> List[Dict[str, Any]]:
    items = []
    try:
        if hasattr(session, "list_resources"):
            rres = await session.list_resources()
            for r in getattr(rres, "resources", []):
                items.append({
                    "uri": _safe_get(r, "uri"),
                    "name": _safe_get(r, "name"),
                    "description": _safe_get(r, "description"),
                    "mimeType": _safe_get(r, "mimeType"),
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
# FastAPI App
# =============================================================================

app = FastAPI(
    title="MCP OpenAPI Bridge (Generic, Self-Discovering)",
    version="4.1.0",
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
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# =============================================================================
# Startup / Discovery
# =============================================================================

@app.on_event("startup")
async def on_startup():
    servers_cfg = getenv_json("MCP_SERVERS", None) or []
    for cfg in servers_cfg:
        cfg_obj = ServerConfig(**cfg)
        DISCOVERY.servers[cfg_obj.alias] = ServerState(cfg_obj)
    # Attempt initial discovery; do not fail startup if servers unavailable
    await do_discover(wait_seconds=int(os.getenv("MCP_DISCOVERY_WAIT", "0")))


async def connect_and_cache(alias: str, st: ServerState):
    if st.connected and st.client:
        return
    if st.cfg.mode == "stdio":
        st.client = await mcp_connect_stdio(st.cfg)
        st.connected = True
    else:
        # Custom modes can be added by implementers
        raise RuntimeError(f"Unsupported mode '{st.cfg.mode}' for server '{alias}'")


async def refresh_server_catalog(alias: str, st: ServerState):
    """Populate tools/prompts/resources and cache."""
    try:
        await connect_and_cache(alias, st)
        tools_raw = await mcp_list_tools(st.client)
        st.prompts = await mcp_list_prompts(st.client)
        st.resources = await mcp_list_resources(st.client)

        st.tools.clear()
        for tr in tools_raw:
            name = tr["name"]
            st.tools[name] = ToolDescriptor(
                name=name,
                description=tr.get("description") or "",
                input_schema=tr.get("input_schema") or {}
            )
    except Exception as e:
        st.connected = False
        st.client = None
        st.tools.clear()
        st.prompts = []
        st.resources = []
        print(f"[discover] Server '{alias}' discovery failed: {e}", file=sys.stderr)


async def do_discover(wait_seconds: int = 0) -> Dict[str, Any]:
    tasks = []
    for alias, st in DISCOVERY.servers.items():
        tasks.append(refresh_server_catalog(alias, st))
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    if wait_seconds > 0:
        await asyncio.sleep(min(wait_seconds, 10))
    return {"servers": DISCOVERY.list_servers()}


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


@app.get("/{server}/tools/list", tags=["discovery"], summary="Tools List")
async def tools_list(server: str = Path(..., description="Server alias")):
    st = DISCOVERY.servers.get(server)
    if not st:
        raise HTTPException(404, f"Unknown server '{server}'")
    # If not connected, try an on-demand discovery (helps first-run UX)
    if not st.connected or not st.tools:
        await refresh_server_catalog(server, st)
    tools = []
    for tname, td in st.tools.items():
        tools.append({
            "name": tname,
            "description": td.description,
            "input_schema": td.input_schema
        })
    return {"server": server, "tools": tools}


# =============================================================================
# Tool Dispatch (generic + granular)
# =============================================================================

RESERVED_QUERY_KEYS = {"args", "dryrun"}

def _collect_query_payload(request: Request) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if request is None:
        return payload
    for k, v in request.query_params.multi_items():
        if k in RESERVED_QUERY_KEYS:
            continue
        payload[k] = v
    return payload


async def ensure_connected(server: str) -> ServerState:
    st = DISCOVERY.servers.get(server)
    if not st:
        raise HTTPException(404, f"Unknown server '{server}'")
    # Opportunistic reconnect/discovery on-demand
    if not st.connected or not st.client:
        await refresh_server_catalog(server, st)
    if not st.connected or not st.client:
        raise HTTPException(503, f"Server '{server}' is not connected")
    return st


def _merge_args(primary: Optional[Dict[str, Any]], qs_json: Optional[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if primary:
        out.update(primary)
    if qs_json:
        try:
            parsed = json.loads(qs_json)
            if isinstance(parsed, dict):
                out.update(parsed)
        except Exception:
            # ignore parse error
            pass
    return out


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

    # If tool unknown, refresh once (helps first ever call)
    if tool not in st.tools:
        await refresh_server_catalog(server, st)

    # Convenience helper endpoints (schema/example/help/try) — generic
    if suffix in ("schema", "example", "help", "invoke", "try"):
        td = st.tools.get(tool)
        if suffix == "schema":
            return (td.input_schema if td else {}) or {}
        if suffix == "example":
            return {"examples": _examples_for_tool(td) if td else []}
        if suffix == "help":
            return {"name": tool, "description": (td.description if td else ""), "schema": (td.input_schema if td else {}) or {}}
        if suffix in ("invoke", "try"):
            if dryrun:
                return {"dryrun": True, "tool": tool, "args": {}}
            return await mcp_call_tool(st.client, tool, {})

    # Build final args
    final_args: Dict[str, Any] = _merge_args(body or {}, args_qs_json)

    # Inject operation/resource if granular path used
    if suffix and suffix not in ("invoke", "schema", "example", "help", "try"):
        parts = suffix.split("/")
        action = parts[0] if len(parts) >= 1 else ""
        kind = parts[1] if len(parts) >= 2 else ""
        if action and "operation" not in final_args:
            final_args["operation"] = action
        if kind and "resource" not in final_args:
            final_args["resource"] = kind

    if dryrun:
        print(f"[dryrun] server={server} tool={tool} suffix='{suffix}' final_args={final_args}", file=sys.stderr)
        return {"dryrun": True, "server": server, "tool": tool, "final_args": final_args}

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
# Per-tool helpers (generic)
# =============================================================================

@app.get("/{server}/tool/{tool}/invoke", tags=["tools", "invoke"], summary="Invoke (GET /invoke)")
async def tool_invoke_get(server: str, tool: str, dryrun: Optional[bool] = Query(False)):
    # Alias of /try
    return await do_tool_call(server=server, tool_path=f"{tool}/try", body={}, args_qs_json=None, dryrun=bool(dryrun))


@app.get("/{server}/tool/{tool}/schema", tags=["tools", "schema"], summary="Tool schema")
async def tool_schema(server: str, tool: str):
    st = DISCOVERY.servers.get(server)
    if not st:
        raise HTTPException(404, f"Unknown server '{server}'")
    if not st.connected or tool not in st.tools:
        await refresh_server_catalog(server, st)
    td = st.tools.get(tool)
    return (td.input_schema if td else {}) or {}


@app.get("/{server}/tool/{tool}/example", tags=["tools", "example"], summary="Tool example")
async def tool_example(server: str, tool: str):
    st = DISCOVERY.servers.get(server)
    if not st:
        raise HTTPException(404, f"Unknown server '{server}'")
    if not st.connected or tool not in st.tools:
        await refresh_server_catalog(server, st)
    td = st.tools.get(tool)
    return {"examples": _examples_for_tool(td) if td else []}


@app.get("/{server}/tool/{tool}/help", tags=["tools", "help"], summary="Tool help")
async def tool_help(server: str, tool: str):
    st = DISCOVERY.servers.get(server)
    if not st:
        raise HTTPException(404, f"Unknown server '{server}'")
    if not st.connected or tool not in st.tools:
        await refresh_server_catalog(server, st)
    td = st.tools.get(tool)
    return {"name": tool, "description": (td.description if td else ""), "schema": (td.input_schema if td else {}) or {}}


@app.get("/{server}/tool/{tool}/try", tags=["tools", "try"], summary="Tool zero-arg try",
         description="Calls this tool with `{}` (no arguments).")
async def tool_try(server: str, tool: str, dryrun: Optional[bool] = Query(False)):
    if dryrun:
        return {"dryrun": True, "tool": tool, "args": {}}
    st = await ensure_connected(server)
    return await mcp_call_tool(st.client, tool, {})


# =============================================================================
# OpenAPI x-* enrichment (LLM guidance)
# =============================================================================

def _examples_for_tool(td: Optional[ToolDescriptor]) -> List[Dict[str, Any]]:
    """
    Create generic examples based on the tool's input_schema (if present).
    We intentionally construct both granular-path and body-based forms.
    """
    if not td or not td.input_schema:
        # Minimal generic examples
        return [
            {"summary": "Call with no arguments (if supported)",
             "GET": f"/{{server}}/tool/{td.name}/try" if td else ""},
            {"summary": "Call with JSON body",
             "POST": f"/{{server}}/tool/{td.name}",
             "body": {"key": "value"}}
        ]

    ex = []
    props = (td.input_schema or {}).get("properties", {})
    # If 'operation' and maybe 'resource' show granular style
    op_enums = props.get("operation", {}).get("enum") if isinstance(props.get("operation"), dict) else None
    res_enums = props.get("resource", {}).get("enum") if isinstance(props.get("resource"), dict) else None

    # A) Granular examples
    if op_enums:
        op = op_enums[0]
        ex.append({
            "summary": "Granular path with action",
            "GET": f"/{{server}}/tool/{td.name}/{op}"
        })
        if res_enums:
            res = res_enums[0]
            ex.append({
                "summary": "Granular path with action + kind",
                "GET": f"/{{server}}/tool/{td.name}/{op}/{res}"
            })

    # B) Body examples (with one or two fields)
    body_ex = {}
    # Fill operation/resource if enums exist
    if op_enums:
        body_ex["operation"] = op_enums[0]
    if res_enums:
        body_ex["resource"] = res_enums[0]

    # Add up to one more property for illustration
    for k, v in props.items():
        if k in body_ex:
            continue
        # pick a simple example value
        if isinstance(v, dict):
            t = v.get("type")
            if t == "string":
                body_ex[k] = v.get("example") or "value"
            elif t == "integer":
                body_ex[k] = 1
            elif t == "number":
                body_ex[k] = 1.0
            elif t == "boolean":
                body_ex[k] = True
        if len(body_ex) >= 3:
            break

    ex.append({
        "summary": "POST with JSON body (generic)",
        "POST": f"/{{server}}/tool/{td.name}",
        "body": body_ex or {"key": "value"}
    })
    return ex


def openapi_extra_blocks() -> Dict[str, Any]:
    x_model_instructions = {
        "callDiscipline": [
            "Use GET with query or POST with JSON. If no args, POST `{}`.",
            "Granular endpoints `/{SERVER}/tool/{TOOL}/{action}[/{kind}]` map path segments to `operation` and `resource`.",
            "Unknown query params are forwarded to the tool payload.",
            "Use `dryrun=true` to preview payload without executing."
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
            "3) Prefer granular path when you know the action/kind.",
            "4) Otherwise, POST JSON body matching the tool schema."
        ],
        "errorFix": [
            "If you see 'expected a request body', try POST `{}`.",
            "If schema validation fails, check `/schema` or `/help`.",
            "Use `dryrun=true` to inspect the payload before calling."
        ],
    }

    x_mcp_tool_catalog: List[Dict[str, Any]] = []
    x_mcp_prompts: Dict[str, Any] = {}
    x_mcp_resources: Dict[str, Any] = {}

    for alias, st in DISCOVERY.servers.items():
        # include prompts/resources for visibility
        x_mcp_prompts[alias] = st.prompts
        x_mcp_resources[alias] = st.resources
        for tname, td in st.tools.items():
            x_mcp_tool_catalog.append({
                "server": alias,
                "tool": tname,
                "description": td.description,
                "schema": td.input_schema or {"type": "object"},
                "naturalExamples": _examples_for_tool(td),
            })

    return {
        "x-model-instructions": x_model_instructions,
        "x-mcp-tool-catalog": x_mcp_tool_catalog,
        "x-mcp-prompts": x_mcp_prompts,
        "x-mcp-resources": x_mcp_resources,
    }


# =============================================================================
# OpenAPI post-processor
# =============================================================================

_original_openapi = app.openapi

def custom_openapi():
    openapi_schema = _original_openapi()
    # Refresh x-* blocks on each openapi generation so they reflect live discovery
    openapi_schema.update(openapi_extra_blocks())
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
