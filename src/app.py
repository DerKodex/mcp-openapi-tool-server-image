#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# --- Optional MCP stdio client (generic) --------------------------------------
try:
    from mcp.client.stdio import stdio_client
    from mcp.types import TextContent
    MCP_AVAILABLE = True
except Exception:
    MCP_AVAILABLE = False


# =============================================================================
# Models / State (generic)
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
        self.last_error: Optional[str] = None

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
            "last_error": st.last_error,
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
# MCP integration (generic)
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
# FastAPI app
# =============================================================================

app = FastAPI(
    title="MCP OpenAPI Bridge (Generic, Self-Discovering)",
    version="4.2.0",
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
# Startup / warm discovery
# =============================================================================

@app.on_event("startup")
async def on_startup():
    servers_cfg = getenv_json("MCP_SERVERS", None) or []
    for cfg in servers_cfg:
        cfg_obj = ServerConfig(**cfg)
        DISCOVERY.servers[cfg_obj.alias] = ServerState(cfg_obj)

    # Immediate discover (with a small wait to give stdio servers time to spawn)
    initial_wait = int(os.getenv("MCP_DISCOVERY_WAIT", "2") or "2")
    await do_discover(wait_seconds=min(max(initial_wait, 0), 10))

    # Warm-up loop: retry a few times in background to populate tools early
    asyncio.create_task(_warmup_discovery())

async def _warmup_discovery():
    # Try up to 6 times over ~18s if we still have zero tools
    for _ in range(6):
        if any(st.tools for st in DISCOVERY.servers.values()):
            return
        await do_discover(wait_seconds=2)
        await asyncio.sleep(1)


async def connect_and_cache(alias: str, st: ServerState):
    if st.connected and st.client:
        return
    try:
        if st.cfg.mode == "stdio":
            st.client = await mcp_connect_stdio(st.cfg)
            st.connected = True
            st.last_error = None
        else:
            raise RuntimeError(f"Unsupported mode '{st.cfg.mode}'")
    except Exception as e:
        st.connected = False
        st.client = None
        st.last_error = str(e)

async def refresh_server_catalog(alias: str, st: ServerState):
    try:
        await connect_and_cache(alias, st)
        if not st.connected or not st.client:
            return
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
        st.last_error = None
    except Exception as e:
        st.connected = False
        st.client = None
        st.tools.clear()
        st.prompts = []
        st.resources = []
        st.last_error = str(e)
        print(f"[discover] '{alias}' failed: {e}", file=sys.stderr)

async def do_discover(wait_seconds: int = 0) -> Dict[str, Any]:
    tasks = [refresh_server_catalog(alias, st) for alias, st in DISCOVERY.servers.items()]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    if wait_seconds > 0:
        await asyncio.sleep(min(wait_seconds, 10))
    return {"servers": DISCOVERY.list_servers()}


# =============================================================================
# Health / info
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
    if not st.connected or not st.tools:
        await refresh_server_catalog(server, st)
    tools = []
    for tname, td in st.tools.items():
        tools.append({"name": tname, "description": td.description, "input_schema": td.input_schema})
    return {"server": server, "tools": tools}


# =============================================================================
# Tool dispatch (generic + granular)
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
    if not st.connected or not st.client:
        await refresh_server_catalog(server, st)
    if not st.connected or not st.client:
        raise HTTPException(503, f"Server '{server}' is not connected: {st.last_error or 'no session'}")
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

    if tool not in st.tools:
        await refresh_server_catalog(server, st)

    # helper suffixes (generic)
    if suffix in ("schema", "example", "help", "invoke", "try"):
        td = st.tools.get(tool)
        if suffix == "schema":
            return (td.input_schema if td else {}) or {}
        if suffix == "example":
            return {"examples": _examples_for_tool(td, server, tool)}
        if suffix == "help":
            return {"name": tool, "description": (td.description if td else ""), "schema": (td.input_schema if td else {}) or {}}
        if suffix in ("invoke", "try"):
            if dryrun:
                return {"dryrun": True, "tool": tool, "args": {}}
            return await mcp_call_tool(st.client, tool, {})

    final_args: Dict[str, Any] = _merge_args(body or {}, args_qs_json)

    # Infer operation/resource from granular path
    if suffix and suffix not in ("invoke", "schema", "example", "help", "try"):
        parts = suffix.split("/")
        action = parts[0] if len(parts) >= 1 else ""
        kind = parts[1] if len(parts) >= 2 else ""
        if action and "operation" not in final_args:
            final_args["operation"] = action
        if kind and "resource" not in final_args:
            final_args["resource"] = kind

    if dryrun:
        print(f"[dryrun] {server} -> {tool_path} :: {final_args}", file=sys.stderr)
        return {"dryrun": True, "server": server, "tool": tool, "tool_path": tool_path, "final_args": final_args}

    try:
        print(f"[invoke] {server} -> {tool_path} :: {final_args}", file=sys.stderr)
        return await mcp_call_tool(st.client, tool, final_args)
    except Exception as e:
        raise HTTPException(502, f"Tool invocation failed for '{tool_path}': {e}")


# Catch-all (generic)

@app.post("/{server}/tool/{tool_path:path}", tags=["tools"], summary="Tool Dispatch")
async def tool_dispatch_post(
    request: Request,
    server: str = Path(..., description="Server alias (e.g., 'mcp')"),
    tool_path: str = Path(..., description="Tool or tool path like 'some_tool/do/thing'"),
    args: Optional[str] = Query(None, description="JSON-encoded args fallback"),
    dryrun: Optional[bool] = Query(False, description="If true, returns the would-be call without executing"),
    body: Optional[Dict[str, Any]] = Body(None),
):
    forwarded = _collect_query_payload(request)
    merged_body = dict(forwarded)
    if body:
        merged_body.update(body)
    result = await do_tool_call(server, tool_path, merged_body, args, bool(dryrun))
    return JSONResponse(result)

@app.get("/{server}/tool/{tool_path:path}", tags=["tools"], summary="Tool Dispatch")
async def tool_dispatch_get(
    request: Request,
    server: str = Path(..., description="Server alias (e.g., 'mcp')"),
    tool_path: str = Path(..., description="Tool or tool path like 'some_tool/do/thing'"),
    args: Optional[str] = Query(None, description="JSON-encoded args fallback"),
    dryrun: Optional[bool] = Query(False, description="If true, returns the would-be call without executing"),
):
    forwarded = _collect_query_payload(request)
    result = await do_tool_call(server, tool_path, forwarded, args, bool(dryrun))
    return JSONResponse(result)


# Explicit granular routes (avoid 404s on common patterns)

@app.post("/{server}/tool/{tool}/{action}/{kind}", tags=["tools"], summary="Granular Tool Dispatch (POST)")
async def granular_post_kind(
    request: Request,
    server: str, tool: str, action: str, kind: str,
    args: Optional[str] = Query(None),
    dryrun: Optional[bool] = Query(False),
    body: Optional[Dict[str, Any]] = Body(None),
):
    forwarded = _collect_query_payload(request)
    merged_body = dict(forwarded)
    if body:
        merged_body.update(body)
    return await tool_dispatch_post(request, server, f"{tool}/{action}/{kind}", args, dryrun, merged_body)

@app.get("/{server}/tool/{tool}/{action}/{kind}", tags=["tools"], summary="Granular Tool Dispatch (GET)")
async def granular_get_kind(
    request: Request,
    server: str, tool: str, action: str, kind: str,
    args: Optional[str] = Query(None),
    dryrun: Optional[bool] = Query(False),
):
    return await tool_dispatch_get(request, server, f"{tool}/{action}/{kind}", args, dryrun)

@app.post("/{server}/tool/{tool}/{action}", tags=["tools"], summary="Granular Tool Dispatch (POST)")
async def granular_post_action(
    request: Request,
    server: str, tool: str, action: str,
    args: Optional[str] = Query(None),
    dryrun: Optional[bool] = Query(False),
    body: Optional[Dict[str, Any]] = Body(None),
):
    forwarded = _collect_query_payload(request)
    merged_body = dict(forwarded)
    if body:
        merged_body.update(body)
    return await tool_dispatch_post(request, server, f"{tool}/{action}", args, dryrun, merged_body)

@app.get("/{server}/tool/{tool}/{action}", tags=["tools"], summary="Granular Tool Dispatch (GET)")
async def granular_get_action(
    request: Request,
    server: str, tool: str, action: str,
    args: Optional[str] = Query(None),
    dryrun: Optional[bool] = Query(False),
):
    return await tool_dispatch_get(request, server, f"{tool}/{action}", args, dryrun)


# Per-tool helpers

@app.get("/{server}/tool/{tool}/invoke", tags=["tools", "invoke"], summary="Invoke (GET /invoke)")
async def tool_invoke_get(server: str, tool: str, dryrun: Optional[bool] = Query(False)):
    return await do_tool_call(server, f"{tool}/try", {}, None, bool(dryrun))

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
    return {"examples": _examples_for_tool(td, server, tool)}

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
# OpenAPI enrichment
# =============================================================================

def _examples_for_tool(td: Optional[ToolDescriptor], server_alias: str, tool_name: str) -> List[Dict[str, Any]]:
    # If we have no schema, still give concrete paths to coax the LLM.
    examples: List[Dict[str, Any]] = [
        {"summary": "Zero-arg try", "GET": f"/{server_alias}/tool/{tool_name}/try"},
        {"summary": "POST with JSON body", "POST": f"/{server_alias}/tool/{tool_name}", "body": {"key": "value"}},
    ]

    props = (td.input_schema or {}).get("properties", {}) if td and td.input_schema else {}
    op_enum = props.get("operation", {}).get("enum") if isinstance(props.get("operation"), dict) else None
    res_enum = props.get("resource", {}).get("enum") if isinstance(props.get("resource"), dict) else None

    if op_enum:
        ex_op = op_enum[0]
        examples.append({"summary": "Granular action", "GET": f"/{server_alias}/tool/{tool_name}/{ex_op}"})
        if res_enum:
            ex_res = res_enum[0]
            examples.append({"summary": "Granular action+kind", "GET": f"/{server_alias}/tool/{tool_name}/{ex_op}/{ex_res}"})
            # Show a query param pattern that many tools accept (namespace as a generic example)
            examples.append({"summary": "Granular with query param",
                             "GET": f"/{server_alias}/tool/{tool_name}/{ex_op}/{ex_res}?namespace=example"})
    return examples

def _fallback_examples() -> List[Dict[str, Any]]:
    # These are generic but concrete — they *intentionally* show a k/v namespace pattern
    # because many resource-oriented tools accept it. This is to prompt the LLM to try granular calls.
    return [
        {"summary": "Discover tools", "GET": "/{server}/tools/list"},
        {"summary": "Get a tool's schema", "GET": "/{server}/tool/{tool}/schema"},
        {"summary": "Try a tool with no args", "GET": "/{server}/tool/{tool}/try"},
        {"summary": "Granular example (action)", "GET": "/{server}/tool/{tool}/get"},
        {"summary": "Granular example (action/kind)", "GET": "/{server}/tool/{tool}/get/{kind}"},
        {"summary": "Granular + query param (typical pattern)", "GET": "/{server}/tool/{tool}/get/{kind}?namespace=vault"},
        {"summary": "POST body example", "POST": "/{server}/tool/{tool}", "body": {"operation": "get", "resource": "{kind}", "namespace": "vault"}},
    ]

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
            "4) Otherwise POST a JSON body matching the tool schema."
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
        x_mcp_prompts[alias] = st.prompts
        x_mcp_resources[alias] = st.resources
        for tname, td in st.tools.items():
            x_mcp_tool_catalog.append({
                "server": alias,
                "tool": tname,
                "description": td.description,
                "schema": td.input_schema or {"type": "object"},
                "naturalExamples": _examples_for_tool(td, alias, tname),
            })

    # If discovery is empty, provide fallback examples to push the LLM to try granular calls
    if not x_mcp_tool_catalog:
        x_mcp_tool_catalog.append({
            "server": "{server}",
            "tool": "{tool}",
            "description": "Fallback generic examples (replace placeholders).",
            "schema": {"type": "object"},
            "naturalExamples": _fallback_examples(),
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
    openapi_schema.update(openapi_extra_blocks())
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
