#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP OpenAPI Bridge — Generic, Self-Discovering HTTP Forwarder (single backend)
==============================================================================
- Public API remains the same (/{server}/tool/... with granular variants)
- Builds args/operation/resource exactly as before
- Forwards to ONE upstream MCP HTTP server (MCP_RPC_URL)
- Self-discovers upstream path prefix, method, and body shape by probing
- Caches a working combo per tool for speed

Run:
  uvicorn app:app --host 0.0.0.0 --port 8080

Env:
  MCP_RPC_URL="http://upstream:8080"            # base; may or may not end with /mcp
  CORS_ALLOW_ORIGINS="*"
  HTTP_TIMEOUT=60
"""

import asyncio
import json
import os
import re
import sys
import textwrap
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlencode

import httpx
from fastapi import FastAPI, Body, Query, Path, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# =============================================================================
# Config / globals
# =============================================================================

FORWARD_BASE: Optional[str] = os.getenv("MCP_RPC_URL")
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "60"))
CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")

# Cache of per-tool routing decisions:
# tool_route_cache[tool] = {
#   "prefix": "/tool" or "" or "/mcp/tool" or "/mcp",
#   "method": "POST" or "GET",
#   "shape": "direct" | "wrapped" | "query" | "flat_query"
# }
tool_route_cache: Dict[str, Dict[str, str]] = {}

# =============================================================================
# Data structures (minimal; discovery list only for info endpoints)
# =============================================================================

class ServerConfig(BaseModel):
    alias: str
    mode: str = Field("http", description="http forwarder")
    cmd: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None

class ToolDescriptor(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None
    inferred_actions: List[str] = Field(default_factory=list)
    inferred_kinds: List[str] = Field(default_factory=list)
    convenience_params: List[str] = Field(default_factory=list)
    output_guidance: Dict[str, Any] = Field(default_factory=dict)
    usage: Dict[str, Any] = Field(default_factory=dict)
    natural_examples: List[Dict[str, Any]] = Field(default_factory=list)

class ServerState:
    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg
        self.connected = True
        self.client = None
        self.tools: Dict[str, ToolDescriptor] = {}  # optional, not required to forward

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

# =============================================================================
# Utilities (arg composition – unchanged)
# =============================================================================

COMMON_K8S_KINDS = [
    "pods", "pod", "deployments", "deployment",
    "services", "nodes", "configmaps", "secrets",
    "namespaces", "ingresses", "statefulsets", "daemonsets", "jobs",
]

CONVENIENCE_PARAMS = [
    "namespace", "name", "labels", "labelSelector", "fieldSelector",
    "container", "sinceSeconds",
]

def compose_argstring(
    args_str: Optional[str],
    namespace: Optional[str] = None,
    name: Optional[str] = None,
    labels: Optional[str] = None,
    labelSelector: Optional[str] = None,
    fieldSelector: Optional[str] = None,
    container: Optional[str] = None,
    sinceSeconds: Optional[int] = None,
) -> str:
    parts: List[str] = []
    if name:
        parts.append(str(name))
    if namespace:
        parts.extend(["-n", namespace])
    sel = labels or labelSelector
    if sel:
        parts.extend(["-l", sel])
    if fieldSelector:
        parts.extend(["--field-selector", fieldSelector])
    if container:
        parts.extend(["-c", container])
    if sinceSeconds and sinceSeconds > 0:
        parts.extend(["--since", f"{sinceSeconds}s"])
    if args_str:
        tail = str(args_str).strip()
        if tail:
            parts.append(tail)
    return " ".join(parts).strip()

async def resolve_arg_payload(
    payload: Optional[Dict[str, Any]],
    query_args_str: Optional[str],
    convenience: Dict[str, Any],
    format_hint: Optional[str]
) -> Dict[str, Any]:
    args_obj = (payload or {}).copy()
    argstring = compose_argstring(
        args_str=query_args_str or args_obj.get("args", ""),
        namespace=convenience.get("namespace", args_obj.get("namespace")),
        name=convenience.get("name", args_obj.get("name")),
        labels=convenience.get("labels", args_obj.get("labels")),
        labelSelector=convenience.get("labelSelector", args_obj.get("labelSelector")),
        fieldSelector=convenience.get("fieldSelector", args_obj.get("fieldSelector")),
        container=convenience.get("container", args_obj.get("container")),
        sinceSeconds=convenience.get("sinceSeconds", args_obj.get("sinceSeconds")),
    )
    operation = args_obj.get("operation") or convenience.get("operation")
    if format_hint in ("json", "yaml"):
        if operation in (None, "get", "api-resources", "api-versions"):
            if not re.search(r"\s\-o\s+(json|yaml)\b", argstring):
                argstring = (argstring + f" -o {format_hint}").strip()
    if argstring:
        args_obj["args"] = argstring
    return args_obj

# =============================================================================
# FastAPI app
# =============================================================================

app = FastAPI(
    title="MCP OpenAPI Bridge (self-discovering HTTP forwarder)",
    version="3.2.0",
    description=textwrap.dedent("""\
        A generic OpenAPI façade for a single MCP HTTP server.
        Builds everything from the upstream by probing and caching the winning route/method/body per tool.
    """),
    servers=[{"url": "http://localhost:8080"}],
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Normalize odd incoming paths (placeholder aliases, spaces, missing server)
@app.middleware("http")
async def normalize_odd_paths(request: Request, call_next):
    raw_path = request.scope.get("path") or ""
    decoded = unquote(raw_path)

    # Map placeholder aliases to /mcp/...
    for bad in ("<server-alias>", "<server>", "server-alias", "server"):
        prefix = f"/{bad}/"
        if decoded.startswith(prefix):
            decoded = "/mcp/" + decoded[len(prefix):]
            break

    # Fix "get pods" -> "get/pods"
    marker = "/tool/"
    if marker in decoded:
        head, tail = decoded.split(marker, 1)
        tail = tail.replace("  ", " ").strip()
        if " " in tail:
            tail = "/".join([p for p in tail.split(" ") if p])
        decoded = head + marker + tail

    # If starts with /tool/, prefix with /mcp
    if decoded.startswith("/tool/"):
        decoded = "/mcp" + decoded

    # If first segment unknown, force to /mcp
    if decoded.startswith("/") and "/tool/" in decoded:
        first = decoded.split("/", 2)[1]
        if first and first not in DISCOVERY.servers:
            decoded = "/mcp/" + decoded.split("/", 2)[2]

    if decoded != raw_path:
        request.scope["path"] = decoded
    return await call_next(request)

@app.on_event("startup")
async def on_startup():
    global FORWARD_BASE
    if not FORWARD_BASE:
        raise RuntimeError("MCP_RPC_URL is required (e.g., http://upstream:8080 or http://upstream:8080/mcp)")
    DISCOVERY.servers["mcp"] = ServerState(ServerConfig(alias="mcp", mode="http"))

    # Try a light-weight probe of upstream OpenAPI (optional)
    try:
        await probe_upstream_openapi()
    except Exception as e:
        print(f"[probe] upstream openapi probe failed (non-fatal): {e}", file=sys.stderr)

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

@app.post("/discover", tags=["discovery"], summary="Discover Endpoint")
async def discover_endpoint(wait: Optional[int] = Query(0, description="Seconds to wait (max 10) for discovery")):
    wait = max(0, min(int(wait or 0), 10))
    if wait:
        await asyncio.sleep(wait)
    # Nothing to do; discovery is on-demand via probe/cache
    return {"servers": DISCOVERY.list_servers()}

@app.get("/discovery/status", tags=["discovery"], summary="Discovery Status")
async def discovery_status():
    return {"servers": DISCOVERY.list_servers()}

# =============================================================================
# Upstream probing / forwarding core
# =============================================================================

async def probe_upstream_openapi():
    """
    Try to fetch upstream openapi.json from several candidate paths to learn a likely prefix.
    Non-fatal; forwarding still uses trial-and-error per tool if this yields nothing.
    """
    candidates = prefix_candidates()
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        for px in candidates:
            for path in ("/openapi.json", "/openapi", "/docs/openapi.json"):
                url = f"{px}{path}"
                try:
                    r = await client.get(url)
                    if r.status_code == 200 and "application/json" in r.headers.get("content-type", ""):
                        data = r.json()
                        # Heuristic: if there's a '/tool/{...}' path, prefer that prefix
                        if isinstance(data, dict) and "paths" in data and any("/tool/" in p for p in data["paths"].keys()):
                            # Cache a default for unknown tools to try this first
                            tool_route_cache["_default"] = {"prefix": "/tool", "method": "POST", "shape": "direct"}
                            print(f"[probe] upstream openapi suggests '/tool' under prefix {px}", file=sys.stderr)
                            return
                except Exception:
                    continue
    # no findings; leave cache empty

def prefix_candidates() -> List[str]:
    """Generate base prefix candidates from MCP_RPC_URL (with and without /mcp)."""
    base = (FORWARD_BASE or "").rstrip("/")
    if not base:
        return []
    cands = [base]
    if base.endswith("/mcp"):
        cands.append(base[:-4])  # without /mcp
    else:
        cands.append(base + "/mcp")
    # de-dup while preserving order
    seen, out = set(), []
    for c in cands:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out

async def try_forward_once(
    base: str,
    prefix: str,
    tool_path: str,
    method: str,
    shape: str,
    payload: Dict[str, Any],
    client: httpx.AsyncClient,
) -> Optional[httpx.Response]:
    """
    Attempt one forward call with the given (prefix, method, shape).
    Return response if 2xx, else None.
    """
    # Build path: prefix may be "", "/tool", "/mcp", "/mcp/tool"
    # If prefix already contains "/tool", don't append it again.
    if "/tool" in prefix:
        path = f"{prefix.rstrip('/')}/{tool_path.lstrip('/')}"
    else:
        path = f"{prefix.rstrip('/')}/tool/{tool_path.lstrip('/')}" if prefix else f"/tool/{tool_path.lstrip('/')}"
    url = f"{base.rstrip('/')}{path}"

    try:
        if method == "POST":
            if shape == "direct":
                resp = await client.post(url, json=payload)
            elif shape == "wrapped":
                resp = await client.post(url, json={"args": payload})
            else:
                return None
        else:  # GET
            if shape == "query":
                q = urlencode({"args": json.dumps(payload)})
                resp = await client.get(f"{url}?{q}")
            elif shape == "flat_query":
                # Flatten top-level keys as query params (values JSON-encoded if not str/int)
                qparams = {}
                for k, v in payload.items():
                    if isinstance(v, (str, int, float, bool)) or v is None:
                        qparams[k] = v
                    else:
                        qparams[k] = json.dumps(v)
                resp = await client.get(f"{url}?{urlencode(qparams)}")
            else:
                return None
    except Exception as e:
        print(f"[forward-try] {method} {url} shape={shape} error={e}", file=sys.stderr)
        return None

    if 200 <= resp.status_code < 300:
        return resp
    # Let 404/405/400/415 fall through so we can try other shapes
    print(f"[forward-try] {method} {url} shape={shape} -> {resp.status_code}", file=sys.stderr)
    return None

async def forward_with_discovery(tool: str, tool_path: str, payload: Dict[str, Any]) -> Any:
    """
    Forward a tool call; if we lack a cached route for this tool,
    probe prefixes/methods/body-shapes until something works, then cache it.
    """
    # Use cached decision if present
    route = tool_route_cache.get(tool) or tool_route_cache.get("_default")
    base_candidates = prefix_candidates()
    shapes_by_method = {
        "POST": ["direct", "wrapped"],
        "GET": ["query", "flat_query"],
    }
    prefix_variants = ["", "/tool", "/mcp", "/mcp/tool"]  # we will dedupe against path construction rules

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        # Fast path if route cached
        if route:
            resp = await try_forward_once(
                base_candidates[0],
                route["prefix"],
                tool_path,
                route["method"],
                route["shape"],
                payload,
                client,
            )
            if resp is not None:
                return await normalize_upstream_response(resp)

        # Otherwise probe permutations
        for base in base_candidates:
            # Try obvious prefixes first, then others
            ordered_prefixes: List[str] = []
            # If base already ends with /mcp, try with '' and '/tool' first (which become /tool/...),
            # then '/mcp' variants in case upstream wants /mcp/tool/...
            if base.endswith("/mcp"):
                ordered_prefixes = ["", "/tool", "/mcp", "/mcp/tool"]
            else:
                ordered_prefixes = ["", "/tool", "/mcp", "/mcp/tool"]

            for method in ("POST", "GET"):
                for shape in shapes_by_method[method]:
                    for prefix in ordered_prefixes:
                        resp = await try_forward_once(base, prefix, tool_path, method, shape, payload, client)
                        if resp is not None:
                            # Cache and return
                            tool_route_cache[tool] = {"prefix": prefix, "method": method, "shape": shape}
                            print(f"[route-cache] tool={tool} -> {tool_route_cache[tool]}", file=sys.stderr)
                            return await normalize_upstream_response(resp)

        # If we reach here, nothing worked
        raise HTTPException(502, f"Failed to discover a working upstream route for tool '{tool}'")

def decode_upstream_json(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except Exception:
        return {"status": "ok", "raw": resp.text}

async def normalize_upstream_response(resp: httpx.Response) -> Any:
    ct = resp.headers.get("content-type", "")
    if resp.status_code >= 400:
        # surface upstream error
        try:
            data = resp.json()
        except Exception:
            data = {"error": resp.text}
        raise HTTPException(resp.status_code, data)
    if "application/json" in ct:
        return decode_upstream_json(resp)
    return {"status": "ok", "raw": resp.text}

# =============================================================================
# Core call path
# =============================================================================

async def ensure_connected(server: str) -> ServerState:
    if not FORWARD_BASE:
        raise HTTPException(503, "MCP_RPC_URL not set; configure the upstream MCP HTTP server base URL.")
    st = DISCOVERY.servers.get("mcp")
    if not st:
        st = ServerState(ServerConfig(alias="mcp", mode="http"))
        DISCOVERY.servers["mcp"] = st
    return st

async def do_tool_call(
    server: str,
    tool_path: str,
    body: Optional[Dict[str, Any]],
    query_args_fallback: Optional[str],
    dryrun: bool,
    format_hint: Optional[str],
) -> Any:
    await ensure_connected(server)

    components = tool_path.split("/")
    tool = components[0]
    suffix = "/".join(components[1:]) if len(components) > 1 else ""

    # Build args from URL suffix (action/kind)
    args: Dict[str, Any] = {}
    convenience: Dict[str, Any] = {}
    if suffix and suffix not in ("invoke", "schema", "example", "help", "try"):
        parts = suffix.split("/")
        action = parts[0]
        kind = parts[1] if len(parts) > 1 else ""
        args["operation"] = action
        args["resource"] = kind

    body = body or {}
    if "args" in body and isinstance(body["args"], dict) and any(k in body["args"] for k in ["operation", "resource", "args"]):
        args.update(body.get("args", {}))
    else:
        args.update(body)

    for p in CONVENIENCE_PARAMS + ["operation", "resource"]:
        if p in body:
            convenience[p] = body[p]

    final_args = await resolve_arg_payload(args, query_args_fallback, convenience, format_hint)

    if dryrun:
        print(f"[dryrun] tool={tool} suffix='{suffix}' final_args={final_args}", file=sys.stderr)
        return {"dryrun": True, "server": server, "tool": tool, "final_args": final_args}

    # Forward with discovery (try & cache)
    print(f"[forward] tool={tool} path='{tool_path}' final_args={final_args}", file=sys.stderr)
    return await forward_with_discovery(tool, tool_path, final_args)

# =============================================================================
# Public routes (unchanged interface)
# =============================================================================

@app.post(
    "/{server}/tool/{tool_path:path}",
    tags=["tools"],
    summary="Tool Dispatch",
    responses={200: {"description": "Successful Response", "content": {"application/json": {}}}},
)
async def tool_dispatch_post(
    server: str = Path(..., description="Server alias (ignored; single-backend)"),
    tool_path: str = Path(..., description="Tool or tool path like 'kubectl_resources/get/pods'"),
    args: Optional[str] = Query(None, description="JSON-encoded args fallback"),
    dryrun: Optional[bool] = Query(False, description="Preview composed call only"),
    format: Optional[str] = Query(None, description="Output preference: json|yaml|text"),
    body: Optional[Dict[str, Any]] = Body(None),
):
    qargs = None
    if args:
        try:
            qargs = json.loads(args)
        except Exception:
            qargs = args
    result = await do_tool_call(
        server,
        tool_path,
        body or (qargs if isinstance(qargs, dict) else None),
        None,
        bool(dryrun),
        format
    )
    return JSONResponse(result)

@app.get(
    "/{server}/tool/{tool_path:path}",
    tags=["tools"],
    summary="Tool Dispatch",
    responses={200: {"description": "Successful Response", "content": {"application/json": {}}}},
)
async def tool_dispatch_get(
    server: str = Path(..., description="Server alias (ignored; single-backend)"),
    tool_path: str = Path(..., description="Tool or tool path like 'kubectl_resources/get/pods'"),
    args: Optional[str] = Query(None, description="JSON-encoded args fallback"),
    dryrun: Optional[bool] = Query(False, description="Preview composed call only"),
    format: Optional[str] = Query(None, description="Output preference: json|yaml|text"),
    namespace: Optional[str] = Query(None),
    name: Optional[str] = Query(None),
    labels: Optional[str] = Query(None),
    labelSelector: Optional[str] = Query(None),
    fieldSelector: Optional[str] = Query(None),
    container: Optional[str] = Query(None),
    sinceSeconds: Optional[int] = Query(None),
):
    qargs_fallback = None
    if args:
        try:
            data = json.loads(args)
            if isinstance(data, dict):
                body = data
            else:
                body = {}
                qargs_fallback = args
        except Exception:
            body = {}
            qargs_fallback = args
    else:
        body = {}

    for k, v in {
        "namespace": namespace, "name": name, "labels": labels,
        "labelSelector": labelSelector, "fieldSelector": fieldSelector,
        "container": container, "sinceSeconds": sinceSeconds,
    }.items():
        if v is not None:
            body[k] = v

    result = await do_tool_call(server, tool_path, body, qargs_fallback, bool(dryrun), format)
    return JSONResponse(result)

# Granular variants (POST/GET with action/kind)
@app.post("/{server}/tool/{tool}/{action}/{kind}", tags=["tools"], summary="Granular Tool Dispatch (POST)")
async def granular_post_kind(
    server: str, tool: str, action: str, kind: str,
    args: Optional[str] = Query(None),
    dryrun: Optional[bool] = Query(False),
    format: Optional[str] = Query(None),
    body: Optional[Dict[str, Any]] = Body(None),
    namespace: Optional[str] = Query(None),
    name: Optional[str] = Query(None),
    labels: Optional[str] = Query(None),
    labelSelector: Optional[str] = Query(None),
    fieldSelector: Optional[str] = Query(None),
    container: Optional[str] = Query(None),
    sinceSeconds: Optional[int] = Query(None),
):
    body = (body or {}).copy()
    for k, v in {
        "namespace": namespace, "name": name, "labels": labels,
        "labelSelector": labelSelector, "fieldSelector": fieldSelector,
        "container": container, "sinceSeconds": sinceSeconds,
    }.items():
        if v is not None:
            body[k] = v
    if args:
        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict):
                body.update(parsed)
        except Exception:
            pass
    return await tool_dispatch_post(server, f"{tool}/{action}/{kind}", None, dryrun, format, body)

@app.get("/{server}/tool/{tool}/{action}/{kind}", tags=["tools"], summary="Granular Tool Dispatch (GET)")
async def granular_get_kind(
    server: str, tool: str, action: str, kind: str,
    args: Optional[str] = Query(None),
    dryrun: Optional[bool] = Query(False),
    format: Optional[str] = Query(None),
    namespace: Optional[str] = Query(None),
    name: Optional[str] = Query(None),
    labels: Optional[str] = Query(None),
    labelSelector: Optional[str] = Query(None),
    fieldSelector: Optional[str] = Query(None),
    container: Optional[str] = Query(None),
    sinceSeconds: Optional[int] = Query(None),
):
    return await tool_dispatch_get(
        server, f"{tool}/{action}/{kind}", args, dryrun, format,
        namespace, name, labels, labelSelector, fieldSelector, container, sinceSeconds
    )

@app.post("/{server}/tool/{tool}/{action}", tags=["tools"], summary="Granular Tool Dispatch (POST, action only)")
async def granular_post_action(
    server: str, tool: str, action: str,
    args: Optional[str] = Query(None),
    dryrun: Optional[bool] = Query(False),
    format: Optional[str] = Query(None),
    body: Optional[Dict[str, Any]] = Body(None),
    namespace: Optional[str] = Query(None),
    name: Optional[str] = Query(None),
    labels: Optional[str] = Query(None),
    labelSelector: Optional[str] = Query(None),
    fieldSelector: Optional[str] = Query(None),
    container: Optional[str] = Query(None),
    sinceSeconds: Optional[int] = Query(None),
):
    body = (body or {}).copy()
    for k, v in {
        "namespace": namespace, "name": name, "labels": labels,
        "labelSelector": labelSelector, "fieldSelector": fieldSelector,
        "container": container, "sinceSeconds": sinceSeconds,
    }.items():
        if v is not None:
            body[k] = v
    if args:
        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict):
                body.update(parsed)
        except Exception:
            pass
    return await tool_dispatch_post(server, f"{tool}/{action}", None, dryrun, format, body)

@app.get("/{server}/tool/{tool}/{action}", tags=["tools"], summary="Granular Tool Dispatch (GET, action only)")
async def granular_get_action(
    server: str, tool: str, action: str,
    args: Optional[str] = Query(None),
    dryrun: Optional[bool] = Query(False),
    format: Optional[str] = Query(None),
    namespace: Optional[str] = Query(None),
    name: Optional[str] = Query(None),
    labels: Optional[str] = Query(None),
    labelSelector: Optional[str] = Query(None),
    fieldSelector: Optional[str] = Query(None),
    container: Optional[str] = Query(None),
    sinceSeconds: Optional[int] = Query(None),
):
    return await tool_dispatch_get(
        server, f"{tool}/{action}", args, dryrun, format,
        namespace, name, labels, labelSelector, fieldSelector, container, sinceSeconds
    )

# Convenience per-tool helpers (no upstream schema read; they’re stubs)
@app.get("/mcp/tool/{tool}/invoke", tags=["tools", "invoke"], summary="Invoke (GET /invoke)")
async def tool_invoke_get(tool: str, **kwargs):
    return await tool_dispatch_get(server="mcp", tool_path=f"{tool}/invoke", **kwargs)

@app.get("/mcp/tool/{tool}/schema", tags=["tools", "schema"], summary="Tool schema")
async def tool_schema(tool: str):
    return {}

@app.get("/mcp/tool/{tool}/example", tags=["tools", "example"], summary="Tool example")
async def tool_example(tool: str):
    return {"naturalExamples": []}

@app.get("/mcp/tool/{tool}/help", tags=["tools", "help"], summary="Tool help")
async def tool_help(tool: str):
    return {
        "name": tool,
        "description": "",
        "inferred_actions": [],
        "inferred_kinds": [],
        "convenience_params": CONVENIENCE_PARAMS,
        "outputGuidance": {},
        "usage": {}
    }

@app.get("/mcp/tool/{tool}/try", tags=["tools", "try"], summary="Tool zero-arg try",
         description="Calls this tool with `{}` (no arguments).")
async def tool_try(tool: str, dryrun: Optional[bool] = Query(False)):
    return await tool_dispatch_get(server="mcp", tool_path=f"{tool}/try", dryrun=dryrun)

# =============================================================================
# OpenAPI x-* enrichment (brief)
# =============================================================================

def openapi_extra_blocks() -> Dict[str, Any]:
    x_model_instructions = {
        "callDiscipline": [
            "Use GET with query or POST with JSON. If no args, POST `{}`.",
            "Granular endpoints `/{SERVER}/tool/{TOOL}/{action}[/{kind}]` map path segments to fields.",
            "Unknown query params are forwarded via arg composition.",
            "Use `dryrun=true` to preview the composed payload.",
        ],
        "typicalFlow": [
            "1) Read `/openapi.json` and `/discovery/status`.",
            "2) Call `/{server}/tool/{tool}/{action}/{kind}` when possible.",
            "3) Prefer `format=json` for machine-readable results.",
        ],
    }
    return {
        "x-model-instructions": x_model_instructions,
        "x-mcp-tool-catalog": [],
        "x-mcp-prompts": {},
        "x-mcp-resources": {}
    }

_original_openapi = app.openapi
def custom_openapi():
    openapi_schema = _original_openapi()
    openapi_schema.update({**openapi_extra_blocks()})
    app.openapi_schema = openapi_schema
    return app.openapi_schema
app.openapi = custom_openapi
