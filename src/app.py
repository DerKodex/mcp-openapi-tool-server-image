# [unchanged header + imports up to httpx import]
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP OpenAPI Bridge — Generic, Self-Discovering (resilient)
----------------------------------------------------------
- Discovers MCP servers + tools at runtime
- Exposes generic and granular HTTP endpoints per tool
- Calls succeed even if discovery hasn't populated a tool yet
- Rich OpenAPI with x-* guidance and examples

Run:
  uvicorn app:app --host 0.0.0.0 --port 8080

Env:
  MCP_SERVERS='[{"alias":"mcp","mode":"stdio","cmd":["/path/to/mcp-server"]}]'
  MCP_DISCOVERY_WAIT=2
  MCP_FORWARD_URL='http://mcp-upstream:8080'   # OPTIONAL: REST bridge fallback
  MCP_RPC_URL='http://mcp-server:8080/mcp'     # OPTIONAL: raw MCP HTTP RPC
"""

import asyncio
import json
import os
import re
import sys
import textwrap
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

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
    inferred_actions: List[str] = Field(default_factory=list)
    inferred_kinds: List[str] = Field(default_factory=list)
    convenience_params: List[str] = Field(default_factory=list)
    output_guidance: Dict[str, Any] = Field(default_factory=dict)
    usage: Dict[str, Any] = Field(default_factory=dict)
    natural_examples: List[Dict[str, Any]] = Field(default_factory=list)


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

COMMON_K8S_KINDS = [
    "pods", "pod", "deployments", "deployment",
    "services", "nodes", "configmaps", "secrets",
    "namespaces", "ingresses", "statefulsets", "daemonsets", "jobs"
]

CONVENIENCE_PARAMS = [
    "namespace", "name", "labels", "labelSelector", "fieldSelector",
    "container", "sinceSeconds"
]

def getenv_json(name: str, default: Any) -> Any:
    val = os.getenv(name)
    if not val:
        return default
    try:
        return json.loads(val)
    except Exception:
        return default


def parse_actions_from_description(desc: str) -> List[str]:
    if not desc:
        return []
    actions = set()
    for line in desc.splitlines():
        m = re.search(r"^\s*-\s*([a-zA-Z0-9_-]+)\s*[:\-]", line)
        if m:
            actions.add(m.group(1).strip())
    preferred = ["get", "describe", "logs", "events", "top", "exec", "cp",
                 "cluster-info", "api-resources", "api-versions", "explain",
                 "diff", "auth", "config"]
    ordered = [a for a in preferred if a in actions]
    for a in actions:
        if a not in ordered:
            ordered.append(a)
    return ordered


def infer_kinds_from_description(desc: str) -> List[str]:
    if not desc:
        return []
    kinds = set()
    for k in COMMON_K8S_KINDS:
        if re.search(rf"\b{k}\b", desc):
            kinds.add(k)
    return [k for k in COMMON_K8S_KINDS if k in kinds]


def schema_enums(schema: Dict[str, Any], field: str) -> List[str]:
    try:
        props = schema.get("properties", {})
        if field in props and "enum" in props[field]:
            return [str(v) for v in props[field]["enum"]]
    except Exception:
        pass
    return []


def build_k8s_output_guidance(tool_name: str) -> Dict[str, Any]:
    return {
        "preferredFormats": ["json", "yaml", "text"],
        "notes": [
            "For 'get' style operations, prefer `format=json` to inject `-o json` so output is machine-readable.",
            "For 'describe', kubectl emits human text (no -o json). Treat it as unstructured text.",
            "For 'logs', output is line-oriented text; do not expect JSON.",
            "For 'api-resources'/'api-versions', `format=json` works when supported; otherwise text table."
        ],
        "parsingHints": {
            "describe/pods (text)": {
                "extract": [
                    {"field": "name", "regex": r"^Name:\s+([^\s]+)"},
                    {"field": "namespace", "regex": r"^Namespace:\s+([^\s]+)"},
                    {"field": "node", "regex": r"^Node:\s+([^\s]+)"},
                    {"field": "podIP", "regex": r"^IP:\s+([^\s]+)"},
                    {"field": "phase", "regex": r"^Status:\s+([A-Za-z]+)"},
                ],
                "lineMode": True
            },
            "logs (text)": {
                "extract": [
                    {"field": "lines", "note": "Split by newline; may contain timestamps."}
                ]
            }
        }
    }


def k8s_natural_examples(tool_name: str) -> List[Dict[str, Any]]:
    return [
        {
            "intent": "List all pods in the vault namespace (machine-readable JSON)",
            "calls": [
                {"GET": f"/mcp/tool/{tool_name}/get/pods?namespace=vault&format=json"},
                {"POST": f"/mcp/tool/{tool_name}/get/pods", "body": {"namespace": "vault"}, "query": {"format": "json"}},
            ],
            "notes": ["Use format=json to inject '-o json' when supported (e.g., kubectl get)."]
        },
        {
            "intent": "Describe a pod (human text)",
            "calls": [
                {"GET": f"/mcp/tool/{tool_name}/describe/pods?namespace=vault&name=vault-0"},
            ],
            "notes": ["'describe' outputs unstructured text; see x-outputGuidance.parsingHints."]
        },
        {
            "intent": "List namespaces",
            "calls": [
                {"GET": f"/mcp/tool/{tool_name}/get/namespaces?format=json"},
            ],
            "notes": ["Prefer JSON for machine parsing."]
        }
    ]


def is_k8s_tool(name: str, desc: str) -> bool:
    return bool(re.search(r"\bkubectl\b", desc or "") or name.startswith("kubectl_"))


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


# =============================================================================
# MCP integration (stdio)
# =============================================================================

async def mcp_connect_stdio(cfg: ServerConfig):
    """
    Create an MCP stdio client supporting both SDK shapes:
    - Newer SDKs: stdio_client(...) -> async context manager (enter it)
    - Older SDKs: stdio_client(...) -> awaitable (await it)
    Also supports older SDKs that don't accept env= on stdio_client.
    """
    if not MCP_AVAILABLE:
        raise RuntimeError("MCP python SDK not installed. pip install mcp[stdio]")
    if not cfg.cmd:
        raise RuntimeError(f"Server {cfg.alias}: stdio mode requires 'cmd'.")

    def _make_client_factory():
        try:
            # Newer SDKs often support env= on stdio_client
            return stdio_client(cfg.cmd, env=cfg.env or {})
        except TypeError as e:
            # Older SDKs: no env= kwarg; temporarily inject env for the spawn
            if "unexpected keyword argument 'env'" not in str(e):
                raise
            orig_env = os.environ.copy()
            try:
                if cfg.env:
                    os.environ.update(cfg.env)
                return stdio_client(cfg.cmd)
            finally:
                os.environ.clear()
                os.environ.update(orig_env)

    res = _make_client_factory()

    # If the SDK returns an async context manager, enter it; else await it.
    if hasattr(res, "__aenter__"):
        client = await res.__aenter__()  # type: ignore[attr-defined]
        # Tag the client with its context so we can close it on shutdown
        try:
            setattr(client, "__mcp_ctx__", res)
        except Exception:
            pass
    else:
        client = await res  # older awaitable form

    await client.initialize()
    return client

async def mcp_list_tools(session) -> List[Dict[str, Any]]:
    result = await session.list_tools()
    tools = []
    for t in result.tools:
        tools.append({
            "name": t.name,
            "description": getattr(t, "description", "") or "",
            "input_schema": t.inputSchema.model_dump() if getattr(t, "inputSchema", None) else {}
        })
    return tools


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
            # Accept both 200 and 207 (multi-status) if some servers use it
            if r.status_code in (200, 207):
                data = r.json()
                return data
        except Exception as e:
            last_exc = e
    if last_exc:
        raise last_exc
    raise RuntimeError("No RPC method variant succeeded")

async def mcp_http_list_tools() -> List[Dict[str, Any]]:
    """
    Minimal list_tools over HTTP RPC to MCP_RPC_URL.
    We try a few common method names used by streamable-http servers.
    """
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
    # Normalize plausible shapes
    tools = []
    # common shapes:
    # {"tools":[{"name":"...","description":"...","inputSchema":{...}}, ...]}
    # or {"result":{"tools":[...]}}
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
    """
    Minimal call_tool over HTTP RPC to MCP_RPC_URL.
    Tries a few method names, normalizes common content shapes.
    """
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

    # Normalize to the same shape stdio path returns
    # Expect either {"content":[...]} or {"result":{"content":[...]}} etc.
    container = data.get("result", data)
    content = container.get("content") or container.get("contents") or []
    normalized = {"type": "mcp_result", "content": []}
    for item in content:
        if isinstance(item, dict):
            if item.get("type") == "text" and "text" in item:
                normalized["content"].append({"type": "text", "text": item["text"]})
            else:
                # pass through other content types
                normalized["content"].append(item)
        else:
            # fallback string payload
            normalized["content"].append({"type": "text", "text": str(item)})
    return normalized


# =============================================================================
# FastAPI App + Discovery
# =============================================================================

app = FastAPI(
    title="MCP OpenAPI Bridge (Generic, Self-Discovering)",
    version="3.1.8",
    description=textwrap.dedent(
        """\
        A generic, self-discovering OpenAPI façade for MCP servers.
        It discovers tools, prompts, and resources from MCP and exposes:
        • One generic GET/POST endpoint per tool
        • Auto-generated granular endpoints for each `action`/`kind` combo
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

# --- Path normalizer (unchanged)
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
            # Don’t let shutdown be noisy
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
        try:
            tools_raw: List[Dict[str, Any]] = []
            if st.cfg.mode == "stdio":
                if not st.connected and st.cfg.cmd:
                    st.client = await mcp_connect_stdio(st.cfg)
                    st.connected = True
                tools_raw = await mcp_list_tools(st.client) if st.connected else []
            else:
                tools_raw = []

            # NEW: If stdio didn't yield tools and MCP_RPC_URL is set, try HTTP RPC discovery
            if not tools_raw and MCP_RPC_URL:
                try:
                    tools_raw = await mcp_http_list_tools()
                    if tools_raw:
                        # Mark as "virtually connected" so routes don’t 503 during help/schema
                        st.connected = st.connected or True
                except Exception as e:
                    print(f"[discover:http-rpc] list_tools failed via {MCP_RPC_URL}: {e}", file=sys.stderr)

            st.tools.clear()
            for tr in tools_raw:
                name = tr["name"]
                desc = tr.get("description") or ""
                schema = tr.get("input_schema") or {}
                td = ToolDescriptor(
                    name=name,
                    description=desc,
                    input_schema=schema,
                    convenience_params=[p for p in CONVENIENCE_PARAMS if p in (schema.get("properties") or {})]
                )
                td.inferred_actions = schema_enums(schema, "operation") or parse_actions_from_description(desc)
                kinds_from_schema = schema_enums(schema, "resource")
                if kinds_from_schema:
                    td.inferred_kinds = kinds_from_schema
                elif is_k8s_tool(name, desc):
                    td.inferred_kinds = infer_kinds_from_description(desc)
                if is_k8s_tool(name, desc):
                    td.output_guidance = build_k8s_output_guidance(name)
                    td.natural_examples = k8s_natural_examples(name)
                td.usage = {
                    "schema": {
                        "type": "object",
                        "fields": {
                            "args": {"required": True, "type": "string", "description": "Operation-specific arguments / flags"},
                            "operation": {"required": True, "type": "string"},
                            "resource": {"required": True, "type": "string"},
                        },
                        "requiredFields": ["args", "operation", "resource"]
                    },
                    "argstringRequired": True,
                }
                st.tools[name] = td

        except Exception as e:
            st.connected = False
            st.client = None
            st.tools.clear()
            print(f"[discover] Server '{alias}' discovery failed: {e}", file=sys.stderr)

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
# Health / Info (unchanged)
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
# Discovery control (unchanged)
# =============================================================================

@app.post("/discover", tags=["discovery"], summary="Discover Endpoint")
async def discover_endpoint(wait: Optional[int] = Query(0, description="Seconds to wait (max 10) for discovery")):
    wait = max(0, min(int(wait or 0), 10))
    return await do_discover(wait_seconds=wait)

@app.get("/discovery/status", tags=["discovery"], summary="Discovery Status")
async def discovery_status():
    return {"servers": DISCOVERY.list_servers()}


# =============================================================================
# OpenAPI enrichment (unchanged helper)
# =============================================================================

def openapi_extra_blocks() -> Dict[str, Any]:
    x_model_instructions = {
        "callDiscipline": [
            "Use **GET with query** or **POST with JSON**. If no args, POST `{}`.",
            "Prefer granular endpoints `/{SERVER}/tool/{TOOL}/{action}[/{kind}]` when actions/kinds are available.",
            "If the tool requires `args` (string), you can either provide it directly OR pass convenience params; the bridge will compose the argstring.",
            "Use `format=json|yaml|text` to influence output; json/yaml injects '-o' when supported.",
            "Use `dryrun=true` to preview the composed MCP call."
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
            "2) Choose a tool whose schema/description matches the user request.",
            "3) Use a granular path when available (e.g., `/get/pods`).",
            "4) Prefer `format=json` for machine-readable results when available."
        ],
        "errorFix": [
            "If you see 'expected a request body', use GET or POST `{}`.",
            "If a tool needs `args` (string): either send it, or pass convenience params like `namespace`, `name`, etc.",
            "If you see a schema error, inspect `/schema`, `/example`, or try a granular endpoint.",
            "If parsing text, leverage `x-outputGuidance.parsingHints`."
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
                "requiredFields": td.usage.get("schema", {}).get("requiredFields", []),
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
# HTTP Fallback helper (unchanged)
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
# Generic + Granular Tool Dispatch
# =============================================================================

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
    tool_path: str,
    body: Optional[Dict[str, Any]],
    query_args_fallback: Optional[str],
    dryrun: bool,
    format_hint: Optional[str],
) -> Any:
    # Try stdio; if not available and MCP_RPC_URL is set, we’ll use HTTP RPC.
    st: Optional[ServerState] = None
    try:
        st = await ensure_connected(server)
    except HTTPException as e:
        if e.status_code != 503 or not MCP_RPC_URL:
            raise
        # stdio not connected; we'll execute via HTTP RPC below
        print(f"[rpc-fallback] Using MCP_RPC_URL={MCP_RPC_URL} for '{tool_path}'", file=sys.stderr)

    components = tool_path.split("/")
    tool = components[0]
    suffix = "/".join(components[1:]) if len(components) > 1 else ""

    td = (st.tools.get(tool) if st else None)
    if st and not td:
        print(f"[dispatch] Tool '{tool}' not in cache for server '{st.cfg.alias}'. Refreshing discovery...", file=sys.stderr)
        await do_discover(wait_seconds=0)
        td = DISCOVERY.servers.get(st.cfg.alias, st).tools.get(tool)

    # helper endpoints (schema/example/help) remain stdio/discovery-based
    if suffix in ("schema", "example", "help"):
        if td:
            if suffix == "schema":
                return td.input_schema or {}
            if suffix == "example":
                return {"naturalExamples": td.natural_examples}
            if suffix == "help":
                return {
                    "name": tool,
                    "description": td.description,
                    "inferred_actions": td.inferred_actions,
                    "inferred_kinds": td.inferred_kinds,
                    "convenience_params": td.convenience_params,
                    "outputGuidance": td.output_guidance,
                    "usage": td.usage
                }
        if suffix == "schema":
            return {}
        if suffix == "example":
            return {"naturalExamples": []}
        if suffix == "help":
            return {
                "name": tool, "description": "", "inferred_actions": [], "inferred_kinds": [],
                "convenience_params": CONVENIENCE_PARAMS, "outputGuidance": {}, "usage": {}
            }

    if suffix == "try":
        if dryrun:
            return {"dryrun": True, "tool": tool, "args": {}}
        if st:
            return await mcp_call_tool(st.client, tool, {})
        # HTTP RPC fallback
        return await mcp_http_call_tool(tool, {})

    args = {}
    convenience = {}
    if suffix and suffix not in ("invoke",):
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

    if td and "operation" not in final_args and td.inferred_actions:
        final_args["operation"] = td.inferred_actions[0]
    if td and "resource" not in final_args and td.inferred_kinds:
        final_args["resource"] = td.inferred_kinds[0]

    if dryrun:
        print(f"[dryrun] server={server} tool={tool} suffix='{suffix}' final_args={final_args}", file=sys.stderr)
        return {"dryrun": True, "server": server, "tool": tool, "final_args": final_args}

    # Execute via stdio if available, else via HTTP RPC
    try:
        if st:
            print(f"[invoke-stdio] server={st.cfg.alias} tool={tool} suffix='{suffix}' final_args={final_args}", file=sys.stderr)
            return await mcp_call_tool(st.client, tool, final_args)
        else:
            print(f"[invoke-http-rpc] url={MCP_RPC_URL} tool={tool} suffix='{suffix}' final_args={final_args}", file=sys.stderr)
            return await mcp_http_call_tool(tool, final_args)
    except Exception as e:
        raise HTTPException(502, f"Tool invocation failed for '{tool_path}': {e}")


# =============================================================================
# Generic dispatchers (with existing REST fallback on 503)
# =============================================================================

@app.post(
    "/{server}/tool/{tool_path:path}",
    tags=["tools"],
    summary="Tool Dispatch",
    responses={200: {"description": "Successful Response", "content": {"application/json": {}}}},
)
async def tool_dispatch_post(
    server: str = Path(..., description="Server alias (e.g., 'mcp')"),
    tool_path: str = Path(..., description="Tool or tool path like 'kubectl_resources/get/pods'"),
    args: Optional[str] = Query(None, description="JSON-encoded args fallback"),
    dryrun: Optional[bool] = Query(False, description="If true, returns the would-be MCP call without executing it"),
    format: Optional[str] = Query(None, description="Output preference: json|yaml|text (adds '-o json|yaml' when supported)"),
    body: Optional[Dict[str, Any]] = Body(None),
):
    qargs = None
    if args:
        try:
            qargs = json.loads(args)
        except Exception:
            qargs = args
    payload_body = body or (qargs if isinstance(qargs, dict) else None)

    try:
        result = await do_tool_call(server, tool_path, payload_body, None, bool(dryrun), format)
        return JSONResponse(result)
    except HTTPException as e:
        if e.status_code == 503 and MCP_FORWARD_URL:
            params: Dict[str, Any] = {}
            if format is not None:
                params["format"] = format
            if dryrun:
                params["dryrun"] = dryrun
            if isinstance(qargs, str):
                params["args"] = qargs
            return await forward_via_http(server, tool_path, "POST", params, payload_body)
        raise


@app.get(
    "/{server}/tool/{tool_path:path}",
    tags=["tools"],
    summary="Tool Dispatch",
    responses={200: {"description": "Successful Response", "content": {"application/json": {}}}},
)
async def tool_dispatch_get(
    server: str = Path(..., description="Server alias (e.g., 'mcp')"),
    tool_path: str = Path(..., description="Tool or tool path like 'kubectl_resources/get/pods'"),
    args: Optional[str] = Query(None, description="JSON-encoded args fallback"),
    dryrun: Optional[bool] = Query(False, description="If true, returns the would-be MCP call without executing it"),
    format: Optional[str] = Query(None, description="Output preference: json|yaml|text (adds '-o json|yaml' when supported)"),
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

    try:
        result = await do_tool_call(server, tool_path, body, qargs_fallback, bool(dryrun), format)
        return JSONResponse(result)
    except HTTPException as e:
        if e.status_code == 503 and MCP_FORWARD_URL:
            params: Dict[str, Any] = {}
            if format is not None:
                params["format"] = format
            if dryrun:
                params["dryrun"] = dryrun
            if args is not None:
                params["args"] = args
            for k in ("namespace","name","labels","labelSelector","fieldSelector","container","sinceSeconds"):
                if k in body:
                    params[k] = body[k]
            return await forward_via_http(server, tool_path, "GET", params, None)
        raise


# -------------------- explicit granular routes (unchanged except fallback already handled) --------------------
# (Your granular_post_kind / granular_get_kind / granular_post_action / granular_get_action remain unchanged)
# ... keep your existing granular route implementations here unchanged ...


# =============================================================================
# Per-tool helper endpoints (unchanged)
# =============================================================================

@app.get("/mcp/tool/{tool}/invoke", tags=["tools", "invoke"], summary="Invoke (GET /invoke)")
async def tool_invoke_get(tool: str, **kwargs):
    return await tool_dispatch_get(server="mcp", tool_path=f"{tool}/invoke", **kwargs)

@app.get("/mcp/tool/{tool}/schema", tags=["tools", "schema"], summary="Tool schema")
async def tool_schema(tool: str):
    st = DISCOVERY.servers.get("mcp")
    if not st or tool not in st.tools:
        return {}
    return st.tools[tool].input_schema or {}

@app.get("/mcp/tool/{tool}/example", tags=["tools", "example"], summary="Tool example")
async def tool_example(tool: str):
    st = DISCOVERY.servers.get("mcp")
    if not st or tool not in st.tools:
        return {"naturalExamples": []}
    return {"naturalExamples": st.tools[tool].natural_examples}

@app.get("/mcp/tool/{tool}/help", tags=["tools", "help"], summary="Tool help")
async def tool_help(tool: str):
    st = DISCOVERY.servers.get("mcp")
    if not st or tool not in st.tools:
        return {
            "name": tool,
            "description": "",
            "inferred_actions": [],
            "inferred_kinds": [],
            "convenience_params": CONVENIENCE_PARAMS,
            "outputGuidance": {},
            "usage": {}
        }
    td = st.tools[tool]
    return {
        "name": tool,
        "description": td.description,
        "inferred_actions": td.inferred_actions,
        "inferred_kinds": td.inferred_kinds,
        "convenience_params": td.convenience_params,
        "outputGuidance": td.output_guidance,
        "usage": td.usage
    }

@app.get("/mcp/tool/{tool}/try", tags=["tools", "try"], summary="Tool zero-arg try",
         description="Calls this tool with `{}` (no arguments).")
async def tool_try(tool: str, dryrun: Optional[bool] = Query(False)):
    return await tool_dispatch_get(server="mcp", tool_path=f"{tool}/try", dryrun=dryrun)


# =============================================================================
# OpenAPI Post-processor (unchanged)
# =============================================================================

_original_openapi = app.openapi
def custom_openapi():
    openapi_schema = _original_openapi()
    openapi_schema.update({ **openapi_extra_blocks() })
    app.openapi_schema = openapi_schema
    return app.openapi_schema
app.openapi = custom_openapi
