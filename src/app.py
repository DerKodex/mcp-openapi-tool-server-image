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
"""

import asyncio
import json
import os
import re
import sys
import textwrap
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Body, Query, Path, HTTPException
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
# MCP integration
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
# FastAPI App + Discovery
# =============================================================================

app = FastAPI(
    title="MCP OpenAPI Bridge (Generic, Self-Discovering)",
    version="3.1.2",
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
            else:
                tools_raw = []

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
# OpenAPI enrichment (x-* blocks)
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
    if not st:
        raise HTTPException(404, f"Unknown server '{server}'")
    if not st.connected or not st.client:
        raise HTTPException(503, f"Server '{server}' is not connected")
    return st


async def do_tool_call(
    server: str,
    tool_path: str,
    body: Optional[Dict[str, Any]],
    query_args_fallback: Optional[str],
    dryrun: bool,
    format_hint: Optional[str],
) -> Any:
    st = await ensure_connected(server)

    components = tool_path.split("/")
    tool = components[0]
    suffix = "/".join(components[1:]) if len(components) > 1 else ""

    # Try to locate descriptor; if missing, refresh once, then fall back to "best-effort"
    td = st.tools.get(tool)
    if not td:
        print(f"[dispatch] Tool '{tool}' not in cache for server '{server}'. Refreshing discovery...", file=sys.stderr)
        await do_discover(wait_seconds=0)
        td = DISCOVERY.servers.get(server, ServerState(ServerConfig(alias=server))).tools.get(tool)

    # Helper endpoints even without td: schema/example/help/try
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
        # No descriptor: provide minimal help
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
        return await mcp_call_tool(st.client, tool, {})

    # Build args from URL suffix when present (action/kind)
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

    # If still missing 'operation'/'resource' and we have a descriptor, inject defaults when sensible
    if td and "operation" not in final_args and td.inferred_actions:
        final_args["operation"] = td.inferred_actions[0]
    if td and "resource" not in final_args and td.inferred_kinds:
        final_args["resource"] = td.inferred_kinds[0]

    if dryrun:
        print(f"[dryrun] server={server} tool={tool} suffix='{suffix}' final_args={final_args}", file=sys.stderr)
        return {
            "dryrun": True,
            "server": server,
            "tool": tool,
            "final_args": final_args
        }

    # BEST-EFFORT CALL even if td is missing — do not 404
    try:
        print(f"[invoke] server={server} tool={tool} suffix='{suffix}' final_args={final_args}", file=sys.stderr)
        return await mcp_call_tool(st.client, tool, final_args)
    except Exception as e:
        # Give a helpful error with context
        raise HTTPException(502, f"Tool invocation failed for '{tool_path}': {e}")


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

    result = await do_tool_call(server, tool_path, body, qargs_fallback, bool(dryrun), format)
    return JSONResponse(result)


# -------------------- NEW: explicit granular routes to avoid 404s --------------------

@app.post(
    "/{server}/tool/{tool}/{action}/{kind}",
    tags=["tools"],
    summary="Granular Tool Dispatch (POST)"
)
async def granular_post_kind(
    server: str,
    tool: str,
    action: str,
    kind: str,
    args: Optional[str] = Query(None, description="JSON-encoded args fallback"),
    dryrun: Optional[bool] = Query(False),
    format: Optional[str] = Query(None, description="Output preference: json|yaml|text"),
    body: Optional[Dict[str, Any]] = Body(None),
    namespace: Optional[str] = Query(None),
    name: Optional[str] = Query(None),
    labels: Optional[str] = Query(None),
    labelSelector: Optional[str] = Query(None),
    fieldSelector: Optional[str] = Query(None),
    container: Optional[str] = Query(None),
    sinceSeconds: Optional[int] = Query(None),
):
    # Merge convenience query params into body just like GET handler
    body = (body or {}).copy()
    for k, v in {
        "namespace": namespace, "name": name, "labels": labels,
        "labelSelector": labelSelector, "fieldSelector": fieldSelector,
        "container": container, "sinceSeconds": sinceSeconds,
    }.items():
        if v is not None:
            body[k] = v

    # Prefer body; if args is JSON dict, merge it too (parity with catch-all)
    if args:
        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict):
                body.update(parsed)
        except Exception:
            pass

    result = await do_tool_call(
        server,
        f"{tool}/{action}/{kind}",
        body,
        None,
        bool(dryrun),
        format,
    )
    return JSONResponse(result)


@app.get(
    "/{server}/tool/{tool}/{action}/{kind}",
    tags=["tools"],
    summary="Granular Tool Dispatch (GET)"
)
async def granular_get_kind(
    server: str,
    tool: str,
    action: str,
    kind: str,
    args: Optional[str] = Query(None, description="JSON-encoded args fallback"),
    dryrun: Optional[bool] = Query(False),
    format: Optional[str] = Query(None, description="Output preference: json|yaml|text"),
    namespace: Optional[str] = Query(None),
    name: Optional[str] = Query(None),
    labels: Optional[str] = Query(None),
    labelSelector: Optional[str] = Query(None),
    fieldSelector: Optional[str] = Query(None),
    container: Optional[str] = Query(None),
    sinceSeconds: Optional[int] = Query(None),
):
    # Reuse the GET catch-all behavior by composing tool_path
    # and passing through query convenience params (via body).
    body: Dict[str, Any] = {}
    for k, v in {
        "namespace": namespace, "name": name, "labels": labels,
        "labelSelector": labelSelector, "fieldSelector": fieldSelector,
        "container": container, "sinceSeconds": sinceSeconds,
    }.items():
        if v is not None:
            body[k] = v

    qargs_fallback: Optional[str] = None
    if args:
        try:
            data = json.loads(args)
            if isinstance(data, dict):
                body.update(data)
            else:
                qargs_fallback = args
        except Exception:
            qargs_fallback = args

    result = await do_tool_call(
        server,
        f"{tool}/{action}/{kind}",
        body,
        qargs_fallback,
        bool(dryrun),
        format,
    )
    return JSONResponse(result)


@app.post(
    "/{server}/tool/{tool}/{action}",
    tags=["tools"],
    summary="Granular Tool Dispatch (POST, action only)"
)
async def granular_post_action(
    server: str,
    tool: str,
    action: str,
    args: Optional[str] = Query(None, description="JSON-encoded args fallback"),
    dryrun: Optional[bool] = Query(False),
    format: Optional[str] = Query(None, description="Output preference: json|yaml|text"),
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

    result = await do_tool_call(
        server,
        f"{tool}/{action}",
        body,
        None,
        bool(dryrun),
        format,
    )
    return JSONResponse(result)


@app.get(
    "/{server}/tool/{tool}/{action}",
    tags=["tools"],
    summary="Granular Tool Dispatch (GET, action only)"
)
async def granular_get_action(
    server: str,
    tool: str,
    action: str,
    args: Optional[str] = Query(None, description="JSON-encoded args fallback"),
    dryrun: Optional[bool] = Query(False),
    format: Optional[str] = Query(None, description="Output preference: json|yaml|text"),
    namespace: Optional[str] = Query(None),
    name: Optional[str] = Query(None),
    labels: Optional[str] = Query(None),
    labelSelector: Optional[str] = Query(None),
    fieldSelector: Optional[str] = Query(None),
    container: Optional[str] = Query(None),
    sinceSeconds: Optional[int] = Query(None),
):
    body: Dict[str, Any] = {}
    for k, v in {
        "namespace": namespace, "name": name, "labels": labels,
        "labelSelector": labelSelector, "fieldSelector": fieldSelector,
        "container": container, "sinceSeconds": sinceSeconds,
    }.items():
        if v is not None:
            body[k] = v

    qargs_fallback: Optional[str] = None
    if args:
        try:
            data = json.loads(args)
            if isinstance(data, dict):
                body.update(data)
            else:
                qargs_fallback = args
        except Exception:
            qargs_fallback = args

    result = await do_tool_call(
        server,
        f"{tool}/{action}",
        body,
        qargs_fallback,
        bool(dryrun),
        format,
    )
    return JSONResponse(result)

# ------------------ end of NEW granular routes ------------------


# =============================================================================
# Per-tool helper endpoints (schema/example/help/try) for convenience
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
# OpenAPI Post-processor: insert x-* fields
# =============================================================================

_original_openapi = app.openapi

def custom_openapi():
    openapi_schema = _original_openapi()
    openapi_schema.update({
        **openapi_extra_blocks()
    })
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
