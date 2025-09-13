# app.py
import os, time, asyncio, re, json
from typing import Any, Dict, Optional, List, Tuple, Union
from fastapi import (
    FastAPI, Body, Response, HTTPException, Depends, Header,
    Request, Query, Path
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
import httpx

# =========================
# Config
# =========================
API_KEY           = os.environ.get("API_KEY", "").strip()
ALLOW_ORIGINS     = os.environ.get("CORS_ALLOW_ORIGINS", "*")
REQUEST_TIMEOUT   = int(os.environ.get("REQUEST_TIMEOUT", "30"))
REFRESH_INTERVAL  = int(os.environ.get("REFRESH_INTERVAL", "10"))  # seconds
DISCOVERY_PUBLIC  = os.environ.get("DISCOVERY_PUBLIC", "true").lower() in ("1","true","t","yes","y","on")
PUBLIC_BASE_URL   = os.environ.get("PUBLIC_BASE_URL", "http://localhost:8080")

# Multiple servers: "ro=http://host:8080/mcp,admin=http://host:8080/mcp"
MCP_SERVERS  = os.environ.get("MCP_SERVERS", "").strip()
MCP_RPC_URL  = os.environ.get("MCP_RPC_URL", "").strip()  # single-server fallback

def parse_servers(spec: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out

SERVERS_CFG = parse_servers(MCP_SERVERS) or ({"mcp": MCP_RPC_URL} if MCP_RPC_URL else {"mcp": "http://localhost:8080/mcp"})

# =========================
# App
# =========================
app = FastAPI(
    title="MCP OpenAPI Bridge (Generic, Self-Discovering)",
    version="2.2.0",
    description=(
        "A generic, self-discovering OpenAPI façade for MCP servers.\n"
        "It discovers tools, prompts, and resources from MCP and exposes:\n"
        "• One generic GET/POST endpoint per tool\n"
        "• Auto-generated granular endpoints for each `action`/`kind` enum combination\n"
        "All endpoints accept GET (query) and POST (JSON); request bodies are optional.\n"
        "Use `{}` for empty POST bodies. If you cannot send a body, use GET with `?args={...}` or `?key=value`."
    ),
)

if ALLOW_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if ALLOW_ORIGINS == "*" else [o.strip() for o in ALLOW_ORIGINS.split(",")],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    if not API_KEY:
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")

PUBLIC_OR_AUTH = [] if DISCOVERY_PUBLIC else [Depends(require_api_key)]

# =========================
# HTTP client
# =========================
_client: Optional[httpx.AsyncClient] = None
async def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
    return _client

# =========================
# MCP state / RPC
# =========================
class ServerState:
    def __init__(self, name: str, rpc_url: str):
        self.name = name
        self.rpc_url = rpc_url
        self.session_id: Optional[str] = None
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.prompts: List[Dict[str, Any]] = []
        self.resources: List[Dict[str, Any]] = []
        self.ready: bool = False
        self.last_error: Optional[str] = None

    async def rpc(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        client = await get_client()
        headers = {}
        if self.session_id:
            headers["mcp-session-id"] = self.session_id
        payload = {"jsonrpc": "2.0", "id": str(int(time.time() * 1000)), "method": method}
        if params is not None:
            payload["params"] = params
        try:
            resp = await client.post(self.rpc_url, json=payload, headers=headers)
            resp.raise_for_status()
        except Exception as e:
            self.ready = False
            self.last_error = f"{type(e).__name__}: {e}"
            raise HTTPException(
                status_code=503,
                detail={
                    "message": f"Failed to connect to MCP server at {self.rpc_url}",
                    "error": str(e),
                    "resolution": [
                        "Verify the MCP service DNS/port, Endpoint, and NetworkPolicy.",
                        f"Check that {self.rpc_url} is correct and reachable from this pod.",
                    ],
                },
            )
        sid = resp.headers.get("mcp-session-id")
        if sid:
            self.session_id = sid
        data = resp.json()
        if "error" in data:
            self.ready = False
            self.last_error = str(data["error"])
            raise HTTPException(
                status_code=502,
                detail={
                    "message": "MCP server returned an error",
                    "mcp_error": data["error"],
                    "resolution": [
                        "Confirm tool name and argument keys match the tool schema.",
                        "Call `/{SERVER}/tools/list` and `/{SERVER}/tool/{TOOL}/schema` to verify fields.",
                        "If RBAC/namespace related, try the admin server or pass `namespace`.",
                    ],
                },
            )
        return data["result"]

    async def ensure_initialized(self):
        if self.session_id:
            return
        await self.rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}})

    async def refresh_all(self):
        await self.ensure_initialized()
        # tools
        tools_result = await self.rpc("tools/list")
        tool_list = tools_result.get("tools", tools_result if isinstance(tools_result, list) else [])
        self.tools = {t["name"]: t for t in tool_list if isinstance(t, dict) and "name" in t}

        # prompts (best-effort)
        try:
            prompts_result = await self.rpc("prompts/list")
            self.prompts = prompts_result.get("prompts", prompts_result if isinstance(prompts_result, list) else [])
        except Exception:
            self.prompts = []

        # resources (best-effort)
        try:
            res_result = await self.rpc("resources/list")
            self.resources = res_result.get("resources", res_result if isinstance(res_result, list) else [])
        except Exception:
            self.resources = []

        self.ready = True
        self.last_error = None

SERVERS: Dict[str, ServerState] = {name: ServerState(name, url) for name, url in SERVERS_CFG.items()}

# =========================
# Helpers
# =========================
def _safe(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)

def example_from_schema(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return {}
    if "example" in schema:
        return schema["example"]
    t = schema.get("type")
    if isinstance(t, list) and t:
        t = t[0]
    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]
    if t == "object" or ("properties" in schema):
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        obj = {}
        for k, v in props.items():
            if "default" in v: obj[k] = v["default"]; continue
            ex = example_from_schema(v)
            if ex == {}:
                vt = v.get("type")
                if vt == "string":
                    ex = (v.get("enum") or ["value"])[0]
                elif vt in ("integer", "number"):
                    ex = 1
                elif vt == "boolean":
                    ex = True
                elif vt == "array":
                    ex = []
                elif vt == "object":
                    ex = {}
                else:
                    ex = "value"
            obj[k] = ex
        if required:
            obj = {k: v for k, v in obj.items() if k in required}
        return obj
    if t == "array":
        return [example_from_schema(schema.get("items", {}))]
    if t == "string":  return schema.get("default") or "value"
    if t in ("integer", "number"): return schema.get("default") or 1
    if t == "boolean": return schema.get("default") or True
    return schema.get("default") or {}

def describe_schema(schema: Any) -> Dict[str, Any]:
    if not isinstance(schema, dict):
        return {"type": "object", "fields": {}, "requiredFields": []}
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields = {}
    for k, v in props.items():
        fields[k] = {
            "required": k in required,
            "type": v.get("type", "any"),
            "enum": v.get("enum", None),
            "default": v.get("default", None),
            "description": v.get("description", None),
            "examples": v.get("examples", None),
        }
    return {"type": schema.get("type", "object"), "fields": fields, "requiredFields": sorted(list(required))}

def parse_args_query(args_q: Optional[str]) -> Dict[str, Any]:
    if not args_q: return {}
    try:
        val = json.loads(args_q)
        return val if isinstance(val, dict) else {}
    except Exception:
        return {}

def coerce_query_params(schema: Dict[str, Any], qp: Dict[str, str]) -> Dict[str, Any]:
    props = (schema or {}).get("properties", {}) if isinstance(schema, dict) else {}
    out: Dict[str, Any] = {}
    for k, v in qp.items():
        if k in props:
            typ = props[k].get("type")
            if typ in (None, "string"): out[k] = v
            elif typ in ("integer", "number"):
                try: out[k] = int(v) if typ == "integer" else float(v)
                except: out[k] = v
            elif typ == "boolean":
                out[k] = v.lower() in ("1","true","t","yes","y","on")
            elif typ == "array":
                out[k] = [s for s in v.split(",") if s != ""]
            else:
                out[k] = v
    return out

def normalize_body(
    body: Optional[Union[Dict[str, Any], list, str, int, float, bool, None]],
    schema: Optional[Dict[str, Any]],
    args_q: Optional[str],
    request: Optional[Request]
) -> Dict[str, Any]:
    base: Dict[str, Any] = {}
    if isinstance(body, dict):
        base = body.get("args", body) if isinstance(body.get("args"), dict) else body or {}
    from_args = parse_args_query(args_q)
    base.update(from_args)
    if request is not None:
        qp = {k: v for k, v in request.query_params.items() if k not in ("args",)}
        base.update(coerce_query_params(schema or {"type":"object"}, qp))
    return base or {}

def iter_action_kind(schema: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    if not isinstance(schema, Dict):
        return ([], [])
    props = schema.get("properties", {})
    action_key = next((k for k in ["action","verb","operation","op"] if k in props), None)
    actions = props.get(action_key, {}).get("enum", []) if action_key else []
    kind_key = next((k for k in ["kind","resource","resources","type"] if k in props), None)
    kinds = props.get(kind_key, {}).get("enum", []) if kind_key else []
    return (actions, kinds)

# ---- Argstring composer (for kubectl-like tools) ----
_CONVENIENCE_KEYS = ["namespace","name","labels","labelSelector","fieldSelector","container","sinceSeconds"]
def _looks_kubectl_like(tool_name: str, description: str, schema: Dict[str, Any]) -> bool:
    text = (tool_name + " " + (description or "")).lower()
    has_ops = any(k in (schema.get("properties") or {}) for k in ("operation","action"))
    has_resource = any(k in (schema.get("properties") or {}) for k in ("resource","kind"))
    has_args = (schema.get("properties") or {}).get("args",{}).get("type") == "string"
    return has_ops and has_resource and has_args and ("kubectl" in text or "kubernetes" in text)

def _append_flag(parts: List[str], flag: str, value: Optional[str]):
    if value is None or value == "": return
    if " " in value:
        parts.append(f"{flag} '{value}'")
    else:
        parts.append(f"{flag} {value}")

def enrich_argstring(call_args: Dict[str, Any], schema: Dict[str, Any], tool_name: str, description: str):
    """If the tool requires args:string and caller passed convenience params,
       auto-compose args (e.g., name + -n namespace + selectors)."""
    props = (schema or {}).get("properties", {})
    if props.get("args", {}).get("type") != "string":
        return
    if not _looks_kubectl_like(tool_name, description, schema):
        return
    # Already provided args? keep and just append missing pieces
    arg_parts: List[str] = []
    raw = str(call_args.get("args") or "").strip()
    if raw:
        arg_parts.append(raw)

    name = call_args.pop("name", None)
    namespace = call_args.pop("namespace", None)
    labels = call_args.pop("labels", None) or call_args.pop("labelSelector", None)
    field_selector = call_args.pop("fieldSelector", None)
    container = call_args.pop("container", None)
    since_seconds = call_args.pop("sinceSeconds", None)

    # name first (positional)
    if name: arg_parts.insert(0, name)
    # namespace
    _append_flag(arg_parts, "-n", namespace)
    # selectors
    if labels: _append_flag(arg_parts, "-l", labels)
    if field_selector: _append_flag(arg_parts, "--field-selector", field_selector)
    # logs-specific convenience
    if container: _append_flag(arg_parts, "-c", container)
    if since_seconds is not None:
        try:
            ss = int(since_seconds)
            _append_flag(arg_parts, "--since", f"{ss}s")
        except Exception:
            pass

    call_args["args"] = " ".join(arg_parts).strip()

# =========================
# Discovery loop
# =========================
_discovery_lock = asyncio.Lock()
_last_discovery: float = 0.0

async def discover_once():
    global _last_discovery
    async with _discovery_lock:
        for name, st in SERVERS.items():
            try:
                await st.refresh_all()
            except Exception:
                pass
        _last_discovery = time.time()

async def background_poller():
    while True:
        try:
            await discover_once()
        except Exception:
            pass
        await asyncio.sleep(REFRESH_INTERVAL)

@app.on_event("startup")
async def startup():
    asyncio.create_task(background_poller())

# =========================
# Health / discovery / tool list
# =========================
@app.get("/livez", tags=["health"])
async def livez(): return {"ok": True}

@app.get("/readyz", tags=["health"])
async def readyz():
    any_ready = any(st.ready for st in SERVERS.values())
    return JSONResponse(
        status_code=200 if any_ready else 503,
        content={"ok": any_ready,
                 "servers": {k: {"ready": v.ready, "tools": list(v.tools.keys()), "last_error": v.last_error} for k,v in SERVERS.items()}},
    )

@app.get("/healthz", tags=["health"])
async def healthz():
    return {"ok": any(st.ready for st in SERVERS.values()),
            "servers": {k: {"ready": v.ready, "rpc_url": v.rpc_url, "tools": list(v.tools.keys()), "last_error": v.last_error}
                        for k,v in SERVERS.items()}}

@app.get("/servers", tags=["info"])
async def servers_info(): return {"servers": {k: v.rpc_url for k, v in SERVERS.items()}}

@app.get("/{server}/tools/list", tags=["discovery"], dependencies=PUBLIC_OR_AUTH)
async def tools_list(server: str):
    st = SERVERS.get(server)
    if not st:
        raise HTTPException(status_code=404, detail={"message": f"Unknown server '{server}'"})
    # ensure recent snapshot
    return {"tools": list(st.tools.values())}

@app.post("/discover", tags=["discovery"], dependencies=PUBLIC_OR_AUTH)
async def discover_endpoint(wait: Optional[int] = Query(default=0, description="Seconds to wait (max 10) for discovery")):
    wait = max(0, min(int(wait or 0), 10))
    task = asyncio.create_task(discover_once())
    if wait:
        try:
            await asyncio.wait_for(task, timeout=wait)
        except asyncio.TimeoutError:
            return {"ok": False, "message": "Discovery still running", "waited": wait}
    return {"ok": True, "lastDiscovery": _last_discovery}

@app.get("/discovery/status", tags=["discovery"])
async def discovery_status():
    return {"lastDiscovery": _last_discovery,
            "servers": {k: {"ready": v.ready, "tools": list(v.tools.keys()),
                            "prompts": [p.get('name') for p in v.prompts],
                            "resources": [r.get('uri','') for r in v.resources],
                            "last_error": v.last_error}
                        for k, v in SERVERS.items()}}

# =========================
# Generic dispatcher for ALL tools & granular forms
# =========================
@app.api_route("/{server}/tool/{tool_path:path}", methods=["GET","POST"], tags=["tools"], dependencies=PUBLIC_OR_AUTH)
async def tool_dispatch(
    request: Request,
    server: str = Path(..., description="Server alias (e.g., 'mcp', 'ro', 'admin')"),
    tool_path: str = Path(..., description="Tool name or tool plus suffix, e.g., 'kubectl_resources', 'kubectl_resources/get/pods', 'kubectl_resources/invoke'"),
    args: Optional[str] = Query(default=None, description="JSON-encoded args fallback"),
    body: Optional[Dict[str, Any]] = Body(default=None, embed=False),
):
    st = SERVERS.get(server)
    if not st or tool_path.strip() == "":
        raise HTTPException(status_code=404, detail={"message": "Unknown server or tool."})

    # Split tool path: tool[/action][/kind or special]
    parts = [p for p in tool_path.split("/") if p != ""]
    tool_name = parts[0]
    suffix = parts[1:]  # e.g. ['get','pods'] or ['invoke'] or ['try']

    tool = st.tools.get(tool_name)
    if not tool:
        # try match by safe-name
        rev = { _safe(k): k for k in st.tools.keys() }
        real = rev.get(tool_name)
        if real: tool = st.tools.get(real); tool_name = real

    if not tool:
        raise HTTPException(status_code=404, detail={"message": f"Tool '{tool_name}' not found on server '{server}'."})

    schema = tool.get("inputSchema") or {"type":"object"}
    description = tool.get("description", "")
    call_args = normalize_body(body, schema, args, request)

    # helper suffixes
    if suffix == ["schema"]:
        return schema
    if suffix == ["example"]:
        return {"name": tool_name, "arguments": example_from_schema(schema)}
    if suffix == ["help"]:
        return {
            "tool": tool_name,
            "server": server,
            "description": description or "No description provided by MCP server.",
            "schema": describe_schema(schema),
            "howToUse": [
                f"POST {PUBLIC_BASE_URL}/{server}/tool/{_safe(tool_name)} with JSON; if no args, send {{}}.",
                f"GET  {PUBLIC_BASE_URL}/{server}/tool/{_safe(tool_name)}?args={{...}} (if body unsupported).",
                f"Try zero-arg: GET {PUBLIC_BASE_URL}/{server}/tool/{_safe(tool_name)}/try",
                "Prefer granular forms like '/{server}/tool/{tool}/{action}[/{kind}]' when available.",
            ],
        }
    if suffix == ["try"]:
        await st.ensure_initialized()
        return await st.rpc("tools/call", {"name": tool_name, "arguments": {}})
    if suffix == ["invoke"]:
        await st.ensure_initialized()
        return await st.rpc("tools/call", {"name": tool_name, "arguments": call_args})

    # granular: /{tool}/{action}[/{kind}]
    actions, kinds = iter_action_kind(schema)
    fixed: Dict[str, Any] = {}
    if len(suffix) >= 1 and suffix[0] not in ("schema","example","help","try","invoke"):
        fixed["action"] = suffix[0]
    if len(suffix) >= 2:
        fixed["kind"] = suffix[1]
    call_args.update(fixed)

    # Compose argstring (kubectl-like tools)
    enrich_argstring(call_args, schema, tool_name, description)

    await st.ensure_initialized()
    return await st.rpc("tools/call", {"name": tool_name, "arguments": call_args})

# =========================
# Error handlers with guidance
# =========================
@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    payload = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
    if "resolution" not in payload:
        tips = [
            "Preferred: POST a JSON object body (use `{}` if no arguments).",
            "Fallbacks: GET '/try' or '/invoke' for zero-arg, or pass `?args={...}` / `?key=value`.",
            "Discover tools via `/{SERVER}/tools/list` (GET) and inspect fields with `/tool/{TOOL}/schema`.",
            "Use `/tool/{TOOL}/example` or granular endpoints like `/{SERVER}/tool/{TOOL}/{action}/{kind}`.",
            "If the tool requires `args` (string), pass it OR provide convenience params like `namespace`, `name`, etc.; the bridge will construct the argstring.",
        ]
        if exc.status_code == 401:
            tips.insert(0, "Include the `X-Api-Key` header if the bridge was configured with API_KEY.")
        if exc.status_code == 422:
            tips.insert(0, "Ensure the request is JSON or use GET with `?args={...}` as a fallback.")
        payload["resolution"] = tips
    return JSONResponse(status_code=exc.status_code, content=payload)

@app.exception_handler(Exception)
async def unhandled_exc_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "message": f"Unhandled error: {type(exc).__name__}",
            "error": str(exc),
            "resolution": [
                "Preferred: POST a JSON object body; if you have no arguments, send `{}`.",
                "Fallbacks: GET '/try' or '/invoke', or pass `?args={...}` / individual `?key=value`.",
                "Retry with a simpler body; check bridge pod logs for stack traces.",
                "Verify MCP server availability and NetworkPolicy.",
            ],
        },
    )

# =========================
# OpenAPI builder (non-blocking) – synthesize spec from MCP data
# =========================
_COMMON_CONVENIENCE = [
    {"name": "namespace", "in": "query", "required": False, "schema": {"type":"string"},
     "description": "Convenience param; auto-translated into argstring (e.g., '-n <namespace>')."},
    {"name": "name", "in": "query", "required": False, "schema": {"type":"string"},
     "description": "Convenience param; resource name; placed positionally before flags in argstring."},
    {"name": "labels", "in": "query", "required": False, "schema": {"type":"string"},
     "description": "Convenience param; label selector; becomes '-l <labels>'."},
    {"name": "labelSelector", "in": "query", "required": False, "schema": {"type":"string"},
     "description": "Convenience alias for 'labels'."},
    {"name": "fieldSelector", "in": "query", "required": False, "schema": {"type":"string"},
     "description": "Convenience param; becomes '--field-selector <expr>'."},
    {"name": "container", "in": "query", "required": False, "schema": {"type":"string"},
     "description": "Convenience param for logs; becomes '-c <container>'."},
    {"name": "sinceSeconds", "in": "query", "required": False, "schema": {"type":"integer"},
     "description": "Convenience param for logs; becomes '--since <N>s'."},
    {"name": "args", "in": "query", "required": False, "schema": {"type":"string"},
     "description": "Raw argstring fallback (JSON body still preferred)."},
]

def _op_params_with_schema(schema: Dict[str, Any], argstring_required: bool, include_convenience=True) -> List[Dict[str, Any]]:
    params: List[Dict[str, Any]] = []
    # Query fallbacks for actual schema fields
    props = (schema or {}).get("properties", {}) if isinstance(schema, dict) else {}
    for k, v in props.items():
        params.append({"name": k, "in": "query", "required": False,
                       "description": v.get("description", f"Query param for '{k}'"),
                       "schema": {"type": v.get("type","string")}})
    if include_convenience and argstring_required:
        # Convenience NL params → argstring composer
        params.extend(_COMMON_CONVENIENCE)
    return params

def _make_request_body(schema: Dict[str, Any], required: bool=False) -> Dict[str, Any]:
    direct = schema or {"type": "object"}
    wrapped = {"type": "object", "properties": {"args": direct}, "required": ["args"]}
    return {
        "required": required,
        "content": {
            "application/json": {
                "schema": {"oneOf": [direct, wrapped]},
                "examples": {
                    "empty":  {"summary": "No arguments", "value": {}},
                    "direct": {"summary": "Direct body", "value": example_from_schema(direct) or {}},
                    "wrapped": {"summary": "Wrapped in args", "value": {"args": example_from_schema(direct) or {}}},
                },
            }
        },
    }

def _parse_examples_from_description(desc: str) -> List[str]:
    """Extract example lines from MCP tool descriptions (best-effort)."""
    out: List[str] = []
    for line in (desc or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("- "):
            out.append(s[2:])
        elif "operation=" in s or "args=" in s or "resource=" in s:
            out.append(s)
    return out[:12]

def _nl_examples_from(tool_name: str, schema: Dict[str, Any], description: str) -> List[Dict[str, Any]]:
    actions, kinds = iter_action_kind(schema)
    props = schema.get("properties", {}) if isinstance(schema, dict) else {}
    has_ns = "namespace" in props or _looks_kubectl_like(tool_name, description, schema)
    exs: List[Dict[str, Any]] = []
    if has_ns:
        if (("action" in props or "operation" in props) and ("kind" in props or "resource" in props)):
            exs.append({
                "intent": "List all pods in the ollama namespace",
                "calls": [
                    {"GET": f"/{{SERVER}}/tool/{_safe(tool_name)}/get/pods?namespace=ollama"},
                    {"POST": f"/{{SERVER}}/tool/{_safe(tool_name)}/get/pods", "body": {"namespace":"ollama"}},
                    {"POST": f"/{{SERVER}}/tool/{_safe(tool_name)}", "body": {"action":"get","kind":"pods","namespace":"ollama","args":""}},
                ],
                "notes": ["If the tool requires `args` (string), the bridge will compose '-n ollama' automatically from `namespace`."]
            })
            exs.append({
                "intent": "Describe the apisix pod in the apisix namespace",
                "calls": [
                    {"GET": f"/{{SERVER}}/tool/{_safe(tool_name)}/describe/pods?namespace=apisix&name=apisix"},
                    {"POST": f"/{{SERVER}}/tool/{_safe(tool_name)}/describe/pods", "body": {"namespace":"apisix","name":"apisix"}},
                ],
                "notes": ["`name` becomes positional; `namespace` becomes '-n apisix' in argstring if needed."]
            })
    # Also surface raw lines from description as hints
    parsed = _parse_examples_from_description(description)
    if parsed:
        exs.append({"intent": "Examples from MCP description", "lines": parsed})
    return exs

def _compose_paths_from_mcp() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    paths: Dict[str, Any] = {}
    components: Dict[str, Any] = {"schemas": {}}

    for server, st in SERVERS.items():
        for tname, t in st.tools.items():
            schema = t.get("inputSchema") or {"type":"object"}
            desc = t.get("description", "") or ""
            safe = _safe(tname)
            tool_base = f"/{server}/tool/{safe}"
            actions, kinds = iter_action_kind(schema)
            comp_name = f"Args_{_safe(server)}_{safe}"
            components["schemas"][comp_name] = schema
            props = schema.get("properties", {}) if isinstance(schema, dict) else {}
            argstring_required = props.get("args", {}).get("type") == "string"

            # Base GET/POST
            for method in ("get","post"):
                opid = f"{server}_{safe}_base_{method}"
                op = {
                    "tags": [f"{server}:{tname}"],
                    "summary": t.get("title", tname),
                    "operationId": opid,
                    "description": (
                        (desc or tname) + "\n\n"
                        "CALLING RULES:\n"
                        "• Prefer POST with a JSON object body. If you have no arguments, send `{}`.\n"
                        "• If you cannot send a body: pass `?args={...}` or individual `?key=value`.\n"
                        "• Granular paths like '/{server}/tool/{tool}/{action}[/{kind}]' imply fixed fields."
                        + ("\n• This tool expects an argstring. You may pass convenience params (`namespace`, `name`, etc.); the bridge will compose it." if argstring_required else "")
                    ).strip(),
                    "parameters": _op_params_with_schema(schema, argstring_required),
                    "responses": {"200": {"description": "OK"}},
                    "x-usage": {
                        "schema": describe_schema(schema),
                        "argstringRequired": argstring_required,
                        "naturalExamples": _nl_examples_from(tname, schema, desc),
                    },
                }
                if method == "post":
                    op["requestBody"] = _make_request_body(schema, required=False)
                paths.setdefault(tool_base, {})[method] = op

            # helpers
            paths.setdefault(f"{tool_base}/invoke", {})["get"]  = {
                "tags": [f"{server}:{tname}", "invoke"],
                "summary": f"{tname} (GET /invoke)",
                "operationId": f"{server}_{safe}_invoke",
                "parameters": _op_params_with_schema(schema, argstring_required),
                "responses": {"200": {"description": "OK"}},
            }
            paths.setdefault(f"{tool_base}/schema", {})["get"]  = {
                "tags": [f"{server}:{tname}", "schema"],
                "summary": f"{tname} schema",
                "operationId": f"{server}_{safe}_schema",
                "responses": {"200": {"description": "OK"}},
            }
            paths.setdefault(f"{tool_base}/example", {})["get"] = {
                "tags": [f"{server}:{tname}", "example"],
                "summary": f"{tname} example",
                "operationId": f"{server}_{safe}_example",
                "responses": {"200": {"description": "OK"}},
            }
            paths.setdefault(f"{tool_base}/help", {})["get"]    = {
                "tags": [f"{server}:{tname}", "help"],
                "summary": f"{tname} help",
                "operationId": f"{server}_{safe}_help",
                "responses": {"200": {"description": "OK"}},
            }
            paths.setdefault(f"{tool_base}/try", {})["get"]     = {
                "tags": [f"{server}:{tname}", "try"],
                "summary": f"{tname} zero-arg try",
                "description": "Calls this tool with `{}` (no arguments).",
                "operationId": f"{server}_{safe}_try",
                "responses": {"200": {"description": "OK"}},
            }

            # Granular: /{action} and /{action}/{kind}
            if actions or kinds:
                # action only
                for action in actions or []:
                    p = f"{tool_base}/{_safe(action)}"
                    for method in ("get","post"):
                        opid = f"{server}_{safe}_{_safe(action)}_{method}"
                        op = {
                            "tags": [f"{server}:{tname}", action],
                            "summary": f"{tname} → {action}",
                            "operationId": opid,
                            "description": f"Fixes `action: \"{action}\"`. Provide only remaining fields."
                                           + ("\nConvenience params (`namespace`, `name`, etc.) are accepted and composed into the argstring." if argstring_required else ""),
                            "parameters": _op_params_with_schema(schema, argstring_required),
                            "responses": {"200": {"description": "OK"}},
                            "x-usage": {
                                "schema": describe_schema(schema),
                                "argstringRequired": argstring_required,
                                "naturalExamples": _nl_examples_from(tname, schema, desc),
                            },
                        }
                        if method == "post":
                            op["requestBody"] = _make_request_body(schema, required=False)
                        paths.setdefault(p, {})[method] = op
                    paths.setdefault(f"{p}/try", {})["get"] = {
                        "tags": [f"{server}:{tname}", action, "try"],
                        "summary": f"{tname} → {action} zero-arg try",
                        "operationId": f"{server}_{safe}_{_safe(action)}_try",
                        "responses": {"200": {"description": "OK"}},
                    }

                # action + kind, or kind only
                aks = []
                if actions and kinds:
                    aks = [(a,k) for a in actions for k in kinds]
                elif kinds:
                    aks = [("_", k) for k in kinds]

                for action, kind in aks:
                    suffix = f"{_safe(kind)}" if action == "_" else f"{_safe(action)}/{_safe(kind)}"
                    p = f"{tool_base}/{suffix}"
                    for method in ("get","post"):
                        opid = f"{server}_{safe}_{suffix.replace('/','_')}_{method}"
                        op = {
                            "tags": [f"{server}:{tname}", *( [] if action=="_" else [action] ), kind],
                            "summary": f"{tname} → {(kind if action=='_' else action+'/'+kind)}",
                            "operationId": opid,
                            "description": (
                                (f"Fixes `kind: \"{kind}\"`." if action=="_" else f"Fixes `action: \"{action}\"` and `kind: \"{kind}\"`. ")
                                + "Provide only remaining fields."
                                + ("\nConvenience params (`namespace`, `name`, etc.) are accepted and composed into the argstring." if argstring_required else "")
                            ),
                            "parameters": _op_params_with_schema(schema, argstring_required),
                            "responses": {"200": {"description": "OK"}},
                            "x-usage": {
                                "schema": describe_schema(schema),
                                "argstringRequired": argstring_required,
                                "naturalExamples": _nl_examples_from(tname, schema, desc),
                            },
                        }
                        if method == "post":
                            op["requestBody"] = _make_request_body(schema, required=False)
                        paths.setdefault(p, {})[method] = op
                    paths.setdefault(f"{p}/try", {})["get"] = {
                        "tags": [f"{server}:{tname}", *( [] if action=="_" else [action] ), kind, "try"],
                        "summary": f"{tname} → {(kind if action=='_' else action+'/'+kind)} zero-arg try",
                        "operationId": f"{server}_{safe}_{suffix.replace('/','_')}_try",
                        "responses": {"200": {"description": "OK"}},
                    }

    return paths, components

def custom_openapi():
    # Base skeleton
    openapi = {
        "openapi": "3.1.0",
        "info": {
            "title": app.title,
            "version": app.version,
            "description": app.description,
        },
        "servers": [{"url": PUBLIC_BASE_URL}],
        "paths": {},
        "components": {"schemas": {}, "securitySchemes": {
            "XApiKey": {"type": "apiKey","in": "header","name": "X-Api-Key","description":"Optional API key (if configured)."}
        }},
        "x-model-instructions": {
            "callDiscipline": [
                "Use **GET with query** or **POST with JSON**. If no args, POST `{}`.",
                "Prefer granular endpoints `/{SERVER}/tool/{TOOL}/{action}[/{kind}]` when enums exist.",
                "If the tool requires `args` (string), you can either provide it directly OR pass convenience params; the bridge will compose the argstring.",
            ],
            "bodyShapes": ["Direct body: `{ ... }`","Wrapped body: `{ \"args\": { ... } }`"],
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
                "4) If you cannot send a body, use GET with `?args={...}` or `?key=value`.",
            ],
            "errorFix": [
                "If you see 'expected a request body', use GET or POST `{}`.",
                "If a tool needs `args` (string): either send it, or pass convenience params like `namespace`, `name`, etc.",
                "If you see a schema error, inspect `/schema`, `/example`, or try a granular endpoint.",
            ],
        },
    }

    # Include fixed routes (health/discovery) from FastAPI
    fixed = get_openapi(title=app.title, version=app.version, description=app.description, routes=app.routes)
    for p, item in (fixed.get("paths") or {}).items():
        openapi["paths"].setdefault(p, item)

    # Synthesize tool paths/components directly from MCP data
    tool_paths, comp = _compose_paths_from_mcp()
    openapi["paths"].update(tool_paths)
    openapi["components"]["schemas"].update(comp.get("schemas", {}))

    # MCP catalog (tools + prompts + resources)
    openapi["x-mcp-tool-catalog"] = [
        {
            "server": sname,
            "tool": tname,
            "description": t.get("description", "No description provided by MCP server."),
            "schema": describe_schema(t.get("inputSchema") or {"type":"object"}),
        }
        for sname, st in SERVERS.items()
        for tname, t in st.tools.items()
    ]
    openapi["x-mcp-prompts"] = { sname: st.prompts for sname, st in SERVERS.items() if st.prompts }
    openapi["x-mcp-resources"] = { sname: st.resources for sname, st in SERVERS.items() if st.resources }

    # Hint if nothing discovered yet
    if not any("/tool/" in p for p in openapi.get("paths", {}).keys()):
        openapi["x-note"] = {
            "message": "No tool paths synthesized yet. Discovery runs asynchronously.",
            "actions": [
                "Call `POST /discover?wait=3` to prompt a discovery cycle.",
                "Then re-fetch `/openapi.json`."
            ]
        }
    return openapi

app.openapi = custom_openapi
