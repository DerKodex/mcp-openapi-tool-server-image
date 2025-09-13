# app.py
import os, time, asyncio, re, json
from typing import Any, Dict, Optional, List, Tuple, Union
from fastapi import FastAPI, Body, Response, HTTPException, Depends, Header, Request, Query
from fastapi.routing import APIRoute
from fastapi import APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
import httpx

# =========================
# Config
# =========================
API_KEY = os.environ.get("API_KEY", "").strip()
ALLOW_ORIGINS = os.environ.get("CORS_ALLOW_ORIGINS", "*")
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "30"))
STARTUP_TIMEOUT = int(os.environ.get("STARTUP_TIMEOUT", "30"))
REFRESH_INTERVAL = int(os.environ.get("REFRESH_INTERVAL", "10"))  # background poller seconds
DISCOVERY_PUBLIC = os.environ.get("DISCOVERY_PUBLIC", "true").lower() in ("1","true","t","yes","y","on")

# Multiple servers: "ro=http://host:8080/mcp,admin=http://host:8080/mcp"
MCP_SERVERS = os.environ.get("MCP_SERVERS", "").strip()
MCP_RPC_URL = os.environ.get("MCP_RPC_URL", "").strip()  # single-server fallback


def parse_servers(spec: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


SERVERS_CFG = parse_servers(MCP_SERVERS)
if not SERVERS_CFG:
    if MCP_RPC_URL:
        SERVERS_CFG = {"mcp": MCP_RPC_URL}
    else:
        SERVERS_CFG = {"mcp": "http://localhost:8080/mcp"}

# =========================
# App
# =========================
app = FastAPI(
    title="MCP OpenAPI Bridge (Generic, Self-Discovering)",
    version="1.6.0",
    description=(
        "A generic, self-discovering OpenAPI façade for MCP servers.\n"
        "At startup (and periodically), it queries each MCP server, discovers tools and their schemas, then exposes:\n"
        "• One generic POST/GET endpoint per tool\n"
        "• Auto-generated granular endpoints per tool for each `action` and/or `kind` enum combination\n"
        "All endpoints accept GET (query) and POST (JSON); request bodies are optional.\n"
        "Use `{}` for empty POST bodies. If you cannot send a body, use GET with `?args={...}` or individual `?key=value`."
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


def _auth_dep():
    def require_api_key(x_api_key: Optional[str] = Header(default=None)):
        if not API_KEY:
            return
        if x_api_key != API_KEY:
            raise HTTPException(status_code=401, detail="invalid api key")
    return require_api_key

# Discovery routes may be public if DISCOVERY_PUBLIC=true
RequireAPIKey = None if DISCOVERY_PUBLIC else Depends(_auth_dep())

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
# Per-server state
# =========================
class ServerState:
    def __init__(self, name: str, rpc_url: str):
        self.name = name
        self.rpc_url = rpc_url
        self.session_id: Optional[str] = None
        self.tools: Dict[str, Dict[str, Any]] = {}  # keyed by tool name
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
                        "Preferred: POST a JSON object body (use `{}` if no arguments).",
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
                        "Always send a JSON object body: `{}` for none, or `{ \"args\": { ... } }`.",
                        "Confirm tool name and argument keys match the tool schema.",
                        "Call `/SERVER/tools/list` and `/SERVER/tool/TOOL/schema` to verify fields.",
                        "If RBAC/namespace related, try the admin server or pass `namespace`.",
                    ],
                },
            )
        return data["result"]

    async def ensure_initialized(self):
        if self.session_id:
            return
        await self.rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}})

    async def refresh_tools(self):
        await self.ensure_initialized()
        result = await self.rpc("tools/list")
        tools: Dict[str, Dict[str, Any]] = {}
        tool_list = result.get("tools", result if isinstance(result, list) else [])
        for t in tool_list:
            if isinstance(t, dict) and "name" in t:
                tools[t["name"]] = t
        self.tools = tools
        self.ready = True
        self.last_error = None


SERVERS: Dict[str, ServerState] = {name: ServerState(name, url) for name, url in SERVERS_CFG.items()}

# =========================
# Helpers
# =========================
def _safe(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)

def example_from_schema(schema: Dict[str, Any]) -> Any:
    if not isinstance(schema, dict):
        return None
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
            if "default" in v:
                obj[k] = v["default"]; continue
            ex = example_from_schema(v)
            if ex is None:
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
    return schema.get("default") or None

def describe_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    desc: Dict[str, Any] = {"type": schema.get("type", "object")}
    props = schema.get("properties", {}) if isinstance(schema, dict) else {}
    required = set(schema.get("required", []))
    fields: Dict[str, Dict[str, Any]] = {}
    for k, v in props.items():
        fields[k] = {
            "required": k in required,
            "type": v.get("type", "any"),
            "enum": v.get("enum", None),
            "default": v.get("default", None),
            "description": v.get("description", None),
        }
    desc["fields"] = fields
    desc["requiredFields"] = sorted(list(required))
    hints = []
    if "namespace" in props:
        hints.append("If the tool targets Kubernetes, include `namespace` for namespace-scoped queries.")
    if "kind" in props and props["kind"].get("enum"):
        hints.append(f"`kind` accepts one of: {props['kind']['enum']}")
    if "action" in props and props["action"].get("enum"):
        hints.append(f"`action` accepts one of: {props['action']['enum']}")
    if not required:
        hints.append("This tool can be called with no arguments; you may send `{}` or use GET /…/try or /…/invoke.")
    desc["usageHints"] = hints
    return desc

def merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a); out.update(b); return out

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
    request: Optional[Request] = None
) -> Dict[str, Any]:
    base: Dict[str, Any] = {}
    if isinstance(body, dict):
        base = body.get("args", body) if isinstance(body.get("args"), dict) else body or {}
    from_args = parse_args_query(args_q)
    base = merge(base, from_args)
    if request is not None:
        qp = {k: v for k, v in request.query_params.items() if k not in ("args",)}
        base = merge(base, coerce_query_params(schema or {"type":"object"}, qp))
    return base or {}

# -------- OpenAPI helpers: request bodies & query fallbacks ----------
def make_request_body(schema: Dict[str, Any], required: bool = False) -> Dict[str, Any]:
    """
    We declare requestBody as NOT required so tool planners won't abort when no body is present.
    Handlers accept an empty body and also coerce query params (?namespace=..., ?args=...).
    """
    direct = schema or {"type": "object"}
    wrapped = {"type": "object", "properties": {"args": direct}, "required": ["args"]}
    return {
        "required": required,  # deliberately False by default
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

def schema_has_required(schema: Dict[str, Any]) -> bool:
    return bool(isinstance(schema, dict) and schema.get("required"))

# Common query params to inject into every tool POST/GET in OpenAPI
_COMMON_QUERY_PARAMS = [
    {
        "name": "args",
        "in": "query",
        "required": False,
        "description": "JSON-encoded arguments fallback when you cannot send a body. Example: ?args={\"namespace\":\"ollama\",\"kind\":\"pods\",\"action\":\"get\"}",
        "schema": {"type": "string"}
    },
    {"name": "namespace", "in": "query", "required": False, "description": "Kubernetes namespace (e.g., 'ollama').", "schema": {"type": "string"}},
    {"name": "name",      "in": "query", "required": False, "description": "Resource name (pod/deployment/etc.).", "schema": {"type": "string"}},
    {"name": "kind",      "in": "query", "required": False, "description": "Resource kind (pods, deployments, services, events, namespaces).", "schema": {"type": "string"}},
    {"name": "action",    "in": "query", "required": False, "description": "Action/verb (get, describe, logs, delete, apply).", "schema": {"type": "string"}},
    {"name": "labels",    "in": "query", "required": False, "description": "Comma-separated label selector (e.g., app=ollama,tier=backend).", "schema": {"type": "string"}},
    {"name": "fieldSelector", "in": "query", "required": False, "description": "Kubernetes field selector string.", "schema": {"type": "string"}},
    {"name": "limit",     "in": "query", "required": False, "description": "Max items to return.", "schema": {"type": "integer"}},
    {"name": "container", "in": "query", "required": False, "description": "Container name (for logs).", "schema": {"type": "string"}},
    {"name": "sinceSeconds","in": "query","required": False, "description": "Only return logs newer than X seconds.", "schema": {"type": "integer"}},
]

_registered: set[str] = set()  # Track mounted routes


# =========================
# Route builders
# =========================
def add_generic_routes(router: APIRouter, server: str):
    st = SERVERS[server]

    @router.get("/", tags=["usage"], summary="Root Quickstart")
    async def root_quickstart():
        # Try to ensure discovery for better openapi.json right away
        try:
            await st.refresh_tools()
            install_routes_for_server(server)
        except Exception:
            pass
        return {
            "howToUse": [
                "Use GET with query **or** POST with JSON. If you have no args, POST `{}`.",
                "Prefer granular paths like `/{server}/tool/{tool}/{action}[/{kind}]` when enums exist.",
                "If you see 'Request body expected', retry with `{}` or use the GET form.",
            ],
        }

    @router.get(f"/{server}/healthz", summary=f"{server} health")
    async def healthz_server():
        # Always 200; report readiness state inside
        return {
            "ok": st.ready,
            "server": server,
            "rpc_url": st.rpc_url,
            "tools": list(st.tools.keys()),
            "last_error": st.last_error,
        }

    @router.get(f"/{server}/usage", summary=f"{server} model usage guide", tags=["usage"])
    async def usage_server():
        return {
            "howToUse": [
                "Preferred: POST with a JSON body (use `{}` if no args).",
                "Fallbacks: GET '/try' or '/invoke' for zero-arg calls.",
                "Or put JSON in query: '?args={...}', or individual '?key=value' params.",
                "You may POST either a direct JSON body `{...}` or wrapped `{ \"args\": { ... } }`.",
                "Prefer granular endpoints like `/{SERVER}/tool/{TOOL}/get/pods` when available.",
                "Discover tools with `/openapi.json` and `GET /{SERVER}/tools/list`.",
                "Inspect fields with `GET /{SERVER}/tool/{TOOL}/schema` and examples with `/example`.",
            ],
            "curlExamples": [
                f"curl -X POST http://localhost:8080/{server}/tool/TOOL_NAME -H 'Content-Type: application/json' -d '{{}}'",
                f"curl -X POST http://localhost:8080/{server}/tool/TOOL_NAME/get/pods -H 'Content-Type: application/json' -d '{{}}'",
                f"curl 'http://localhost:8080/{server}/tool/TOOL_NAME/try'",
                f"curl 'http://localhost:8080/{server}/tool/TOOL_NAME/invoke?namespace=vault'",
                f"curl 'http://localhost:8080/{server}/tool/TOOL_NAME?args=%7B%22namespace%22%3A%22vault%22%7D'",
            ],
            "tip": "If you see 'expected a request body', resend with `{}` OR use GET '/try' or '/invoke' OR `?args={...}`.",
        }

    # tools/list: GET is public if DISCOVERY_PUBLIC, POST always allowed with API key
    @router.get(
        f"/{server}/tools/list",
        summary=f"{server} MCP tool listing (GET)",
        tags=["discovery"],
        dependencies=[] if DISCOVERY_PUBLIC else [Depends(_auth_dep())],
    )
    async def tools_list_get():
        await st.refresh_tools()
        return {"tools": list(st.tools.values())}

    @router.post(
        f"/{server}/tools/list",
        summary=f"{server} MCP tool listing (POST)",
        tags=["discovery"],
        dependencies=[] if DISCOVERY_PUBLIC else [Depends(_auth_dep())],
        openapi_extra={"requestBody": make_request_body({"type":"object"})},
    )
    async def tools_list_post(_body: Optional[Dict[str, Any]] = Body(default=None, embed=False)):
        await st.refresh_tools()
        return {"tools": list(st.tools.values())}

    # raw tools/call — keep POST only
    @router.post(
        f"/{server}/tools/call",
        summary=f"{server} raw MCP tool call",
        tags=["advanced"],
        dependencies=[] if DISCOVERY_PUBLIC else [Depends(_auth_dep())],
        openapi_extra={
            "x-instructions": (
                "Call a tool by name. Provide arguments directly or under an `args` object. "
                "If you cannot send a body, pass `name` and arguments via query: "
                "`?name=TOOL&args={...}` or `?name=TOOL&namespace=vault`."
            ),
            "requestBody": make_request_body({"type": "object"}),
        },
    )
    async def tools_call(
        request: Request,
        body: Optional[Dict[str, Any]] = Body(default=None, embed=False),
        name: Optional[str] = Header(default=None, description="Optional tool name header alternative"),
        tool: Optional[str] = Header(default=None, description="Alternative header for tool name"),
        q_name: Optional[str] = Query(default=None, description="Tool name (query fallback)"),
        args: Optional[str] = Query(default=None, description="JSON-encoded args (query fallback)"),
    ):
        base_args = normalize_body(body, {"type":"object"}, args, request)
        tool_name = base_args.pop("name", None) or name or tool or q_name
        if not tool_name:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Tool name not provided.",
                    "resolution": [
                        "Pass the tool name in the JSON body as `{ \"name\": \"<tool>\", ... }`, or",
                        "Use the `name:` or `tool:` header, or",
                        "Set `?name=<tool>` in the query string, or",
                        "Call a specific tool endpoint like `/{SERVER}/tool/{TOOL}`.",
                    ],
                },
            )
        await st.ensure_initialized()
        return await st.rpc("tools/call", {"name": tool_name, "arguments": base_args})


def add_schema_helpers(router: APIRouter, server: str, tool: Dict[str, Any]):
    tname = tool["name"]
    pname = _safe(tname)

    async def schema_handler():
        return tool.get("inputSchema") or {"type": "object"}

    async def example_handler():
        ex = example_from_schema(tool.get("inputSchema") or {"type": "object"})
        return {"name": tname, "arguments": ex or {}}

    async def help_handler():
        schema = tool.get("inputSchema") or {"type": "object"}
        return {
            "tool": tname,
            "server": server,
            "description": tool.get("description", "No description provided by MCP server."),
            "schema": describe_schema(schema),
            "howToUse": [
                f"POST to `/{server}/tool/{pname}`. Always send a JSON body; if no args, send `{{}}`.",
                "Fallbacks: GET '/try' for zero-arg, or pass `?args={...}` / `?key=value`, or GET '/invoke'.",
                "You may send a direct body or `{ \"args\": { ... } }`.",
                f"Fetch an example body from `/{server}/tool/{pname}/example`.",
                f"View schema at `/{server}/tool/{pname}/schema`.",
                "If granular endpoints exist (action/kind), prefer those—they pre-fill some fields.",
            ],
        }

    async def try_handler():
        return await SERVERS[server].rpc("tools/call", {"name": tname, "arguments": {}})

    router.add_api_route(f"/{server}/tool/{pname}/schema", schema_handler, ["GET"],
                         name=f"{server}:{tname} schema", summary=f"{server} → {tname} schema",
                         operation_id=f"{server}_{tname}_schema", tags=[f"{server}:{tname}", "schema"],
                         dependencies=[] if DISCOVERY_PUBLIC else [Depends(_auth_dep())])
    router.add_api_route(f"/{server}/tool/{pname}/example", example_handler, ["GET"],
                         name=f"{server}:{tname} example", summary=f"{server} → {tname} example",
                         operation_id=f"{server}_{tname}_example", tags=[f"{server}:{tname}", "example"],
                         dependencies=[] if DISCOVERY_PUBLIC else [Depends(_auth_dep())])
    router.add_api_route(f"/{server}/tool/{pname}/help", help_handler, ["GET"],
                         name=f"{server}:{tname} help", summary=f"{server} → {tname} help",
                         operation_id=f"{server}_{tname}_help", tags=[f"{server}:{tname}", "help"],
                         dependencies=[] if DISCOVERY_PUBLIC else [Depends(_auth_dep())])
    router.add_api_route(f"/{server}/tool/{pname}/try", try_handler, ["GET"],
                         name=f"{server}:{tname} try", summary=f"{server} → {tname} zero-arg try",
                         description="GET calls this tool with `{}`.", operation_id=f"{server}_{tname}_try",
                         tags=[f"{server}:{tname}", "try"],
                         dependencies=[] if DISCOVERY_PUBLIC else [Depends(_auth_dep())])


def add_tool_route(router: APIRouter, server: str, tool: Dict[str, Any]):
    st = SERVERS[server]
    tname = tool["name"]; pname = _safe(tname)
    schema = tool.get("inputSchema") or {"type": "object"}
    desc = tool.get("description") or f"Call MCP tool {tname} on server {server}. Prefer granular endpoints when possible."
    state = st; toolname = tname

    # POST (primary)
    async def handler_post(
        request: Request,
        body: Optional[Dict[str, Any]] = Body(default=None, embed=False),
        args: Optional[str] = Query(default=None, description="JSON-encoded args (query fallback)"),
    ) -> Any:
        call_args = normalize_body(body, schema, args, request)
        await state.ensure_initialized()
        return await state.rpc("tools/call", {"name": toolname, "arguments": call_args})

    post_route = APIRoute(
        path=f"/{server}/tool/{pname}",
        endpoint=handler_post,
        methods=["POST"],
        name=f"{server}:{tname}",
        summary=desc,
        operation_id=f"{server}_{tname}",
        tags=[f"{server}:{tname}"],
        dependencies=[] if DISCOVERY_PUBLIC else [Depends(_auth_dep())],
        openapi_extra={
            "x-source-description": tool.get("description", None),
            "x-instructions": (
                "Preferred: POST a JSON body. If no args, send `{}`. "
                "Fallbacks: pass args in `?args={...}` or as individual `?key=value`. "
                "Granular endpoints like `/{SERVER}/tool/{TOOL}/{action}/{kind}` are preferred when available."
            ),
            "requestBody": make_request_body(schema),
        },
    )
    router.routes.append(post_route)

    # GET (query-based invoke)
    async def handler_get(
        request: Request,
        args: Optional[str] = Query(default=None, description="JSON-encoded args (query fallback)"),
    ) -> Any:
        call_args = normalize_body({}, schema, args, request)
        await state.ensure_initialized()
        return await state.rpc("tools/call", {"name": toolname, "arguments": call_args})

    get_route = APIRoute(
        path=f"/{server}/tool/{pname}",
        endpoint=handler_get,
        methods=["GET"],
        name=f"{server}:{tname} (GET)",
        summary=f"{desc} (GET query allowed)",
        operation_id=f"{server}_{tname}_get",
        tags=[f"{server}:{tname}"],
        dependencies=[] if DISCOVERY_PUBLIC else [Depends(_auth_dep())],
    )
    # Add explicit /invoke alias to strongly hint planners
    invoke_route = APIRoute(
        path=f"/{server}/tool/{pname}/invoke",
        endpoint=handler_get,
        methods=["GET"],
        name=f"{server}:{tname} invoke",
        summary=f"{desc} (GET /invoke)",
        operation_id=f"{server}_{tname}_invoke",
        tags=[f"{server}:{tname}", "invoke"],
        dependencies=[] if DISCOVERY_PUBLIC else [Depends(_auth_dep())],
    )
    router.routes.extend([get_route, invoke_route])


def iter_action_kind(schema: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    if not isinstance(schema, Dict):
        return ([], [])
    props = schema.get("properties", {})
    action_key = next((k for k in ["action","verb","operation","op"] if k in props), None)
    actions = props.get(action_key, {}).get("enum", []) if action_key else []
    kind_key = next((k for k in ["kind","resource","resources"] if k in props), None)
    kinds = props.get(kind_key, {}).get("enum", []) if kind_key else []
    return (actions, kinds)

def add_granular_routes(router: APIRouter, server: str, tool: Dict[str, Any]):
    st = SERVERS[server]
    schema = tool.get("inputSchema") or {"type": "object"}
    actions, kinds = iter_action_kind(schema)
    if not actions and not kinds:
        return
    tname = tool["name"]; pname = _safe(tname)

    # Action-only
    for action in actions or []:
        fixed = {"action": action}
        reduced_schema = json.loads(json.dumps(schema))
        props = reduced_schema.setdefault("properties", {})
        req = set(reduced_schema.get("required", []))
        props.pop("action", None); req.discard("action")
        reduced_schema["required"] = list(req)
        op_id = f"{server}_{tname}_{_safe(action)}"
        state = st; toolname = tname; fixed_local = dict(fixed)
        base_path = f"/{server}/tool/{pname}/{_safe(action)}"

        async def handler_action_post(
            request: Request,
            body: Optional[Dict[str, Any]] = Body(default=None, embed=False),
            args: Optional[str] = Query(default=None, description="JSON-encoded args (query fallback)"),
        ) -> Any:
            call_args = normalize_body(body, reduced_schema, args, request)
            call_args.update(fixed_local)
            await state.ensure_initialized()
            return await state.rpc("tools/call", {"name": toolname, "arguments": call_args})

        async def handler_action_get(
            request: Request,
            args: Optional[str] = Query(default=None, description="JSON-encoded args (query fallback)"),
        ) -> Any:
            call_args = normalize_body({}, reduced_schema, args, request)
            call_args.update(fixed_local)
            await state.ensure_initialized()
            return await state.rpc("tools/call", {"name": toolname, "arguments": call_args})

        # POST
        router.routes.append(APIRoute(
            path=base_path, endpoint=handler_action_post, methods=["POST"],
            name=f"{server}:{tname}:{action}",
            summary=f"{server} → {tname} → {action}",
            description=f"Fixes `action: \"{action}\"`. Provide only remaining fields.",
            operation_id=op_id, tags=[f"{server}:{tname}", action],
            dependencies=[] if DISCOVERY_PUBLIC else [Depends(_auth_dep())],
            openapi_extra={"requestBody": make_request_body(reduced_schema)},
        ))
        # GET and /invoke
        router.routes.append(APIRoute(
            path=base_path, endpoint=handler_action_get, methods=["GET"],
            name=f"{server}:{tname}:{action} (GET)",
            summary=f"{server} → {tname} → {action} (GET query allowed)",
            operation_id=f"{op_id}_get", tags=[f"{server}:{tname}", action],
            dependencies=[] if DISCOVERY_PUBLIC else [Depends(_auth_dep())],
        ))
        router.routes.append(APIRoute(
            path=f"{base_path}/invoke", endpoint=handler_action_get, methods=["GET"],
            name=f"{server}:{tname}:{action}:invoke",
            summary=f"{server} → {tname} → {action} (GET /invoke)",
            operation_id=f"{op_id}_invoke", tags=[f"{server}:{tname}", action, "invoke"],
            dependencies=[] if DISCOVERY_PUBLIC else [Depends(_auth_dep())],
        ))

        if not reduced_schema.get("required"):
            async def handler_action_try():
                return await st.rpc("tools/call", {"name": tname, "arguments": dict(fixed_local)})
            router.add_api_route(
                path=f"{base_path}/try", endpoint=handler_action_try, methods=["GET"],
                name=f"{server}:{tname}:{action}:try",
                summary=f"{server} → {tname} → {action} zero-arg try",
                description="GET calls this action with `{}` (fixed fields implied).",
                operation_id=f"{op_id}_try", tags=[f"{server}:{tname}", action, "try"],
                dependencies=[] if DISCOVERY_PUBLIC else [Depends(_auth_dep())],
            )

    # Action + kind (or kind only)
    for action in actions or ["_"]:
        for kind in kinds or []:
            if action == "_":
                fixed = {"kind": kind}; suffix = _safe(kind)
                summary = f"{server} → {tname} → {kind}"
                desc = f"Fixes `kind: \"{kind}\"`."
                tags = [f"{server}:{tname}", kind]
                op_id = f"{server}_{tname}_{_safe(kind)}"
            else:
                fixed = {"action": action, "kind": kind}
                suffix = f"{_safe(action)}/{_safe(kind)}"
                summary = f"{server} → {tname} → {action}/{kind}"
                desc = f"Fixes `action: \"{action}\"` and `kind: \"{kind}\"`."
                tags = [f"{server}:{tname}", action, kind]
                op_id = f"{server}_{tname}_{_safe(action)}_{_safe(kind)}"

            reduced_schema = json.loads(json.dumps(schema))
            props = reduced_schema.setdefault("properties", {})
            req = set(reduced_schema.get("required", []))
            for k in ("action","kind"): props.pop(k, None); req.discard(k)
            reduced_schema["required"] = list(req)

            state = st; toolname = tname; fixed_local = dict(fixed)
            base_path = f"/{server}/tool/{pname}/{suffix}"

            async def handler_action_kind_post(
                request: Request,
                body: Optional[Dict[str, Any]] = Body(default=None, embed=False),
                args: Optional[str] = Query(default=None, description="JSON-encoded args (query fallback)"),
            ) -> Any:
                call_args = normalize_body(body, reduced_schema, args, request)
                call_args.update(fixed_local)
                await state.ensure_initialized()
                return await state.rpc("tools/call", {"name": toolname, "arguments": call_args})

            async def handler_action_kind_get(
                request: Request,
                args: Optional[str] = Query(default=None, description="JSON-encoded args (query fallback)"),
            ) -> Any:
                call_args = normalize_body({}, reduced_schema, args, request)
                call_args.update(fixed_local)
                await state.ensure_initialized()
                return await state.rpc("tools/call", {"name": toolname, "arguments": call_args})

            # POST
            router.routes.append(APIRoute(
                path=base_path, endpoint=handler_action_kind_post, methods=["POST"],
                name=f"{server}:{tname}:{summary}", summary=summary,
                description=f"Convenience endpoint. {desc} Provide only remaining fields.",
                operation_id=op_id, tags=tags,
                dependencies=[] if DISCOVERY_PUBLIC else [Depends(_auth_dep())],
                openapi_extra={"requestBody": make_request_body(reduced_schema)},
            ))
            # GET and /invoke
            router.routes.append(APIRoute(
                path=base_path, endpoint=handler_action_kind_get, methods=["GET"],
                name=f"{server}:{tname}:{summary} (GET)",
                summary=f"{summary} (GET query allowed)",
                operation_id=f"{op_id}_get", tags=tags,
                dependencies=[] if DISCOVERY_PUBLIC else [Depends(_auth_dep())],
            ))
            router.routes.append(APIRoute(
                path=f"{base_path}/invoke", endpoint=handler_action_kind_get, methods=["GET"],
                name=f"{server}:{tname}:{summary}:invoke",
                summary=f"{summary} (GET /invoke)",
                operation_id=f"{op_id}_invoke", tags=tags + ["invoke"],
                dependencies=[] if DISCOVERY_PUBLIC else [Depends(_auth_dep())],
            ))

            if not reduced_schema.get("required"):
                async def handler_action_kind_try():
                    return await st.rpc("tools/call", {"name": tname, "arguments": dict(fixed_local)})
                router.add_api_route(
                    path=f"{base_path}/try", endpoint=handler_action_kind_try, methods=["GET"],
                    name=f"{server}:{tname}:{summary}:try",
                    summary=f"{summary} zero-arg try",
                    description="GET calls this action/kind with `{}` (fixed fields implied).",
                    operation_id=f"{op_id}_try", tags=tags + ["try"],
                    dependencies=[] if DISCOVERY_PUBLIC else [Depends(_auth_dep())],
                )

# =========================
# Startup / refresh (resilient)
# =========================
def install_routes_for_server(server: str):
    st = SERVERS[server]
    router = APIRouter()
    add_generic_routes(router, server)
    for tool in st.tools.values():
        key_base = f"{server}::{tool['name']}"
        if key_base + "::base" not in _registered:
            add_schema_helpers(router, server, tool)
            add_tool_route(router, server, tool)
            _registered.add(key_base + "::base")
        if key_base + "::granular" not in _registered:
            add_granular_routes(router, server, tool)
            _registered.add(key_base + "::granular")
    app.include_router(router)

async def background_poller():
    # Keep trying forever; install routes as soon as servers are reachable
    while True:
        for name, st in SERVERS.items():
            try:
                await st.refresh_tools()
                install_routes_for_server(name)
            except Exception:
                # error is recorded in st.last_error/st.ready
                pass
        await asyncio.sleep(REFRESH_INTERVAL)

@app.on_event("startup")
async def startup():
    # Do NOT raise on startup; just kick off the poller
    asyncio.create_task(background_poller())

# Liveness: always 200 once app is running
@app.get("/livez", tags=["health"])
async def livez():
    return {"ok": True}

# Readiness: ready if ANY server is ready
@app.get("/readyz", tags=["health"])
async def readyz():
    any_ready = any(st.ready for st in SERVERS.values())
    status = 200 if any_ready else 503
    return JSONResponse(
        status_code=status,
        content={
            "ok": any_ready,
            "servers": {k: {"ready": v.ready, "tools": list(v.tools.keys()), "last_error": v.last_error} for k, v in SERVERS.items()},
        },
    )

# Health (informational; never used for pod death)
@app.get("/healthz", tags=["health"])
async def healthz():
    return {
        "ok": any(st.ready for st in SERVERS.values()),
        "servers": {k: {"ready": v.ready, "rpc_url": v.rpc_url, "tools": list(v.tools.keys()), "last_error": v.last_error}
                    for k, v in SERVERS.items()},
    }

@app.get("/servers", tags=["info"])
async def servers():
    return {"servers": {k: v.rpc_url for k, v in SERVERS.items()}}

@app.post("/refresh", tags=["admin"], dependencies=[] if DISCOVERY_PUBLIC else [Depends(_auth_dep())])
async def refresh():
    for name, st in SERVERS.items():
        await st.refresh_tools()
        install_routes_for_server(name)
    return {"ok": True, "servers": {k: list(v.tools.keys()) for k, v in SERVERS.items()}}

# =========================
# Error handlers with guidance
# =========================
@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict):
        payload = exc.detail
    else:
        payload = {"message": str(exc.detail)}
    if "resolution" not in payload:
        tips = [
            "Preferred: POST a JSON object body (use `{}` if no arguments).",
            "Fallbacks: GET '/try' or '/invoke' for zero-arg, or pass `?args={...}` / individual `?key=value`.",
            "Discover tools via `/{SERVER}/tools/list` and inspect fields with `/tool/{TOOL}/schema`.",
            "Use `/tool/{TOOL}/example` or granular endpoints.",
        ]
        if exc.status_code == 401:
            tips.insert(0, "Include the `X-Api-Key` header if the bridge was configured with API_KEY.")
        if exc.status_code == 422:
            tips.insert(0, "Ensure the body is a JSON object (or use GET '/try' / '/invoke' / `?args={...}` as a fallback).")
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
# Custom OpenAPI with explicit model guidance
# =========================
def _build_k8s_cookbook() -> List[Dict[str, Any]]:
    cookbook: List[Dict[str, Any]] = []
    for server, st in SERVERS.items():
        for t in st.tools.values():
            name = t.get("name", "")
            schema = t.get("inputSchema") or {}
            props = schema.get("properties", {}) if isinstance(schema, dict) else {}
            # Generic K8s intents if tool looks like a k8s tool
            if ("kube" in name or "kubectl" in name or "k8" in name) and ("kind" in props or "action" in props):
                tool_path = f"/{server}/tool/{_safe(name)}"
                cookbook.extend([
                    {
                        "intent": "List all pods in the <namespace> namespace",
                        "preferred": f"POST {tool_path}/get/pods",
                        "body": {"namespace": "<namespace>"},
                        "alternatives": [
                            {"call": f"GET  {tool_path}/get/pods?namespace=<namespace>"},
                            {"call": f"GET  {tool_path}/get/pods/invoke?namespace=<namespace>"},
                            {"call": f"POST {tool_path}", "body": {"action":"get","kind":"pods","namespace":"<namespace>"}},
                            {"call": f"GET  {tool_path}?namespace=<namespace>&action=get&kind=pods"},
                            {"call": f"GET  {tool_path}?args=%7B%22namespace%22%3A%22<namespace>%22%2C%22kind%22%3A%22pods%22%2C%22action%22%3A%22get%22%7D"},
                        ],
                    },
                    {
                        "intent": "Describe the <pod> pod in the <namespace> namespace",
                        "preferred": f"POST {tool_path}/get/pods",
                        "body": {"namespace": "<namespace>", "name": "<pod>"},
                        "alternatives": [
                            {"call": f"GET {tool_path}/get/pods?namespace=<namespace>&name=<pod>"},
                            {"call": f"POST {tool_path}", "body": {"action":"get","kind":"pods","namespace":"<namespace>","name":"<pod>"}}
                        ]
                    },
                    {
                        "intent": "Get logs for the <pod> pod in the <namespace> namespace",
                        "preferred": f"POST {tool_path}/logs/pods",
                        "body": {"namespace": "<namespace>", "name": "<pod>", "container": "<optional>"},
                    },
                ])
    return cookbook

def _augment_operation_docs(openapi_schema: Dict[str, Any]):
    """
    Make each tool operation extremely explicit:
    - requestBody not required (but supported).
    - Common K8s params in query.
    - Natural-language examples.
    - Preserve MCP tool descriptions as x-source-description.
    """
    paths = openapi_schema.get("paths", {})
    for path, methods in list(paths.items()):
        if "/tool/" not in path:
            continue
        for verb, op in list(methods.items()):
            if verb.lower() not in ("post", "get"):
                continue

            # Ensure requestBody exists and is NOT required for POST
            if verb.lower() == "post":
                rb = op.get("requestBody") or {}
                rb["required"] = False
                op["requestBody"] = rb

            # Add common query params if not present
            existing = {(p.get("name"), p.get("in")) for p in op.get("parameters", [])}
            params = op.setdefault("parameters", [])
            for qp in _COMMON_QUERY_PARAMS:
                key = (qp["name"], qp["in"])
                if key not in existing:
                    params.append(qp)

            # Strengthen description and usage
            guidance = (
                "CALLING RULES:\n"
                "• Prefer POST with a JSON object body. If you have no arguments, send `{}`.\n"
                "• If you cannot send a body: use `?args={...}` or individual `?key=value` such as `?namespace=ollama`.\n"
                "• Prefer granular paths like '/{server}/tool/{tool}/get/pods' when action/kind are implied.\n"
                "• If you see 'Request body expected', retry the same call with an empty body `{}` or use GET.\n"
            )
            op["description"] = ((op.get("description") or "") + "\n\n" + guidance).strip()

            # Add a canonical NL example
            examples = op.setdefault("x-naturalExamples", [])
            if not any(e.get("intent") == "List all pods in the ollama namespace" for e in examples):
                examples.append({
                    "intent": "List all pods in the ollama namespace",
                    "how": [
                        "Extract `namespace=ollama` from the phrase.",
                        "Use a granular endpoint if available: '/…/get/pods'.",
                        "Otherwise call the generic tool with `{ \"action\":\"get\", \"kind\":\"pods\", \"namespace\":\"ollama\" }`.",
                        "If you cannot send a body, pass `?namespace=ollama` or `?args={\"namespace\":\"ollama\",\"kind\":\"pods\",\"action\":\"get\"}`."
                    ]
                })

def _build_tool_catalog() -> List[Dict[str, Any]]:
    catalog: List[Dict[str, Any]] = []
    for server, st in SERVERS.items():
        for t in st.tools.values():
            name = t.get("name","")
            schema = t.get("inputSchema") or {"type":"object"}
            desc = t.get("description") or "No description provided by MCP server."
            actions, kinds = iter_action_kind(schema)
            catalog.append({
                "server": server,
                "tool": name,
                "description": desc,
                "schema": describe_schema(schema),
                "granular": {
                    "actions": actions,
                    "kinds": kinds,
                    "paths": {
                        "generic": f"/{server}/tool/{_safe(name)}",
                        "invoke": f"/{server}/tool/{_safe(name)}/invoke",
                        "example": f"/{server}/tool/{_safe(name)}/example",
                        "schema":  f"/{server}/tool/{_safe(name)}/schema",
                        "help":    f"/{server}/tool/{_safe(name)}/help",
                        "try":     f"/{server}/tool/{_safe(name)}/try",
                    }
                },
            })
    return catalog

async def _ensure_discovered_now():
    # Synchronously attempt discovery right before emitting /openapi.json
    deadline = time.time() + STARTUP_TIMEOUT
    last_err = None
    while time.time() < deadline:
        try:
            for name, st in SERVERS.items():
                try:
                    await st.refresh_tools()
                    install_routes_for_server(name)
                except Exception as e:
                    last_err = e
            return
        except Exception as e:
            last_err = e
        await asyncio.sleep(1)
    if last_err:
        # Don’t crash; just proceed with limited schema
        pass

def custom_openapi_sync(openapi_routes):
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=openapi_routes,
    )
    openapi_schema["x-model-instructions"] = {
        "callDiscipline": [
            "Use **GET with query** or **POST with JSON**. If no args, POST `{}`.",
            "Prefer granular endpoints `/{SERVER}/tool/{TOOL}/{action}[/{kind}]` when enums exist.",
        ],
        "bodyShapes": ["Direct body: `{ ... }`", "Wrapped body: `{ \"args\": { ... } }`"],
        "discovery": [
            "List tools: `GET or POST /{SERVER}/tools/list` (GET is public if DISCOVERY_PUBLIC=true).",
            "Per-tool schema: `GET /{SERVER}/tool/{TOOL}/schema`.",
            "Per-tool example: `GET /{SERVER}/tool/{TOOL}/example`.",
            "Per-tool help: `GET /{SERVER}/tool/{TOOL}/help`.",
            "Zero-argument test: `GET /{SERVER}/tool/{TOOL}/try` when available.",
        ],
        "typicalFlow": [
            "1) Read `/openapi.json` and `/SERVER/tools/list`.",
            "2) For a user intent like “<action> <kind> …”, pick the granular path `/{SERVER}/tool/{TOOL}/{action}/{kind}`.",
            "3) If unsure which fields are required, call `/schema` or `/example` first.",
            "4) Prefer GET if you cannot send a body; otherwise POST `{}` or the minimal JSON.",
        ],
        "errorFix": [
            "If you see 'expected a request body', use GET or POST `{}`.",
            "If you see a schema error, inspect `/schema`, `/example`, or try a granular endpoint.",
        ],
    }
    openapi_schema["x-cookbook"] = _build_k8s_cookbook()
    openapi_schema["x-mcp-tool-catalog"] = _build_tool_catalog()
    openapi_schema.setdefault("components", {}).setdefault("securitySchemes", {})
    openapi_schema["components"]["securitySchemes"]["XApiKey"] = {
        "type": "apiKey",
        "in": "header",
        "name": "X-Api-Key",
        "description": "Optional API key; set if the bridge was started with API_KEY.",
    }
    _augment_operation_docs(openapi_schema)
    return openapi_schema

# Replace app.openapi with a version that forces discovery first
def custom_openapi():
    # Force a synchronous discovery attempt so tool routes appear in the schema
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    try:
        if loop.is_running():
            # running under uvicorn; schedule a one-shot task and wait briefly
            # (best-effort; if not possible, schema will reflect whatever is installed)
            fut = asyncio.run_coroutine_threadsafe(_ensure_discovered_now(), loop)
            try: fut.result(timeout=STARTUP_TIMEOUT)
            except Exception: pass
        else:
            loop.run_until_complete(_ensure_discovered_now())
    except Exception:
        pass

    # Build with the (hopefully) fully-installed routes
    app.openapi_schema = custom_openapi_sync(app.routes)
    return app.openapi_schema

app.openapi = custom_openapi
