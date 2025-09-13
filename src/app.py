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
DISCOVERY_PUBLIC = os.environ.get("DISCOVERY_PUBLIC", "true").lower() in ("1","true","yes","on")

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
    version="1.5.0",
    description=(
        "A generic, self-discovering OpenAPI façade for MCP servers.\n"
        "At startup (and periodically), it queries each MCP server, discovers tools and their schemas, "
        "then exposes:\n"
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

# -------------------------
# Auth helpers
# -------------------------
def _needs_key() -> bool:
    return bool(API_KEY)

def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    if not _needs_key():
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")

def maybe_require_api_key_for(discovery: bool):
    # Allow unauthenticated discovery GETs if DISCOVERY_PUBLIC, otherwise require the key
    if discovery and DISCOVERY_PUBLIC:
        return []
    return [Depends(require_api_key)]

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
                        "If RBAC/namespace related, try a different server or pass required scope fields.",
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
        hints.append("If the tool targets a scoped domain (e.g., Kubernetes), include a scope like `namespace` when relevant.")
    for k in ("kind", "resource", "resources"):
        if k in props and props[k].get("enum"):
            hints.append(f"`{k}` accepts one of: {props[k]['enum']}")
    for k in ("action","verb","operation","op"):
        if k in props and props[k].get("enum"):
            hints.append(f"`{k}` accepts one of: {props[k]['enum']}")
    if not required:
        hints.append("This tool can be called with no arguments; you may send `{}` or use GET /…/try.")
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

def make_request_body(schema: Dict[str, Any], required: bool = False) -> Dict[str, Any]:
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

_COMMON_QUERY_PARAMS = [
    {"name": "args", "in": "query", "required": False,
     "description": "JSON-encoded arguments fallback (e.g. ?args={\"action\":\"get\",\"kind\":\"items\"})",
     "schema": {"type": "string"}},
]

_registered: set[str] = set()  # Track mounted routes


# =========================
# Routes
# =========================
@app.get("/", tags=["usage"])
async def root_quickstart():
    return {
        "message": "MCP OpenAPI Bridge (Generic)",
        "discoveryPublic": DISCOVERY_PUBLIC,
        "next": ["GET /{server}/usage", "GET /{server}/tools/list"],
        "tip": "Use **GET with query** or **POST with JSON**; if no args, send `{}`.",
    }

def add_generic_routes(router: APIRouter, server: str):
    st = SERVERS[server]

    @router.get(f"/{server}/healthz", summary=f"{server} health")
    async def healthz_server():
        return {
            "ok": st.ready,
            "server": server,
            "rpc_url": st.rpc_url,
            "tools": list(st.tools.keys()),
            "last_error": st.last_error,
        }

    @router.get(f"/{server}/usage", summary=f"{server} usage guide", tags=["usage"])
    async def usage_server():
        return {
            "howToUse": [
                "Preferred: **GET with query** or **POST with JSON** (use `{}` if no args).",
                "Every tool endpoint supports POST with optional body and GET with `?args={...}`.",
                "When a tool's schema has `action` and/or `kind` enums, granular endpoints are created for each combo.",
                "Use `/tool/{TOOL}/schema` + `/example` to inspect fields and see sample bodies."
            ],
            "discovery": [
                f"GET /{server}/tools/list",
                f"GET /{server}/tool/<TOOL>/schema",
                f"GET /{server}/tool/<TOOL>/example",
                f"GET /{server}/tool/<TOOL>/help",
                f"GET /{server}/tool/<TOOL>/try",
            ],
            "tip": "If you see 'expected a request body', resend the same call with an empty JSON body `{}` or use GET with query.",
        }

    # tools/list: GET (public discovery if enabled) + POST (auth if configured)
    @router.get(
        f"/{server}/tools/list",
        summary=f"{server} list MCP tools (GET)",
        dependencies=maybe_require_api_key_for(discovery=True),
    )
    async def tools_list_get():
        await st.refresh_tools()
        return {"tools": list(st.tools.values())}

    @router.post(
        f"/{server}/tools/list",
        summary=f"{server} list MCP tools (POST)",
        dependencies=maybe_require_api_key_for(discovery=False),
        openapi_extra={
            "x-instructions": "Lists tools on this MCP server. Send `{}` if you have no arguments.",
            "x-priority": "low",
            "requestBody": make_request_body({"type": "object"}),
        },
    )
    async def tools_list_post(_body: Optional[Dict[str, Any]] = Body(default=None, embed=False)):
        await st.refresh_tools()
        return {"tools": list(st.tools.values())}

    # tools/call: GET + POST (auth if configured)
    @router.get(
        f"/{server}/tools/call",
        summary=f"{server} raw MCP tool call (GET)",
        dependencies=maybe_require_api_key_for(discovery=False),
        openapi_extra={"x-priority": "medium"},
    )
    async def tools_call_get(
        request: Request,
        name: Optional[str] = Query(default=None),
        args: Optional[str] = Query(default=None),
    ):
        if not name:
            raise HTTPException(status_code=422, detail={"message": "Missing ?name=<tool>", "resolution": [
                "Provide ?name and either ?args={} or individual ?key=value matching the tool schema."
            ]})
        call_args = normalize_body({}, {"type": "object"}, args, request)
        await st.ensure_initialized()
        return await st.rpc("tools/call", {"name": name, "arguments": call_args})

    @router.post(
        f"/{server}/tools/call",
        summary=f"{server} raw MCP tool call (POST)",
        dependencies=maybe_require_api_key_for(discovery=False),
        openapi_extra={
            "x-instructions": (
                "Call a tool by name. Provide arguments directly or under an `args` object. "
                "If you cannot send a body, use the GET form with `?name=TOOL&args={...}`."
            ),
            "x-priority": "medium",
            "requestBody": make_request_body({"type": "object"}),
        },
    )
    async def tools_call_post(
        request: Request,
        body: Optional[Dict[str, Any]] = Body(default=None, embed=False),
        name: Optional[str] = Header(default=None),
        tool: Optional[str] = Header(default=None),
        q_name: Optional[str] = Query(default=None),
        args: Optional[str] = Query(default=None),
    ):
        base_args = normalize_body(body, {"type":"object"}, args, request)
        tool_name = base_args.pop("name", None) or name or tool or q_name
        if not tool_name:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Tool name not provided.",
                    "resolution": [
                        "Pass `{\"name\":\"<tool>\", ...}` in the body, or header `name:`/`tool:`, or `?name=<tool>`.",
                        "Or call a specific endpoint like `/{SERVER}/tool/{TOOL}`.",
                    ],
                },
            )
        await st.ensure_initialized()
        return await st.rpc("tools/call", {"name": tool_name, "arguments": base_args})


# -------------------------
# Tool routes (GET+POST, body optional)
# -------------------------
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
        # Build hints about enums to guide models
        props = schema.get("properties", {}) if isinstance(schema, dict) else {}
        actions = None
        kinds = None
        for k in ("action","verb","operation","op"):
            if k in props and props[k].get("enum"):
                actions = props[k]["enum"]; break
        for k in ("kind","resource","resources"):
            if k in props and props[k].get("enum"):
                kinds = props[k]["enum"]; break

        return {
            "tool": tname,
            "server": server,
            "description": tool.get("description", "No description provided by MCP server."),
            "schema": describe_schema(schema),
            "howToUse": [
                f"POST to `/{server}/tool/{pname}`. If no args, send `{{}}`.",
                "Or use GET with `?args={...}` or individual `?key=value`.",
                "When action/kind enums exist, granular endpoints are created: "
                f"`/{server}/tool/{pname}/<action>` and `/{server}/tool/{pname}/<action>/<kind>`.",
                "Prefer granular endpoints when your intent maps to a known action/kind.",
                "If you see 'Request body expected', resend the same call with an empty JSON body `{}`.",
            ],
            "actions": actions or [],
            "kinds": kinds or [],
        }

    async def try_handler():
        return await SERVERS[server].rpc("tools/call", {"name": tname, "arguments": {}})

    # Discovery endpoints are GET and public if DISCOVERY_PUBLIC=true
    router.add_api_route(f"/{server}/tool/{pname}/schema", schema_handler, ["GET"],
                         name=f"{server}:{tname} schema", summary=f"{server} → {tname} schema",
                         operation_id=f"{server}_{tname}_schema", tags=[f"{server}:{tname}", "schema"],
                         dependencies=maybe_require_api_key_for(discovery=True),
                         openapi_extra={"x-priority": "low"})
    router.add_api_route(f"/{server}/tool/{pname}/example", example_handler, ["GET"],
                         name=f"{server}:{tname} example", summary=f"{server} → {tname} example",
                         operation_id=f"{server}_{tname}_example", tags=[f"{server}:{tname}", "example"],
                         dependencies=maybe_require_api_key_for(discovery=True),
                         openapi_extra={"x-priority": "low"})
    router.add_api_route(f"/{server}/tool/{pname}/help", help_handler, ["GET"],
                         name=f"{server}:{tname} help", summary=f"{server} → {tname} help",
                         operation_id=f"{server}_{tname}_help", tags=[f"{server}:{tname}", "help"],
                         dependencies=maybe_require_api_key_for(discovery=True),
                         openapi_extra={"x-priority": "low"})
    router.add_api_route(f"/{server}/tool/{pname}/try", try_handler, ["GET"],
                         name=f"{server}:{tname} try", summary=f"{server} → {tname} zero-arg try",
                         description="GET calls this tool with `{}`.", operation_id=f"{server}_{tname}_try",
                         tags=[f"{server}:{tname}", "try"],
                         dependencies=maybe_require_api_key_for(discovery=True),
                         openapi_extra={"x-priority": "low"})


def add_tool_route(router: APIRouter, server: str, tool: Dict[str, Any]):
    st = SERVERS[server]
    tname = tool["name"]; pname = _safe(tname)
    schema = tool.get("inputSchema") or {"type": "object"}
    desc = tool.get("description") or f"Call MCP tool {tname} on server {server}. Prefer granular endpoints when possible."
    state = st; toolname = tname

    async def handler_get(
        request: Request,
        args: Optional[str] = Query(default=None, description="JSON-encoded args (query fallback)"),
    ) -> Any:
        call_args = normalize_body({}, schema, args, request)
        await state.ensure_initialized()
        return await state.rpc("tools/call", {"name": toolname, "arguments": call_args})

    async def handler_post(
        request: Request,
        body: Optional[Dict[str, Any]] = Body(default=None, embed=False),
        args: Optional[str] = Query(default=None, description="JSON-encoded args (query fallback)"),
    ) -> Any:
        call_args = normalize_body(body, schema, args, request)
        await state.ensure_initialized()
        return await state.rpc("tools/call", {"name": toolname, "arguments": call_args})

    # GET
    router.add_api_route(
        path=f"/{server}/tool/{pname}",
        endpoint=handler_get,
        methods=["GET"],
        name=f"{server}:{tname}:get",
        summary=f"{desc} (GET).",
        operation_id=f"{server}_{tname}_get",
        tags=[f"{server}:{tname}"],
        dependencies=maybe_require_api_key_for(discovery=False),
        openapi_extra={"x-priority": "medium"},
    )

    # POST
    route = APIRoute(
        path=f"/{server}/tool/{pname}",
        endpoint=handler_post,
        methods=["POST"],
        name=f"{server}:{tname}",
        summary=desc,
        operation_id=f"{server}_{tname}",
        tags=[f"{server}:{tname}"],
        dependencies=maybe_require_api_key_for(discovery=False),
        openapi_extra={
            "x-instructions": (
                "Preferred: POST a JSON body. If no args, send `{}`. Or use the GET form with query params."
            ),
            "x-priority": "medium",
            "requestBody": make_request_body(schema),
        },
    )
    router.routes.append(route)


def iter_action_kind(schema: Dict[str, Any]) -> Tuple[List[str], List[str], Optional[str], Optional[str]]:
    if not isinstance(schema, Dict):
        return ([], [], None, None)
    props = schema.get("properties", {})
    action_key = next((k for k in ["action","verb","operation","op"] if k in props), None)
    actions = props.get(action_key, {}).get("enum", []) if action_key else []
    kind_key = next((k for k in ["kind","resource","resources"] if k in props), None)
    kinds = props.get(kind_key, {}).get("enum", []) if kind_key else []
    return (actions, kinds, action_key, kind_key)

def add_granular_routes(router: APIRouter, server: str, tool: Dict[str, Any]):
    st = SERVERS[server]
    schema = tool.get("inputSchema") or {"type": "object"}
    actions, kinds, action_key, kind_key = iter_action_kind(schema)
    if not actions and not kinds:
        return
    tname = tool["name"]; pname = _safe(tname)

    # base reducer helper
    def strip_fields(s: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
        r = json.loads(json.dumps(s))
        props = r.setdefault("properties", {})
        req = set(r.get("required", []))
        for f in fields:
            props.pop(f, None); req.discard(f)
        r["required"] = list(req)
        return r

    # Action-only
    for action in actions or []:
        fixed = {action_key: action} if action_key else {"action": action}
        reduced_schema = strip_fields(schema, [action_key] if action_key else ["action"])
        op_id = f"{server}_{tname}_{_safe(action)}"
        state = st; toolname = tname; fixed_local = dict(fixed)
        path = f"/{server}/tool/{pname}/{_safe(action)}"

        async def handler_action_get(request: Request, args: Optional[str] = Query(default=None)) -> Any:
            call_args = normalize_body({}, reduced_schema, args, request)
            call_args.update(fixed_local)
            await state.ensure_initialized()
            return await state.rpc("tools/call", {"name": toolname, "arguments": call_args})

        async def handler_action_post(
            request: Request,
            body: Optional[Dict[str, Any]] = Body(default=None, embed=False),
            args: Optional[str] = Query(default=None),
        ) -> Any:
            call_args = normalize_body(body, reduced_schema, args, request)
            call_args.update(fixed_local)
            await state.ensure_initialized()
            return await state.rpc("tools/call", {"name": toolname, "arguments": call_args})

        router.add_api_route(
            path=path, endpoint=handler_action_get, methods=["GET"],
            name=f"{server}:{tname}:{action}:get",
            summary=f"{server} → {tname} → {action} (GET)",
            operation_id=f"{op_id}_get", tags=[f"{server}:{tname}", action],
            dependencies=maybe_require_api_key_for(discovery=False),
            openapi_extra={
                "x-priority": "high",
                "x-naturalExamples": [{"intent": f"{action} (no kind specified)", "call": f"GET {path}"}],
                "description": (
                    f"CALLING RULES:\n"
                    f"• Prefer **GET with query** or **POST with JSON**. If no remaining fields, send `{{}}` (POST).\n"
                    f"• Fixed `{action_key or 'action'}` is implied by the path.\n"
                ),
            },
        )
        route = APIRoute(
            path=path, endpoint=handler_action_post, methods=["POST"],
            name=f"{server}:{tname}:{action}",
            summary=f"{server} → {tname} → {action}",
            operation_id=op_id, tags=[f"{server}:{tname}", action],
            dependencies=maybe_require_api_key_for(discovery=False),
            openapi_extra={
                "x-priority": "high",
                "x-naturalExamples": [{"intent": f"{action} (no kind specified)", "body": {}}],
                "requestBody": make_request_body(reduced_schema),
                "description": (
                    f"CALLING RULES:\n"
                    f"• POST with JSON body (send `{{}}` if none) or use GET with query.\n"
                    f"• Fixed `{action_key or 'action'}` is implied by the path.\n"
                ),
            },
        )
        router.routes.append(route)

        if not reduced_schema.get("required"):
            async def handler_action_try():
                return await st.rpc("tools/call", {"name": tname, "arguments": dict(fixed_local)})
            router.add_api_route(
                path=f"{path}/try", endpoint=handler_action_try, methods=["GET"],
                name=f"{server}:{tname}:{action}:try",
                summary=f"{server} → {tname} → {action} zero-arg try",
                description="GET calls this action with `{}` (fixed fields implied).",
                operation_id=f"{op_id}_try", tags=[f"{server}:{tname}", action, "try"],
                dependencies=maybe_require_api_key_for(discovery=True),
                openapi_extra={"x-priority": "low"},
            )

    # Action + kind (or kind only)
    for action in (actions or ["_"]):
        for kind in kinds or []:
            if action == "_":
                fixed = {kind_key: kind} if kind_key else {"kind": kind}
                suffix = _safe(kind)
                summary = f"{server} → {tname} → {kind}"
                tags = [f"{server}:{tname}", kind]
                op_id = f"{server}_{tname}_{_safe(kind)}"
                drop_fields = [kind_key] if kind_key else ["kind"]
            else:
                fixed = { (action_key or "action"): action, (kind_key or "kind"): kind }
                suffix = f"{_safe(action)}/{_safe(kind)}"
                summary = f"{server} → {tname} → {action}/{kind}"
                tags = [f"{server}:{tname}", action, kind]
                op_id = f"{server}_{tname}_{_safe(action)}_{_safe(kind)}"
                drop_fields = [ (action_key or "action"), (kind_key or "kind") ]

            reduced_schema = json.loads(json.dumps(schema))
            props = reduced_schema.setdefault("properties", {})
            req = set(reduced_schema.get("required", []))
            for k in drop_fields: props.pop(k, None); req.discard(k)
            reduced_schema["required"] = list(req)

            state = st; toolname = tname; fixed_local = dict(fixed)
            path = f"/{server}/tool/{pname}/{suffix}"

            async def handler_action_kind_get(request: Request, args: Optional[str] = Query(default=None)) -> Any:
                call_args = normalize_body({}, reduced_schema, args, request)
                call_args.update(fixed_local)
                await state.ensure_initialized()
                return await state.rpc("tools/call", {"name": toolname, "arguments": call_args})

            async def handler_action_kind_post(
                request: Request,
                body: Optional[Dict[str, Any]] = Body(default=None, embed=False),
                args: Optional[str] = Query(default=None),
            ) -> Any:
                call_args = normalize_body(body, reduced_schema, args, request)
                call_args.update(fixed_local)
                await state.ensure_initialized()
                return await state.rpc("tools/call", {"name": toolname, "arguments": call_args})

            router.add_api_route(
                path=path, endpoint=handler_action_kind_get, methods=["GET"],
                name=f"{server}:{tname}:{summary}:get",
                summary=f"{summary} (GET)", operation_id=f"{op_id}_get", tags=tags,
                dependencies=maybe_require_api_key_for(discovery=False),
                openapi_extra={
                    "x-priority": "high",
                    "x-naturalExamples": [{"intent": f"{action} {kind}" if action != "_" else f"{kind}", "call": f"GET {path}"}],
                    "description": (
                        "CALLING RULES:\n"
                        "• Prefer **GET with query** or **POST with JSON**. If no remaining fields, send `{}` (POST).\n"
                        f"• Fixed fields implied by the path: {', '.join([f'{k}={v!r}' for k,v in fixed_local.items()])}\n"
                    ),
                },
            )
            route = APIRoute(
                path=path, endpoint=handler_action_kind_post, methods=["POST"],
                name=f"{server}:{tname}:{summary}", summary=summary,
                operation_id=op_id, tags=tags,
                dependencies=maybe_require_api_key_for(discovery=False),
                openapi_extra={
                    "x-priority": "high",
                    "x-naturalExamples": [{"intent": f"{action} {kind}" if action != "_" else f"{kind}", "body": {}}],
                    "requestBody": make_request_body(reduced_schema),
                    "description": (
                        "CALLING RULES:\n"
                        "• POST with JSON body (send `{}` if none) or use GET with query.\n"
                        f"• Fixed fields implied by the path: {', '.join([f'{k}={v!r}' for k,v in fixed_local.items()])}\n"
                    ),
                },
            )
            router.routes.append(route)

            if not reduced_schema.get("required"):
                async def handler_action_kind_try():
                    return await st.rpc("tools/call", {"name": tname, "arguments": dict(fixed_local)})
                router.add_api_route(
                    path=f"{path}/try", endpoint=handler_action_kind_try, methods=["GET"],
                    name=f"{server}:{tname}:{summary}:try",
                    summary=f"{summary} zero-arg try",
                    description="GET calls this action/kind with `{}` (fixed fields implied).",
                    operation_id=f"{op_id}_try", tags=tags + ["try"],
                    dependencies=maybe_require_api_key_for(discovery=True),
                    openapi_extra={"x-priority": "low"},
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
    while True:
        for name, st in SERVERS.items():
            try:
                await st.refresh_tools()
                install_routes_for_server(name)
            except Exception:
                pass
        await asyncio.sleep(REFRESH_INTERVAL)

@app.on_event("startup")
async def startup():
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
            "discoveryPublic": DISCOVERY_PUBLIC,
            "servers": {k: {"ready": v.ready, "tools": list(v.tools.keys()), "last_error": v.last_error} for k, v in SERVERS.items()},
        },
    )

@app.get("/healthz", tags=["health"])
async def healthz():
    return {
        "ok": any(st.ready for st in SERVERS.values()),
        "discoveryPublic": DISCOVERY_PUBLIC,
        "servers": {k: {"ready": v.ready, "rpc_url": v.rpc_url, "tools": list(v.tools.keys()), "last_error": v.last_error}
                    for k, v in SERVERS.items()},
    }

@app.get("/servers", tags=["info"])
async def servers():
    # Mark as low priority so planners don't overuse it
    return {"servers": {k: v.rpc_url for k, v in SERVERS.items()}, "x-priority": "low"}

@app.post("/refresh", tags=["admin"], dependencies=[Depends(require_api_key)])
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
            "Use **GET with query** or **POST with JSON** (send `{}` if no args).",
            "Prefer granular endpoints `/{SERVER}/tool/{TOOL}/{action}` or `/{action}/{kind}` when enums exist.",
            "Discover tools via `/SERVER/tools/list` and inspect `/tool/{TOOL}/schema`.",
        ]
        if exc.status_code == 401:
            tips.insert(0, "Include header `X-Api-Key` if the bridge was configured with API_KEY.")
        if exc.status_code == 422:
            tips.insert(0, "Provide missing parameters via body `{}` or query `?key=value` / `?args={...}`.")
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
                "Use **GET with query** or **POST with JSON**; if no args, send `{}`.",
                "Prefer granular endpoints where `action`/`kind` are implied by the path.",
                "Retry with a simpler body; check bridge pod logs for stack traces.",
                "Verify MCP server availability and NetworkPolicy.",
            ],
        },
    )

# =========================
# Custom OpenAPI with explicit model guidance
# =========================
def _derive_intents_from_tools() -> List[Dict[str, Any]]:
    intents: List[Dict[str, Any]] = []
    for server, st in SERVERS.items():
        for name, t in st.tools.items():
            schema = t.get("inputSchema") or {}
            props = schema.get("properties", {}) if isinstance(schema, dict) else {}
            action_key = next((k for k in ["action","verb","operation","op"] if k in props), None)
            kind_key = next((k for k in ["kind","resource","resources"] if k in props), None)
            actions = props.get(action_key, {}).get("enum", []) if action_key else []
            kinds = props.get(kind_key, {}).get("enum", []) if kind_key else []

            # Build a few generic NL intents based on enums (no domain assumptions)
            if actions and kinds:
                for a in actions[:5]:
                    for k in kinds[:5]:
                        intents.append({
                            "utterance": f"{a} {k}",
                            "endpoint": f"/{server}/tool/{_safe(name)}/{_safe(a)}/{_safe(k)}",
                        })
            elif actions:
                for a in actions[:8]:
                    intents.append({
                        "utterance": f"{a}",
                        "endpoint": f"/{server}/tool/{_safe(name)}/{_safe(a)}",
                    })
            elif kinds:
                for k in kinds[:8]:
                    intents.append({
                        "utterance": f"{k}",
                        "endpoint": f"/{server}/tool/{_safe(name)}/{_safe(k)}",
                    })
    return intents

def _augment_operation_docs(openapi_schema: Dict[str, Any]):
    paths = openapi_schema.get("paths", {})
    for path, methods in list(paths.items()):
        for verb, op in list(methods.items()):
            if verb.lower() not in ("post", "get"):
                continue

            # Ensure requestBody exists and is NOT required for POST
            if verb.lower() == "post":
                rb = op.get("requestBody") or {}
                rb["required"] = False
                op["requestBody"] = rb

            # Add common query param to tool POST/GET operations
            if "/tool/" in path:
                existing = {(p.get("name"), p.get("in")) for p in op.get("parameters", [])}
                params = op.setdefault("parameters", [])
                for qp in _COMMON_QUERY_PARAMS:
                    key = (qp["name"], qp["in"])
                    if key not in existing:
                        params.append(qp)

            # Strengthen description and usage
            guidance = (
                "CALLING RULES:\n"
                "• Prefer **GET with query** or **POST with JSON**. If you have no args, POST `{}`.\n"
                "• Granular paths like '/{server}/tool/{tool}/{action}[/{kind}]' imply fixed fields.\n"
                "• If you see 'Request body expected', retry with `{}` or use the GET form.\n"
            )
            op["description"] = ((op.get("description") or "") + "\n\n" + guidance).strip()

            # Mark priority (discovery endpoints low)
            if path.endswith("/tools/list") or path.endswith("/schema") or path.endswith("/example") or path.endswith("/help") or path.endswith("/try"):
                op.setdefault("x-priority", "low")
            elif "/tool/" in path:
                op.setdefault("x-priority", "high")
            else:
                op.setdefault("x-priority", "medium")

            # Natural-language example (generic)
            examples = op.setdefault("x-naturalExamples", [])
            if "/tool/" in path and verb.lower() == "post":
                examples.append({
                    "intent": "Call this tool with minimal body",
                    "body": {}
                })
            if "/tool/" in path and verb.lower() == "get":
                examples.append({
                    "intent": "Call this tool with query (no body)",
                    "call": f"GET {path}"
                })

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    # Global servers hint
    openapi_schema["servers"] = [{"url": "http://localhost:8080"}]

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
            "1) Read `/openapi.json` and `/SERVER/tools/list` (GET).",
            "2) For a user intent like “<action> <kind> …”, pick the granular path `/{SERVER}/tool/{TOOL}/{action}/{kind}`.",
            "3) If unsure which fields are required, call `/schema` or `/example` first.",
            "4) Prefer GET if you cannot send a body; otherwise POST `{}` or the minimal JSON.",
        ],
        "errorFix": [
            "If you see 'expected a request body', use GET or POST `{}`.",
            "If you see a schema error, inspect `/schema`, `/example`, or try a granular endpoint.",
        ],
    }

    # Intent map derived from discovered tools
    openapi_schema["x-intents"] = _derive_intents_from_tools()

    # Cookbook derived from intents (compact)
    openapi_schema["x-cookbook"] = [
        {"pattern": "<action> <kind>", "use": "/{SERVER}/tool/{TOOL}/{action}/{kind}", "notes": "Send `{}` if no other fields are needed."},
        {"pattern": "<action>", "use": "/{SERVER}/tool/{TOOL}/{action}", "notes": "Fixed `action` from path; add other fields as needed."},
        {"pattern": "<kind>", "use": "/{SERVER}/tool/{TOOL}/{kind}", "notes": "Fixed `kind` from path; add other fields as needed."},
    ]

    openapi_schema.setdefault("components", {}).setdefault("securitySchemes", {})
    openapi_schema["components"]["securitySchemes"]["XApiKey"] = {
        "type": "apiKey",
        "in": "header",
        "name": "X-Api-Key",
        "description": "Optional API key; set if the bridge was started with API_KEY.",
    }

    _augment_operation_docs(openapi_schema)
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
