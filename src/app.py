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
        # Useful for dev; can be overridden by env
        SERVERS_CFG = {"mcp": "http://localhost:8080/mcp"}

# =========================
# App
# =========================
app = FastAPI(
    title="MCP OpenAPI Bridge (Granular + Multi-Server)",
    version="0.7.0",
    description=(
        "A self-discovering OpenAPI façade for MCP servers. "
        "It generates detailed, example-rich endpoints for each MCP tool and adds granular paths when it detects "
        "`action` and/or `kind` enums in the tool schema."
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
        except httpx.ConnectError as e:
            raise HTTPException(
                status_code=503,
                detail={
                    "message": f"Failed to connect to MCP server at {self.rpc_url}",
                    "error": str(e),
                    "resolution": [
                        "Always send a JSON object request body (use `{}` if no arguments).",
                        "Verify the MCP server Service/Endpoint is reachable from this pod.",
                        f"Check that {self.rpc_url} is correct.",
                        "Ensure NetworkPolicies allow this pod to reach the MCP service on TCP/8080 (or your port).",
                    ],
                },
            )
        sid = resp.headers.get("mcp-session-id")
        if sid:
            self.session_id = sid
        data = resp.json()
        if "error" in data:
            raise HTTPException(
                status_code=502,
                detail={
                    "message": "MCP server returned an error",
                    "mcp_error": data["error"],
                    "resolution": [
                        "Always send a JSON object request body: direct `{...}` or wrapped `{ \"args\": { ... } }`.",
                        "Confirm the tool name and argument keys match the MCP tool schema.",
                        "Call `/SERVER/tools/list` to discover tools and `/SERVER/tool/TOOL/schema` for fields.",
                        "If RBAC/namespace related, try the admin server or pass the proper `namespace`.",
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
    """Map individual query params to schema-typed fields (best-effort)."""
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
                # accept comma-separated
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
    """
    Accept multiple shapes to be model-friendly:
    - None -> {}
    - {"args": {...}} -> {...}
    - {...} -> {...}
    - Also merge query (?args={...}) and individual query params that match schema fields.
    """
    base: Dict[str, Any] = {}
    if isinstance(body, dict):
        base = body.get("args", body) if isinstance(body.get("args"), dict) else body or {}
    # merge ?args={} first
    from_args = parse_args_query(args_q)
    base = merge(base, from_args)
    # merge individual ?key=value that match schema
    if request is not None:
        qp = {k: v for k, v in request.query_params.items() if k not in ("args",)}
        base = merge(base, coerce_query_params(schema or {"type":"object"}, qp))
    return base or {}

def make_request_body(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Advertise both direct-body and {args:{}} body forms; mark body as REQUIRED
    so planners try to send `{}`. We ALSO support GET/Query fallbacks in code.
    """
    direct = schema or {"type": "object"}
    wrapped = {"type": "object", "properties": {"args": direct}, "required": ["args"]}
    return {
        "required": True,
        "content": {
            "application/json": {
                "schema": {"oneOf": [direct, wrapped]},
                "examples": {
                    "empty": {"summary": "No arguments", "value": {}},
                    "direct": {"summary": "Direct body", "value": example_from_schema(direct) or {}},
                    "wrapped": {"summary": "Wrapped in args", "value": {"args": example_from_schema(direct) or {}}},
                },
            }
        },
    }

def schema_has_required(schema: Dict[str, Any]) -> bool:
    return bool(isinstance(schema, dict) and schema.get("required"))

# Track mounted routes to avoid duplicates on /refresh
_registered: set[str] = set()

# =========================
# Route builders
# =========================
def add_generic_routes(router: APIRouter, server: str):
    st = SERVERS[server]

    @router.get(f"/{server}/healthz", summary=f"{server} health")
    async def healthz_server():
        try:
            await st.ensure_initialized()
            return {"ok": True, "server": server, "tools": list(st.tools.keys())}
        except Exception as e:
            return Response(content=f"not ready: {e}", status_code=503)

    @router.get(f"/{server}/usage", summary=f"{server} model usage guide", tags=["usage"])
    async def usage_server():
        return {
            "howToUse": [
                "Preferred: POST with a JSON body (use `{}` if no args).",
                "Fallbacks (for clients that cannot send a body):",
                "  • Use GET on '/try' endpoints for zero-argument calls.",
                "  • Put JSON in query: '?args={...}', or individual '?key=value' params.",
                "You may POST either a direct JSON body `{...}` or wrapped `{ \"args\": { ... } }`.",
                "Prefer granular endpoints like `/{SERVER}/tool/{TOOL}/get/pods` when available.",
                "Discover tools with `POST /{SERVER}/tools/list` (send `{}` even if empty).",
                "Inspect fields with `GET /{SERVER}/tool/{TOOL}/schema` and examples with `/example`.",
            ],
            "curlExamples": [
                f"curl -X POST http://localhost:8080/{server}/tool/TOOL_NAME -H 'Content-Type: application/json' -d '{{}}'",
                f"curl -X POST http://localhost:8080/{server}/tool/TOOL_NAME/get/pods -H 'Content-Type: application/json' -d '{{}}'",
                f"curl 'http://localhost:8080/{server}/tool/TOOL_NAME/try'",
                f"curl 'http://localhost:8080/{server}/tool/TOOL_NAME?args=%7B%22namespace%22%3A%22vault%22%7D'",
            ],
            "tip": "If you see 'expected a request body', resend with `{}` OR use GET '/try' or '?args={...}'.",
        }

    @router.post(
        f"/{server}/tools/list",
        summary=f"{server} raw MCP tool listing",
        dependencies=[Depends(require_api_key)],
        openapi_extra={
            "x-instructions": "Lists tools on this MCP server. Send `{}` if you have no arguments.",
            "requestBody": make_request_body({"type": "object"}),
        },
    )
    async def tools_list(_body: Optional[Dict[str, Any]] = Body(default=None, embed=False)):
        await st.refresh_tools()
        return {"tools": list(st.tools.values())}

    @router.post(
        f"/{server}/tools/call",
        summary=f"{server} raw MCP tool call",
        dependencies=[Depends(require_api_key)],
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
        # Body + query normalization
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
                "Fallbacks: GET '/try' for zero-arg, or pass `?args={...}` / `?key=value`.",
                "You may send a direct body or `{ \"args\": { ... } }`.",
                f"Fetch an example body from `/{server}/tool/{pname}/example`.",
                f"View schema at `/{server}/tool/{pname}/schema`.",
                "If granular endpoints exist (action/kind), prefer those—they pre-fill some fields.",
            ],
            "curlExamples": [
                f"curl -X POST http://localhost:8080/{server}/tool/{pname} -H 'Content-Type: application/json' -d '{{}}'",
                f"curl 'http://localhost:8080/{server}/tool/{pname}/try'",
                f"curl 'http://localhost:8080/{server}/tool/{pname}?args=%7B%22namespace%22%3A%22vault%22%7D'",
            ],
        }

    async def try_handler(request: Request):
        # call with {}
        return await SERVERS[server].rpc("tools/call", {"name": tname, "arguments": {}})

    router.add_api_route(
        path=f"/{server}/tool/{pname}/schema",
        endpoint=schema_handler,
        methods=["GET"],
        name=f"{server}:{tname} schema",
        summary=f"{server} → {tname} schema",
        operation_id=f"{server}_{tname}_schema",
        tags=[f"{server}:{tname}", "schema"],
    )
    router.add_api_route(
        path=f"/{server}/tool/{pname}/example",
        endpoint=example_handler,
        methods=["GET"],
        name=f"{server}:{tname} example",
        summary=f"{server} → {tname} example",
        operation_id=f"{server}_{tname}_example",
        tags=[f"{server}:{tname}", "example"],
    )
    router.add_api_route(
        path=f"/{server}/tool/{pname}/help",
        endpoint=help_handler,
        methods=["GET"],
        name=f"{server}:{tname} help",
        summary=f"{server} → {tname} help",
        operation_id=f"{server}_{tname}_help",
        tags=[f"{server}:{tname}", "help"],
    )
    router.add_api_route(
        path=f"/{server}/tool/{pname}/try",
        endpoint=try_handler,
        methods=["GET"],
        name=f"{server}:{tname} try",
        summary=f"{server} → {tname} zero-arg try",
        description="Convenience GET that calls this tool with an empty argument object `{}`.",
        operation_id=f"{server}_{tname}_try",
        tags=[f"{server}:{tname}", "try"],
    )


def add_tool_route(router: APIRouter, server: str, tool: Dict[str, Any]):
    st = SERVERS[server]
    tname = tool["name"]
    pname = _safe(tname)
    schema = tool.get("inputSchema") or {"type": "object"}
    desc = tool.get("description") or f"Call MCP tool {tname} on server {server}. Prefer granular endpoints when possible."
    state = st; toolname = tname

    async def handler(
        request: Request,
        body: Optional[Dict[str, Any]] = Body(default=None, embed=False),
        args: Optional[str] = Query(default=None, description="JSON-encoded args (query fallback)"),
    ) -> Any:
        call_args = normalize_body(body, schema, args, request)
        await state.ensure_initialized()
        return await state.rpc("tools/call", {"name": toolname, "arguments": call_args})

    route = APIRoute(
        path=f"/{server}/tool/{pname}",
        endpoint=handler,
        methods=["POST"],
        name=f"{server}:{tname}",
        summary=desc,
        operation_id=f"{server}_{tname}",
        tags=[f"{server}:{tname}"],
        dependencies=[Depends(require_api_key)],
        openapi_extra={
            "x-instructions": (
                "Preferred: POST a JSON body. If no args, send `{}`. "
                "Fallbacks: pass args in `?args={...}` or as individual `?key=value`. "
                "Granular endpoints like `/{SERVER}/tool/{TOOL}/{action}/{kind}` are preferred when available."
            ),
            "requestBody": make_request_body(schema),
        },
    )
    router.routes.append(route)


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
        # strip fixed
        props = reduced_schema.setdefault("properties", {})
        req = set(reduced_schema.get("required", []))
        props.pop("action", None)
        req.discard("action")
        reduced_schema["required"] = list(req)
        op_id = f"{server}_{tname}_{_safe(action)}"

        state = st; toolname = tname; fixed_local = dict(fixed)
        path = f"/{server}/tool/{pname}/{_safe(action)}"

        async def handler_action(
            request: Request,
            body: Optional[Dict[str, Any]] = Body(default=None, embed=False),
            args: Optional[str] = Query(default=None, description="JSON-encoded args (query fallback)"),
        ) -> Any:
            call_args = normalize_body(body, reduced_schema, args, request)
            call_args.update(fixed_local)
            await state.ensure_initialized()
            return await state.rpc("tools/call", {"name": toolname, "arguments": call_args})

        ex = example_from_schema(reduced_schema) or {}
        route = APIRoute(
            path=path,
            endpoint=handler_action,
            methods=["POST"],
            name=f"{server}:{tname}:{action}",
            summary=f"{server} → {tname} → {action}",
            description=(f"Fixes `action: \"{action}\"`. Provide only the remaining fields in the body (if any)."),
            operation_id=op_id,
            tags=[f"{server}:{tname}", action],
            dependencies=[Depends(require_api_key)],
            openapi_extra={
                "x-instructions": (
                    "Preferred: POST with a JSON body. If no remaining fields, send `{}`. "
                    "Fallbacks: `?args={...}` or individual `?key=value` matching schema fields."
                ),
                "requestBody": make_request_body(reduced_schema),
            },
        )
        router.routes.append(route)

        # GET /try if no required fields remain
        if not schema_has_required(reduced_schema):
            async def handler_action_try():
                return await st.rpc("tools/call", {"name": tname, "arguments": dict(fixed_local)})
            router.add_api_route(
                path=f"{path}/try",
                endpoint=handler_action_try,
                methods=["GET"],
                name=f"{server}:{tname}:{action}:try",
                summary=f"{server} → {tname} → {action} zero-arg try",
                description="Convenience GET that calls this action with `{}` (fixed fields implied).",
                operation_id=f"{op_id}_try",
                tags=[f"{server}:{tname}", action, "try"],
            )

    # Action + kind (or kind only)
    for action in actions or ["_"]:
        for kind in kinds or []:
            if action == "_":
                fixed = {"kind": kind}
                suffix = _safe(kind)
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
            for k in ("action","kind"):
                props.pop(k, None); req.discard(k)
            reduced_schema["required"] = list(req)

            state = st; toolname = tname; fixed_local = dict(fixed)
            path = f"/{server}/tool/{pname}/{suffix}"

            async def handler_action_kind(
                request: Request,
                body: Optional[Dict[str, Any]] = Body(default=None, embed=False),
                args: Optional[str] = Query(default=None, description="JSON-encoded args (query fallback)"),
            ) -> Any:
                call_args = normalize_body(body, reduced_schema, args, request)
                call_args.update(fixed_local)
                await state.ensure_initialized()
                return await state.rpc("tools/call", {"name": toolname, "arguments": call_args})

            ex = example_from_schema(reduced_schema) or {}
            route = APIRoute(
                path=path,
                endpoint=handler_action_kind,
                methods=["POST"],
                name=f"{server}:{tname}:{summary}",
                summary=summary,
                description=f"Convenience endpoint. {desc} Provide only remaining fields (if any).",
                operation_id=op_id,
                tags=tags,
                dependencies=[Depends(require_api_key)],
                openapi_extra={
                    "x-instructions": (
                        "Preferred: POST with a JSON body. If no remaining fields, send `{}`. "
                        "Fallbacks: `?args={...}` or individual `?key=value`."
                    ),
                    "requestBody": make_request_body(reduced_schema),
                },
            )
            router.routes.append(route)

            # GET /try if no required fields remain
            if not schema_has_required(reduced_schema):
                async def handler_action_kind_try():
                    return await st.rpc("tools/call", {"name": tname, "arguments": dict(fixed_local)})
                router.add_api_route(
                    path=f"{path}/try",
                    endpoint=handler_action_kind_try,
                    methods=["GET"],
                    name=f"{server}:{tname}:{summary}:try",
                    summary=f"{summary} zero-arg try",
                    description="Convenience GET that calls this action/kind with `{}` (fixed fields implied).",
                    operation_id=f"{op_id}_try",
                    tags=tags + ["try"],
                )

# =========================
# Startup / refresh
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

@app.on_event("startup")
async def startup():
    deadline = time.time() + STARTUP_TIMEOUT
    for name, st in SERVERS.items():
        last_err: Optional[Exception] = None
        while time.time() < deadline:
            try:
                await st.refresh_tools()
                install_routes_for_server(name)
                last_err = None
                break
            except Exception as e:
                last_err = e
                await asyncio.sleep(1)
        if last_err:
            raise RuntimeError(f"[{name}] startup failed: {last_err}")

@app.get("/healthz", tags=["health"])
async def healthz():
    try:
        for st in SERVERS.values():
            await st.ensure_initialized()
        return {"ok": True, "servers": list(SERVERS.keys())}
    except Exception as e:
        return Response(status_code=503, content=f"not ready: {e}")

@app.get("/servers", tags=["info"])
async def servers():
    return {"servers": {k: v.rpc_url for k, v in SERVERS.items()}}

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
            "Preferred: POST a JSON object body (use `{}` if no arguments).",
            "Fallbacks: GET '/try' for zero-arg, or pass `?args={...}` / individual `?key=value`.",
            "Discover tools via `/{SERVER}/tools/list` and inspect fields with `/tool/{TOOL}/schema`.",
            "Use `/tool/{TOOL}/example` or granular endpoints.",
        ]
        if exc.status_code == 401:
            tips.insert(0, "Include the `X-Api-Key` header if the bridge was configured with API_KEY.")
        if exc.status_code == 422:
            tips.insert(0, "Ensure the body is a JSON object (or use GET '/try' / `?args={...}` as a fallback).")
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
                "Fallbacks: GET '/try' or pass `?args={...}` / individual `?key=value`.",
                "Retry with a simpler body; check bridge pod logs for stack traces.",
                "Verify MCP server availability and NetworkPolicy.",
            ],
        },
    )

# =========================
# Custom OpenAPI: global, explicit model instructions
# =========================
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    openapi_schema["x-model-instructions"] = {
        "callDiscipline": [
            "Preferred transport is POST with a JSON object body. If you have no arguments, send `{}`.",
            "If you cannot send a body, use GET '/try' (zero-arg) or pass arguments in the query string:",
            "  • `?args={...}` (JSON-encoded), or",
            "  • individual `?key=value` matching schema field names.",
        ],
        "bodyShapes": [
            "Direct body: `{ ... }`",
            "Wrapped body: `{ \"args\": { ... } }`",
        ],
        "discovery": [
            "List tools: `/{SERVER}/tools/list` (send `{}` even if empty).",
            "Per-tool schema: `/{SERVER}/tool/{TOOL}/schema`.",
            "Per-tool example: `/{SERVER}/tool/{TOOL}/example`.",
            "Per-tool help: `/{SERVER}/tool/{TOOL}/help`.",
            "Zero-argument test: `/{SERVER}/tool/{TOOL}/try` and for granular paths `.../try` when available.",
        ],
        "preference": "Prefer granular endpoints like `/{SERVER}/tool/{TOOL}/{action}/{kind}` when available.",
        "typicalFlow": [
            "1) `GET /{SERVER}/usage`.",
            "2) `POST /{SERVER}/tools/list` with `{}`.",
            "3) `GET /{SERVER}/tool/{TOOL}/schema` and `/example` (understand fields).",
            "4) `POST /{SERVER}/tool/{TOOL}` (or granular path) with `{}` or the minimal example.",
        ],
        "errorFix": [
            "If you see 'expected a request body', resend with `{}` OR use GET '/try' OR `?args={...}`.",
            "If you see a schema error, consult `/schema`, `/example`, or use a granular endpoint.",
        ],
    }
    openapi_schema.setdefault("components", {}).setdefault("securitySchemes", {})
    openapi_schema["components"]["securitySchemes"]["XApiKey"] = {
        "type": "apiKey",
        "in": "header",
        "name": "X-Api-Key",
        "description": "Optional API key; set if the bridge was started with API_KEY.",
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
