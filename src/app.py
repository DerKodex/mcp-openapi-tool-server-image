import os, time, asyncio, re, json
from typing import Any, Dict, Optional, List, Tuple, Union
from fastapi import FastAPI, Body, Response, HTTPException, Depends, Header, Request
from fastapi.routing import APIRoute
from fastapi import APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
    version="0.5.0",
    description=(
        "A self-discovering OpenAPI façade for MCP servers. "
        "It generates detailed, example-rich endpoints for each MCP tool and adds granular paths when it detects "
        "`action` and/or `kind` enums in the tool schema.\n\n"
        "## How models should call these endpoints\n"
        "- You may POST either `{...arguments...}` **or** `{ \"args\": { ...arguments... } }`.\n"
        "- If a tool needs **no arguments**, you can POST an **empty body** `{}` (or omit the body entirely).\n"
        "- Prefer granular routes like `/SERVER/tool/TOOL/get/pods` when they exist; they fix fields for you.\n"
        "- Use `/SERVER/tool/TOOL/example` to fetch a minimal example object and `/SERVER/tool/TOOL/schema` to see the JSON Schema.\n"
        "- If you get an error, read the `resolution` hints included in the response."
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
                        "Confirm the tool name and argument keys match the MCP tool schema.",
                        "Try calling /SERVER/tools/list to discover available tools.",
                        "If RBAC- or namespace-related, try the admin server or adjust tool arguments (e.g., namespace).",
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
        # Expect either {"tools":[...]} or a bare list
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
                obj[k] = v["default"]
                continue
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

    if t == "string":
        return schema.get("default") or "value"
    if t in ("integer", "number"):
        return schema.get("default") or 1
    if t == "boolean":
        return schema.get("default") or True

    return schema.get("default") or None


def strip_fixed_fields(schema: Dict[str, Any], fixed: Dict[str, Any]) -> Dict[str, Any]:
    if not schema or schema.get("type") != "object":
        return schema
    schema = json.loads(json.dumps(schema))
    props = schema.setdefault("properties", {})
    req = set(schema.get("required", []))
    for k in list(fixed.keys()):
        props.pop(k, None)
        if k in req:
            req.remove(k)
    schema["required"] = list(req)
    return schema


def iter_action_kind(schema: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    if not isinstance(schema, Dict):
        return ([], [])
    props = schema.get("properties", {})
    action_key = next((k for k in ["action", "verb", "operation", "op"] if k in props), None)
    actions = props.get(action_key, {}).get("enum", []) if action_key else []
    kind_key = next((k for k in ["kind", "resource", "resources"] if k in props), None)
    kinds = props.get(kind_key, {}).get("enum", []) if kind_key else []
    return (actions, kinds)


def describe_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Produce a verbose, model-friendly description of a JSON schema: required keys, optional keys, enums, and notes."""
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
    # quick hints
    hints = []
    if "namespace" in props:
        hints.append("If the tool supports Kubernetes, include `namespace` for namespace-scoped queries.")
    if "kind" in props and props["kind"].get("enum"):
        hints.append(f"`kind` accepts one of: {props['kind']['enum']}")
    if "action" in props and props["action"].get("enum"):
        hints.append(f"`action` accepts one of: {props['action']['enum']}")
    if not required:
        hints.append("This tool can be called with **no arguments**; send `{}` or omit the body.")
    desc["usageHints"] = hints
    return desc


def normalize_body(body: Optional[Union[Dict[str, Any], list, str, int, float, bool, None]]) -> Dict[str, Any]:
    """
    Accept multiple shapes to be model-friendly:
    - None -> {}
    - {"args": {...}} -> {...}
    - {...} -> {...}
    Any other type -> {}
    """
    if body is None:
        return {}
    if isinstance(body, dict):
        if "args" in body and isinstance(body["args"], dict):
            return body["args"]
        return body
    # If a model accidentally sends a primitive/array, treat as no args.
    return {}


def make_request_body(schema: Dict[str, Any], required_default: bool) -> Dict[str, Any]:
    """
    Advertise both direct-body and {args:{}} body forms, and make it optional so tools can be called with no args.
    """
    # If schema is empty object, still expose oneOf to teach the "args" form.
    direct = schema or {"type": "object"}
    wrapped = {"type": "object", "properties": {"args": direct}, "required": ["args"]}
    return {
        "required": required_default and bool(direct.get("required")),  # still false for most, so body can be omitted
        "content": {
            "application/json": {
                "schema": {"oneOf": [direct, wrapped]},
                "examples": {
                    "direct": {"summary": "Direct body", "value": example_from_schema(direct) or {}},
                    "wrapped": {"summary": "Wrapped in args", "value": {"args": example_from_schema(direct) or {}}},
                    "empty": {"summary": "No arguments", "value": {}},
                },
            }
        },
    }


# Track what we’ve already mounted to avoid duplicates on /refresh
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

    @router.get(
        f"/{server}/usage",
        summary=f"{server} model usage guide",
        tags=["usage"],
    )
    async def usage_server():
        return {
            "howToUse": [
                "POST either a direct JSON body `{...}` or a wrapped body `{ \"args\": { ... } }`.",
                "If no arguments are needed, POST `{}` or omit the body.",
                "Prefer granular endpoints like `/SERVER/tool/TOOL/get/pods` when available.",
                "Call `/SERVER/tools/list` to discover tools (requires API key if enabled).",
                "Use `/SERVER/tool/TOOL/schema` and `/SERVER/tool/TOOL/example` to learn arguments.",
            ],
            "tip": "If you receive an 'args required' type error from a client, switch to the wrapped body form: `{ \"args\": { ... } }`.",
        }

    @router.post(
        f"/{server}/tools/list",
        summary=f"{server} raw MCP tool listing",
        dependencies=[Depends(require_api_key)],
    )
    async def tools_list():
        await st.refresh_tools()
        return {"tools": list(st.tools.values())}

    @router.post(
        f"/{server}/tools/call",
        summary=f"{server} raw MCP tool call",
        dependencies=[Depends(require_api_key)],
        openapi_extra={
            "x-instructions": (
                "You can pass arguments directly as the request body, or wrapped inside an `args` object. "
                "Both `{...}` and `{ \"args\": { ... } }` are accepted. You may also send `{}` for no-arg tools."
            ),
            "requestBody": make_request_body({"type": "object"}, required_default=False),
        },
    )
    async def tools_call(
        body: Optional[Dict[str, Any]] = Body(default=None, embed=False),
        name: Optional[str] = Header(default=None, description="Optional tool name header alternative"),
        tool: Optional[str] = Header(default=None, description="Alternative header for tool name"),
    ):
        args = normalize_body(body)
        tool_name = args.pop("name", None) or name or tool  # allow passing name in args or headers
        if not tool_name:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Tool name not provided.",
                    "resolution": [
                        "Pass the tool name in the JSON body as `{\"name\":\"<tool>\", ...}` or",
                        "Use the `name:` or `tool:` HTTP header to specify the tool name.",
                        "Or call a specific tool endpoint like `/SERVER/tool/TOOL`.",
                    ],
                },
            )
        await st.ensure_initialized()
        return await st.rpc("tools/call", {"name": tool_name, "arguments": args})


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
                f"POST to `/{server}/tool/{_safe(tname)}` with either direct body or `{{\"args\":{{...}}}}`.",
                f"Fetch an example body from `/{server}/tool/{_safe(tname)}/example`.",
                f"View schema at `/{server}/tool/{_safe(tname)}/schema`.",
                "If granular endpoints exist (action/kind), prefer those—they pre-fill some fields.",
            ],
        }

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


def add_tool_route(router: APIRouter, server: str, tool: Dict[str, Any]):
    st = SERVERS[server]
    tname = tool["name"]
    pname = _safe(tname)
    schema = tool.get("inputSchema") or {"type": "object"}
    desc = (
        tool.get("description")
        or f"Call MCP tool {tname} on server {server}. Prefer granular endpoints below when they match your need."
    )

    state = st
    toolname = tname

    async def handler(body: Optional[Dict[str, Any]] = Body(default=None, embed=False)) -> Any:
        args = normalize_body(body)
        await state.ensure_initialized()
        return await state.rpc("tools/call", {"name": toolname, "arguments": args})

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
                "You may POST either a direct JSON body or `{ \"args\": { ... } }`. "
                "If no arguments are needed, POST `{}` or omit the body. "
                "Use granular endpoints like `/SERVER/tool/TOOL/{action}/{kind}` when available."
            ),
            "requestBody": make_request_body(schema, required_default=False),
        },
    )
    router.routes.append(route)


def add_granular_routes(router: APIRouter, server: str, tool: Dict[str, Any]):
    st = SERVERS[server]
    schema = tool.get("inputSchema") or {"type": "object"}
    actions, kinds = iter_action_kind(schema)
    if not actions and not kinds:
        # still add a /help for completeness
        return

    tname = tool["name"]
    pname = _safe(tname)

    # Action-only: /{server}/tool/{tool}/{action}
    for action in actions or []:
        fixed = {"action": action}
        reduced_schema = strip_fixed_fields(schema, fixed)
        op_id = f"{server}_{tname}_{_safe(action)}"

        state = st
        toolname = tname
        fixed_local = dict(fixed)

        async def handler_action(body: Optional[Dict[str, Any]] = Body(default=None, embed=False)) -> Any:
            args = normalize_body(body)
            args.update(fixed_local)
            await state.ensure_initialized()
            return await state.rpc("tools/call", {"name": toolname, "arguments": args})

        ex = example_from_schema(reduced_schema) or {}
        path = f"/{server}/tool/{pname}/{_safe(action)}"
        route = APIRoute(
            path=path,
            endpoint=handler_action,
            methods=["POST"],
            name=f"{server}:{tname}:{action}",
            summary=f"{server} → {tname} → {action}",
            description=(f"Fixes `action: \"{action}\"`. Provide only the remaining fields in the body, if any."),
            operation_id=op_id,
            tags=[f"{server}:{tname}", action],
            dependencies=[Depends(require_api_key)],
            openapi_extra={
                "x-instructions": (
                    f"Use this when the desired action is '{action}'. "
                    "You may POST `{}` for action-only calls or add remaining fields. "
                    "Direct body or `{ \"args\": { ... } }` are both accepted."
                ),
                "requestBody": make_request_body(reduced_schema, required_default=False),
            },
        )
        router.routes.append(route)

        # Action help
        async def handler_action_help():
            return {
                "tool": tname,
                "server": server,
                "action": action,
                "fixed": fixed_local,
                "remainingSchema": describe_schema(reduced_schema),
                "exampleBody": ex,
                "howToUse": [
                    f"POST to `{path}` with optional remaining fields.",
                    "You can send `{}` if no additional fields are required.",
                    "Body can be direct or wrapped as `{ \"args\": { ... } }`.",
                ],
            }

        router.add_api_route(
            path=f"{path}/help",
            endpoint=handler_action_help,
            methods=["GET"],
            name=f"{server}:{tname}:{action}:help",
            summary=f"{server} → {tname} → {action} help",
            operation_id=f"{op_id}_help",
            tags=[f"{server}:{tname}", action, "help"],
        )

    # Action + kind (or kind only): /{server}/tool/{tool}/{action}/{kind}
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

            reduced_schema = strip_fixed_fields(schema, fixed)

            state = st
            toolname = tname
            fixed_local = dict(fixed)
            path = f"/{server}/tool/{pname}/{suffix}"

            async def handler_action_kind(body: Optional[Dict[str, Any]] = Body(default=None, embed=False)) -> Any:
                args = normalize_body(body)
                args.update(fixed_local)
                await state.ensure_initialized()
                return await state.rpc("tools/call", {"name": toolname, "arguments": args})

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
                        "Prefer this endpoint when both action and resource kind are known. "
                        "You can POST `{}` for no-arg calls, or include the remaining fields. "
                        "Direct body or `{ \"args\": { ... } }` are both accepted."
                    ),
                    "requestBody": make_request_body(reduced_schema, required_default=False),
                },
            )
            router.routes.append(route)

            # Help for action/kind
            async def handler_action_kind_help():
                return {
                    "tool": tname,
                    "server": server,
                    "fixed": fixed_local,
                    "remainingSchema": describe_schema(reduced_schema),
                    "exampleBody": ex,
                    "howToUse": [
                        f"POST to `{path}` with optional remaining fields.",
                        "You can send `{}` if no additional fields are required.",
                        "Body can be direct or wrapped as `{ \"args\": { ... } }`.",
                    ],
                }

            router.add_api_route(
                path=f"{path}/help",
                endpoint=handler_action_kind_help,
                methods=["GET"],
                name=f"{server}:{tname}:{summary}:help",
                summary=f"{summary} help",
                operation_id=f"{op_id}_help",
                tags=tags + ["help"],
            )


# =========================
# Startup / refresh
# =========================
def install_routes_for_server(server: str):
    """Idempotently mount routes for a server & its tools."""
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
    # If the detail is already a dict with resolution tips, just return it.
    if isinstance(exc.detail, dict):
        payload = exc.detail
    else:
        payload = {"message": str(exc.detail)}
    # Add generic guidance if missing
    if "resolution" not in payload:
        tips = [
            "If this is an argument error, POST either a direct body `{...}` or `{ \"args\": { ... } }`.",
            "Call `/SERVER/tools/list` and `/SERVER/tool/TOOL/schema` to inspect available tools and fields.",
            "Use `/SERVER/tool/TOOL/example` to get a minimal valid body.",
            "Prefer granular endpoints like `/SERVER/tool/TOOL/get/pods` when available.",
        ]
        # Hint for auth
        if exc.status_code == 401:
            tips.insert(0, "Include the `X-Api-Key` header if the bridge was configured with API_KEY.")
        if exc.status_code == 422:
            tips.insert(0, "Ensure the request body is a JSON object (or `{ \"args\": { ... } }`).")
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
                "Retry the request with a simpler body or `{}`.",
                "Check bridge pod logs for stack traces.",
                "Verify MCP server availability and NetworkPolicy.",
            ],
        },
    )
