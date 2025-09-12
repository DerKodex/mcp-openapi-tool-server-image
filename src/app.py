import os, time, asyncio, re, json
from typing import Any, Dict, Optional, List, Tuple
from fastapi import FastAPI, Body, Response, HTTPException, Depends, Header
from fastapi.routing import APIRoute
from fastapi import APIRouter
from fastapi.middleware.cors import CORSMiddleware
import httpx

# -------------------------
# Config
# -------------------------
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

# -------------------------
# App
# -------------------------
app = FastAPI(
    title="MCP OpenAPI Bridge (Granular + Multi-Server)",
    version="0.4.0",
    description="Self-discovering OpenAPI façade for MCP servers with granular, example-rich endpoints."
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

# -------------------------
# HTTP client
# -------------------------
_client: Optional[httpx.AsyncClient] = None
async def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
    return _client

# -------------------------
# Per-server state
# -------------------------
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
        resp = await client.post(self.rpc_url, json=payload, headers=headers)
        resp.raise_for_status()
        sid = resp.headers.get("mcp-session-id")
        if sid:
            self.session_id = sid
        data = resp.json()
        if "error" in data:
            raise HTTPException(status_code=502, detail={"mcp_error": data["error"]})
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

# -------------------------
# Helpers (granular)
# -------------------------
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

# Track what we’ve already mounted to avoid duplicates on /refresh
_registered: set[str] = set()

def add_generic_routes(router: APIRouter, server: str):
    st = SERVERS[server]

    @router.get(f"/{server}/healthz", summary=f"{server} health")
    async def healthz_server():
        try:
            await st.ensure_initialized()
            return {"ok": True, "server": server, "tools": list(st.tools.keys())}
        except Exception as e:
            return Response(content=f"not ready: {e}", status_code=503)

    @router.post(f"/{server}/tools/list", summary=f"{server} raw MCP tool listing", dependencies=[Depends(require_api_key)])
    async def tools_list():
        await st.refresh_tools()
        return {"tools": list(st.tools.values())}

    @router.post(f"/{server}/tools/call", summary=f"{server} raw MCP tool call", dependencies=[Depends(require_api_key)])
    async def tools_call(name: str = Body(..., embed=True),
                         arguments: Optional[Dict[str, Any]] = Body(default=None, embed=True)):
        await st.ensure_initialized()
        return await st.rpc("tools/call", {"name": name, "arguments": arguments or {}})

def add_schema_helpers(router: APIRouter, server: str, tool: Dict[str, Any]):
    st = SERVERS[server]
    tname = tool["name"]
    pname = _safe(tname)

    async def schema_handler():
        return tool.get("inputSchema") or {"type": "object"}

    async def example_handler():
        ex = example_from_schema(tool.get("inputSchema") or {"type": "object"})
        return {"name": tname, "arguments": ex or {}}

    router.add_api_route(
        path=f"/{server}/tool/{pname}/schema",
        endpoint=schema_handler,
        methods=["GET"],
        name=f"{server}:{tname} schema",
        summary=f"{server} → {tname} schema",
        operation_id=f"{server}_{tname}_schema",
        tags=[f"{server}:{tname}"],
    )
    router.add_api_route(
        path=f"/{server}/tool/{pname}/example",
        endpoint=example_handler,
        methods=["GET"],
        name=f"{server}:{tname} example",
        summary=f"{server} → {tname} example",
        operation_id=f"{server}_{tname}_example",
        tags=[f"{server}:{tname}"],
    )

def add_tool_route(router: APIRouter, server: str, tool: Dict[str, Any]):
    st = SERVERS[server]
    tname = tool["name"]
    pname = _safe(tname)
    schema = tool.get("inputSchema") or {"type": "object"}
    desc = (tool.get("description") or f"Call MCP tool {tname} on server {server}. "
           f"Prefer granular endpoints below when they match your need.")

    async def handler(body: Dict[str, Any] = Body(default={}, embed=False), _tool=tname, _st=st) -> Any:
        await _st.ensure_initialized()
        return await _st.rpc("tools/call", {"name": _tool, "arguments": body or {}})

    route = APIRoute(
        path=f"/{server}/tool/{pname}",
        endpoint=handler,
        methods=["POST"],
        name=f"{server}:{tname}",
        summary=desc,
        operation_id=f"{server}_{tname}",  # visible as function name in some planners
        tags=[f"{server}:{tname}"],
        dependencies=[Depends(require_api_key)],
        openapi_extra={
            "x-instructions": (
                "Generic form of the tool. "
                "Use a more specific /{server}/tool/{tool}/{action}[/{kind}] endpoint if available. "
                "Send JSON matching the schema; omit fields fixed by granular endpoints."
            ),
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": schema,
                        "examples": {
                            "typical": {
                                "summary": "Typical call",
                                "value": example_from_schema(schema) or {}
                            }
                        }
                    }
                }
            }
        },
    )
    router.routes.append(route)

def add_granular_routes(router: APIRouter, server: str, tool: Dict[str, Any]):
    st = SERVERS[server]
    schema = tool.get("inputSchema") or {"type": "object"}
    actions, kinds = iter_action_kind(schema)
    if not actions and not kinds:
        return

    tname = tool["name"]
    pname = _safe(tname)

    # Action-only: /{server}/tool/{tool}/{action}
    for action in actions or []:
        fixed = {"action": action}
        reduced_schema = strip_fixed_fields(schema, fixed)
        op_id = f"{server}_{tname}_{_safe(action)}"

        async def handler_action(
            body: Dict[str, Any] = Body(default={}, embed=False),
            _fixed=fixed, _tool=tname, _st=st
        ) -> Any:
            args = body or {}
            args.update(_fixed)
            await _st.ensure_initialized()
            return await _st.rpc("tools/call", {"name": _tool, "arguments": args})

        ex = example_from_schema(reduced_schema) or {}
        path = f"/{server}/tool/{pname}/{_safe(action)}"
        route = APIRoute(
            path=path,
            endpoint=handler_action,
            methods=["POST"],
            name=f"{server}:{tname}:{action}",
            summary=f"{server} → {tname} → {action}",
            description=(f"Fixes `action: \"{action}\"`. Provide only the remaining fields in the body."),
            operation_id=op_id,
            tags=[f"{server}:{tname}", action],
            dependencies=[Depends(require_api_key)],
            openapi_extra={
                "x-instructions": (
                    f"Use this when the desired action is '{action}'. "
                    "Do not include the fixed field `action` in the request body; it is implied by the path."
                ),
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": reduced_schema,
                            "examples": {
                                "typical": {"summary": "Typical", "value": ex}
                            }
                        }
                    }
                }
            },
        )
        router.routes.append(route)

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

            async def handler_action_kind(
                body: Dict[str, Any] = Body(default={}, embed=False),
                _fixed=fixed, _tool=tname, _st=st
            ) -> Any:
                args = body or {}
                args.update(_fixed)
                await _st.ensure_initialized()
                return await _st.rpc("tools/call", {"name": _tool, "arguments": args})

            ex = example_from_schema(reduced_schema) or {}
            route = APIRoute(
                path=f"/{server}/tool/{pname}/{suffix}",
                endpoint=handler_action_kind,
                methods=["POST"],
                name=f"{server}:{tname}:{summary}",
                summary=summary,
                description=f"Convenience endpoint. {desc} Provide only remaining fields.",
                operation_id=op_id,
                tags=tags,
                dependencies=[Depends(require_api_key)],
                openapi_extra={
                    "x-instructions": (
                        "Prefer this endpoint when both action and resource kind are known. "
                        "Omit any fixed fields; they are implied by the path."
                    ),
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": reduced_schema,
                                "examples": {
                                    "typical": {"summary": "Typical", "value": ex}
                                }
                            }
                        }
                    }
                },
            )
            router.routes.append(route)

# -------------------------
# Startup / refresh
# -------------------------
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
