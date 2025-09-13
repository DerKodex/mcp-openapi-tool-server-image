# app.py
import os, time, asyncio, re, json, hashlib
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
API_KEY            = os.environ.get("API_KEY", "").strip()
ALLOW_ORIGINS      = os.environ.get("CORS_ALLOW_ORIGINS", "*")
REQUEST_TIMEOUT    = int(os.environ.get("REQUEST_TIMEOUT", "30"))
REFRESH_INTERVAL   = int(os.environ.get("REFRESH_INTERVAL", "10"))  # seconds
DISCOVERY_PUBLIC   = os.environ.get("DISCOVERY_PUBLIC", "true").lower() in ("1","true","t","yes","y","on")
PUBLIC_BASE_URL    = os.environ.get("PUBLIC_BASE_URL", "http://localhost:8080")

# Optional: capture small output samples per tool to embed in the spec (LLM training wheels)
CAPTURE_SAMPLES             = os.environ.get("CAPTURE_SAMPLES", "false").lower() in ("1","true","t","yes","y","on")
CAPTURE_SAMPLE_BYTES        = int(os.environ.get("CAPTURE_SAMPLE_BYTES", "4000"))  # truncate
CAPTURE_SAMPLE_TIMEOUT_SECS = int(os.environ.get("CAPTURE_SAMPLE_TIMEOUT_SECS", "6"))

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
    version="3.0.0",
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
        # cached samples {toolName: {"args":{...}, "output":"...", "ts":...}}
        self.samples: Dict[str, Dict[str, Any]] = {}

    async def rpc(self, method: str, params: Optional[Dict[str, Any]] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
        client = await get_client()
        headers = {}
        if self.session_id:
            headers["mcp-session-id"] = self.session_id
        payload = {"jsonrpc": "2.0", "id": str(int(time.time() * 1000)), "method": method}
        if params is not None:
            payload["params"] = params
        try:
            resp = await client.post(self.rpc_url, json=payload, headers=headers, timeout=timeout or REQUEST_TIMEOUT)
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

        # capture small samples (non-blocking best-effort)
        if CAPTURE_SAMPLES:
            await self.capture_samples_best_effort()

        self.ready = True
        self.last_error = None

    async def capture_samples_best_effort(self):
        # For each tool, try zero-arg {} if possible
        for tname, tool in list(self.tools.items())[:10]:  # cap total
            if tname in self.samples:
                continue
            schema = tool.get("inputSchema") or {"type": "object"}
            # Construct a tiny example args (our generator is safe)
            example_args = example_from_schema(schema) or {}
            # Don't try obviously heavy kubectl combos (we keep it empty unless it has enums)
            try:
                result = await self.rpc(
                    "tools/call",
                    {"name": tname, "arguments": example_args},
                    timeout=CAPTURE_SAMPLE_TIMEOUT_SECS,
                )
                out = result
            except Exception:
                # Ignore failures; store nothing
                continue
            # Store truncated JSON string as sample
            try:
                txt = json.dumps(out, ensure_ascii=False)
            except Exception:
                txt = str(out)
            self.samples[tname] = {
                "args": example_args,
                "output": (txt[:CAPTURE_SAMPLE_BYTES] + ("…" if len(txt) > CAPTURE_SAMPLE_BYTES else "")),
                "ts": int(time.time()),
            }

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
        qp = {k: v for k, v in request.query_params.items() if k not in ("args","dryrun","format")}
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

def enrich_argstring(call_args: Dict[str, Any], schema: Dict[str, Any], tool_name: str, description: str, fmt: Optional[str]):
    """If the tool requires args:string and caller passed convenience params,
       auto-compose args (e.g., name + -n namespace + selectors). Also add -o json/yaml if format= specified."""
    props = (schema or {}).get("properties", {})
    if props.get("args", {}).get("type") != "string":
        return
    looks_kube = _looks_kubectl_like(tool_name, description, schema)
    if not looks_kube:
        return

    parts: List[str] = []
    raw = str(call_args.get("args") or "").strip()
    if raw:
        parts.append(raw)

    name = call_args.pop("name", None)
    namespace = call_args.pop("namespace", None)
    labels = call_args.pop("labels", None) or call_args.pop("labelSelector", None)
    field_selector = call_args.pop("fieldSelector", None)
    container = call_args.pop("container", None)
    since_seconds = call_args.pop("sinceSeconds", None)

    # positional first
    if name: parts.insert(0, name)
    _append_flag(parts, "-n", namespace)
    if labels: _append_flag(parts, "-l", labels)
    if field_selector: _append_flag(parts, "--field-selector", field_selector)
    if container: _append_flag(parts, "-c", container)
    if since_seconds is not None:
        try:
            ss = int(since_seconds)
            _append_flag(parts, "--since", f"{ss}s")
        except Exception:
            pass

    # -o <format> for certain operations (only when not already present)
    argline = " ".join(parts)
    if fmt in ("json","yaml") and "-o " not in argline and "--output" not in argline:
        op = (call_args.get("operation") or call_args.get("action") or "").lower()
        # Only inject for read-ish operations where kubectl supports -o
        if op in ("get","api-resources","api-versions"):
            parts.append(f"-o {fmt}")
        elif op == "logs" and fmt == "json":
            # logs do not support -o json; skip
            pass

    call_args["args"] = " ".join(parts).strip()

def parse_bool(v: Optional[str]) -> Optional[bool]:
    if v is None:
        return None
    s = v.lower().strip()
    if s in ("1","true","t","yes","y","on"): return True
    if s in ("0","false","f","no","n","off"): return False
    return None

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

@app.get("/servers", tags=["info"], summary="Servers Info")
async def servers_info(): return {"servers": {k: v.rpc_url for k, v in SERVERS.items()}}

@app.get("/{server}/tools/list", tags=["discovery"], dependencies=PUBLIC_OR_AUTH, summary="Tools List")
async def tools_list(server: str):
    st = SERVERS.get(server)
    if not st:
        raise HTTPException(status_code=404, detail={"message": f"Unknown server '{server}'"})
    return {
        "tools": list(st.tools.values()),
        "prompts": st.prompts,
        "resources": st.resources,
        "samples": st.samples,
    }

@app.post("/discover", tags=["discovery"], dependencies=PUBLIC_OR_AUTH, summary="Discover Endpoint")
async def discover_endpoint(wait: Optional[int] = Query(default=0, description="Seconds to wait (max 10) for discovery")):
    wait = max(0, min(int(wait or 0), 10))
    task = asyncio.create_task(discover_once())
    if wait:
        try:
            await asyncio.wait_for(task, timeout=wait)
        except asyncio.TimeoutError:
            return {"ok": False, "message": "Discovery still running", "waited": wait}
    return {"ok": True, "lastDiscovery": _last_discovery}

@app.get("/discovery/status", tags=["discovery"], summary="Discovery Status")
async def discovery_status():
    return {"lastDiscovery": _last_discovery,
            "servers": {k: {"ready": v.ready, "tools": list(v.tools.keys()),
                            "prompts": [p.get('name') for p in v.prompts],
                            "resources": [r.get('uri','') for r in v.resources],
                            "samples": {tn: {"args": sm["args"], "preview": sm["output"][:200]} for tn, sm in v.samples.items()},
                            "last_error": v.last_error}
                        for k, v in SERVERS.items()}}

# =========================
# Generic dispatcher for ALL tools & granular forms
# =========================
@app.api_route("/{server}/tool/{tool_path:path}", methods=["GET","POST"], tags=["tools"], dependencies=PUBLIC_OR_AUTH, summary="Tool Dispatch")
async def tool_dispatch(
    request: Request,
    server: str = Path(..., description="Server alias (e.g., 'mcp', 'ro', 'admin')"),
    tool_path: str = Path(..., description="Tool name or tool plus suffix, e.g., 'kubectl_resources', 'kubectl_resources/get/pods', 'kubectl_resources/invoke'"),
    args: Optional[str] = Query(default=None, description="JSON-encoded args fallback"),
    dryrun: Optional[bool] = Query(default=False, description="If true, returns the would-be MCP call without executing it"),
    format_hint: Optional[str] = Query(alias="format", default=None, description="Output preference: json|yaml|text (adds '-o json|yaml' when supported)"),
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
                "Use 'format=json|yaml' to request machine-readable output when supported (-o flag injection).",
                "Use 'dryrun=true' to preview the composed call.",
            ],
            "outputGuidance": output_guidance_for(tool_name, description, schema),
        }
    if suffix == ["try"]:
        await st.ensure_initialized()
        return await st.rpc("tools/call", {"name": tool_name, "arguments": {}})
    if suffix == ["invoke"]:
        await st.ensure_initialized()
        enrich_argstring(call_args, schema, tool_name, description, format_hint)
        if dryrun:
            return {"dryrun": {"name": tool_name, "arguments": call_args}}
        return await st.rpc("tools/call", {"name": tool_name, "arguments": call_args})

    # granular: /{action}[/{kind}]
    actions, kinds = iter_action_kind(schema)
    fixed: Dict[str, Any] = {}
    if len(suffix) >= 1 and suffix[0] not in ("schema","example","help","try","invoke"):
        fixed["action"] = suffix[0]
        fixed["operation"] = suffix[0]
    if len(suffix) >= 2:
        fixed["kind"] = suffix[1]
        fixed["resource"] = suffix[1]
    call_args.update(fixed)

    # Compose argstring (kubectl-like tools) and add -o when requested
    enrich_argstring(call_args, schema, tool_name, description, format_hint)

    await st.ensure_initialized()
    if dryrun:
        return {"dryrun": {"name": tool_name, "arguments": call_args}}
    return await st.rpc("tools/call", {"name": tool_name, "arguments": call_args})

# =========================
# Output guidance (for LLMs)
# =========================
def output_guidance_for(tool_name: str, description: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    looks_kube = _looks_kubectl_like(tool_name, description, schema)
    if not looks_kube:
        return {"preferredFormats": ["json","yaml","text"], "notes": ["Format depends on tool implementation."]}
    notes = [
        "For 'get' style operations, prefer `format=json` to inject `-o json` so output is machine-readable.",
        "For 'describe', kubectl emits human text (no -o json). Treat it as unstructured text.",
        "For 'logs', output is line-oriented text; do not expect JSON.",
        "For 'api-resources'/'api-versions', `format=json` works when supported; otherwise text table.",
    ]
    # Simple parse hints for common describe/logs
    parsing = {
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
            "extract": [{"field": "lines", "note": "Split by newline; may contain timestamps."}]
        }
    }
    return {
        "preferredFormats": ["json","yaml","text"],
        "notes": notes,
        "parsingHints": parsing
    }

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
            "Discover tools via `/{SERVER}/tools/list` and inspect fields with `/tool/{TOOL}/schema`.",
            "Use `format=json|yaml` to inject an output flag when supported (e.g., kubectl get).",
            "If the tool requires `args` (string), pass it OR provide convenience params like `namespace`, `name` etc.; the bridge will construct the argstring.",
            "Use `dryrun=true` to preview the exact MCP call.",
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
# OpenAPI builder – synthesize spec from MCP data
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
    {"name": "format", "in": "query", "required": False, "schema": {"type":"string","enum":["json","yaml","text"]},
     "description": "Output hint (adds '-o json|yaml' when supported)."},
    {"name": "dryrun", "in": "query", "required": False, "schema": {"type":"boolean"},
     "description": "If true, returns the would-be MCP call without executing it."},
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
    if include_convenience:
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
    out: List[str] = []
    for line in (desc or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("- "):
            out.append(s[2:])
        elif "operation=" in s or "args=" in s or "resource=" in s or "action=" in s or "kind=" in s:
            out.append(s)
    return out[:16]

def _nl_examples_from(server: str, tool_name: str, schema: Dict[str, Any], description: str) -> List[Dict[str, Any]]:
    props = schema.get("properties", {}) if isinstance(schema, dict) else {}
    has_ns = "namespace" in props or _looks_kubectl_like(tool_name, description, schema)
    exs: List[Dict[str, Any]] = []
    if has_ns:
        exs.append({
            "intent": "List all pods in the apisix namespace (machine-readable)",
            "calls": [
                {"GET": f"/{server}/tool/{_safe(tool_name)}/get/pods?namespace=apisix&format=json"},
                {"POST": f"/{server}/tool/{_safe(tool_name)}/get/pods", "body": {"namespace":"apisix"}, "query": {"format":"json"}},
            ],
            "notes": ["Use format=json to inject '-o json' when supported (e.g., kubectl get)."]
        })
        exs.append({
            "intent": "Describe the apisix pod (human text)",
            "calls": [
                {"GET": f"/{server}/tool/{_safe(tool_name)}/describe/pods?namespace=apisix&name=apisix"},
            ],
            "notes": ["Describe emits text; parse with `x-outputGuidance.parsingHints`."]
        })
    parsed = _parse_examples_from_description(description)
    if parsed:
        exs.append({"intent": "Examples from MCP description", "lines": parsed})
    return exs

def _response_block(schema: Dict[str, Any], tool_name: str, description: str) -> Dict[str, Any]:
    looks_kube = _looks_kubectl_like(tool_name, description, schema)
    if looks_kube:
        return {
            "200": {
                "description": "OK",
                "content": {
                    "application/json": {"schema": {"oneOf": [{"type":"object"},{"type":"array"}]}},
                    "application/yaml": {"schema": {"type":"string"}},
                    "text/plain": {"schema": {"type":"string"}}
                }
            }
        }
    # generic
    return {"200": {"description": "OK", "content": {"application/json": {"schema": {}}, "text/plain": {"schema": {"type":"string"}}}}}

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
            looks_kube = _looks_kubectl_like(tname, desc, schema)
            x_output = output_guidance_for(tname, desc, schema)
            output_samples = st.samples.get(tname)

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
                        + ("\n• Use `format=json|yaml|text` to request the output shape; json/yaml injects '-o <format>' when supported." if looks_kube else "")
                    ).strip(),
                    "parameters": _op_params_with_schema(schema, argstring_required),
                    "responses": _response_block(schema, tname, desc),
                    "x-usage": {
                        "schema": describe_schema(schema),
                        "argstringRequired": argstring_required,
                        "naturalExamples": _nl_examples_from(server, tname, schema, desc),
                        "outputGuidance": x_output,
                        **({"outputSample": output_samples} if output_samples else {}),
                    },
                }
                if method == "post":
                    op["requestBody"] = _make_request_body(schema, required=False)
                paths.setdefault(tool_base, {})[method] = op

            # helpers
            for helper, tag in (("invoke","invoke"),("schema","schema"),("example","example"),("help","help"),("try","try")):
                opid = f"{server}_{safe}_{helper}"
                paths.setdefault(f"{tool_base}/{helper}", {})["get"]  = {
                    "tags": [f"{server}:{tname}", tag],
                    "summary": f"{tname} ({helper})",
                    "operationId": opid,
                    "parameters": _op_params_with_schema(schema, argstring_required) if helper=="invoke" else [],
                    "responses": _response_block(schema, tname, desc),
                    **({"description":"Calls this tool with `{}` (no arguments)."} if helper=="try" else {}),
                }

            # Granular: /{action} and /{action}/{kind}
            if actions or kinds:
                for action in actions or []:
                    p = f"{tool_base}/{_safe(action)}"
                    for method in ("get","post"):
                        opid = f"{server}_{safe}_{_safe(action)}_{method}"
                        op = {
                            "tags": [f"{server}:{tname}", action],
                            "summary": f"{tname} → {action}",
                            "operationId": opid,
                            "description": f"Fixes `action: \"{action}\"`. Provide only remaining fields."
                                           + ("\nConvenience params (`namespace`, `name`, etc.) are accepted and composed into the argstring." if argstring_required else "")
                                           + ("\nUse `format=json|yaml|text` to control output shape." if looks_kube else ""),
                            "parameters": _op_params_with_schema(schema, argstring_required),
                            "responses": _response_block(schema, tname, desc),
                            "x-usage": {
                                "schema": describe_schema(schema),
                                "argstringRequired": argstring_required,
                                "naturalExamples": _nl_examples_from(server, tname, schema, desc),
                                "outputGuidance": x_output,
                                **({"outputSample": output_samples} if output_samples else {}),
                            },
                        }
                        if method == "post":
                            op["requestBody"] = _make_request_body(schema, required=False)
                        paths.setdefault(p, {})[method] = op
                    paths.setdefault(f"{p}/try", {})["get"] = {
                        "tags": [f"{server}:{tname}", action, "try"],
                        "summary": f"{tname} → {action} zero-arg try",
                        "operationId": f"{server}_{safe}_{_safe(action)}_try",
                        "responses": _response_block(schema, tname, desc),
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
                                (f"Fixes `kind: \"{kind}\"`. " if action=="_" else f"Fixes `action: \"{action}\"` and `kind: \"{kind}\"`. ")
                                + "Provide only remaining fields."
                                + ("\nConvenience params (`namespace`, `name`, etc.) are accepted and composed into the argstring." if argstring_required else "")
                                + ("\nUse `format=json|yaml|text` to control output shape." if looks_kube else "")
                            ),
                            "parameters": _op_params_with_schema(schema, argstring_required),
                            "responses": _response_block(schema, tname, desc),
                            "x-usage": {
                                "schema": describe_schema(schema),
                                "argstringRequired": argstring_required,
                                "naturalExamples": _nl_examples_from(server, tname, schema, desc),
                                "outputGuidance": x_output,
                                **({"outputSample": output_samples} if output_samples else {}),
                            },
                        }
                        if method == "post":
                            op["requestBody"] = _make_request_body(schema, required=False)
                        paths.setdefault(p, {})[method] = op
                    paths.setdefault(f"{p}/try", {})["get"] = {
                        "tags": [f"{server}:{tname}", *( [] if action=="_" else [action] ), kind, "try"],
                        "summary": f"{tname} → {(kind if action=='_' else action+'/'+kind)} zero-arg try",
                        "operationId": f"{server}_{safe}_{suffix.replace('/','_')}_try",
                        "responses": _response_block(schema, tname, desc),
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
                "Use `format=json|yaml|text` to influence output; json/yaml injects '-o' when supported.",
                "Use `dryrun=true` to preview the composed MCP call.",
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
                "4) Prefer `format=json` for machine-readable results when available.",
            ],
            "errorFix": [
                "If you see 'expected a request body', use GET or POST `{}`.",
                "If a tool needs `args` (string): either send it, or pass convenience params like `namespace`, `name`, etc.",
                "If you see a schema error, inspect `/schema`, `/example`, or try a granular endpoint.",
                "If parsing text, leverage `x-outputGuidance.parsingHints`.",
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

    # MCP catalog (tools + prompts + resources + samples)
    openapi["x-mcp-tool-catalog"] = [
        {
            "server": sname,
            "tool": tname,
            "description": t.get("description", "No description provided by MCP server."),
            "schema": describe_schema(t.get("inputSchema") or {"type":"object"}),
            **({"outputSample": SERVERS[sname].samples.get(tname)} if SERVERS[sname].samples.get(tname) else {}),
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
