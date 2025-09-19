#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MCP OpenAPI Bridge — Generic, Driver-Aware, Redis-backed
--------------------------------------------------------
All runtime config is sourced from a single YAML manifest mounted at
`/etc/mcp/driver-manifest.yaml` (or MANIFEST_PATH/DRIVER_MANIFEST_PATH).

- Transport (stdio/http) comes from spec.transport
- Servers list comes from spec.transport.servers
- Cache/Redis comes from spec.cache.registry.redis
- Drivers come from spec.drivers
- DB migrator config comes from spec.migrator.db
"""

import asyncio
import importlib
import inspect
import json
from linecache import cache
import os
import re
import stat
import sys
import textwrap
import threading
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Mapping
from urllib.parse import unquote
from manifest_validator import validate_manifest_yaml  # import your validator

from migrator import Migrator
import httpx
from fastapi import Body, FastAPI, HTTPException, Path, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from mcp_openapi.manifest_loader import load_manifest
from mcp_openapi.driver_loader import REGISTRY, RedisCache, InMemoryCache  # type: ignore

# =============================================================================
# Manifest: load once, make it the source of truth
# =============================================================================

DEFAULT_MANIFEST_PATH = "/etc/mcp/driver-manifest.yaml"
MANIFEST_PATH = os.getenv("MANIFEST_PATH") or os.getenv("DRIVER_MANIFEST_PATH") or DEFAULT_MANIFEST_PATH

MANIFEST: Optional[Dict[str, Any]] = None
try:
    txt = open(MANIFEST_PATH, "r").read()
    validate_manifest_yaml(txt)          # raises if invalid
    MANIFEST = load_manifest(MANIFEST_PATH)  # safe to parse after validation
    os.environ["DRIVER_MANIFEST_PATH"] = MANIFEST_PATH
except ValidationError as ve:
    print(f"[manifest] Invalid driver-manifest.yaml: {ve}", file=sys.stderr)
    sys.exit(1)  # or raise to abort startup
except Exception as e:
    print(f"[manifest] Failed to load manifest from {MANIFEST_PATH}: {e}", file=sys.stderr)
    MANIFEST = None

# After MANIFEST is loaded:
openapi_ns = (
    (MANIFEST.get("spec", {}).get("openapiCache") or {}).get("namespace")
    or (MANIFEST.get("metadata") or {}).get("name")
    or "openapi"
)
openapi_ttl = int(
    (MANIFEST.get("spec", {}).get("openapiCache") or {}).get("ttlSeconds")
    or (MANIFEST.get("spec", {}).get("redis") or {}).get("default_ttl_seconds")
    or 600
)
# Defaults for health state
MIGRATIONS_DONE: bool = True
MIGRATION_ERROR: str = ""

# =============================================================================
# Transport config (prefer manifest; no env fallback unless manifest absent)
# =============================================================================

MCP_STDIO_ENABLED = False
MCP_FORCE_STDIO = False
MCP_STDIO_INIT_TIMEOUT = 45
MCP_STDIO_PREFLIGHT = True
MCP_STDIO_PREFLIGHT_CONFIG = False
MCP_STDIO_EXTRA_ARGS = ""
MCP_RPC_URL: Optional[str] = None
MCP_FORWARD_URL: Optional[str] = None

if MANIFEST and "transport" in MANIFEST.get("spec", {}):
    t = MANIFEST["spec"]["transport"]
    stdio = t.get("stdio", {}) or {}
    MCP_STDIO_ENABLED = bool(stdio.get("enabled", False))
    MCP_FORCE_STDIO = bool(stdio.get("force", False))
    MCP_STDIO_INIT_TIMEOUT = int(stdio.get("initTimeoutSeconds", 45))
    MCP_STDIO_PREFLIGHT = bool(stdio.get("preflight", True))
    MCP_STDIO_PREFLIGHT_CONFIG = bool(stdio.get("preflightConfig", False))
    MCP_STDIO_EXTRA_ARGS = " ".join(stdio.get("extraArgs", []) or [])
    MCP_RPC_URL = (t.get("rpc", {}) or {}).get("url") or None
    MCP_FORWARD_URL = (t.get("forward", {}) or {}).get("url") or None
else:
    # Only fall back to env if manifest has no transport block
    MCP_RPC_URL = os.getenv("MCP_RPC_URL") or None
    MCP_FORWARD_URL = os.getenv("MCP_FORWARD_URL") or None

# Load stdio client if enabled
MCP_AVAILABLE = False
stdio_client = None
MCPClientSession = None
StdioParamsType = None
if MCP_STDIO_ENABLED or MCP_FORCE_STDIO:
    try:
        from mcp.client.stdio import stdio_client as _stdio_client  # type: ignore
        stdio_client = _stdio_client
        MCP_AVAILABLE = True
    except Exception:
        MCP_AVAILABLE = False
    try:
        from mcp.client.session import ClientSession as MCPClientSession  # type: ignore
    except Exception:
        try:
            from mcp.client.session import Session as MCPClientSession  # type: ignore
        except Exception:
            MCPClientSession = None
    try:
        from mcp.client.stdio import StdioServerParameters as StdioParamsType  # type: ignore
    except Exception:
        StdioParamsType = None  # fallback

# =============================================================================
# Models / in-memory discovery
# =============================================================================

class ServerConfig(BaseModel):
    alias: str
    mode: str = Field("stdio", description="stdio | custom")
    cmd: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    cwd: Optional[str] = None

class ToolDescriptor(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None
    examples: List[Dict[str, Any]] = Field(default_factory=list)

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

def getenv_json(name: str, default: Any) -> Any:
    val = os.getenv(name)
    if not val:
        return default
    try:
        return json.loads(val)
    except Exception:
        return default

def _safe_yaml_load(txt: str) -> Any:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("PyYAML is required to parse YAML manifests/config") from e
    return yaml.safe_load(txt)

def _mask(s: str, keep: int = 6) -> str:
    if not s:
        return s
    return s if len(s) <= keep else s[:keep] + "*" * (len(s) - keep)

def _schema_enums(schema: Dict[str, Any]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    try:
        props = (schema or {}).get("properties", {})
        for k, v in (props or {}).items():
            if isinstance(v, dict) and "enum" in v and isinstance(v["enum"], list):
                out[k] = [str(x) for x in v["enum"]]
    except Exception:
        pass
    return out

def _deep_merge(dst: Dict[str, Any], src: Mapping[str, Any]) -> None:
    """
    Recursively merge the key/value pairs from ``src`` into ``dst``.

    Nested dictionaries are merged; all other values replace the destination.
    """
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)  # type: ignore[arg-type]
        else:
            dst[k] = v

def _schema_fields(schema: Dict[str, Any]) -> List[str]:
    try:
        props = (schema or {}).get("properties", {})
        return list(props.keys())
    except Exception:
        return []

def _build_examples_from_schema(tool_name: str, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    props = _schema_fields(schema)
    enums = _schema_enums(schema)
    examples: List[Dict[str, Any]] = []

    def sample_body() -> Dict[str, Any]:
        body: Dict[str, Any] = {}
        for p in props:
            body[p] = (enums[p][0] if p in enums and enums[p] else f"<{p}>")
        return body

    body = sample_body()
    examples.append({
        "intent": f"Call '{tool_name}' with a JSON body",
        "call": {"POST": f"/mcp/tool/{tool_name}", "body": body},
        "notes": ["Body is forwarded as-is to the MCP tool."]
    })
    examples.append({
        "intent": f"GET invoke '{tool_name}' with JSON-encoded args",
        "call": {"GET": f"/mcp/tool/{tool_name}?args=" + json.dumps(body)},
        "notes": ["The 'args' query param must be JSON-encoded."]
    })
    interesting = [k for k in ("operation", "resource", "action", "kind", "type") if k in props]
    if interesting:
        body2 = sample_body()
        examples.append({
            "intent": f"Explicitly set {', '.join(interesting)} for '{tool_name}'",
            "call": {"POST": f"/mcp/tool/{tool_name}", "body": body2},
            "notes": ["Values shown use the first enum choices when available."]
        })
    return examples

# =============================================================================
# MCP (HTTP RPC)
# =============================================================================

async def rpc_try_methods(client: httpx.AsyncClient, url: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    last_exc: Optional[Exception] = None
    for payload in candidates:
        try:
            r = await client.post(url, json=payload, timeout=60)
            if r.status_code in (200, 207):
                return r.json()
        except Exception as e:
            last_exc = e
    if last_exc:
        raise last_exc
    raise RuntimeError("No RPC method variant succeeded")

async def mcp_http_list_tools() -> List[Dict[str, Any]]:
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
    tools = []
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
    container = data.get("result", data)
    content = container.get("content") or container.get("contents") or []
    normalized = {"type": "mcp_result", "content": []}
    for item in content:
        if isinstance(item, dict):
            if item.get("type") == "text" and "text" in item:
                normalized["content"].append({"type": "text", "text": item["text"]})
            else:
                normalized["content"].append(item)
        else:
            normalized["content"].append({"type": "text", "text": str(item)})
    return normalized

# =============================================================================
# MCP (stdio)
# =============================================================================

async def _enter_ctx(cm):
    aenter = getattr(cm, "__aenter__", None)
    if not callable(aenter):
        raise TypeError("Expected an async context manager")
    entered = aenter()
    return await entered if inspect.isawaitable(entered) else entered

async def _exit_ctx(cm):
    if not cm:
        return
    aexit = getattr(cm, "__aexit__", None)
    if callable(aexit):
        try:
            maybe = aexit(None, None, None)
            if inspect.isawaitable(maybe):
                await maybe
        except Exception:
            pass

def _is_executable(path: str) -> bool:
    try:
        st = os.stat(path)
        if not stat.S_ISREG(st.st_mode):
            return False
        return bool(st.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
    except FileNotFoundError:
        return False
    except Exception:
        return False

def _split_cmd(cfg_cmd: Optional[List[str] | str]) -> Tuple[str, List[str]]:
    if isinstance(cfg_cmd, list) and cfg_cmd:
        return str(cfg_cmd[0]), [str(x) for x in cfg_cmd[1:]]
    if isinstance(cfg_cmd, str) and cfg_cmd:
        return cfg_cmd, []
    raise RuntimeError("cmd must be a non-empty list[str] or str")

def _env_for(cfg: ServerConfig) -> Dict[str, str]:
    env = dict(os.environ)
    if cfg.env:
        env.update(cfg.env)
    return env

def _make_desc(command: str, args: List[str], env: Dict[str, str], cwd: str):
    if StdioParamsType is not None:
        return StdioParamsType(
            command=command, args=args, env=env, cwd=cwd,
            encoding="utf-8", stderr_encoding="utf-8",
            encoding_error_handler="replace", stderr_encoding_error_handler="replace",
        )
    return SimpleNamespace(
        command=command, args=args, env=env, cwd=cwd,
        encoding="utf-8", stderr_encoding="utf-8",
        encoding_error_handler="replace", stderr_encoding_error_handler="replace",
    )

def _with_extra_args(args: List[str]) -> List[str]:
    if not MCP_STDIO_EXTRA_ARGS:
        return args
    extra = [a for a in MCP_STDIO_EXTRA_ARGS.split() if a]
    return args + extra

def _arg_permutations(base_args: List[str]) -> List[List[str]]:
    perms: List[List[str]] = []

    def has_pair(k: str) -> bool:
        try:
            i = base_args.index(k)
            return i < len(base_args) - 1
        except ValueError:
            return False

    def has_equals(k: str) -> bool:
        return any(s.startswith(k + "=") for s in base_args)

    perms.append(list(base_args))

    if has_pair("--transport"):
        i = base_args.index("--transport")
        if i < len(base_args) - 1:
            v = base_args[i + 1]
            perms.append(base_args[:i] + [f"--transport={v}"] + base_args[i + 2:])

    if has_equals("--transport"):
        for s in base_args:
            if s.startswith("--transport="):
                v = s.split("=", 1)[1]
                rest = [x for x in base_args if x != s]
                perms.append(["--transport", v] + rest)
                break

    if not has_pair("--transport") and not has_equals("--transport"):
        perms.append(base_args + ["--transport=stdio"])
        perms.append(base_args + ["--transport", "stdio"])

    seen = set()
    uniq: List[List[str]] = []
    for p in perms:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            uniq.append(_with_extra_args(p))
    return uniq

async def _run_short(command: str, args: List[str], env: Dict[str, str], cwd: str, tag: str, timeout_s: float = 6.0):
    try:
        proc = await asyncio.create_subprocess_exec(
            command, *args, cwd=cwd, env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            outs, errs = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass
            print(f"[stdio-preflight] {tag} timeout: {command} {' '.join(args)}", file=sys.stderr)
            return
        out_s = (outs or b"")[:400].decode("utf-8", "replace")
        err_s = (errs or b"")[:400].decode("utf-8", "replace")
        code = proc.returncode
        print(f"[stdio-preflight] {tag} rc={code}; stdout[:400]=\n{out_s}\n--- stderr[:400]=\n{err_s}", file=sys.stderr)
    except FileNotFoundError:
        print(f"[stdio-preflight] {tag} NOT FOUND: {command}", file=sys.stderr)
    except Exception as e:
        print(f"[stdio-preflight] {tag} failed: {type(e).__name__}: {e}", file=sys.stderr)

async def _preflight_all(command: str, args: List[str], env: Dict[str, str], cwd: str):
    if not MCP_STDIO_PREFLIGHT:
        return
    await _run_short(command, ["--version"], env, cwd, "version")
    await _run_short(command, ["--help"], env, cwd, "help")
    if MCP_STDIO_PREFLIGHT_CONFIG:
        await _run_short(command, args, env, cwd, "configured-args", timeout_s=4.0)

async def mcp_connect_stdio(cfg: ServerConfig) -> Any:
    if not (MCP_STDIO_ENABLED or MCP_FORCE_STDIO):
        raise RuntimeError("MCP stdio disabled by manifest.")
    if not MCP_AVAILABLE or MCPClientSession is None or stdio_client is None:
        raise RuntimeError("MCP stdio client/session not available.")
    if not cfg.cmd:
        raise RuntimeError(f"Server {cfg.alias}: stdio mode requires 'cmd'.")

    command, base_args = _split_cmd(cfg.cmd)
    if os.path.isabs(command) and not _is_executable(command):
        raise RuntimeError(f"Executable not found or not executable: '{command}'")
    env = _env_for(cfg)
    cwd = cfg.cwd or os.getcwd()

    await _preflight_all(command, base_args, env, cwd)

    attempts = _arg_permutations(base_args)
    errors: List[str] = []

    for idx, args in enumerate(attempts, start=1):
        desc = _make_desc(command, args, env, cwd)
        stdio_cm = None
        session_ctx = None
        try:
            stdio_cm = stdio_client(desc)  # descriptor-only
            rw = await _enter_ctx(stdio_cm)
            if not isinstance(rw, (tuple, list)) or len(rw) != 2:
                raise TypeError("stdio_client did not yield (read_stream, write_stream)")
            read_stream, write_stream = rw[0], rw[1]
            session_ctx = MCPClientSession(read_stream, write_stream)
            session = await _enter_ctx(session_ctx)

            init = getattr(session, "initialize", None)
            if callable(init):
                maybe = init()
                if inspect.isawaitable(maybe):
                    await asyncio.wait_for(maybe, timeout=MCP_STDIO_INIT_TIMEOUT)

            setattr(session, "__stdio_ctx__", stdio_cm)
            setattr(session, "__session_ctx__", session_ctx)
            setattr(session, "__stdio_args__", list(args))
            return session
        except Exception as e:
            await _exit_ctx(session_ctx)
            await _exit_ctx(stdio_cm)
            errors.append(f"#{idx} args={args} -> {type(e).__name__}: {e}")

    raise RuntimeError("Failed to open stdio_client for "
                       f"'{cfg.alias}': " + " | ".join(errors))

async def mcp_list_tools_stdio(session) -> List[Dict[str, Any]]:
    result = await session.list_tools()
    raw = getattr(result, "tools", result)
    out: List[Dict[str, Any]] = []
    for t in raw:
        name = getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else None)
        if not name:
            continue
        desc = getattr(t, "description", None)
        if desc is None and isinstance(t, dict):
            desc = t.get("description", "")
        schema = getattr(t, "inputSchema", None)
        if schema is None and isinstance(t, dict):
            schema = t.get("inputSchema") or t.get("input_schema") or {}
        if hasattr(schema, "model_dump"):
            schema = schema.model_dump()
        out.append({"name": name, "description": desc or "", "input_schema": schema or {}})
    return out

async def mcp_call_tool_stdio(session, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    call = await session.call_tool(tool_name, args)
    content_seq = getattr(call, "content", None) or call
    normalized = {"type": "mcp_result", "content": []}
    try:
        from mcp.types import TextContent as _TC  # type: ignore
    except Exception:
        _TC = None  # type: ignore
    for item in content_seq:
        if _TC and isinstance(item, _TC):
            normalized["content"].append({"type": "text", "text": item.text})
        else:
            payload = item.model_dump() if hasattr(item, "model_dump") else (item if isinstance(item, dict) else {})
            ctype = payload.get("type") or "unknown"
            normalized["content"].append(payload if ctype != "unknown" else {"type": "unknown", "data": payload})
    return normalized

# =============================================================================
# Manifest helpers
# =============================================================================

def _parse_manifest_to_specs(path: str) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        doc = _safe_yaml_load(f.read())
    if not isinstance(doc, dict) or "spec" not in doc:
        raise ValueError("Invalid DriverManifest: missing top-level 'spec'")
    spec = doc.get("spec") or {}
    iid = spec.get("instanceId")
    drivers = spec.get("drivers") or []
    out: List[Dict[str, Any]] = []
    for d in drivers:
        mod = d.get("module")
        if not mod:
            name = d.get("name", "<unnamed>")
            raise ValueError(f"Driver entry '{name}' missing 'module'")
        out.append({"module": mod, "config": d.get("config")})
    return iid, out

def _apply_manifest_to_env_and_registry(manifest: Dict[str, Any]) -> None:
    """Map manifest cache/instance into env + live registry before driver load."""
    spec = manifest.get("spec", {}) or {}

    # Instance ID influences cache namespace, etc.
    iid = spec.get("instanceId")
    if iid:
        os.environ["INSTANCE_ID"] = str(iid)
        REGISTRY.instance_id = str(iid)

    # Redis config: prefer spec.redis, fallback to old spec.cache.registry.redis
    redis_cfg = spec.get("redis") or ((spec.get("cache") or {}).get("registry") or {}).get("redis") or {}
    redis_url = redis_cfg.get("url")
    key_prefix = redis_cfg.get("key_prefix") or redis_cfg.get("namespace") or "driver"
    default_ttl = str(redis_cfg.get("default_ttl_seconds", 600))
    # set envs consumed by RedisCache()
    if redis_url:
        os.environ["REDIS_URL"] = redis_url
    os.environ["DRIVER_CACHE_NS_PREFIX"] = key_prefix
    os.environ["DRIVER_CACHE_DEFAULT_TTL"] = default_ttl

    # Make sure REGISTRY.cache matches desired backend
    try:
        if os.getenv("REDIS_URL"):
            REGISTRY.cache = RedisCache(url=redis_url, prefix=key_prefix, default_ttl=int(default_ttl))
        else:
            REGISTRY.cache = InMemoryCache(prefix=key_prefix, default_ttl=int(default_ttl))
    except Exception as e:
        print(f"[cache] failed to initialize cache from manifest: {e}", file=sys.stderr)

async def _reload_drivers_from_manifest() -> Dict[str, Any]:
    path = os.getenv("DRIVER_MANIFEST_PATH")
    if not path:
        return {"error": "No DRIVER_MANIFEST_PATH set"}
    iid, specs = _parse_manifest_to_specs(path)
    if iid and iid != REGISTRY.instance_id:
        os.environ["INSTANCE_ID"] = iid
        REGISTRY.instance_id = iid
        REGISTRY.cache = RedisCache() if os.getenv("REDIS_URL") else InMemoryCache()
    os.environ["DRIVERS"] = json.dumps(specs)
    return await REGISTRY.load_from_env()

# =============================================================================
# FastAPI app + CORS (from manifest if available)
# =============================================================================

cors = MANIFEST["spec"].get("cors", {}) if MANIFEST else {}
allow_origins = cors.get("allow_origins", ["*"])
allow_methods = cors.get("allow_methods", ["*"])
allow_headers = cors.get("allow_headers", ["*"])

app = FastAPI(
    title="MCP OpenAPI Bridge (Generic, Driver-Aware)",
    version="9.2.0",
    description=textwrap.dedent(
        """Generic OpenAPI façade for MCP servers. All configuration is read from a single manifest."""
    ),
    servers=[{"url": "http://localhost:8080"}],
    openapi_url="/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins, allow_credentials=True,
    allow_methods=allow_methods, allow_headers=allow_headers,
)

# =============================================================================
# URL normalization
# =============================================================================

_ALIAS_RE = re.compile(r"^/mcp/tool/call_([^_/]+)_(.+)$")

@app.middleware("http")
async def normalize_odd_paths(request: Request, call_next):
    raw_path = request.scope.get("path") or ""
    decoded = unquote(raw_path)

    m = _ALIAS_RE.match(decoded)
    if m:
        _alias, tool = m.group(1), m.group(2)
        decoded = f"/mcp/tool/{tool}"

    for bad in ("<server-alias>", "<server>", "server-alias", "server"):
        prefix = f"/{bad}/"
        if decoded.startswith(prefix):
            decoded = "/mcp/" + decoded[len(prefix):]

    marker = "/tool/"
    if marker in decoded:
        head, tail = decoded.split(marker, 1)
        tail = "/".join([p for p in tail.replace("  ", " ").strip().split(" ") if p])
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

# =============================================================================
# Startup / Shutdown
# =============================================================================

@app.on_event("startup")
async def on_startup():
    global MIGRATIONS_DONE, MIGRATION_ERROR

    # 1) Apply manifest to env + registry (instance id, cache)
    try:
        if MANIFEST:
            _apply_manifest_to_env_and_registry(MANIFEST)
    except Exception as e:
        print(f"[startup] cache/instance application failed: {e}", file=sys.stderr)

    # 2) Run DB migrations (from manifest)
    try:
        MIGRATIONS_DONE = True
        MIGRATION_ERROR = ""
        if MANIFEST and MANIFEST["spec"].get("migrator", {}).get("run", False):
            db_cfg = MANIFEST["spec"]["migrator"]["db"]
            os.environ["DB_HOST"] = db_cfg.get("host", "localhost")
            os.environ["DB_PORT"] = str(db_cfg.get("port", 5432))
            os.environ["DB_NAME"] = db_cfg.get("name", "postgres")
            os.environ["DB_SCHEMA"] = db_cfg.get("schema", "public")
            os.environ["DB_SSLMODE"] = db_cfg.get("sslmode", "prefer")
            os.environ["DB_USER_FILE"] = db_cfg.get("usernameFile", "")
            os.environ["DB_PASS_FILE"] = db_cfg.get("passwordFile", "")
            os.environ["DB_MIGRATIONS_DIR"] = db_cfg.get("migrationsDir", "/app/migrations")
            os.environ["DB_SEEDS_DIR"] = db_cfg.get("seedsDir", "/app/seeds")
            os.environ["RUN_MIGRATIONS"] = "1"
        Migrator().run_on_startup()
    except Exception as e:
        MIGRATIONS_DONE = False
        MIGRATION_ERROR = str(e)
        print(f"[migrator] FAILED: {e}", file=sys.stderr)

    # 3) Load servers from manifest transport.servers (no env dependency)
    servers_cfg = []
    if MANIFEST and "transport" in MANIFEST.get("spec", {}) and "servers" in MANIFEST["spec"]["transport"]:
        servers_cfg = MANIFEST["spec"]["transport"]["servers"]
    for cfg in servers_cfg:
        cfg_obj = ServerConfig(**cfg)
        DISCOVERY.servers[cfg_obj.alias] = ServerState(cfg_obj)
    if "mcp" not in DISCOVERY.servers:
        DISCOVERY.servers["mcp"] = ServerState(ServerConfig(alias="mcp"))

    # 4) Discover tools
    try:
        wait = 0  # purely manifest-driven; no env wait
        await do_discover(wait_seconds=wait)
    except Exception as e:
        print(f"Validation failed:\n{e}", file=sys.stderr)

    # 5) Load drivers (prefer manifest)
    try:
        if MANIFEST:
            drivers = MANIFEST["spec"].get("drivers", []) or []
            driver_specs = [{"module": d["module"], "config": d.get("config")} for d in drivers]
            os.environ["DRIVERS"] = json.dumps(driver_specs)
        status = await REGISTRY.load_from_env()
        print(f"[drivers] loaded: {status}", file=sys.stderr)
    except Exception as e:
        print(f"[drivers] load failed: {e}", file=sys.stderr)

@app.on_event("shutdown")
async def on_shutdown():
    try:
        await REGISTRY.close_all()
    except Exception:
        pass

# =============================================================================
# Discovery / Health
# =============================================================================

def refresh_servers_from_env() -> bool:
    # kept for compatibility; not used in manifest-first flow
    updated = False
    servers_cfg = getenv_json("MCP_SERVERS", None) or []
    for cfg in servers_cfg:
        cfg_obj = ServerConfig(**cfg)
        st = DISCOVERY.servers.get(cfg_obj.alias)
        if not st or (st.cfg != cfg_obj):
            DISCOVERY.servers[cfg_obj.alias] = ServerState(cfg_obj)
            updated = True
    return updated

async def do_discover(wait_seconds: int = 0) -> Dict[str, Any]:
    for alias, st in list(DISCOVERY.servers.items()):
        tools_raw: List[Dict[str, Any]] = []
        prefer_stdio = MCP_FORCE_STDIO

        if not prefer_stdio and MCP_RPC_URL:
            try:
                tools_raw = await mcp_http_list_tools()
                if tools_raw:
                    st.connected = True
            except Exception as e:
                print(f"[discover:http-rpc] list_tools failed via {MCP_RPC_URL}: {e}", file=sys.stderr)

        if (prefer_stdio or not tools_raw) and (MCP_STDIO_ENABLED or MCP_FORCE_STDIO) and st.cfg.mode == "stdio" and st.cfg.cmd:
            try:
                if not st.connected or st.client is None:
                    st.client = await mcp_connect_stdio(st.cfg)
                    st.connected = True
                tools_raw = await mcp_list_tools_stdio(st.client)
            except Exception as e:
                print(f"[discover] stdio connect/list failed for '{alias}': {e}", file=sys.stderr)
                st.connected = False
                st.client = None

        st.tools.clear()
        for tr in tools_raw:
            name = tr["name"]
            desc = tr.get("description") or ""
            schema = tr.get("input_schema") or {}
            examples = _build_examples_from_schema(name, schema or {})
            st.tools[name] = ToolDescriptor(name=name, description=desc, input_schema=schema, examples=examples)

    if wait_seconds > 0:
        await asyncio.sleep(min(wait_seconds, 10))

    return {
        "servers": [{
            "alias": alias,
            "connected": st.connected,
            "tools": [t for t in st.tools.keys()]
        } for alias, st in DISCOVERY.servers.items()]
    }

@app.get("/livez", tags=["health"])
async def livez():
    return {"status": "ok"}

@app.get("/readyz", tags=["health"])
async def readyz():
    return {"status": "ok", "servers": DISCOVERY.list_servers()}

@app.get("/healthz", tags=["health"])
async def healthz():
    if not MIGRATIONS_DONE:
        return JSONResponse({"status":"starting","migrations":"pending","error":MIGRATION_ERROR}, status_code=503)
    return {"status": "ok"}

@app.get("/servers", tags=["info"])
async def servers_info():
    return {"servers": DISCOVERY.list_servers()}

@app.get("/drivers/status", tags=["info"])
async def drivers_status():
    return await REGISTRY.describe_all()

# =============================================================================
# Introspection / reload
# =============================================================================

@app.get("/introspect/cache/status", tags=["introspect"])
async def introspect_cache_status():
    base = {
        "namespace": REGISTRY.cache.namespace(),
        "defaultTTL": int(os.getenv("DRIVER_CACHE_DEFAULT_TTL", "600")),
        "maxEntries": os.getenv("DRIVER_CACHE_MAX_ENTRIES", "10000"),
        "prefixEnv": os.getenv("DRIVER_CACHE_NS_PREFIX", "driver"),
        "instanceId": REGISTRY.instance_id,
        "kind": type(REGISTRY.cache).__name__,
    }
    r = getattr(REGISTRY.cache, "_r", None)
    if r is not None:
        try:
            pong = r.ping()
        except Exception as e:
            pong = f"error: {type(e).__name__}: {e}"
        try:
            dbsize = r.dbsize()
        except Exception as e:
            dbsize = f"error: {type(e).__name__}: {e}"
        try:
            info_server = r.info("server")
            version = info_server.get("redis_version")
        except Exception:
            version = None
        base["redis"] = {
            "url": _mask(os.getenv("REDIS_URL", "")),
            "ping": pong,
            "dbsize": dbsize,
            "version": version,
        }
    else:
        size = None
        try:
            size = len(getattr(REGISTRY.cache, "_d", {}))
        except Exception:
            pass
        base["memory"] = {"entries": size, "maxEntries": int(os.getenv("DRIVER_CACHE_MAX_ENTRIES", "10000"))}
    return base

@app.post("/introspect/cache/selftest", tags=["introspect"])
async def introspect_cache_selftest():
    import random, string
    ns = "introspect"
    key = "selftest-" + "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    payload = {"ok": True, "key": key}
    try:
        REGISTRY.cache.set(ns, key, payload, ttl_seconds=30)
        got = REGISTRY.cache.get(ns, key)
        success = (got == payload)
        return {"success": success, "written": payload, "readBack": got, "cache": await introspect_cache_status()}
    finally:
        try:
            REGISTRY.cache.delete(ns, key)
        except Exception:
            pass

@app.get("/introspect/env/core", tags=["introspect"])
async def introspect_env_core():
    return {
        "INSTANCE_ID": os.getenv("INSTANCE_ID", "default"),
        "REDIS_URL": _mask(os.getenv("REDIS_URL", "")),
        "DRIVER_CACHE_DEFAULT_TTL": os.getenv("DRIVER_CACHE_DEFAULT_TTL", "600"),
        "DRIVER_CACHE_MAX_ENTRIES": os.getenv("DRIVER_CACHE_MAX_ENTRIES", "10000"),
        "DRIVER_CACHE_NS_PREFIX": os.getenv("DRIVER_CACHE_NS_PREFIX", "driver"),
        "DRIVER_MANIFEST_PATH": os.getenv("DRIVER_MANIFEST_PATH", ""),
        "DRIVERS_present": bool(os.getenv("DRIVERS")),
        "DRIVERS_length": len(os.getenv("DRIVERS", "")),
    }

@app.post("/introspect/drivers/reload", tags=["introspect"])
async def introspect_drivers_reload():
    try:
        if os.getenv("DRIVER_MANIFEST_PATH"):
            status = await _reload_drivers_from_manifest()
        else:
            status = await REGISTRY.load_from_env()
        return status
    except Exception as e:
        raise HTTPException(500, f"Driver reload failed: {e}")

# =============================================================================
# Status page
# =============================================================================

@app.get("/status", include_in_schema=False)
async def html_status():
    cache = await introspect_cache_status()
    drivers = await REGISTRY.describe_all()
    html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>MCP OpenAPI — Status</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; }}
code, pre {{ background: #f6f8fa; padding: 2px 4px; border-radius: 4px; }}
section {{ margin-bottom: 2rem; }}
summary {{ font-weight: 600; }}
.kv td {{ padding: .25rem .5rem; vertical-align: top; }}
.kv td:first-child {{ color: #555; }}
</style>
</head>
<body>
  <h1>MCP OpenAPI — Status</h1>

  <section>
    <div class="summary">Instance</div>
    <table class="kv">
      <tr><td>Instance ID</td><td><code>{cache.get("instanceId")}</code></td></tr>
      <tr><td>Cache Kind</td><td><code>{cache.get("kind")}</code></td></tr>
      <tr><td>Cache Namespace</td><td><code>{cache.get("namespace")}</code></td></tr>
    </table>
  </section>

  <section>
    <div class="summary">Cache</div>
    <pre>{json.dumps(cache, indent=2, ensure_ascii=False)}</pre>
  </section>

  <section>
    <div class="summary">Drivers</div>
    <pre>{json.dumps(drivers, indent=2, ensure_ascii=False)}</pre>
  </section>

  <section>
    <div class="summary">Useful endpoints</div>
    <ul>
      <li><code>/introspect/cache/status</code></li>
      <li><code>/introspect/cache/selftest</code> (POST)</li>
      <li><code>/introspect/drivers/reload</code> (POST)</li>
      <li><code>/introspect/env/core</code></li>
      <li><code>/drivers/status</code></li>
      <li><code>/discovery/status</code></li>
      <li><code>/discover</code> (POST)</li>
    </ul>
  </section>
</body>
</html>
"""
    return Response(content=html, media_type="text/html")

# =============================================================================
# Discovery control
# =============================================================================

@app.post("/discover", tags=["discovery"])
async def discover_endpoint(wait: Optional[int] = Query(0)):
    wait = max(0, min(int(wait or 0), 10))
    # env refresh kept for compatibility with legacy env-based config
    refresh_servers_from_env()
    return await do_discover(wait_seconds=wait)

@app.get("/discovery/status", tags=["discovery"])
async def discovery_status():
    return {"servers": DISCOVERY.list_servers()}

# =============================================================================
# HTTP fallback
# =============================================================================

async def forward_via_http(server: str, tool_path: str, method: str, params: Dict[str, Any], body: Optional[Dict[str, Any]]):
    if not MCP_FORWARD_URL:
        raise HTTPException(503, "No MCP server connected and MCP_FORWARD_URL not set")
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
# Tool dispatch
# =============================================================================

async def do_tool_call(server: str, tool_name: str, body: Optional[Dict[str, Any]], qargs: Optional[str], dryrun: bool) -> Any:
    if body is None:
        if qargs is not None:
            try:
                parsed = json.loads(qargs)
                args = parsed if isinstance(parsed, dict) else {"args": parsed}
            except Exception:
                args = {"args": qargs}
        else:
            args = {}
    else:
        args = body

    if dryrun:
        return {"dryrun": True, "server": server, "tool": tool_name, "args": args}

    prefer_stdio = MCP_FORCE_STDIO

    if not prefer_stdio and MCP_RPC_URL:
        try:
            return await mcp_http_call_tool(tool_name, args)
        except Exception as e:
            if MCP_FORWARD_URL:
                return await forward_via_http(server, tool_name, "POST", {}, args)
            raise HTTPException(502, f"HTTP RPC invocation failed: {e}")

    if MCP_STDIO_ENABLED or MCP_FORCE_STDIO:
        st = DISCOVERY.servers.get(server)
        if not st or not st.cfg.cmd:
            raise HTTPException(503, "stdio requested but not configured (missing server/cmd)")
        try:
            if not st.connected or st.client is None:
                st.client = await mcp_connect_stdio(st.cfg)
                st.connected = True
                try:
                    await do_discover(0)
                except Exception:
                    pass
            return await mcp_call_tool_stdio(st.client, tool_name, args)
        except Exception as e:
            raise HTTPException(502, f"Stdio invocation failed: {e}")

    raise HTTPException(503, "No usable transport: set RPC url in manifest or enable stdio in manifest")

@app.post("/{server}/tool/{tool_name:path}", tags=["tools"])
async def tool_dispatch_post(
    server: str = Path(..., description="Server alias (e.g., 'mcp')"),
    tool_name: str = Path(..., description="Exact tool name as exposed by MCP"),
    dryrun: Optional[bool] = Query(False),
    body: Optional[Dict[str, Any]] = Body(None),
):
    try:
        result = await do_tool_call(server, tool_name, body, None, bool(dryrun))
        return JSONResponse(result)
    except HTTPException as e:
        if e.status_code == 503 and MCP_FORWARD_URL:
            params: Dict[str, Any] = {}
            if dryrun:
                params["dryrun"] = dryrun
            return await forward_via_http(server, tool_name, "POST", params, body or {})
        raise

@app.get("/{server}/tool/{tool_name:path}", tags=["tools"])
async def tool_dispatch_get(
    server: str = Path(..., description="Server alias (e.g., 'mcp')"),
    tool_name: str = Path(..., description="Exact tool name as exposed by MCP"),
    dryrun: Optional[bool] = Query(False),
    args: Optional[str] = Query(None, description="JSON-encoded arguments"),
):
    try:
        result = await do_tool_call(server, tool_name, None, args, bool(dryrun))
        return JSONResponse(result)
    except HTTPException as e:
        if e.status_code == 503 and MCP_FORWARD_URL:
            params: Dict[str, Any] = {}
            if dryrun:
                params["dryrun"] = dryrun
            if args is not None:
                params["args"] = args
            return await forward_via_http(server, tool_name, "GET", params, None)
        raise

@app.get("/mcp/tool/{tool}/schema", tags=["tools", "schema"])
async def tool_schema(tool: str):
    st = DISCOVERY.servers.get("mcp")
    if not st or tool not in st.tools:
        return {}
    return st.tools[tool].input_schema or {}

@app.get("/mcp/tool/{tool}/help", tags=["tools", "help"])
async def tool_help(tool: str):
    st = DISCOVERY.servers.get("mcp")
    if not st or tool not in st.tools:
        return {
            "name": tool,
            "description": "",
            "notes": [
                "Arguments are forwarded to the MCP tool exactly as you send them.",
                "Use /mcp/tool/{tool}/schema and /mcp/tool/{tool}/example for guidance."
            ],
        }
    td = st.tools[tool]
    return {
        "name": tool,
        "description": td.description,
        "notes": [
            "Arguments are forwarded to the MCP tool exactly as you send them.",
            "Use /mcp/tool/{tool}/schema and /mcp/tool/{tool}/example for guidance."
        ],
    }

@app.get("/mcp/tool/{tool}/example", tags=["tools", "example"])
async def tool_example(tool: str):
    st = DISCOVERY.servers.get("mcp")
    if not st or tool not in st.tools:
        return {"examples": []}
    return {"examples": st.tools[tool].examples}

@app.get("/mcp/tool/{tool}/try", tags=["tools", "try"], description="Calls this tool with `{}` (no arguments).")
async def tool_try(tool: str, dryrun: Optional[bool] = Query(False)):
    return await tool_dispatch_get(server="mcp", tool_name=tool, dryrun=dryrun)

# =============================================================================
# OpenAPI augmentation (unchanged except for version and minor comments)
# =============================================================================

def _inject_tool_operations(openapi_schema: Dict[str, Any]) -> None:
    paths = openapi_schema.setdefault("paths", {})
    tags = openapi_schema.setdefault("tags", [])

    try:
        for d in REGISTRY.loaded.values():
            inst = d.instance
            if hasattr(inst, "openapi_tags"):
                for t in (inst.openapi_tags() or []):
                    if not any(existing.get("name") == t.get("name") for existing in tags):
                        tags.append(t)
    except Exception:
        pass

    for alias, st in DISCOVERY.servers.items():
        if alias != "mcp":
            continue
        for tname, td in st.tools.items():
            p = f"/mcp/tool/{tname}"
            if p not in paths:
                paths[p] = {}

            post_op = {
                "tags": ["tools"],
                "summary": f"Call MCP tool '{tname}'",
                "description": (td.description or "").strip(),
                "operationId": f"call_{alias}_{tname}",
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": td.input_schema or {"type": "object"}}}
                },
                "responses": {"200": {"description": "Successful Response", "content": {"application/json": {"schema": {"type": "object"}}}}},
                "x-mcp-tool-name": tname,
                "x-usage-hints": [
                    "Prefer POST with a body matching the tool schema.",
                    "If a field is not applicable, send empty string ''."
                ],
            }

            get_op = {
                "tags": ["tools"],
                "summary": f"Call MCP tool '{tname}' (GET)",
                "description": "If you cannot POST JSON, use ?args={...} as a JSON object.",
                "operationId": f"call_{alias}_{tname}_get",
                "parameters": [
                    {"name": "args", "in": "query", "required": False, "schema": {"type": "string"}},
                    {"name": "dryrun", "in": "query", "required": False, "schema": {"type": "boolean", "default": False}},
                ],
                "responses": {"200": {"description": "Successful Response", "content": {"application/json": {"schema": {"type": "object"}}}}},
                "x-mcp-tool-name": tname,
            }

            for d in REGISTRY.loaded.values():
                inst = d.instance
                if hasattr(inst, "tool_guidance"):
                    try:
                        extra = inst.tool_guidance(tname, td) or {}
                        if extra:
                            for op in (post_op, get_op):
                                for k, v in extra.items():
                                    if k in op and isinstance(op[k], list) and isinstance(v, list):
                                        op[k] = op[k] + v
                                    elif k in op and isinstance(op[k], dict) and isinstance(v, dict):
                                        op[k] = {**op[k], **v}
                                    else:
                                        op[k] = v
                    except Exception as e:
                        print(f"[drivers] tool_guidance error from {d.modpath} for {tname}: {e}", file=sys.stderr)

            if "post" not in paths[p]:
                paths[p]["post"] = post_op
            if "get" not in paths[p]:
                paths[p]["get"] = get_op

def openapi_extra_blocks() -> Dict[str, Any]:
    x_model_instructions = {
        "usage": [
            "Prefer POST to /mcp/tool/{tool} with a JSON body matching the tool schema.",
            "If a parameter isn't needed, send empty string ''.",
            "Use GET with ?args={...} only when you must pass query JSON.",
        ],
        "discovery": [
            "List tools: GET /{SERVER}/tools/list",
            "Per-tool schema: GET /{SERVER}/tool/{TOOL}/schema",
            "Per-tool examples: GET /{SERVER}/tool/{TOOL}/example",
            "Zero-argument test: GET /{SERVER}/tool/{TOOL}/try"
        ]
    }

    x_driver_data_summaries: Dict[str, Any] = {}
    for d in REGISTRY.loaded.values():
        inst = d.instance
        if hasattr(inst, "summarize_cache"):
            try:
                summary = inst.summarize_cache()
                if summary:
                    x_driver_data_summaries[getattr(inst, "name", d.modpath)] = summary
            except Exception as e:
                print(f"[drivers] summarize_cache failed for {d.modpath}: {e}", file=sys.stderr)

        if hasattr(inst, "extend_model_instructions"):
            try:
                inst.extend_model_instructions(x_model_instructions)
            except Exception as e:
                print(f"[drivers] extend_model_instructions failed for {d.modpath}: {e}", file=sys.stderr)

    x_mcp_tool_catalog: List[Dict[str, Any]] = []
    for alias, st in DISCOVERY.servers.items():
        for tname, td in st.tools.items():
            x_mcp_tool_catalog.append({
                "server": alias,
                "tool": tname,
                "description": td.description,
                "schema": td.input_schema or {"type": "object"},
                "requiredFields": (td.input_schema or {}).get("required", []),
                "examples": td.examples,
            })
    x_instance = {"id": os.getenv("INSTANCE_ID", "default")}
    x_cache = {"prefix": REGISTRY.cache.namespace()}

    return {
        "x-model-instructions": x_model_instructions,
        "x-driver-data-summaries": x_driver_data_summaries,
        "x-instance": x_instance,
        "x-cache": x_cache,
        "x-mcp-tool-catalog": x_mcp_tool_catalog,
        "x-mcp-prompts": {},
        "x-mcp-resources": {}
    }

@app.get("/{server}/tools/list", tags=["discovery"])
async def tools_list(server: str = Path(..., description="Server alias")):
    st = DISCOVERY.servers.get(server)
    if not st:
        raise HTTPException(404, f"Unknown server '{server}'")
    tools = []
    for tname, td in st.tools.items():
        tools.append({"name": tname, "description": td.description, "input_schema": td.input_schema})
    return {"server": server, "tools": tools}

_original_openapi = app.openapi
def custom_openapi():
    cache_ns = openapi_ns
    cache_key = "openapi_schema"
    ttl = openapi_ttl

    # Check cache first
    try:
        cached = REGISTRY.cache.get(cache_ns, cache_key)
        if cached:
            app.openapi_schema = cached
            return cached
    except Exception as e:
        print(f"[openapi] cache lookup failed: {e}", file=sys.stderr)

    # Compute the schema as before
    openapi_schema = _original_openapi()
    openapi_schema.update({**openapi_extra_blocks()})
    _inject_tool_operations(openapi_schema)
    # Merge augmentation data from Redis into each operation
    try:
        cache = REGISTRY.cache
        for driver_name, d in REGISTRY.loaded.items():
            # Augmentation entries should be cached under 'augmentations' per driver
            aug_entries = cache.get(driver_name, "augmentations")  # type: ignore[attr-defined]
            if not aug_entries:
                continue
            for row in aug_entries:
                path = row.get("path")
                method = (row.get("method") or "").lower()
                ext = (
                    row.get("extensions")
                    or row.get("extension")
                    or row.get("extensions_json")
                )
                if not path or not method or not isinstance(ext, dict):
                    continue
                op = (
                    openapi_schema
                    .get("paths", {})
                    .get(path, {})
                    .get(method)
                )
                if op and isinstance(op, dict):
                    _deep_merge(op, ext)
    except Exception as e:
        print(f"[openapi] failed to apply augmentation data: {e}", file=sys.stderr)
        
    try:
        REGISTRY.cache.set(cache_ns, cache_key, openapi_schema, ttl_seconds=ttl)
    except Exception as e:
        print(f"[openapi] failed to cache schema: {e}", file=sys.stderr)

    app.openapi_schema = openapi_schema
    return openapi_schema
app.openapi = custom_openapi

# =============================================================================
# Background thread for migrations and Redis synchronization
# =============================================================================

def run_migrations_and_sync():
    """Run database migrations and periodically synchronize data into Redis."""
    global MIGRATIONS_DONE, MIGRATION_ERROR

    # Run migrations on startup
    try:
        if MANIFEST and MANIFEST["spec"].get("migrator", {}).get("run", False):
            db_cfg = MANIFEST["spec"]["migrator"]["db"]
            os.environ["DB_HOST"] = db_cfg.get("host", "localhost")
            os.environ["DB_PORT"] = str(db_cfg.get("port", 5432))
            os.environ["DB_NAME"] = db_cfg.get("name", "postgres")
            os.environ["DB_SCHEMA"] = db_cfg.get("schema", "public")
            os.environ["DB_SSLMODE"] = db_cfg.get("sslmode", "prefer")
            os.environ["DB_USER_FILE"] = db_cfg.get("usernameFile", "")
            os.environ["DB_PASS_FILE"] = db_cfg.get("passwordFile", "")
            os.environ["DB_MIGRATIONS_DIR"] = db_cfg.get("migrationsDir", "/app/migrations")
            os.environ["DB_SEEDS_DIR"] = db_cfg.get("seedsDir", "/app/seeds")
            os.environ["RUN_MIGRATIONS"] = "1"
            Migrator().run_on_startup()
            MIGRATIONS_DONE = True
            MIGRATION_ERROR = ""
    except Exception as e:
        MIGRATIONS_DONE = False
        MIGRATION_ERROR = str(e)
        print(f"[migrator] FAILED: {e}", file=sys.stderr)

    # Periodically synchronize data into Redis
    sync_interval = int(os.getenv("REDIS_SYNC_INTERVAL", "300"))  # Default: 5 minutes
    while True:
        try:
            print("[sync] Synchronizing data into Redis...")
            # Add your synchronization logic here
            # Example: REGISTRY.cache.set("namespace", "key", {"data": "value"})
        except Exception as e:
            print(f"[sync] FAILED: {e}", file=sys.stderr)
        time.sleep(sync_interval)

# Start the background thread
threading.Thread(target=run_migrations_and_sync, daemon=True).start()

# Local dev
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8080")),
        reload=bool(os.getenv("RELOAD", "")),
    )

@app.on_event("startup")
async def initialize_app():
    """Initialize the app by running migrations and starting synchronization."""
    # Wait for migrations to complete
    global MIGRATIONS_DONE, MIGRATION_ERROR
    while not MIGRATIONS_DONE:
        if MIGRATION_ERROR:
            raise RuntimeError(f"Migration failed: {MIGRATION_ERROR}")
        await asyncio.sleep(1)

    # Start the periodic synchronization task
    asyncio.create_task(synchronize_data_to_redis())

# In app.py, replace the existing synchronize_data_to_redis() stub with:

async def synchronize_data_to_redis() -> None:
    """
    Periodically synchronize data to Redis.

    This function relies on each driver’s sync jobs (e.g. the Yugabyte driver)
    to refresh their data in Redis.  After waiting for the configured interval,
    it invalidates the cached OpenAPI document so that it will be rebuilt on
    the next request.
    """
    # Pull the interval from the manifest (spec.redis.default_ttl_seconds or fallback)
    # or use REDIS_SYNC_INTERVAL as a last resort.  This keeps configuration
    # entirely in the manifest if desired.
    sync_interval = int(
        (MANIFEST.get("spec", {}).get("redis") or {}).get("default_ttl_seconds", 300)
    )
    while True:
        try:
            print("[sync] Synchronizing data to Redis...")

            # Drivers' _run_job() coroutines are already running in the background
            # (scheduled by the DriverRegistry), so we don't fetch data here.
            # We simply wait for the interval and then invalidate the cached
            # OpenAPI document so that any new augmentation data is picked up.
            await asyncio.sleep(sync_interval)

            # Invalidate the cached /openapi.json so it will be rebuilt.
            try:
                REGISTRY.cache.delete(openapi_ns, "openapi_schema")
                print(f"[sync] Cleared cached OpenAPI document in namespace {openapi_ns}")
            except Exception as e:
                print(f"[sync] Failed to delete cache entry: {e}", file=sys.stderr)

        except Exception as e:
            # Catch exceptions in the loop to prevent it from exiting silently.
            print(f"[sync] FAILED: {e}", file=sys.stderr)
            await asyncio.sleep(sync_interval)

