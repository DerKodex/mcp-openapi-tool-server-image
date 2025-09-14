# mcp_openapi/drivers/yugabyte_driver.py
from __future__ import annotations
import asyncio
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field, ValidationError

from mcp_openapi.driver_base import CacheAPI, Driver

try:
    import psycopg
    from psycopg_pool import AsyncConnectionPool
except Exception as e:
    raise RuntimeError("psycopg[pool] is required for yugabyte_driver") from e

# -------------------- Config (Keycloak-like manifest) --------------------

class VaultK8sAuth(BaseModel):
    method: str = Field("kubernetes", const=True)
    mount: str = "kubernetes"
    role: str
    jwtFile: str = "/var/run/secrets/kubernetes.io/serviceaccount/token"

class VaultAppRoleAuth(BaseModel):
    method: str = Field("approle", const=True)
    mount: str = "approle"
    role_id: str
    secret_id: str

class VaultDatabaseCfg(BaseModel):
    mount: str = "database"
    role: str
    renewBefore: int = 120

class VaultSpec(BaseModel):
    address: str
    auth: VaultK8sAuth | VaultAppRoleAuth
    database: VaultDatabaseCfg

class YugabyteSpec(BaseModel):
    host: str
    port: int = 5433
    dbname: str
    sslmode: str = "prefer"
    options: Dict[str, Any] = Field(default_factory=dict)

class InitSpec(BaseModel):
    schema: Optional[str] = None
    createIfMissing: bool = True
    ddl: List[str] = Field(default_factory=list)

class PoolSpec(BaseModel):
    min: int = 1
    max: int = 5
    statementTimeoutMs: int = 60000

class CacheSpec(BaseModel):
    ttlSeconds: int = 600

class SyncJobSpec(BaseModel):
    name: str
    query: str
    key: str
    intervalSeconds: int = 60
    mode: str = "rows"            # rows | list | kv
    ttlSeconds: Optional[int] = None

class SyncSpec(BaseModel):
    jobs: List[SyncJobSpec] = Field(default_factory=list)

class YugabyteDriverConfig(BaseModel):
    apiVersion: str = "mcp.openapi/v1alpha1"
    kind: str = "YugabyteDriverConfig"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    spec: Dict[str, Any]

    def normalize(self) -> Dict[str, Any]:
        s = self.spec
        return {
            "vault": VaultSpec.model_validate(s["vault"]).model_dump(),
            "yugabyte": YugabyteSpec.model_validate(s["yugabyte"]).model_dump(),
            "init": InitSpec.model_validate(s.get("init", {})).model_dump(),
            "pool": PoolSpec.model_validate(s.get("pool", {})).model_dump(),
            "cache": CacheSpec.model_validate(s.get("cache", {})).model_dump(),
            "sync": SyncSpec.model_validate(s.get("sync", {})).model_dump(),
        }

# -------------------- Driver --------------------

class DriverImpl(Driver):
    name = "yugabyte"

    def __init__(self) -> None:
        self.cfg: Dict[str, Any] = {}
        self.cache: CacheAPI | None = None

        self._client = httpx.AsyncClient(timeout=20)
        self._vault_token: Optional[str] = None
        self._vault_token_exp: Optional[float] = None

        self._db_username: Optional[str] = None
        self._db_password: Optional[str] = None
        self._db_lease_id: Optional[str] = None
        self._db_lease_exp: Optional[float] = None

        self._pool: Optional[AsyncConnectionPool] = None
        self._lock = asyncio.Lock()
        self._tasks: List[asyncio.Task] = []

        self._instance_id = os.getenv("INSTANCE_ID", "default")

    # ---------- helpers ----------

    def _default_schema(self) -> str:
        base = f"mcp_{self._instance_id}"
        s = re.sub(r"[^a-zA-Z0-9_]", "_", base)
        if re.match(r"^[0-9]", s):
            s = "m_" + s
        return s[:63].lower()

    def _schema(self) -> str:
        init = self.cfg.get("init", {})
        return (init.get("schema") or self._default_schema()).lower()

    # ---------- lifecycle ----------

    async def load(self, config: Dict[str, Any], cache: CacheAPI) -> None:
        try:
            if "spec" in config:
                self.cfg = YugabyteDriverConfig.model_validate(config).normalize()
            else:
                _ = VaultSpec.model_validate(config["vault"])
                _ = YugabyteSpec.model_validate(config["yugabyte"])
                self.cfg = config
        except ValidationError as e:
            raise RuntimeError(f"Yugabyte driver config invalid: {e}") from e

        self.cfg.setdefault("init", {})
        self.cfg["init"].setdefault("schema", self._default_schema())

        self.cache = cache
        await self._ensure_pool()
        await self._ensure_schema()
        await self._start_sync_jobs()

    async def describe(self) -> Dict[str, Any]:
        return {
            "instanceId": self._instance_id,
            "redisPrefix": self.cache.namespace() if self.cache else None,
            "schema": self._schema(),
            "vault": {
                "address": self.cfg["vault"]["address"],
                "db_role": self.cfg["vault"]["database"]["role"],
                "db_mount": self.cfg["vault"]["database"]["mount"],
                "auth_method": self.cfg["vault"]["auth"]["method"],
            },
            "yugabyte": {k: self.cfg["yugabyte"][k] for k in ("host", "port", "dbname", "sslmode")},
            "pool": self.cfg["pool"],
            "syncJobs": [j if isinstance(j, dict) else j.model_dump() for j in self.cfg.get("sync", {}).get("jobs", [])],
            "state": {
                "have_token": bool(self._vault_token),
                "have_db_creds": bool(self._db_username and self._db_password),
                "pool_open": bool(self._pool and not self._pool.closed),
                "lease_exp_epoch": self._db_lease_exp,
                "running_tasks": len(self._tasks),
            }
        }

    async def enrich_openapi(self, openapi_schema: Dict[str, Any]) -> None:
        tables = self._cache_get("tables")
        openapi_schema.setdefault("x-instance", {})["id"] = self._instance_id
        openapi_schema.setdefault("x-cache", {})["prefix"] = self.cache.namespace() if self.cache else "memory"
        openapi_schema.setdefault("x-drivers", {})["yugabyte"] = {
            "schema": self._schema(),
            "capabilities": [
                "Vault dynamic credentials (renew/rotate)",
                "Per-instance schema bootstrap",
                "Background sync Yugabyte → Redis",
            ],
            "redisKeys": [
                f"{self.cache.namespace()}:{self.name}:tables" if self.cache else f"memory:{self.name}:tables",
                f"{self.cache.namespace()}:{self.name}:driver_kv" if self.cache else f"memory:{self.name}:driver_kv",
            ],
            "tables": tables or [],
        }

    def openapi_tags(self):
        return [{"name": "driver:yugabyte", "description": "Yugabyte/Redis integration"}]

    def tool_guidance(self, tool_name: str, td: Any) -> Dict[str, Any]:
        # Provide generic hints that help tools like kubectl-like MCP servers:
        return {"x-usage-hints": ["Use concrete resource names and namespaces when available."]}

    def extend_model_instructions(self, inst: Dict[str, Any]) -> None:
        inst.setdefault("yugabyte", {})["cachePrefix"] = self.cache.namespace() if self.cache else "memory"

    def summarize_cache(self) -> Dict[str, Any]:
        return {
            "redisPrefix": self.cache.namespace() if self.cache else "memory",
            "tables_known": len(self._cache_get("tables") or []),
        }

    async def close(self) -> None:
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except Exception:
                pass
        self._tasks.clear()
        try:
            if self._pool and not self._pool.closed:
                await self._pool.close()
        finally:
            await self._client.aclose()

    # ---------- small cache helper ----------

    def _cache_set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        assert self.cache is not None
        self.cache.set(self.name, key, value, ttl_seconds=ttl)

    def _cache_get(self, key: str) -> Optional[Any]:
        assert self.cache is not None
        return self.cache.get(self.name, key)

    # ---------- Vault (login/renew/creds) ----------

    async def _vault_login(self) -> None:
        v = self.cfg["vault"]
        addr: str = v["address"].rstrip("/")
        auth = v["auth"]
        if auth["method"] == "kubernetes":
            with open(auth.get("jwtFile", "/var/run/secrets/kubernetes.io/serviceaccount/token"), "r", encoding="utf-8") as f:
                jwt = f.read().strip()
            url = f"{addr}/v1/auth/{auth.get('mount','kubernetes')}/login"
            payload = {"jwt": jwt, "role": auth["role"]}
        else:
            url = f"{addr}/v1/auth/{auth.get('mount','approle')}/login"
            payload = {"role_id": auth["role_id"], "secret_id": auth["secret_id"]}
        r = await self._client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()["auth"]
        self._vault_token = data["client_token"]
        ttl = int(data.get("lease_duration", 3600))
        self._vault_token_exp = time.time() + max(30, ttl - 30)

    def _vault_headers(self) -> Dict[str, str]:
        if not self._vault_token:
            raise RuntimeError("Vault token missing")
        return {"X-Vault-Token": self._vault_token}

    async def _ensure_vault_token(self) -> None:
        if not self._vault_token or not self._vault_token_exp or time.time() >= self._vault_token_exp:
            await self._vault_login()

    async def _fetch_db_creds(self) -> None:
        await self._ensure_vault_token()
        v = self.cfg["vault"]
        addr: str = v["address"].rstrip("/")
        path = f"{v['database']['mount'].strip('/')}/creds/{v['database']['role']}"
        url = f"{addr}/v1/{path}"
        r = await self._client.get(url, headers=self._vault_headers())
        r.raise_for_status()
        data = r.json()
        self._db_username = data["data"]["username"]
        self._db_password = data["data"]["password"]
        self._db_lease_id = data.get("lease_id")
        ldur = int(data.get("lease_duration", 3600))
        self._db_lease_exp = time.time() + max(60, ldur - 30)

    async def _renew_lease(self) -> bool:
        if not self._db_lease_id:
            return False
        await self._ensure_vault_token()
        v = self.cfg["vault"]
        addr: str = v["address"].rstrip("/")
        url = f"{addr}/v1/sys/leases/renew"
        payload = {"lease_id": self._db_lease_id}
        r = await self._client.put(url, headers=self._vault_headers(), json=payload)
        if r.status_code >= 400:
            return False
        data = r.json()
        ldur = int(data.get("lease_duration", 3600))
        self._db_lease_exp = time.time() + max(60, ldur - 30)
        return True

    # ---------- Pool / DSN ----------

    def _dsn(self, user: str, pwd: str) -> str:
        y = self.cfg["yugabyte"]
        opts = y.get("options", {})
        params = {
            "host": y["host"],
            "port": y["port"],
            "dbname": y["dbname"],
            "sslmode": y.get("sslmode", "prefer"),
            "user": user,
            "password": pwd,
            **opts,
        }
        parts = [f"{k}={json.dumps(str(v))[1:-1]}" for k, v in params.items()]
        return " ".join(parts)

    async def _ensure_pool(self) -> None:
        async with self._lock:
            renew_before = int(self.cfg["vault"]["database"].get("renewBefore", 120))
            now = time.time()
            need_new = False

            if not self._db_username or not self._db_password or not self._db_lease_exp:
                await self._fetch_db_creds()
                need_new = True
            elif (self._db_lease_exp - now) <= max(30, renew_before):
                if not await self._renew_lease():
                    await self._fetch_db_creds()
                    need_new = True

            if need_new or not self._pool or self._pool.closed:
                if self._pool and not self._pool.closed:
                    await self._pool.close()
                dsn = self._dsn(self._db_username, self._db_password)  # type: ignore
                p = self.cfg["pool"]
                self._pool = AsyncConnectionPool(
                    conninfo=dsn,
                    min_size=int(p.get("min", 1)),
                    max_size=int(p.get("max", 5)),
                    kwargs={"options": f"-c statement_timeout={int(p.get('statementTimeoutMs', 60000))}"},
                    open=False,
                )
                await self._pool.open()

    # ---------- Schema bootstrap ----------

    async def _ensure_schema(self) -> None:
        init = self.cfg["init"]
        if not init.get("createIfMissing", True):
            return
        ddl = list(init.get("ddl", []))
        schema = self._schema()
        if not ddl:
            ddl = [
                f"create schema if not exists {schema}",
                f"""create table if not exists {schema}.driver_kv (
                        key text primary key,
                        value jsonb not null,
                        updated_at timestamptz not null default now()
                    )""",
                f"""create table if not exists {schema}.audit_log (
                        id bigserial primary key,
                        at timestamptz not null default now(),
                        actor text not null,
                        action text not null,
                        attrib jsonb not null default '{{}}'::jsonb
                    )"""
            ]
        await self._ensure_pool()
        assert self._pool is not None
        async with self._pool.connection() as aconn:
            async with aconn.transaction():
                for stmt in ddl:
                    await aconn.execute(stmt)

    # ---------- Sync jobs Yugabyte -> Redis ----------

    async def _start_sync_jobs(self) -> None:
        spec = self.cfg.get("sync", {}) or {}
        jobs = spec.get("jobs", [])
        for j in jobs:
            if not isinstance(j, SyncJobSpec):
                j = SyncJobSpec(**j)
            self._tasks.append(asyncio.create_task(self._run_job(j)))

        if not jobs:
            schema = self._schema()
            default = SyncJobSpec(
                name="tables",
                key="tables",
                mode="list",
                intervalSeconds=60,
                ttlSeconds=self.cfg["cache"].get("ttlSeconds", 600),
                query=(
                    "select table_schema||'.'||table_name as fqn "
                    "from information_schema.tables "
                    f"where table_schema in ('public','{schema}') order by 1"
                ),
            )
            self._tasks.append(asyncio.create_task(self._run_job(default)))

    async def _run_job(self, job: SyncJobSpec) -> None:
        while True:
            try:
                await self._ensure_pool()
                assert self._pool is not None
                async with self._pool.connection() as aconn:
                    rows = await aconn.execute(job.query)
                    result = await rows.fetchall() if hasattr(rows, "fetchall") else rows
                    if job.mode == "kv":
                        d: Dict[str, Any] = {}
                        for r in result:
                            k = r[0]
                            v = r[1]
                            try:
                                d[k] = v if isinstance(v, (dict, list)) else json.loads(v)
                            except Exception:
                                d[k] = v
                        payload = d
                    elif job.mode == "list":
                        payload = [r[0] for r in result]
                    else:
                        if hasattr(result, "columns"):
                            cols = [c.name for c in result.columns]
                            payload = [dict(zip(cols, r)) for r in result]
                        else:
                            payload = [list(r) for r in result]
                    self._cache_set(job.key, payload, ttl=job.ttlSeconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._cache_set(f"job:{job.name}:error", {"error": f"{type(e).__name__}: {e}", "ts": time.time()}, ttl=job.ttlSeconds or 300)
            await asyncio.sleep(max(5, job.intervalSeconds))
