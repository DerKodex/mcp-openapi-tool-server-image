# mcp_openapi/driver_loader.py
from __future__ import annotations
import importlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from mcp_openapi.driver_base import CacheAPI

# -------------------- Redis-backed cache --------------------

class RedisCache(CacheAPI):
    def __init__(self):
        try:
            import redis  # redis>=4
        except Exception as e:
            raise RuntimeError("RedisCache requires 'redis' package") from e

        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._r = redis.Redis.from_url(url, decode_responses=True)
        self._default_ttl = int(os.getenv("DRIVER_CACHE_DEFAULT_TTL", "600"))
        base_prefix = os.getenv("DRIVER_CACHE_NS_PREFIX", "driver")
        instance = os.getenv("INSTANCE_ID", "default")
        self._prefix = f"{base_prefix}:{instance}"

    def _k(self, ns: str, key: str) -> str:
        return f"{self._prefix}:{ns}:{key}"

    def get(self, ns: str, key: str):
        val = self._r.get(self._k(ns, key))
        if val is None:
            return None
        try:
            return json.loads(val)
        except Exception:
            return val

    def set(self, ns: str, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        s = json.dumps(value)
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        if ttl and ttl > 0:
            self._r.setex(self._k(ns, key), ttl, s)
        else:
            self._r.set(self._k(ns, key), s)

    def delete(self, ns: str, key: str) -> None:
        self._r.delete(self._k(ns, key))

    def namespace(self) -> str:
        return self._prefix

# -------------------- In-memory fallback --------------------

from dataclasses import dataclass

@dataclass
class _Item:
    value: Any
    exp: Optional[float]

class InMemoryCache(CacheAPI):
    def __init__(self):
        self._d: Dict[Tuple[str, str], _Item] = {}
        self._default_ttl = int(os.getenv("DRIVER_CACHE_DEFAULT_TTL", "600"))
        self._max = int(os.getenv("DRIVER_CACHE_MAX_ENTRIES", "10000"))
        base_prefix = os.getenv("DRIVER_CACHE_NS_PREFIX", "driver")
        instance = os.getenv("INSTANCE_ID", "default")
        self._prefix = f"{base_prefix}:{instance}"

    def _now(self) -> float:
        return time.time()

    def _prune(self):
        if len(self._d) <= self._max:
            return
        now = self._now()
        for k in [k for k, it in self._d.items() if it.exp and it.exp <= now]:
            self._d.pop(k, None)
        while len(self._d) > self._max:
            self._d.pop(next(iter(self._d)))

    def get(self, ns: str, key: str):
        it = self._d.get((f"{self._prefix}:{ns}", key))
        if not it:
            return None
        if it.exp and it.exp <= self._now():
            self._d.pop((f"{self._prefix}:{ns}", key), None)
            return None
        return it.value

    def set(self, ns: str, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        exp = self._now() + ttl if ttl > 0 else None
        self._d[(f"{self._prefix}:{ns}", key)] = _Item(value=value, exp=exp)
        self._prune()

    def delete(self, ns: str, key: str) -> None:
        self._d.pop((f"{self._prefix}:{ns}", key), None)

    def namespace(self) -> str:
        return self._prefix

# -------------------- driver loader --------------------

@dataclass
class LoadedDriver:
    modpath: str
    instance: Any
    config: Dict[str, Any] = field(default_factory=dict)

class DriverRegistry:
    """
    Loads drivers from env DRIVERS JSON:
      DRIVERS='[
        {"module":"mcp_openapi.drivers.yugabyte_driver","config":"/config/yugabyte-driver.yaml"}
      ]'
    """
    def __init__(self):
        if os.getenv("REDIS_URL"):
            self.cache: CacheAPI = RedisCache()
        else:
            self.cache: CacheAPI = InMemoryCache()
        self.loaded: Dict[str, LoadedDriver] = {}
        self.instance_id = os.getenv("INSTANCE_ID", "default")

    def _read_config(self, blob: Any) -> Dict[str, Any]:
        if blob is None:
            return {}
        if isinstance(blob, dict):
            return blob
        if isinstance(blob, str):
            if os.path.exists(blob):
                with open(blob, "r", encoding="utf-8") as f:
                    txt = f.read()
                try:
                    import yaml
                    return yaml.safe_load(txt)
                except Exception:
                    return json.loads(txt)
            return json.loads(blob)
        return {}

    async def load_from_env(self) -> Dict[str, Any]:
        raw = os.getenv("DRIVERS", "[]")
        try:
            arr = json.loads(raw)
        except Exception:
            arr = []
        summary = []
        for spec in arr:
            modname = spec.get("module")
            conf_blob = spec.get("config")
            if not modname:
                continue
            try:
                mod = importlib.import_module(modname)
                instance = getattr(mod, "DriverImpl")()
                cfg = self._read_config(conf_blob)
                await instance.load(cfg, self.cache)
                name = getattr(instance, "name", modname)
                self.loaded[name] = LoadedDriver(modpath=modname, instance=instance, config=cfg)
                summary.append({"name": name, "module": modname, "loaded": True})
            except Exception as e:
                summary.append({"module": modname, "loaded": False, "error": f"{type(e).__name__}: {e}"})
        return {"instanceId": self.instance_id, "drivers": summary, "cachePrefix": self.cache.namespace()}

    async def close_all(self):
        for d in list(self.loaded.values()):
            try:
                await d.instance.close()
            except Exception:
                pass

    async def enrich_openapi(self, openapi_schema: Dict[str, Any]) -> None:
        openapi_schema.setdefault("x-instance", {})["id"] = self.instance_id
        openapi_schema.setdefault("x-cache", {})["prefix"] = self.cache.namespace()
        for d in self.loaded.values():
            try:
                await d.instance.enrich_openapi(openapi_schema)
            except Exception:
                pass

    async def describe_all(self) -> Dict[str, Any]:
        out = []
        for name, d in self.loaded.items():
            try:
                out.append({"name": name, "module": d.modpath, "describe": await d.instance.describe()})
            except Exception as e:
                out.append({"name": name, "module": d.modpath, "error": f"{type(e).__name__}: {e}"})
        return {"instanceId": self.instance_id, "cachePrefix": self.cache.namespace(), "drivers": out}

# singleton
REGISTRY = DriverRegistry()
