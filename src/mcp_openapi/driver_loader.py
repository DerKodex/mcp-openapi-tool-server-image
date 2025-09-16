# mcp_openapi/driver_loader.py
from __future__ import annotations

import importlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List

from mcp_openapi.driver_base import CacheAPI

# -------------------- helpers --------------------

def _safe_load_yaml(txt: str) -> Any:
    try:
        import yaml  # type: ignore
        return yaml.safe_load(txt)
    except Exception as e:
        raise RuntimeError("YAML parsing failed (install PyYAML?): " + str(e)) from e

def _load_manifest_from_env() -> Optional[Dict[str, Any]]:
    """
    Try to load a manifest YAML if DRIVER_MANIFEST_PATH or MANIFEST_PATH is set.
    Returns a dict or None.
    """
    path = (
        os.getenv("DRIVER_MANIFEST_PATH")
        or os.getenv("MANIFEST_PATH")
        or "/etc/mcp/driver-manifest.yaml"
    )
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return _safe_load_yaml(f.read())
    except Exception:
        return None

def _coerce_json_or_yaml_path(s: str) -> Dict[str, Any]:
    """
    If s is a file path, load YAML/JSON from it; otherwise parse as JSON.
    """
    if os.path.exists(s):
        with open(s, "r", encoding="utf-8") as f:
            txt = f.read()
        # Guess YAML by extension, else try JSON then YAML
        lower = s.lower()
        if lower.endswith((".yaml", ".yml")):
            return _safe_load_yaml(txt) or {}
        try:
            return json.loads(txt)
        except Exception:
            return _safe_load_yaml(txt) or {}
    # Not a file; parse as JSON
    return json.loads(s)

# -------------------- Redis-backed cache --------------------

class RedisCache(CacheAPI):
    def __init__(
        self,
        url: Optional[str] = None,
        default_ttl: Optional[int] = None,
        prefix: Optional[str] = None,
    ):
        try:
            import redis  # redis>=4
        except Exception as e:
            raise RuntimeError("RedisCache requires the 'redis' package") from e

        url_env = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._r = redis.Redis.from_url(url or url_env, decode_responses=True)

        self._default_ttl = int(
            default_ttl
            if default_ttl is not None
            else os.getenv("DRIVER_CACHE_DEFAULT_TTL", "600")
        )
        base_prefix = os.getenv("DRIVER_CACHE_NS_PREFIX", "driver")
        instance = os.getenv("INSTANCE_ID", "default")
        self._prefix = prefix or f"{base_prefix}:{instance}"

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

@dataclass
class _Item:
    value: Any
    exp: Optional[float]

class InMemoryCache(CacheAPI):
    def __init__(
        self,
        default_ttl: Optional[int] = None,
        prefix: Optional[str] = None,
        max_entries: Optional[int] = None,
    ):
        self._d: Dict[Tuple[str, str], _Item] = {}
        self._default_ttl = int(
            default_ttl
            if default_ttl is not None
            else os.getenv("DRIVER_CACHE_DEFAULT_TTL", "600")
        )
        self._max = int(
            max_entries
            if max_entries is not None
            else os.getenv("DRIVER_CACHE_MAX_ENTRIES", "10000")
        )
        base_prefix = os.getenv("DRIVER_CACHE_NS_PREFIX", "driver")
        instance = os.getenv("INSTANCE_ID", "default")
        self._prefix = prefix or f"{base_prefix}:{instance}"

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
    Loads drivers from:
      1) DRIVERS environment variable (JSON array of {module, config})
      2) If absent, from the manifest 'spec.drivers' (when DRIVER_MANIFEST_PATH/MANIFEST_PATH is set)

    Cache + instanceId also prefer the manifest when present (spec.cache, spec.instanceId),
    else fall back to environment variables.
    """
    def __init__(self):
        self.loaded: Dict[str, LoadedDriver] = {}

        # Read manifest (optional)
        manifest = _load_manifest_from_env()
        spec = (manifest or {}).get("spec") or {}

        # Instance id (manifest overrides env if env wasn’t set)
        env_instance = os.getenv("INSTANCE_ID")
        man_instance = spec.get("instanceId")
        if env_instance:
            self.instance_id = env_instance
        else:
            self.instance_id = man_instance or "default"
            os.environ["INSTANCE_ID"] = self.instance_id  # surface to others

        # Cache config (manifest optional)
        cache_cfg = spec.get("cache") or {}
        cache_kind = (cache_cfg.get("kind") or "").lower()  # "redis" | "memory" | ""
        cache_prefix = cache_cfg.get("prefix")
        cache_default_ttl = cache_cfg.get("defaultTTL")
        cache_max_entries = cache_cfg.get("maxEntries")
        cache_url = None

        # Support nested "redis: {url: ...}" or top-level "url"
        if isinstance(cache_cfg.get("redis"), dict):
            cache_url = cache_cfg["redis"].get("url")
        cache_url = cache_url or cache_cfg.get("url") or os.getenv("REDIS_URL")

        # Choose cache backend:
        use_redis = bool(cache_url) or bool(os.getenv("REDIS_URL"))
        if cache_kind == "memory":
            use_redis = False
        elif cache_kind == "redis":
            use_redis = True

        if use_redis:
            self.cache = RedisCache(
                url=cache_url,
                default_ttl=cache_default_ttl,
                prefix=cache_prefix or None,
            )
            # Make REDIS_URL visible to anyone else (introspection endpoints, etc.)
            if cache_url and not os.getenv("REDIS_URL"):
                os.environ["REDIS_URL"] = cache_url
        else:
            self.cache = InMemoryCache(
                default_ttl=cache_default_ttl,
                prefix=cache_prefix or None,
                max_entries=cache_max_entries,
            )

    def _read_config(self, blob: Any) -> Dict[str, Any]:
        if blob is None:
            return {}
        if isinstance(blob, dict):
            return blob
        if isinstance(blob, str):
            return _coerce_json_or_yaml_path(blob)
        return {}

    def _drivers_from_env_or_manifest(self) -> List[Dict[str, Any]]:
        # 1) DRIVERS env (JSON)
        raw = os.getenv("DRIVERS")
        if raw:
            try:
                arr = json.loads(raw)
                if isinstance(arr, list):
                    return arr
            except Exception:
                pass

        # 2) Fallback: manifest spec.drivers
        manifest = _load_manifest_from_env()
        if manifest and isinstance(manifest, dict):
            spec = manifest.get("spec") or {}
            drivers = spec.get("drivers") or []
            # Normalize: allow entries with {"name": "...", "module": "...", "config": ...}
            out: List[Dict[str, Any]] = []
            for d in drivers:
                if not isinstance(d, dict):
                    continue
                mod = d.get("module") or d.get("name")
                if not mod:
                    continue
                out.append({"module": mod, "config": d.get("config")})
            return out

        return []

    async def load_from_env(self) -> Dict[str, Any]:
        """
        (Re)loads drivers using DRIVERS env or manifest fallback.
        Closes previously loaded ones first.
        """
        await self.close_all()
        self.loaded.clear()

        specs = self._drivers_from_env_or_manifest()
        summary = []
        for spec in specs:
            modname = spec.get("module")
            conf_blob = spec.get("config")
            if not modname:
                continue
            try:
                mod = importlib.import_module(modname)
                instance = getattr(mod, "DriverImpl")()
                cfg = self._read_config(conf_blob)
                await instance.load(cfg, self.cache)
                name = getattr(instance, "name", None) or modname
                self.loaded[name] = LoadedDriver(modpath=modname, instance=instance, config=cffg if (cffg := cfg) else {})
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
