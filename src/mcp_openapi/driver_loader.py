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
    """
    Parse YAML text safely. Raises a RuntimeError with a helpful hint if PyYAML
    isn't installed or parsing fails.
    """
    try:
        import yaml  # type: ignore
        return yaml.safe_load(txt)
    except Exception as e:
        raise RuntimeError("YAML parsing failed (install PyYAML?): " + str(e)) from e


def _load_manifest_from_env() -> Optional[Dict[str, Any]]:
    """
    Try to load a manifest YAML if DRIVER_MANIFEST_PATH or MANIFEST_PATH is set,
    or from the default /etc/mcp/driver-manifest.yaml. Returns a dict or None.
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
            return _safe_load_yaml(f.read()) or {}
    except Exception:
        return None


def _coerce_json_or_yaml_path(s: str) -> Dict[str, Any]:
    """
    If 's' is a file path, load YAML/JSON from it; otherwise parse as JSON.
    Returns a dict (empty on failure).
    """
    try:
        if os.path.exists(s):
            with open(s, "r", encoding="utf-8") as f:
                txt = f.read()
            lower = s.lower()
            if lower.endswith((".yaml", ".yml")):
                return _safe_load_yaml(txt) or {}
            try:
                return json.loads(txt)
            except Exception:
                return _safe_load_yaml(txt) or {}
        # Not a file; parse as JSON
        return json.loads(s)
    except Exception:
        return {}


# -------------------- Redis-backed cache --------------------


class RedisCache(CacheAPI):
    """
    Redis-backed CacheAPI implementation.

    - URL comes from the manifest spec.redis.url, or is constructed from host/port/db/password,
      with REDIS_URL as a last resort.
    - Key space is namespaced as: <prefix>:<instance>:<ns>:<key>
    """

    def __init__(
        self,
        url: Optional[str] = None,
        default_ttl: Optional[int] = None,
        prefix: Optional[str] = None,
    ):
        try:
            import redis  # redis>=4
        except Exception as e:
            raise RuntimeError("RedisCache requires the 'redis' package (pip install redis)") from e

        # URL: explicit arg > env
        url_env = os.getenv("REDIS_URL")
        self._r = redis.Redis.from_url(url or url_env or "redis://localhost:6379/0", decode_responses=True)

        # TTL / prefix / instance
        self._default_ttl = int(
            default_ttl
            if default_ttl is not None
            else int(os.getenv("DRIVER_CACHE_DEFAULT_TTL", "600"))
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
            self._r.setex(self._k(ns, key), int(ttl), s)
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
    """
    Simple in-memory cache with TTL and soft max size.
    """

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
            else int(os.getenv("DRIVER_CACHE_DEFAULT_TTL", "600"))
        )
        self._max = int(
            max_entries
            if max_entries is not None
            else int(os.getenv("DRIVER_CACHE_MAX_ENTRIES", "10000"))
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
        # Drop expired first
        for k in [k for k, it in self._d.items() if it.exp and it.exp <= now]:
            self._d.pop(k, None)
        # If still too large, drop arbitrary items (FIFO-ish)
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
        exp = self._now() + int(ttl) if ttl and int(ttl) > 0 else None
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

    Cache + instanceId also prefer the manifest when present (spec.redis/spec.instanceId),
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
            # Surface to the rest of the process (e.g., caches, drivers)
            os.environ["INSTANCE_ID"] = self.instance_id

        # -------------------- Redis / Cache config --------------------
        # Prefer spec.redis; fallback to legacy spec.cache.registry.redis layout.
        redis_cfg = spec.get("redis") or ((spec.get("cache") or {}).get("registry") or {}).get("redis") or {}

        cache_kind = (redis_cfg.get("kind") or "").lower()  # "redis" | "memory" | ""
        # Prefer a fully-specified namespace when provided; then 'prefix'; then 'key_prefix'
        cache_prefix = (
            redis_cfg.get("namespace")
            or redis_cfg.get("prefix")
            or redis_cfg.get("key_prefix")
        )
        cache_default_ttl = redis_cfg.get("default_ttl_seconds") or redis_cfg.get("defaultTTL")
        cache_max_entries = redis_cfg.get("maxEntries")

        # Build URL from either 'url' or host/port/db/password
        cache_url = redis_cfg.get("url") or os.getenv("REDIS_URL")
        if not cache_url:
            host = redis_cfg.get("host")
            port = redis_cfg.get("port")
            db = redis_cfg.get("db")
            pwd = redis_cfg.get("password")
            if host and port is not None:
                auth = f":{pwd}@" if (pwd not in (None, "")) else ""
                dbpart = f"/{int(db)}" if db is not None else ""
                cache_url = f"redis://{auth}{host}:{int(port)}{dbpart}"

        # Choose cache backend
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
            # Optionally expose REDIS_URL to the rest of the process (for introspection)
            if cache_url and not os.getenv("REDIS_URL"):
                os.environ["REDIS_URL"] = cache_url
        else:
            self.cache = InMemoryCache(
                default_ttl=cache_default_ttl,
                prefix=cache_prefix or None,
                max_entries=cache_max_entries,
            )

    # -------------------- driver discovery --------------------

    def _read_config(self, blob: Any) -> Dict[str, Any]:
        if blob is None:
            return {}
        if isinstance(blob, dict):
            return blob
        if isinstance(blob, str):
            return _coerce_json_or_yaml_path(blob)
        return {}

    def _drivers_from_env_or_manifest(self) -> List[Dict[str, Any]]:
        """
        Returns a list of {"module": <str>, "config": <dict>} driver specs.
        Priority: DRIVERS env var (JSON) > manifest spec.drivers.
        """
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
            out: List[Dict[str, Any]] = []
            for d in drivers:
                if not isinstance(d, dict):
                    continue
                mod = d.get("module")
                if not mod:
                    # If someone only provided a 'name' without a python module path,
                    # skip it rather than trying to import a non-module.
                    continue
                out.append({"module": mod, "config": d.get("config")})
            return out

        return []

    # -------------------- lifecycle --------------------

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
                self.loaded[name] = LoadedDriver(modpath=modname, instance=instance, config=cfg or {})
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


# Singleton registry used by the app
REGISTRY = DriverRegistry()
