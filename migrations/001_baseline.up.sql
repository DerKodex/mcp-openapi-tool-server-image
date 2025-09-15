-- Assumes the app sets search_path to your schema (the migrator does this).
-- Idempotent: IF NOT EXISTS everywhere.

-- Helper: updated_at touch trigger
CREATE OR REPLACE FUNCTION touch_updated_at()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  NEW.updated_at := now();
  RETURN NEW;
END$$;

-- 1) Generic KV (config, small blobs)
CREATE TABLE IF NOT EXISTS mcp_kv (
  k         text PRIMARY KEY,
  v         jsonb NOT NULL,
  updated_at timestamptz NOT NULL DEFAULT now()
);
DROP TRIGGER IF EXISTS trg_mcp_kv_touch ON mcp_kv;
CREATE TRIGGER trg_mcp_kv_touch
BEFORE UPDATE ON mcp_kv
FOR EACH ROW EXECUTE FUNCTION touch_updated_at();

-- 2) Cache with TTL (drivers + server can store denormalized views)
CREATE TABLE IF NOT EXISTS mcp_cache (
  k           text PRIMARY KEY,
  v           jsonb NOT NULL,
  expires_at  timestamptz NOT NULL,
  updated_at  timestamptz NOT NULL DEFAULT now()
);
DROP TRIGGER IF EXISTS trg_mcp_cache_touch ON mcp_cache;
CREATE TRIGGER trg_mcp_cache_touch
BEFORE UPDATE ON mcp_cache
FOR EACH ROW EXECUTE FUNCTION touch_updated_at();

-- 3) Tool catalog mirror (what the server exposes at runtime)
CREATE TABLE IF NOT EXISTS mcp_tool_catalog (
  tool_name     text PRIMARY KEY,
  description   text DEFAULT '',
  input_schema  jsonb NOT NULL DEFAULT '{}'::jsonb,
  examples      jsonb NOT NULL DEFAULT '[]'::jsonb,
  updated_at    timestamptz NOT NULL DEFAULT now()
);
DROP TRIGGER IF EXISTS trg_mcp_tool_catalog_touch ON mcp_tool_catalog;
CREATE TRIGGER trg_mcp_tool_catalog_touch
BEFORE UPDATE ON mcp_tool_catalog
FOR EACH ROW EXECUTE FUNCTION touch_updated_at();

-- 4) Sync cursors & job state (e.g., background fetchers)
CREATE TABLE IF NOT EXISTS mcp_sync_state (
  source        text PRIMARY KEY,         -- e.g., 'k8s:pods', 'driver:yugabyte'
  last_cursor   text DEFAULT NULL,
  last_run_at   timestamptz DEFAULT NULL,
  next_run_at   timestamptz DEFAULT NULL,
  meta          jsonb NOT NULL DEFAULT '{}'::jsonb,
  updated_at    timestamptz NOT NULL DEFAULT now()
);
DROP TRIGGER IF EXISTS trg_mcp_sync_state_touch ON mcp_sync_state;
CREATE TRIGGER trg_mcp_sync_state_touch
BEFORE UPDATE ON mcp_sync_state
FOR EACH ROW EXECUTE FUNCTION touch_updated_at();

-- 5) Lightweight locks (cooperative, not advisory)
CREATE TABLE IF NOT EXISTS mcp_locks (
  name   text PRIMARY KEY,
  owner  text NOT NULL,
  until  timestamptz NOT NULL,
  meta   jsonb NOT NULL DEFAULT '{}'::jsonb
);

-- 6) Audit log (append-only)
CREATE TABLE IF NOT EXISTS mcp_audit (
  id        bigserial PRIMARY KEY,
  category  text NOT NULL,           -- e.g., 'driver', 'api', 'auth'
  action    text NOT NULL,           -- e.g., 'sync', 'call_tool', 'cache_write'
  subject   text DEFAULT NULL,       -- free-form identity or resource key
  payload   jsonb NOT NULL DEFAULT '{}'::jsonb,
  at        timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_mcp_audit_at ON mcp_audit (at DESC);
CREATE INDEX IF NOT EXISTS idx_mcp_audit_cat_action ON mcp_audit (category, action);

-- 7) Event stream (simple queue/event bus)
CREATE TABLE IF NOT EXISTS mcp_events (
  id         bigserial PRIMARY KEY,
  type       text NOT NULL,          -- e.g., 'refresh', 'reconcile', 'notice'
  data       jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  processed  boolean NOT NULL DEFAULT false
);
CREATE INDEX IF NOT EXISTS idx_mcp_events_created ON mcp_events (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_mcp_events_processed ON mcp_events (processed, created_at);

-- 8) Optional driver registry snapshot (what modules loaded last)
CREATE TABLE IF NOT EXISTS mcp_drivers (
  name       text PRIMARY KEY,       -- module path, e.g. 'mcp_openapi.drivers.yugabyte_driver'
  version    text DEFAULT NULL,
  config     jsonb NOT NULL DEFAULT '{}'::jsonb,
  updated_at timestamptz NOT NULL DEFAULT now()
);
DROP TRIGGER IF EXISTS trg_mcp_drivers_touch ON mcp_drivers;
CREATE TRIGGER trg_mcp_drivers_touch
BEFORE UPDATE ON mcp_drivers
FOR EACH ROW EXECUTE FUNCTION touch_updated_at();
