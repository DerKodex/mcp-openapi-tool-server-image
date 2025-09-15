-- JSONB indexes (no extensions required; safe on Yugabyte)

-- GIN indexes for searching JSON content
CREATE INDEX IF NOT EXISTS idx_mcp_kv_v_gin          ON mcp_kv USING GIN (v);
CREATE INDEX IF NOT EXISTS idx_mcp_cache_v_gin       ON mcp_cache USING GIN (v);
CREATE INDEX IF NOT EXISTS idx_mcp_tool_schema_gin   ON mcp_tool_catalog USING GIN (input_schema);
CREATE INDEX IF NOT EXISTS idx_mcp_tool_examples_gin ON mcp_tool_catalog USING GIN (examples);
CREATE INDEX IF NOT EXISTS idx_mcp_sync_meta_gin     ON mcp_sync_state USING GIN (meta);
CREATE INDEX IF NOT EXISTS idx_mcp_drivers_cfg_gin   ON mcp_drivers USING GIN (config);

-- Cache expiry helper
CREATE INDEX IF NOT EXISTS idx_mcp_cache_expiry ON mcp_cache (expires_at);
