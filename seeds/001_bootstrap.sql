-- Safe bootstrap data; re-runnable thanks to ON CONFLICT.

INSERT INTO mcp_kv (k, v)
VALUES ('server.banner', '{"msg":"MCP OpenAPI server is alive"}')
ON CONFLICT (k) DO UPDATE SET v = EXCLUDED.v;

INSERT INTO mcp_tool_catalog (tool_name, description, input_schema, examples)
VALUES ('_placeholder', 'placeholder tool', '{}'::jsonb, '[]'::jsonb)
ON CONFLICT (tool_name) DO NOTHING;
