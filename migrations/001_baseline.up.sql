CREATE TABLE mcp_openapi_augmentations (
  id            SERIAL PRIMARY KEY,
  path          TEXT NOT NULL,
  method        TEXT NOT NULL,
  summary       TEXT,
  description   TEXT,
  auth_required BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE mcp_openapi_usage_hints (
  augmentation_id INTEGER NOT NULL REFERENCES mcp_openapi_augmentations(id) ON DELETE CASCADE,
  hint            TEXT NOT NULL,
  id              SERIAL PRIMARY KEY
);

CREATE TABLE mcp_openapi_param_hints (
  augmentation_id INTEGER NOT NULL REFERENCES mcp_openapi_augmentations(id) ON DELETE CASCADE,
  name            TEXT NOT NULL,
  data_type       TEXT,
  allowed_values  TEXT[],   -- or a separate table if you want more structure
  example_value   TEXT,
  default_value   TEXT,
  description     TEXT,
  id              SERIAL PRIMARY KEY
);

CREATE TABLE mcp_openapi_examples (
  augmentation_id INTEGER NOT NULL REFERENCES mcp_openapi_augmentations(id) ON DELETE CASCADE,
  example_index   INTEGER NOT NULL, -- allows multiple examples per augmentation
  user_prompt     TEXT,
  args_json       JSONB NOT NULL,
  id              SERIAL PRIMARY KEY
);
