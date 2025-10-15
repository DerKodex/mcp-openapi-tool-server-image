-- Optional: start clean
-- DROP TABLE IF EXISTS mcp_openapi_examples        CASCADE;
-- DROP TABLE IF EXISTS mcp_openapi_param_hints     CASCADE;
-- DROP TABLE IF EXISTS mcp_openapi_usage_hints     CASCADE;
-- DROP TABLE IF EXISTS mcp_openapi_augmentations   CASCADE;

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
  id              SERIAL PRIMARY KEY,
  -- prevent duplicate hints per augmentation (optional but recommended)
  CONSTRAINT uq_usage_hints_aug_hint UNIQUE (augmentation_id, hint)
);

CREATE TABLE mcp_openapi_param_hints (
  augmentation_id INTEGER NOT NULL REFERENCES mcp_openapi_augmentations(id) ON DELETE CASCADE,
  name            TEXT NOT NULL,
  data_type       TEXT,
  allowed_values  TEXT[],   -- or a separate table if you want more structure
  example_value   TEXT,
  default_value   TEXT,
  description     TEXT,
  id              SERIAL PRIMARY KEY,
  -- enable ON CONFLICT (augmentation_id, name)
  CONSTRAINT uq_param_hints_aug_name UNIQUE (augmentation_id, name)
);

CREATE TABLE mcp_openapi_examples (
  augmentation_id INTEGER NOT NULL REFERENCES mcp_openapi_augmentations(id) ON DELETE CASCADE,
  example_index   INTEGER NOT NULL, -- allows multiple examples per augmentation
  user_prompt     TEXT,
  args_json       JSONB NOT NULL,
  id              SERIAL PRIMARY KEY,
  -- enable ON CONFLICT (augmentation_id, example_index)
  CONSTRAINT uq_examples_aug_idx UNIQUE (augmentation_id, example_index)
);

-- (Optional) helpful indexes (FK columns are often queried)
CREATE INDEX IF NOT EXISTS idx_usage_hints_aug_id  ON mcp_openapi_usage_hints(augmentation_id);
CREATE INDEX IF NOT EXISTS idx_param_hints_aug_id  ON mcp_openapi_param_hints(augmentation_id);
CREATE INDEX IF NOT EXISTS idx_examples_aug_id     ON mcp_openapi_examples(augmentation_id);
