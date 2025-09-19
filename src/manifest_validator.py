# manifest_validator.py

import yaml
from typing import List, Optional
from pydantic import BaseModel, ValidationError, Field

class Metadata(BaseModel):
    name: str

class RedisConfig(BaseModel):
    url: str
    namespace: str
    key_prefix: str = Field(..., alias="key_prefix")
    default_ttl_seconds: int

class OpenapiCacheConfig(BaseModel):
    namespace: str
    ttlSeconds: int

class TransportStdioConfig(BaseModel):
    enabled: bool
    force: bool
    initTimeoutSeconds: int
    preflight: bool
    preflightConfig: bool
    extraArgs: List[str]

class TransportServerConfig(BaseModel):
    alias: str
    mode: str
    cmd: List[str]
    env: Optional[dict] = None
    cwd: Optional[str] = None

class TransportConfig(BaseModel):
    prefer: str
    stdio: TransportStdioConfig
    rpc: dict
    forward: dict
    servers: List[TransportServerConfig]

class CorsConfig(BaseModel):
    allow_origins: List[str]
    allow_methods: List[str]
    allow_headers: List[str]

class DbConfig(BaseModel):
    host: str
    port: int
    name: str
    schema: str
    sslmode: str
    usernameFile: Optional[str] = None
    passwordFile: Optional[str] = None
    migrationsDir: Optional[str] = None
    seedsDir: Optional[str] = None

class MigratorConfig(BaseModel):
    run: bool
    db: DbConfig

class DriverEntry(BaseModel):
    name: str
    module: str
    config: dict  # you can model this further if desired

class Spec(BaseModel):
    instanceId: str
    redis: RedisConfig
    openapiCache: Optional[OpenapiCacheConfig] = None
    transport: TransportConfig
    cors: CorsConfig
    migrator: MigratorConfig
    drivers: List[DriverEntry]

class DriverManifestConfig(BaseModel):
    apiVersion: str
    kind: str
    metadata: Metadata
    spec: Spec

def validate_manifest_yaml(yaml_text: str) -> None:
    """
    Parse and validate a manifest YAML string.
    Raises ValidationError on failure.
    """
    data = yaml.safe_load(yaml_text)
    DriverManifestConfig.parse_obj(data)

# Example usage:
# with open("driver-manifest.yaml") as f:
#     yaml_text = f.read()
# try:
#     validate_manifest_yaml(yaml_text)
#     print("Manifest is valid.")
# except ValidationError as exc:
#     print(f"Manifest validation failed: {exc}")
