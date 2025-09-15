import os, re, hashlib, importlib.util, sys, glob
from datetime import datetime
from typing import List, Tuple, Optional
import psycopg

FILENAME_RE = re.compile(r"^(\d{3,})_(.+?)\.(up\.sql|py)$")

class Migrator:
    def __init__(self):
        # Connection from env/files (works with Vault agent files)
        self.host     = os.getenv("DB_HOST", "localhost")
        self.port     = int(os.getenv("DB_PORT", "5432"))
        self.database = os.getenv("DB_NAME", "postgres")
        self.schema   = os.getenv("DB_SCHEMA", "public")
        self.sslmode  = os.getenv("DB_SSLMODE", "prefer")

        # User/pass can be plaintext or file paths (Vault pattern)
        self.user     = self._read_val("DB_USER", "DB_USER_FILE")
        self.password = self._read_val("DB_PASSWORD", "DB_PASS_FILE")

        # Paths
        self.migrations_dir = os.getenv("DB_MIGRATIONS_DIR", "/app/migrations")
        self.seeds_dir      = os.getenv("DB_SEEDS_DIR", "/app/seeds")

        # Behavior knobs
        self.reset                   = os.getenv("DB_RESET", "0").lower() in ("1","true","yes")
        self.allow_changed_checksums = os.getenv("DB_MIGRATION_ALLOW_CHANGED", "0").lower() in ("1","true","yes")

        self.mig_table  = f'{self.schema}.mcp_migrations'
        self.seed_table = f'{self.schema}.mcp_seeds'

    def _read_val(self, env_key: str, file_env_key: str) -> Optional[str]:
        if os.getenv(env_key): 
            return os.getenv(env_key)
        f = os.getenv(file_env_key)
        if f and os.path.exists(f):
            with open(f, "r", encoding="utf-8") as fh:
                return fh.read().strip()
        return None

    # -------------------- public API --------------------
    def run_on_startup(self):
        if os.getenv("RUN_MIGRATIONS", "0").lower() not in ("1","true","yes"):
            return  # disabled
        with self._connect() as conn:
            conn.execute("SET client_min_messages TO WARNING;")
            self._ensure_schema(conn)
            if self.reset:
                self._drop_schema(conn)
                self._ensure_schema(conn)

            self._ensure_meta_tables(conn)

            applied = self._load_applied(conn)
            todo = self._discover_migrations(self.migrations_dir)

            for version, name, kind, path in todo:
                checksum = self._file_sha256(path)
                prev = applied.get(version)
                if prev:
                    # already applied
                    if prev["checksum"] != checksum and not self.allow_changed_checksums:
                        raise RuntimeError(
                          f"Checksum changed for migration {version}_{name}!\n"
                          f" was={prev['checksum']}\n now={checksum}\n"
                          " (set DB_MIGRATION_ALLOW_CHANGED=1 to override)"
                        )
                    continue

                # apply within a transaction
                with conn.transaction():
                    if kind == "up.sql":
                        sql = open(path, "r", encoding="utf-8").read()
                        conn.execute(sql)
                    elif kind == "py":
                        self._exec_python_migration(conn, path)
                    else:
                        raise RuntimeError(f"Unknown migration kind: {kind}")

                    conn.execute(
                        f"INSERT INTO {self.mig_table}(version,name,checksum,executed_at) VALUES(%s,%s,%s,now());",
                        (version, name, checksum)
                    )
                print(f"[migrator] applied {version}_{name}.{kind}")

            # seeds (best-effort, only once each)
            if os.path.isdir(self.seeds_dir):
                seeded = self._load_seeded(conn)
                for spath in sorted(glob.glob(os.path.join(self.seeds_dir, "*"))):
                    sname = os.path.basename(spath)
                    if sname in seeded:
                        continue
                    with conn.transaction():
                        if sname.endswith(".sql"):
                            sql = open(spath, "r", encoding="utf-8").read()
                            conn.execute(sql)
                        elif sname.endswith(".py"):
                            self._exec_python_seed(conn, spath)
                        else:
                            continue
                        conn.execute(
                            f"INSERT INTO {self.seed_table}(name,checksum,executed_at) VALUES(%s,%s,now());",
                            (sname, self._file_sha256(spath))
                        )
                    print(f"[migrator] seeded {sname}")

    # -------------------- internals --------------------
    def _connect(self):
        dsn = (
            f"host={self.host} port={self.port} dbname={self.database} "
            f"user={self.user} password={self.password} sslmode={self.sslmode}"
        )
        # autocommit False by default; we manage transactions
        conn = psycopg.connect(dsn)
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {self._qi(self.schema)};")
        conn.execute(f"SET search_path TO {self._qi(self.schema)};")
        return conn

    def _qi(self, ident: str) -> str:
        return '"' + ident.replace('"', '""') + '"'

    def _ensure_schema(self, conn):
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {self._qi(self.schema)};")
        conn.execute(f"SET search_path TO {self._qi(self.schema)};")

    def _drop_schema(self, conn):
        print(f"[migrator] dropping schema {self.schema} (CASCADE)")
        conn.execute(f"DROP SCHEMA IF EXISTS {self._qi(self.schema)} CASCADE;")
        conn.execute(f"CREATE SCHEMA {self._qi(self.schema)};")
        conn.execute(f"SET search_path TO {self._qi(self.schema)};")

    def _ensure_meta_tables(self, conn):
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.mig_table}(
              id bigserial PRIMARY KEY,
              version integer NOT NULL UNIQUE,
              name text NOT NULL,
              checksum text NOT NULL,
              executed_at timestamptz NOT NULL DEFAULT now()
            );
        """)
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.seed_table}(
              id bigserial PRIMARY KEY,
              name text NOT NULL UNIQUE,
              checksum text NOT NULL,
              executed_at timestamptz NOT NULL DEFAULT now()
            );
        """)

    def _discover_migrations(self, root: str) -> List[Tuple[int,str,str,str]]:
        if not os.path.isdir(root):
            return []
        found: List[Tuple[int,str,str,str]] = []
        for fname in os.listdir(root):
            m = FILENAME_RE.match(fname)
            if not m:
                continue
            version = int(m.group(1))
            name    = m.group(2)
            kind    = m.group(3)   # up.sql | py
            found.append((version, name, kind, os.path.join(root, fname)))
        found.sort(key=lambda t: t[0])
        return found

    def _file_sha256(self, path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _load_applied(self, conn) -> dict:
        rows = conn.execute(f"SELECT version, name, checksum FROM {self.mig_table} ORDER BY version;").fetchall()
        return { int(r[0]): {"name": r[1], "checksum": r[2]} for r in rows }

    def _load_seeded(self, conn) -> set:
        rows = conn.execute(f"SELECT name FROM {self.seed_table};").fetchall()
        return { r[0] for r in rows }

    def _exec_python_migration(self, conn, path: str):
        mod = self._load_module(path)
        if not hasattr(mod, "up"):
            raise RuntimeError(f"{path} missing up(conn) function")
        mod.up(conn)

    def _exec_python_seed(self, conn, path: str):
        mod = self._load_module(path)
        fn = getattr(mod, "seed", None) or getattr(mod, "up", None)
        if not fn:
            raise RuntimeError(f"{path} missing seed(conn) or up(conn)")
        fn(conn)

    def _load_module(self, path: str):
        name = "mig_" + re.sub(r"[^a-zA-Z0-9_]", "_", os.path.basename(path))
        spec = importlib.util.spec_from_file_location(name, path)
        if not spec or not spec.loader:
            raise RuntimeError(f"Cannot load module from {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
