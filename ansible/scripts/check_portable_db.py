"""Phase 8: Verify mlflow.container.db has all paths repointed to /app/mlruns."""
import sqlite3
import sys

conn = sqlite3.connect("mlflow.container.db")
rows = conn.execute("SELECT name, storage_location FROM model_versions").fetchall()
conn.close()

bad = [(name, loc) for name, loc in rows if loc and "/app/mlruns" not in loc]
if bad:
    print(f"FAIL — Unrepointed paths: {bad}")
    sys.exit(1)

print(f"All {len(rows)} model_version paths point to /app/mlruns")
