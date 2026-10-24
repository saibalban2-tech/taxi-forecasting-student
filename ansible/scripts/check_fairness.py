"""Phase 6: Parse fairness reports and print summary."""
import json
import pathlib
import sys

base = pathlib.Path("artifacts/responsible")
for problem in ["demand", "fare"]:
    f = base / f"{problem}_fairness.json"
    if not f.exists():
        print(f"MISSING: {f}")
        sys.exit(1)
    data = json.loads(f.read_text())
    dims = data.get("dimensions", [])
    flagged = [d for d in dims if d.get("flagged")]
    print(f"{problem}: {len(dims)} dimensions, {len(flagged)} flagged")
