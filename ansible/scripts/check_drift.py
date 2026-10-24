"""Phase 5: Parse drift report and print summary."""
import json
import pathlib
import sys

base = pathlib.Path("artifacts/drift")
for problem in ["demand", "fare"]:
    f = base / f"{problem}_drift_metrics.json"
    if not f.exists():
        print(f"MISSING: {f}")
        sys.exit(1)
    data = json.loads(f.read_text())
    results = data.get("feature_results", {})
    n_drifted = sum(1 for v in results.values() if v.get("drifted"))
    print(f"{problem}: {len(results)} features checked, {n_drifted} drifted")
