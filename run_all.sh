#!/usr/bin/env bash
# =============================================================================
# run_all.sh — NYC Taxi Forecasting: Golden-Path End-to-End Execution
# =============================================================================
# Runs every capstone phase from a clean state, logging each step.
# Prerequisites (one-time):
#   python -m venv .venv
#   .venv/bin/pip install -r requirements.txt
#   .venv/bin/pip install -r requirements-dashboard.txt
#
# Usage:
#   ./run_all.sh            # run all phases
#   ./run_all.sh --skip-download   # skip TLC data download (use existing raw data)
# =============================================================================

set -euo pipefail

PYTHON=".venv/bin/python"
LOG_DIR="logs"
N_TRIALS="${OPTUNA_TRIALS:-5}"       # override: OPTUNA_TRIALS=25 ./run_all.sh
SKIP_DOWNLOAD="${1:-}"

mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
section() { echo; echo "══════════════════════════════════════════"; echo "  $*"; echo "══════════════════════════════════════════"; }

# ─── verify python ─────────────────────────────────────────────────────────
if [ ! -f "$PYTHON" ]; then
  echo "ERROR: .venv not found. Run: python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

# =============================================================================
section "PHASE 1 — Data Pipeline"
# =============================================================================
log "Running data pipeline (ingest → clean → aggregate → features → splits)..."

PIPELINE_ARGS=""
if [ "$SKIP_DOWNLOAD" = "--skip-download" ]; then
  PIPELINE_ARGS="--skip-download"
  log "(skipping TLC download — using existing raw data)"
fi

$PYTHON -m scripts.run_pipeline $PIPELINE_ARGS 2>&1 | tee "$LOG_DIR/phase1_pipeline.log"
log "Phase 1 complete. Processed data in data/processed/, features in data/features/"


# =============================================================================
section "PHASE 2 — Model Training"
# =============================================================================
log "Resetting MLflow database..."
rm -f mlflow.db mlflow.container.db
rm -rf mlruns/

log "Training demand + fare models (Optuna trials=$N_TRIALS per estimator)..."
$PYTHON -m scripts.train_models --n-trials "$N_TRIALS" 2>&1 | tee "$LOG_DIR/phase2_training.log"
log "Phase 2 complete. Models registered in MLflow registry."


# =============================================================================
section "PHASE 3 — Model Serving (smoke test)"
# =============================================================================
log "Starting FastAPI on port 8000..."
$PYTHON -m uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 &
API_PID=$!
# Trap to kill the API when this script exits
trap "log 'Stopping API (pid $API_PID)...'; kill $API_PID 2>/dev/null || true" EXIT

# Wait for API to be ready
log "Waiting for API startup..."
for i in {1..30}; do
  if $PYTHON -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" 2>/dev/null; then
    break
  fi
  sleep 1
done

log "Running inference smoke tests..."
$PYTHON - <<'PYEOF' 2>&1 | tee "$LOG_DIR/phase3_serving.log"
import json, urllib.request

def post(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://localhost:8000{path}",
        data=data, headers={"Content-Type": "application/json"}
    )
    return json.loads(urllib.request.urlopen(req, timeout=30).read())

health  = json.loads(urllib.request.urlopen("http://localhost:8000/health").read())
print("Health:", json.dumps(health, indent=2))
assert health["status"] == "ok", "API not healthy"
assert health["demand_model_loaded"], "Demand model not loaded"
assert health["fare_model_loaded"], "Fare model not loaded"

fare = post("/predict/fare", {
    "trip_distance": 3.4, "pu_location_id": 132, "do_location_id": 230,
    "passenger_count": 1, "pickup_datetime": "2024-01-15T18:30:00"
})
print("Fare prediction:", fare)

demand = post("/predict/demand", {"zone": 132, "target_hour": "2024-01-31T20:00:00"})
print("Demand prediction:", demand)

batch = post("/forecast/demand", {"horizon_hours": 3, "zones": [132, 186]})
print(f"Batch forecast: {len(batch['points'])} points")

print("\nPhase 3 PASSED — both models serving correctly.")
PYEOF

log "Phase 3 complete."


# =============================================================================
section "PHASE 5 — Monitoring & Drift Detection"
# =============================================================================
log "Generating drift reports for both problems (simulated drift)..."
$PYTHON -m scripts.monitoring.demo_drift_report --problem all --mode drift 2>&1 | tee "$LOG_DIR/phase5_drift.log"
log "Phase 5 complete. Reports in artifacts/drift/"


# =============================================================================
section "PHASE 6 — Responsible AI"
# =============================================================================
log "Running fairness + SHAP analysis for both models..."
$PYTHON -m scripts.responsible.run_analysis 2>&1 | tee "$LOG_DIR/phase6_responsible.log"
log "Phase 6 complete. Model cards in docs/model_cards/, plots in artifacts/responsible/"


# =============================================================================
section "PHASE 8 — Golden Path: Docker & Portability"
# =============================================================================
log "Creating portable MLflow container database..."
$PYTHON scripts/make_mlflow_portable.py \
  --src mlflow.db \
  --dest mlflow.container.db \
  --new-base file:///app/mlruns 2>&1 | tee "$LOG_DIR/phase8_portable.log"
log "mlflow.container.db written (artifact paths repointed to /app/mlruns)."

log "Docker compose build + up (detached, API + MLflow UI)..."
if command -v docker &>/dev/null; then
  docker compose -f docker/docker-compose.yml build --quiet 2>&1 | tee -a "$LOG_DIR/phase8_portable.log"
  docker compose -f docker/docker-compose.yml up -d 2>&1 | tee -a "$LOG_DIR/phase8_portable.log"
  log "Containers started. API on :8000, MLflow UI on :5000."
  log "To stop: docker compose -f docker/docker-compose.yml down"
else
  log "Docker not available — skipping container build."
fi

log "Phase 8 complete."


# =============================================================================
section "ALL PHASES COMPLETE"
# =============================================================================
echo
echo "  Data pipeline  : data/processed/, data/features/"
echo "  MLflow DB      : mlflow.db, mlflow.container.db"
echo "  MLflow runs    : mlruns/"
echo "  Artifacts      : artifacts/demand/, artifacts/fare/, artifacts/drift/, artifacts/responsible/"
echo "  Model cards    : docs/model_cards/"
echo "  CI/CD workflow : .github/workflows/ci.yml"
echo "  Logs           : $LOG_DIR/"
echo
echo "  Start services (re-run after the script exits):"
echo "    API:       $PYTHON -m uvicorn src.serving.api:app --port 8000"
echo "    Dashboard: $PYTHON -m streamlit run dashboard/app.py --server.port 8501"
echo "    MLflow UI: $PYTHON -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000"
echo
