# NYC Taxi Forecasting — MLOps Capstone

End-to-end MLOps platform for two NYC Yellow Taxi prediction tasks:

- **Demand:** hourly pickup count per taxi zone (XGBoost / LightGBM, recursive multi-hour forecast)
- **Fare:** `fare_amount` per trip (XGBoost / LightGBM, booking-time estimate)

Covers every production phase: data ingestion → cleaning → feature engineering → hyperparameter tuning (Optuna) → MLflow tracking & registry → FastAPI serving → batch forecasting → CI/CD → drift detection → responsible AI (SHAP + fairness) → Streamlit dashboard.

---

## Quick Start (Local)

### Prerequisites
- Python 3.11
- Docker (for container demo)
- ~4 GB disk space for data + models

### 1 — Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dashboard.txt
```

### 2 — Run the full pipeline (one command)

```bash
./run_all.sh
```

This executes all 8 phases end-to-end from a clean state:
data pipeline → model training → serving smoke test → drift reports → responsible AI → golden-path verification.

### 3 — Run phases individually

```bash
# Phase 1: Data pipeline
python -m scripts.run_pipeline

# Phase 2: Model training (--n-trials 5 for speed; default 25 for full quality)
python -m scripts.train_models --n-trials 5

# Phase 3: Start the API
python -m uvicorn src.serving.api:app --port 8000

# Phase 5: Generate drift report
python -m scripts.monitoring.demo_drift_report --problem all --mode drift

# Phase 6: Responsible AI analysis
python -m scripts.responsible.run_analysis

# Phase 7: Start the dashboard (requires API running on :8000)
python -m streamlit run dashboard/app.py --server.port 8501

# Phase 8: Make MLflow DB portable for Docker, then run containers
python scripts/make_mlflow_portable.py
docker compose -f docker/docker-compose.yml up --build
```

### 4 — Run tests

```bash
python -m pytest tests/unit/    # 102 unit tests, no network/model required
python -m pytest tests/         # + integration smoke tests
```

---

## API Reference

Once the API is running on `http://localhost:8000`:

```bash
# Health
curl http://localhost:8000/health

# Fare prediction
curl -X POST http://localhost:8000/predict/fare \
  -H "Content-Type: application/json" \
  -d '{"trip_distance":3.4,"pu_location_id":132,"do_location_id":230,
       "passenger_count":1,"pickup_datetime":"2024-01-15T18:30:00"}'

# Demand prediction
curl -X POST http://localhost:8000/predict/demand \
  -H "Content-Type: application/json" \
  -d '{"zone":132,"target_hour":"2024-01-31T20:00:00"}'

# Recursive batch forecast (6 hours, 3 zones)
curl -X POST http://localhost:8000/forecast/demand \
  -H "Content-Type: application/json" \
  -d '{"horizon_hours":6,"zones":[132,186,237]}'

# SHAP explanation
curl -X POST http://localhost:8000/explain/fare \
  -H "Content-Type: application/json" \
  -d '{"trip_distance":3.4,"pu_location_id":132,"do_location_id":230,
       "passenger_count":1,"pickup_datetime":"2024-01-15T18:30:00"}'

# Swagger UI
open http://localhost:8000/docs
```

---

## CI/CD

GitHub Actions workflow at `.github/workflows/ci.yml`:

| Job | What it checks |
|-----|---------------|
| `lint` | `ruff check` across all source |
| `unit-tests` | 102 unit tests, ≥ 75 % coverage gate |
| `integration-tests` | API + pipeline smoke tests (offline, synthetic data) |
| `docker-build-api` | Dockerfile builds cleanly (builder stage; model artefacts trained locally) |
| `docker-build-dashboard` | Full dashboard image build (reads committed artefacts) |

---

## Docker

```bash
# After running training (mlruns/ and mlflow.container.db must exist):
python scripts/make_mlflow_portable.py
docker compose -f docker/docker-compose.yml up --build

# Services:
#   API      → http://localhost:8000  (Swagger at /docs)
#   MLflow UI → http://localhost:5000

# Dashboard container:
docker build -f docker/Dockerfile.dashboard -t taxi-dashboard .
docker run -p 8501:8501 -e API_BASE_URL=http://host.docker.internal:8000 taxi-dashboard
```

---

## Project Layout

| Path | Responsibility |
|------|---------------|
| `src/data/` | Ingest, clean, aggregate, split |
| `src/features/` | Calendar, demand lags/rolling, fare features |
| `src/models/` | Optuna tuning, train, evaluate, SHAP, model card |
| `src/serving/` | FastAPI app, inference, batch forecast, registry |
| `src/monitoring/` | Capture parsing, PSI + KS drift, performance comparison |
| `src/responsible/` | SHAP explainability, subgroup fairness |
| `src/aws/` | S3 helpers, SageMaker processor wrappers |
| `dashboard/` | Streamlit showcase (6 sections) |
| `scripts/` | CLI entry points for every phase |
| `tests/` | 102 unit tests + integration smoke tests |
| `docker/` | Dockerfile (API), Dockerfile.dashboard, docker-compose |
| `infra/terraform/` | Optional AWS infrastructure templates |
| `.github/workflows/` | CI/CD pipeline |
| `run_all.sh` | Golden-path end-to-end orchestration script |

---

## Model Results (Jan 2024 TLC data)

| Problem | Algorithm | Val RMSE | Test RMSE | Test R² |
|---------|-----------|----------|-----------|---------|
| Demand (pickups/hour/zone) | LightGBM | 8.38 | 8.59 | 0.971 |
| Fare (USD) | LightGBM | 3.37 | 3.47 | 0.955 |

Tuned with Optuna TPE (5–25 trials per estimator), selected by validation RMSE on a strict chronological 80/10/10 split.

---

## Monitoring & Responsible AI

- **Drift**: PSI (threshold 0.2) + KS test per feature. Reports in `artifacts/drift/`.
- **Fairness**: Disparity ratio (worst/best RMSE) across borough, time-of-day, weekend, airport zones. Flagged > 1.5×.
- **Explainability**: SHAP `TreeExplainer` — global summaries in `artifacts/responsible/`, per-request contributions via `/explain/*` API.
- **Model cards**: `docs/model_cards/demand.md`, `docs/model_cards/fare.md`.

---

## Notes

- `mlruns/`, `mlflow.db`, and `data/` are gitignored — run the pipeline locally to generate them.
- AWS, Docker, and Terraform are **optional**. All phases work fully offline.
- The dashboard connects to the API over HTTP; start the API before the dashboard.
