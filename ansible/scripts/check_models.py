"""Phase 2: Verify MLflow model registration and print best run metrics."""
import sys
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = mlflow.MlflowClient()

demand = client.search_model_versions("name='taxi-demand-forecaster'")
fare   = client.search_model_versions("name='taxi-fare-predictor'")

if not demand or not fare:
    print("MISSING_MODELS")
    sys.exit(1)

print(f"demand_v{demand[0].version} fare_v{fare[0].version}")

for exp_name in ["taxi-demand-forecasting", "taxi-fare-prediction"]:
    exp = client.get_experiment_by_name(exp_name)
    if not exp:
        continue
    runs = client.search_runs(
        exp.experiment_id,
        order_by=["metrics.test_rmse ASC"],
        max_results=1,
    )
    if runs:
        m = runs[0].data.metrics
        print(
            f"{exp_name}: test_r2={m.get('test_r2', 0):.3f}"
            f" test_rmse={m.get('test_rmse', 0):.3f}"
        )
