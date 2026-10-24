"""Load models from the MLflow Model Registry.

The training pipeline registers the best model per problem under the names in
``config.models`` (``taxi-demand-forecaster`` / ``taxi-fare-predictor``). Here we
resolve a registry reference (a version number or ``"latest"``) and load the
model using the appropriate native MLflow flavor, configuring the tracking URI
first.

MLflow 3.x model-centric storage note
--------------------------------------
Starting in MLflow 3, registered models are stored at a model-centric path
(``mlruns/{exp_id}/models/{model_id}/artifacts/``) recorded in the DB column
``model_versions.storage_location``.  The Python ``ModelVersion`` object does
NOT expose this field, so we read it directly from the SQLite DB when the
tracking URI is a local SQLite file.  The ``make_mlflow_portable.py`` script
rewrites those paths so they point to ``/app/mlruns`` inside the container.
"""

from __future__ import annotations

import logging
import sqlite3
import urllib.parse
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException

from src.config import Config, get_config
from src.models.common import registry_model_name
from src.utils.mlflow_utils import configure_mlflow

logger = logging.getLogger(__name__)
_CONTAINER_MLRUNS = Path("/app/mlruns")


def model_uri(problem: str, config: Config | None = None) -> str:
    """Build the ``models:/`` URI for a problem's registered model."""
    config = config or get_config()
    name = registry_model_name(problem, config)
    stage = str(config.serving.get("model_stage", "latest"))
    return f"models:/{name}/{stage}"


def _sqlite_storage_locations(tracking_uri: str) -> dict[str, str]:
    """Read ``{model_name: storage_location}`` directly from a SQLite tracking DB.

    Returns an empty dict if the URI is not a SQLite file or the column does
    not exist (older MLflow versions without model-centric storage).
    """
    uri = tracking_uri or ""
    # Accept both  sqlite:///path  and  sqlite:////abs/path  forms.
    if not uri.startswith("sqlite:///"):
        return {}
    path_part = uri[len("sqlite:///"):]
    # urllib-decode in case the path was percent-encoded.
    db_path = Path(urllib.parse.unquote(path_part))
    if not db_path.exists():
        return {}
    try:
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute(
                "SELECT name, storage_location FROM model_versions"
            ).fetchall()
            return {name: loc for name, loc in rows if loc}
        except sqlite3.OperationalError:
            return {}
        finally:
            conn.close()
    except Exception:
        return {}


def _load_by_uri(uri: str, estimator: str) -> object:
    """Load a model artifact from *uri* using the appropriate native flavor."""
    if estimator == "xgboost":
        return mlflow.xgboost.load_model(uri)
    if estimator == "lightgbm":
        return mlflow.lightgbm.load_model(uri)
    return mlflow.sklearn.load_model(uri)


# Alias for backwards-compatibility with existing monkeypatches in tests.
_load_native = _load_by_uri


def _estimator_from_run(client: mlflow.MlflowClient, run_id: str) -> str:
    """Return the 'estimator' tag recorded on a training run, or 'sklearn'."""
    try:
        run = client.get_run(run_id)
        return run.data.tags.get("estimator", "sklearn")
    except Exception:
        return "sklearn"


def _resolve_version(client: mlflow.MlflowClient, name: str, stage: str):
    """Return the ModelVersion object for *name* at *stage* / version number."""
    if stage.isdigit():
        return client.get_model_version(name, stage)
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise MlflowException(f"No registered versions found for {name}")
    return max(versions, key=lambda v: int(v.version))


def _container_uri_from_storage(storage_loc: str) -> str | None:
    """Rebase a host *storage_location* path to the mounted ``/app/mlruns``.

    Returns the rebased ``file://`` URI if the subpath exists, else ``None``.
    """
    marker = "/mlruns/"
    candidate = (storage_loc or "").replace("\\", "/")
    if marker not in candidate:
        return None
    rel = candidate.split(marker, 1)[1]
    path = _CONTAINER_MLRUNS / rel
    return path.as_uri() if path.exists() else None


def load_model(problem: str, config: Config | None = None):
    """Load the registered model for ``problem`` from the MLflow registry."""
    config = config or get_config()
    configure_mlflow(config)
    name = registry_model_name(problem, config)
    stage = str(config.serving.get("model_stage", "latest"))
    tracking_uri = config.mlflow.get("tracking_uri", "")
    logger.info("Loading model for %s (name=%s stage=%s)", problem, name, stage)

    # Read storage_location from SQLite directly — MLflow 3.x stores models at
    # a model-centric path that the Python ModelVersion object does not expose.
    storage_map = _sqlite_storage_locations(tracking_uri)

    client = mlflow.MlflowClient()
    version = _resolve_version(client, name, stage)
    estimator = _estimator_from_run(client, version.run_id)

    # ── Primary: use storage_location from DB (MLflow 3.x model-centric path)
    storage_loc = storage_map.get(name, "")
    if storage_loc:
        logger.info(
            "Loading %s model from storage_location=%s (estimator=%s)",
            problem, storage_loc, estimator,
        )
        try:
            return _load_by_uri(storage_loc, estimator)
        except (MlflowException, OSError) as exc:
            logger.warning("Primary load failed (%s); trying container fallback.", exc)
            if _CONTAINER_MLRUNS.is_dir():
                container_uri = _container_uri_from_storage(storage_loc)
                if container_uri:
                    logger.info("Container fallback: %s", container_uri)
                    return mlflow.pyfunc.load_model(container_uri)
            raise

    # ── Fallback: classic runs:/ artifact path (MLflow <3 or remote tracking)
    runs_uri = f"runs:/{version.run_id}/model"
    logger.info(
        "No storage_location in DB; loading %s from %s (estimator=%s)",
        problem, runs_uri, estimator,
    )
    try:
        return _load_by_uri(runs_uri, estimator)
    except (MlflowException, OSError):
        if not _CONTAINER_MLRUNS.is_dir():
            raise
        # Last resort: scan version.source for a /mlruns/ segment and rebase.
        container_uri = _container_uri_from_storage(str(version.source))
        if container_uri:
            logger.info("Container fallback from source: %s", container_uri)
            return mlflow.pyfunc.load_model(container_uri)
        raise
