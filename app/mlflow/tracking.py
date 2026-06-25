from __future__ import annotations

from pathlib import Path
from typing import Protocol


class ExperimentTracker(Protocol):
    def log_run(
        self,
        *,
        run_name: str,
        parameters: dict[str, object],
        metrics: dict[str, float],
        artifacts: list[Path] | None = None,
    ) -> str | None:
        ...


class NoOpExperimentTracker:
    def log_run(
        self,
        *,
        run_name: str,
        parameters: dict[str, object],
        metrics: dict[str, float],
        artifacts: list[Path] | None = None,
    ) -> str | None:
        return None


class MLflowExperimentTracker:
    def __init__(
        self,
        *,
        tracking_uri: str | None,
        experiment_name: str,
    ) -> None:
        try:
            import mlflow
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "MLflow is not installed. Install with `pip install -e '.[platform]'`."
            ) from exc

        self._mlflow = mlflow
        if tracking_uri:
            self._mlflow.set_tracking_uri(tracking_uri)
        self._mlflow.set_experiment(experiment_name)

    def log_run(
        self,
        *,
        run_name: str,
        parameters: dict[str, object],
        metrics: dict[str, float],
        artifacts: list[Path] | None = None,
    ) -> str | None:
        with self._mlflow.start_run(run_name=run_name) as run:
            self._mlflow.log_params({key: str(value) for key, value in parameters.items()})
            self._mlflow.log_metrics(metrics)
            for artifact in artifacts or []:
                if artifact.exists():
                    self._mlflow.log_artifact(str(artifact))
            return str(run.info.run_id)
