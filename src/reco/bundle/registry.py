"""MLflow logging and alias-based promotion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reco.bundle.pipeline import candidate_version, read_report
from reco.bundle.writer import write_pointer
from reco.config import runtime_paths
from reco.evaluation.gate import evaluate_gate


class PromotionRejected(RuntimeError):
    """Raised when a candidate fails the regression gate."""


def _log_mlflow(bundle_path: Path, report: dict[str, Any], home: Path | None) -> str:
    import mlflow
    from mlflow import MlflowClient

    class BundleReferenceModel(mlflow.pyfunc.PythonModel):
        """Registry placeholder; production serving reads the logged portable bundle directly."""

        def predict(self, context: Any, model_input: Any, params: Any = None) -> Any:
            return model_input

    paths = runtime_paths(home)
    tracking_uri = f"sqlite:///{paths['mlflow_db'].as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    experiment = client.get_experiment_by_name("netflix-reco")
    if experiment is None:
        experiment_id = client.create_experiment(
            "netflix-reco", artifact_location=(paths["home"] / "mlruns").resolve().as_uri()
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    with mlflow.start_run(run_name=str(bundle_path.name)):
        mlflow.log_params({"bundle_version": bundle_path.name, "interaction_source": "simulated"})
        for model_name, values in report["models"].items():
            for metric, value in values.items():
                mlflow.log_metric(f"{model_name}.{metric}", float(value))
        mlflow.log_artifacts(str(bundle_path), artifact_path="bundle")
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=BundleReferenceModel(),
            artifacts={"portable_bundle": str(bundle_path)},
        )
    registered = mlflow.register_model(model_info.model_uri, "netflix-reco")
    client.set_registered_model_alias("netflix-reco", "champion", registered.version)
    return str(registered.version)


def promote_bundle(
    home: Path | None = None,
    version: str | None = None,
    *,
    track_mlflow: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """Gate, register, alias, and atomically promote one bundle.

    ``force`` is reserved for explicit rollback to a previously promoted version.
    """
    paths = runtime_paths(home)
    selected_version = version or candidate_version(home)
    bundle_path = paths["models"] / selected_version
    report = read_report(bundle_path)
    reasons = list(report["promotion_gate"].get("reasons", []))
    champion_path = paths["champion"]
    if champion_path.is_file() and not force:
        champion_version = champion_path.read_text(encoding="utf-8").strip()
        if champion_version != selected_version:
            champion_report = read_report(paths["models"] / champion_version)
            passed, regression_reasons = evaluate_gate(
                report["models"]["hybrid"],
                report["models"]["popularity"],
                champion_report["models"]["hybrid"],
            )
            if not passed:
                reasons.extend(regression_reasons)
    if (not report["promotion_gate"].get("passed", False) or reasons) and not force:
        raise PromotionRejected("; ".join(sorted(set(reasons))) or "candidate gate failed")
    registry_version = _log_mlflow(bundle_path, report, home) if track_mlflow else "not-tracked"
    write_pointer(champion_path, selected_version)
    return {"bundle_version": selected_version, "registry_version": registry_version, "alias": "champion"}
