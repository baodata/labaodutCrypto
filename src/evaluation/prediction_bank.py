"""Versioned, label-free forecasts shared by portfolio decision models."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.labels import LABEL_DEFINITION, LABEL_TARGET, RETURN_BASIS, session_calendar
from src.utils.artifacts import atomic_csv, atomic_json, file_hashes


SCHEMA_VERSION = "prediction_bank_v1"
KEY = ["fold_id", "split", "model_id", "horizon_sessions", "decision_date", "ticker"]
DATE_COLUMNS = ["decision_date", "execution_date", "label_end_date", "training_start",
                "training_cutoff", "last_training_label_date", "selection_cutoff"]
COLUMNS = KEY + ["execution_date", "label_end_date", "prediction", "fit_id",
                 "training_start", "training_cutoff", "last_training_label_date",
                 "selection_cutoff", "prediction_kind", "target", "label_definition", "return_basis"]


def fitted_weight_paths(directory: Path, fits: dict) -> list[Path]:
    """Only bank-local relative paths can be referenced by fitted artifacts."""
    root = directory.resolve()
    paths = []
    for name in sorted({fit["weights_artifact"] for fit in fits.values() if "weights_artifact" in fit}):
        path = (root / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(root):
            raise ValueError("Fitted weights must be stored inside the prediction bank directory.")
        paths.append(path)
    return paths


class PredictionBank:
    """No realized outcomes or in-sample predictions are allowed in this store.

    Cutoffs are inclusive Close dates. All fitted state and selection metadata
    must precede each decision date. This verifies the contract/provenance; it
    cannot prove that an arbitrary external predictor actually followed it.
    """

    def __init__(self, predictions: pd.DataFrame, metadata: dict, fits: dict):
        self.predictions = predictions.copy()
        self.metadata = dict(metadata)
        self.fits = dict(fits)
        self.validate()

    def validate(self) -> None:
        frame = self.predictions
        if self.metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported prediction bank schema version.")
        if not frame.columns.is_unique or set(frame.columns) != set(COLUMNS) or frame.empty:
            raise ValueError("Prediction schema must match exactly and contain forecasts, never realized labels.")
        if frame.isna().any().any() or frame.duplicated(KEY).any():
            raise ValueError("Missing metadata or duplicate prediction keys.")
        calendar = session_calendar(self.metadata["session_calendar"])
        tickers = self.metadata["tickers"]
        if not tickers or len(set(tickers)) != len(tickers):
            raise ValueError("Prediction bank needs a unique ordered ticker universe.")
        if set(frame.ticker) != set(tickers):
            raise ValueError("Prediction universe differs from metadata.")
        for name in DATE_COLUMNS:
            values = pd.DatetimeIndex(pd.to_datetime(frame[name]))
            if values.tz is not None or values.hasnans or not values.equals(values.normalize()):
                raise ValueError(f"Invalid session dates in {name}.")
            if not values.isin(calendar).all():
                raise ValueError(f"{name} contains dates outside the recorded session calendar.")
            frame[name] = values.strftime("%Y-%m-%d")
        if frame.duplicated(KEY).any():
            raise ValueError("Duplicate prediction keys after normalizing session dates.")
        horizons = frame.horizon_sessions.to_numpy()
        if (not np.issubdtype(horizons.dtype, np.integer)
                or np.issubdtype(horizons.dtype, np.bool_) or np.any(horizons <= 0)):
            raise ValueError("Prediction horizons must be positive integer trading sessions.")
        if not pd.api.types.is_numeric_dtype(frame.prediction) or not np.isfinite(frame.prediction.to_numpy()).all():
            raise ValueError("Predictions must be finite decimal returns.")
        for name, value in {"target": LABEL_TARGET, "label_definition": LABEL_DEFINITION,
                            "return_basis": RETURN_BASIS, "prediction_kind": "expected_cross_sectional_excess_return"}.items():
            if not frame[name].eq(value).all() or self.metadata.get(name) != value:
                raise ValueError(f"Prediction semantics mismatch: {name}.")
        if not frame.split.isin(["train_oof", "inner", "outer"]).all():
            raise ValueError("Only causal train_oof, inner or outer forecasts are permitted.")
        for name in ("model_id", "fold_id", "fit_id"):
            if not frame[name].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
                raise ValueError(f"Empty or invalid {name}.")
        if not ((frame.training_start <= frame.last_training_label_date)
                & (frame.last_training_label_date <= frame.training_cutoff)
                & (frame.training_cutoff < frame.decision_date)
                & (frame.selection_cutoff <= frame.training_cutoff)).all():
            raise ValueError("Training/selection must end before the forecast decision.")
        decisions = calendar.get_indexer(pd.to_datetime(frame.decision_date))
        executions = calendar.get_indexer(pd.to_datetime(frame.execution_date))
        endpoints = calendar.get_indexer(pd.to_datetime(frame.label_end_date))
        if not (np.array_equal(executions, decisions + 1)
                and np.array_equal(endpoints, decisions + horizons)):
            raise ValueError("Forecast dates violate Close[t] -> Open[t+1] -> Close[t+h].")
        # A model may differ across horizons/folds, never within one day's asset panel.
        groups = frame.groupby(KEY[:-1], sort=False)
        if not groups.size().eq(len(tickers)).all():
            raise ValueError("Each prediction date requires exactly one row per configured ticker.")
        for name in [c for c in COLUMNS if c not in KEY + ["prediction"]]:
            if not groups[name].nunique().eq(1).all():
                raise ValueError(f"Inconsistent per-day fit metadata: {name}.")
        for fit_id, rows in frame.groupby("fit_id", sort=False):
            fit = self.fits.get(fit_id)
            if fit is None:
                raise ValueError(f"Missing fitted-model artifact: {fit_id}.")
            for name in ["model_id", "fold_id", "split", "horizon_sessions", "training_start",
                         "training_cutoff", "last_training_label_date", "selection_cutoff"]:
                if not rows[name].eq(fit[name]).all():
                    raise ValueError(f"Fitted-model provenance mismatch: {fit_id}/{name}.")
            if fit.get("scaler_fit_end", fit["training_cutoff"]) > fit["training_cutoff"]:
                raise ValueError("Scaler fitted beyond the declared training cutoff.")
        protocol = self.metadata["protocol"]
        if (protocol.get("evaluation_scope") != "validation"
                or self.metadata.get("evaluation_scope") != "validation"
                or self.metadata.get("test_evaluated") is not False
                or calendar[-1] > pd.Timestamp(protocol["development_end"])):
            raise ValueError("This prediction bank schema only permits bounded development research.")
        folds = {fold["name"]: fold for fold in protocol["folds"]}
        if len(folds) != len(protocol["folds"]) or not set(frame.fold_id).issubset(folds):
            raise ValueError("Unknown or duplicate protocol folds.")
        for (fold_id, split), rows in frame.groupby(["fold_id", "split"], sort=False):
            fold = folds[fold_id]
            if split == "train_oof":
                begin, end, fit_end = calendar[0], fold["train_end"], fold["train_end"]
            elif split == "inner":
                begin, end, fit_end = fold["inner_start"], fold["inner_end"], fold["train_end"]
            else:
                begin, end, fit_end = fold["outer_start"], fold["outer_end"], fold["inner_end"]
            begin, end, fit_end = [pd.Timestamp(value).strftime("%Y-%m-%d") for value in (begin, end, fit_end)]
            if not ((rows.decision_date >= begin) & (rows.label_end_date <= end)
                    & (rows.training_cutoff <= fit_end)).all():
                raise ValueError(f"Prediction or label crosses its protocol phase: {fold_id}/{split}.")
        self.predictions = frame.loc[:, COLUMNS]

    def save(self, directory: str | Path) -> None:
        self.validate()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        names = ["predictions.csv", "prediction_fits.json", "prediction_bank_manifest.json"]
        if any((directory / name).exists() for name in names):
            raise ValueError("Prediction bank artifacts already exist; use a new run directory.")
        atomic_csv(self.predictions, directory / names[0], index=False)
        atomic_json(directory / names[1], self.fits)
        artifacts = [directory / name for name in names[:2]] + fitted_weight_paths(directory, self.fits)
        payload = dict(self.metadata, row_count=len(self.predictions),
                       artifact_hashes=file_hashes(artifacts, directory.resolve()))
        atomic_json(directory / names[2], payload)

    @classmethod
    def load(cls, directory: str | Path) -> "PredictionBank":
        """Read forecasts/fitted state only; never load labels as a side effect."""
        directory = Path(directory).resolve()
        metadata = json.loads((directory / "prediction_bank_manifest.json").read_text())
        files = [directory / name for name in ("predictions.csv", "prediction_fits.json")]
        fits = json.loads(files[1].read_text())
        if file_hashes(files + fitted_weight_paths(directory, fits), directory) != metadata.get("artifact_hashes"):
            raise ValueError("Prediction bank artifact hashes differ from the manifest.")
        predictions = pd.read_csv(files[0], dtype={"ticker": str, "model_id": str, "fold_id": str, "fit_id": str})
        if len(predictions) != metadata.get("row_count"):
            raise ValueError("Prediction bank row count differs from the manifest.")
        return cls(predictions, metadata, fits)

    def panel(self, *, model_id: str, fold_id: str, split: str, horizon: int) -> pd.DataFrame:
        """Date x asset forecasts; explicit scope prevents mixing overlapping folds."""
        frame = self.predictions
        selected = frame[(frame.model_id == model_id) & (frame.fold_id == fold_id)
                         & (frame.split == split) & (frame.horizon_sessions == horizon)]
        if selected.empty:
            raise ValueError("No predictions for the requested model/fold/split/horizon.")
        return selected.pivot(index="decision_date", columns="ticker", values="prediction").reindex(
            columns=self.metadata["tickers"]
        ).sort_index()

    def training_panel(self, *, model_id: str, fold_id: str, horizon: int) -> pd.DataFrame:
        """Allocator training always uses the causal OOF split."""
        return self.panel(model_id=model_id, fold_id=fold_id, split="train_oof", horizon=horizon)

    def evaluation_frame_from_directory(self, directory: str | Path) -> pd.DataFrame:
        """Explicit offline access to hash-checked labels from the same data snapshot."""
        directory = Path(directory).resolve()
        metadata = json.loads((directory / "labels_manifest.json").read_text())
        if file_hashes([directory / "labels.csv"], directory) != metadata.get("artifact_hashes"):
            raise ValueError("Label artifact hashes differ from the manifest.")
        for name in ("label_definition", "return_basis", "tickers", "data_hashes", "protocol_hash"):
            if metadata.get(name) != self.metadata.get(name):
                raise ValueError(f"Prediction/label provenance differs: {name}.")
        return self.evaluation_frame(pd.read_csv(directory / "labels.csv", dtype={"ticker": str}))

    def evaluation_frame(self, labels: pd.DataFrame) -> pd.DataFrame:
        """Explicit offline join. Never pass its realized columns to an allocator."""
        keys = ["decision_date", "ticker", "horizon_sessions", "label_definition", "return_basis"]
        required = keys + ["execution_date", "label_end_date", "realized_return",
                           "realized_cross_sectional_excess_return"]
        if any(name not in labels for name in required) or labels.duplicated(keys).any():
            raise ValueError("Labels must contain unique, fully specified executable outcomes.")
        outcomes = labels.loc[:, required].copy()
        for name in ("decision_date", "execution_date", "label_end_date"):
            outcomes[name] = pd.to_datetime(outcomes[name]).dt.strftime("%Y-%m-%d")
        result = self.predictions.merge(outcomes, on=keys, how="left", validate="many_to_one",
                                        suffixes=("", "_realized"))
        realized = ["realized_return", "realized_cross_sectional_excess_return"]
        if not np.isfinite(result[realized].to_numpy()).all():
            raise ValueError("Missing or nonfinite realized labels for prediction evaluation.")
        if any(not result[name].eq(result[name + "_realized"]).all()
               for name in ("execution_date", "label_end_date")):
            raise ValueError("Realized label timing differs from forecast timing.")
        return result.drop(columns=["execution_date_realized", "label_end_date_realized"])
