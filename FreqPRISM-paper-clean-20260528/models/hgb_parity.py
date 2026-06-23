from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from torch.utils.data import DataLoader

from data.datasets import ArtifactPriorImageDataset, ImageSample, artifact_prior_collate, collect_labeled_images, limit_per_label
from networks.artifact_prior import ArtifactPriorFeatureExtractor, CodecTextureConfig
from utils.progress import progress_iter


def aggregate_probabilities(probabilities: np.ndarray, aggregation: str) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("probabilities must be a 2D array of shape [N, V]")
    if aggregation == "mean_prob":
        return values.mean(axis=1).astype(np.float32)
    if aggregation == "mean_logit":
        clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
        logits = np.log(clipped / (1.0 - clipped))
        mean_logit = logits.mean(axis=1)
        return (1.0 / (1.0 + np.exp(-mean_logit))).astype(np.float32)
    if aggregation in {"", "none"}:
        if values.shape[1] != 1:
            raise ValueError("none aggregation requires exactly one probability view")
        return values[:, 0].astype(np.float32)
    raise ValueError("aggregation must be one of: none, mean_prob, mean_logit")


@dataclass(frozen=True)
class HGBTrainConfig:
    source_root: str | Path
    output_dir: str | Path
    image_size: int = 256
    batch_size: int = 16
    num_workers: int = 4
    device: str = "cuda"
    max_iter: int = 450
    learning_rate: float = 0.03
    max_leaf_nodes: int = 127
    l2_regularization: float = 0.0001
    random_state: int = 20260518
    max_samples_per_label: int = 0
    train_variant: str = "clean"


@dataclass(frozen=True)
class HGBTrainResult:
    estimator_path: Path
    protocol_path: Path
    progress_log_path: Path
    feature_dim: int
    train_log_loss: float


def _device_name(requested: str) -> str:
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return str(requested)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve(strict=False))
    return value


def _config_json(config: HGBTrainConfig) -> dict[str, Any]:
    return {key: _jsonable(value) for key, value in asdict(config).items()}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_features(
    source_root: str | Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: str,
    max_samples_per_label: int = 0,
    train_variant: str = "clean",
    samples: list[ImageSample] | None = None,
    show_progress: bool = True,
    progress_desc: str = "APSD features",
) -> tuple[np.ndarray, np.ndarray, int]:
    selected_samples = list(samples) if samples is not None else collect_labeled_images(source_root)
    selected_samples = limit_per_label(selected_samples, int(max_samples_per_label))
    labels_present = {sample.label for sample in selected_samples}
    if labels_present != {0, 1}:
        raise ValueError("HGB source training requires both real and fake labels")
    torch_device = torch.device(_device_name(device))
    extractor = ArtifactPriorFeatureExtractor(CodecTextureConfig(image_size=int(image_size))).to(torch_device)
    extractor.eval()
    dataset = ArtifactPriorImageDataset(selected_samples, image_size=int(image_size), variant=train_variant)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=artifact_prior_collate,
    )
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in progress_iter(
            loader,
            total=len(loader),
            desc=str(progress_desc),
            unit="batch",
            enabled=bool(show_progress),
        ):
            images = batch["images"].to(torch_device)  # type: ignore[index]
            batch_features = extractor(images).detach().cpu().numpy().astype(np.float32)
            features.append(batch_features)
            labels.append(batch["labels"].detach().cpu().numpy().astype(np.int64))  # type: ignore[index]
    if not features:
        raise ValueError("no features extracted")
    feature_array = np.concatenate(features, axis=0)
    label_array = np.concatenate(labels, axis=0)
    return feature_array, label_array, int(feature_array.shape[1])


def _binary_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities.astype(np.float64), 1e-6, 1.0 - 1e-6)
    y = labels.astype(np.float64)
    return float(-(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)).mean())


def train_hgb_source_only(config: HGBTrainConfig) -> HGBTrainResult:
    _seed_everything(int(config.random_state))
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_log_path = output_dir / "progress.jsonl"
    progress_log_path.write_text("")

    features, labels, feature_dim = extract_features(
        source_root=config.source_root,
        image_size=int(config.image_size),
        batch_size=int(config.batch_size),
        num_workers=int(config.num_workers),
        device=str(config.device),
        max_samples_per_label=int(config.max_samples_per_label),
        train_variant=str(config.train_variant),
    )
    with progress_log_path.open("a") as handle:
        handle.write(
            json.dumps(
                {
                    "phase": "extract_features",
                    "sample_count": int(labels.shape[0]),
                    "feature_dim": int(feature_dim),
                    "train_variant": str(config.train_variant),
                    "target_labels_used": False,
                },
                sort_keys=True,
            )
            + "\n"
        )

    estimator = HistGradientBoostingClassifier(
        max_iter=int(config.max_iter),
        learning_rate=float(config.learning_rate),
        max_leaf_nodes=int(config.max_leaf_nodes),
        l2_regularization=float(config.l2_regularization),
        random_state=int(config.random_state),
    )
    estimator.fit(features, labels)
    probabilities = estimator.predict_proba(features)[:, 1]
    train_log_loss = _binary_log_loss(labels, probabilities)

    estimator_path = output_dir / "artifact_prior_hgb_parity.joblib"
    joblib.dump(
        {
            "estimator": estimator,
            "feature_dim": int(feature_dim),
            "image_size": int(config.image_size),
            "train_config": _config_json(config),
            "target_labels_used": False,
        },
        estimator_path,
    )
    protocol_path = output_dir / "training_protocol.json"
    protocol_path.write_text(
        json.dumps(
            {
                "model_name": "ArtifactPriorHGBParity",
                "estimator": str(estimator_path.resolve(strict=False)),
                "progress_log": str(progress_log_path.resolve(strict=False)),
                "train_config": _config_json(config),
                "feature_dim": int(feature_dim),
                "train_sample_count": int(labels.shape[0]),
                "train_log_loss": float(train_log_loss),
                "target_labels_used": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    with progress_log_path.open("a") as handle:
        handle.write(
            json.dumps(
                {
                    "phase": "train_hgb",
                    "train_log_loss": float(train_log_loss),
                    "target_labels_used": False,
                },
                sort_keys=True,
            )
            + "\n"
        )
    return HGBTrainResult(
        estimator_path=estimator_path,
        protocol_path=protocol_path,
        progress_log_path=progress_log_path,
        feature_dim=int(feature_dim),
        train_log_loss=float(train_log_loss),
    )
