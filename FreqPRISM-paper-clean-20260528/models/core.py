from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

CODEC_FAMILIES: tuple[str, ...] = ("codec_block",)
SOURCE_ONLY_TRANSFER_FAMILIES: tuple[str, ...] = (
    "recompression_stability",
    "chroma_luma_coupling",
    "residual_spectrum",
    "texture_chroma_ratio",
    "texture_phase",
    "rich_stride_stats",
)


def family_indices(family_slices: Mapping[str, slice], family_names: Sequence[str]) -> np.ndarray:
    indices: list[int] = []
    for name in family_names:
        if name not in family_slices:
            raise ValueError(f"unknown feature family: {name}")
        family_slice = family_slices[name]
        start = int(family_slice.start or 0)
        stop = int(family_slice.stop or 0)
        if stop <= start:
            raise ValueError(f"feature family has empty slice: {name}")
        indices.extend(range(start, stop))
    return np.asarray(sorted(set(indices)), dtype=np.int64)


def probabilities_to_logits(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped)).astype(np.float32)


def _validate_xy(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2:
        raise ValueError("features must be 2D")
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError("labels must be 1D and match features")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be binary 0/1")
    return x, y


@dataclass
class FeatureExpert:
    model: Any
    feature_indices: np.ndarray
    family_names: tuple[str, ...]
    learner: str
    target_labels_used: bool = False

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32)
        return np.asarray(self.model.predict_proba(x[:, self.feature_indices]), dtype=np.float32)


@dataclass
class SourceCombiner:
    model: Any
    target_labels_used: bool = False

    def design_matrix(self, codec_probabilities: np.ndarray, transfer_probabilities: np.ndarray) -> np.ndarray:
        codec_logits = probabilities_to_logits(codec_probabilities)
        transfer_logits = probabilities_to_logits(transfer_probabilities)
        codec_margin = np.abs(codec_logits)
        return np.stack([codec_logits, transfer_logits, codec_margin], axis=1).astype(np.float32)

    def predict_proba_from_scores(self, codec_probabilities: np.ndarray, transfer_probabilities: np.ndarray) -> np.ndarray:
        matrix = self.design_matrix(codec_probabilities, transfer_probabilities)
        return np.asarray(self.model.predict_proba(matrix)[:, 1], dtype=np.float32)


@dataclass
class ResidualLogitCombiner:
    alpha: float
    target_labels_used: bool = False

    def predict_proba_from_scores(self, codec_probabilities: np.ndarray, transfer_probabilities: np.ndarray) -> np.ndarray:
        codec_logits = probabilities_to_logits(codec_probabilities).astype(np.float64)
        transfer_logits = probabilities_to_logits(transfer_probabilities).astype(np.float64)
        combined = codec_logits + float(self.alpha) * transfer_logits
        return (1.0 / (1.0 + np.exp(-combined))).astype(np.float32)


@dataclass
class ConfidenceGatedResidualCombiner:
    alpha: float
    margin_threshold: float
    target_labels_used: bool = False

    def predict_proba_from_scores(self, codec_probabilities: np.ndarray, transfer_probabilities: np.ndarray) -> np.ndarray:
        codec_logits = probabilities_to_logits(codec_probabilities).astype(np.float64)
        transfer_logits = probabilities_to_logits(transfer_probabilities).astype(np.float64)
        gated = np.abs(codec_logits) < float(self.margin_threshold)
        combined = codec_logits.copy()
        combined[gated] = combined[gated] + float(self.alpha) * transfer_logits[gated]
        return (1.0 / (1.0 + np.exp(-combined))).astype(np.float32)


@dataclass
class V2Detector:
    codec_expert: FeatureExpert
    transfer_expert: FeatureExpert
    combiner: SourceCombiner
    image_size: int
    train_config: dict[str, Any]
    target_labels_used: bool = False

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        codec = self.codec_expert.predict_proba(features)[:, 1]
        transfer = self.transfer_expert.predict_proba(features)[:, 1]
        return self.combiner.predict_proba_from_scores(codec, transfer)


def fit_codec_hgb_expert(
    features: np.ndarray,
    labels: np.ndarray,
    feature_indices: np.ndarray,
    family_names: Sequence[str] = CODEC_FAMILIES,
    *,
    max_iter: int = 200,
    learning_rate: float = 0.03,
    max_leaf_nodes: int = 127,
    l2_regularization: float = 0.0001,
    random_state: int = 20260519,
) -> FeatureExpert:
    x, y = _validate_xy(features, labels)
    indices = np.asarray(feature_indices, dtype=np.int64)
    model = HistGradientBoostingClassifier(
        max_iter=int(max_iter),
        learning_rate=float(learning_rate),
        max_leaf_nodes=int(max_leaf_nodes),
        l2_regularization=float(l2_regularization),
        random_state=int(random_state),
    )
    model.fit(x[:, indices], y)
    return FeatureExpert(model=model, feature_indices=indices, family_names=tuple(family_names), learner="hgb")


def fit_logistic_expert(
    features: np.ndarray,
    labels: np.ndarray,
    feature_indices: np.ndarray,
    family_names: Sequence[str] = SOURCE_ONLY_TRANSFER_FAMILIES,
    *,
    c: float = 0.25,
    random_state: int = 20260519,
) -> FeatureExpert:
    x, y = _validate_xy(features, labels)
    indices = np.asarray(feature_indices, dtype=np.int64)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=float(c), class_weight="balanced", random_state=int(random_state)),
    )
    model.fit(x[:, indices], y)
    return FeatureExpert(model=model, feature_indices=indices, family_names=tuple(family_names), learner="logistic")


def fit_source_combiner(codec_probabilities: np.ndarray, transfer_probabilities: np.ndarray, labels: np.ndarray) -> SourceCombiner:
    y = np.asarray(labels, dtype=np.int64)
    if y.ndim != 1:
        raise ValueError("labels must be 1D")
    combiner = SourceCombiner(
        model=LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced", random_state=20260519)
    )
    matrix = combiner.design_matrix(codec_probabilities, transfer_probabilities)
    if matrix.shape[0] != y.shape[0]:
        raise ValueError("score arrays and labels must have matching length")
    combiner.model.fit(matrix, y)
    return combiner


def fit_residual_alpha_combiner(
    codec_probabilities: np.ndarray,
    transfer_probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    alpha_grid: Sequence[float] = (-0.25, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.25),
) -> ResidualLogitCombiner:
    y = np.asarray(labels, dtype=np.int64)
    if y.ndim != 1:
        raise ValueError("labels must be 1D")
    best_alpha = 0.0
    best_score: tuple[float, float, float] | None = None
    for alpha in alpha_grid:
        scores = ResidualLogitCombiner(alpha=float(alpha)).predict_proba_from_scores(codec_probabilities, transfer_probabilities)
        ap = float(average_precision_score(y, scores))
        auc = float(roc_auc_score(y, scores))
        # Prefer ranking first, then smaller absolute transfer influence.
        candidate = (ap + auc, -abs(float(alpha)), float(alpha))
        if best_score is None or candidate > best_score:
            best_score = candidate
            best_alpha = float(alpha)
    return ResidualLogitCombiner(alpha=best_alpha)
