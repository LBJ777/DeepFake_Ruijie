from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile

from data.datasets import apply_variant, pil_to_tensor
from models.core import ResidualLogitCombiner
from models.hgb_parity import aggregate_probabilities
from networks.artifact_prior import ArtifactPriorFeatureExtractor, CodecTextureConfig
from .native_tiles import (
    aggregate_tile_scores,
    combine_whole_tile_aux_signed_delta_guard_scores,
    extract_native_tiles,
    native_tile_boxes,
)
from networks.residual_prior import infer_residual_prior_scores, load_residual_prior_model
from networks.score_blend import logit_blend
from networks.semantic_prior import load_openai_clip


ImageFile.LOAD_TRUNCATED_IMAGES = True

SCORING_MODES = {"baseline", "equivalent_fast", "aggressive_fast"}


def _normalize_scoring_mode(value: object, *, gpu_preprocess: bool) -> str:
    if value is None:
        return "aggressive_fast" if bool(gpu_preprocess) else "baseline"
    mode = str(value).strip()
    if not mode:
        return "aggressive_fast" if bool(gpu_preprocess) else "baseline"
    if mode not in SCORING_MODES:
        raise ValueError(f"unsupported scoring_mode: {mode}")
    return mode


@dataclass(frozen=True)
class UnifiedDetectorConfig:
    root: Path
    artifact_model_path: Path
    semantic_probe_path: Path
    residual_prior_path: Path
    artifact_image_size: int
    artifact_variants: tuple[str, ...]
    semantic_model_name: str
    semantic_variants: tuple[str, ...]
    semantic_download_root: Path
    residual_inference_image_size: int | None
    tile_mode: str
    tile_size: int
    tile_grid_size: int
    beta: float
    alpha_low_pos: float
    alpha_low_neg: float
    alpha_high_pos: float
    alpha_high_neg: float
    alpha_high_neg_guard: float
    tile_delta_threshold: float
    high_res_threshold: float
    gamma: float
    threshold: float
    artifact_forward_batch_size: int
    artifact_tile_batch_size: int
    semantic_forward_batch_size: int
    gpu_preprocess: bool
    scoring_mode: str

    @classmethod
    def from_root(cls, root: str | Path, config_name: str = "apfreq_train100k_full.yaml") -> "UnifiedDetectorConfig":
        import yaml

        project_root = Path(root).resolve(strict=False)
        raw = yaml.safe_load((project_root / "configs" / config_name).read_text())
        artifacts = dict(raw["artifacts"])
        artifact = dict(raw["artifact_prior"])
        semantic = dict(raw["semantic_prior"])
        residual = dict(raw["residual_prior"])
        composition = dict(raw["composition"])
        runtime = dict(raw.get("runtime") or {})
        gpu_preprocess = bool(runtime.get("gpu_preprocess", False))
        scoring_mode = _normalize_scoring_mode(runtime.get("scoring_mode"), gpu_preprocess=gpu_preprocess)
        return cls(
            root=project_root,
            artifact_model_path=project_root / artifacts["artifact_model"],
            semantic_probe_path=project_root / artifacts["semantic_probe"],
            residual_prior_path=project_root / artifacts["residual_prior"],
            artifact_image_size=int(artifact["image_size"]),
            artifact_variants=tuple(str(item) for item in artifact["eval_variants"]),
            semantic_model_name=str(semantic["model_name"]),
            semantic_variants=tuple(str(item) for item in semantic["eval_variants"]),
            semantic_download_root=Path(str(semantic.get("download_root", Path.home() / ".cache" / "clip"))),
            residual_inference_image_size=(
                None if residual.get("inference_image_size") is None else int(residual["inference_image_size"])
            ),
            tile_mode=str(composition["tile_mode"]),
            tile_size=int(composition["tile_size"]),
            tile_grid_size=int(composition["tile_grid_size"]),
            beta=float(composition["beta"]),
            alpha_low_pos=float(composition["alpha_low_pos"]),
            alpha_low_neg=float(composition["alpha_low_neg"]),
            alpha_high_pos=float(composition["alpha_high_pos"]),
            alpha_high_neg=float(composition["alpha_high_neg"]),
            alpha_high_neg_guard=float(composition["alpha_high_neg_guard"]),
            tile_delta_threshold=float(composition["tile_delta_threshold"]),
            high_res_threshold=float(composition["high_res_threshold"]),
            gamma=float(composition["gamma"]),
            threshold=float(composition["threshold"]),
            artifact_forward_batch_size=int(runtime.get("artifact_forward_batch_size", 64)),
            artifact_tile_batch_size=int(
                runtime.get("artifact_tile_batch_size", runtime.get("artifact_forward_batch_size", 64))
            ),
            semantic_forward_batch_size=int(runtime.get("semantic_forward_batch_size", runtime.get("semantic_batch_size", 32))),
            gpu_preprocess=gpu_preprocess,
            scoring_mode=scoring_mode,
        )

    def with_artifact_overrides(
        self,
        *,
        artifact_model_path: str | Path | None = None,
        semantic_probe_path: str | Path | None = None,
        residual_prior_path: str | Path | None = None,
    ) -> "UnifiedDetectorConfig":
        updates: dict[str, Path] = {}
        if artifact_model_path:
            updates["artifact_model_path"] = Path(artifact_model_path).resolve(strict=False)
        if semantic_probe_path:
            updates["semantic_probe_path"] = Path(semantic_probe_path).resolve(strict=False)
        if residual_prior_path:
            updates["residual_prior_path"] = Path(residual_prior_path).resolve(strict=False)
        return replace(self, **updates)

    def with_runtime_overrides(
        self,
        *,
        scoring_mode: str | None = None,
        gpu_preprocess: bool | None = None,
    ) -> "UnifiedDetectorConfig":
        updated_gpu_preprocess = self.gpu_preprocess if gpu_preprocess is None else bool(gpu_preprocess)
        updated_scoring_mode = (
            self.scoring_mode
            if scoring_mode is None
            else _normalize_scoring_mode(scoring_mode, gpu_preprocess=updated_gpu_preprocess)
        )
        return replace(
            self,
            gpu_preprocess=updated_gpu_preprocess,
            scoring_mode=updated_scoring_mode,
        )


def _device(name: str) -> torch.device:
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(str(name))


class UnifiedArtifactDetector:
    def __init__(self, config: UnifiedDetectorConfig, *, device: str = "cpu", load_semantic: bool = True) -> None:
        self.config = config
        self.device = _device(device)
        self.artifact_payload: dict[str, Any] = joblib.load(config.artifact_model_path)
        self.semantic_probe = joblib.load(config.semantic_probe_path)
        self.artifact_extractor = ArtifactPriorFeatureExtractor(
            CodecTextureConfig(image_size=config.artifact_image_size)
        ).to(self.device)
        self.artifact_extractor.eval()
        self.semantic_model = None
        self.semantic_preprocess = None
        if load_semantic:
            self.semantic_model, self.semantic_preprocess = load_openai_clip(
                config.semantic_model_name,
                device=self.device,
                download_root=config.semantic_download_root,
            )
        self.residual_model = load_residual_prior_model(config.residual_prior_path, device=str(self.device))

    def cache_fingerprint(self) -> str:
        payload = {
            "artifact_model_path": str(self.config.artifact_model_path.resolve(strict=False)),
            "semantic_probe_path": str(self.config.semantic_probe_path.resolve(strict=False)),
            "residual_prior_path": str(self.config.residual_prior_path.resolve(strict=False)),
            "artifact_image_size": int(self.config.artifact_image_size),
            "artifact_variants": list(self.config.artifact_variants),
            "semantic_model_name": str(self.config.semantic_model_name),
            "semantic_variants": list(self.config.semantic_variants),
            "residual_inference_image_size": self.config.residual_inference_image_size,
            "tile_mode": str(self.config.tile_mode),
            "tile_size": int(self.config.tile_size),
            "tile_grid_size": int(self.config.tile_grid_size),
            "beta": float(self.config.beta),
            "alpha_low_pos": float(self.config.alpha_low_pos),
            "alpha_low_neg": float(self.config.alpha_low_neg),
            "alpha_high_pos": float(self.config.alpha_high_pos),
            "alpha_high_neg": float(self.config.alpha_high_neg),
            "alpha_high_neg_guard": float(self.config.alpha_high_neg_guard),
            "tile_delta_threshold": float(self.config.tile_delta_threshold),
            "high_res_threshold": float(self.config.high_res_threshold),
            "gamma": float(self.config.gamma),
            "threshold": float(self.config.threshold),
            "gpu_preprocess": bool(self.config.gpu_preprocess),
            "scoring_mode": str(self.config.scoring_mode),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    def _score_artifact_tensor_batch(self, tensors: torch.Tensor) -> np.ndarray:
        if tensors.ndim != 4 or tensors.shape[1] != 3:
            raise ValueError("artifact tensors must have shape BCHW with 3 channels")
        chunks: list[np.ndarray] = []
        batch_size = max(1, int(self.config.artifact_forward_batch_size))
        with torch.no_grad():
            for start in range(0, int(tensors.shape[0]), batch_size):
                batch = tensors[start : start + batch_size].to(self.device)
                chunks.append(self.artifact_extractor(batch).detach().cpu().numpy().astype(np.float32))
        features = np.concatenate(chunks, axis=0)
        codec_scores = self.artifact_payload["codec"].predict_proba(features)[:, 1]
        chroma_scores = self.artifact_payload["chroma"].predict_proba(features)[:, 1]
        return ResidualLogitCombiner(alpha=float(self.artifact_payload["alpha"])).predict_proba_from_scores(
            codec_scores,
            chroma_scores,
        ).astype(np.float32)

    def _effective_scoring_mode(self) -> str:
        if self.config.scoring_mode == "baseline" and bool(self.config.gpu_preprocess):
            return "aggressive_fast"
        return str(self.config.scoring_mode)

    def _score_rgb_images(self, images: Sequence[Image.Image], variants: Sequence[str]) -> np.ndarray:
        tensors: list[torch.Tensor] = []
        for image in images:
            for variant in variants:
                tensors.append(
                    pil_to_tensor(
                        apply_variant(image, self.config.artifact_image_size, variant),
                        self.config.artifact_image_size,
                    )
                )
        if not tensors:
            return np.asarray([], dtype=np.float32)
        scores = self._score_artifact_tensor_batch(torch.stack(tensors))
        return aggregate_probabilities(scores.reshape(len(images), len(tuple(variants))), "mean_logit").astype(np.float32)

    def score_artifact_whole_paths(self, paths: Sequence[str | Path]) -> np.ndarray:
        chunks: list[np.ndarray] = []
        path_list = list(paths)
        chunk_size = max(1, int(self.config.artifact_forward_batch_size))
        for start in range(0, len(path_list), chunk_size):
            images: list[Image.Image] = []
            try:
                for path in path_list[start : start + chunk_size]:
                    images.append(Image.open(path).convert("RGB"))
                chunks.append(self._score_rgb_images(images, self.config.artifact_variants))
            finally:
                for image in images:
                    image.close()
        if not chunks:
            return np.asarray([], dtype=np.float32)
        return np.concatenate(chunks, axis=0).astype(np.float32)

    def _read_rgb_tensor(self, path: str | Path) -> torch.Tensor:
        try:
            from torchvision.io import ImageReadMode, read_image

            tensor = read_image(str(path), mode=ImageReadMode.RGB)
        except Exception:
            with Image.open(path) as image:
                array = np.asarray(image.convert("RGB"), dtype=np.uint8)
            tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return tensor.to(device=self.device, dtype=torch.float32).div(255.0)

    def _resize_tensor_batch(self, batch: torch.Tensor) -> torch.Tensor:
        image_size = int(self.config.artifact_image_size)
        if batch.shape[-2:] == (image_size, image_size):
            return batch.contiguous()
        resized = F.interpolate(
            batch,
            size=(image_size, image_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        return resized.clamp(0.0, 1.0).contiguous()

    def _pad_tile_tensor(self, tile: torch.Tensor) -> torch.Tensor:
        tile_size = int(self.config.tile_size)
        height, width = int(tile.shape[-2]), int(tile.shape[-1])
        pad_bottom = max(0, tile_size - height)
        pad_right = max(0, tile_size - width)
        if pad_bottom == 0 and pad_right == 0:
            return tile.contiguous()
        padded = F.pad(tile.unsqueeze(0), (0, pad_right, 0, pad_bottom), mode="replicate").squeeze(0)
        return padded[..., :tile_size, :tile_size].contiguous()

    def _score_artifact_tile_paths_gpu(self, paths: Sequence[str | Path]) -> tuple[np.ndarray, np.ndarray]:
        scores_by_image: list[list[float]] = [[] for _ in paths]
        max_sides: list[int] = []
        pending_tensors: list[torch.Tensor] = []
        pending_indices: list[int] = []
        tile_batch_size = max(1, int(self.config.artifact_tile_batch_size))

        def flush_pending() -> None:
            if not pending_tensors:
                return
            batch = torch.stack(pending_tensors).to(self.device)
            batch_scores = self._score_artifact_tensor_batch(batch)
            for image_index, score in zip(pending_indices, batch_scores):
                scores_by_image[int(image_index)].append(float(score))
            pending_tensors.clear()
            pending_indices.clear()

        for image_index, path in enumerate(paths):
            tensor = self._read_rgb_tensor(path)
            height, width = int(tensor.shape[-2]), int(tensor.shape[-1])
            max_sides.append(max(width, height))
            if max(width, height) <= int(self.config.tile_size):
                prepared = self._resize_tensor_batch(tensor.unsqueeze(0))[0]
                pending_tensors.append(prepared)
                pending_indices.append(image_index)
            else:
                boxes = native_tile_boxes(width, height, self.config.tile_size, self.config.tile_grid_size)
                for left, top, right, bottom in boxes:
                    tile = tensor[:, int(top) : int(bottom), int(left) : int(right)]
                    tile = self._pad_tile_tensor(tile)
                    tile = self._resize_tensor_batch(tile.unsqueeze(0))[0]
                    pending_tensors.append(tile)
                    pending_indices.append(image_index)
                    if len(pending_tensors) >= tile_batch_size:
                        flush_pending()
            if len(pending_tensors) >= tile_batch_size:
                flush_pending()
        flush_pending()

        scores = [aggregate_tile_scores(image_scores, self.config.tile_mode) for image_scores in scores_by_image]
        return np.asarray(scores, dtype=np.float32), np.asarray(max_sides, dtype=np.float32)

    def _score_artifact_tile_paths_pil_batched(self, paths: Sequence[str | Path]) -> tuple[np.ndarray, np.ndarray]:
        scores_by_image: list[list[float]] = [[] for _ in paths]
        max_sides: list[int] = []
        pending_tensors: list[torch.Tensor] = []
        pending_indices: list[int] = []
        tile_batch_size = max(1, int(self.config.artifact_tile_batch_size))

        def flush_pending() -> None:
            if not pending_tensors:
                return
            batch_scores = self._score_artifact_tensor_batch(torch.stack(pending_tensors))
            for image_index, score in zip(pending_indices, batch_scores):
                scores_by_image[int(image_index)].append(float(score))
            pending_tensors.clear()
            pending_indices.clear()

        for image_index, path in enumerate(paths):
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                width, height = rgb.size
                max_sides.append(max(int(width), int(height)))
                if max(width, height) <= self.config.tile_size:
                    tiles = [rgb]
                else:
                    boxes = native_tile_boxes(width, height, self.config.tile_size, self.config.tile_grid_size)
                    tiles = extract_native_tiles(rgb, boxes, tile_size=self.config.tile_size)
                for tile in tiles:
                    pending_tensors.append(pil_to_tensor(tile, self.config.artifact_image_size, "clean"))
                    pending_indices.append(image_index)
                    if len(pending_tensors) >= tile_batch_size:
                        flush_pending()
        flush_pending()

        scores = [aggregate_tile_scores(image_scores, self.config.tile_mode) for image_scores in scores_by_image]
        return np.asarray(scores, dtype=np.float32), np.asarray(max_sides, dtype=np.float32)

    def score_artifact_tile_paths(self, paths: Sequence[str | Path]) -> tuple[np.ndarray, np.ndarray]:
        mode = self._effective_scoring_mode()
        if mode == "aggressive_fast":
            return self._score_artifact_tile_paths_gpu(paths)
        if mode == "equivalent_fast":
            return self._score_artifact_tile_paths_pil_batched(paths)
        scores: list[float] = []
        max_sides: list[int] = []
        for path in paths:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                width, height = rgb.size
                max_sides.append(max(int(width), int(height)))
                if max(width, height) <= self.config.tile_size:
                    scores.append(float(self._score_rgb_images([rgb], ("clean",))[0]))
                    continue
                boxes = native_tile_boxes(width, height, self.config.tile_size, self.config.tile_grid_size)
                tiles = extract_native_tiles(rgb, boxes, tile_size=self.config.tile_size)
                tile_scores = self._score_rgb_images(tiles, ("clean",))
                scores.append(aggregate_tile_scores(tile_scores, self.config.tile_mode))
        return np.asarray(scores, dtype=np.float32), np.asarray(max_sides, dtype=np.float32)

    def score_semantic_paths(self, paths: Sequence[str | Path]) -> np.ndarray:
        if self.semantic_model is None or self.semantic_preprocess is None:
            raise RuntimeError("semantic prior was not loaded")
        if self._effective_scoring_mode() in {"equivalent_fast", "aggressive_fast"}:
            return self._score_semantic_paths_batched(paths)
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for path in paths:
                tensors = []
                with Image.open(path) as image:
                    for variant in self.config.semantic_variants:
                        tensors.append(self.semantic_preprocess(apply_variant(image, self.config.artifact_image_size, variant)))
                encoded = self.semantic_model.encode_image(torch.stack(tensors).to(self.device))
                features = encoded.detach().float().cpu().numpy().astype(np.float32)
                variant_scores = self.semantic_probe.predict_proba(features)
                chunks.append(aggregate_probabilities(variant_scores.reshape(1, -1), "mean_logit"))
        return np.concatenate(chunks, axis=0).astype(np.float32)

    def _score_semantic_paths_batched(self, paths: Sequence[str | Path]) -> np.ndarray:
        if self.semantic_model is None or self.semantic_preprocess is None:
            raise RuntimeError("semantic prior was not loaded")
        scores_by_image: list[list[float]] = [[] for _ in paths]
        pending_tensors: list[torch.Tensor] = []
        pending_indices: list[int] = []
        batch_size = max(1, int(self.config.semantic_forward_batch_size))

        def flush_pending() -> None:
            if not pending_tensors:
                return
            with torch.no_grad():
                batch = torch.stack(pending_tensors).to(self.device)
                encoded = self.semantic_model.encode_image(batch)
                features = encoded.detach().float().cpu().numpy().astype(np.float32)
            batch_scores = self.semantic_probe.predict_proba(features)
            for image_index, score in zip(pending_indices, batch_scores):
                scores_by_image[int(image_index)].append(float(score))
            pending_tensors.clear()
            pending_indices.clear()

        for image_index, path in enumerate(paths):
            with Image.open(path) as image:
                for variant in self.config.semantic_variants:
                    prepared = apply_variant(image, self.config.artifact_image_size, variant)
                    pending_tensors.append(self.semantic_preprocess(prepared))
                    pending_indices.append(image_index)
                    if len(pending_tensors) >= batch_size:
                        flush_pending()
        flush_pending()
        scores = [
            aggregate_probabilities(np.asarray(image_scores, dtype=np.float32).reshape(1, -1), "mean_logit")[0]
            for image_scores in scores_by_image
        ]
        return np.asarray(scores, dtype=np.float32)

    def score_residual_prior_paths(self, paths: Sequence[str | Path], *, batch_size: int = 32) -> np.ndarray:
        return infer_residual_prior_scores(
            paths,
            model=self.residual_model,
            device=str(self.device),
            image_size=self.config.residual_inference_image_size,
            batch_size=batch_size,
        )

    def score_component_paths(
        self,
        paths: Sequence[str | Path],
        *,
        residual_batch_size: int = 32,
    ) -> dict[str, np.ndarray]:
        whole = self.score_artifact_whole_paths(paths)
        tile, max_side = self.score_artifact_tile_paths(paths)
        semantic = self.score_semantic_paths(paths)
        base = combine_whole_tile_aux_signed_delta_guard_scores(
            whole,
            tile,
            semantic,
            high_res_mask=max_side > self.config.high_res_threshold,
            beta=self.config.beta,
            alpha_low_pos=self.config.alpha_low_pos,
            alpha_low_neg=self.config.alpha_low_neg,
            alpha_high_pos=self.config.alpha_high_pos,
            alpha_high_neg=self.config.alpha_high_neg,
            alpha_high_neg_guard=self.config.alpha_high_neg_guard,
            tile_delta_threshold=self.config.tile_delta_threshold,
        )
        residual = self.score_residual_prior_paths(paths, batch_size=residual_batch_size)
        final_fixed = logit_blend(base, residual, self.config.gamma).astype(np.float32)
        return {
            "W": whole.astype(np.float32),
            "T": tile.astype(np.float32),
            "S": semantic.astype(np.float32),
            "R": residual.astype(np.float32),
            "max_side": max_side.astype(np.float32),
            "final_fixed": final_fixed,
        }

    def score_paths(self, paths: Sequence[str | Path], *, residual_batch_size: int = 32) -> np.ndarray:
        return self.score_component_paths(paths, residual_batch_size=residual_batch_size)["final_fixed"].astype(np.float32)
