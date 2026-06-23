#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.datasets import ImageSample, collect_labeled_images, limit_per_label, select_per_label
from data.manifests import load_image_samples_from_manifest
from models.core import CODEC_FAMILIES, family_indices, fit_codec_hgb_expert, fit_logistic_expert
from models.hgb_parity import aggregate_probabilities, extract_features
from networks.artifact_prior import ArtifactPriorFeatureExtractor, CodecTextureConfig
from networks.residual_prior import build_residual_prior_model, build_residual_prior_transform
from networks.semantic_prior import (
    extract_clip_features_resumable,
    fit_clip_linear_probe,
    load_openai_clip,
    parse_clip_variant_spec,
)
from utils.metrics import binary_metrics, write_target_report
from utils.progress import progress_iter


ImageFile.LOAD_TRUNCATED_IMAGES = True


def _safe_token(text: str) -> str:
    return str(text).replace("/", "_").replace(":", "_").replace(",", "_").replace("+", "p").replace("-", "m")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _family_slices(image_size: int) -> dict[str, slice]:
    extractor = ArtifactPriorFeatureExtractor(CodecTextureConfig(image_size=image_size))
    return dict(extractor.feature_family_slices(include_rollups=True))


def _fine_chroma_indices(families: dict[str, slice]) -> np.ndarray:
    chroma = families["chroma_luma_coupling"]
    start = int(chroma.start or 0)
    return np.arange(start + 10, start + 28, dtype=np.int64)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_source_training_samples(source_root: str | Path, train_manifest: str | Path | None = None) -> list[ImageSample]:
    if train_manifest is not None and str(train_manifest):
        return load_image_samples_from_manifest(train_manifest)
    return collect_labeled_images(source_root)


def train_artifact_prior(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    families = _family_slices(int(args.image_size))
    source_samples = load_source_training_samples(args.source_root, args.train_manifest or None)
    print(
        f"[artifact] source={args.source_root} train_per_label={int(args.train_per_label)} "
        f"variant={args.train_variant} batch_size={int(args.batch_size)} samples={len(source_samples)}",
        flush=True,
    )
    features, labels, feature_dim = extract_features(
        source_root=args.source_root,
        image_size=int(args.image_size),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=str(args.device),
        max_samples_per_label=int(args.train_per_label),
        train_variant=str(args.train_variant),
        samples=source_samples,
        show_progress=not bool(args.no_progress),
        progress_desc="artifact APSD features",
    )
    print(f"[artifact] fitting codec/chroma experts on {labels.shape[0]} feature rows", flush=True)
    codec = fit_codec_hgb_expert(
        features,
        labels,
        family_indices(families, CODEC_FAMILIES),
        CODEC_FAMILIES,
        max_iter=int(args.codec_max_iter),
        random_state=20260519,
    )
    chroma = fit_logistic_expert(
        features,
        labels,
        _fine_chroma_indices(families),
        ("chroma_ratio_corr_grad",),
        random_state=20260519,
    )
    model_path = out / "artifact_prior_models.joblib"
    payload = {
        "codec": codec,
        "chroma": chroma,
        "feature_dim": int(feature_dim),
        "image_size": int(args.image_size),
        "alpha": float(args.artifact_alpha),
        "target_labels_used": False,
    }
    joblib.dump(payload, model_path)
    protocol = {
        "stage": "artifact_prior",
        "source_root": str(Path(args.source_root).resolve(strict=False)),
        "train_per_label": int(args.train_per_label),
        "train_manifest": str(Path(args.train_manifest).resolve(strict=False)) if args.train_manifest else "",
        "train_sample_count": int(len(source_samples)),
        "train_variant": str(args.train_variant),
        "feature_dim": int(feature_dim),
        "artifact_alpha": float(args.artifact_alpha),
        "target_labels_used": False,
        "model_path": str(model_path.resolve(strict=False)),
    }
    (out / "training_protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(json.dumps(protocol, indent=2, sort_keys=True))


def _aggregate_variant_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    variants: tuple[str, ...],
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    view_count = len(variants)
    if view_count == 1:
        return labels.astype(np.int64), scores.astype(np.float32)
    score_matrix = scores.reshape(scores.shape[0] // view_count, view_count)
    label_matrix = labels.reshape(labels.shape[0] // view_count, view_count)
    if not np.all(label_matrix == label_matrix[:, :1]):
        raise ValueError("labels differ across semantic views")
    return label_matrix[:, 0].astype(np.int64), aggregate_probabilities(score_matrix, mode).astype(np.float32)


def train_semantic_prior(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = out / "feature_cache"
    score_cache = out / "score_cache"
    model_token = _safe_token(str(args.clip_model))
    train_variants = parse_clip_variant_spec(str(args.semantic_train_variants))
    holdout_variants = parse_clip_variant_spec(str(args.semantic_holdout_variants))
    eval_variants = parse_clip_variant_spec(str(args.semantic_eval_variants))
    model, preprocess = load_openai_clip(str(args.clip_model), device=str(args.device), download_root=str(args.clip_download_root))
    print(
        f"[semantic] source={args.source_root} train_per_label={int(args.train_per_label)} "
        f"holdout_per_label={int(args.holdout_per_label)} model={args.clip_model}",
        flush=True,
    )
    source_all = load_source_training_samples(args.source_root, args.train_manifest or None)
    source_train_all = source_all
    source_train = select_per_label(source_train_all, max_per_label=int(args.train_per_label), skip_per_label=0)
    if args.holdout_manifest:
        source_holdout = load_image_samples_from_manifest(args.holdout_manifest)
        if int(args.holdout_per_label) > 0:
            source_holdout = select_per_label(source_holdout, max_per_label=int(args.holdout_per_label), skip_per_label=0)
    else:
        source_holdout = select_per_label(
            source_all,
            max_per_label=int(args.holdout_per_label),
            skip_per_label=int(args.train_per_label),
        )
    train_features, train_labels, _, _ = extract_clip_features_resumable(
        cache_path=cache_dir / f"source_train_{model_token}_{int(args.train_per_label)}_{_safe_token(','.join(train_variants))}.npz",
        model=model,
        preprocess=preprocess,
        samples=source_train,
        image_size=int(args.image_size),
        variants=train_variants,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=str(args.device),
        cache_chunk_samples=int(args.cache_chunk_samples),
        show_progress=not bool(args.no_progress),
        progress_desc="semantic source-train CLIP",
    )
    holdout_features, holdout_labels_expanded, _, _ = extract_clip_features_resumable(
        cache_path=cache_dir / f"source_holdout_{model_token}_{int(args.holdout_per_label)}_{_safe_token(','.join(holdout_variants))}.npz",
        model=model,
        preprocess=preprocess,
        samples=source_holdout,
        image_size=int(args.image_size),
        variants=holdout_variants,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=str(args.device),
        cache_chunk_samples=int(args.cache_chunk_samples),
        show_progress=not bool(args.no_progress),
        progress_desc="semantic source-holdout CLIP",
    )
    print(f"[semantic] fitting linear probe on {train_labels.shape[0]} feature rows", flush=True)
    probe = fit_clip_linear_probe(
        train_features,
        train_labels,
        clip_model_name=str(args.clip_model),
        c=float(args.linear_c),
        random_state=int(args.random_state),
        train_config=vars(args),
    )
    model_path = out / "semantic_probe.joblib"
    joblib.dump(probe, model_path)
    holdout_scores_expanded = probe.predict_proba(holdout_features)
    holdout_labels, holdout_scores = _aggregate_variant_scores(
        holdout_scores_expanded,
        holdout_labels_expanded,
        holdout_variants,
        str(args.tta_aggregation),
    )
    source_holdout_metrics = binary_metrics(holdout_labels, holdout_scores)
    target_mean = None
    if not bool(args.skip_target_report):
        packed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        all_target = collect_labeled_images(args.target_root)
        target_groups = sorted({sample.group for sample in all_target})
        for group in progress_iter(
            target_groups,
            total=len(target_groups),
            desc="semantic target report",
            unit="gen",
            enabled=not bool(args.no_progress),
        ):
            group_samples = limit_per_label([sample for sample in all_target if sample.group == group], int(args.target_per_label))
            cache_path = score_cache / f"{group}_{model_token}_{_safe_token(','.join(eval_variants))}.npz"
            if cache_path.exists():
                cached = np.load(cache_path, allow_pickle=False)
                labels = cached["labels"].astype(np.int64)
                scores = cached["scores"].astype(np.float32)
            else:
                features, expanded_labels, _, _ = extract_clip_features_resumable(
                    cache_path=cache_dir / f"target_{group}_{model_token}_{int(args.target_per_label)}_{_safe_token(','.join(eval_variants))}.npz",
                    model=model,
                    preprocess=preprocess,
                    samples=group_samples,
                    image_size=int(args.image_size),
                    variants=eval_variants,
                    batch_size=int(args.batch_size),
                    num_workers=int(args.num_workers),
                    device=str(args.device),
                    cache_chunk_samples=int(args.cache_chunk_samples),
                    show_progress=not bool(args.no_progress),
                    progress_desc=f"semantic target {group} CLIP",
                )
                expanded_scores = probe.predict_proba(features)
                labels, scores = _aggregate_variant_scores(expanded_scores, expanded_labels, eval_variants, str(args.tta_aggregation))
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache_path, labels=labels, scores=scores)
            packed[group] = (labels, scores)
        target_mean = write_target_report(out / "target_report", packed, 0.5)
    protocol = {
        "stage": "semantic_prior",
        "model_path": str(model_path.resolve(strict=False)),
        "clip_model": str(args.clip_model),
        "train_per_label": int(args.train_per_label),
        "holdout_per_label": int(args.holdout_per_label),
        "train_manifest": str(Path(args.train_manifest).resolve(strict=False)) if args.train_manifest else "",
        "holdout_manifest": str(Path(args.holdout_manifest).resolve(strict=False)) if args.holdout_manifest else "",
        "train_sample_count": int(len(source_train)),
        "holdout_sample_count": int(len(source_holdout)),
        "source_holdout": source_holdout_metrics,
        "target_mean": target_mean,
        "target_labels_used_for_selection": False,
        "target_labels_used_for_final_metrics_only": target_mean is not None,
    }
    (out / "training_protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    _write_csv(out / "summary.csv", [{"split": "source_holdout", **source_holdout_metrics}])
    print(json.dumps(protocol, indent=2, sort_keys=True))


class ResidualSourceDataset(Dataset):
    def __init__(self, samples: list[ImageSample], *, image_size: int, train: bool) -> None:
        self.samples = list(samples)
        self.transform = build_residual_prior_transform(image_size=int(image_size), train=bool(train))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, torch.tensor(int(sample.label), dtype=torch.float32), str(sample.path.resolve(strict=False))


def train_residual_prior(args: argparse.Namespace) -> None:
    _seed_all(int(args.random_state))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    samples = load_source_training_samples(args.source_root, args.train_manifest or None)
    if int(args.max_samples_per_label) > 0:
        samples = limit_per_label(samples, int(args.max_samples_per_label))
    if {sample.label for sample in samples} != {0, 1}:
        raise ValueError("residual prior source training requires both labels")
    print(
        f"[residual] source={args.source_root} samples={len(samples)} epochs={int(args.epochs)} "
        f"batch_size={int(args.batch_size)} image_size={int(args.residual_train_image_size)}",
        flush=True,
    )
    device = torch.device(args.device if not str(args.device).startswith("cuda") or torch.cuda.is_available() else "cpu")
    dataset = ResidualSourceDataset(samples, image_size=int(args.residual_train_image_size), train=True)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=(device.type != "cpu"),
        drop_last=False,
    )
    model = build_residual_prior_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.residual_lr), weight_decay=float(args.weight_decay))
    loss_fn = torch.nn.BCEWithLogitsLoss()
    progress_path = out / "progress.jsonl"
    progress_path.write_text("")
    for epoch in range(int(args.epochs)):
        losses: list[float] = []
        model.train()
        max_batches = len(loader)
        if int(args.max_steps) > 0:
            max_batches = min(max_batches, int(args.max_steps))
        epoch_iter = progress_iter(
            loader,
            total=max_batches,
            desc=f"residual epoch {epoch + 1}/{int(args.epochs)}",
            unit="batch",
            enabled=not bool(args.no_progress),
        )
        for step, (images, labels, _paths) in enumerate(epoch_iter, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images).reshape(labels.shape[0], -1)[:, 0]
            loss = loss_fn(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            if int(args.max_steps) > 0 and step >= int(args.max_steps):
                break
        checkpoint = {
            "model": model.state_dict(),
            "epoch": int(epoch),
            "clean_protocol": {
                "source_root": str(Path(args.source_root).resolve(strict=False)),
                "train_manifest": str(Path(args.train_manifest).resolve(strict=False)) if args.train_manifest else "",
                "residual_train_image_size": int(args.residual_train_image_size),
                "random_state": int(args.random_state),
                "target_labels_used": False,
            },
        }
        checkpoint_path = out / f"checkpoint-{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        row = {
            "epoch": int(epoch),
            "train_loss": float(sum(losses) / max(len(losses), 1)),
            "checkpoint": str(checkpoint_path.resolve(strict=False)),
            "target_labels_used": False,
        }
        with progress_path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser("Train unified detector internal components from source-only data")
    parser.add_argument("--stage", choices=("artifact", "semantic", "residual"), required=True)
    parser.add_argument("--source_root", default=str(PROJECT_ROOT / "dataset" / "train_100k" / "progan_train"))
    parser.add_argument("--target_root", default=str(PROJECT_ROOT / "dataset" / "AIGCDetectBenchmark_test"))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train_per_label", type=int, default=5000)
    parser.add_argument("--holdout_per_label", type=int, default=5000)
    parser.add_argument("--target_per_label", type=int, default=200)
    parser.add_argument("--train_variant", default="expand:clean,jpeg50,jpeg50,resize50,blur1")
    parser.add_argument("--codec_max_iter", type=int, default=200)
    parser.add_argument("--artifact_alpha", type=float, default=-0.40)
    parser.add_argument("--clip_model", default="ViT-L/14")
    parser.add_argument("--clip_download_root", default="/data/lizihao/.cache/clip")
    parser.add_argument("--cache_chunk_samples", type=int, default=512)
    parser.add_argument("--semantic_train_variants", default="clean")
    parser.add_argument("--semantic_holdout_variants", default="clean")
    parser.add_argument("--semantic_eval_variants", default="clean")
    parser.add_argument("--tta_aggregation", choices=("mean_prob", "mean_logit"), default="mean_logit")
    parser.add_argument("--linear_c", type=float, default=1.0)
    parser.add_argument("--skip_target_report", action="store_true")
    parser.add_argument("--train_manifest", default="")
    parser.add_argument("--holdout_manifest", default="")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--residual_train_image_size", type=int, default=256)
    parser.add_argument("--residual_lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_samples_per_label", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--random_state", type=int, default=20260519)
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args()
    if args.stage == "artifact":
        train_artifact_prior(args)
    elif args.stage == "semantic":
        train_semantic_prior(args)
    elif args.stage == "residual":
        train_residual_prior(args)
    else:
        raise ValueError(args.stage)


if __name__ == "__main__":
    main()
