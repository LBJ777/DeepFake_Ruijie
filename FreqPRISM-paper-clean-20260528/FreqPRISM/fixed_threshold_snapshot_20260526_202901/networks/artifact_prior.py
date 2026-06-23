from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from functools import cached_property

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class CodecTextureConfig:
    image_size: int = 256
    dct_block: int = 8
    fft_bands: int = 4
    feature_mode: str = "codec_texture"


@dataclass(frozen=True)
class ArtifactPriorOutput:
    logit: Tensor
    features: Tensor


@dataclass(frozen=True)
class FeatureFamily:
    name: str
    start: int
    stop: int
    kind: str = "base"

    @property
    def dim(self) -> int:
        return int(self.stop - self.start)

    @property
    def slice(self) -> slice:
        return slice(int(self.start), int(self.stop))


def _validate_rgb(images: Tensor) -> None:
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must be a BCHW RGB tensor")


def _kernel(values: list[list[float]], device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.tensor(values, device=device, dtype=dtype).reshape(1, 1, len(values), len(values[0]))


def _pad(tensor: Tensor, padding: int | tuple[int, int, int, int]) -> Tensor:
    if isinstance(padding, int):
        pad = (padding, padding, padding, padding)
    else:
        pad = padding
    return F.pad(tensor, pad, mode="reflect" if min(tensor.shape[-2:]) > max(pad) else "replicate")


def _view_stats(view: Tensor) -> Tensor:
    flat = view.flatten(2).float()
    quantiles = torch.quantile(
        flat,
        torch.tensor([0.1, 0.5, 0.9], device=flat.device, dtype=flat.dtype),
        dim=-1,
    ).permute(1, 2, 0)
    return torch.cat(
        [
            flat.mean(dim=-1),
            flat.std(dim=-1, unbiased=False),
            flat.abs().mean(dim=-1),
            flat.amax(dim=-1),
            flat.amin(dim=-1),
            quantiles.reshape(flat.shape[0], -1),
        ],
        dim=1,
    )


def _flat_stats(values: Tensor) -> Tensor:
    flat = values.flatten(1).float()
    quantiles = torch.quantile(
        flat,
        torch.tensor([0.5, 0.9], device=flat.device, dtype=flat.dtype),
        dim=1,
    ).permute(1, 0)
    return torch.cat(
        [
            flat.mean(dim=1, keepdim=True),
            flat.std(dim=1, unbiased=False, keepdim=True),
            flat.amax(dim=1, keepdim=True),
            quantiles,
        ],
        dim=1,
    )


def _boundary_features(diff: Tensor, period: int, axis: str) -> Tensor:
    if axis == "x":
        positions = torch.arange(diff.shape[-1], device=diff.device)
        mask = ((positions + 1) % int(period) == 0).reshape(1, 1, 1, diff.shape[-1])
    elif axis == "y":
        positions = torch.arange(diff.shape[-2], device=diff.device)
        mask = ((positions + 1) % int(period) == 0).reshape(1, 1, diff.shape[-2], 1)
    else:
        raise ValueError(f"unsupported boundary axis: {axis}")
    if not bool(mask.any()) or not bool((~mask).any()):
        return torch.zeros(diff.shape[0], 3, device=diff.device, dtype=diff.dtype)
    boundary = diff.masked_select(mask.expand_as(diff)).reshape(diff.shape[0], -1)
    off_boundary = diff.masked_select((~mask).expand_as(diff)).reshape(diff.shape[0], -1)
    boundary_mean = boundary.mean(dim=1, keepdim=True)
    off_mean = off_boundary.mean(dim=1, keepdim=True)
    return torch.cat([boundary_mean, off_mean, boundary_mean / off_mean.clamp_min(1e-6)], dim=1)


def _color_correlations(images: Tensor) -> Tensor:
    flat = images.flatten(2).float()
    centered = flat - flat.mean(dim=2, keepdim=True)
    denom = centered.std(dim=2, unbiased=False).clamp_min(1e-6)
    rg = (centered[:, 0] * centered[:, 1]).mean(dim=1) / (denom[:, 0] * denom[:, 1])
    rb = (centered[:, 0] * centered[:, 2]).mean(dim=1) / (denom[:, 0] * denom[:, 2])
    gb = (centered[:, 1] * centered[:, 2]).mean(dim=1) / (denom[:, 1] * denom[:, 2])
    return torch.stack([rg, rb, gb], dim=1)


def _phase_mean_stats(diff: Tensor, period: int, axis: str) -> Tensor:
    if axis == "x":
        positions = torch.arange(diff.shape[-1], device=diff.device)
        phase_view = (1, 1, 1, diff.shape[-1])
    elif axis == "y":
        positions = torch.arange(diff.shape[-2], device=diff.device)
        phase_view = (1, 1, diff.shape[-2], 1)
    else:
        raise ValueError(f"unsupported phase axis: {axis}")
    means = []
    for phase in range(int(period)):
        mask = (positions % int(period) == phase).reshape(phase_view)
        selected = diff.masked_select(mask.expand_as(diff)).reshape(diff.shape[0], -1)
        means.append(selected.mean(dim=1, keepdim=True))
    values = torch.cat(means, dim=1)
    values = values / values.mean(dim=1, keepdim=True).clamp_min(1e-6)
    return _flat_stats(values.reshape(values.shape[0], 1, 1, values.shape[1]))


def _patch_mean_std_stats(values: Tensor, grid: int) -> Tensor:
    values_f = values.float()
    patch_mean = F.adaptive_avg_pool2d(values_f, output_size=(grid, grid))
    patch_square_mean = F.adaptive_avg_pool2d(values_f.square(), output_size=(grid, grid))
    patch_std = (patch_square_mean - patch_mean.square()).clamp_min(0.0).sqrt()
    return torch.cat([_flat_stats(patch_mean), _flat_stats(patch_std)], dim=1)


def _neighbor_correlation(values: Tensor, shift_y: int, shift_x: int) -> Tensor:
    height, width = values.shape[-2:]
    if shift_y >= height or shift_x >= width:
        return torch.zeros(values.shape[0], 1, device=values.device, dtype=values.dtype)
    first = values[..., : height - shift_y, : width - shift_x].flatten(1).float()
    second = values[..., shift_y:, shift_x:].flatten(1).float()
    first = first - first.mean(dim=1, keepdim=True)
    second = second - second.mean(dim=1, keepdim=True)
    denom = first.std(dim=1, unbiased=False).clamp_min(1e-6) * second.std(dim=1, unbiased=False).clamp_min(1e-6)
    return ((first * second).mean(dim=1) / denom).unsqueeze(1)


def _make_dct_kernels(block: int, device: torch.device, dtype: torch.dtype, pairs: tuple[tuple[int, int], ...]) -> Tensor:
    coords = torch.arange(block, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernels = []
    for u, v in pairs:
        alpha_u = math.sqrt(1.0 / block) if u == 0 else math.sqrt(2.0 / block)
        alpha_v = math.sqrt(1.0 / block) if v == 0 else math.sqrt(2.0 / block)
        kernel = alpha_u * alpha_v
        kernel = kernel * torch.cos(((2.0 * yy + 1.0) * u * math.pi) / (2.0 * block))
        kernel = kernel * torch.cos(((2.0 * xx + 1.0) * v * math.pi) / (2.0 * block))
        kernels.append(kernel)
    return torch.stack(kernels, dim=0)


def _rgb_to_ycbcr(images: Tensor) -> Tensor:
    r, g, b = images[:, 0:1], images[:, 1:2], images[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b
    return torch.cat([y, cb, cr], dim=1)


class ArtifactPriorFeatureExtractor(nn.Module):
    view_channels = {"rgb_light": 3, "residual": 3, "highpass": 6, "dct_local": 12, "fft_global": 4}

    def __init__(self, config: CodecTextureConfig | None = None) -> None:
        super().__init__()
        self.config = config or CodecTextureConfig()
        if self.config.dct_block < 2:
            raise ValueError("dct_block must be at least 2")
        if self.config.feature_mode != "codec_texture":
            raise ValueError("only codec_texture feature_mode is implemented for APSD V0")

    @property
    def feature_dim(self) -> int:
        sample = torch.zeros(1, 3, int(self.config.image_size), int(self.config.image_size))
        with torch.no_grad():
            return int(self(sample).shape[1])

    @cached_property
    def feature_families(self) -> OrderedDict[str, FeatureFamily]:
        sample = torch.zeros(1, 3, int(self.config.image_size), int(self.config.image_size))
        blocks: list[tuple[str, Tensor]] = []
        with torch.no_grad():
            views = self._views(sample)
            for name, view in views.items():
                blocks.append((f"view_{name}", _view_stats(view)))
            blocks.append(("rich_spatial", self._rich_spatial_stats(sample)))
            blocks.append(("texture_artifact", self._texture_artifact_stats(sample)))
            recompression_blocks = self._recompression_stability_blocks(sample)
            for name, values in recompression_blocks.items():
                if name == "recompression_stability":
                    continue
                blocks.append((name, values))
            blocks.append(("residual_cooccurrence", self._residual_cooccurrence_stats(sample)))
            blocks.append(("residual_spectrum", self._residual_spectrum_stats(sample)))
            blocks.append(("residual_tail_shape", self._residual_tail_shape_stats(sample)))
            blocks.append(("chroma_luma_coupling", self._chroma_luma_coupling_stats(sample)))
            blocks.append(("patch_spectrum_heterogeneity", self._patch_spectrum_heterogeneity_stats(sample)))
            blocks.append(("codec_block", self._codec_block_stats(sample)))
        families: OrderedDict[str, FeatureFamily] = OrderedDict()
        cursor = 0
        for name, values in blocks:
            width = int(values.shape[1])
            families[name] = FeatureFamily(name=name, start=cursor, stop=cursor + width)
            cursor += width
        families["transfer_core"] = FeatureFamily(
            name="transfer_core",
            start=families["rich_spatial"].start,
            stop=families["recompression_q32_resize50"].stop,
            kind="rollup",
        )
        families["recompression_stability"] = FeatureFamily(
            name="recompression_stability",
            start=families["recompression_q32"].start,
            stop=families["recompression_q32_resize50"].stop,
            kind="rollup",
        )
        families["rich_stride_stats"] = FeatureFamily(
            name="rich_stride_stats",
            start=families["rich_spatial"].start,
            stop=families["rich_spatial"].start + 30,
            kind="rollup",
        )
        families["rich_boundary_p8"] = FeatureFamily(
            name="rich_boundary_p8",
            start=families["rich_spatial"].start + 30,
            stop=families["rich_spatial"].start + 36,
            kind="rollup",
        )
        families["rich_boundary_p16"] = FeatureFamily(
            name="rich_boundary_p16",
            start=families["rich_spatial"].start + 36,
            stop=families["rich_spatial"].start + 42,
            kind="rollup",
        )
        families["rich_local_residual"] = FeatureFamily(
            name="rich_local_residual",
            start=families["rich_spatial"].start + 42,
            stop=families["rich_spatial"].start + 47,
            kind="rollup",
        )
        families["rich_color_corr"] = FeatureFamily(
            name="rich_color_corr",
            start=families["rich_spatial"].start + 47,
            stop=families["rich_spatial"].stop,
            kind="rollup",
        )
        families["texture_patch_grid4"] = FeatureFamily(
            name="texture_patch_grid4",
            start=families["texture_artifact"].start,
            stop=families["texture_artifact"].start + 40,
            kind="rollup",
        )
        families["texture_patch_grid8"] = FeatureFamily(
            name="texture_patch_grid8",
            start=families["texture_artifact"].start + 40,
            stop=families["texture_artifact"].start + 80,
            kind="rollup",
        )
        families["texture_neighbor"] = FeatureFamily(
            name="texture_neighbor",
            start=families["texture_artifact"].start + 80,
            stop=families["texture_artifact"].start + 86,
            kind="rollup",
        )
        families["texture_phase"] = FeatureFamily(
            name="texture_phase",
            start=families["texture_artifact"].start + 86,
            stop=families["texture_artifact"].start + 116,
            kind="rollup",
        )
        families["texture_fft_angular"] = FeatureFamily(
            name="texture_fft_angular",
            start=families["texture_artifact"].start + 116,
            stop=families["texture_artifact"].start + 133,
            kind="rollup",
        )
        families["texture_color_corr"] = FeatureFamily(
            name="texture_color_corr",
            start=families["texture_artifact"].start + 133,
            stop=families["texture_artifact"].start + 136,
            kind="rollup",
        )
        families["texture_chroma_ratio"] = FeatureFamily(
            name="texture_chroma_ratio",
            start=families["texture_artifact"].start + 136,
            stop=families["texture_artifact"].stop,
            kind="rollup",
        )
        families["codec_dct_low"] = FeatureFamily(
            name="codec_dct_low",
            start=families["codec_block"].start,
            stop=families["codec_block"].start + 24,
            kind="rollup",
        )
        families["codec_dct_mid"] = FeatureFamily(
            name="codec_dct_mid",
            start=families["codec_block"].start + 24,
            stop=families["codec_block"].start + 48,
            kind="rollup",
        )
        families["codec_dct_high"] = FeatureFamily(
            name="codec_dct_high",
            start=families["codec_block"].start + 48,
            stop=families["codec_block"].start + 72,
            kind="rollup",
        )
        families["codec_energy_ratio"] = FeatureFamily(
            name="codec_energy_ratio",
            start=families["codec_block"].start + 72,
            stop=families["codec_block"].start + 120,
            kind="rollup",
        )
        families["codec_jpeg_grid_y"] = FeatureFamily(
            name="codec_jpeg_grid_y",
            start=families["codec_block"].start + 120,
            stop=families["codec_block"].start + 136,
            kind="rollup",
        )
        families["codec_jpeg_grid_cb"] = FeatureFamily(
            name="codec_jpeg_grid_cb",
            start=families["codec_block"].start + 136,
            stop=families["codec_block"].start + 152,
            kind="rollup",
        )
        families["codec_jpeg_grid_cr"] = FeatureFamily(
            name="codec_jpeg_grid_cr",
            start=families["codec_block"].start + 152,
            stop=families["codec_block"].stop,
            kind="rollup",
        )
        families["all"] = FeatureFamily(name="all", start=0, stop=cursor, kind="rollup")
        return families

    def feature_family_slices(self, include_rollups: bool = False) -> OrderedDict[str, slice]:
        return OrderedDict(
            (name, family.slice)
            for name, family in self.feature_families.items()
            if include_rollups or family.kind == "base"
        )

    def forward(self, images: Tensor) -> Tensor:
        _validate_rgb(images)
        views = self._views(images)
        features = [_view_stats(view) for view in views.values()]
        features.append(self._rich_spatial_stats(images))
        features.append(self._texture_artifact_stats(images))
        features.extend(
            values
            for name, values in self._recompression_stability_blocks(images).items()
            if name != "recompression_stability"
        )
        features.append(self._residual_cooccurrence_stats(images))
        features.append(self._residual_spectrum_stats(images))
        features.append(self._residual_tail_shape_stats(images))
        features.append(self._chroma_luma_coupling_stats(images))
        features.append(self._patch_spectrum_heterogeneity_stats(images))
        features.append(self._codec_block_stats(images))
        return torch.nan_to_num(torch.cat(features, dim=1), nan=0.0, posinf=1e6, neginf=-1e6)

    def _views(self, images: Tensor) -> OrderedDict[str, Tensor]:
        return OrderedDict(
            [
                ("rgb_light", images),
                ("residual", self._residual(images)),
                ("highpass", self._highpass(images)),
                ("dct_local", self._dct_local(images)),
                ("fft_global", self._fft_global(images)),
            ]
        )

    def _residual(self, images: Tensor) -> Tensor:
        local_mean = F.avg_pool2d(images, kernel_size=5, stride=1, padding=2, count_include_pad=False)
        return images - local_mean

    def _highpass(self, images: Tensor) -> Tensor:
        gray = images.mean(dim=1, keepdim=True)
        sobel_x = _kernel([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], images.device, images.dtype) / 8.0
        sobel_y = _kernel([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], images.device, images.dtype) / 8.0
        laplace = _kernel([[0, 1, 0], [1, -4, 1], [0, 1, 0]], images.device, images.dtype) / 4.0
        edge_x = F.conv2d(_pad(gray, 1), sobel_x)
        edge_y = F.conv2d(_pad(gray, 1), sobel_y)
        edge_mag = torch.sqrt(edge_x.float().square() + edge_y.float().square() + 1e-12).to(dtype=images.dtype)
        rgb_laplace = F.conv2d(_pad(images, 1), laplace.repeat(3, 1, 1, 1), groups=3)
        return torch.cat([edge_x, edge_y, edge_mag, rgb_laplace], dim=1)

    def _dct_local(self, images: Tensor) -> Tensor:
        block = int(self.config.dct_block)
        pairs = ((0, 1), (1, 0), (1, 1), (2, 0))
        basis = _make_dct_kernels(block, images.device, images.dtype, pairs)
        basis = basis - basis.mean(dim=(-2, -1), keepdim=True)
        basis = basis / basis.abs().sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        weight = basis.unsqueeze(1).repeat(3, 1, 1, 1)
        pad_total = block - 1
        padded = _pad(images, (pad_total // 2, pad_total - pad_total // 2, pad_total // 2, pad_total - pad_total // 2))
        coeffs = F.conv2d(padded, weight, groups=3).abs()
        return torch.log1p(coeffs * float(block))

    def _fft_global(self, images: Tensor) -> Tensor:
        gray = images.float().mean(dim=1, keepdim=True)
        fft = torch.fft.fftshift(torch.fft.fft2(gray, norm="ortho"), dim=(-2, -1))
        magnitude = torch.log1p(torch.abs(fft))
        magnitude = magnitude / magnitude.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        height, width = gray.shape[-2:]
        yy = torch.fft.fftshift(torch.fft.fftfreq(height, device=images.device)).float()
        xx = torch.fft.fftshift(torch.fft.fftfreq(width, device=images.device)).float()
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        radius = torch.sqrt(grid_x.square() + grid_y.square())
        radius = radius / radius.amax().clamp_min(1e-6)
        bands = []
        edges = torch.linspace(0.0, 1.0, steps=int(self.config.fft_bands) + 1, device=images.device)
        for index in range(int(self.config.fft_bands)):
            mask = (radius >= edges[index]) & (radius <= edges[index + 1] if index == int(self.config.fft_bands) - 1 else radius < edges[index + 1])
            mask_f = mask.reshape(1, 1, height, width).to(magnitude.dtype)
            value = (magnitude * mask_f).sum(dim=(-2, -1), keepdim=True) / mask_f.sum().clamp_min(1.0)
            bands.append(value.expand(-1, 1, height, width))
        return torch.cat(bands, dim=1).to(dtype=images.dtype)

    def _rich_spatial_stats(self, images: Tensor) -> Tensor:
        gray = images.float().mean(dim=1, keepdim=True)
        features = []
        for stride in (1, 2, 4):
            features.append(_flat_stats((gray[..., stride:] - gray[..., :-stride]).abs()))
            features.append(_flat_stats((gray[..., stride:, :] - gray[..., :-stride, :]).abs()))
        hdiff1 = (gray[..., 1:] - gray[..., :-1]).abs()
        vdiff1 = (gray[..., 1:, :] - gray[..., :-1, :]).abs()
        for period in (8, 16):
            features.append(_boundary_features(hdiff1, period=period, axis="x"))
            features.append(_boundary_features(vdiff1, period=period, axis="y"))
        local_mean = F.avg_pool2d(gray, kernel_size=5, stride=1, padding=2, count_include_pad=False)
        features.append(_flat_stats((gray - local_mean).abs()))
        features.append(_color_correlations(images.float()))
        return torch.cat(features, dim=1)

    def _fft_angular_features(self, images: Tensor) -> Tensor:
        gray = images.float().mean(dim=1, keepdim=True)
        fft = torch.fft.fftshift(torch.fft.fft2(gray, norm="ortho"), dim=(-2, -1))
        magnitude = torch.log1p(torch.abs(fft))
        height, width = gray.shape[-2:]
        yy = torch.fft.fftshift(torch.fft.fftfreq(height, device=images.device)).float()
        xx = torch.fft.fftshift(torch.fft.fftfreq(width, device=images.device)).float()
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        radius = torch.sqrt(grid_x.square() + grid_y.square())
        angle = torch.atan2(grid_y, grid_x)
        valid = radius > 0.08
        sectors = []
        for sector in range(8):
            lower = -math.pi + sector * (2.0 * math.pi / 8.0)
            upper = -math.pi + (sector + 1) * (2.0 * math.pi / 8.0)
            mask = valid & (angle >= lower) & (angle <= upper if sector == 7 else angle < upper)
            mask_f = mask.reshape(1, 1, height, width).to(magnitude.dtype)
            sectors.append((magnitude * mask_f).sum(dim=(-2, -1)) / mask_f.sum().clamp_min(1.0))
        sector_values = torch.cat(sectors, dim=1)
        sector_ratio = sector_values / sector_values.sum(dim=1, keepdim=True).clamp_min(1e-6)
        entropy = -(sector_ratio.clamp_min(1e-8) * sector_ratio.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
        radial = []
        for lower, upper in ((0.05, 0.18), (0.18, 0.34), (0.34, 0.72)):
            mask = (radius >= lower) & (radius < upper)
            mask_f = mask.reshape(1, 1, height, width).to(magnitude.dtype)
            radial.append((magnitude * mask_f).sum(dim=(-2, -1)) / mask_f.sum().clamp_min(1.0))
        low, mid, high = [item.reshape(images.shape[0], 1) for item in radial]
        radial_ratios = torch.cat([mid / low.clamp_min(1e-6), high / mid.clamp_min(1e-6), high / low.clamp_min(1e-6)], dim=1)
        return torch.cat([sector_ratio, _flat_stats(sector_ratio.reshape(sector_ratio.shape[0], 1, 1, -1)), entropy, radial_ratios], dim=1)

    def _texture_artifact_stats(self, images: Tensor) -> Tensor:
        images_f = images.float()
        gray = images_f.mean(dim=1, keepdim=True)
        local_mean = F.avg_pool2d(gray, kernel_size=7, stride=1, padding=3, count_include_pad=False)
        gray_residual = gray - local_mean
        hdiff = F.pad((gray[..., 1:] - gray[..., :-1]).abs(), (0, 1, 0, 0), mode="replicate")
        vdiff = F.pad((gray[..., 1:, :] - gray[..., :-1, :]).abs(), (0, 0, 0, 1), mode="replicate")
        edge_mean = 0.5 * (hdiff + vdiff)
        rgb_mean = F.avg_pool2d(images_f, kernel_size=5, stride=1, padding=2, count_include_pad=False)
        rgb_residual = images_f - rgb_mean
        luma_residual = rgb_residual.mean(dim=1, keepdim=True)
        chroma_residual = (rgb_residual - luma_residual).abs().mean(dim=1, keepdim=True)
        features = []
        for grid in (4, 8):
            for patch_map in (gray_residual.abs(), edge_mean, luma_residual.abs(), chroma_residual):
                features.append(_patch_mean_std_stats(patch_map, grid=grid))
        for values in (gray_residual, chroma_residual):
            for shift_y, shift_x in ((0, 1), (1, 0), (1, 1)):
                features.append(_neighbor_correlation(values, shift_y=shift_y, shift_x=shift_x))
        hdiff1 = (gray[..., 1:] - gray[..., :-1]).abs()
        vdiff1 = (gray[..., 1:, :] - gray[..., :-1, :]).abs()
        for period in (4, 8, 16):
            features.append(_phase_mean_stats(hdiff1, period=period, axis="x"))
            features.append(_phase_mean_stats(vdiff1, period=period, axis="y"))
        features.append(self._fft_angular_features(images_f))
        features.append(_color_correlations(rgb_residual))
        features.append(_flat_stats(chroma_residual / luma_residual.abs().clamp_min(1e-6)))
        return torch.cat(features, dim=1)

    def _residual_cooccurrence_stats(self, images: Tensor) -> Tensor:
        gray = images.float().mean(dim=1, keepdim=True)
        local = F.avg_pool2d(gray, kernel_size=5, stride=1, padding=2, count_include_pad=False)
        residual = gray - local
        scale = residual.flatten(1).std(dim=1, unbiased=False).reshape(-1, 1, 1, 1).clamp_min(1e-6)
        normalized = (residual / scale).clamp(-3.0, 3.0)
        bins = torch.zeros_like(normalized, dtype=torch.long)
        bins = bins + (normalized >= -0.5).long()
        bins = bins + (normalized >= 0.0).long()
        bins = bins + (normalized >= 0.5).long()
        features = []
        for shift_y, shift_x in ((0, 1), (1, 0), (1, 1), (0, 2), (2, 0)):
            first = bins[..., : bins.shape[-2] - shift_y if shift_y else bins.shape[-2], : bins.shape[-1] - shift_x if shift_x else bins.shape[-1]]
            second = bins[..., shift_y:, shift_x:]
            joint = first * 4 + second
            one_hot = F.one_hot(joint.reshape(joint.shape[0], -1), num_classes=16).float()
            probs = one_hot.mean(dim=1)
            matrix = probs.reshape(probs.shape[0], 4, 4)
            entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
            diagonal = matrix.diagonal(dim1=1, dim2=2).sum(dim=1, keepdim=True)
            anti_diagonal = torch.stack([matrix[:, 0, 3], matrix[:, 1, 2], matrix[:, 2, 1], matrix[:, 3, 0]], dim=1).sum(dim=1, keepdim=True)
            sign_agree = matrix[:, :2, :2].sum(dim=(1, 2), keepdim=True).reshape(-1, 1) + matrix[:, 2:, 2:].sum(dim=(1, 2), keepdim=True).reshape(-1, 1)
            features.append(torch.cat([probs, entropy, diagonal, anti_diagonal, sign_agree], dim=1))
        return torch.cat(features, dim=1)

    def _residual_spectrum_stats(self, images: Tensor) -> Tensor:
        gray = images.float().mean(dim=1, keepdim=True)
        local = F.avg_pool2d(gray, kernel_size=7, stride=1, padding=3, count_include_pad=False)
        residual = gray - local
        fft = torch.fft.fftshift(torch.fft.fft2(residual, norm="ortho"), dim=(-2, -1))
        magnitude = torch.log1p(torch.abs(fft))
        height, width = gray.shape[-2:]
        yy = torch.fft.fftshift(torch.fft.fftfreq(height, device=images.device)).float()
        xx = torch.fft.fftshift(torch.fft.fftfreq(width, device=images.device)).float()
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        radius = torch.sqrt(grid_x.square() + grid_y.square())
        radius = radius / radius.amax().clamp_min(1e-6)
        band_values = []
        edges = torch.linspace(0.04, 1.0, steps=9, device=images.device)
        for index in range(8):
            mask = (radius >= edges[index]) & (radius < edges[index + 1] if index < 7 else radius <= edges[index + 1])
            mask_f = mask.reshape(1, 1, height, width).to(magnitude.dtype)
            band_values.append(((magnitude * mask_f).sum(dim=(-2, -1)) / mask_f.sum().clamp_min(1.0)).reshape(images.shape[0], 1))
        bands = torch.cat(band_values, dim=1)
        band_ratio = bands / bands.sum(dim=1, keepdim=True).clamp_min(1e-6)
        adjacent = band_ratio[:, 1:] / band_ratio[:, :-1].clamp_min(1e-6)
        high_low = band_ratio[:, -2:].sum(dim=1, keepdim=True) / band_ratio[:, :2].sum(dim=1, keepdim=True).clamp_min(1e-6)
        entropy = -(band_ratio.clamp_min(1e-8) * band_ratio.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
        return torch.cat([band_ratio, adjacent, high_low, entropy, _flat_stats(band_ratio.reshape(band_ratio.shape[0], 1, 1, -1))], dim=1)

    def _residual_tail_shape_stats(self, images: Tensor) -> Tensor:
        gray = images.float().mean(dim=1, keepdim=True)
        local = F.avg_pool2d(gray, kernel_size=7, stride=1, padding=3, count_include_pad=False)
        residual = (gray - local).flatten(1).float()
        abs_residual = residual.abs()
        scale = abs_residual.median(dim=1, keepdim=True).values.clamp_min(1e-6)
        normalized = residual / scale
        abs_normalized = normalized.abs()
        quantiles = torch.quantile(
            normalized,
            torch.tensor([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99], device=images.device, dtype=normalized.dtype),
            dim=1,
        ).permute(1, 0)
        abs_quantiles = torch.quantile(
            abs_normalized,
            torch.tensor([0.5, 0.75, 0.9, 0.95, 0.99], device=images.device, dtype=normalized.dtype),
            dim=1,
        ).permute(1, 0)
        tail_rates = torch.cat(
            [
                (abs_normalized > threshold).float().mean(dim=1, keepdim=True)
                for threshold in (1.0, 2.0, 3.0, 4.0)
            ],
            dim=1,
        )
        centered = normalized - normalized.mean(dim=1, keepdim=True)
        std = centered.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
        skew = (centered / std).pow(3).mean(dim=1, keepdim=True)
        kurtosis = (centered / std).pow(4).mean(dim=1, keepdim=True)
        iqr = (quantiles[:, 5:6] - quantiles[:, 3:4]).abs()
        robust_tail = (abs_quantiles[:, -1:] / abs_quantiles[:, 0:1].clamp_min(1e-6))
        return torch.cat([quantiles, abs_quantiles, tail_rates, skew, kurtosis, iqr, robust_tail], dim=1)

    def _chroma_luma_coupling_stats(self, images: Tensor) -> Tensor:
        ycbcr = _rgb_to_ycbcr(images.float())
        local = F.avg_pool2d(ycbcr, kernel_size=7, stride=1, padding=3, count_include_pad=False)
        residual = ycbcr - local
        y = residual[:, 0:1]
        cb = residual[:, 1:2]
        cr = residual[:, 2:3]
        chroma_abs = 0.5 * (cb.abs() + cr.abs())
        luma_abs = y.abs()
        features = [
            _flat_stats(luma_abs),
            _flat_stats(chroma_abs),
            _flat_stats(chroma_abs / luma_abs.clamp_min(1e-6)),
        ]
        flat = residual.flatten(2)
        centered = flat - flat.mean(dim=2, keepdim=True)
        denom = centered.std(dim=2, unbiased=False).clamp_min(1e-6)
        yc = (centered[:, 0] * centered[:, 1]).mean(dim=1, keepdim=True) / (denom[:, 0:1] * denom[:, 1:2])
        yr = (centered[:, 0] * centered[:, 2]).mean(dim=1, keepdim=True) / (denom[:, 0:1] * denom[:, 2:3])
        cr_corr = (centered[:, 1] * centered[:, 2]).mean(dim=1, keepdim=True) / (denom[:, 1:2] * denom[:, 2:3])
        features.append(torch.cat([yc, yr, cr_corr], dim=1))
        y_h = (y[..., 1:] - y[..., :-1]).abs()
        y_v = (y[..., 1:, :] - y[..., :-1, :]).abs()
        c_h = (chroma_abs[..., 1:] - chroma_abs[..., :-1]).abs()
        c_v = (chroma_abs[..., 1:, :] - chroma_abs[..., :-1, :]).abs()
        features.append(_flat_stats(c_h / y_h.clamp_min(1e-6)))
        features.append(_flat_stats(c_v / y_v.clamp_min(1e-6)))
        features.append(_patch_mean_std_stats(chroma_abs / luma_abs.clamp_min(1e-6), grid=8))
        return torch.cat(features, dim=1)

    def _patch_spectrum_heterogeneity_stats(self, images: Tensor) -> Tensor:
        gray = images.float().mean(dim=1, keepdim=True)
        local = F.avg_pool2d(gray, kernel_size=7, stride=1, padding=3, count_include_pad=False)
        residual = gray - local
        low = F.avg_pool2d(residual.abs(), kernel_size=5, stride=1, padding=2, count_include_pad=False)
        high_x = F.pad((residual[..., 1:] - residual[..., :-1]).abs(), (0, 1, 0, 0), mode="replicate")
        high_y = F.pad((residual[..., 1:, :] - residual[..., :-1, :]).abs(), (0, 0, 0, 1), mode="replicate")
        high = 0.5 * (high_x + high_y)
        mid = (residual.abs() - low).abs()
        patch_features = []
        for component in (low, mid, high):
            patch_mean = F.adaptive_avg_pool2d(component, output_size=(8, 8)).flatten(1)
            normalized = patch_mean / patch_mean.mean(dim=1, keepdim=True).clamp_min(1e-6)
            entropy = -(normalized / normalized.sum(dim=1, keepdim=True).clamp_min(1e-6)).clamp_min(1e-8)
            entropy = (entropy * entropy.log()).sum(dim=1, keepdim=True).neg()
            patch_features.extend(
                [
                    patch_mean.mean(dim=1, keepdim=True),
                    patch_mean.std(dim=1, unbiased=False, keepdim=True),
                    patch_mean.amax(dim=1, keepdim=True),
                    patch_mean.amin(dim=1, keepdim=True),
                    patch_mean.std(dim=1, unbiased=False, keepdim=True) / patch_mean.mean(dim=1, keepdim=True).clamp_min(1e-6),
                    entropy,
                ]
            )
        low_map = F.adaptive_avg_pool2d(low, output_size=(8, 8)).flatten(1)
        mid_map = F.adaptive_avg_pool2d(mid, output_size=(8, 8)).flatten(1)
        high_map = F.adaptive_avg_pool2d(high, output_size=(8, 8)).flatten(1)
        patch_features.extend(
            [
                (mid_map / low_map.clamp_min(1e-6)).mean(dim=1, keepdim=True),
                (high_map / mid_map.clamp_min(1e-6)).mean(dim=1, keepdim=True),
                (high_map / low_map.clamp_min(1e-6)).mean(dim=1, keepdim=True),
            ]
        )
        return torch.cat(patch_features, dim=1)

    def _block_dct_coefficients(self, values: Tensor, block: int = 8) -> Tensor:
        if values.shape[-2] < block or values.shape[-1] < block:
            return values.new_zeros(values.shape[0], values.shape[1], 0, block * block)
        height = (values.shape[-2] // block) * block
        width = (values.shape[-1] // block) * block
        patches = values[..., :height, :width].float().unfold(2, block, block).unfold(3, block, block)
        patches = patches.contiguous().reshape(values.shape[0], values.shape[1], -1, block, block)
        patches = patches - patches.mean(dim=(-2, -1), keepdim=True)
        basis = _make_dct_kernels(block, values.device, patches.dtype, tuple((u, v) for u in range(block) for v in range(block)))
        return torch.einsum("bcnxy,kxy->bcnk", patches, basis)

    def _coefficient_group_stats(self, coefficients: Tensor, indices: list[int]) -> Tensor:
        if coefficients.shape[2] == 0:
            return coefficients.new_zeros(coefficients.shape[0], coefficients.shape[1] * 8)
        selected = coefficients[..., indices].abs().flatten(2)
        return _view_stats(selected.reshape(selected.shape[0], selected.shape[1], 1, selected.shape[2]))

    def _block_energy_ratio_stats(self, coefficients: Tensor) -> Tensor:
        if coefficients.shape[2] == 0:
            return coefficients.new_zeros(coefficients.shape[0], coefficients.shape[1] * 2 * 8)
        abs_coeffs = coefficients.abs()
        low = abs_coeffs[..., [1, 2, 8, 9]].mean(dim=-1)
        mid = abs_coeffs[..., [3, 4, 10, 11, 16, 17, 18, 24, 25]].mean(dim=-1)
        high = abs_coeffs[..., [5, 6, 7, 12, 13, 14, 15, 19, 20, 21, 22, 23, 26, 27, 28, 29, 30, 31]].mean(dim=-1)
        ratios = torch.stack([mid / low.clamp_min(1e-6), high / mid.clamp_min(1e-6)], dim=-1)
        return _view_stats(ratios.permute(0, 1, 3, 2).reshape(ratios.shape[0], ratios.shape[1] * ratios.shape[-1], 1, ratios.shape[2]))

    def _jpeg_boundary_grid_stats(self, channel: Tensor, period: int = 8) -> Tensor:
        hdiff = (channel[..., 1:] - channel[..., :-1]).abs()
        vdiff = (channel[..., 1:, :] - channel[..., :-1, :]).abs()
        return torch.cat(
            [
                _boundary_features(hdiff, period=period, axis="x"),
                _boundary_features(vdiff, period=period, axis="y"),
                _phase_mean_stats(hdiff, period=period, axis="x"),
                _phase_mean_stats(vdiff, period=period, axis="y"),
            ],
            dim=1,
        )

    def _codec_block_stats(self, images: Tensor) -> Tensor:
        ycbcr = _rgb_to_ycbcr(images.float())
        coefficients = self._block_dct_coefficients(ycbcr, block=8)
        features = [
            self._coefficient_group_stats(coefficients, [1, 2, 8, 9]),
            self._coefficient_group_stats(coefficients, [3, 4, 10, 11, 16, 17, 18, 24, 25]),
            self._coefficient_group_stats(coefficients, [5, 6, 7, 12, 13, 14, 15, 19, 20, 21, 22, 23, 26, 27, 28, 29, 30, 31]),
            self._block_energy_ratio_stats(coefficients),
        ]
        for channel_index in range(3):
            features.append(self._jpeg_boundary_grid_stats(ycbcr[:, channel_index : channel_index + 1], period=8))
        return torch.cat(features, dim=1)

    def _quantized_reconstruction(self, images: Tensor, levels: int, scale: float) -> Tensor:
        if scale < 1.0:
            height, width = images.shape[-2:]
            small_h = max(8, int(round(height * float(scale))))
            small_w = max(8, int(round(width * float(scale))))
            values = F.interpolate(images.float(), size=(small_h, small_w), mode="bicubic", align_corners=False)
            values = F.interpolate(values, size=(height, width), mode="bicubic", align_corners=False)
        else:
            values = images.float()
        quantized = torch.round(values.clamp(0.0, 1.0) * float(levels - 1)) / float(levels - 1)
        return quantized.clamp(0.0, 1.0)

    def _recompression_stability_blocks(self, images: Tensor) -> OrderedDict[str, Tensor]:
        gray = images.float().mean(dim=1, keepdim=True)
        features: OrderedDict[str, Tensor] = OrderedDict()
        for levels, scale in ((32, 1.0), (64, 1.0), (32, 0.5)):
            reconstructed = self._quantized_reconstruction(images, levels=levels, scale=scale)
            residual = (gray - reconstructed.mean(dim=1, keepdim=True)).abs()
            block_features = [
                _flat_stats(residual),
                _patch_mean_std_stats(residual, grid=8),
            ]
            hdiff = (residual[..., 1:] - residual[..., :-1]).abs()
            vdiff = (residual[..., 1:, :] - residual[..., :-1, :]).abs()
            block_features.extend(
                [
                    _boundary_features(hdiff, period=8, axis="x"),
                    _boundary_features(vdiff, period=8, axis="y"),
                ]
            )
            name = (
                "recompression_q32"
                if levels == 32 and scale == 1.0
                else "recompression_q64"
                if levels == 64 and scale == 1.0
                else "recompression_q32_resize50"
            )
            features[name] = torch.cat(block_features, dim=1)
        features["recompression_stability"] = torch.cat(list(features.values()), dim=1)
        return features


class ArtifactPriorV0(nn.Module):
    def __init__(self, config: CodecTextureConfig | None = None, hidden_dim: int = 128, dropout: float = 0.05) -> None:
        super().__init__()
        self.feature_extractor = ArtifactPriorFeatureExtractor(config)
        self.register_buffer("feature_mean", torch.zeros(self.feature_extractor.feature_dim))
        self.register_buffer("feature_scale", torch.ones(self.feature_extractor.feature_dim))
        self.input_norm = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(self.feature_extractor.feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 1),
        )

    @property
    def feature_dim(self) -> int:
        return self.feature_extractor.feature_dim

    def set_feature_normalizer(self, mean: Tensor, scale: Tensor) -> None:
        if mean.shape != self.feature_mean.shape or scale.shape != self.feature_scale.shape:
            raise ValueError("normalizer tensors must match feature_dim")
        self.feature_mean.copy_(mean.to(device=self.feature_mean.device, dtype=self.feature_mean.dtype))
        self.feature_scale.copy_(scale.to(device=self.feature_scale.device, dtype=self.feature_scale.dtype).clamp_min(1e-6))

    def forward(self, images: Tensor, return_details: bool = False) -> Tensor | ArtifactPriorOutput:
        with torch.no_grad():
            features = self.feature_extractor(images)
        normalized = (features - self.feature_mean[None, :]) / self.feature_scale[None, :].clamp_min(1e-6)
        logit = self.head(self.input_norm(normalized)).squeeze(1)
        if return_details:
            return ArtifactPriorOutput(logit=logit, features=features)
        return logit
