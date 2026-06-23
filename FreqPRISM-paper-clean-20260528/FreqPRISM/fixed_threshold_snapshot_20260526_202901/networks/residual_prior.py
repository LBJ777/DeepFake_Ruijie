from __future__ import annotations

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from typing import Any, cast, Dict, List, Optional, Union
import numpy as np
from pathlib import Path
from typing import Sequence

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1, zero_init_residual=False):
        super(ResNet, self).__init__()

        self.unfoldSize = 2
        self.unfoldIndex = 0
        assert self.unfoldSize > 1
        assert -1 < self.unfoldIndex and self.unfoldIndex < self.unfoldSize*self.unfoldSize
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64 , layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc1 = nn.Linear(512 * block.expansion, 1)
        self.fc1 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    def interpolate(self, img, factor):
        return F.interpolate(F.interpolate(img, scale_factor=factor, mode='nearest', recompute_scale_factor=True), scale_factor=1/factor, mode='nearest', recompute_scale_factor=True)
    def forward(self, x):
        # n,c,w,h = x.shape
        # if -1*w%2 != 0: x = x[:,:,:w%2*-1,:      ]
        # if -1*h%2 != 0: x = x[:,:,:      ,:h%2*-1]
        # factor = 0.5
        # x_half = F.interpolate(x, scale_factor=factor, mode='nearest', recompute_scale_factor=True)
        # x_re   = F.interpolate(x_half, scale_factor=1/factor, mode='nearest', recompute_scale_factor=True)
        # NPR  = x - x_re
        # n,c,w,h = x.shape
        # if w%2 == 1 : x = x[:,:,:-1,:]
        # if h%2 == 1 : x = x[:,:,:,:-1]
        NPR  = x - self.interpolate(x, 0.5)

        x = self.conv1(NPR*2.0/3.0)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


def _torch_load(path: str | Path, *, map_location: str = "cpu"):
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def crop_to_even(image):
    width, height = image.size
    even_width = width - (width % 2)
    even_height = height - (height % 2)
    if even_width <= 0 or even_height <= 0:
        raise ValueError(f"image is too small to crop to even dimensions: {image.size}")
    if even_width == width and even_height == height:
        return image
    return image.crop((0, 0, even_width, even_height))


def _unwrap_state_dict(checkpoint_obj) -> dict:
    if isinstance(checkpoint_obj, dict):
        for key in ("model", "state_dict", "net", "network"):
            value = checkpoint_obj.get(key)
            if isinstance(value, dict):
                return value
        if all(hasattr(value, "shape") for value in checkpoint_obj.values()):
            return checkpoint_obj
    raise ValueError("checkpoint must be a raw state_dict or contain model/state_dict")


def _strip_module_prefix(state_dict: dict) -> dict:
    return {
        str(key)[7:] if str(key).startswith("module.") else str(key): value
        for key, value in state_dict.items()
    }


def build_residual_prior_model():
    return resnet50(num_classes=1)


def load_residual_prior_model(checkpoint: str | Path, *, device: str = "cpu"):
    import torch

    torch_device = torch.device(device if not str(device).startswith("cuda") or torch.cuda.is_available() else "cpu")
    model = build_residual_prior_model()
    state_dict = _strip_module_prefix(_unwrap_state_dict(_torch_load(checkpoint, map_location="cpu")))
    model.load_state_dict(state_dict, strict=True)
    model.to(torch_device)
    model.eval()
    return model


def build_residual_prior_transform(*, image_size: int | None = 256, train: bool = False):
    from torchvision import transforms

    steps = [transforms.Lambda(crop_to_even)]
    if image_size is not None:
        steps.append(transforms.Resize((int(image_size), int(image_size))))
    if train:
        steps.append(transforms.RandomHorizontalFlip())
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(steps)


def infer_residual_prior_scores(
    paths: Sequence[str | Path],
    *,
    model,
    device: str = "cpu",
    image_size: int | None = 256,
    batch_size: int = 32,
) -> np.ndarray:
    import torch
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    torch_device = torch.device(device if not str(device).startswith("cuda") or torch.cuda.is_available() else "cpu")
    transform = build_residual_prior_transform(image_size=image_size, train=False)
    effective_batch_size = 1 if image_size is None else max(1, int(batch_size))
    scores: list[float] = []
    batch: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for path in paths:
            with Image.open(path) as image:
                batch.append(transform(image.convert("RGB")))
            if len(batch) >= effective_batch_size:
                images = torch.stack(batch).to(torch_device)
                logits = model(images).reshape(images.shape[0], -1)[:, 0]
                scores.extend(torch.sigmoid(logits).detach().cpu().numpy().astype(float).tolist())
                batch.clear()
        if batch:
            images = torch.stack(batch).to(torch_device)
            logits = model(images).reshape(images.shape[0], -1)[:, 0]
            scores.extend(torch.sigmoid(logits).detach().cpu().numpy().astype(float).tolist())
    return np.asarray(scores, dtype=np.float32)
