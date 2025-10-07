"""Fast feed-forward style transfer network (Johnson et al. 2016 style architecture).

This module defines the TransformerNet architecture and utilities to apply a
pre-trained style (one model per style) to an input image quickly.

Weights are NOT included. Place your trained *.pth files under models/ with
filenames like:  starry.pth  mosaic.pth  udnie.pth

Each state_dict should match the TransformerNet below (common in many public
implementations). You can obtain compatible weights from style transfer
repositories that publish MIT-licensed weights, or train your own.

Usage (programmatic):
    img = fast_stylize(Path('content.jpg'), 'mosaic')

Environment variables:
    FAST_STYLE_MODELS_DIR (optional) - override models directory path.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        layers = []
        if upsample:
            layers.append(nn.Upsample(mode="nearest", scale_factor=upsample))
        padding = kernel_size // 2
        layers.append(nn.ReflectionPad2d(padding))
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride))
        super().__init__(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvLayer(channels, channels, 3, 1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, 3, 1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # type: ignore
        y = self.relu(self.in1(self.conv1(x)))
        y = self.in2(self.conv2(y))
        return x + y


class TransformerNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.conv1 = ConvLayer(3, 32, 9, 1)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, 3, 2)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, 3, 2)
        self.in3 = nn.InstanceNorm2d(128, affine=True)

        # Residuals
        self.res = nn.Sequential(
            *[ResidualBlock(128) for _ in range(5)]
        )

        # Decoder / Upsample
        self.deconv1 = ConvLayer(128, 64, 3, 1, upsample=2)
        self.in4 = nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = ConvLayer(64, 32, 3, 1, upsample=2)
        self.in5 = nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, 9, 1)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x):  # type: ignore
        y = self.relu(self.in1(self.conv1(x)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        # Many implementations clamp after adding original dynamic range; here we map via tanh to [-1,1] then scale.
        return y


def _models_dir() -> Path:
    env = os.getenv("FAST_STYLE_MODELS_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "models"


def available_styles() -> List[str]:
    d = _models_dir()
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.pth"))


def load_style_model(style_name: str) -> TransformerNet:
    path = _models_dir() / f"{style_name}.pth"
    if not path.exists():
        raise FileNotFoundError(f"Model weights not found: {path} (expected {style_name}.pth)")
    state = torch.load(path, map_location=DEVICE)
    net = TransformerNet().to(DEVICE)
    # Some checkpoints may save with keys 'state_dict'
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    # Remove potential 'module.' prefixes
    cleaned = {k.replace('module.', ''): v for k, v in state.items()}
    net.load_state_dict(cleaned, strict=False)
    net.eval()
    return net


_preprocess = T.Compose([
    T.ToTensor(),
])

_postprocess = T.Compose([
    T.Lambda(lambda t: t.clamp(0, 255)),
    T.Lambda(lambda t: t / 255.0),
    T.ToPILImage(),
])


def fast_stylize(content_path: Path, style_name: str, max_size: int = 720) -> Image.Image:
    img = Image.open(content_path).convert("RGB")
    if max(img.size) > max_size:
        scale = max_size / max(img.size)
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)
    t = _preprocess(img) * 255.0  # scale to 0-255 like many trainings
    inp = t.unsqueeze(0).to(DEVICE)
    net = load_style_model(style_name)
    with torch.no_grad():
        out = net(inp)
    out = out.squeeze(0).cpu()
    return _postprocess(out)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m src.fast_style <content_image> <style_name> [output]")
        raise SystemExit(1)
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("fast_output.jpg")
    img = fast_stylize(Path(sys.argv[1]), sys.argv[2])
    img.save(out)
    print("Saved to", out)
