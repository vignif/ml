import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image, ImageOps
from pathlib import Path
from functools import lru_cache
from typing import Callable, Optional, Dict, Any

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_IMAGE_SIDE = int(os.getenv("ST_MAX_IMAGE_SIDE", "512"))
MIN_IMAGE_SIDE = int(os.getenv("ST_MIN_IMAGE_SIDE", "32"))  # must be >= 32 for 5 VGG19 poolings

# Simple fast style transfer using a pre-trained VGG19 feature extractor and iterative optimization.
# For production you'd likely want a pre-trained transformer network for speed.

def load_image(path: Path, max_size: int | None = None) -> torch.Tensor:
    if max_size is None:
        max_size = MAX_IMAGE_SIDE
    img = Image.open(path).convert("RGB")
    w, h = img.size
    # Downscale if needed for max side constraint
    largest = max(w, h)
    if largest > max_size:
        scale = max_size / largest
        w, h = int(w * scale), int(h * scale)
        img = img.resize((w, h), Image.LANCZOS)
    # Ensure minimum side so VGG pooling does not collapse to zero
    smallest = min(w, h)
    if smallest < MIN_IMAGE_SIDE:
        # Instead of upscaling entire image (which could exceed max_size), pad evenly
        pad_w = max(0, MIN_IMAGE_SIDE - w)
        pad_h = max(0, MIN_IMAGE_SIDE - h)
        if pad_w or pad_h:
            # (left, top, right, bottom)
            left = pad_w // 2
            right = pad_w - left
            top = pad_h // 2
            bottom = pad_h - top
            img = ImageOps.expand(img, border=(left, top, right, bottom), fill=0)
            w, h = img.size
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0).to(DEVICE)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.clone().detach().cpu().squeeze(0)
    unnorm = T.Normalize(
        mean=[-m/s for m, s in zip([0.485, 0.456, 0.406],[0.229,0.224,0.225])],
        std=[1/s for s in [0.229,0.224,0.225]]
    )
    t = unnorm(t).clamp(0,1)
    return T.ToPILImage()(t)


def get_vgg_features():
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(DEVICE).eval()
    for p in vgg.parameters():
        p.requires_grad_(False)
    return vgg

# Create a single global VGG instance (downloads weights once) to reuse across requests.
_VGG = get_vgg_features()

STYLE_LAYERS = {0, 5, 10, 19, 28}  # relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 indices
CONTENT_LAYER = 21  # relu4_2


def extract_features(x: torch.Tensor, model: nn.Module):
    features = {}
    for i, layer in enumerate(model):
        x = layer(x)
        if i in STYLE_LAYERS:
            features[f"style_{i}"] = x
        if i == CONTENT_LAYER:
            features["content"] = x
    return features


def gram_matrix(t: torch.Tensor) -> torch.Tensor:
    b, c, h, w = t.size()
    features = t.view(c, h * w)
    gram = features @ features.t() / (c * h * w)
    return gram


@lru_cache(maxsize=16)
def _cached_style_grams(style_path_str: str, max_side: int):
    style_img = load_image(Path(style_path_str), max_size=max_side)
    style_feats = extract_features(style_img, _VGG)
    return {k: gram_matrix(v) for k, v in style_feats.items() if k.startswith("style_")}


def stylize_with_progress(
    content_path: Path,
    style_path: Path,
    *,
    steps: int = 250,
    style_weight: float = 1e6,
    content_weight: float = 1.0,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    callback_every: int = 1,
    should_cancel: Optional[Callable[[], bool]] = None,
    preview_every: Optional[int] = None,
) -> Image.Image:
    """Style transfer with optional progress callback.

    progress_cb receives a dict with keys: step (1-based), total_steps, content_loss, style_loss, total_loss, elapsed_seconds.
    """
    import time
    start = time.time()
    content = load_image(content_path, None)
    max_side = content.shape[-1]
    content_feats = extract_features(content, _VGG)
    style_grams = _cached_style_grams(str(style_path.resolve()), max_side)

    target = content.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([target], lr=0.02)

    for step in range(1, steps + 1):
        target_feats = extract_features(target, _VGG)
        content_loss = nn.functional.mse_loss(target_feats['content'], content_feats['content'])
        style_loss = 0.0
        for k, v in target_feats.items():
            if k.startswith('style_'):
                target_gram = gram_matrix(v)
                style_gram = style_grams[k]
                style_loss += nn.functional.mse_loss(target_gram, style_gram)
        total_loss = content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if progress_cb and (step % callback_every == 0 or step == steps):
            payload = {
                "step": step,
                "total_steps": steps,
                "content_loss": float(content_loss.item()),
                "style_loss": float(style_loss.item()),
                "total_loss": float(total_loss.item()),
                "elapsed_seconds": time.time() - start,
            }
            if preview_every and (step % preview_every == 0 or step == steps):
                try:
                    preview_img = tensor_to_pil(target)
                    # Optionally downscale large preview
                    max_w = 512
                    if preview_img.width > max_w:
                        ratio = max_w / preview_img.width
                        preview_img = preview_img.resize((max_w, int(preview_img.height * ratio)))
                    payload["_preview_image"] = preview_img
                    payload["preview_step"] = step
                except Exception:
                    pass
            progress_cb(payload)

        if should_cancel and should_cancel():
            break
        if step % 50 == 0:
            print(f"Step {step}/{steps} - content: {content_loss.item():.4f} style: {style_loss.item():.4f}")

    return tensor_to_pil(target)


def stylize(content_path: Path, style_path: Path, steps: int = 250, style_weight: float = 1e6, content_weight: float = 1.0) -> Image.Image:
    return stylize_with_progress(content_path, style_path, steps=steps, style_weight=style_weight, content_weight=content_weight)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m src.style_transfer <content_image> <style_image> [out]")
        raise SystemExit(1)
    out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("output.jpg")
    img = stylize(Path(sys.argv[1]), Path(sys.argv[2]))
    img.save(out_path)
    print("Saved to", out_path)
