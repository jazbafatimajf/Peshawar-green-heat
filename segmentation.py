"""Vegetation / built-up segmentation: NDVI baseline + lightweight U-Net style model."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, jaccard_score

from .config import MODELS_DIR, NDVI_VEG_THRESHOLD
from .data_access import compute_ndvi, ndvi_vegetation_mask


class TinyUNet(nn.Module):
    """Minimal U-Net for 4-band input -> 2-class (veg / non-veg) logits.
    Intentionally small so it trains quickly on a few hand-labelled tiles.
    """

    def __init__(self, in_ch: int = 4, out_ch: int = 2):
        super().__init__()
        def conv_block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True),
            )

        self.enc1 = conv_block(in_ch, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = conv_block(32, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = conv_block(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = conv_block(32, 16)
        self.head = nn.Conv2d(16, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.up2(b)
        # pad if needed for odd sizes
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = torch.nn.functional.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = torch.nn.functional.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)


def stack_to_tensor(stack: Dict[str, np.ndarray]) -> torch.Tensor:
    """B02,B03,B04,B08 -> (1,4,H,W)"""
    bands = [stack[b] for b in ["B02", "B03", "B04", "B08"]]
    arr = np.stack(bands, axis=0)  # 4,H,W
    return torch.from_numpy(arr).unsqueeze(0).float()


def baseline_predict(stack: Dict[str, np.ndarray], threshold: float = NDVI_VEG_THRESHOLD) -> np.ndarray:
    """Simple NDVI threshold baseline. Returns binary mask (1=vegetation)."""
    ndvi = compute_ndvi(stack)
    return ndvi_vegetation_mask(ndvi, threshold)


def model_predict(
    stack: Dict[str, np.ndarray],
    model: Optional[nn.Module] = None,
    weights_path: Optional[Path] = None,
) -> np.ndarray:
    """Run TinyUNet if available, else fall back to baseline."""
    if model is None:
        model = TinyUNet()
        if weights_path and weights_path.exists():
            model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        else:
            # No trained weights yet → use baseline
            return baseline_predict(stack)
    model.eval()
    with torch.no_grad():
        x = stack_to_tensor(stack)
        logits = model(x)
        pred = logits.argmax(dim=1).squeeze(0).numpy().astype(np.uint8)
    return pred


def evaluate_against_labels(
    pred: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """Pixel-wise F1 and IoU for vegetation class (1)."""
    y_true = labels.ravel()
    y_pred = pred.ravel()
    # Restrict to valid pixels if labels use 255 as ignore
    valid = y_true < 255
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if len(y_true) == 0:
        return {"f1": 0.0, "iou": 0.0, "veg_frac_true": 0.0, "veg_frac_pred": 0.0}
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    iou = jaccard_score(y_true, y_pred, average="binary", zero_division=0)
    return {
        "f1": float(f1),
        "iou": float(iou),
        "veg_frac_true": float(y_true.mean()),
        "veg_frac_pred": float(y_pred.mean()),
    }


def train_tiny_unet(
    tiles: list,  # list of (stack_dict, label_array)
    epochs: int = 20,
    lr: float = 1e-3,
    out_path: Optional[Path] = None,
) -> TinyUNet:
    """Train on a handful of hand-labelled tiles. Returns model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for stack, label in tiles:
            x = stack_to_tensor(stack).to(device)
            y = torch.from_numpy(label.astype(np.int64)).unsqueeze(0).to(device)
            # Resize label if needed
            if y.shape[-2:] != x.shape[-2:]:
                y = torch.nn.functional.interpolate(
                    y.float().unsqueeze(0), size=x.shape[-2:], mode="nearest"
                ).squeeze(0).long()
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} loss={total_loss/len(tiles):.4f}")

    out_path = out_path or MODELS_DIR / "tiny_unet_veg.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved weights to {out_path}")
    return model


# Synthetic label helper for quick smoke tests
def make_synthetic_labels(ndvi: np.ndarray, noise: float = 0.05) -> np.ndarray:
    """Create noisy labels from NDVI for testing the training loop."""
    base = (ndvi >= NDVI_VEG_THRESHOLD).astype(np.uint8)
    flip = np.random.random(base.shape) < noise
    base[flip] = 1 - base[flip]
    return base
