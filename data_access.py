"""Download and clip Sentinel-2 / Landsat windows via Planetary Computer STAC."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import rasterio
from rasterio.io import MemoryFile
from rasterio.windows import from_bounds
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.crs import CRS

from .config import DATA_DIR, PESHAWAR_BBOX, S2_BANDS


STAC_SEARCH = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SIGN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"


def sign_href(href: str) -> str:
    r = requests.get(SIGN_URL, params={"href": href}, timeout=30)
    r.raise_for_status()
    return r.json()["href"]


def search_sentinel2(
    bbox: List[float] = PESHAWAR_BBOX,
    datetime: str = "2024-06-01/2024-09-30",
    max_cloud: float = 5.0,
    limit: int = 5,
) -> List[dict]:
    body = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "datetime": datetime,
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "limit": limit,
    }
    r = requests.post(STAC_SEARCH, json=body, timeout=60)
    r.raise_for_status()
    return r.json().get("features", [])


def search_landsat(
    bbox: List[float] = PESHAWAR_BBOX,
    datetime: str = "2024-06-01/2024-09-30",
    max_cloud: float = 10.0,
    limit: int = 3,
) -> List[dict]:
    body = {
        "collections": ["landsat-c2-l2"],
        "bbox": bbox,
        "datetime": datetime,
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "limit": limit,
    }
    r = requests.post(STAC_SEARCH, json=body, timeout=60)
    r.raise_for_status()
    return r.json().get("features", [])


def _read_band_window(
    asset_href: str,
    bbox: List[float],
    out_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, dict]:
    """Read a COG window for the given geographic bbox (WGS84). Returns array and profile."""
    from rasterio.warp import transform_bounds

    signed = sign_href(asset_href)
    with rasterio.open(signed) as src:
        # Transform WGS84 bbox into the raster CRS (usually UTM)
        if src.crs and src.crs.to_epsg() != 4326:
            bbox_proj = transform_bounds("EPSG:4326", src.crs, *bbox)
        else:
            bbox_proj = bbox
        window = from_bounds(*bbox_proj, transform=src.transform)
        window = window.round_offsets().round_lengths()
        if window.width <= 0 or window.height <= 0:
            raise ValueError(f"Empty window for bbox {bbox} against {src.bounds}")
        data = src.read(
            1,
            window=window,
            out_shape=out_shape,
            resampling=rasterio.enums.Resampling.bilinear,
        )
        transform = src.window_transform(window)
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "height": int(data.shape[0]),
            "width": int(data.shape[1]),
            "count": 1,
            "crs": src.crs,
            "transform": transform,
            "nodata": None,
        }
        return data.astype(np.float32), profile


def read_band_window(asset_href, bbox, out_shape=None):
    """Same as above, but network reads give up after ~25 s (x2 tries) instead of hanging forever."""
    with rasterio.Env(GDAL_HTTP_TIMEOUT="25", GDAL_HTTP_CONNECTTIMEOUT="15",
                      GDAL_HTTP_MAX_RETRY="1", GDAL_HTTP_RETRY_DELAY="1"):
        return _read_band_window(asset_href, bbox, out_shape)


def download_s2_stack(
    bbox: List[float] = PESHAWAR_BBOX,
    out_dir: Optional[Path] = None,
    max_cloud: float = 5.0,
) -> Dict[str, np.ndarray]:
    """Download a small multi-band stack (B02,B03,B04,B08) for the AOI.
    Returns dict of band name -> array (float32 reflectance-ish).
    """
    out_dir = out_dir or DATA_DIR / "s2"
    out_dir.mkdir(parents=True, exist_ok=True)

    features = search_sentinel2(bbox=bbox, max_cloud=max_cloud, limit=3)
    if not features:
        raise RuntimeError("No low-cloud Sentinel-2 scenes found for AOI")

    # Prefer lowest cloud
    features = sorted(features, key=lambda f: f["properties"].get("eo:cloud_cover", 100))
    feat = features[0]
    print(f"Using scene: {feat['id']} cloud={feat['properties'].get('eo:cloud_cover')}")

    stack = {}
    profile = None
    for band in S2_BANDS:
        href = feat["assets"][band]["href"]
        arr, prof = read_band_window(href, bbox)
        # Sentinel-2 L2A is scaled reflectance * 10000 typically
        arr = np.clip(arr / 10000.0, 0, 1)
        stack[band] = arr
        profile = prof
        # Save individual for inspection (uint16 scaled)
        out_path = out_dir / f"{feat['id']}_{band}.tif"
        save_prof = dict(prof)
        save_prof["dtype"] = "uint16"
        with rasterio.open(out_path, "w", **save_prof) as dst:
            dst.write((arr * 10000).astype(np.uint16), 1)

    # Save meta
    meta_path = out_dir / "last_scene.txt"
    meta_path.write_text(f"{feat['id']}\n{feat['properties'].get('datetime')}\n")
    return stack


def compute_ndvi(stack: Dict[str, np.ndarray]) -> np.ndarray:
    nir = stack["B08"]
    red = stack["B04"]
    denom = nir + red
    ndvi = np.where(denom > 1e-6, (nir - red) / denom, 0.0)
    return ndvi.astype(np.float32)


def ndvi_vegetation_mask(ndvi: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    return (ndvi >= threshold).astype(np.uint8)


if __name__ == "__main__":
    print("Searching and downloading sample stack...")
    stack = download_s2_stack()
    ndvi = compute_ndvi(stack)
    print("NDVI shape:", ndvi.shape, "range:", ndvi.min(), ndvi.max())
    mask = ndvi_vegetation_mask(ndvi)
    print("Vegetation fraction:", mask.mean())
