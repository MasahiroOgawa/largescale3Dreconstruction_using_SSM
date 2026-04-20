"""Parse ETH3D COLMAP calibration files (cameras.txt, images.txt).

Returns per-image intrinsics (3, 3) and extrinsics (4, 4, world-to-camera),
adjusted to the center-square-crop + resize applied by
`ssm3d.data.eth3d.load_eth3d_scene`. That lets downstream code build pixel-level
warps that match the loaded RGB/depth tensors without re-scaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class ETH3DCams:
    intrinsics: dict[str, np.ndarray]  # image_name -> (3, 3) adjusted to image_size
    extrinsics: dict[str, np.ndarray]  # image_name -> (4, 4) w2c
    orig_sizes: dict[str, tuple[int, int]]  # image_name -> (H, W)


def _quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    # COLMAP quaternion convention: (qw, qx, qy, qz), Hamilton.
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz),     2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [    2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz),     2 * (qy * qz - qx * qw)],
        [    2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float32)


def _parse_cameras(path: Path) -> dict[str, dict]:
    """COLMAP cameras.txt: `ID MODEL WIDTH HEIGHT fx fy cx cy ...`."""
    out: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        cam_id = parts[0]
        out[cam_id] = {
            "width": float(parts[2]),
            "height": float(parts[3]),
            "fx": float(parts[4]),
            "fy": float(parts[5]),
            "cx": float(parts[6]),
            "cy": float(parts[7]),
        }
    return out


def _parse_images(path: Path) -> dict[str, dict]:
    """COLMAP images.txt: alternating lines; first line per image is the pose."""
    out: dict[str, dict] = {}
    lines = path.read_text().splitlines()
    idx = 0
    pose_line_count = 0
    while idx < len(lines):
        line = lines[idx].strip()
        idx += 1
        if not line or line.startswith("#"):
            continue
        pose_line_count += 1
        if pose_line_count % 2 == 0:
            # skip the 2D-3D point correspondences line
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        cam_id = parts[8]
        name = Path(parts[9]).name  # COLMAP may use "dslr_images_undistorted/DSC_*.JPG"
        out[name] = {
            "quat": (qw, qx, qy, qz),
            "trans": (tx, ty, tz),
            "camera_id": cam_id,
        }
    return out


def load_eth3d_cams(
    scene_dir: Path,
    image_size: int,
    image_names: Optional[list[str]] = None,
) -> ETH3DCams:
    """Parse COLMAP calib + return intrinsics rescaled for the center-square crop+resize.

    The `load_eth3d_scene` pipeline applies:
        crop box = center square of min(H, W)
        resize  = (image_size, image_size) bilinear
    We apply the same transform to the intrinsic matrix so projected points land
    on the resized image. Extrinsics (world-to-camera) are unchanged.
    """
    scene_dir = Path(scene_dir)
    calib_candidates = [
        scene_dir / "dslr_calibration_undistorted",
        scene_dir / scene_dir.name / "dslr_calibration_undistorted",
        scene_dir / "dslr_calibration_jpg",
        scene_dir / scene_dir.name / "dslr_calibration_jpg",
    ]
    calib = next((c for c in calib_candidates if c.exists()), None)
    if calib is None:
        raise FileNotFoundError(f"No dslr_calibration_* under {scene_dir}")

    cameras = _parse_cameras(calib / "cameras.txt")
    images = _parse_images(calib / "images.txt")

    intrinsics: dict[str, np.ndarray] = {}
    extrinsics: dict[str, np.ndarray] = {}
    orig_sizes: dict[str, tuple[int, int]] = {}

    for name, info in images.items():
        if image_names is not None and name not in image_names:
            continue
        cam = cameras.get(info["camera_id"])
        if cam is None:
            continue
        orig_h, orig_w = int(cam["height"]), int(cam["width"])
        s = min(orig_h, orig_w)
        crop_left = (orig_w - s) // 2
        crop_top = (orig_h - s) // 2
        scale = image_size / s

        # Intrinsic after crop: cx -= crop_left, cy -= crop_top
        # Intrinsic after resize: multiply fx, fy, cx, cy by `scale`
        fx = cam["fx"] * scale
        fy = cam["fy"] * scale
        cx = (cam["cx"] - crop_left) * scale
        cy = (cam["cy"] - crop_top) * scale
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        R = _quat_to_rot(*info["quat"])
        t = np.array(info["trans"], dtype=np.float32)
        E = np.eye(4, dtype=np.float32)
        E[:3, :3] = R
        E[:3, 3] = t

        intrinsics[name] = K
        extrinsics[name] = E
        orig_sizes[name] = (orig_h, orig_w)

    return ETH3DCams(intrinsics=intrinsics, extrinsics=extrinsics, orig_sizes=orig_sizes)
