"""Render a TAPVid-style point-tracking video.

For each frame, draw predicted-visible tracks as filled circles and
predicted-occluded tracks as hollow circles, with a fading 8-frame
trajectory tail per track. Track IDs are colour-coded with a perceptually
uniform colormap (Matplotlib's tab20).

Output: MP4 via OpenCV's VideoWriter.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _colormap(n: int) -> np.ndarray:
    """`n` BGR uint8 colors from a hash-like distribution. No matplotlib dep."""
    rng = np.random.default_rng(0)
    hsv = np.zeros((n, 1, 3), dtype=np.uint8)
    hsv[:, 0, 0] = (rng.integers(0, 180, size=n)).astype(np.uint8)
    hsv[:, 0, 1] = 200
    hsv[:, 0, 2] = 230
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[:, 0]
    return bgr


def project_to_uv(xyz: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pinhole projection: (..., 3) world XYZ → (..., 2) pixel uv.

    Tracks are in *camera coordinates of the first frame* (TAPVid-3D
    convention §3.1), so a static camera at the canonical pose is assumed.
    """
    Z = np.clip(xyz[..., 2:3], 1e-6, None)
    uv = (xyz[..., :2] / Z) * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
    return uv


def render_tracking_video(
    frames_rgb_uint8: np.ndarray,    # (F, H, W, 3)
    pred_tracks_NT3: np.ndarray,     # (N, F, 3)
    pred_visibility_NT: np.ndarray,  # (N, F) in [0, 1]
    intrinsics_K: np.ndarray,        # (3, 3)
    out_path: str | Path,
    fps: int = 30,
    tail_len: int = 8,
    radius: int = 4,
    vis_thresh: float = 0.5,
) -> Path:
    """Write an MP4 video with overlaid predicted point tracks."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    F, H, W, _ = frames_rgb_uint8.shape
    N = pred_tracks_NT3.shape[0]
    uv = project_to_uv(pred_tracks_NT3, intrinsics_K)        # (N, F, 2)

    colors = _colormap(N)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    if not vw.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter at {out_path}")

    for t in range(F):
        frame_bgr = cv2.cvtColor(frames_rgb_uint8[t], cv2.COLOR_RGB2BGR).copy()

        # Trajectory tails (oldest faded, newest opaque)
        for n in range(N):
            for k in range(1, min(tail_len, t) + 1):
                p0 = uv[n, t - k]
                p1 = uv[n, t - k + 1]
                if not (np.isfinite(p0).all() and np.isfinite(p1).all()):
                    continue
                p0 = tuple(int(round(v)) for v in p0)
                p1 = tuple(int(round(v)) for v in p1)
                if min(p0 + p1) < -50 or max(p0[0], p1[0]) > W + 50 or max(p0[1], p1[1]) > H + 50:
                    continue
                alpha = 1.0 - (k - 1) / max(1, tail_len)
                bgr = tuple(int(c * alpha) for c in colors[n].tolist())
                cv2.line(frame_bgr, p0, p1, bgr, thickness=1, lineType=cv2.LINE_AA)

        # Current-frame markers
        for n in range(N):
            p = uv[n, t]
            if not np.isfinite(p).all():
                continue
            cx, cy = (int(round(p[0])), int(round(p[1])))
            if not (0 <= cx < W and 0 <= cy < H):
                continue
            color = tuple(int(c) for c in colors[n].tolist())
            if pred_visibility_NT[n, t] >= vis_thresh:
                cv2.circle(frame_bgr, (cx, cy), radius, color, thickness=-1, lineType=cv2.LINE_AA)
            else:
                cv2.circle(frame_bgr, (cx, cy), radius, color, thickness=1, lineType=cv2.LINE_AA)

        vw.write(frame_bgr)
    vw.release()
    return out_path
