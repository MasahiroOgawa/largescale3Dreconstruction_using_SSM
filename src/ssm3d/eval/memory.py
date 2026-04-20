"""Lightweight memory-usage measurement for the DA3 vs SSM-3D head-to-head.

We compare two quantities per model:
  - `param_count` — static parameter count (exact).
  - `peak_rss_delta_mb` — peak process RSS increase during inference, sampled
    by a background thread at 10 ms. Captures Python + C-allocated tensor
    memory on CPU; on CUDA runs, supplement with `torch.cuda.max_memory_allocated`.

The sampler is a deliberately simple polling thread — coarse but
language-agnostic. Use the `RSSPoller` context manager to scope a measurement
around a code block.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import psutil
import torch


def param_count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


@dataclass
class MemoryReport:
    param_count: int
    peak_rss_delta_mb: float
    peak_cuda_mb: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "param_count": float(self.param_count),
            "peak_rss_delta_mb": float(self.peak_rss_delta_mb),
            "peak_cuda_mb": float(self.peak_cuda_mb),
        }


class RSSPoller:
    """Sampling thread that tracks peak RSS between `__enter__` and `__exit__`."""

    def __init__(self, interval: float = 0.01) -> None:
        self.interval = interval
        self._stop = threading.Event()
        self._proc = psutil.Process()
        self.baseline = 0
        self.peak = 0
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            rss = self._proc.memory_info().rss
            if rss > self.peak:
                self.peak = rss
            self._stop.wait(self.interval)

    def __enter__(self) -> "RSSPoller":
        self.baseline = self._proc.memory_info().rss
        self.peak = self.baseline
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    @property
    def peak_delta_mb(self) -> float:
        return (self.peak - self.baseline) / (1024 * 1024)


def snapshot_cuda_peak(device: str | torch.device) -> float:
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))


def reset_cuda_peak(device: str | torch.device) -> None:
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
