# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ParamSpec, TypeVar

import psutil

P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class PeakRSSSampler:
    """Sample process RSS in the background and retain the observed peak.

    A before/after RSS comparison can miss short-lived allocation spikes during model
    fitting. This sampler records RSS throughout the measured region at a configurable
    interval. It measures process resident-set size as exposed by ``psutil``; it is not
    a Python-allocation profiler and does not include memory owned by other processes.
    """

    interval_seconds: float = 0.01
    process: psutil.Process = field(default_factory=psutil.Process)
    peak_rss_bytes: int = field(init=False, default=0)
    samples: int = field(init=False, default=0)
    _stop_event: threading.Event = field(init=False, repr=False)
    _thread: threading.Thread | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self._stop_event = threading.Event()

    def _sample_once(self) -> None:
        rss = int(self.process.memory_info().rss)
        self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
        self.samples += 1

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self._sample_once()

    def start(self) -> PeakRSSSampler:
        if self._thread is not None:
            raise RuntimeError("PeakRSSSampler is already running")
        self._stop_event.clear()
        self._sample_once()
        self._thread = threading.Thread(
            target=self._run,
            name="vertimosaic-peak-rss",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> int:
        if self._thread is None:
            return self.peak_rss_bytes
        self._sample_once()
        self._stop_event.set()
        self._thread.join(timeout=max(1.0, self.interval_seconds * 10.0))
        self._thread = None
        return self.peak_rss_bytes

    def __enter__(self) -> PeakRSSSampler:
        return self.start()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()


def measure_peak_rss(
    callable_: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs
) -> tuple[R, int]:
    """Run a callable while sampling RSS and return ``(result, peak_rss_bytes)``."""

    with PeakRSSSampler() as sampler:
        result = callable_(*args, **kwargs)
    return result, sampler.peak_rss_bytes
