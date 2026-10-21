"""Time-window gating for a frame-based optical scene."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ShutterPlan:
    """One set of exposure windows sharing a start frame and frame rate."""

    start_frame: int
    frame_rate_hz: float
    exposure_windows_ms: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.start_frame, int) or isinstance(self.start_frame, bool) or self.start_frame < 0:
            raise ValueError("start_frame must be a non-negative integer")
        if not np.isfinite(self.frame_rate_hz) or self.frame_rate_hz <= 0:
            raise ValueError("frame_rate_hz must be finite and positive")
        if not self.exposure_windows_ms:
            raise ValueError("at least one shutter window is required")
        if any(not np.isfinite(value) or value <= 0 for value in self.exposure_windows_ms):
            raise ValueError("shutter windows must be finite and positive")

    @property
    def maximum_window_s(self) -> float:
        return max(self.exposure_windows_ms) / 1000.0


def shutter_frame_count(plan: ShutterPlan) -> int:
    """Number of source frames required to cover the longest exposure."""
    return max(1, int(np.ceil(plan.maximum_window_s * plan.frame_rate_hz - 1.0e-12)))


def crop_scene_at_shutter(
    scene_frames: np.ndarray,
    plan: ShutterPlan,
) -> np.ndarray:
    """Select the configured source-frame interval for the common shutter."""
    frames = np.asarray(scene_frames)
    if frames.ndim != 3 or frames.shape[0] == 0:
        raise ValueError("scene_frames must be nonempty with shape (frames, height, width)")
    stop = plan.start_frame + shutter_frame_count(plan)
    if stop > frames.shape[0]:
        raise ValueError(
            f"scene has {frames.shape[0]} frames, but shutter needs frame {stop - 1}"
        )
    return frames[plan.start_frame:stop]
