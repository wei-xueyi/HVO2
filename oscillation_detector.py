"""Shared-window detector for phase changes in simulated pixel arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from video_process import PixelReactivationError


@dataclass(frozen=True)
class OscillationDetection:
    start_time_s: np.ndarray
    stop_time_s: np.ndarray
    duration_s: np.ndarray
    active: np.ndarray


def detect_oscillation_windows(
    phase_rad: np.ndarray,
    time_s: np.ndarray,
    *,
    detector_window_s: float,
    phase_change_tolerance_rad: float = 1.0e-12,
    pixel_offset: tuple[int, int] = (0, 0),
) -> OscillationDetection:
    """Require a full active window before onset and after the last change."""
    phase = np.asarray(phase_rad, dtype=float)
    time = np.asarray(time_s, dtype=float)
    if phase.ndim != 3 or phase.shape[2] != time.size or time.size < 2:
        raise ValueError("phase_rad must have shape (height, width, len(time_s))")
    if time.ndim != 1 or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        raise ValueError("time_s must be finite and strictly increasing")
    if not np.all(np.isfinite(phase)):
        raise ValueError("phase_rad must be finite")
    if not np.isfinite(detector_window_s) or detector_window_s <= 0:
        raise ValueError("detector_window_s must be finite and positive")
    if not np.isfinite(phase_change_tolerance_rad) or phase_change_tolerance_rad < 0:
        raise ValueError("phase_change_tolerance_rad must be finite and non-negative")

    changing = np.abs(np.diff(phase, axis=2)) > phase_change_tolerance_rad
    active_count = np.count_nonzero(changing, axis=2)
    first_index = np.argmax(changing, axis=2)
    last_index = changing.shape[2] - 1 - np.argmax(changing[:, :, ::-1], axis=2)
    has_activity = active_count > 0
    span_count = np.where(has_activity, last_index - first_index + 1, 0)
    reactivated = has_activity & (active_count != span_count)
    if np.any(reactivated):
        coordinates = np.argwhere(reactivated)
        coords = [
            (int(y + pixel_offset[0]), int(x + pixel_offset[1]))
            for y, x in coordinates
        ]
        raise PixelReactivationError(coords, len(coords))

    start = np.full(phase.shape[:2], np.nan, dtype=float)
    stop = np.full(phase.shape[:2], np.nan, dtype=float)
    duration = np.full(phase.shape[:2], np.nan, dtype=float)
    start_candidate = time[first_index] + detector_window_s
    stop_candidate = time[last_index + 1] + detector_window_s
    measurable = (
        has_activity
        & ((time[last_index + 1] - time[first_index]) >= detector_window_s - 1.0e-12)
        & (stop_candidate <= time[-1] + 1.0e-12)
    )
    start[measurable] = start_candidate[measurable]
    stop[measurable] = stop_candidate[measurable]
    duration[measurable] = stop_candidate[measurable] - start_candidate[measurable]
    return OscillationDetection(start, stop, duration, measurable)
