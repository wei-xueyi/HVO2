"""Photoelectric scene-to-current conversion and time-sampled detector output."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shutter import ShutterPlan


@dataclass(frozen=True)
class PhotocurrentTrace:
    """Current samples shaped (height, width, time), with a shared time axis."""

    time_s: np.ndarray
    current_ma: np.ndarray
    source_frame_index: np.ndarray
    maximum_exposure_s: float


def scene_frames_to_photocurrent(
    scene_frames: np.ndarray,
    *,
    code_max: float,
    current_min_ma: float,
    current_max_ma: float,
    gain: float,
) -> np.ndarray:
    """Linearly map scene codes to a photocurrent frame sequence."""
    frames = np.asarray(scene_frames, dtype=np.float64)
    if frames.ndim != 3 or frames.shape[0] == 0:
        raise ValueError("scene_frames must be nonempty with shape (frames, height, width)")
    if not np.all(np.isfinite(frames)):
        raise ValueError("scene_frames must contain finite values")
    if not np.isfinite(code_max) or code_max <= 0:
        raise ValueError("code_max must be finite and positive")
    if not all(np.isfinite(v) for v in (current_min_ma, current_max_ma, gain)) or gain <= 0:
        raise ValueError("current range and gain must be finite; gain must be positive")

    normalized = np.clip(frames / code_max, 0.0, 1.0)
    return (current_min_ma + (current_max_ma - current_min_ma) * normalized) * gain


def sample_photocurrent_through_shutter(
    photocurrent_frames_ma: np.ndarray,
    plan: ShutterPlan,
    *,
    simulation_dt_s: float,
    detector_settle_s: float,
) -> PhotocurrentTrace:
    """Sample the gated current at the model time step and append a dark tail."""
    frames = np.asarray(photocurrent_frames_ma, dtype=np.float64)
    if frames.ndim != 3 or frames.shape[0] == 0:
        raise ValueError("photocurrent_frames_ma must be nonempty and three-dimensional")
    if not np.all(np.isfinite(frames)):
        raise ValueError("photocurrent frames must be finite")
    if not np.isfinite(simulation_dt_s) or simulation_dt_s <= 0:
        raise ValueError("simulation_dt_s must be finite and positive")
    if not np.isfinite(detector_settle_s) or detector_settle_s < 0:
        raise ValueError("detector_settle_s must be finite and non-negative")

    duration_s = plan.maximum_window_s
    final_time_s = duration_s + detector_settle_s + simulation_dt_s
    regular = np.arange(int(np.floor(final_time_s / simulation_dt_s)) + 1) * simulation_dt_s
    # Preserve exact shutter and source-frame boundaries even with an arbitrary dt.
    shutters = np.asarray(plan.exposure_windows_ms) / 1000.0
    frame_edges = np.arange(1, frames.shape[0]) / plan.frame_rate_hz
    events = np.concatenate((regular, shutters, shutters + detector_settle_s,
                             frame_edges[frame_edges < duration_s], [final_time_s]))
    time_s = np.unique(np.round(events, 14))
    n_samples = time_s.size
    height, width = frames.shape[1:]
    current_ma = np.zeros((height, width, n_samples), dtype=np.float64)
    in_window = time_s < duration_s - 1.0e-14
    source_frame_index = np.floor(time_s * plan.frame_rate_hz + 1.0e-10).astype(np.int64)
    source_frame_index = np.minimum(source_frame_index, frames.shape[0] - 1)
    active_indices = np.flatnonzero(in_window)
    for index in active_indices:
        current_ma[:, :, index] = frames[source_frame_index[index]]

    return PhotocurrentTrace(
        time_s=time_s,
        current_ma=current_ma,
        source_frame_index=source_frame_index,
        maximum_exposure_s=duration_s,
    )


def make_photocurrent_trace(
    scene_frames: np.ndarray,
    plan: ShutterPlan,
    *,
    code_max: float,
    current_min_ma: float,
    current_max_ma: float,
    gain: float,
    simulation_dt_s: float,
    detector_settle_s: float,
) -> PhotocurrentTrace:
    """Apply the shutter first, then convert its scene frames to photocurrent."""
    from shutter import crop_scene_at_shutter

    captured_scene = crop_scene_at_shutter(scene_frames, plan)
    frame_currents = scene_frames_to_photocurrent(
        captured_scene,
        code_max=code_max,
        current_min_ma=current_min_ma,
        current_max_ma=current_max_ma,
        gain=gain,
    )
    cropped_plan = ShutterPlan(
        start_frame=0,
        frame_rate_hz=plan.frame_rate_hz,
        exposure_windows_ms=plan.exposure_windows_ms,
    )
    return sample_photocurrent_through_shutter(
        frame_currents,
        cropped_plan,
        simulation_dt_s=simulation_dt_s,
        detector_settle_s=detector_settle_s,
    )
