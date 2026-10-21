"""Exposure-specific outputs from phase arrays and oscillator detectors."""

from __future__ import annotations

import numpy as np

from oscillation_detector import OscillationDetection, detect_oscillation_windows
from readout_modules import accumulate_phase_pulses


def read_phase_array(
    phase_rad: np.ndarray,
    time_s: np.ndarray,
    *,
    detector_window_s: float,
    maximum_pulses: float,
) -> tuple[OscillationDetection, np.ndarray]:
    detection = detect_oscillation_windows(
        phase_rad,
        time_s,
        detector_window_s=detector_window_s,
    )
    pulses = accumulate_phase_pulses(
        phase_rad,
        time_s,
        detection.start_time_s,
        detection.stop_time_s,
        detection.active,
        maximum_pulses=maximum_pulses,
    )
    return detection, pulses


def read_hvo2_phase_array(
    phase_rad: np.ndarray,
    time_s: np.ndarray,
    *,
    detector_window_s: float,
    maximum_pulses: float,
) -> tuple[OscillationDetection, np.ndarray, np.ndarray]:
    detection, pulses = read_phase_array(
        phase_rad,
        time_s,
        detector_window_s=detector_window_s,
        maximum_pulses=maximum_pulses,
    )
    completion_time = np.where(detection.active, detection.stop_time_s, np.nan)
    return detection, pulses, completion_time


def read_intrinsic_vo2_windows(
    phase_rad: np.ndarray,
    time_s: np.ndarray,
    shutter_windows_ms: tuple[float, ...],
    *,
    detector_window_s: float,
    maximum_pulses: float,
) -> dict[float, tuple[OscillationDetection, np.ndarray]]:
    """Read one longest intrinsic phase trace at each shared shutter window."""
    outputs = {}
    for window_ms, (window_phase, window_time) in truncate_phase_windows(
        phase_rad, time_s, shutter_windows_ms, detector_window_s=detector_window_s
    ).items():
        outputs[window_ms] = read_phase_array(
            window_phase,
            window_time,
            detector_window_s=detector_window_s,
            maximum_pulses=maximum_pulses,
        )
    return outputs


def truncate_phase_windows(phase_rad, time_s, shutter_windows_ms, *, detector_window_s):
    """Return real phase/time arrays, with exact closure and a constant dark tail.

    This performs shutter gating only; detection and counting remain independent.
    """
    phase = np.asarray(phase_rad, dtype=float)
    time = np.asarray(time_s, dtype=float)
    if (time.ndim != 1 or time.size < 2 or phase.ndim != 3
            or phase.shape[2] != time.size or not np.all(np.isfinite(time))
            or np.any(np.diff(time) <= 0) or not np.all(np.isfinite(phase))):
        raise ValueError("phase/time arrays must be finite with a strictly increasing time axis")
    if not np.isfinite(detector_window_s) or detector_window_s <= 0:
        raise ValueError("detector window must be finite and positive")
    outputs = {}
    for window_ms in shutter_windows_ms:
        end = window_ms / 1000.0
        confirmed = end + detector_window_s
        if not np.isfinite(end) or end <= time[0] or confirmed > time[-1] + 1.0e-12:
            raise ValueError("shutter and confirmation tail must lie within the phase trace")
        target_time = np.unique(np.round(np.concatenate((time[time < confirmed - 1.0e-14],
                                                       [end, confirmed])), 14))
        query = np.minimum(target_time, end)
        lo = np.clip(np.searchsorted(time, query, side='right') - 1, 0, time.size - 2)
        fraction = (query - time[lo]) / (time[lo + 1] - time[lo])
        window_phase = phase[:, :, lo] + (phase[:, :, lo + 1] - phase[:, :, lo]) * fraction
        outputs[window_ms] = (window_phase, target_time)
    return outputs
