"""Time-resolved phase trajectories for finite and intrinsic VO2 arrays."""

from __future__ import annotations

import warnings
import numpy as np

from finite_oscillation_model import BelowThresholdCurrentWarning, FiniteOscillationNeuron, ModelParameters
from photodetector_model import PhotocurrentTrace
from video_process import InsufficientCurrentError


def _parameter_maps(
    current_map: np.ndarray,
    parameters: ModelParameters,
    *,
    finite_lifetime: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model = FiniteOscillationNeuron(parameters)
    unique, inverse = np.unique(np.abs(current_map), return_inverse=True)
    frequencies = np.fromiter(
        (model.frequency_hz(float(current)) for current in unique),
        dtype=float,
        count=unique.size,
    )
    strengthening = np.fromiter(
        (model.strengthening_rate_per_s(float(current)) if finite_lifetime else 0.0 for current in unique),
        dtype=float,
        count=unique.size,
    )
    shape = current_map.shape
    return frequencies[inverse].reshape(shape), strengthening[inverse].reshape(shape)


def run_intrinsic_vo2_phase_array(
    trace: PhotocurrentTrace,
    parameters: ModelParameters,
) -> np.ndarray:
    """Integrate unbounded phase while the shutter is open; return (H,W,T)."""
    return _integrate_phase(trace, parameters, finite_lifetime=False)


def run_hvo2_phase_array(
    trace: PhotocurrentTrace,
    parameters: ModelParameters,
) -> np.ndarray:
    """Integrate phase with the HVO2 hidden strengthening state; return (H,W,T)."""
    return _integrate_phase(trace, parameters, finite_lifetime=True)


def _integrate_phase(
    trace: PhotocurrentTrace,
    parameters: ModelParameters,
    *,
    finite_lifetime: bool,
) -> np.ndarray:
    currents = trace.current_ma
    time_s = trace.time_s
    height, width, n_times = currents.shape
    if time_s.ndim != 1 or time_s.size != n_times or n_times < 2:
        raise ValueError("trace time axis must match current_ma and contain at least two samples")
    dt_s = np.diff(time_s)
    if not np.all(np.isfinite(time_s)) or np.any(dt_s <= 0) or not np.all(np.isfinite(currents)):
        raise ValueError("current samples must be finite and time must strictly increase")
    magnitudes = np.abs(currents)
    if not np.any(magnitudes >= parameters.minimum_oscillation_current_ma):
        raise InsufficientCurrentError()
    if np.any((magnitudes > 1.0e-15) & (magnitudes < parameters.minimum_oscillation_current_ma)):
        warnings.warn("部分电流太小无法起振；停振后再次起振将触发复振错误。",
                      BelowThresholdCurrentWarning, stacklevel=2)
    del magnitudes

    phase = np.zeros((height, width, n_times), dtype=np.float64)
    h = np.zeros((height, width), dtype=np.float64) if finite_lifetime else None
    frame_maps = {}
    for source_frame in np.unique(trace.source_frame_index):
        frame_currents = currents[:, :, np.flatnonzero(trace.source_frame_index == source_frame)[0]]
        frame_maps[int(source_frame)] = _parameter_maps(frame_currents, parameters, finite_lifetime=finite_lifetime)

    for index in range(n_times - 1):
        step_s = float(dt_s[index])
        phase[:, :, index + 1] = phase[:, :, index]
        frequency_map, strengthening_map = frame_maps[int(trace.source_frame_index[index])]
        current_map = currents[:, :, index]
        above_threshold = (
            np.abs(current_map) >= parameters.minimum_oscillation_current_ma
        ) & (np.abs(current_map) > 1.0e-15)

        if finite_lifetime:
            active = above_threshold & (h < 1.0 - 1.0e-12)
            phase[:, :, index + 1] += np.where(
                active,
                2.0 * np.pi * frequency_map * step_s,
                0.0,
            )
            h_next = h.copy()
            h_next[active] = np.minimum(
                1.0,
                h[active] + strengthening_map[active] * step_s,
            )
            input_on = np.abs(current_map) > 1.0e-15
            # Match the single-device model: any nonzero input holds saturated h.
            recovering = ~active & ~(input_on & (h >= 1.0)) & (h > 0.0)
            h_next[recovering] = np.maximum(
                0.0,
                h[recovering] - step_s / parameters.recovery_time_s,
            )
            h = h_next
        else:
            phase[:, :, index + 1] += np.where(
                above_threshold,
                2.0 * np.pi * frequency_map * step_s,
                0.0,
            )

    return phase
