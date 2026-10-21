"""Intrinsic VO2 oscillation, pulse counting, and grayscale helpers.

The fitted oscillation frequency is distinct from the hardware readout
frequency, which is the capped integer count divided by the exposure window.
"""

from __future__ import annotations

import numpy as np

from finite_oscillation_model import FiniteOscillationNeuron, ModelParameters


def intrinsic_vo2_frequency_map(
    current_ma_map: np.ndarray,
    *,
    model_parameters: ModelParameters | None = None,
) -> np.ndarray:
    """Return each pixel's fitted intrinsic VO2 frequency in Hz."""
    currents = np.asarray(current_ma_map, dtype=float)
    if currents.ndim != 2 or currents.size == 0:
        raise ValueError("current_ma_map must be a nonempty 2D array")
    if not np.all(np.isfinite(currents)):
        raise ValueError("current_ma_map must contain finite values")

    model = FiniteOscillationNeuron(parameters=model_parameters or ModelParameters())
    unique_currents, inverse = np.unique(np.abs(currents), return_inverse=True)
    frequencies_hz = np.fromiter(
        (model.frequency_hz(float(current_ma)) for current_ma in unique_currents),
        dtype=float,
        count=unique_currents.size,
    )
    return frequencies_hz[inverse].reshape(currents.shape)


def map_intrinsic_vo2_frequency_to_grayscale(frequency_hz_map: np.ndarray) -> np.ndarray:
    """Map one image's minimum and maximum frequencies to 0 and 255."""
    frequencies = np.asarray(frequency_hz_map, dtype=float)
    if frequencies.ndim != 2 or frequencies.size == 0:
        raise ValueError("frequency_hz_map must be a nonempty 2D array")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies < 0.0):
        raise ValueError("frequency_hz_map must contain finite non-negative values")

    minimum = float(frequencies.min())
    maximum = float(frequencies.max())
    if np.isclose(maximum, minimum):
        fill_value = 0 if maximum == 0.0 else 255
        return np.full(frequencies.shape, fill_value, dtype=np.uint8)

    normalized = (frequencies - minimum) / (maximum - minimum)
    return np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)


def simulate_intrinsic_vo2_exposure(
    current_ma_map: np.ndarray,
    exposure_time_ms: float,
    *,
    max_pulses: int = 256,
    model_parameters: ModelParameters | None = None,
) -> np.ndarray:
    """Accumulate input-dependent VO2 pulses without a finite lifetime."""
    if not np.isfinite(exposure_time_ms) or exposure_time_ms <= 0:
        raise ValueError("exposure_time_ms must be finite and positive")
    if not isinstance(max_pulses, (int, np.integer)) or isinstance(max_pulses, (bool, np.bool_)) or max_pulses <= 0:
        raise ValueError("max_pulses must be a positive integer")

    frequencies_hz = intrinsic_vo2_frequency_map(
        current_ma_map,
        model_parameters=model_parameters,
    )
    pulse_counts = np.floor(
        frequencies_hz * (exposure_time_ms / 1000.0)
    )
    return np.minimum(pulse_counts, max_pulses).astype(np.uint16)
