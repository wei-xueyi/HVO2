"""Pulse accumulation and CMOS charge integration for pixel arrays."""

from __future__ import annotations

import numpy as np

ELECTRON_CHARGE_C = 1.602176634e-19


def _phase_at(phase_rad: np.ndarray, time_s: np.ndarray, query_s: np.ndarray) -> np.ndarray:
    lo = np.searchsorted(time_s, query_s, side="right") - 1
    lo = np.clip(lo, 0, time_s.size - 2)
    hi = lo + 1
    y, x = np.indices(query_s.shape)
    fraction = (query_s - time_s[lo]) / (time_s[hi] - time_s[lo])
    return phase_rad[y, x, lo] + fraction * (phase_rad[y, x, hi] - phase_rad[y, x, lo])


def accumulate_phase_pulses(
    phase_rad: np.ndarray,
    time_s: np.ndarray,
    active_start_s: np.ndarray,
    active_stop_s: np.ndarray,
    valid_mask: np.ndarray,
    *,
    minimum_pulses: float = 0.0,
    maximum_pulses: float = 255.0,
) -> np.ndarray:
    """Return fractional phase cycles clipped to the configured counter range."""
    phase = np.asarray(phase_rad, dtype=float)
    starts = np.asarray(active_start_s, dtype=float)
    stops = np.asarray(active_stop_s, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool)
    time_s = np.asarray(time_s, dtype=float)
    if (time_s.ndim != 1 or time_s.size < 2 or phase.shape[-1] != time_s.size
            or not np.all(np.isfinite(time_s)) or np.any(np.diff(time_s) <= 0)
            or not np.all(np.isfinite(phase))):
        raise ValueError("phase/time must be finite and time must strictly increase")
    if phase.ndim != 3 or starts.shape != phase.shape[:2] or stops.shape != starts.shape:
        raise ValueError("phase and detector window map shapes do not match")
    if valid.shape != starts.shape:
        raise ValueError("valid_mask must match the pixel map shape")
    if not np.isfinite(minimum_pulses) or minimum_pulses < 0:
        raise ValueError("minimum_pulses must be finite and non-negative")
    if not np.isfinite(maximum_pulses) or maximum_pulses <= minimum_pulses:
        raise ValueError("maximum_pulses must be finite and greater than minimum_pulses")
    if np.any(valid & (~np.isfinite(starts) | ~np.isfinite(stops)
                      | (starts < time_s[0] - 1.0e-12) | (stops > time_s[-1] + 1.0e-12)
                      | (stops < starts))):
        raise ValueError("detector intervals must be finite, ordered, and within the trace")

    counts = np.full(starts.shape, minimum_pulses, dtype=float)
    if np.any(valid):
        safe_starts = np.where(valid, starts, time_s[0])
        safe_stops = np.where(valid, stops, time_s[0])
        phase_start = _phase_at(phase, time_s, safe_starts)
        phase_stop = _phase_at(phase, time_s, safe_stops)
        counts[valid] = np.clip(
            (phase_stop[valid] - phase_start[valid]) / (2.0 * np.pi),
            minimum_pulses,
            maximum_pulses,
        )
    return counts


def integrate_cmos_charge(
    photocurrent_ma_map: np.ndarray,
    exposure_time_s: float,
    *,
    amplifier_gain: float,
    minimum_charge_electrons: float = 0.0,
    maximum_charge_electrons: float = 12_000.0,
) -> np.ndarray:
    """Integrate current and saturate it to the CMOS charge range."""
    current = np.asarray(photocurrent_ma_map, dtype=float)
    if current.ndim != 2 or current.size == 0 or not np.all(np.isfinite(current)):
        raise ValueError("photocurrent_ma_map must be a nonempty finite 2D map")
    if not np.isfinite(exposure_time_s) or exposure_time_s <= 0:
        raise ValueError("exposure_time_s must be finite and positive")
    if not np.isfinite(amplifier_gain) or amplifier_gain <= 0:
        raise ValueError("amplifier_gain must be finite and positive")
    _validate_cmos_charge_range(minimum_charge_electrons, maximum_charge_electrons)
    ideal_charge = np.maximum(current, 0.0) * 1.0e-3 / ELECTRON_CHARGE_C
    ideal_charge *= amplifier_gain * exposure_time_s
    return np.clip(ideal_charge, minimum_charge_electrons, maximum_charge_electrons)


def integrate_cmos_current_trace(
    photocurrent_trace,
    shutter_windows_ms: tuple[float, ...],
    *,
    amplifier_gain: float,
    minimum_charge_electrons: float = 0.0,
    maximum_charge_electrons: float = 12_000.0,
) -> dict[float, np.ndarray]:
    """Integrate the shutter-gated time-varying current for each shared window."""
    current = np.asarray(photocurrent_trace.current_ma, dtype=float)
    time_s = np.asarray(photocurrent_trace.time_s, dtype=float)
    if current.ndim != 3 or current.shape[2] != time_s.size or time_s.size < 2:
        raise ValueError("photocurrent trace and its time axis do not match")
    steps = np.diff(time_s)
    if (not np.all(np.isfinite(time_s)) or np.any(steps <= 0)
            or not np.all(np.isfinite(current))):
        raise ValueError("CMOS current must be finite and time must strictly increase")
    if not shutter_windows_ms or any(not np.isfinite(w) or w <= 0 for w in shutter_windows_ms):
        raise ValueError("shutter windows must be finite and positive")
    if not np.isfinite(amplifier_gain) or amplifier_gain <= 0:
        raise ValueError("amplifier_gain must be finite and positive")
    _validate_cmos_charge_range(minimum_charge_electrons, maximum_charge_electrons)
    output = {}
    for exposure_ms in shutter_windows_ms:
        exposure_s = exposure_ms / 1000.0
        if time_s[0] != 0 or exposure_s > photocurrent_trace.maximum_exposure_s + 1.0e-12:
            raise ValueError("CMOS exposure must lie within the common shutter trace")
        weights = np.clip(exposure_s - time_s[:-1], 0.0, steps)
        integrated_current_as = np.einsum('hwt,t->hw', current[:, :, :-1], weights)
        ideal_charge = np.maximum(integrated_current_as, 0.0) * (
            amplifier_gain * 1.0e-3 / ELECTRON_CHARGE_C
        )
        output[exposure_ms] = np.clip(
            ideal_charge,
            minimum_charge_electrons,
            maximum_charge_electrons,
        )
    return output


def map_cmos_charge(
    charge_electrons,
    *,
    minimum_charge_electrons: float = 0.0,
    maximum_charge_electrons: float = 12_000.0,
):
    """Clip charge and map the fixed sensor range to 8-bit gray."""
    charge = np.asarray(charge_electrons, dtype=float)
    if charge.ndim != 2 or charge.size == 0 or not np.all(np.isfinite(charge)):
        raise ValueError("charge_electrons must be a nonempty finite 2D map")
    _validate_cmos_charge_range(minimum_charge_electrons, maximum_charge_electrons)
    clipped = np.clip(charge, minimum_charge_electrons, maximum_charge_electrons)
    normalized = (clipped - minimum_charge_electrons) / (
        maximum_charge_electrons - minimum_charge_electrons
    )
    return np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)


def _validate_cmos_charge_range(minimum_charge_electrons, maximum_charge_electrons):
    if (not np.isfinite(minimum_charge_electrons)
            or not np.isfinite(maximum_charge_electrons)
            or minimum_charge_electrons < 0
            or maximum_charge_electrons <= minimum_charge_electrons):
        raise ValueError("CMOS maximum charge must be finite and greater than a non-negative minimum")
