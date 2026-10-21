"""Regression tests for non-finite VO2 exposure pulse accumulation."""

import contextlib
import io
import json
import unittest
from pathlib import Path

import numpy as np

from finite_oscillation_model import FiniteOscillationNeuron
from intrinsic_vo2_model import (
    intrinsic_vo2_frequency_map,
    map_intrinsic_vo2_frequency_to_grayscale,
    simulate_intrinsic_vo2_exposure,
)


class IntrinsicVO2ExposureTests(unittest.TestCase):
    def test_frequency_threshold_is_reused(self):
        currents = np.array([[4.549, 4.55, 5.0]])
        counts = simulate_intrinsic_vo2_exposure(currents, 10.0)
        expected = np.array([[0, 204, 206]], dtype=np.uint16)
        np.testing.assert_array_equal(counts, expected)

    def test_each_exposure_starts_at_zero_and_counts_full_window_cycles(self):
        current = 7.32
        frequency_hz = FiniteOscillationNeuron().frequency_hz(current)
        for exposure_ms in (10.0, 40.0, 90.0):
            with self.subTest(exposure_ms=exposure_ms):
                expected = min(int(np.floor(frequency_hz * exposure_ms / 1000.0)), 256)
                counts = simulate_intrinsic_vo2_exposure(np.full((2, 3), current), exposure_ms)
                self.assertEqual(counts.dtype, np.uint16)
                np.testing.assert_array_equal(counts, np.full((2, 3), expected, dtype=np.uint16))

    def test_long_exposures_saturate_at_256_pulses(self):
        counts = simulate_intrinsic_vo2_exposure(np.array([[7.0, 7.5]]), 40.0)
        np.testing.assert_array_equal(counts, np.full((1, 2), 256, dtype=np.uint16))

    def test_frequency_map_reuses_threshold_and_is_independent_of_exposure(self):
        currents = np.array([[4.549, 4.55, 5.0]])
        frequencies = intrinsic_vo2_frequency_map(currents)
        model = FiniteOscillationNeuron()
        np.testing.assert_array_equal(
            frequencies,
            np.array([[0.0, model.frequency_hz(4.55), model.frequency_hz(5.0)]]),
        )

    def test_frequency_grayscale_uses_each_image_minimum_and_maximum(self):
        frequencies = np.array([[100.0, 125.0, 150.0]])
        grayscale = map_intrinsic_vo2_frequency_to_grayscale(frequencies)
        np.testing.assert_array_equal(grayscale, np.array([[0, 127, 255]], dtype=np.uint8))

    def test_constant_frequency_image_matches_hvo2_normalization_convention(self):
        np.testing.assert_array_equal(
            map_intrinsic_vo2_frequency_to_grayscale(np.full((2, 2), 21000.0)),
            np.full((2, 2), 255, dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            map_intrinsic_vo2_frequency_to_grayscale(np.zeros((2, 2))),
            np.zeros((2, 2), dtype=np.uint8),
        )

    def test_accepts_custom_pulse_cap(self):
        counts = simulate_intrinsic_vo2_exposure(np.array([[7.32]]), 40.0, max_pulses=128)
        self.assertEqual(int(counts[0, 0]), 128)

    def test_notebook_phase_detector_and_accumulator_are_connected(self):
        from photodetector_model import make_photocurrent_trace
        from phase_array import run_hvo2_phase_array, run_intrinsic_vo2_phase_array
        from oscillation_detector import detect_oscillation_windows
        from array_readout import truncate_phase_windows
        from readout_modules import accumulate_phase_pulses
        from shutter import ShutterPlan
        from finite_oscillation_model import ModelParameters

        windows = (5.0, 10.0, 90.0)
        trace = make_photocurrent_trace(
            np.full((5, 1, 2), 7.32), ShutterPlan(0, 50.0, windows),
            code_max=10.0, current_min_ma=0.0, current_max_ma=10.0,
            gain=1.0, simulation_dt_s=0.0001, detector_settle_s=0.0003,
        )
        parameters = ModelParameters()
        scope = dict(
            np=np, PHOTOCURRENT_TRACE=trace, HVO2_MODEL_PARAMETERS=parameters,
            SHUTTER_WINDOWS_MS=windows, OSCILLATION_DETECTOR_WINDOW_S=0.0003,
            PHASE_CHANGE_TOLERANCE_RAD=1e-12, PULSE_ACCUMULATOR_MIN=0.0,
            PULSE_ACCUMULATOR_MAX=255.0,
            run_intrinsic_vo2_phase_array=run_intrinsic_vo2_phase_array,
            truncate_phase_windows=truncate_phase_windows,
            detect_oscillation_windows=detect_oscillation_windows,
            accumulate_phase_pulses=accumulate_phase_pulses,
            HVO2_PHASE_ARRAY=run_hvo2_phase_array(trace, parameters),
        )
        notebook = json.loads(Path(__file__).with_name('exposure_whole_process.ipynb').read_text(encoding='utf-8'))
        cells = {c['id']: c for c in notebook['cells']}
        with contextlib.redirect_stdout(io.StringIO()):
            for key in ('intrinsic-array', 'oscillation-detectors', 'pulse-accumulator'):
                exec(compile(''.join(cells[key]['source']), key, 'exec'), scope)
        f = FiniteOscillationNeuron().frequency_hz(7.32)
        for ms in windows:
            detection = scope['INTRINSIC_VO2_DETECTIONS'][ms]
            counts = scope['INTRINSIC_VO2_PULSE_COUNT_MAPS'][ms]
            np.testing.assert_allclose(detection.duration_s, ms / 1000)
            np.testing.assert_allclose(counts, min(255, f * (ms / 1000 - .0003)))


    def test_rejects_invalid_inputs(self):
        invalid_cases = (
            (np.array([]), 10.0, {}),
            (np.array([7.0]), 10.0, {}),
            (np.array([[np.nan]]), 10.0, {}),
            (np.array([[7.0]]), 0.0, {}),
            (np.array([[7.0]]), np.inf, {}),
            (np.array([[np.inf]]), 10.0, {}),
            (np.array([[7.0]]), 10.0, {"max_pulses": 0}),
            (np.array([[7.0]]), 10.0, {"max_pulses": 1.5}),
        )
        for current, exposure_ms, kwargs in invalid_cases:
            with self.subTest(current=current, exposure_ms=exposure_ms, kwargs=kwargs):
                with self.assertRaises(ValueError):
                    simulate_intrinsic_vo2_exposure(current, exposure_ms, **kwargs)

        for frequencies in (np.array([]), np.array([21000.0]), np.array([[np.nan]]), np.array([[-1.0]])):
            with self.subTest(frequencies=frequencies):
                with self.assertRaises(ValueError):
                    map_intrinsic_vo2_frequency_to_grayscale(frequencies)


if __name__ == "__main__":
    unittest.main(verbosity=2)
