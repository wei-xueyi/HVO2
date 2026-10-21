"""Tests for shutter, detector, phase-array, and readout boundaries."""

import unittest

import numpy as np

from array_readout import read_hvo2_phase_array, read_intrinsic_vo2_windows
from finite_oscillation_model import ModelParameters
from oscillation_detector import detect_oscillation_windows
from photodetector_model import (
    PhotocurrentTrace,
    make_photocurrent_trace,
    scene_frames_to_photocurrent,
)
from phase_array import run_hvo2_phase_array, run_intrinsic_vo2_phase_array
from readout_modules import (
    accumulate_phase_pulses,
    integrate_cmos_current_trace,
    map_cmos_charge,
)
from shutter import ShutterPlan, crop_scene_at_shutter, shutter_frame_count
from video_process import PixelReactivationError


class HardwarePipelineTests(unittest.TestCase):
    def setUp(self):
        self.dt = 1.0e-4
        self.windows_ms = (5.0, 10.0, 90.0)
        self.plan = ShutterPlan(0, 50.0, self.windows_ms)
        self.parameters = ModelParameters()

    def test_shutter_crops_source_and_reports_frame_requirement(self):
        scene = np.arange(8, dtype=float)[:, None, None]
        plan = ShutterPlan(2, 50.0, self.windows_ms)
        self.assertEqual(shutter_frame_count(plan), 5)
        np.testing.assert_array_equal(crop_scene_at_shutter(scene, plan)[:, 0, 0], [2, 3, 4, 5, 6])
        with self.assertRaises(ValueError):
            crop_scene_at_shutter(scene[:5], plan)

    def test_photodetector_maps_scene_and_gates_after_longest_shutter(self):
        scene = np.array([[[0.0]], [[128.0]], [[255.0]], [[255.0]], [[255.0]]])
        photo_frames = scene_frames_to_photocurrent(
            scene,
            code_max=255.0,
            current_min_ma=0.0,
            current_max_ma=10.0,
            gain=1.0,
        )
        np.testing.assert_allclose(photo_frames[:, 0, 0], [0.0, 1280 / 255, 10.0, 10.0, 10.0])
        trace = make_photocurrent_trace(
            scene,
            self.plan,
            code_max=255.0,
            current_min_ma=0.0,
            current_max_ma=10.0,
            gain=1.0,
            simulation_dt_s=self.dt,
            detector_settle_s=0.0003,
        )
        self.assertEqual(trace.current_ma.shape, (1, 1, trace.time_s.size))
        self.assertLessEqual(trace.time_s[-1], 0.0905)
        self.assertTrue(np.all(trace.current_ma[:, :, trace.time_s >= 0.09] == 0.0))

    def test_hvo2_and_intrinsic_phase_outputs_share_shape(self):
        source = np.full((3, 1, 1), 7.32)
        plan = ShutterPlan(0, 50.0, (2.0, 5.0))
        trace = make_photocurrent_trace(
            source,
            plan,
            code_max=10.0,
            current_min_ma=0.0,
            current_max_ma=10.0,
            gain=1.0,
            simulation_dt_s=self.dt,
            detector_settle_s=0.0003,
        )
        hvo = run_hvo2_phase_array(trace, self.parameters)
        intrinsic = run_intrinsic_vo2_phase_array(trace, self.parameters)
        self.assertEqual(hvo.shape, intrinsic.shape)
        self.assertEqual(hvo.shape, (1, 1, trace.time_s.size))
        self.assertGreater(hvo[0, 0, -1], 0.0)
        self.assertGreater(intrinsic[0, 0, -1], hvo[0, 0, -1])

    def test_detector_applies_matching_activation_and_release_delay(self):
        dt = 0.0001
        time = np.arange(101) * dt
        phase = np.zeros((1, 1, time.size))
        running = (time[:-1] >= 0.001) & (time[:-1] < 0.004)
        phase[0, 0, 1:] = np.cumsum(running.astype(float) * (2 * np.pi * 2000 * dt))
        result, pulses, completed = read_hvo2_phase_array(
            phase,
            time,
            detector_window_s=0.0003,
            maximum_pulses=255.0,
        )
        self.assertAlmostEqual(result.start_time_s[0, 0], 0.0013)
        self.assertAlmostEqual(result.stop_time_s[0, 0], 0.0043)
        self.assertAlmostEqual(result.duration_s[0, 0], 0.003)
        self.assertAlmostEqual(pulses[0, 0], 5.4)
        self.assertAlmostEqual(completed[0, 0], 0.0043)

    def test_short_activity_never_activates_detector_and_window_clips_intrinsic(self):
        dt = 0.0001
        time = np.arange(101) * dt
        phase = np.zeros((1, 1, time.size))
        phase[0, 0, 1:3] = 1.0
        phase[0, 0, 2:] = 2.0
        result = detect_oscillation_windows(phase, time, detector_window_s=0.0003)
        self.assertFalse(result.active.any())

        phase[0, 0] = np.arange(time.size) * 0.1
        windows = read_intrinsic_vo2_windows(
            phase,
            time,
            (2.0, 5.0),
            detector_window_s=0.0003,
            maximum_pulses=255.0,
        )
        self.assertAlmostEqual(windows[2.0][0].stop_time_s[0, 0], 0.0023)
        self.assertAlmostEqual(windows[5.0][0].stop_time_s[0, 0], 0.0053)

    def test_detector_flags_restart_after_a_phase_gap(self):
        time = np.arange(101) * 0.0001
        phase = np.zeros((1, 1, time.size))
        phase[0, 0, :20] = np.arange(20) * 0.1
        phase[0, 0, 40:] = 1.9 + np.arange(61) * 0.1
        with self.assertRaises(PixelReactivationError):
            detect_oscillation_windows(phase, time, detector_window_s=0.0003)

    def test_phase_pulse_count_is_fractional_and_saturates_at_255(self):
        time = np.arange(11) * 0.001
        phase = np.zeros((1, 2, time.size))
        phase[0, 0] = 2 * np.pi * 1250.0 * time
        phase[0, 1] = 2 * np.pi * 40000.0 * time
        counts = accumulate_phase_pulses(
            phase,
            time,
            np.zeros((1, 2)),
            np.full((1, 2), 0.01),
            np.ones((1, 2), dtype=bool),
            maximum_pulses=255.0,
        )
        np.testing.assert_allclose(counts, [[12.5, 255.0]])

    def test_phase_pulse_accumulator_clamps_both_bounds_including_inactive_pixels(self):
        time = np.array([0.0, 1.0])
        phase = np.array([[[-4 * np.pi, -4 * np.pi], [0.0, 24 * np.pi],
                           [0.0, 2 * np.pi]]])
        counts = accumulate_phase_pulses(
            phase,
            time,
            np.zeros((1, 3)),
            np.ones((1, 3)),
            np.array([[True, True, False]]),
            minimum_pulses=3.0,
            maximum_pulses=9.0,
        )
        np.testing.assert_array_equal(counts, [[3.0, 9.0, 3.0]])

    def test_phase_pulse_accumulator_rejects_invalid_limits(self):
        phase = np.zeros((1, 1, 2))
        args = (phase, np.array([0.0, 1.0]), np.zeros((1, 1)),
                np.ones((1, 1)), np.ones((1, 1), dtype=bool))
        for minimum, maximum in ((-1.0, 10.0), (np.nan, 10.0), (5.0, 5.0), (6.0, 5.0)):
            with self.subTest(minimum=minimum, maximum=maximum):
                with self.assertRaises(ValueError):
                    accumulate_phase_pulses(*args, minimum_pulses=minimum,
                                            maximum_pulses=maximum)

    def test_cmos_integrates_shared_windows_with_configured_gain(self):
        trace = PhotocurrentTrace(
            time_s=np.arange(11) * 0.001,
            current_ma=np.full((1, 1, 11), 1.0),
            source_frame_index=np.zeros(11, dtype=int),
            maximum_exposure_s=0.01,
        )
        output = integrate_cmos_current_trace(
            trace,
            (5.0, 10.0),
            amplifier_gain=1.0e-13,
        )
        self.assertLess(output[5.0][0, 0], output[10.0][0, 0])

    def test_cmos_charge_is_clipped_and_mapped_to_fixed_sensor_range(self):
        image = map_cmos_charge(np.array([[-100.0, 0.0, 6000.0, 12000.0, 24000.0]]))
        np.testing.assert_array_equal(image, [[0, 0, 127, 255, 255]])

        trace = PhotocurrentTrace(
            time_s=np.array([0.0, 0.01]),
            current_ma=np.full((1, 2, 2), 1.0),
            source_frame_index=np.zeros(2, dtype=int),
            maximum_exposure_s=0.01,
        )
        charge = integrate_cmos_current_trace(
            trace,
            (10.0,),
            amplifier_gain=1e-5,
            maximum_charge_electrons=1000.0,
        )[10.0]
        np.testing.assert_array_equal(charge, [[1000.0, 1000.0]])

    def test_non_grid_shutters_and_frame_changes_keep_exact_times(self):
        from array_readout import truncate_phase_windows
        from finite_oscillation_model import FiniteOscillationNeuron
        plan = ShutterPlan(1, 123.0, (1.13, 12.37, 24.41))
        scene = np.array([6., 7., 7.2, 7.4, 7.5])[:, None, None]
        trace = make_photocurrent_trace(
            scene, plan, code_max=10., current_min_ma=0., current_max_ma=10.,
            gain=1., simulation_dt_s=.00017, detector_settle_s=.0003,
        )
        phase = run_intrinsic_vo2_phase_array(trace, self.parameters)
        model = FiniteOscillationNeuron()
        # Exact piecewise-constant current integral at the non-grid source boundary.
        first_frame = np.flatnonzero(np.isclose(trace.time_s, 1 / 123., atol=1e-13, rtol=0))[0]
        self.assertAlmostEqual(phase[0, 0, first_frame] / (2 * np.pi), model.frequency_hz(7.) / 123.)
        windows = read_intrinsic_vo2_windows(phase, trace.time_s, plan.exposure_windows_ms,
                                            detector_window_s=.0003, maximum_pulses=255.)
        for ms, (det, _) in windows.items():
            self.assertAlmostEqual(det.duration_s[0, 0], ms / 1000.)
        charges = integrate_cmos_current_trace(trace, plan.exposure_windows_ms,
                     amplifier_gain=1e-13)
        from readout_modules import ELECTRON_CHARGE_C
        expected_integral = 7 / 123 + 7.2 / 123 + 7.4 / 123 + 7.5 * (.02441 - 3 / 123)
        self.assertAlmostEqual(charges[24.41][0, 0], expected_integral * 1e-16 / ELECTRON_CHARGE_C)
        self.assertEqual(shutter_frame_count(ShutterPlan(0, 50., (20.,))), 1)

    def test_hvo_phase_matches_core_and_saturated_state_stays_locked(self):
        from finite_oscillation_model import FiniteOscillationNeuron
        import warnings
        model = FiniteOscillationNeuron()
        time = np.arange(1001) * .0001
        currents = np.where(time < .02, 7.32, np.where(time < .06, 4., 7.32))
        trace = PhotocurrentTrace(time, currents[None, None],
                                 np.where(time < .02, 0, np.where(time < .06, 1, 2)), .1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            phase = run_hvo2_phase_array(trace, self.parameters)
            reference = model.simulate(.1, .0001, lambda t: 7.32 if t < .02 else (4. if t < .06 else 7.32))
        np.testing.assert_allclose(phase[0, 0], reference.theta_rad)
        self.assertAlmostEqual(phase[0, 0, -1], phase[0, 0, 200])

    def test_all_subthreshold_current_reports_chinese_error(self):
        from video_process import InsufficientCurrentError
        trace = PhotocurrentTrace(np.arange(5) * .0001, np.full((1, 1, 5), 4.54), np.zeros(5, dtype=int), .0004)
        for runner in (run_hvo2_phase_array, run_intrinsic_vo2_phase_array):
            with self.assertRaisesRegex(InsufficientCurrentError, '电流太小无法起振'):
                runner(trace, self.parameters)

    def test_invalid_detector_and_accumulator_intervals_fail(self):
        time = np.array([0., .001, .002])
        phase = np.zeros((1, 1, 3))
        with self.assertRaises(ValueError):
            detect_oscillation_windows(phase, time[::-1], detector_window_s=.0003)
        with self.assertRaises(ValueError):
            accumulate_phase_pulses(phase, time, np.zeros((1, 1)), np.full((1, 1), .003),
                                    np.ones((1, 1), bool))


if __name__ == "__main__":
    unittest.main(verbosity=2)
