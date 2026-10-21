"""有限振荡 H-VO2 神经元的两状态最小模型。

模型只保留两个内部状态：

* ``theta``：快变量，表示振荡相位，用来生成周期性脉冲。
* ``h``：慢变量，表示归一化的氢稳定导电状态强化程度。

当输入电流不为零且 ``h < 1`` 时，器件保持振荡，同时 ``h`` 以
电流相关的速率增长。当 ``h`` 达到上限 1 后，振荡停止；只要输入
仍然存在，``h`` 会被锁定在停振状态。输入撤去后，``h`` 线性恢复到
零，因此完全饱和的器件会在 ``recovery_time_s`` 后完成恢复。

这个模块刻意保持为最小物理模型，不显式引入温度场、空间氢浓度或
相界面位置。这些影响被降阶到两个电流依赖函数：
``frequency_hz`` 和 ``strengthening_rate_per_s``。后者默认由拟合的
振荡持续时间曲线反推。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np


CurrentFunction = Callable[[float], float]
WaveformFunction = Callable[[np.ndarray], np.ndarray]
DurationCurveFunction = Callable[[float], float]

MIN_OSCILLATION_CURRENT_MA = 4.55
DURATION_CURVE_KNEE_CURRENT_MA = 6.099924


class BelowThresholdCurrentWarning(RuntimeWarning):
    """A nonzero input is below the current required to start oscillation."""


def fitted_oscillation_duration_ms(current_ma: float) -> float:
    """Default digitized duration fit; replace this callable when refitting data."""
    current = abs(float(current_ma))
    if current <= DURATION_CURVE_KNEE_CURRENT_MA:
        return 90.0

    delta_current = current - DURATION_CURVE_KNEE_CURRENT_MA
    return (
        79.4 * np.exp(-((delta_current / 0.2613) ** 2))
        + 10.6 * np.exp(-((delta_current / 1.2998) ** 2))
    )


#全部参数定义，初始参数确定后不许修改
@dataclass(frozen=True)
class ModelParameters:
    """所有模拟条件共享的全局参数。

    电流单位为 mA，时间单位为秒，频率单位为 Hz，电阻单位为欧姆。
    """

    # 频率拟合 f[kHz] = 18.56248169 + 0.42087232 * abs(current[mA])，
    # 以下用兼容的参考点形式存储，frequency_hz 仍返回 Hz。
    reference_current_ma: float = 7.32
    reference_frequency_hz: float = 21_643.2670724
    frequency_slope_hz_per_ma: float = 420.87232
    minimum_frequency_hz: float = 1.0

    # 强化速率回退模型使用的参考点；启用 duration_curve_ms 时不使用。
    reference_strengthening_rate_per_s: float = 75.76
    strengthening_slope_per_s_per_ma: float = 256.63
    minimum_strengthening_rate_per_s: float = 9.52
    minimum_oscillation_current_ma: float = MIN_OSCILLATION_CURRENT_MA
    duration_curve_ms: DurationCurveFunction | None = fitted_oscillation_duration_ms

    recovery_time_s: float = 1.0

    initial_high_resistance_ohm: float = 2_000.0
    initial_low_resistance_ohm: float = 500.0
    stopped_intermediate_resistance_ohm: float = 1_100.0
    #参数合法性检查
    def __post_init__(self) -> None:
        if self.minimum_frequency_hz <= 0:
            raise ValueError("minimum_frequency_hz must be positive")
        if self.minimum_strengthening_rate_per_s <= 0:
            raise ValueError("minimum_strengthening_rate_per_s must be positive")
        if (
            not np.isfinite(self.minimum_oscillation_current_ma)
            or self.minimum_oscillation_current_ma <= 0
        ):
            raise ValueError("minimum_oscillation_current_ma must be finite and positive")
        if self.duration_curve_ms is not None and not callable(self.duration_curve_ms):
            raise TypeError("duration_curve_ms must be callable or None")
        if self.recovery_time_s <= 0:
            raise ValueError("recovery_time_s must be positive")
        if self.initial_high_resistance_ohm <= self.initial_low_resistance_ohm:
            raise ValueError(
                "initial_high_resistance_ohm must exceed "
                "initial_low_resistance_ohm"
            )
        if self.stopped_intermediate_resistance_ohm <= 0:
            raise ValueError("stopped_intermediate_resistance_ohm must be positive")


#一轮仿真的全部输出
@dataclass(frozen=True)
class SimulationResult:
    """一次模型仿真的时间分辨输出。"""
    #输出均以数组形式储存
    time_s: np.ndarray
    current_ma: np.ndarray
    theta_rad: np.ndarray
    h: np.ndarray
    resistance_ohm: np.ndarray
    voltage_v: np.ndarray
    oscillating: np.ndarray

    @property
    def response_duration_s(self) -> float:
        """估算器件处于振荡状态的持续时间。"""

        active_indices = np.flatnonzero(self.oscillating)
        if active_indices.size == 0:
            return 0.0
        if self.time_s.size < 2:
            return 0.0
        dt_s = float(self.time_s[1] - self.time_s[0])
        return float(
            self.time_s[active_indices[-1]]
            - self.time_s[active_indices[0]]
            + dt_s
        )

    @property
    def phase_cycle_count(self) -> float:
        """快相位变量累计的周期数。(就是振荡周期数,是一个浮点数,可以有小数部分)"""

        if self.theta_rad.size < 2:
            return 0.0
        return float((self.theta_rad[-1] - self.theta_rad[0]) / (2.0 * np.pi))

    @property
    def complete_spike_count(self) -> int:
        """完整周期的整数个数，可作为第一版 spike 数估计。"""
        #这里为什么要分开计算？
        return int(np.floor(self.phase_cycle_count + 1.0e-12))

#整个模型的核心类,包含了振荡频率计算,强化速率计算,寿命计算,周期数计算,仿真等方法
class FiniteOscillationNeuron:
    """两状态有限振荡神经元模型。"""

    #支持自定义参数和波形函数,默认是正弦波
    def __init__(
        self,
        parameters: ModelParameters | None = None,
        waveform: WaveformFunction | None = None,
    ) -> None:
        self.parameters = parameters or ModelParameters()
        self.waveform = waveform or np.sin

    def frequency_hz(self, current_ma: float) -> float:
        """Return fitted frequency in Hz; inputs below the firing threshold return zero."""
        p = self.parameters
        if abs(current_ma) < p.minimum_oscillation_current_ma:
            return 0.0
        value = p.reference_frequency_hz + p.frequency_slope_hz_per_ma * (
            abs(current_ma) - p.reference_current_ma
        )
        return max(p.minimum_frequency_hz, value)

    def strengthening_rate_per_s(self, current_ma: float) -> float:
        """计算慢变量 ``h`` 的电流相关强化速率。"""
        p = self.parameters
        current = abs(current_ma)
        if current < p.minimum_oscillation_current_ma:
            return 0.0
        if p.duration_curve_ms is not None:
            duration_ms = float(p.duration_curve_ms(current))
            if not np.isfinite(duration_ms) or duration_ms <= 0.0:
                raise ValueError("duration_curve_ms must return a finite positive duration")
            return 1000.0 / duration_ms

        value = p.reference_strengthening_rate_per_s + (
            p.strengthening_slope_per_s_per_ma
            * (current - p.reference_current_ma)
        )
        return max(p.minimum_strengthening_rate_per_s, value)

    def intrinsic_lifetime_s(self, current_ma: float, initial_h: float = 0.0) -> float:
        """解析计算 ``h`` 到达上限前的内禀振荡寿命。"""

        self._validate_h(initial_h)
        if initial_h >= 1.0 or abs(current_ma) < self.parameters.minimum_oscillation_current_ma:
            return 0.0
        return (1.0 - initial_h) / self.strengthening_rate_per_s(current_ma)

    def expected_cycle_count(
        self,
        current_ma: float,
        pulse_width_s: float,
        initial_h: float = 0.0,
    ) -> float:
        """解析估计恒定矩形脉冲下的振荡周期数。"""

        if pulse_width_s < 0:
            raise ValueError("pulse_width_s must be non-negative")
        response_s = min(
            pulse_width_s,
            self.intrinsic_lifetime_s(current_ma, initial_h=initial_h),
        )
        return self.frequency_hz(current_ma) * response_s   #只是估算，可能差几个周期

    #最核心的函数
    def simulate(
        self,
        duration_s: float,  #总仿真时长
        dt_s: float,    #仿真时间步长
        current: CurrentFunction,   #电流函数,输入时间返回电流值
        *,
        initial_h: float = 0.0,
        initial_theta_rad: float = 0.0,
    ) -> SimulationResult:
        """模拟任意输入电流波形。

        ``current(t)`` 必须返回单位为 mA 的电流。正负电流只通过幅值
        影响 ``h``，因此反向电流不会提供额外的复位机制。
        """

        #参数合法性检查
        if duration_s < 0:
            raise ValueError("duration_s must be non-negative")
        if dt_s <= 0:
            raise ValueError("dt_s must be positive")
        self._validate_h(initial_h)

        # 先用固定步长构造时间轴；若总时长不能被步长整除，
        # 末尾会额外补上精确的终止时间点。
        n_steps = int(np.floor(duration_s / dt_s + 1.0e-12)) + 1
        time_s = np.arange(n_steps, dtype=float) * dt_s     #时间数组
        if time_s[-1] < duration_s - dt_s * 1.0e-9:
            time_s = np.append(time_s, duration_s)

        # 将输入电流函数离散化，后续每个时间点都使用同一份电流数组。
        current_ma = np.asarray([float(current(t)) for t in time_s], dtype=float)
        if not np.all(np.isfinite(current_ma)):
            raise ValueError("current(t) must return finite values")
        below_threshold = (
            (np.abs(current_ma) > 1.0e-15)
            & (np.abs(current_ma) < self.parameters.minimum_oscillation_current_ma)
        )
        if np.any(below_threshold):
            warnings.warn(
                "Nonzero input below 4.55 mA detected; those samples do not oscillate.",
                BelowThresholdCurrentWarning,
                stacklevel=2,
            )

        # 初始化输出数组
        theta_rad = np.empty_like(time_s)
        h = np.empty_like(time_s)
        resistance_ohm = np.empty_like(time_s)
        voltage_v = np.empty_like(time_s)
        oscillating = np.zeros(time_s.shape, dtype=bool)

        # initial_h 允许模拟“不完全恢复后再次输入”的情况。
        theta_rad[0] = float(initial_theta_rad)
        h[0] = float(initial_h)

        for index, t_s in enumerate(time_s):
            input_on = not np.isclose(current_ma[index], 0.0, atol=1.0e-15)
            above_oscillation_threshold = (
                abs(current_ma[index]) >= self.parameters.minimum_oscillation_current_ma
            )
            oscillating[index] = (
                input_on
                and above_oscillation_threshold
                and h[index] < 1.0 - 1.0e-12
            )

            # 电阻先由当前相位和慢变量计算，再在恒流条件下用 V=IR
            # 得到输出电压；这里不显式模拟电容充放电。
            resistance_ohm[index] = self._instantaneous_resistance_ohm(
                theta_rad[index], h[index], oscillating[index]
            )
            voltage_v[index] = current_ma[index] * 1.0e-3 * resistance_ohm[index]

            if index == time_s.size - 1:
                break

            step_s = float(time_s[index + 1] - t_s)
            # 先把当前状态复制到下一步，后续根据输入和慢变量更新。
            theta_rad[index + 1] = theta_rad[index]
            h[index + 1] = h[index]


            if input_on and above_oscillation_threshold and h[index] < 1.0:
                # 输入开启且尚未停振：相位按频率前进，慢变量线性强化。
                # 当前版本使用 dh/dt=k(I)，便于直接由寿命反推 k。
                theta_rad[index + 1] += (
                    2.0 * np.pi * self.frequency_hz(current_ma[index]) * step_s
                )
                h[index + 1] = min(
                    1.0,
                    h[index]
                    + self.strengthening_rate_per_s(current_ma[index]) * step_s,
                )
            elif input_on and h[index] >= 1.0:
                # 饱和停振状态：输入仍在时，h 被锁定在 1。
                h[index + 1] = 1.0
            elif h[index] > 0.0:
                # 零阶线性恢复：若 h=1，则恰好在 T_rec 后恢复到 0。
                h[index + 1] = max(
                    0.0,
                    h[index] - step_s / self.parameters.recovery_time_s,
                )

        return SimulationResult(
            time_s=time_s,
            current_ma=current_ma,
            theta_rad=theta_rad,
            h=h,
            resistance_ohm=resistance_ohm,
            voltage_v=voltage_v,
            oscillating=oscillating,
        )

    def simulate_pulse(
        self,
        current_ma: float,
        pulse_width_s: float,
        *,
        post_pulse_s: float = 0.0,
        dt_s: float = 1.0e-5,
        initial_h: float = 0.0,
        initial_theta_rad: float = 0.0,
    ) -> SimulationResult:
        """模拟一个矩形脉冲，以及脉冲结束后的零输入恢复区间。"""

        if pulse_width_s < 0:
            raise ValueError("pulse_width_s must be non-negative")
        if post_pulse_s < 0:
            raise ValueError("post_pulse_s must be non-negative")

        def pulse(t_s: float) -> float:
            # 输入开启阶段返回固定电流，超过脉冲宽度后返回零电流。
            return current_ma if t_s < pulse_width_s else 0.0

        return self.simulate(
            duration_s=pulse_width_s + post_pulse_s,
            dt_s=dt_s,
            current=pulse,
            initial_h=initial_h,
            initial_theta_rad=initial_theta_rad,
        )

    def _instantaneous_resistance_ohm(
        self,
        theta_rad: float,
        h: float,
        oscillating: bool,
    ) -> float:
        p = self.parameters
        remaining_fraction = 1.0 - h
        # h=0 时中心电阻为初始高低阻的中点；h=1 时中心电阻
        # 收敛到停振后的中间电阻。
        initial_center = 0.5 * (
            p.initial_high_resistance_ohm + p.initial_low_resistance_ohm
        )
        center = p.stopped_intermediate_resistance_ohm + remaining_fraction * (
            initial_center - p.stopped_intermediate_resistance_ohm
        )
        if not oscillating:
            return center

        # 振荡半幅随 (1-h) 线性缩小，因此 h 增大时电阻摆幅和
        # 电压摆幅会同步衰减。
        initial_half_span = 0.5 * (
            p.initial_high_resistance_ohm - p.initial_low_resistance_ohm
        )
        waveform_value = float(
            np.asarray(self.waveform(np.asarray(theta_rad, dtype=float))).reshape(())
        )
        waveform_value = float(np.clip(waveform_value, -1.0, 1.0))
        return center + remaining_fraction * initial_half_span * waveform_value

    @staticmethod
    def _validate_h(h: float) -> None:
        if not np.isfinite(h) or not 0.0 <= h <= 1.0:
            raise ValueError("h must be a finite value in the interval [0, 1]")


if __name__ == "__main__":
    model = FiniteOscillationNeuron()
    result = model.simulate_pulse(
        current_ma=7.32,
        pulse_width_s=0.050,
        post_pulse_s=1.050,
        dt_s=1.0e-5,
    )
    print(f"frequency: {model.frequency_hz(7.32):.1f} Hz")
    print(f"intrinsic lifetime: {model.intrinsic_lifetime_s(7.32)*1e3:.2f} ms")
    print(f"simulated response: {result.response_duration_s*1e3:.2f} ms")
    print(f"complete phase cycles: {result.complete_spike_count}")
    print(f"final h after recovery: {result.h[-1]:.6f}")
