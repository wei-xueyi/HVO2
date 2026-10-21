from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

# 使用已有的有限振荡模型作为单像素动力学核心，不直接修改原模型文件。
from finite_oscillation_model import FiniteOscillationNeuron, ModelParameters

# Keep existing notebook imports working while intrinsic VO2 lives separately.
from intrinsic_vo2_model import (
    intrinsic_vo2_frequency_map,
    map_intrinsic_vo2_frequency_to_grayscale,
    simulate_intrinsic_vo2_exposure,
)


# 统一的错误码，表示“停振后又发生了二次起振”。
ERROR_CODE_REACTIVATE_AFTER_STOP = "E_REACTIVATE_AFTER_STOP"
# 默认视频采样率：50 fps，和之前的设计一致。
DEFAULT_FPS = 50.0
# 模型内部默认以 1 ms 步长积分；停振观察窗口固定为 3 个积分步长。
DEFAULT_SIMULATION_DT_S = 1.0e-4
STOP_WINDOW_STEPS = 3


class InsufficientCurrentError(RuntimeError):
    """整个输入阵列的电流均低于起振门槛。"""

    def __init__(self) -> None:
        super().__init__("电流太小无法起振")

#复振报错
class PixelReactivationError(RuntimeError):
    """像素在振荡中断后再次起振，包括低电流引起的中断。

    复振检查覆盖整段仿真轨迹；首次饱和停振后仍观察 3 个步长，
    再记录完成时间。
    """

    def __init__(self, pixel_indices: Iterable[tuple[int, int]], count: int) -> None:
        self.pixel_indices = list(pixel_indices)
        self.count = count
        self.error_code = ERROR_CODE_REACTIVATE_AFTER_STOP
        message = (
            f"{self.error_code}: 复振，检测到 {count} 个像素停振后再次起振。"
        )
        super().__init__(message)

#储存每个像素的统计结果
@dataclass(frozen=True)
class PixelResult:
    """单个像素在首次停振后的统计结果。

    这里保存的是该像素最终输出所需要的三类指标：
    - 响应持续时间
    - 脉冲数
    - 脉冲频率
    - 完成输出的时间
    """

    response_duration_s: float
    pulse_count: int
    pulse_frequency_hz: float
    completion_time_s: float

class VideoPixelProcessor:
    """把视频帧转成逐像素电流序列，再交给 H-VO2 模型阵列进行仿真。

    整体流程：
    1. 读取视频并降到灰度
    2. 把灰度值映射成光强和电流
    3. 逐像素建立步进电流信号
    4. 调用有限振荡模型模拟一次单像素响应
    5. 统计首次停振并保存输出结果
    6. 在完成度阈值下生成 33%、66%、100% 的结果图
    """

    def __init__(
        self,
        *,
        fps: float = DEFAULT_FPS,
        current_min_ma: float = 0.0,
        current_max_ma: float = 10.0,
        gain: float = 1.0,
        simulation_dt_s: float = DEFAULT_SIMULATION_DT_S,
        model_parameters: ModelParameters | None = None,
    ) -> None:
        # 视频帧率用于把离散帧索引转换成真实时间。
        self.fps = float(fps)
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        self.frame_dt_s = 1.0 / self.fps

        # 线性映射参数：像素亮度 -> 光强 -> 电流。
        self.current_min_ma = float(current_min_ma)
        self.current_max_ma = float(current_max_ma)
        self.gain = float(gain)

        self.simulation_dt_s = float(simulation_dt_s)
        if not np.isfinite(self.simulation_dt_s) or self.simulation_dt_s <= 0:
            raise ValueError("simulation_dt_s must be a finite positive number")
        self.stop_window_steps = STOP_WINDOW_STEPS

        # 直接复用模型默认参数集合。如果需要进一步细化硬件拟合，
        # 可在外部传 ModelParameters 实例覆盖。
        self.model_parameters = model_parameters or ModelParameters()

    def read_video_grayscale(self, video_path: str | Path, start_frame: int = 0) -> np.ndarray:
        """读取视频并转换成灰度序列，输出形状为 (帧数, 高, 宽)。

        这样后续每个像素都可以按时间轴看成一个电流序列：
        current_array[frame_index, y, x]
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"video file not found: {video_path}")

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"failed to open video: {video_path}")

        frames: list[np.ndarray] = []
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index >= start_frame:
                # OpenCV 读取的是 BGR 图。直接用浮点数计算亮度，避免
                # cvtColor 输出 8 位灰度时的取整，保留通道混合产生的小数层次。
                bgr = frame.astype(np.float32)
                luminance = (
                    0.114 * bgr[:, :, 0]
                    + 0.587 * bgr[:, :, 1]
                    + 0.299 * bgr[:, :, 2]
                )
                frames.append(luminance)
            frame_index += 1
        capture.release()

        if not frames:
            raise ValueError(f"no frames available in video: {video_path}")

        return np.stack(frames, axis=0)

    def pixel_to_current(self, gray_frames: np.ndarray) -> np.ndarray:
        """将像素灰度转成电流阵列。

        将灰度归一化到 0~1 后，线性映射到电流范围：
            current = current_min_ma + (current_max_ma - current_min_ma) * normalized
        最后乘上 gain 做增益放大。
        """
        gray_frames = np.asarray(gray_frames, dtype=np.float32)
        if gray_frames.ndim != 3:
            raise ValueError("gray_frames must be 3D with shape (frames, height, width)")

        normalized = gray_frames / 255.0
        current = self.current_min_ma + (self.current_max_ma - self.current_min_ma) * normalized
        return (current) * self.gain

    def validate_current_array(self, current_array: np.ndarray) -> None:
        """在仿真和出图前，区分电流过小与观察时间不足。"""
        current_array = np.asarray(current_array)
        if current_array.ndim != 3 or current_array.size == 0:
            raise ValueError("current_array must be nonempty with shape (frames, height, width)")
        if not np.all(np.isfinite(current_array)):
            raise ValueError("current_array must contain finite values")
        if np.all(
            np.abs(current_array) < self.model_parameters.minimum_oscillation_current_ma
        ):
            raise InsufficientCurrentError()

    def _make_step_current_function(
        self,
        frame_currents: np.ndarray,
    ) -> tuple[float, callable]:
        """把离散帧电流序列转成时间连续的阶梯电流函数。

        例如：50 fps 下，电流序列 [1, 2, 3, ...] 会被解释为：
        0~20 ms 取 1；20~40 ms 取 2；40~60 ms 取 3。
        这就符合你说的“逐帧阶梯波形”的要求。
        """
        total_duration_s = float(frame_currents.size) * self.frame_dt_s

        def current_function(t_s: float) -> float:
            if t_s < 0.0:
                return float(frame_currents[0])
            if t_s >= total_duration_s:
                return 0.0
            index = int(np.floor(t_s / self.frame_dt_s))
            index = min(max(index, 0), frame_currents.size - 1)
            return float(frame_currents[index])

        return total_duration_s, current_function

    def _extract_first_stop_summary(
        self,
        result: object,
    ) -> tuple[bool, float | None, bool, tuple[int, int] | None]:
        """判断一个像素是否“首次稳定停振”，并检查是否发生二次起振。

        返回值含义：
        (是否完成输出, 完成输出的时间, 是否发生复振, 复振位置)

        判定规则：
        - 只要 h 达到 1 且系统当前不再振荡，就视为停振发生
        - 任意振荡区间之间出现中断后再次起振，均判定为“异常复振”
        - 复振检查覆盖整个轨迹，不限于饱和停振后的 3 步
        - 首次饱和停振后观察 3 步，记录窗口末端作为输出时刻
        """
        h = np.asarray(result.h, dtype=float)
        oscillating = np.asarray(result.oscillating, dtype=bool)
        time_s = np.asarray(result.time_s, dtype=float)

        # 先检查所有振荡区间，避免把低电流间隔计入频率统计。
        # 起振前的等待不算复振；一旦开始振荡，任何中断后的再次起振都报错。
        active_indices = np.flatnonzero(oscillating)
        gaps = np.flatnonzero(np.diff(active_indices) > 1)
        if gaps.size:
            gap_index = int(gaps[0])
            stop_idx = int(active_indices[gap_index]) + 1
            restart_idx = int(active_indices[gap_index + 1])
            return False, None, True, (stop_idx, restart_idx)

        # 找出第一次出现“h>=1 且当前不再振荡”的时间点。
        committed_stop = np.flatnonzero((h >= 1.0 - 1.0e-9) & (~oscillating))
        if committed_stop.size == 0:
            return False, None, False, None

        first_stop_idx = int(committed_stop[0])
        # 窗口末端距首次停振完整经过 stop_window_steps 个仿真步长。
        completion_idx = min(first_stop_idx + self.stop_window_steps, time_s.size - 1)
        completion_time_s = float(time_s[completion_idx])
        return True, completion_time_s, False, None

    def _simulate_single_pixel(
        self,
        current_sequence: np.ndarray,
        *,
        pixel_xy: tuple[int, int] | None = None,
    ) -> tuple[PixelResult | None, bool, list[tuple[int, int]]]:
        """对单个像素执行一次完整仿真并返回该像素的统计结果。

        这是最核心的单像素入口：
        - 输入：当前像素在所有帧上的电流值序列
        - 输出：
          1. PixelResult（若正常完成）
          2. 是否复振异常
          3. 复振像素坐标列表
        """
        model = FiniteOscillationNeuron(parameters=self.model_parameters)
        total_duration_s, current_function = self._make_step_current_function(current_sequence)
        observation_duration_s = self.stop_window_steps * self.simulation_dt_s
        simulation_duration_s = total_duration_s + observation_duration_s

        # Continuous above-threshold input cannot outlast the longest fitted
        # duration among its sampled currents, so avoid simulating the full clip.
        if np.all(
            np.abs(current_sequence)
            >= self.model_parameters.minimum_oscillation_current_ma
        ):
            sampled_currents = np.unique(np.abs(current_sequence))
            maximum_lifetime_s = max(
                processor_lifetime_s
                for processor_lifetime_s in (
                    model.intrinsic_lifetime_s(float(current_ma))
                    for current_ma in sampled_currents
                )
            )
            simulation_duration_s = min(
                simulation_duration_s,
                maximum_lifetime_s + observation_duration_s + self.simulation_dt_s,
            )

        sim_result = model.simulate(
            duration_s=simulation_duration_s,
            dt_s=self.simulation_dt_s,
            current=current_function,
            initial_h=0.0,
            initial_theta_rad=0.0,
        )

        completed, completion_time_s, reactivated, reactivation_index = self._extract_first_stop_summary(sim_result)
        if reactivated:
            if reactivation_index is not None and pixel_xy is not None:
                return None, True, [(pixel_xy[0], pixel_xy[1])]
            return None, True, []

        if not completed or completion_time_s is None:
            return None, False, []

        # 这三类值用作最终图像的灰度映射。
        duration_s = float(sim_result.response_duration_s)
        if duration_s <= 0.0:
            pulse_count = 0
            pulse_frequency_hz = 0.0
        else:
            pulse_count = int(sim_result.complete_spike_count)
            pulse_frequency_hz = float(sim_result.phase_cycle_count / duration_s)

        pixel_result = PixelResult(
            response_duration_s=duration_s,
            pulse_count=pulse_count,
            pulse_frequency_hz=pulse_frequency_hz,
            completion_time_s=float(completion_time_s),
        )
        return pixel_result, False, []

    def process_video(
        self,
        video_path: str | Path,
        *,
        title: str,
        start_frame: int = 0,
        output_root: str | Path | None = None,
        completion_divisions: int = 3,
        metrics: Iterable[str] = ("duration", "pulse_frequency", "pulse_count"),
        show_images: bool = False,
    ) -> Path:
        """处理整段视频，并按指定等分数生成完成度灰度图。

        关键步骤：
        1. 从视频读取灰度帧
        2. 将每个像素的亮度转换成电流序列
        3. 逐像素仿真并记录首次停振完成时间
        4. 检查是否有二次起振异常
        5. 按完成时间先后分组，输出指定等分完成度的图像
        """
        if not isinstance(completion_divisions, int) or isinstance(completion_divisions, bool):
            raise TypeError("completion_divisions must be an integer")
        if not 1 <= completion_divisions <= 100:
            raise ValueError("completion_divisions must be between 1 and 100")
        if isinstance(metrics, str):
            selected_metrics = (metrics,)
        else:
            selected_metrics = tuple(dict.fromkeys(metrics))
        available_metrics = {"duration", "pulse_frequency", "pulse_count"}
        if not selected_metrics:
            raise ValueError("metrics must contain at least one metric")
        unknown_metrics = set(selected_metrics) - available_metrics
        if unknown_metrics:
            raise ValueError(
                f"unsupported metrics: {', '.join(sorted(unknown_metrics))}; "
                f"choose from {', '.join(sorted(available_metrics))}"
            )

        frames = self.read_video_grayscale(video_path, start_frame=start_frame)
        current_array = self.pixel_to_current(frames)
        self.validate_current_array(current_array)
        n_frames, height, width = current_array.shape

        # 三个结果图都用二维数组表示：每个像素都对应一个数值。
        response_map = np.full((height, width), np.nan, dtype=float)
        pulse_count_map = np.full((height, width), np.nan, dtype=float)
        pulse_frequency_map = np.full((height, width), np.nan, dtype=float)
        completion_time_map = np.full((height, width), np.nan, dtype=float)

        reactivated_pixels: list[tuple[int, int]] = []

        for y in range(height):
            for x in range(width):
                current_sequence = current_array[:, y, x]
                pixel_result, reactivated, reactivation_indices = self._simulate_single_pixel(
                    current_sequence,
                    pixel_xy=(y, x),
                )
                if reactivated:
                    reactivated_pixels.extend(reactivation_indices)
                    continue
                if pixel_result is None:
                    continue

                response_map[y, x] = pixel_result.response_duration_s
                pulse_count_map[y, x] = float(pixel_result.pulse_count)
                pulse_frequency_map[y, x] = pixel_result.pulse_frequency_hz
                completion_time_map[y, x] = pixel_result.completion_time_s

        # 若出现复振，则直接抛出错误；这与需求要求一致。
        if reactivated_pixels:
            unique_pixels = sorted(set(reactivated_pixels))
            raise PixelReactivationError(unique_pixels, len(unique_pixels))

        # 生成输出目录：MM-DD-HH-MM-标题-startN
        date_label = datetime.now().strftime("%m-%d-%H-%M")
        safe_title = str(title).strip().replace(" ", "_").replace("/", "-")
        output_dir = Path(output_root) if output_root is not None else Path.cwd()
        folder_name = f"{date_label}-{safe_title}-start{start_frame}"
        output_dir = output_dir / folder_name
        output_dir.mkdir(parents=True, exist_ok=True)

        completed_mask = np.isfinite(completion_time_map)
        completed_flat_indices = np.flatnonzero(completed_mask)
        if completed_flat_indices.size == 0:
            raise RuntimeError("no pixels completed the first stop criterion; cannot build output maps")

        # 这里必须把“像素编号”和“完成时间”分开处理：
        # 1) 先拿到所有已完成像素的扁平索引
        # 2) 再按 completion_time_map.ravel()[flat_index] 的大小排序
        # 3) 最后把排序后的像素编号映射回 2D 坐标
        # 这样 33% / 66% / 100% 的含义就是“按完成时间排序后取前 N% 的像素”，
        # 而不是“按空间上半区/下半区取像素”。
        flat_completion_map = completion_time_map.ravel()
        completed_times = flat_completion_map[completed_flat_indices]
        sorted_order = np.argsort(completed_times, kind="mergesort")
        completed_count_total = completed_flat_indices.size
        all_metric_values = {
            "duration": response_map,
            "pulse_count": pulse_count_map,
            "pulse_frequency": pulse_frequency_map,
        }
        selected_metric_values = {
            name: all_metric_values[name] for name in selected_metrics
        }

        # 按指定等分数生成结果；最后一组明确包含全部已完成像素。
        for division_index in range(1, completion_divisions + 1):
            pct = division_index * 100 // completion_divisions
            if division_index == completion_divisions:
                target_count = completed_count_total
            else:
                target_count = int(
                    np.round(completed_count_total * division_index / completion_divisions)
                )
                target_count = max(1, min(target_count, completed_count_total))

            selected_flat = completed_flat_indices[sorted_order[:target_count]]
            selected_y, selected_x = np.unravel_index(selected_flat, completion_time_map.shape)
            mask = np.zeros_like(completed_mask, dtype=bool)
            mask[selected_y, selected_x] = True

            self._write_metric_images(
                output_dir=output_dir,
                percentage=pct,
                mask=mask,
                metric_values=selected_metric_values,
                show_images=show_images,
            )

        return output_dir

    def _write_metric_images(
        self,
        *,
        output_dir: Path,
        percentage: int,
        mask: np.ndarray,
        metric_values: dict[str, np.ndarray],
        show_images: bool,
    ) -> None:
        """将某一百分比完成度下的三类结果写成灰度图。

        规则：
        - 未完成输出的像素用 0（黑色）填充
        - 已完成输出的像素按其数值在当前有效范围内归一化成 0~255
        - 输出文件名格式：33%-duration.png / 66%-pulse_count.png 等
        """
        for metric_name, values in metric_values.items():
            image = np.zeros(values.shape, dtype=np.uint8)
            valid_mask = mask & np.isfinite(values)
            if np.any(valid_mask):
                valid_values = values[valid_mask]
                min_value = np.min(valid_values)
                max_value = np.max(valid_values)
                if np.isclose(max_value, min_value):
                    normalized = np.zeros_like(valid_values, dtype=np.float32)
                    normalized[:] = 255.0
                else:
                    normalized = (valid_values - min_value) / (max_value - min_value)
                    normalized = normalized * 255.0
                image_values = np.zeros_like(values, dtype=np.float32)
                image_values[valid_mask] = normalized
                image = np.clip(image_values, 0.0, 255.0).astype(np.uint8)

            filename = f"{percentage}%-{metric_name}.png"
            # 这里将单通道图再转成 3 通道，因为 cv2.imwrite 更稳定地保存 .png
            cv2.imwrite(str(output_dir / filename), cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))

            if show_images:
                cv2.imshow(filename, image)

        if show_images:
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def process_video_file(
    video_path: str | Path,
    *,
    title: str,
    start_frame: int = 0,
    output_root: str | Path | None = None,
    fps: float = DEFAULT_FPS,
    current_min_ma: float = 0.0,
    current_max_ma: float = 10.0,
    gain: float = 1.0,
    simulation_dt_s: float = DEFAULT_SIMULATION_DT_S,
    completion_divisions: int = 3,
    metrics: Iterable[str] = ("duration", "pulse_frequency", "pulse_count"),
    show_images: bool = False,
) -> Path:
    """对外暴露的简洁接口：给定视频路径和标题，直接生成结果图目录。

    典型调用方式：
    process_video_file(
        video_path="example.mp4",
        title="demo",
        start_frame=0,
        output_root="./results",
    )
    """
    processor = VideoPixelProcessor(
        fps=fps,
        current_min_ma=current_min_ma,
        current_max_ma=current_max_ma,
        gain=gain,
        simulation_dt_s=simulation_dt_s,
    )
    return processor.process_video(
        video_path=video_path,
        title=title,
        start_frame=start_frame,
        output_root=output_root,
        completion_divisions=completion_divisions,
        metrics=metrics,
        show_images=show_images,
    )


if __name__ == "__main__":
    # 这是一个示例入口：按相对路径寻找当前项目目录中的示例视频。
    # 如果不在这里，就要求调用方显式传入视频路径。
    project_root = Path(__file__).resolve().parent
    sample_video = project_root / "Weizmann_data" / "shahar_side.avi"
    if sample_video.exists():
        output_dir = process_video_file(
            sample_video,
            title="shahar_side_plus732_retest",
            start_frame=0,
            output_root=project_root / "outputs",
            fps=50.0,
            current_min_ma=2.0,
            current_max_ma=12.0,
            gain=1.0,
            simulation_dt_s=1.0e-3,
            completion_divisions=5,
            metrics=("pulse_frequency"), #"duration", "pulse_frequency", "pulse_count"
            show_images=False,
        )
        print(f"Output images saved to: {output_dir}")
    else:
        print("No sample video file was found. Please pass an actual video path to process_video_file().")
