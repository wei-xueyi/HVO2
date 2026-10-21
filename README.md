# HVO₂ HDR 图像识别实验

主线是利用 HVO₂ 的有限振荡与提前停振特性研究 HDR 图像识别。视频是模拟时变光照的一种输入方式。

## 当前硬件映射架构

Notebook 的板块按信号经过的硬件排列，所有实验参数放在 0.1，全局板块说明位于对应代码单元之前：

| 序号 | 模块 | 输入与输出 |
| --- | --- | --- |
| 0.1 | 全局参数 | 输入来源、起始帧、三档共用快门、积分步长、探测器确认时间、模型参数、增益、CMOS 电荷上下限及映射设置 |
| 0.2 | 合成 HDR 输入 | 生成 PQ 场景、预览图及 Notebook 直方图；不属于硬件 |
| 1 | 快门 | 参数区配置的同一组三档曝光窗口作用于 HVO₂、本征 VO₂ 与 CMOS |
| 2 | 光电探测器阵列（photodetector array） | 裁切场景，映射亮度码到光电流，产生时间分辨电流矩阵 |
| 3.1 | HVO₂ 阵列 | 电流驱动有限振荡，输出相位矩阵；隐变量 `h` 只在阵列仿真内部使用 |
| 3.2 | 本征 VO₂ 阵列 | 共用最长相位轨迹，再按各快门截取；不带 `h`，不会自行停振 |
| 4 | 振荡检测器阵列 | 按参数区检测窗口确认开始与结束，输出逐像素有效时长 |
| 5.1 | 脉冲累积器 | 检测有效区间内累计分数周期，按配置下限和上限饱和 |
| 5.2 | CMOS | 对同一光电流积分并裁剪到 0.1 参数区配置的 CMOS 电荷上下限，使用固定单一增益 |
| 6.1 | 脉冲映射 | 脉冲数除以有效检测时长，保存完整频率图和 HVO₂ 完成度分组图，并显示直方图 |
| 6.2 | 电荷映射 | 按 0.1 参数区配置的固定电荷范围映射到 8-bit，Notebook 展示直方图 |

快门时间、起始帧和检测器窗口以 Notebook 参数区为准；积分步长为 0.1 ms。时间轴显式加入快门边沿、输入视频帧边界和检测器确认边界，避免事件落在离散步长之间时造成累计误差。

## 当前文件职责

| 文件 | 职责 |
| --- | --- |
| `finite_oscillation_model.py` | HVO₂ 单器件动力学、频率拟合、有限寿命、恢复与相位统计 |
| `intrinsic_vo2_model.py` | 本征 VO₂ 的拟合频率图、整数脉冲累计与截顶、频率图灰度映射 |
| `video_process.py` | HVO₂ 像素仿真、视频输入、电流映射、异常检测及脚本入口 |
| `HDR_create.py` | 合成 HDR 输入与 PQ 编解码，默认仅保存 8-bit 预览 |
| `exposure_whole_process.ipynb` | 实验参数、流程组织、三类曝光比较和直方图展示 |
| `shutter.py` | 共用快门窗口及输入帧裁切 |
| `photodetector_model.py` | 场景到电流映射和经过快门的电流时间轨迹 |
| `phase_array.py` | HVO₂、本征 VO₂ 的相位阵列演化 |
| `oscillation_detector.py` | 从相位变化提取逐像素检测区间 |
| `array_readout.py`、`readout_modules.py` | 窗口读出、脉冲累积与 CMOS 电荷积分 |
| `exposure_mapping.py`、`notebook_viz.py` | 映射、完成度分组、图像和 Notebook 直方图 |
| `validate_exposure_notebook.py` | 不改写实验 Notebook 的干净内核全流程验证 |
| `test_intrinsic_vo2_exposure.py` | 本征 VO₂ 及 Notebook 计数读出的回归测试 |
| `PROJECT_CONTEXT.md` | 研究目标、用户确认的假设、读出电路调研和变更记录 |

`video_process.py` 仍重新导出本征 VO₂ 的三个旧函数名，现有 Notebook 无需修改导入。新代码可以直接从 `intrinsic_vo2_model` 导入。

## 数据与输出

- `duration-inputcurrent.csv`、`资源 6.svg`：持续时间拟合依据。
- `Weizmann_data/`：光照输入视频素材。
- `synthetic_hdr_preview_8bit.png`：合成 HDR 输入预览；完整矩阵在内存中传递。
- `outputs/<时间>-HDR/`：分别保存 `HVO2-output`、`intrinsic-VO2-output` 和 `traditional-output` 的结果图。
- 统计直方图仅在 Notebook 展示。

## 读出定义与 CMOS 校准结果

1. 拟合振荡频率由输入电流决定，是模型内部量。
2. HVO₂ 和本征 VO₂ 的当前读出频率均为按累积器配置范围裁剪后的脉冲数除以检测器有效时长；本征 VO₂ 检测时长等于快门时长，HVO₂ 时长由有限振荡轨迹决定。
4. 频率灰度映射按各图自身的最小值与最大值归一化；不能凭灰度级数直接认定器件物理动态范围。

CMOS 补偿系数和电荷下限、上限直接放在 Notebook 0.1 参数区，不再读取旧转换参考式或按快门计算增益。5.2 按配置边界裁剪电荷，6.2 按相同边界映射到 8-bit，并保留重构前普通曝光的向下取整量化方式。

## 后续整理顺序

后续整理优先继续拆分仍留在 Notebook 的图像输出逻辑；实验参数和三档输出已经集中在硬件流程中。

## 验证

在 HVO2 环境、此目录下运行：

```powershell
python -B -m unittest -v test_intrinsic_vo2_exposure
```

运行全尺寸 notebook：

```powershell
python validate_exposure_notebook.py
```

验证器使用临时 HVO2 内核从头执行，将带输出副本与统计写入 `validation/`，不把执行输出写回实验 Notebook。

2026-10-21 最新全尺寸 HDR 验证：25,920 像素，输入相位矩阵 `(144, 180, 511)`，Notebook 执行通过，显示 17 张预览/直方图图片并生成 11 张硬件结果图。当前快门为 49.5/50/50.5 ms，累积器范围为 1065–1080；按固定 CMOS 增益，三档电荷均超过 12,000 e⁻ 并饱和为白图，详情见 `validation/report.json`。
