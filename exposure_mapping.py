"""Shared output masks and min/max image mapping for pixel arrays."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def select_completion_groups(
    completion_time_map: np.ndarray,
    divisions: int,
) -> list[tuple[int, np.ndarray]]:
    completion_time = np.asarray(completion_time_map, dtype=float)
    if completion_time.ndim != 2:
        raise ValueError("completion_time_map must be two-dimensional")
    if not isinstance(divisions, int) or isinstance(divisions, bool) or not 1 <= divisions <= 100:
        raise ValueError("divisions must be an integer in [1, 100]")

    completed_mask = np.isfinite(completion_time)
    indices = np.flatnonzero(completed_mask)
    if indices.size == 0:
        raise RuntimeError("no detected pixels are available for completion groups")

    times = completion_time.ravel()[indices]
    ordered = indices[np.argsort(times, kind="mergesort")]
    groups = []
    for group_index in range(1, divisions + 1):
        percentage = group_index * 100 // divisions
        count = ordered.size if group_index == divisions else int(
            np.round(ordered.size * group_index / divisions)
        )
        count = max(1, min(count, ordered.size))
        mask = np.zeros(completed_mask.shape, dtype=bool)
        mask.ravel()[ordered[:count]] = True
        groups.append((percentage, mask))
    return groups


def normalize_metric_map(values: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    metric = np.asarray(values, dtype=float)
    if metric.ndim != 2:
        raise ValueError("values must be two-dimensional")
    valid = np.isfinite(metric)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != metric.shape:
            raise ValueError("valid_mask must match values")
        valid &= mask

    result = np.zeros(metric.shape, dtype=np.uint8)
    if not np.any(valid):
        return result
    subset = metric[valid]
    low, high = float(subset.min()), float(subset.max())
    if np.isclose(low, high):
        result[valid] = 255
    else:
        result[valid] = np.clip((subset - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
    return result


def write_completion_metric_images(
    output_dir: Path,
    percentage: int,
    mask: np.ndarray,
    metric_maps: dict[str, np.ndarray],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric_name, values in metric_maps.items():
        image = normalize_metric_map(values, mask)
        path = output_dir / f"{percentage}%-{metric_name}.png"
        if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)):
            raise OSError(f"failed to save exposure metric image: {path}")
