"""Small inline histogram renderer used by the exposure notebook."""

from __future__ import annotations

import cv2
import numpy as np
from IPython.display import Image, display


def display_histogram(
    values: np.ndarray,
    *,
    bins: int,
    title: str,
    x_label: str,
    value_range: tuple[float, float] | None = None,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        print(f"No finite values for histogram: {title}")
        return np.zeros(bins, dtype=np.int64)

    counts, edges = np.histogram(values, bins=bins, range=value_range)
    width, height = 1120, 440
    left, right, top, bottom = 88, 26, 62, 82
    plot_width = width - left - right
    plot_height = height - top - bottom
    plot_bottom = top + plot_height
    max_count = max(int(counts.max()), 1)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    cv2.putText(canvas, title, (left, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (45, 55, 62), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Pixels", (18, top - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (75, 85, 92), 1, cv2.LINE_AA)
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = int(round(plot_bottom - fraction * plot_height))
        tick = int(round(max_count * fraction))
        cv2.line(canvas, (left, y), (width - right, y), (228, 233, 236), 1)
        cv2.putText(canvas, str(tick), (12, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (75, 85, 92), 1, cv2.LINE_AA)

    slot_width = plot_width / bins
    for index, count in enumerate(counts):
        bar_height = int(round(int(count) / max_count * plot_height))
        x0 = int(round(left + index * slot_width + 2))
        x1 = int(round(left + (index + 1) * slot_width - 2))
        y0 = plot_bottom - bar_height
        cv2.rectangle(canvas, (x0, y0), (max(x1, x0 + 1), plot_bottom - 1), (143, 111, 59), thickness=-1)

    cv2.line(canvas, (left, top), (left, plot_bottom), (75, 85, 92), 2)
    cv2.line(canvas, (left, plot_bottom), (width - right, plot_bottom), (75, 85, 92), 2)
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = int(round(left + plot_width * fraction))
        tick = edges[0] + (edges[-1] - edges[0]) * fraction
        cv2.putText(canvas, f"{tick:.4g}", (x - 22, plot_bottom + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (75, 85, 92), 1, cv2.LINE_AA)
    cv2.putText(canvas, x_label, (left + plot_width // 2 - 80, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (55, 65, 72), 1, cv2.LINE_AA)

    success, png_data = cv2.imencode(".png", canvas)
    if not success:
        raise RuntimeError("failed to encode inline histogram")
    display(Image(data=png_data.tobytes()))
    return counts
