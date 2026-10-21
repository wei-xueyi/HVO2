from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


WIDTH = 180
HEIGHT = 144
MAX_PQ_CODE = 1023
MAX_PQ_LUMINANCE_NITS = 10_000.0

M1 = 2610.0 / 16384.0
M2 = 2523.0 / 32.0
C1 = 3424.0 / 4096.0
C2 = 2413.0 / 128.0
C3 = 2392.0 / 128.0

OUTPUT_DIR = Path(__file__).resolve().parent
PQ_IMAGE_PATH = OUTPUT_DIR / "synthetic_hdr10_pq_10bit.png"
PREVIEW_IMAGE_PATH = OUTPUT_DIR / "synthetic_hdr_preview_8bit.png"
MATRIX_PATH = OUTPUT_DIR / "synthetic_hdr_pq_codes.npy"


def pq_encode_luminance(luminance_nits: np.ndarray) -> np.ndarray:
    """Encode absolute luminance in nits into normalized ST 2084 PQ code values."""
    luminance = np.asarray(luminance_nits, dtype=np.float64)
    normalized_luminance = np.clip(luminance / MAX_PQ_LUMINANCE_NITS, 0.0, 1.0)
    powered = np.power(normalized_luminance, M1)
    encoded = np.power(
        (C1 + C2 * powered) / (1.0 + C3 * powered),
        M2,
    )
    return np.clip(encoded, 0.0, 1.0)


def pq_decode_codes(pq_codes: np.ndarray) -> np.ndarray:
    """Decode 10-bit ST 2084 PQ code values into absolute luminance in nits."""
    codes = np.asarray(pq_codes, dtype=np.float64)
    encoded = np.clip(codes / MAX_PQ_CODE, 0.0, 1.0)
    powered = np.power(encoded, 1.0 / M2)
    numerator = np.maximum(powered - C1, 0.0)
    denominator = C2 - C3 * powered
    normalized_luminance = np.power(numerator / denominator, 1.0 / M1)
    return normalized_luminance * MAX_PQ_LUMINANCE_NITS


def _draw_text_mask(text: str, origin: tuple[int, int], scale: float, thickness: int) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.putText(
        mask,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        255,
        thickness,
        lineType=cv2.LINE_8,
    )
    return mask > 0


def create_hdr10_test_matrix(output_dir: str | Path = OUTPUT_DIR) -> np.ndarray:
    """Create a synthetic 180x144 PQ matrix and save only its 8-bit preview.

    The returned uint16 matrix contains legal 10-bit PQ code values in [0, 1023].
    Optional PQ PNG and NumPy file output is retained below as comments.
    The PQ PNG uses a 16-bit container with sample values in [0, 1023].
    """
    y_grid, x_grid = np.mgrid[0:HEIGHT, 0:WIDTH]
    x_normalized = x_grid / (WIDTH - 1)
    y_normalized = y_grid / (HEIGHT - 1)

    # A smooth diagonal ramp joins the bright source and dark shadow.
    transition = 0.5 * (x_normalized + y_normalized)

    # Use one mirrored Gaussian profile for both corner regions.
    bright_center = (29.0, 24.0)
    dark_center = (WIDTH - 1 - bright_center[0], HEIGHT - 1 - bright_center[1])
    sigma_x, sigma_y = 75.0, 84.0
    bright_falloff = np.exp(
        -(((x_grid - bright_center[0]) / sigma_x) ** 2
          + ((y_grid - bright_center[1]) / sigma_y) ** 2)
    )
    dark_falloff = np.exp(
        -(((x_grid - dark_center[0]) / sigma_x) ** 2
          + ((y_grid - dark_center[1]) / sigma_y) ** 2)
    )

    # Construct the background directly in PQ code space. Both the diagonal
    # ramp and Gaussian contrast are antisymmetric around code value 511.5.
    background_codes_float = (
        511.5
        + 25.0 * (1.0 - 2.0 * transition)
        + 475.0 * (bright_falloff - dark_falloff)
    )
    rounded_codes = np.clip(
        np.floor(background_codes_float + 0.5),
        0,
        MAX_PQ_CODE,
    ).astype(np.uint16)

    # Round only one pixel from each mirrored pair, then define its partner
    # as the exact 10-bit complement to avoid tie-rounding asymmetry.
    y_indices, x_indices = np.indices((HEIGHT, WIDTH))
    flat_indices = y_indices * WIDTH + x_indices
    mirrored_flat_indices = (HEIGHT - 1 - y_indices) * WIDTH + (WIDTH - 1 - x_indices)
    primary_pixels = flat_indices < mirrored_flat_indices
    pq_matrix = np.where(
        primary_pixels,
        rounded_codes,
        MAX_PQ_CODE - rounded_codes[::-1, ::-1],
    ).astype(np.uint16)
    background_pq_matrix = pq_matrix.copy()

    # Shift each letter's center-background code toward neutral PQ code 512.
    letter_a = _draw_text_mask("A", (14, 57), scale=2.0, thickness=5)
    letter_b = _draw_text_mask("B", (77, 87), scale=2.0, thickness=5)
    letter_c = _draw_text_mask("C", (126, 132), scale=2.0, thickness=5)
    for letter_mask, offset_ratio in (
        (letter_a, 0.10),
        (letter_b, 0.10),
        (letter_c, 0.15),
    ):
        coordinates = np.argwhere(letter_mask)
        if coordinates.size == 0:
            raise ValueError("letter mask is empty")
        center_y, center_x = np.rint(coordinates.mean(axis=0)).astype(int)
        center_code = float(background_pq_matrix[center_y, center_x])
        fill_code = np.rint(center_code + offset_ratio * (512.0 - center_code))
        pq_matrix[letter_mask] = np.uint16(np.clip(fill_code, 0, MAX_PQ_CODE))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pq_image_path = output_dir / PQ_IMAGE_PATH.name
    preview_image_path = output_dir / PREVIEW_IMAGE_PATH.name
    matrix_path = output_dir / MATRIX_PATH.name

    # 暂停保存 10-bit PQ 图，保留输出代码供后续恢复。
    # if not cv2.imwrite(str(pq_image_path), pq_matrix):
    #     raise OSError(f"failed to save 10-bit PQ grayscale image: {pq_image_path}")

    preview = np.rint(pq_matrix.astype(np.float32) * (255.0 / MAX_PQ_CODE)).astype(np.uint8)
    if not cv2.imwrite(str(preview_image_path), preview):
        raise OSError(f"failed to save 8-bit preview image: {preview_image_path}")

    # np.save(matrix_path, pq_matrix)
    return pq_matrix


if __name__ == "__main__":
    matrix = create_hdr10_test_matrix()
    decoded_luminance = pq_decode_codes(matrix)
    print(f"Matrix shape: {matrix.shape}")
    print(f"Matrix dtype: {matrix.dtype}")
    print(f"PQ code range: {int(matrix.min())}..{int(matrix.max())}")
    print(f"Distinct PQ codes: {np.unique(matrix).size}")
    print(f"Decoded luminance range: {decoded_luminance.min():.6f}..{decoded_luminance.max():.1f} nits")
    # print(f"Saved 10-bit PQ PNG: {PQ_IMAGE_PATH}")
    print(f"Saved 8-bit preview: {PREVIEW_IMAGE_PATH}")
    # print(f"Saved matrix: {MATRIX_PATH}")
