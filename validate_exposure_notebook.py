"""Run the exposure notebook in a fresh HVO2 kernel, without rewriting its source.

Run using a Python environment containing nbclient and nbformat, for example:
    D:/Anaconda/python.exe validate_exposure_notebook.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel-python', default='D:/Anaconda/envs/HVO2/python.exe')
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    source = project / 'exposure_whole_process.ipynb'
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = project / 'validation'
    destination.mkdir(exist_ok=True)
    notebook = nbformat.read(source, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    notebook.cells.append(nbformat.v4.new_code_cell('''
# Validation-only checks; this cell does not belong to the experiment notebook.
import json
assert HVO2_PHASE_ARRAY.shape == INTRINSIC_VO2_PHASE_ARRAY.shape
assert HVO2_PHASE_ARRAY.shape[:2] == SCENE_PQ_CODES.shape
assert np.all(HVO2_DETECTION.active)
assert np.all(np.isfinite(HVO2_FREQUENCY_MAP_HZ))
assert np.nanmax(HVO2_PULSE_COUNT_MAP) <= PULSE_ACCUMULATOR_MAX
assert np.nanmin(HVO2_PULSE_COUNT_MAP) >= PULSE_ACCUMULATOR_MIN
assert CMOS_GAIN == CMOS_AMPLIFIER_GAIN
for ms in SHUTTER_WINDOWS_MS:
    det = INTRINSIC_VO2_DETECTIONS[ms]
    np.testing.assert_allclose(det.duration_s, ms / 1000, atol=1e-12)
    assert det.active.all()
    counts = INTRINSIC_VO2_PULSE_COUNT_MAPS[ms]
    assert np.isfinite(counts).all() and (counts <= PULSE_ACCUMULATOR_MAX).all()
    assert (counts >= PULSE_ACCUMULATOR_MIN).all()
    frequency = INTRINSIC_VO2_FREQUENCY_MAPS_HZ[ms]
    np.testing.assert_allclose(frequency, counts / det.duration_s)
    charge = CMOS_CHARGE_MAPS[ms]
    assert np.all(charge >= CMOS_MIN_CHARGE_ELECTRONS)
    assert np.all(charge <= CMOS_MAX_CHARGE_ELECTRONS)
    expected_image = np.clip(
        (charge - CMOS_MIN_CHARGE_ELECTRONS)
        * (255.0 / (CMOS_MAX_CHARGE_ELECTRONS - CMOS_MIN_CHARGE_ELECTRONS)),
        0.0,
        255.0,
    ).astype(np.uint8)
    np.testing.assert_array_equal(CMOS_IMAGES[ms], expected_image)
images = sorted(RESULT_BUNDLE_DIR.rglob('*.png'))
assert len(images) == HVO2_COMPLETION_DIVISIONS * len(HVO2_OUTPUT_METRICS) + 7
for path in images:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert image is not None and image.shape == SCENE_PQ_CODES.shape, str(path)
report = dict(
    result_dir=str(RESULT_BUNDLE_DIR), pixels=int(SCENE_PQ_CODES.size),
    phase_shape=list(HVO2_PHASE_ARRAY.shape), cmos_gain=CMOS_GAIN,
    hvo2_duration_ms=[float(np.nanmin(HVO2_DURATION_MAP_S)*1000), float(np.nanmax(HVO2_DURATION_MAP_S)*1000)],
    hvo2_frequency_hz=[float(np.nanmin(HVO2_FREQUENCY_MAP_HZ)), float(np.nanmax(HVO2_FREQUENCY_MAP_HZ))],
    intrinsic={str(ms): dict(
        duration_ms=float(np.mean(INTRINSIC_VO2_DETECTIONS[ms].duration_s)*1000),
        pulses_min=float(np.min(INTRINSIC_VO2_PULSE_COUNT_MAPS[ms])),
        pulses_max=float(np.max(INTRINSIC_VO2_PULSE_COUNT_MAPS[ms])),
        saturated_pixels=int(np.count_nonzero(INTRINSIC_VO2_PULSE_COUNT_MAPS[ms] >= PULSE_ACCUMULATOR_MAX)),
        gray_levels=int(np.unique(normalize_metric_map(INTRINSIC_VO2_FREQUENCY_MAPS_HZ[ms])).size),
    ) for ms in SHUTTER_WINDOWS_MS},
    cmos={str(ms): dict(gray_levels=int(np.unique(CMOS_IMAGES[ms]).size),
                       charge_min=float(np.min(CMOS_CHARGE_MAPS[ms])),
                       charge_max=float(np.max(CMOS_CHARGE_MAPS[ms])),
                       saturated_pixels=int(np.count_nonzero(
                           CMOS_CHARGE_MAPS[ms] >= CMOS_MAX_CHARGE_ELECTRONS)))
          for ms in SHUTTER_WINDOWS_MS},
    result_images=[str(p.relative_to(RESULT_BUNDLE_DIR)) for p in images],
)
(PROJECT_DIR / 'validation' / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
'''))
    with tempfile.TemporaryDirectory(prefix='hvo2-kernel-') as temporary:
        spec_dir = Path(temporary) / 'hvo2-validation'
        spec_dir.mkdir()
        (spec_dir / 'kernel.json').write_text(json.dumps(dict(
            argv=[str(Path(args.kernel_python).resolve()), '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            display_name='HVO2 validation', language='python',
        )), encoding='utf-8')
        manager = KernelManager(kernel_name='hvo2-validation',
                                kernel_spec_manager=KernelSpecManager(kernel_dirs=[temporary]))
        def progress(cell, cell_index, **kwargs):
            print(f"Running cell {cell_index + 1}: {cell.get('id', '')}", flush=True)
        client = NotebookClient(notebook, km=manager, timeout=240,
                                resources={'metadata': {'path': str(project)}},
                                on_cell_execute=progress)
        try:
            with client.setup_kernel(cwd=str(project), cleanup_kc=True):
                client.execute()
        finally:
            nbformat.write(notebook, destination / 'exposure_whole_process.executed.ipynb')
    if hashlib.sha256(source.read_bytes()).hexdigest() != original_hash:
        raise RuntimeError('Source notebook changed during validation; validate the new version separately')
    inline_images = sum('image/png' in output.get('data', {})
                        for c in notebook.cells if c.cell_type == 'code' for output in c.outputs)
    errors = [o for c in notebook.cells if c.cell_type == 'code' for o in c.outputs if o.output_type == 'error']
    assert not errors
    assert inline_images == 17, f'Unexpected inline preview/histogram count: {inline_images}'
    print(f'PASS: {inline_images} inline images; verified output sizes and readout statistics.', flush=True)
    print((destination / 'report.json').read_text(encoding='utf-8'), flush=True)


if __name__ == '__main__':
    main()
