import pytest
import numpy as np
import pfspinup.pfio as pfio
from pfspinup.common import calculate_surface_storage


def test_surface_storage(metadata, test_data_dir):
    dx = metadata['ComputationalGrid.DX']
    dy = metadata['ComputationalGrid.DY']
    mask = metadata.input_data('mask')

    pressure_files, index_list, timing_list = metadata.output_files('pressure', ignore_missing=True)
    saturation_files, _, _ = metadata.output_files('saturation', index_list=index_list)
    nt = len(index_list)

    surface_storage = np.zeros((nt,))
    for i, (pressure_file, saturation_file) in enumerate(zip(pressure_files, saturation_files)):
        pressure = pfio.pfread(pressure_file)
        saturation = pfio.pfread(saturation_file)

        pressure[mask == 0] = np.nan
        saturation[mask == 0] = np.nan

        surface_storage[i, ...] = np.sum(
            calculate_surface_storage(pressure, dx, dy, mask=mask),
            axis=(0, 1)
        )

    assert np.allclose(surface_storage, np.load(f'{test_data_dir}/surface_storage.npy'), equal_nan=True)


@pytest.mark.xfail
def test_surface_storage_data_accessor(run, test_data_dir):
    data = run.data_accessor
    nt = len(data.times)

    surface_storage = np.zeros((nt,))
    for i in data.times:
        surface_storage[i, ...] = np.sum(
            data.surface_storage,
            axis=(0, 1)
        )
        data.time += 1

    assert np.allclose(surface_storage, np.load(f'{test_data_dir}/surface_storage.npy'), equal_nan=True)