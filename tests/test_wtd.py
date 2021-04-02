import pytest
import numpy as np
import pfspinup.pfio as pfio
from pfspinup.common import calculate_water_table_depth


def test_water_table_depth(metadata, test_data_dir):
    dz = metadata.dz()
    nx = metadata['ComputationalGrid.NX']
    ny = metadata['ComputationalGrid.NY']
    mask = metadata.input_data('mask')

    pressure_files, index_list, timing_list = metadata.output_files('pressure', ignore_missing=True)
    saturation_files, _, _ = metadata.output_files('saturation', index_list=index_list)
    nt = len(index_list)

    wtd = np.zeros((nt, ny, nx))
    for i, (pressure_file, saturation_file) in enumerate(zip(pressure_files, saturation_files)):
        pressure = pfio.pfread(pressure_file)
        saturation = pfio.pfread(saturation_file)

        pressure[mask == 0] = np.nan
        saturation[mask == 0] = np.nan

        wtd[i, ...] = calculate_water_table_depth(pressure, saturation, dz)

    assert np.allclose(wtd, np.load(f'{test_data_dir}/wtd.npy'), equal_nan=True)


@pytest.mark.xfail
def test_water_table_depth_data_accessor(run, test_data_dir):
    data = run.data_accessor
    nt = len(data.times)
    ny = run.ComputationalGrid.NY
    nx = run.ComputationalGrid.NX

    wtd = np.zeros((nt, ny, nx))
    for i in data.times:
        wtd[i, ...] = data.wtd
        data.time += 1

    assert np.allclose(wtd, np.load(f'{test_data_dir}/wtd.npy'), equal_nan=True)