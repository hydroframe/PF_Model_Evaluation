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


# WTD calculation checks for a single grid column
# All array values are bottom -> top layer
_dz = np.array([100, 1, 0.6, 0.3, 0.1])  # total thickness = 102, depth of last layer = 52


def test_water_table_depth_0():
    # water table depth below bottom domain
    pressure = np.array([-52, -53, -53.5, -53.6, -53.7])
    saturation = np.array([0.1, 0.3, 0.2, 0.1, 0.6])
    wtd = calculate_water_table_depth(pressure, saturation, _dz).item()
    assert np.allclose(wtd, 102)


def test_water_table_depth_1():
    # water table depth 1m above surface
    pressure = np.array([53, 2.5, 1.7, 1.25, 1.05])
    saturation = np.array([1, 1, 1, 1, 1])
    wtd = calculate_water_table_depth(pressure, saturation, _dz).item()
    assert np.allclose(wtd, 0)


def test_water_table_depth_2():
    # water table depth within the domain and unsaturated above
    # pressure[1:] are NOT correct values, but you'd have something<0
    saturation = np.array([1, 0.7, 0.2, 0.1, 0.1])
    pressure = np.array([50.3, -0.1, -0.5, -0.7, -1])
    wtd = calculate_water_table_depth(pressure, saturation, _dz).item()
    assert np.allclose(wtd, 1.7)


def test_water_table_depth_3():
    # water table depth within the domain and above it a layer of saturated soil (to ignore)
    pressure = np.array([50.3, -0.1, 0.7, 0.25, 0.05])
    saturation = np.array([1, 0.7, 0.8, 1, 1])
    wtd = calculate_water_table_depth(pressure, saturation, _dz).item()
    assert np.allclose(wtd, 1.7)
