import os.path
import numpy as np
import pfspinup.pfio as pfio
from pfspinup.common import calculate_overland_flow_grid, calculate_overland_flow


TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def test_overland_flow_grid(metadata, test_data_dir):
    mask = pfio.pfread(os.path.join(TEST_DATA_DIR, 'mask.pfb'))
    dx = metadata['ComputationalGrid.DX']
    dy = metadata['ComputationalGrid.DY']
    nx = metadata['ComputationalGrid.NX']
    ny = metadata['ComputationalGrid.NY']
    slopex = metadata.slope_x()
    slopey = metadata.slope_y()
    mannings = metadata.get_single_domain_value('Mannings')

    pressure_files, index_list, timing_list = metadata.output_files('pressure', ignore_missing=True)
    saturation_files, _, _ = metadata.output_files('saturation', index_list=index_list)
    nt = len(index_list)

    overland_flow = np.zeros((nt, ny, nx))
    for i, (pressure_file, saturation_file) in enumerate(zip(pressure_files, saturation_files)):
        pressure = pfio.pfread(pressure_file)
        saturation = pfio.pfread(saturation_file)

        pressure[mask == 0] = np.nan
        saturation[mask == 0] = np.nan

        overland_flow[i, ...] = calculate_overland_flow_grid(pressure, slopex, slopey, mannings, dx, dy, mask=mask, flow_method='OverlandFlow')

    assert np.allclose(overland_flow, np.load(f'{test_data_dir}/overland_flow_grid.npy'), equal_nan=True)


def test_overland_flow(metadata, test_data_dir):
    mask = pfio.pfread(os.path.join(TEST_DATA_DIR, 'mask.pfb'))
    dx = metadata['ComputationalGrid.DX']
    dy = metadata['ComputationalGrid.DY']
    slopex = metadata.slope_x()
    slopey = metadata.slope_y()
    mannings = metadata.get_single_domain_value('Mannings')

    pressure_files, index_list, timing_list = metadata.output_files('pressure', ignore_missing=True)
    saturation_files, _, _ = metadata.output_files('saturation', index_list=index_list)
    nt = len(index_list)

    overland_flow = np.zeros(nt)
    for i, (pressure_file, saturation_file) in enumerate(zip(pressure_files, saturation_files)):
        pressure = pfio.pfread(pressure_file)
        saturation = pfio.pfread(saturation_file)

        pressure[mask == 0] = np.nan
        saturation[mask == 0] = np.nan

        overland_flow[i] = calculate_overland_flow(pressure, slopex, slopey, mannings, dx, dy, mask=mask, flow_method='OverlandFlow')

    assert(np.allclose(
        overland_flow,
        [
            1169.00366915, 1127.6027474, 1085.62307393, 1045.49776147, 1008.03974685,
            974.37857363,  944.02119589, 914.69246618,  873.87922975,  794.06115505,
            656.39695043,  493.84083632, 355.66981807,  238.63788558,  151.63077188,
            88.81032296,   48.21051155,  26.82130173,   18.91468012,   21.77857951,
            25.952812,     30.68154449,  35.65929305,   41.19038182,   47.19080077
         ]
    ))


def test_overland_flow_kinematic_grid(metadata, test_data_dir):
    mask = pfio.pfread(os.path.join(TEST_DATA_DIR, 'mask.pfb'))
    dx = metadata['ComputationalGrid.DX']
    dy = metadata['ComputationalGrid.DY']
    nx = metadata['ComputationalGrid.NX']
    ny = metadata['ComputationalGrid.NY']
    slopex = metadata.slope_x()
    slopey = metadata.slope_y()
    mannings = metadata.get_single_domain_value('Mannings') * np.ones((ny, nx))

    pressure_files, index_list, timing_list = metadata.output_files('pressure', ignore_missing=True)
    saturation_files, _, _ = metadata.output_files('saturation', index_list=index_list)
    nt = len(index_list)

    overland_flow = np.zeros((nt, ny, nx))

    for i, (pressure_file, saturation_file) in enumerate(zip(pressure_files, saturation_files)):
        pressure = pfio.pfread(pressure_file)
        pressure[mask == 0] = np.nan

        overland_flow[i, ...] = calculate_overland_flow_grid(pressure, slopex, slopey, mannings, dx, dy, mask=mask, flow_method='OverlandKinematic')

    assert np.allclose(overland_flow, np.load(f'{test_data_dir}/overland_flow_kinematic_grid.npy'), equal_nan=True)
