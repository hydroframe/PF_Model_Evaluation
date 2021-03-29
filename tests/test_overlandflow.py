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
            6117.29495448, 5984.23850997, 5842.64921322, 5700.70976294, 5563.31285031,
            5435.45154667, 5318.53236169, 5209.34719981, 5072.78125123, 4832.69184322,
            4424.93198645, 3889.77077796, 3305.03738111, 2706.69214599, 2180.25177262,
            1750.6914993,  1438.27595013, 1270.31481713, 1224.03278596, 1307.4280435,
            1384.83212691, 1446.02558485, 1491.18668748, 1525.35330985, 1552.47024012
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
