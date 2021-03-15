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
    nx = metadata['ComputationalGrid.NX']
    ny = metadata['ComputationalGrid.NY']
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
            6766.73834572, 6626.98038291, 6477.57177402, 6328.5727824,  6185.20026083,
            6053.13966686, 5933.42110441, 5821.14253547, 5670.67827593, 5382.77888711,
            4875.2074138,  4203.36770264, 3481.27122382, 2782.88844656, 2222.85805693,
            1780.56437809, 1462.61708404, 1294.92192591, 1256.35663079, 1359.44428856,
            1456.63523202, 1535.90613387, 1596.81943196, 1644.6891018,  1683.5897374
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
