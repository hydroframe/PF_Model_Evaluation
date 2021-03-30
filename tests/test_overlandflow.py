import os.path
import numpy as np
import pfspinup.pfio as pfio
from pfspinup.common import calculate_overland_flow_grid, calculate_overland_flow


TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def test_overland_flow_grid(metadata, test_data_dir):
    mask = metadata.mask
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
    mask = metadata.mask
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

    print('---------')
    print(overland_flow)
    print('----------')

    assert(np.allclose(
        overland_flow,
        [
            26115.33843648, 25592.72853644, 25055.0265084,  24529.07250998, 24017.55359466,
            23524.21743305, 23041.32461742, 22551.64303395, 21908.1469928,  20803.21280048,
            18987.50872112, 16599.32430781, 13966.92527992, 11469.4984031,  9402.580625,
            7825.23226117,  6711.42010401,  6056.84280502,  5818.99682113,  6065.77226212,
            6338.3873967,   6598.95402581,  6848.57274397,  7104.98202936,  7342.22940416
        ]
    ))


def test_overland_flow_kinematic_grid(metadata, test_data_dir):
    mask = metadata.mask
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
