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
            832.06602917, 804.4646979,  774.88432232, 745.75580475, 717.98294567,
            692.65697961, 669.48470259, 646.7472578,  614.13714173, 549.56170034,
            440.77811209, 325.3447892,  240.93064517, 174.20029318, 118.85148868,
            75.86020225,  45.32835147,  26.82107036,  18.91468012,  21.63910474,
            24.44418158,  26.94064145,  29.17335288,  31.52454615,  34.0745836
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
