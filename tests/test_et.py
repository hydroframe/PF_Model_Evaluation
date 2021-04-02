import pytest
import numpy as np
import pfspinup.pfio as pfio
from pfspinup.common import calculate_evapotranspiration


def test_et(run, metadata, test_data_dir):
    assert metadata['Solver.EvapTransFile'] != metadata['Solver.EvapTransFileTransient'], \
        'Only one of Solver.EvapTrans.FileName, Solver.EvapTransFileTransient can be set in order to calculate evapotranspiration'

    dx = metadata['ComputationalGrid.DX']
    dy = metadata['ComputationalGrid.DY']
    dz = metadata.dz()
    mask = metadata.input_data('mask')

    data = run.data_accessor
    nt = len(data.times)
    nz = metadata['ComputationalGrid.NZ']
    ny = metadata['ComputationalGrid.NY']
    nx = metadata['ComputationalGrid.NX']
    filename = metadata['Solver.EvapTrans.FileName']

    evapotranspiration = np.zeros((nt, nz, ny, nx))
    for i in data.times:
        if metadata['Solver.EvapTransFile']:
            et = pfio.pfread(filename)
        else:
            et = pfio.pfread(f'{filename}.{i:0>5}.pfb')

        evapotranspiration[i, ...] = calculate_evapotranspiration(et, dx, dy, dz, mask=mask)

    assert np.allclose(evapotranspiration, np.load(f'{test_data_dir}/evapotranspiration.npy'), equal_nan=True)


@pytest.mark.xfail
def test_et_data_accessor(run, test_data_dir):
    data = run.data_accessor
    nt = len(data.times)
    nz = run.ComputationalGrid.NZ
    ny = run.ComputationalGrid.NY
    nx = run.ComputationalGrid.NX

    evapotranspiration = np.zeros((nt, nz, ny, nx))
    for i in data.times:
        evapotranspiration[i, ...] = data.et
        data.time += 1

    assert np.allclose(evapotranspiration, np.load(f'{test_data_dir}/evapotranspiration.npy'), equal_nan=True)