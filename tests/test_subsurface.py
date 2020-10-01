import os.path
import numpy as np
import pfspinup.pfio as pfio
from pfspinup.common import calculate_subsurface_storage


def test_subsurface_storage(run_dir, run_name, test_data_dir):

    pressure_file = os.path.join(run_dir, f'{run_name}.out.press.00000.pfb')
    saturation_file = os.path.join(run_dir, f'{run_name}.out.satur.00000.pfb')

    pressure = pfio.pfread(pressure_file)
    saturation = pfio.pfread(saturation_file)
    porosity = pfio.pfread(os.path.join(run_dir, f'{run_name}.out.porosity.pfb'))
    specific_storage = pfio.pfread(os.path.join(run_dir, f'{run_name}.out.specific_storage.pfb'))

    dx = dy = 1000
    # thickness of layers, frop bottom to top
    thickness = np.array([1000, 100, 50, 25, 10, 5, 1, 0.6, 0.3, 0.1])

    storage = calculate_subsurface_storage(pressure, saturation, porosity, specific_storage, dx, dy, thickness)
    assert np.allclose(storage, np.load(f'{test_data_dir}/subsurface00.npy'))
