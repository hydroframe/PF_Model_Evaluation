import os.path
import numpy as np
import pfspinup.pfio as pfio
from pfspinup.common import calculate_surface_storage


def test_surface_storage(run_dir, run_name, test_data_dir):

    pressure_file = os.path.join(run_dir, f'{run_name}.out.press.00000.pfb')
    pressure = pfio.pfread(pressure_file)

    dx = dy = 1000

    storage = calculate_surface_storage(pressure, dx, dy)
    assert np.allclose(storage, np.load(f'{test_data_dir}/surface00.npy'))
