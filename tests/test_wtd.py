import os.path
import numpy as np
import pfspinup.pfio as pfio
from pfspinup.common import water_table_depth


def test_water_table_depth(run_dir, run_name, test_data_dir):

    pressure_file = os.path.join(run_dir, f'{run_name}.out.press.00030.pfb')
    saturation_file = os.path.join(run_dir, f'{run_name}.out.satur.00030.pfb')

    pressure = pfio.pfread(pressure_file)
    saturation = pfio.pfread(saturation_file)

    # thickness of layers, from bottom to top
    thickness = np.array([1000, 100, 50, 25, 10, 5, 1, 0.6, 0.3, 0.1])

    wtd = water_table_depth(pressure, saturation, thickness)
    assert np.allclose(wtd, np.load(f'{test_data_dir}/wtd30.npy'), equal_nan=True)
