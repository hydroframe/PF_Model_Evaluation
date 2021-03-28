import numpy as np
import pfspinup.pfio as pfio
from pfspinup.common import calculate_subsurface_storage


def test_subsurface_storage(metadata, test_data_dir):
    dx = metadata['ComputationalGrid.DX']
    dy = metadata['ComputationalGrid.DY']
    dz = metadata.dz()

    porosity = metadata.input_data('porosity')
    specific_storage = metadata.input_data('specific storage')
    mask = metadata.input_data('mask')

    pressure_files, index_list, timing_list = metadata.output_files('pressure', ignore_missing=True)
    saturation_files, _, _ = metadata.output_files('saturation', index_list=index_list)
    nt = len(index_list)

    subsurface_storage = np.zeros((nt,))
    for i, (pressure_file, saturation_file) in enumerate(zip(pressure_files, saturation_files)):
        pressure = pfio.pfread(pressure_file)
        saturation = pfio.pfread(saturation_file)

        pressure[mask == 0] = np.nan
        saturation[mask == 0] = np.nan

        subsurface_storage[i, ...] = np.sum(
            calculate_subsurface_storage(porosity, pressure, saturation, specific_storage, dx, dy, dz, mask=mask),
            axis=(0, 1, 2)
        )

    assert np.allclose(subsurface_storage, np.load(f'{test_data_dir}/subsurface_storage.npy'), equal_nan=True)
