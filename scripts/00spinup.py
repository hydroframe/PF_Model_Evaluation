import os.path
from glob import glob
import numpy as np
from pfspinup import pfio
from pfspinup.common import calculate_surface_storage, calculate_subsurface_storage, calculate_water_table_depth
from pfspinup.pfmetadata import PFMetadata

RUN_DIR = '../pfspinup/data/example_run'
RUN_NAME = 'icom_spinup1'


if __name__ == '__main__':

    metadata = PFMetadata(f'{RUN_DIR}/{RUN_NAME}.out.pfmetadata')

    pressure_files = sorted(glob(f'{RUN_DIR}/*.out.press.*.pfb'))
    saturation_files = sorted(glob(f'{RUN_DIR}/*.out.satur.*.pfb'))

    porosity_file = metadata.config['inputs']['porosity']['data'][0]['file']
    porosity = pfio.pfread(os.path.join(RUN_DIR, porosity_file))
    specific_storage_file = metadata.config['inputs']['specific storage']['data'][0]['file']
    specific_storage = pfio.pfread(os.path.join(RUN_DIR, specific_storage_file))

    # pixel size
    dx = metadata['ComputationalGrid.DX']
    dy = metadata['ComputationalGrid.DY']

    nx = metadata['ComputationalGrid.NX']
    ny = metadata['ComputationalGrid.NY']
    nz = metadata['ComputationalGrid.NZ']

    # Thickness of each layer, bottom to top
    thickness = metadata.nz_list('dzScale') * metadata['ComputationalGrid.DZ']

    nt = len(pressure_files)
    subsurface_storage = np.zeros((nt, nx, ny))
    surface_storage = np.zeros((nt, nx, ny))
    wtd = np.zeros((nt, nx, ny))

    for i, (pressure_file, saturation_file) in enumerate(zip(pressure_files, saturation_files)):

        pressure = pfio.pfread(pressure_file)
        saturation = pfio.pfread(saturation_file)

        # sub-surface storage
        subsurface_storage_i = calculate_subsurface_storage(pressure, saturation, porosity, specific_storage, dx, dy, thickness)
        subsurface_storage_i[subsurface_storage_i == np.min(subsurface_storage_i)] = np.nan
        subsurface_storage[i, ...] = subsurface_storage_i

        # surface storage
        surface_storage_i = calculate_surface_storage(pressure, dx, dy)
        surface_storage_i[surface_storage_i == np.min(surface_storage_i)] = np.nan
        surface_storage[i, ...] = surface_storage_i

        # water table depth
        wtd_i = calculate_water_table_depth(pressure, saturation, thickness)
        wtd[i, ...] = wtd_i

    # read the recharge file
    # this step is used for checking the percentage of subsurface storage relative to the recharge
    # which is useful for spinning up checking.
    pme_file = metadata['Solver.EvapTrans.FileName']
    recharge_layers = pfio.pfread(os.path.join(RUN_DIR, pme_file))

    # recharge = recharge rate at top layer * dx * dy * top layer thickness *
    #            the interval between two consecutive outputs (at steady state)
    # TODO: Get the 100.0 from the metadata file + index difference in successive files
    recharge = recharge_layers[-1, :, :] * dx * dy * thickness[-1] * 100.0
    percent_change = np.zeros((nt-1, nx, ny))
    for i in range(nt-1):
        substorage_change = subsurface_storage[i+1, :, :] - subsurface_storage[i, :, :]
        percent_change_i = abs(substorage_change) / recharge * 100
        percent_change[i, ...] = percent_change_i
