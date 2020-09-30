import os.path
from glob import glob
import numpy as np
from pfspinup import pfio
from pfspinup.common import surface_storage, subsurface_storage, water_table_depth
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

    TotSubstor = np.array([]).reshape(0, nx, ny)
    TotSurfstor = np.array([]).reshape(0, nx, ny)
    TotWtd = np.array([]).reshape(0, nx, ny)
    SubStorMean = [0] * nt
    SurfStorMean = [0] * nt
    wtd_mean = [0] * nt

    for i, (pressure_file, saturation_file) in enumerate(zip(pressure_files, saturation_files)):

        pressure = pfio.pfread(pressure_file)
        saturation = pfio.pfread(saturation_file)

        # sub-surface storage
        SubStor = subsurface_storage(pressure, saturation, porosity, specific_storage, dx, dy, thickness)
        SubStor[SubStor == np.min(SubStor)] = np.nan
        TotSubstor = np.vstack([TotSubstor, SubStor[np.newaxis, ...]])
        SubStorMean[i] = np.nanmean(SubStor)

        # surface storage
        SurfStor = surface_storage(pressure, dx, dy)
        SurfStor[SurfStor == np.min(SurfStor)] = np.nan
        TotSurfstor = np.vstack([TotSurfstor, SurfStor[np.newaxis, ...]])
        SurfStorMean[i] = np.nanmean(SurfStor)

        # water table depth
        wtd = water_table_depth(pressure, saturation, thickness)
        TotWtd = np.vstack([TotWtd, wtd[np.newaxis, ...]])
        wtd_mean[i] = np.nanmean(wtd)

    # read the recharge file
    # this step is used for checking the percentage of subsurface storage relative to the recharge
    # which is useful for spinning up checking.
    pme_file = metadata['Solver.EvapTrans.FileName']
    recharge_layers = pfio.pfread(os.path.join(RUN_DIR, pme_file))

    # recharge = recharge rate at top layer * dx * dy * top layer thickness *
    #            the interval between two consecutive outputs (at steady state)
    # TODO: Get the 100.0 from the metadata file + index difference in successive files
    rech = recharge_layers[-1, :, :] * dx * dy * thickness[-1] * 100.0
    pctchange = [0] * (nt-1)
    Totpctchg_grid = np.array([]).reshape(0, nx, ny)
    for i in range(nt-1):
        SubstorChange = TotSubstor[(i+1), :, :] - TotSubstor[i, :, :]
        pctchange[i] = np.nansum(abs(SubstorChange)) / np.nansum(rech) * 100
        pctchange_grid = abs(SubstorChange) / rech * 100
        Totpctchg_grid = np.vstack([Totpctchg_grid, pctchange_grid[np.newaxis, ...]])
