# %%
import os.path
from glob import glob
import numpy as np
from pfspinup import pfio
from pfspinup.common import calculate_surface_storage, calculate_subsurface_storage, calculate_water_table_depth
from pfspinup.pfmetadata import PFMetadata

# %%
RUN_DIR = 'pfspinup/data/example_run'
RUN_NAME = 'icom_spinup1'
metadata = PFMetadata(f'{RUN_DIR}/{RUN_NAME}.out.pfmetadata')

# %%
# Get the grid information from the metadata
# Resolultion
dx = metadata['ComputationalGrid.DX']
dy = metadata['ComputationalGrid.DY']

# Extent
nx = metadata['ComputationalGrid.NX']
ny = metadata['ComputationalGrid.NY']
nz = metadata['ComputationalGrid.NZ']

# Thickness of each layer, bottom to top
thickness = metadata.nz_list('dzScale') * metadata['ComputationalGrid.DZ']

#Timing
dump_interval = float(metadata.config['inputs']['configuration']['data']['TimingInfo.DumpInterval'])
dt = float(metadata.config['inputs']['configuration']['data']['TimingInfo.BaseUnit'])
start_count = int(metadata.config['inputs']['configuration']['data']['TimingInfo.StartCount'])
start_time = float(metadata.config['inputs']['configuration']['data']['TimingInfo.StartTime'])

# %%
# Read in the non-time varying values
# Read in porosity and specific storage
porosity_file = metadata.config['inputs']['porosity']['data'][0]['file']
porosity = pfio.pfread(os.path.join(RUN_DIR, porosity_file))

specific_storage_file = metadata.config['inputs']['specific storage']['data'][0]['file']
specific_storage = pfio.pfread(os.path.join(RUN_DIR, specific_storage_file))

# Need to upload this file
#mask = pfio.pfread(os.path.join(RUN_DIR, '.out.mask.pfb'))
#icom_spinup1.out.mask.pfb
# %%
# Make a list of output file numbers based on the pressure files in the directory
pressure_files = sorted(glob(f'{RUN_DIR}/*.out.press.*.pfb'))
file_list = [int(str.split(file, '.')[-2]) for file in pressure_files]

#alternatively come up with a list of file numbers
file_list = range(0,31,10)
file_list = [10,30]

nfile = len(file_list)

# Calculate the output times for your file lists 
# I think we could make this a function 
out_times = [(file-start_count) * dt *
             dump_interval + start_time for file in file_list]


# %%
# Calcualte 
# make empty arrays
subsurface_storage = np.zeros((nfile, nx, ny))
surface_storage = np.zeros((nfile, nx, ny))
wtd = np.zeros((nfile, nx, ny))

# What if you didn't have the pressure file and saturation files for the same timesteps? This could cause mismatch
#for i, (pressure_file, saturation_file) in enumerate(zip(pressure_files, saturation_files)):

#for i, in range(nfile):
for i, _file_list in enumerate(file_list):
    print(i, _file_list)
    #read in pressure file
    pressure_file = "%s.out.press.%05d.pfb"  % (RUN_NAME, _file_list)
    pressure = pfio.pfread(os.path.join(RUN_DIR, pressure_file))
    print(pressure_file, "read")

    # Read in saturation file --NEEDS to put up an error if  file not found!
    saturation_file="%s.out.satur.%05d.pfb" % (RUN_NAME, _file_list)
    saturation=pfio.pfread(os.path.join(RUN_DIR, saturation_file))
    print(saturation_file, "read")

    # sub-surface storage  - mask should be an input to this
    # Should use dz not thickness 
    subsurface_storage_i = calculate_subsurface_storage(
            pressure, saturation, porosity, specific_storage, dx, dy, thickness)
    # Can be removed once mask is added to function
    subsurface_storage_i[subsurface_storage_i ==
                         np.min(subsurface_storage_i)] = np.nan
    subsurface_storage[i, ...] = subsurface_storage_i
    
    # surface storage
    # should only be when pressure is >0
    surface_storage_i = calculate_surface_storage(pressure, dx, dy)
    surface_storage_i[surface_storage_i ==
            np.min(surface_storage_i)] = np.nan
    surface_storage[i, ...] = surface_storage_i

    # water table depth
    #mn ij part of  the function  seems unnecessary 
    wtd_i = calculate_water_table_depth(pressure, saturation, thickness)
    wtd[i, ...] = wtd_i

    # read the recharge file
    # this step is used for checking the percentage of subsurface storage relative to the recharge
    # which is useful for spinning up checking.
    # Need to think about how to sum this when you have postive fluxes from the bottom
    # Function EvapTrans_sum
    pme_file = metadata['Solver.EvapTrans.FileName']
    recharge_layers = pfio.pfread(os.path.join(RUN_DIR, pme_file))
    #confirm if this is a 2D or 3D file

    # Functions we want
    # - Total Subsurface storage - 1 value domain total of surface storage [l3]
    # - Total Surface storage - 1 value domain total of surface storage [L3]
    # - EvapTrans - 2D ouput that sums recharge in every column sum((value*dx*dy*dz), across all layers) [l3/T]
    # - EvapTrans Total - 1 value domain total of the previous one
    # - Overland flow sum - save for next time 
    # - Boundary fluxes using patch values -- lower priority


    # recharge = recharge rate at top layer * dx * dy * top layer thickness *
    #            the interval between two consecutive outputs (at steady state)
    # TODO: Get the 100.0 from the metadata file + index difference in successive files
    recharge = recharge_layers[-1, :, :] * dx * dy * thickness[-1] * 100.0
    percent_change = np.zeros((nt-1, nx, ny))
    for i in range(nt-1):
        substorage_change = subsurface_storage[i+1, :, :] - subsurface_storage[i, :, :]
        percent_change_i = abs(substorage_change) / recharge * 100
        percent_change[i, ...] = percent_change_i


