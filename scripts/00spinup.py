import numpy as np
from pfspinup import pfio
from pfspinup.common import calculate_surface_storage, calculate_subsurface_storage, calculate_water_table_depth, \
    calculate_evapotranspiration, calculate_overland_flow
from pfspinup.pfmetadata import PFMetadata

RUN_DIR = '../pfspinup/data/example_run'
RUN_NAME = 'LW_CLM_Ex4'

metadata = PFMetadata(f'{RUN_DIR}/{RUN_NAME}.out.pfmetadata')

# ------------------------------------------
# Get relevant information from the metadata
# ------------------------------------------

# Resolution
dx = metadata['ComputationalGrid.DX']
dy = metadata['ComputationalGrid.DY']
# Thickness of each layer, bottom to top
dz = metadata.dz()

# Extent
nx = metadata['ComputationalGrid.NX']
ny = metadata['ComputationalGrid.NY']
nz = metadata['ComputationalGrid.NZ']

# ------------------------------------------
# Get numpy arrays from metadata
# ------------------------------------------

# ------------------------------------------
# Time-invariant values
# ------------------------------------------
porosity = metadata.input_data('porosity')
specific_storage = metadata.input_data('specific storage')
mask = metadata.input_data('mask')
# Note that only time-invariant ET flux values are supported for now
et_flux_values = metadata.et_flux()  # shape (nz, nx, ny) - units 1/T.

slopex = metadata.slope_x()  # shape (nx, ny)
slopey = metadata.slope_y()  # shape (nx, ny)
mannings = metadata.get_single_domain_value('Mannings')  # scalar value

# ------------------------------------------
# Time-variant values
# ------------------------------------------
# Get as many pressure files as are available, while also getting their corresponding index IDs and timing info
pressure_files, index_list, timing_list = metadata.output_files('pressure', ignore_missing=True)

# By explicitly passing in the index_list in the call below, we insist that all saturation files corresponding
# to the pressure files be present.
saturation_files, _, _ = metadata.output_files('saturation', index_list=index_list)
# no. of time steps
nt = len(index_list)

# ------------------------------------------
# Initialization
# ------------------------------------------
# Arrays for total values (across all layers), with time as the first axis
subsurface_storage = np.zeros((nt,))
surface_storage = np.zeros((nt,))
wtd = np.zeros((nt, nx, ny))
et = np.zeros((nt,))
overland_flow = np.zeros((nt, nx, ny))

# ------------------------------------------
# Loop through time steps
# ------------------------------------------
for i, (pressure_file, saturation_file) in enumerate(zip(pressure_files, saturation_files)):
    pressure = pfio.pfread(pressure_file)
    saturation = pfio.pfread(saturation_file)

    # We set pressure/saturation values outside our mask to np.nan to indicate missing values.
    # Individual functions that deal with these arrays are written to handle np.nan values properly.
    pressure[mask == 0] = np.nan
    saturation[mask == 0] = np.nan

    # total subsurface storage for this time step is the summation of substorage surface across all x/y/z slices
    subsurface_storage[i, ...] = np.sum(
        calculate_subsurface_storage(mask, porosity, pressure, saturation, specific_storage, dx, dy, dz),
        axis=(0, 1, 2)
    )

    # total surface storage for this time step is the summation of substorage surface across all x/y slices
    surface_storage[i, ...] = np.sum(
        calculate_surface_storage(mask, pressure, dx, dy),
        axis=(0, 1)
    )

    wtd[i, ...] = calculate_water_table_depth(pressure, saturation, dz)

    if et_flux_values is not None:
        # total ET for this time step is the summation of ET values across all x/y/z slices
        et[i, ...] = np.sum(
            calculate_evapotranspiration(mask, et_flux_values, dx, dy, dz),
            axis=(0, 1, 2)
        )

    overland_flow[i, ...] = calculate_overland_flow(mask, pressure, slopex, slopey, mannings, dx, dy)
