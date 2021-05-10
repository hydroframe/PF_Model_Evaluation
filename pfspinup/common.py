import warnings
from parflow.tools.hydrology import calculate_water_table_depth, calculate_subsurface_storage, \
    calculate_surface_storage, calculate_evapotranspiration, calculate_overland_flow_grid, calculate_overland_flow

warnings.warn('The pfspinup package is no longer maintained. '
              'Please use hydrology functions directly from parflow.tools.hydrology')
