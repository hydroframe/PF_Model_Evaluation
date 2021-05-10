![CI](https://github.com/hydroframe/PF_Model_Evaluation/workflows/CI/badge.svg?branch=master)

# PF Model Evaluation
This repository is now superseded by the [hydrology module](https://github.com/parflow/parflow/blob/master/pftools/python/parflow/tools/hydrology.py) in Pftools (version 1.2.0 and above).

All of the several `calculate_*` functions previously available from `pfspinup.common`

```
from pfspinup.common import calculate_surface_storage, calculate_subsurface_storage, calculate_water_table_depth, \
    calculate_evapotranspiration, calculate_overland_flow_grid
```

can be imported from `parflow.tools.hydrology` instead:

```
from parflow.tools.hydrology import calculate_surface_storage, calculate_subsurface_storage, calculate_water_table_depth, \
    calculate_evapotranspiration, calculate_overland_flow_grid
```

However, you will almost certainly want to use the `DataAccessor` class for a `Run` object:

```
from parflow import Run

# Create a Run object from the .pfidb file
run = Run.from_definition('/path/to/pfidb/file')
# Get the DataAccessor object corresponding to the Run object
data = run.data

# iterate through the timesteps of the DataAccessor object
# i goes from 0 to n_timesteps - 1
for i in data.times:

    # nz-by-ny-by-nx array of ET values (bottom to top layer)
    print(data.et)
    
    # ny-by-nx array of overland flow values - 'OverlandKinematic' flow method
    print(data.overland_flow_grid())

    # ny-by-nx array of overland flow values - 'OverlandFlow' flow method
    print(data.overland_flow_grid(flow_method='OverlandFlow'))

    # Total outflow for the domain (scalar value) - 'OverlandKinematic' flow method
    print(data.overland_flow())
        
    # Total outflow for the domain (scalar value) - 'OverlandFlow' flow method
    print(data.overland_flow(flow_method='OverlandFlow'))
     
    # nz-by-ny-by-nx array of subsurface storage values (bottom to top layer)
    print(data.subsurface_storage)

    # ny-by-nx array of surface storage values
    print(data.surface_storage)

    # ny-by-nx array of water table depth values
    print(data.wtd)
        
```

The `DataAccessor` object thus obtained also has lots of other useful properties, like:

```
data.slope_x
data.slope_y
data.mask
data.mannings
data.pressure
data.saturation
data.computed_porosity
data.computed_permeability_x
data.computed_permeability_y
data.computed_permeability_zs
```