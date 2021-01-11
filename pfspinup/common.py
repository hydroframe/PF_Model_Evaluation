import numpy as np


def calculate_water_table_depth(pressure, saturation, dz):
    """
    Calculate water table depth from the land surface
    :param pressure: A nz-by-ny-by-nx ndarray of pressure values (bottom layer to top layer)
    :param saturation: A nz-by-ny-by-nx ndarray of saturation values (bottom layer to top layer)
    :param dz: An ndarray of shape (nz,) of thickness values (bottom layer to top layer)
    :return: A ny-by-nx ndarray of water table depth values (measured from the top)
    """
    # Depth of the center of each layer to the top (bottom layer to top layer)
    _depth = (np.cumsum(dz[::-1]) - (dz[::-1]/2))[::-1]
    # Make 3D with shape (nz, 1, 1) to allow subsequent operations
    depth = _depth[:, np.newaxis, np.newaxis]

    def _first_saturated_layer(col):
        # Return the 0-index of the first fully saturated layer of a grid column,
        # measured from the top layer.
        return np.sum(col == 1) - 1

    # Indices of first saturated layer across the grid, measured from the top
    z_indices = np.apply_along_axis(_first_saturated_layer, axis=0, arr=saturation)  # shape (ny, nx)
    # Make 3D with shape (1, ny, nx) to allow subsequent operations
    z_indices = z_indices[np.newaxis, ...]

    saturation_depth = np.take_along_axis(depth, z_indices, axis=0)  # shape (1, ny, nx)
    ponding_depth = np.take_along_axis(pressure, z_indices, axis=0)  # shape (1, ny, nx)
    wtd = saturation_depth - ponding_depth  # shape (1, ny, nx)

    return wtd.squeeze(axis=0)  # shape (ny, nx)


def calculate_subsurface_storage(mask, porosity, pressure, saturation, specific_storage, dx, dy, dz):
    """
    Calculate gridded subsurface storage across several layers.

    For each layer in the subsurface, storage consists of two parts
      - incompressible subsurface storage
        (porosity * saturation * depth of this layer) * dx * dy
      - compressible subsurface storage
        (pressure * saturation * specific storage * depth of this layer) * dx * dy

    :param mask: A nz-by-ny-by-nx ndarray of mask values (bottom layer to top layer)
    :param porosity: A nz-by-ny-by-nx ndarray of porosity values (bottom layer to top layer)
    :param pressure: A nz-by-ny-by-nx ndarray of pressure values (bottom layer to top layer)
    :param saturation: A nz-by-ny-by-nx ndarray of saturation values (bottom layer to top layer)
    :param specific_storage: A nz-by-ny-by-nx ndarray of specific storage values (bottom layer to top layer)
    :param dx: Length of a grid element in the x direction
    :param dy: Length of a grid element in the y direction
    :param dz: Thickness of a grid element in the z direction (bottom layer to top layer)
    :return: A nz-by-ny-by-nx ndarray of subsurface storage values, spanning all layers (bottom to top)
    """
    dz = dz[:, np.newaxis, np.newaxis]  # make 3d so we can broadcast the multiplication below
    incompressible = porosity * saturation * dz * dx * dy
    compressible = pressure * saturation * specific_storage * dz * dx * dy
    total = incompressible + compressible
    total[mask == 0] = 0  # output values for points outside the mask are clamped to 0
    return total


def calculate_surface_storage(mask, pressure, dx, dy):
    """
    Calculate gridded surface storage on the top layer.

    Surface storage is given by:
      Pressure at the top layer * dx * dy (for pressure values > 0)

    :param mask: A nz-by-ny-by-nx ndarray of mask values (bottom layer to top layer)
    :param pressure: A nz-by-ny-by-nx ndarray of pressure values (bottom layer to top layer)
    :param dx: Length of a grid element in the x direction
    :param dy: Length of a grid element in the y direction
    :return: An ny-by-nx ndarray of surface storage values
    """
    surface_mask = mask[-1, ...]
    total = pressure[-1, ...] * dx * dy
    total[total < 0] = 0  # surface storage is 0 when pressure < 0
    total[surface_mask == 0] = 0  # output values for points outside the mask are clamped to 0
    return total


def calculate_evapotranspiration(mask, et, dx, dy, dz):
    """
    Calculate gridded evapotranspiration across several layers.

    :param mask: A nz-by-ny-by-nx ndarray of mask values (bottom layer to top layer)
    :param et: A nz-by-ny-by-nx ndarray of evapotranspiration flux values with units 1/T (bottom layer to top layer)
    :param dx: Length of a grid element in the x direction
    :param dy: Length of a grid element in the y direction
    :param dz: Thickness of a grid element in the z direction (bottom layer to top layer)
    :return: A nz-by-ny-by-nx ndarray of evapotranspiration values (units L^3/T), spanning all layers (bottom to top)
    """
    dz = dz[:, np.newaxis, np.newaxis]  # make 3d so we can broadcast the multiplication below
    total = et * dz * dx * dy
    total[mask == 0] = 0  # output values for points outside the mask are clamped to 0
    return total


def _overland_flow(pressure_top, slopex, slopey, mannings, dx, dy):

    # Calculate fluxes across east and north faces

    # ---------------
    # The x direction
    # ---------------
    qx = -(np.sign(slopex) * (np.abs(slopex) ** 0.5) / mannings) * (pressure_top ** (5 / 3)) * dy

    # Upwinding to get flux across the east face of cells - based on qx[i] if it is positive and qx[i+1] if negative
    qeast = np.maximum(0, qx[:, :-1]) - np.maximum(0, -qx[:, 1:])

    # Add the left boundary - pressures outside domain are 0 so flux across this boundary only occurs when
    # qx[0] is negative
    qeast = np.hstack([-np.maximum(0, -qx[:, 0])[:, np.newaxis], qeast])

    # Add the right boundary - pressures outside domain are 0 so flux across this boundary only occurs when
    # qx[-1] is positive
    qeast = np.hstack([qeast, np.maximum(0, qx[:, -1])[:, np.newaxis]])

    # ---------------
    # The y direction
    # ---------------
    qy = -(np.sign(slopey) * (np.abs(slopey) ** 0.5) / mannings) * (pressure_top ** (5 / 3)) * dx

    # Upwinding to get flux across the north face of cells - based in qy[j] if it is positive and qy[j+1] if negative
    qnorth = np.maximum(0, qy[:-1, :]) - np.maximum(0, -qy[1:, :])

    # Add the top boundary - pressures outside domain are 0 so flux across this boundary only occurs when
    # qy[0] is negative
    qnorth = np.vstack([-np.maximum(0, -qy[0, :]), qnorth])

    # Add the bottom boundary - pressures outside domain are 0 so flux across this boundary only occurs when
    # qy[-1] is positive
    qnorth = np.vstack([qnorth, np.maximum(0, qy[-1, :])])

    return qeast, qnorth


def _overland_flow_kinematic(mask, pressure_top, slopex, slopey, mannings, dx, dy, epsilon):

    # We will be tweaking the slope values for this algorithm, so we make a copy
    slopex = slopex.copy()
    slopey = slopey.copy()

    # We're only interested in the surface mask, as an ny-by-nx array
    mask = mask[-1, ...]

    # Find all patterns of the form
    #  -------
    # | 0 | 1 |
    #  -------
    # and copy the slopex values from the '1' cells to the corresponding '0' cells
    _x, _y = np.where(np.diff(mask, axis=1, append=0) == 1)
    slopex[(_x, _y)] = slopex[(_x, _y + 1)]

    # Find all patterns of the form
    #  ---
    # | 0 |
    # | 1 |
    #  ---
    # and copy the slopey values from the '1' cells to the corresponding '0' cells
    _x, _y = np.where(np.diff(mask, axis=0, append=0) == 1)
    slopey[(_x, _y)] = slopey[(_x + 1, _y)]

    slope = np.maximum(epsilon, np.hypot(slopex, slopey))

    # Upwind pressure - this is for the north and east face of all cells
    # The slopes are calculated across these boundaries so the upper x/y boundaries are included in these
    # calculations. The lower x/y boundaries are added further down as q_x0/q_y0
    pressure_top_padded = np.pad(pressure_top[:, 1:], ((0, 0,), (0, 1)))  # pad right
    pupwindx = np.maximum(0, np.sign(slopex) * pressure_top_padded) + np.maximum(0, -np.sign(slopex) * pressure_top)
    pressure_top_padded = np.pad(pressure_top[1:, :], ((0, 1,), (0, 0)))  # pad bottom
    pupwindy = np.maximum(0, np.sign(slopey) * pressure_top_padded) + np.maximum(0, -np.sign(slopey) * pressure_top)

    flux_factor = np.sqrt(slope) * mannings
    # Flux across the x/y directions
    q_x = -slopex / flux_factor * pupwindx ** (5 / 3) * dy
    q_y = -slopey / flux_factor * pupwindy ** (5 / 3) * dx

    # Fix the lower x boundary
    # Use the slopes of the first column
    q_x0 = -slopex[:, 0] / flux_factor[:, 0] * np.maximum(0, np.sign(slopex[:, 0]) * pressure_top[:, 0]) ** (5 / 3) * dy
    qeast = np.hstack([q_x0[:, np.newaxis], q_x])

    # Fix the lower y boundary
    # Use the slopes of the first row
    q_y0 = -slopey[0, :] / flux_factor[0, :] * np.maximum(0, np.sign(slopey[0, :]) * pressure_top[0, :]) ** (5 / 3) * dx
    qnorth = np.vstack([q_y0, q_y])

    return qeast, qnorth


def calculate_overland_flow(mask, pressure, slopex, slopey, mannings, dx, dy, kinematic=True, epsilon=1e-5):
    """
    Calculate overland flow

    :param mask: A nz-by-ny-by-nx ndarray of mask values (bottom layer to top layer)
    :param pressure: A nz-by-ny-by-nx ndarray of pressure values (bottom layer to top layer)
    :param slopex: ny-by-nx
    :param slopey: ny-by-nx
    :param mannings: scalar value
    :param dx: Length of a grid element in the x direction
    :param dy: Length of a grid element in the y direction
    :param kinematic: Whether to use the 'kinematic' algorithm to calculate overland flow.
    :param epsilon: Minimum slope magnitude for solver. Only applicable if kinematic=True.
        This is set using the Solver.OverlandKinematic.Epsilon key in Parflow.
    :return: A ny-by-nx ndarray of overland flow values
    """
    pressure_top = pressure[-1, ...].copy()
    pressure_top = np.nan_to_num(pressure_top)
    pressure_top[pressure_top < 0] = 0

    if kinematic:
        qeast, qnorth = _overland_flow_kinematic(mask, pressure_top, slopex, slopey, mannings, dx, dy, epsilon)
    else:
        qeast, qnorth = _overland_flow(pressure_top, slopex, slopey, mannings, dx, dy)

    # ---------------
    # Total Outflow
    # ---------------

    # Outflow is a positive qeast[i,j] or qnorth[i,j] or a negative qeast[i,j-1], qnorth[i-1,j]
    outflow = np.maximum(0, qeast[:, 1:]) + np.maximum(0, -qeast[:, :-1]) + \
              np.maximum(0, qnorth[1:, :]) + np.maximum(0, -qnorth[:-1, :])

    # Set the outflow values outside the mask to 0
    outflow[mask[-1, ...] == 0] = 0

    return outflow
