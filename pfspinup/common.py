import numpy as np


def calculate_water_table_depth(pressure, saturation, dz):
    """
    Calculate water table depth from the land surface
    :param pressure: A nz-by-nx-by-ny ndarray of pressure values (bottom layer to top layer)
    :param saturation: A nz-by-nx-by-ny ndarray of saturation values (bottom layer to top layer)
    :param dz: An ndarray of shape (nz,) of thickness values (bottom layer to top layer)
    :return: A nx-by-ny ndarray of water table depth values (measured from the top)
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
    z_indices = np.apply_along_axis(_first_saturated_layer, axis=0, arr=saturation)  # shape (nx, ny)
    # Make 3D with shape (1, nx, ny) to allow subsequent operations
    z_indices = z_indices[np.newaxis, ...]

    saturation_depth = np.take_along_axis(depth, z_indices, axis=0)  # shape (1, nx, ny)
    ponding_depth = np.take_along_axis(pressure, z_indices, axis=0)  # shape (1, nx, ny)
    wtd = saturation_depth - ponding_depth  # shape (1, nx, ny)

    return wtd.squeeze(axis=0)  # shape (nx, ny)


def calculate_subsurface_storage(mask, porosity, pressure, saturation, specific_storage, dx, dy, dz):
    """
    Calculate gridded subsurface storage across several layers.

    For each layer in the subsurface, storage consists of two parts
      - incompressible subsurface storage
        (porosity * saturation * depth of this layer) * dx * dy
      - compressible subsurface storage
        (pressure * saturation * specific storage * depth of this layer) * dx * dy

    :param mask: A nz-by-nx-by-ny ndarray of mask values (bottom layer to top layer)
    :param porosity: A nz-by-nx-by-ny ndarray of porosity values (bottom layer to top layer)
    :param pressure: A nz-by-nx-by-ny ndarray of pressure values (bottom layer to top layer)
    :param saturation: A nz-by-nx-by-ny ndarray of saturation values (bottom layer to top layer)
    :param specific_storage: A nz-by-nx-by-ny ndarray of specific storage values (bottom layer to top layer)
    :param dx: Length of a grid element in the x direction
    :param dy: Length of a grid element in the y direction
    :param dz: Thickness of a grid element in the z direction (bottom layer to top layer)
    :return: A nz-by-nx-by-ny ndarray of subsurface storage values, spanning all layers (bottom to top)
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

    :param mask: A nz-by-nx-by-ny ndarray of mask values (bottom layer to top layer)
    :param pressure: A nz-by-nx-by-ny ndarray of pressure values (bottom layer to top layer)
    :param dx: Length of a grid element in the x direction
    :param dy: Length of a grid element in the y direction
    :return: An nx-by-ny ndarray of surface storage values
    """
    surface_mask = mask[-1, ...]
    total = pressure[-1, ...] * dx * dy
    total[total < 0] = 0  # surface storage is 0 when pressure < 0
    total[surface_mask == 0] = 0  # output values for points outside the mask are clamped to 0
    return total


def calculate_evapotranspiration(mask, et, dx, dy, dz):
    """
    Calculate gridded evapotranspiration across several layers.

    :param mask: A nz-by-nx-by-ny ndarray of mask values (bottom layer to top layer)
    :param et: A nz-by-nx-by-ny ndarray of evapotranspiration flux values with units 1/T (bottom layer to top layer)
    :param dx: Length of a grid element in the x direction
    :param dy: Length of a grid element in the y direction
    :param dz: Thickness of a grid element in the z direction (bottom layer to top layer)
    :return: A nz-by-nx-by-ny ndarray of evapotranspiration values (units L^3/T), spanning all layers (bottom to top)
    """
    dz = dz[:, np.newaxis, np.newaxis]  # make 3d so we can broadcast the multiplication below
    total = et * dz * dx * dy
    total[mask == 0] = 0  # output values for points outside the mask are clamped to 0
    return total


def calculate_overland_flow(mask, pressure, slopex, slopey, mannings, dx, dy):
    """
    Calculate overland flow

    This function implements the 'OverlandFlow' algorithm (as opposed to the 'OverlandKinematic' algorithm)

    :param mask: A nz-by-nx-by-ny ndarray of mask values (bottom layer to top layer)
    :param pressure: A nz-by-nx-by-ny ndarray of pressure values (bottom layer to top layer)
    :param slopex: nx-by-ny
    :param slopey: nx-by-ny
    :param mannings: scalar value
    :param dx: Length of a grid element in the x direction
    :param dy: Length of a grid element in the y direction
    :return: A nx-by-ny ndarray of overland flow values
    """
    pressure = pressure[-1, ...].copy()
    pressure[pressure < 0] = 0

    # Calculate fluxes across east and north faces

    # ---------------
    # The x direction
    # ---------------
    qx = -(np.sign(slopex) * (np.abs(slopex) ** 0.5) / mannings) * (pressure ** (5/3)) * dy

    # Upwinding to get flux across the east face of cells - based on qx[i] if it is positive and qx[i+1] if negative
    qeast = np.maximum(0, qx[:-1, :]) - np.maximum(0, -qx[1:, :])

    # Add the left boundary - pressures outside domain are 0 so flux across this boundary only occurs when
    # qx[0] is negative
    qeast = np.vstack([-np.maximum(0, -qx[0, :]), qeast])

    # Add the right boundary - pressures outside domain are 0 so flux across this boundary only occurs when
    # qx[-1] is positive
    qeast = np.vstack([qeast, np.maximum(0, qx[-1, :])])

    # ---------------
    # The y direction
    # ---------------
    qy = -(np.sign(slopey) * (np.abs(slopey) ** 0.5) / mannings) * (pressure ** (5/3)) * dx

    # Upwinding to get flux across the north face of cells - based in qy[j] if it is positive and qy[j+1] if negative
    qnorth = np.maximum(0, qy[:, :-1]) - np.maximum(0, -qy[:, 1:])

    # Add the bottom boundary - pressures outside domain are 0 so flux across this boundary only occurs when
    # qy[0] is negative
    qnorth = np.hstack([-np.maximum(0, -qy[:, 0])[:, np.newaxis], qnorth])

    # Add the right boundary - pressures outside domain are 0 so flux across this boundary only occurs when
    # qy[-1] is positive
    qnorth = np.hstack([qnorth, np.maximum(0, qy[:, -1])[:, np.newaxis]])

    # ---------------
    # Total Outflow
    # ---------------

    # Outflow is a positive qeast[i,j] or qnorth[i,j] or a negative qeast[i-1,j], qnorth[i,j-1]
    outflow = np.maximum(0, qeast[1:, :]) + np.maximum(0, -qeast[:-1, :]) + \
              np.maximum(0, qnorth[:, 1:]) + np.maximum(0, -qnorth[:, :-1])

    return outflow


