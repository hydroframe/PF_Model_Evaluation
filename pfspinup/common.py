import numpy as np


def water_table_depth(press, satur, thickness):
    # calculate water table depth
    # this water table depth is the depth of water table from the land surface

    # depth_to_center is the depth of the center in each layer to the top
    depth_to_center = (np.cumsum(thickness[::-1]) - (thickness[::-1]/2))[::-1]
    depth = {i: depth_to_center[i] for i in range(len(thickness))}

    satur_where = np.sum(satur == 1, axis=0) - 1    # find the first layer that is saturated from the top
    max_wtd = np.vectorize(depth.get)(satur_where)  # use depth dictionary find the depth of center in this layer to the top
    max_wtd = max_wtd.astype('float')
    m, n = satur_where.shape
    i, j = np.ogrid[:m, :n]
    sel_press = press[satur_where, i, j]            # read the pressure (i.e. ponding depth) at this layer
    wtd = max_wtd - sel_press                       # removing the magnitude of ponding depth to the max_wtd
    
    return wtd


def subsurface_storage(press, satur, porosity, spec_stor, dx, dy, dz):
    """
    Calculate gridded subsurface storage across several layers.

    For each layer in the subsurface, storage consists of two parts
      - incompressible subsurface storage
        (porosity * saturation * depth of this layer) * dx * dy
      - compressible subsurface storage
        (pressure * saturation * specific storage * depth of this layer) * dx * dy

    :param press: A nz-by-nx-by-ny ndarray of pressure values (bottom layer to top layer)
    :param satur: A nz-by-nx-by-ny ndarray of saturation values (bottom layer to top layer)
    :param porosity: A nz-by-nx-by-ny ndarray of porosity values (bottom layer to top layer)
    :param spec_stor: A nz-by-nx-by-ny ndarray of specific storage values (bottom layer to top layer)
    :param dx: Length of a grid element in the x direction
    :param dy: Length of a grid element in the y direction
    :param dz: Thickness of a grid element in the z direction (bottom layer to top layer)
    :return: A nx-by-ny ndarray of subsurface storage values, spanning all layers
    """
    total = np.zeros_like(porosity[0, :, :])
    for i, _dz in enumerate(dz):
        incompressible = porosity[i, :, :] * satur[i, :, :] * _dz * dx * dy
        compressible = press[i, :, :] * satur[i, :, :] * spec_stor[i, :, :] * _dz * dx * dy
        total += incompressible + compressible
    return total


def surface_storage(press, dx, dy):
    """
    Calculate gridded surface storage on the top layer.

    Surface storage is given by:
      Pressure at the top layer * dx * dy

    :param press: A nz-by-nx-by-ny ndarray of pressure values (bottom layer to top layer)
    :param dx: Length of a grid element in the x direction
    :param dy: Length of a grid element in the y direction
    :return: An nx-by-ny ndarray of surface storage values
    """
    return press[-1, :, :] * dx * dy
