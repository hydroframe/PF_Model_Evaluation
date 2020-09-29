import numpy as np
from parflowio.pyParflowio import PFData


def pfread(pfbfile):
    """
    Read a pfb file and return data as an ndarray
    :param pfbfile: path to pfb file
    :return: An ndarray of ndim=3

    TODO: parflowio seems to read arrays such that the rows (i.e. axis=1) are reversed w.r.t what pfio gives us
    Hence the np.flip
    """
    pfb_data = PFData(pfbfile)
    pfb_data.loadHeader()
    pfb_data.loadData()
    arr = pfb_data.getDataAsArray()
    pfb_data.close()
    assert arr.ndim == 3, 'Only 3D arrays are supported'
    return np.flip(arr, axis=1)
