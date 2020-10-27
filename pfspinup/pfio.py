import os.path
from parflowio.pyParflowio import PFData


def pfread(pfbfile):
    """
    Read a pfb file and return data as an ndarray
    :param pfbfile: path to pfb file
    :return: An ndarray of ndim=3, with shape (nz, ny, nx)
    """
    if not os.path.exists(pfbfile):
        raise RuntimeError(f'{pfbfile} not found')

    pfb_data = PFData(pfbfile)
    pfb_data.loadHeader()
    pfb_data.loadData()
    arr = pfb_data.moveDataArray()
    pfb_data.close()
    assert arr.ndim == 3, 'Only 3D arrays are supported'
    return arr
