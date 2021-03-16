import tempfile
import numpy as np
import pfspinup.pfio as pfio


def test_read_write():
    arr = np.random.random((5, 40, 30))
    with tempfile.NamedTemporaryFile() as f:
        pfio.pfwrite(arr, f.name, overwrite=True)
        arr2 = pfio.pfread(f.name)
        assert np.allclose(arr, arr2)
