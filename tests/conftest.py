import os.path
import pytest
import pfspinup
from pfspinup.pfmetadata import PFMetadata

RUN_DIR = os.path.join(os.path.dirname(pfspinup.__file__), 'data/example_run')
RUN_NAME = 'LW_CLM_Ex4'

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


@pytest.fixture(scope='module')
def run_dir():
    return RUN_DIR


@pytest.fixture(scope='module')
def run_name():
    return RUN_NAME


@pytest.fixture(scope='module')
def test_data_dir():
    return TEST_DATA_DIR


@pytest.fixture(scope='module')
def metadata():
    return PFMetadata(f'{RUN_DIR}/{RUN_NAME}.out.pfmetadata')
