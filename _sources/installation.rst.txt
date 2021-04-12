Installation
============

Create and Activate a new Python 3 environment
**********************************************

Assuming you have a Python 3 interpreter installed with venv, and available on your path, create and activate a new
environment using:

::

    python3 -m venv env
    source env/bin/activate

Test the package
****************

Use `pytest` to make sure all tests run correctly by doing:

::

    cd /path/to/git/clone/folder
    PYTHONPATH=. pytest tests

Install the package
*******************

If the tests pass, install the package for the currently active environment:

::

    cd /path/to/git/clone/folder
    pip install .

.. note::
    Installing the package installs to the ``site-packages`` folder of your active environment.
    This is only desirable if you are not going to be doing any development on the package,
    but simply want to run scripts that depend on the package. If you are doing any development work on this package,
    will most certainly want to a *developer mode* installation, using `pip install -e .[dev]`

Generating Documentation
************************

Sphinx Documentation of the source (a local copy of what you're looking at right now) can be generated using:

::

    cd /path/to/git/clone/folder/docs
    sphinx-apidoc -f -o ./source ../pfspinup -H Modules
    make clean
    make html

The built html files can be found at ``/path/to/git/clone/folder/docs/build/html``