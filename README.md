![CI](https://github.com/hydroframe/PF_Model_Evaluation/workflows/CI/badge.svg?branch=master)

# PF Model Evaluation
This is a codebase *under active development*, for evaluating Parflow simulations and spinup.

**For full documentation see [the docs](https://hydroframe.github.io/PF_Model_Evaluation/).**

## Installation

This code works on Python 3. Creating a new virtual environment with Python 3
is the easiest option.

### Virtualenv

```
python3 -m venv env
source env/bin/activate
pip install -e .[dev]
```

We recommend installing the package in `develop` mode (the `-e` flag in `pip install -e .`),
as well as any libraries helpful for developers (notice the `[dev]` specification)
while this package is actively under development, so that you can tweak your local copy of the code
easily if you need to, and observe the changes.
