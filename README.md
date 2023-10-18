# DOLFINx helpers

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Tests](https://github.com/UCL/dxh/actions/workflows/tests.yml/badge.svg)](https://github.com/UCL/dxh/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/UCL/dxh/graph/badge.svg?token=z0y0mDsvIx)](https://codecov.io/gh/UCL/dxh)
[![Linting](https://github.com/UCL/dxh/actions/workflows/linting.yml/badge.svg)](https://github.com/UCL/dxh/actions/workflows/linting.yml)
[![Documentation](https://github.com/UCL/dxh/actions/workflows/docs.yml/badge.svg)](https://github-pages.ucl.ac.uk/dxh/)
[![Licence][licence-badge]](./LICENCE.md)

<!--
[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]
-->

<!-- prettier-ignore-start -->
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/dxh
[conda-link]:               https://github.com/conda-forge/dxh-feedstock
[pypi-link]:                https://pypi.org/project/dxh/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/dxh
[pypi-version]:             https://img.shields.io/pypi/v/dxh
[licence-badge]:             https://img.shields.io/badge/License-MIT-yellow.svg
<!-- prettier-ignore-end -->

A collection of helper functions for working with [DOLFINx's Python interface](https://docs.fenicsproject.org/dolfinx/main/python/)
and visualizing objects using [Matplotlib](https://matplotlib.org/).

This project is developed in collaboration with the [Centre for Advanced Research Computing](https://ucl.ac.uk/arc), University College London.

## Documentation

Documentation can be viewed at https://github-pages.ucl.ac.uk/dxh/

## About

### Project team

Current members

- Erik Burman ([burmanerik](https://github.com/burmanerik))
- Sam Cunliffe ([samcunliffe](https://github.com/samcunliffe))
- Deepika Garg ([deepikagarg20](https://github.com/deepikagarg20))
- Krishnakumar Gopalakrishnan ([krishnakumarg1984](https://github.com/krishnakumarg1984))
- Matt Graham ([matt-graham](https://github.com/matt-graham))
- Janosch Preuss ([janoschpreuss](https://github.com/janoschpreuss))

Former members

- Anastasis Georgoulas ([ageorgou](https://github.com/ageorgou))
- Jamie Quinn ([JamieJQuinn](https://github.com/JamieJQuinn))

### Research software engineering contact

Centre for Advanced Research Computing, University College London
([arc.collaborations@ucl.ac.uk](mailto:arc.collaborations@ucl.ac.uk))

## Built with

- [FEniCSx](https://fenicsproject.org/)
- [Matplotlib](https://matplotlib.org/)
- [NumPy](https://numpy.org/)

## Getting started

### Prerequisites

Compatible with Python 3.9 and 3.10.
[We recommend DOLFINx v0.7.0 or above to be installed](https://github.com/FEniCS/dolfinx#installation) although we support v0.6.0 for now.

### Installation

To install the latest development using `pip` run

```sh
pip install git+https://github.com/UCL/dxh.git
```

Alternatively create a local clone of the repository with

```sh
git clone https://github.com/UCL/dxh.git
```

and then install in editable mode by running

```sh
pip install -e .
```

from the root of your clone of the repository.

### Running tests

Tests can be run across all compatible Python versions in isolated environments using
[`tox`](https://tox.wiki/en/latest/) by running

```sh
tox
```

from the root of the repository, or to run tests with Python 3.9 specifically run

```sh
tox -e test-py39
```

substituting `py39` for `py310` to run tests with Python 3.10.

To run tests manually in a Python environment with `pytest` installed run

```sh
pytest tests
```

again from the root of the repository.

### Building documentation

HTML documentation can be built locally using `tox` by running

```sh
tox -e docs
```

from the root of the repository with the output being written to `docs/_build/html`.

## Acknowledgements

This work was funded by a grant from the the Engineering and Physical Sciences Research Council (EPSRC).
