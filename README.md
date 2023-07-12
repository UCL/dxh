# DOLFINx helpers

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Tests](https://github.com/UCL/dxh/actions/workflows/tests.yml/badge.svg)](https://github.com/UCL/dxh/actions/workflows/tests.yml)
[![Linting](https://github.com/UCL/dxh/actions/workflows/linting.yml/badge.svg)](https://github.com/UCL/dxh/actions/workflows/linting.yml)
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

Collection of helper functions for working with DOLFINx Python interface.

This project is developed in collaboration with the [Centre for Advanced Research Computing](https://ucl.ac.uk/arc), University College London.

## About

### Project team

Current members

- Matt Graham ([matt-graham](https://github.com/matt-graham))
- Krishnakumar Gopalakrishnan ([krishnakumarg1984](https://github.com/krishnakumarg1984))
- Deepika Garg ([deepikagarg20](https://github.com/deepikagarg20))
- Janosh Preuss ([janoschpreuss](https://github.com/janoschpreuss))
- Erik Burman ([burmanerik](https://github.com/burmanerik))

Former members

- Anastasis Georgoulas ([ageorgou](https://github.com/ageorgou))
- Jamie Quinn ([JamieJQuinn](https://github.com/JamieJQuinn))

### Research software engineering contact

Centre for Advanced Research Computing, University College London
([arc-collab@ucl.ac.uk](mailto:arc-collab@ucl.ac.uk))

## Built with

- [FEniCSx](https://fenicsproject.org/)
- [Matplotlib](https://matplotlib.org/)
- [NumPy](https://numpy.org/)

## Getting started

### Prerequisites

Compatible with Python 3.9 and 3.10. [Requires DOLFINx v0.6.0 or above to be installed](https://github.com/FEniCS/dolfinx#installation).

### Installation

To install the latest development using `pip` run

```sh
pip install git+https://https://github.com/UCL/dxh.git
```

### Running tests

Tests can be run across all compatible Python versions with [`tox`](https://tox.wiki/en/latest/)
or to run tests manually in a Python environment with `pytest` installed run

```sh
pytest tests
```

from the root of the repository.

## Acknowledgements

This work was funded by a grant from the the Engineering and Physical Sciences Research Council (EPSRC).
