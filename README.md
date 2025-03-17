# MCNPy

[![Version](https://img.shields.io/badge/version-0.2.1-blue.svg)](https://github.com/monleon96/MCNPy)
[![Documentation Status](https://readthedocs.org/projects/mcnpy/badge/?version=latest)](https://mcnpy.readthedocs.io/en/latest/?badge=latest)

A Python package for working with MCNP input and output files. MCNPy provides a lightweight alternative to mcnptools, offering essential functionality for parsing, analyzing, and manipulating MCNP files in Python.

## Features

- Parse and manipulate MCNP input files (materials, PERT cards)
- Read and analyze MCTAL output files
- Compute sensitivity data
- Generate and visualize sensitivity profiles
- Create Sensitivity Data Files (SDF)

## Installation

```bash
pip install mcnpy
```

## Quick Start

```python
import mcnpy

# Read an MCNP input file
inputfile = "path/to/input_file"
input_data = mcnpy.read_mcnp(inputfile)

# Read a MCTAL file
mctalfile = "path/to/mctal_file"
mctal = mcnpy.read_mctal(mctalfile)

# Access materials
materials = input_data.materials

# Compute sensitivity data
from mcnpy.sensitivities import compute_sensitivity
sens_data = compute_sensitivity(inputfile, mctalfile, tally=4, nuclide=26056, label='Sensitivity Fe-56')
```

## Documentation

For complete documentation, examples, and API reference, visit:
[MCNPy Documentation](https://mcnpy.readthedocs.io/en/latest/)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
