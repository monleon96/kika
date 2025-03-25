Overview
========

MCNPy is a Python library built to make working with MCNP (Monte Carlo N-Particle) input and output files easier and more intuitive. It provides a modern, lightweight alternative to mcnptools, focusing on a clear, class-based design to streamline your MCNP workflows.

Originally created for personal use, MCNPy is now available publicly to support the nuclear engineering and radiation transport community. Development will continue according to the author's own work requirements. Community contributions are warmly welcomed via GitHub at `https://github.com/monleon96/MCNPy.git <https://github.com/monleon96/MCNPy.git>`_, as long as they fit with the project's overall vision.

Current Features
----------------

Input Processing
~~~~~~~~~~~~~~~~

- Parsing MCNP input files specifically for materials and PERT cards.
- Conversion between atomic and weight fractions for materials.
- Generation and printing of materials in MCNP input format.

MCTAL Analysis
~~~~~~~~~~~~~~

- Comprehensive parsing of MCNP tally output (MCTAL) files.
- Extraction of F4 tally results, including uncertainties.
- Access to Tally Fluctuation Chart (TFC) data.
- Conversion of tally results to pandas DataFrames.
- Visualization of tally convergence.
- Parsing for other tally types may be partially supported but has not been extensively tested.

Sensitivity Analysis
~~~~~~~~~~~~~~~~~~~~

- Preparation of MCNP input files for sensitivity calculations using PERT cards.
- Processing and analysis of MCNP perturbation calculations.
- Generation and visualization of sensitivity profiles.
- Creation of Sensitivity Data Files (SDF) compatible with SCALE.
- Analysis of linearity and visualization of sensitivity coefficients.

