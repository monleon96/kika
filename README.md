# MCNPy

[![Version](https://img.shields.io/badge/version-0.2.5-blue.svg)](https://github.com/monleon96/MCNPy)
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

## Streamlit UI (KIKA)

### How to run locally

1. Install dependencies (Poetry or pip) including the extra requirements in `streamlit_app/requirements.txt`.
2. Copy `.env.sample` to `.env` and provide secrets:
   - `KIKA_SECRET_KEY` – random 32+ character string.
   - SMTP credentials (`KIKA_SMTP_*`) if you want to send emails locally.
   - Optional: point `KIKA_DB_PATH` to a custom location, or accept the default SQLite file created under `streamlit_app/data/`.
3. Launch the UI with Streamlit:
   ```bash
   streamlit run streamlit_app/KIKA.py
   ```
4. Use the admin helper to seed accounts if needed:
   ```bash
   python streamlit_app/scripts/manage_users.py create-user --name "Test User" --email user@example.com
   ```

### Email verification flow

1. A new user registers via the Streamlit UI. Their account is created with `email_verified = 0`.
2. MCNPy sends a signed verification link to the provided email using `itsdangerous` tokens (48‑hour expiry).
3. The user clicks the link, the app validates the token, flips `email_verified` to `1`, and stores feedback in the UI.
4. Until verification succeeds the login is blocked, and the user is prompted to request another link.
5. The verification link can be resent from the login tab or via the CLI:
   ```bash
   python streamlit_app/scripts/manage_users.py send-verification --email user@example.com
   ```
6. Admins can also list and manage accounts with `streamlit_app/scripts/manage_users.py`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
