"""Parser for legacy SDF (Sensitivity Data File) format.

The goal of this module is to allow round‑trip fidelity with the writer
implemented in ``sdf.SDFData.write_file``. Parsing a file and then writing the
returned object again MUST yield a byte‑for‑byte identical file (except for
possible trailing newlines if the original file had platform specific line
endings which we normalise to ``\n`` on read).  We therefore:

1. Preserve ordering of reactions exactly as they appear in the file.
2. Do not attempt to 'normalise' floating point formatting – we store floats as
   Python floats but reproduce writer formatting which matches the original
   writer (scientific notation with 6 decimals, width 14, right aligned, 5 per line).
3. Reconstruct the perturbation energy boundaries in ascending order (writer
   reverses them when outputting).  The file lists the boundaries in descending
   order immediately under the line ``energy boundaries:`` with 5 values per line.

The legacy block structure for each reaction (as produced by
``SDFData._format_reaction_data``) is:

<nuclideSymbol(ljust13)><reactionName(ljust17)><ZAID(>5)><MT(>7)><\n>
"      0      0"\n
"  0.000000E+00  0.000000E+00      0      0"\n
<5 derived scalar values (ignored on parse, recomputed on write)>\n
<group sensitivities reversed (5 per line)>\n
<group errors reversed (5 per line)>\n
We intentionally IGNORE the 5 scalar values while parsing because the writer
recomputes them; keeping the parsed numbers could conceal inconsistencies.

API
---
parse_sdf(path: str) -> SDFData
    Parse file and return ``SDFData`` instance.

roundtrip_equal(path: str, tmp_dir: Optional[str] = None) -> bool
    Convenience helper that parses, writes to a temporary directory and
    performs textual comparison with the original file.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import re

from .sdf import SDFData, SDFReactionData
from kika._constants import ATOMIC_NUMBER_TO_SYMBOL, MT_TO_REACTION

HEADER_RE = re.compile(r"^(?P<title>.+) MCNP to SCALE sdf (?P<ngroups>\d+)gr\s*$")
NGROUP_LINE_RE = re.compile(r"^\s*(?P<ngroups>\d+) number of neutron groups\s*$")
NPROF_LINE_RE = re.compile(r"^\s*(?P<nprofiles>\d+)\s+ number of sensitivity profiles\s+(?P<nprofiles2>\d+) are region integrated\s*$")
R0_LINE_RE = re.compile(r"^\s*(?P<r0>[+-]?\d?\.\d+E[+-]\d+) \+/-\s+(?P<e0>[+-]?\d?\.\d+E[+-]\d+)\s*$")
REACTION_HEADER_RE = re.compile(r"^(?P<form>.{13})(?P<reac>.{17})(?P<zaid>\d{5})(?P<mt>\s*\d+)\s*$")
# The second and third fixed lines inside a reaction block are literal
FIXED_LINE_1 = "      0      0"
FIXED_LINE_2 = "  0.000000E+00  0.000000E+00      0      0"

FLOAT_PATTERN = r"[+-]?\d?\.\d+E[+-]\d+"


def _parse_scientific_numbers(line: str) -> List[float]:
    return [float(x) for x in re.findall(FLOAT_PATTERN, line)]


def read_sdf(path: str) -> SDFData:
    """Parse an SDF file returning an ``SDFData`` instance.

    Parameters
    ----------
    path : str
        Path to SDF file.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)

    with p.open("r") as f:
        lines = [ln.rstrip("\n") for ln in f]

    idx = 0
    if idx >= len(lines):
        raise ValueError("Empty SDF file")

    m = HEADER_RE.match(lines[idx])
    if not m:
        raise ValueError(f"Header line malformed: '{lines[idx]}'")
    title = m.group("title")
    ngroups_declared = int(m.group("ngroups"))
    idx += 1

    if idx >= len(lines) or not NGROUP_LINE_RE.match(lines[idx]):
        raise ValueError("Missing/invalid neutron groups line")
    idx += 1

    m = NPROF_LINE_RE.match(lines[idx])
    if not m:
        raise ValueError("Missing/invalid sensitivity profiles line")
    nprofiles_declared = int(m.group("nprofiles"))
    # optional cross check second number
    if int(m.group("nprofiles2")) != nprofiles_declared:
        raise ValueError("Mismatch in profile counts on line 3")
    idx += 1

    m = R0_LINE_RE.match(lines[idx])
    if not m:
        raise ValueError("Missing/invalid r0/e0 line")
    r0 = float(m.group("r0"))
    e0 = float(m.group("e0"))
    idx += 1

    if idx >= len(lines) or lines[idx].strip() != "energy boundaries:":
        raise ValueError("Expected 'energy boundaries:' line")
    idx += 1

    # Collect energy boundary numbers until we have ngroups+1 values.
    energy_values: List[float] = []
    while idx < len(lines) and len(energy_values) < (ngroups_declared + 1):
        nums = _parse_scientific_numbers(lines[idx])
        energy_values.extend(nums)
        idx += 1
    if len(energy_values) != (ngroups_declared + 1):
        raise ValueError("Energy boundaries count mismatch")

    # File stores descending order; convert to ascending for internal object.
    pert_energies = list(reversed(energy_values))

    reactions: List[SDFReactionData] = []

    # Parse reaction blocks until end of file
    while idx < len(lines):
        line = lines[idx]
        if not line.strip():
            # skip blank lines (should not normally happen, but be tolerant)
            idx += 1
            continue
        m = REACTION_HEADER_RE.match(line)
        if not m:
            raise ValueError(f"Malformed reaction header at line {idx+1}: {line}")
        header_reac_raw = m.group("reac")  # includes padding
        reaction_name = header_reac_raw.strip()
        zaid = int(m.group("zaid"))
        mt = int(m.group("mt"))
        idx += 1

        if idx >= len(lines) or lines[idx] != FIXED_LINE_1:
            raise ValueError(f"Missing fixed line 1 after reaction header at line {idx+1}")
        idx += 1
        if idx >= len(lines) or lines[idx] != FIXED_LINE_2:
            raise ValueError(f"Missing fixed line 2 after reaction header at line {idx+1}")
        idx += 1

        # Scalar values line (5 numbers) - read and discard
        if idx >= len(lines):
            raise ValueError("Unexpected EOF reading scalar values line")
        scalar_nums = _parse_scientific_numbers(lines[idx])
        if len(scalar_nums) != 5:
            raise ValueError(f"Expected 5 scalar values, got {len(scalar_nums)} at line {idx+1}")
        idx += 1

        # Read groupwise sensitivities (reversed order) until we have ngroups numbers
        sens_vals: List[float] = []
        while idx < len(lines) and len(sens_vals) < ngroups_declared:
            nums = _parse_scientific_numbers(lines[idx])
            if not nums:
                break
            sens_vals.extend(nums)
            idx += 1
        if len(sens_vals) != ngroups_declared:
            raise ValueError("Sensitivity values count mismatch")

        # Read errors (reversed order) until we have ngroups numbers
        err_vals: List[float] = []
        while idx < len(lines) and len(err_vals) < ngroups_declared:
            nums = _parse_scientific_numbers(lines[idx])
            if not nums:
                break
            err_vals.extend(nums)
            idx += 1
        if len(err_vals) != ngroups_declared:
            raise ValueError("Error values count mismatch")

        # Reverse back to ascending energy order
        sens_vals.reverse()
        err_vals.reverse()

        # Construct reaction data (nuclide symbol & reaction name resolved in __post_init__).
        # Provide reaction_name so unknown MT numbers are preserved.
        reaction = SDFReactionData(zaid=zaid, mt=mt, sensitivity=sens_vals, error=err_vals, reaction_name=reaction_name)
        reactions.append(reaction)

    if len(reactions) != nprofiles_declared:
        # Not fatal, but signal inconsistency.
        raise ValueError(f"Profile count mismatch: declared {nprofiles_declared}, parsed {len(reactions)}")

    # Attempt to reconstruct original energy label from filename if it contains two scientific numbers
    # separated by '_' (e.g. 1.00e-11_1.96e+01) to keep filename stable.
    energy_label = f"{pert_energies[0]:.2E}_{pert_energies[-1]:.2E}"  # fallback
    fname = p.name
    m_energy = re.search(r"(\d\.\d+e[+-]?\d+_\d\.\d+e[+-]?\d+)", fname, re.IGNORECASE)
    if m_energy:
        energy_label = m_energy.group(1)

    sdf_obj = SDFData(title=title, energy=energy_label, pert_energies=pert_energies, r0=r0, e0=e0, data=reactions)
    return sdf_obj


def roundtrip_equal(path: str, tmp_dir: Optional[str] = None) -> bool:
    """Roundtrip parse and write a file; return True if identical.

    The output filename is constructed by ``SDFData.write_file``; we compare
    the text content of the newly written file with the original path.
    """
    import tempfile, filecmp, os
    sdf = read_sdf(path)
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="sdf_rt_")
    sdf.write_file(tmp_dir)
    # Determine produced filename
    produced = Path(tmp_dir) / f"{sdf.title}_{sdf.energy}.sdf".replace(' ', '_').replace('/', '_').replace('\\', '_')
    if not produced.exists():
        return False
    # Compare textual content
    with open(path, 'r') as f1, open(produced, 'r') as f2:
        return f1.read() == f2.read()
