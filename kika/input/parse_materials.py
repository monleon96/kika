import re
from typing import List, Optional, Tuple

from .material import Material


def _parse_nuclide_token(token: str) -> Tuple[int, Optional[str]]:
    """Return (zaid, library_suffix) parsed from token like '1001.80c' or '1001'."""
    if "." in token:
        nuclide_parts = token.split(".")
        return int(nuclide_parts[0]), nuclide_parts[1]
    return int(token), None


def read_material(lines: List[str], start_index: int) -> Tuple[Optional[Material], int]:
    """Read and parse a material card from MCNP input lines."""
    i = start_index
    line = lines[i].strip()
    card_lines: List[str] = []

    while line.endswith("&"):
        card_lines.append(line.rstrip("&").strip())
        i += 1
        if i < len(lines):
            line = lines[i].strip()
        else:
            break
    card_lines.append(line)
    full_line = " ".join(card_lines)

    if "$" in full_line:
        full_line = full_line.split("$")[0].strip()

    tokens = full_line.split()
    match = re.match(r"^m(\d+)", tokens[0])
    if not match:
        return None, i + 1

    material_id = int(match.group(1))
    material_obj = Material(id=material_id)

    inferred_fraction_type: Optional[str] = None
    idx = 1
    while idx < len(tokens):
        tok = tokens[idx]
        lower_tok = tok.lower()

        if lower_tok.startswith("nlib="):
            material_obj.libs["nlib"] = tok.split("=", 1)[1]
            idx += 1
            continue
        if lower_tok.startswith("plib="):
            material_obj.libs["plib"] = tok.split("=", 1)[1]
            idx += 1
            continue
        if lower_tok.startswith("ylib="):
            material_obj.libs["ylib"] = tok.split("=", 1)[1]
            idx += 1
            continue
        if lower_tok.startswith("lib="):
            lib_value = tok.split("=", 1)[1]
            lib_key = material_obj._infer_lib_key_from_suffix(lib_value)
            if lib_key is not None:
                material_obj.libs[lib_key] = lib_value
            idx += 1
            continue

        # At this point we should be reading ZAID/fraction pairs.
        nuclide_spec = tok
        try:
            zaid, specific_lib = _parse_nuclide_token(nuclide_spec)
        except ValueError:
            idx += 1
            continue

        if idx + 1 >= len(tokens):
            break

        try:
            raw_fraction = float(tokens[idx + 1])
        except ValueError:
            idx += 1
            continue

        current_type = "wo" if raw_fraction < 0.0 else "ao"
        if inferred_fraction_type is None:
            inferred_fraction_type = current_type
        elif inferred_fraction_type != current_type:
            raise ValueError(f"Material {material_id} has mixed atomic/weight fractions")

        material_obj.add_nuclide(zaid=zaid, fraction=abs(raw_fraction), library=specific_lib)
        idx += 2

    i += 1
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("c ") or not line:
            i += 1
            continue

        if "$" in line:
            line = line.split("$")[0].strip()

        if not line:
            i += 1
            continue

        if not re.match(r"^\s*\d+", line):
            break

        if re.match(r"^m\d+", line):
            break

        parts = line.split()
        idx = 0
        while idx < len(parts):
            nuclide_spec = parts[idx]
            try:
                zaid, specific_lib = _parse_nuclide_token(nuclide_spec)
            except ValueError:
                idx += 1
                continue

            if idx + 1 >= len(parts):
                break

            try:
                raw_fraction = float(parts[idx + 1])
            except ValueError:
                idx += 1
                continue

            current_type = "wo" if raw_fraction < 0.0 else "ao"
            if inferred_fraction_type is None:
                inferred_fraction_type = current_type
            elif inferred_fraction_type != current_type:
                raise ValueError(f"Material {material_id} has mixed atomic/weight fractions")

            material_obj.add_nuclide(zaid=zaid, fraction=abs(raw_fraction), library=specific_lib)
            idx += 2

        i += 1

    material_obj.fraction_type = inferred_fraction_type or "ao"
    return material_obj, i
