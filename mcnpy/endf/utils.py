"""
Utility functions for ENDF file parsing and writing.

Contains helper functions for handling the specific formatting requirements of ENDF files.
"""
import re
from typing import Dict,Union, List, Optional, Tuple, Sequence, Any
from .classes.mt import MT
from .classes.mf1.mf1mt import MT451
from .classes.mf import MF
import numpy as np
import math
from numpy.typing import ArrayLike

def format_endf_number(value: Union[int, float, None], width: int = 11) -> str:
    """
    Format a number according to ENDF specifications.

    The output is an 11-character field made up as follows:
      - The first character is '-' if the number is negative or a blank if positive.
      - The number is written in scientific notation without an 'E'.
      - When the exponent (after normalization) has only one digit (|exponent| < 10),
        the mantissa is printed with 6 decimal digits and the exponent with one digit.
      - When the exponent has two digits (|exponent| >= 10), the mantissa is printed with 5 decimal digits and the exponent with two digits.
      
    For example:
      - A number like -3.14159e-1 will be formatted as "-3.141590-1".
      - A number like 1.234567e+5 will be formatted as " 1.234567+5".
      - A number like 1.0e10 will be formatted as " 1.00000+10".

    Args:
        value: The number to be formatted. If None, returns a blank field.
        width: The total field width (default is 11 characters).

    Returns:
        A string representing the formatted number in ENDF style.
    """
    if value is None:
        return " " * width

    # Special handling for zero: use exponent 0 (one-digit) and 6 decimal places.
    if value == 0:
        return " 0.000000+0"

    sign_char = "-" if value < 0 else " "
    abs_val = abs(value)
    exponent = int(math.floor(math.log10(abs_val)))
    mantissa = abs_val / (10 ** exponent)

    # Select the number of decimals based on the exponent.
    # Use 6 decimals if |exponent| < 10, else use 5 decimals.
    # Adjust the mantissa if rounding would push it to 10 or more.
    while True:
        prec = 6 if abs(exponent) < 10 else 5
        mantissa_str = f"{mantissa:1.{prec}f}"
        if float(mantissa_str) < 10:
            break
        mantissa /= 10
        exponent += 1

    # Format the exponent: one digit if |exponent| < 10, two digits otherwise.
    if abs(exponent) < 10:
        exp_str = f"{abs(exponent):d}"
    else:
        exp_str = f"{abs(exponent):02d}"
    exp_sign = '+' if exponent >= 0 else '-'

    formatted = f"{sign_char}{mantissa_str}{exp_sign}{exp_str}"
    return formatted.rjust(width)


# Format constants for ENDF data types
ENDF_FORMAT_FLOAT = 'float'       # Scientific notation (e.g., " 1.234567+5")
ENDF_FORMAT_INT = 'int'           # Integer format (e.g., "         11")
ENDF_FORMAT_INT_ZERO = 'int_zero' # Integer with zero rendered as 0 (not blank)
ENDF_FORMAT_BLANK = 'blank'       # Blank field
ENDF_FORMAT_PRESERVE = 'preserve' # Use value's own type to determine format


def format_endf_data_line(values: Sequence[Union[int, float, None]], 
                         mat: int, mf: int, mt: int, line_num: int = 0,
                         formats: Optional[List[str]] = None) -> str:
    """
    Format a complete ENDF line with both data and identification parts.
    
    Args:
        values: Sequence of up to 6 numeric values for the data part
        mat: Material number
        mf: File number
        mt: Section number
        line_num: Line sequence number (optional)
        formats: Optional list of format types for each value (ENDF_FORMAT_*)
        
    Returns:
        Formatted 80-character ENDF line
    """
    # Format the data part (columns 1-66)
    data_part = ""
    
    # Apply formats if provided, otherwise use default formatting
    if formats:
        # Make sure formats list matches values length
        format_list = formats + [ENDF_FORMAT_PRESERVE] * (len(values) - len(formats))
        format_list = format_list[:len(values)]
        
        for i, (value, fmt) in enumerate(zip(values, format_list)):
            if fmt == ENDF_FORMAT_INT and value is not None:
                # Format as integer, always show zero (don't use blank for zero)
                data_part += f"{int(value):11d}"
            elif fmt == ENDF_FORMAT_INT_ZERO and value is not None:
                # Format as integer, with zero as actual zero
                data_part += f"{int(value):11d}"
            elif fmt == ENDF_FORMAT_BLANK or value is None:
                # Blank field
                data_part += " " * 11
            else:
                # Default to float format
                data_part += format_endf_number(value)
    else:
        # Use default formatting based on value type
        for i, value in enumerate(values[:6]):
            data_part += format_endf_number(value)
    
    # Pad to 66 characters if needed
    data_part = data_part.ljust(66)
    
    # Format the identification part (columns 67-80)
    id_part = f"{mat:4d}{mf:2d}{mt:3d}{line_num:5d}"
    
    return data_part + id_part


def parse_number(text: str) -> Union[float, int, None]:
    """
    Parse an ENDF-formatted number.
    
    ENDF uses a special format where numbers can be written in forms like:
    "1.234+5" meaning 1.234×10^5
    
    Args:
        text: The text representation of the number
        
    Returns:
        Parsed number as float or int, or None if parsing fails
    """
    text = text.strip()
    if not text:
        return None
    
    try:
        # Try standard float parsing first
        value = float(text)
        # Return as int if it's a whole number
        if value.is_integer():
            return int(value)
        return value
    except ValueError:
        # Handle ENDF-specific format where "+" or "-" might be used instead of "E"
        # For example, "1.234+5" instead of "1.234E+5"
        match = re.search(r'([-+]?\d*\.\d*)([+-]\d+)', text)
        if match:
            try:
                mantissa = float(match.group(1))
                exponent = int(match.group(2))
                value = mantissa * (10 ** exponent)
                if value.is_integer():
                    return int(value)
                return value
            except (ValueError, IndexError):
                pass
                
        # If all parsing fails
        return None


def parse_line(line: str) -> Dict[str, Any]:
    """
    Parse a standard ENDF record line into its components.
    
    Args:
        line: An 80-character ENDF line
        
    Returns:
        Dictionary with parsed components
    """
    result = {}
    
    # Parse data fields (columns 1-66)
    if len(line) >= 66:
        data_part = line[:66]
        # ENDF format typically has 6 fields of 11 characters each
        for i in range(6):
            field_name = f"C{i+1}"
            start = i * 11
            end = start + 11
            if end <= len(data_part):
                field_value = data_part[start:end].strip()
                result[field_name] = parse_number(field_value)
    
    # Parse identification fields (columns 67-80)
    if len(line) >= 75:
        result["MAT"] = int(line[66:70]) if line[66:70].strip() else None
        result["MF"] = int(line[70:72]) if line[70:72].strip() else None
        result["MT"] = int(line[72:75]) if line[72:75].strip() else None
        
    if len(line) >= 80:
        result["SEQ"] = int(line[75:80]) if line[75:80].strip() else None
    
    return result


def parse_endf_id(line: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Parse the identification fields from an ENDF line.
    
    ENDF format specifies:
    - Columns 67-70 (0-indexed: 66-69): MAT number
    - Columns 71-72 (0-indexed: 70-71): MF number
    - Columns 73-75 (0-indexed: 72-74): MT number
    
    Args:
        line: A line from an ENDF file
        
    Returns:
        Tuple of (MAT, MF, MT) numbers
    """
    if len(line) < 75:
        return None, None, None
    
    try:
        # ENDF format has specific columns for MAT, MF, MT
        mat_str = line[66:70].strip()
        mf_str = line[70:72].strip()
        mt_str = line[72:75].strip()
        
        # Convert to integers, handling empty strings
        mat = int(mat_str) if mat_str else None
        mf = int(mf_str) if mf_str else None
        mt = int(mt_str) if mt_str else None
        
        return mat, mf, mt
    except ValueError as e:
        # This might happen if the fields contain non-numeric data
        return None, None, None


def group_lines_by_mt_with_positions(lines: List[str]) -> Tuple[Dict[int, List[str]], Dict[int, int]]:
    """
    Group lines by MT numbers and track their line counts.
    
    Args:
        lines: List of string lines
        
    Returns:
        Tuple of:
            - Dictionary mapping MT numbers to lists of lines
            - Dictionary mapping MT numbers to line counts
    """
    result: Dict[int, List[str]] = {}
    line_counts: Dict[int, int] = {}
    current_mt = None
    current_lines: List[str] = []
    
    for i, line in enumerate(lines):
        # Parse MT number from the line
        try:
            _, _, mt = parse_endf_id(line)
            
            # Skip MT=0 as a data section (it's a marker)
            if mt == 0:
                # If we were collecting a section, finalize it before the MT=0 marker
                if current_mt is not None and current_lines:
                    result[current_mt] = current_lines
                    line_counts[current_mt] = len(current_lines)
                    current_mt = None
                    current_lines = []
                continue
            
            # Handle section changes
            if current_mt is None:
                # Start a new section
                current_mt = mt
                current_lines = [line]
            elif mt != current_mt:
                # Complete the previous section
                result[current_mt] = current_lines
                line_counts[current_mt] = len(current_lines)
                
                # Start a new section
                current_mt = mt
                current_lines = [line]
            else:
                # Continue current section
                current_lines.append(line)
        except Exception:
            # If we can't parse the line, just add it to the current section if we have one
            if current_mt is not None:
                current_lines.append(line)
    
    # Add the last section if needed
    if current_mt is not None and current_lines:
        result[current_mt] = current_lines
        line_counts[current_mt] = len(current_lines)
    
    return result, line_counts


# Interpolation scheme codes and their meanings based on ENDF format specification
INTERPOLATION_SCHEMES = {
    1: "constant in x (histogram)",
    2: "linear-linear",
    3: "linear-log",
    4: "log-linear",
    5: "log-log",
    6: "special one-dimensional interpolation for charged-particle cross sections",
    11: "method of corresponding points (interpolation law 1)",
    12: "method of corresponding points (interpolation law 2)",
    13: "method of corresponding points (interpolation law 3)",
    14: "method of corresponding points (interpolation law 4)",
    15: "method of corresponding points (interpolation law 5)",
    21: "unit base interpolation (interpolation law 1)",
    22: "unit base interpolation (interpolation law 2)",
    23: "unit base interpolation (interpolation law 3)",
    24: "unit base interpolation (interpolation law 4)",
    25: "unit base interpolation (interpolation law 5)"
}

def get_interpolation_scheme_name(scheme_code):
    """
    Get the descriptive name of an interpolation scheme based on its code.
    
    Parameters:
        scheme_code (int): The interpolation scheme code (INT in ENDF format)
        
    Returns:
        str: The descriptive name of the interpolation scheme
    """
    return INTERPOLATION_SCHEMES.get(scheme_code, f"Unknown scheme ({scheme_code})")

def describe_interpolation_region(nbt, int_code):
    """
    Generate a descriptive string for an interpolation region.
    
    Parameters:
        nbt (int): The NBT value indicating the upper bound of points for this interpolation
        int_code (int): The interpolation scheme code (INT)
        
    Returns:
        str: A descriptive string for this interpolation region
    """
    scheme_name = get_interpolation_scheme_name(int_code)
    return f"Points up to {nbt} use {scheme_name}"



def _regionize(nbt_int_pairs: Sequence[Tuple[int, int]], np_len: int) -> List[Tuple[int, int, int]]:
    """
    Convert ENDF (NBT, INT) pairs into 0-based [start_idx, end_idx, INT] regions.
    NBT is 1-based index of the *last* point in each region in ENDF.
    """
    if not nbt_int_pairs:
        # single region with default linear
        return [(0, np_len - 1, 2)]
    regions: List[Tuple[int, int, int]] = []
    start = 0
    for nbt, int_code in nbt_int_pairs:
        end = min(max(nbt - 1, 0), np_len - 1)
        if end >= start:
            regions.append((start, end, int_code))
        start = min(max(nbt, 0), np_len)  # next region starts at nbt (1-based → 0-based)
        if start >= np_len:
            break
    # Guard if list does not cover the tail
    if regions and regions[-1][1] < np_len - 1:
        regions.append((regions[-1][1], np_len - 1, regions[-1][2]))
    if not regions:
        regions = [(0, np_len - 1, 2)]
    return regions


def _base_int_code(int_code: int) -> int:
    """Map 11–15 → 1–5 and 21–25 → 1–5 for 1-D use."""
    if int_code >= 10:
        return int_code % 10 if int_code % 10 != 0 else 5
    return int_code


def _interp_pair(x: float, x1: float, y1: float, x2: float, y2: float, int_code: int) -> float:
    """
    Interpolate y(x) between (x1,y1) and (x2,y2) using ENDF INT code semantics (1–5).
    For INT=6 or unsupported codes → fall back to linear-linear.
    """
    if x1 == x2:
        return y1
    t = (x - x1) / (x2 - x1)
    code = _base_int_code(int_code)
    if code == 1:  # histogram/constant
        return y1
    elif code == 2:  # lin-lin
        return (1.0 - t) * y1 + t * y2
    elif code == 3:  # lin-log (y linear in ln x)
        if x1 <= 0 or x2 <= 0 or x <= 0:
            return (1.0 - t) * y1 + t * y2
        lx1, lx2, lx = math.log(x1), math.log(x2), math.log(x)
        tt = (lx - lx1) / (lx2 - lx1)
        return (1.0 - tt) * y1 + tt * y2
    elif code == 4:  # log-lin (ln y linear in x)
        if y1 <= 0 or y2 <= 0:
            return (1.0 - t) * y1 + t * y2
        ln_y = (1.0 - t) * math.log(y1) + t * math.log(y2)
        return math.exp(ln_y)
    elif code == 5:  # log-log (ln y linear in ln x)
        if y1 <= 0 or y2 <= 0 or x1 <= 0 or x2 <= 0 or x <= 0:
            return (1.0 - t) * y1 + t * y2
        lx1, lx2, lx = math.log(x1), math.log(x2), math.log(x)
        tt = (lx - lx1) / (lx2 - lx1)
        ln_y = (1.0 - tt) * math.log(y1) + tt * math.log(y2)
        return math.exp(ln_y)
    else:  # fallback for INT=6 etc.
        return (1.0 - t) * y1 + t * y2


def interpolate_1d_endf(
    x_grid: ArrayLike,
    y_grid: ArrayLike,
    nbt_int_pairs: Sequence[Tuple[int, int]],
    xq: Union[float, ArrayLike],
    out_of_range: str = "zero",
) -> Union[float, np.ndarray]:
    """
    ENDF one-dimensional interpolation using (NBT, INT) regions (Table of INT codes).
    - out_of_range: 'zero'  → return 0 outside grid
                    'hold'  → hold edge value
    """
    x = np.asarray(x_grid, dtype=float)
    y = np.asarray(y_grid, dtype=float)
    if x.size == 0:
        if np.ndim(xq) == 0:
            return 0.0
        return np.zeros_like(np.asarray(xq, dtype=float))
    regions = _regionize(nbt_int_pairs, len(x))
    xq_arr = np.asarray([xq], dtype=float).ravel() if np.ndim(xq) == 0 else np.asarray(xq, dtype=float)
    out = np.zeros_like(xq_arr, dtype=float)

    for i, xv in enumerate(xq_arr):
        if xv < x[0]:
            out[i] = 0.0 if out_of_range == "zero" else y[0]
            continue
        if xv > x[-1]:
            out[i] = 0.0 if out_of_range == "zero" else y[-1]
            continue
        # find region and interval
        # quick locate index k with x[k] <= xv <= x[k+1]
        k = np.searchsorted(x, xv, side="right") - 1
        k = min(max(k, 0), len(x) - 2)
        # find region with k inside [start, end]
        int_code = 2
        for start, end, ic in regions:
            if start <= k + 1 <= end:
                int_code = ic
                break
        out[i] = _interp_pair(xv, x[k], y[k], x[k + 1], y[k + 1], int_code)

    if np.ndim(xq) == 0:
        return float(out[0])
    return out


def project_tabulated_to_legendre(
    mu: ArrayLike,
    fmu: ArrayLike,
    max_order: int,
    ang_nbt_int: Optional[Sequence[Tuple[int, int]]] = None,
    quad_order: int = 64,
) -> np.ndarray:
    """
    Compute Legendre coefficients a_l up to max_order from tabulated f(μ) on μ∈[-1,1].
    Uses Gauss–Legendre quadrature on an ENDF-interpolated f(μ) (respects angular INT codes).
    a_l = (2l+1)/2 ∫_{-1}^{1} P_l(μ) f(μ) dμ
    """
    mu = np.asarray(mu, dtype=float)
    fmu = np.asarray(fmu, dtype=float)
    if mu.size == 0 or fmu.size == 0:
        return np.zeros(max_order + 1, dtype=float)

    # GL nodes/weights
    mu_q, w_q = np.polynomial.legendre.leggauss(quad_order)
    # Interpolate f to GL nodes with ENDF angular interpolation (default linear)
    f_q = interpolate_1d_endf(mu, fmu, ang_nbt_int or [(len(mu), 2)], mu_q, out_of_range="hold")

    # Normalize on [-1,1] using the same quadrature
    norm = float(np.sum(f_q * w_q))
    if abs(norm) > 1e-15:
        f_q = f_q / norm

    # Project
    coeffs = np.zeros(max_order + 1, dtype=float)
    for l in range(max_order + 1):
        P_l = np.polynomial.legendre.legval(mu_q, [0] * l + [1])  # evaluate P_l(μ)
        coeffs[l] = (2 * l + 1) / 2.0 * float(np.sum(P_l * f_q * w_q))
    return coeffs


def auto_trim_legendre_tail(
    coeffs_by_l: Dict[int, Union[float, np.ndarray]],
    tol: float = 1e-6,
    min_order: int = 0
) -> Dict[int, Union[float, np.ndarray]]:
    """
    Auto-trim to smallest L such that sum_{ℓ>L} |a_ℓ| < tol.
    If values are arrays over energies, use a *global* L that satisfies the condition for all energies,
    so dictionary keys remain consistent.
    
    Parameters
    ----------
    coeffs_by_l : dict
        Dictionary mapping order l to coefficient values
    tol : float
        Tolerance for trimming
    min_order : int
        Minimum order to keep (ensures at least orders 0 to min_order are returned)
    """
    if not coeffs_by_l:
        return coeffs_by_l
        
    # collect in order
    max_l = max(coeffs_by_l) if coeffs_by_l else 0
    arrs = [np.atleast_1d(coeffs_by_l.get(l, 0.0)) for l in range(max_l + 1)]
    A = np.vstack(arrs)  # shape: (L+1, nE)
    absA = np.abs(A)
    # tail sums S_l = sum_{j>l} |a_j|
    tail = np.flipud(np.cumsum(np.flipud(absA), axis=0))  # S_l includes |a_l|; we want >l, so shift
    tail_gt = np.vstack([tail[1:, :], np.zeros((1, tail.shape[1]))])  # shift up
    # For each energy (column), find smallest L with tail_gt[L] < tol
    per_energy_L = [int(np.argmax(tail_gt[:, j] < tol)) for j in range(tail_gt.shape[1])]
    L_global = max(per_energy_L) if per_energy_L else max_l
    
    # Ensure we keep at least up to min_order
    L_global = max(L_global, min_order)
    
    return {l: coeffs_by_l[l] for l in range(L_global + 1)}


def pick_mixed_branch(E: float, E_leg: np.ndarray, E_tab: np.ndarray) -> str:
    """
    For LTT=3, decide which branch to use at energy E.
    - If within Legendre range: 'leg'
    - If within tabulated range: 'tab'
    - If between disjoint ranges: pick the closer boundary
    """
    has_leg = E_leg.size > 0
    has_tab = E_tab.size > 0
    if not has_leg and not has_tab:
        return "none"
    if has_leg and (E <= E_leg.max()):
        return "leg"
    if has_tab and (E >= E_tab.min()):
        return "tab"
    if has_leg and has_tab:
        return "leg" if abs(E - E_leg.max()) <= abs(E - E_tab.min()) else "tab"
    return "leg" if has_leg else "tab"


def segment_int_codes(ne: int, nbt_int_pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
    """
    Build an array of length (ne-1) with the INT code for each energy interval [k, k+1].
    ENDF NBT's are 1-based indices of the *last* point in the region.
    """
    if not nbt_int_pairs:
        nbt_int_pairs = [(ne, 2)]  # default linear across full grid

    seg = np.full(ne - 1, 2, dtype=int)
    start = 0
    for nbt, ic in nbt_int_pairs:
        # region covers points [start ... end], so intervals [start ... end-1]
        end = max(0, min(nbt - 1, ne - 1))
        if end > start:
            seg[start:end] = ic
        start = max(0, min(nbt, ne - 1))
        if start >= ne - 1:
            break
    return seg


def interp_energy_values(E0: float, f0: np.ndarray,
                          E1: float, f1: np.ndarray,
                          E: float, int_code: int) -> np.ndarray:
    """
    Vectorized interpolation of y(E) between (E0,f0) and (E1,f1) under ENDF INT code (1..5).
    Falls back to linear where logs are invalid.
    """
    if E0 == E1:
        return np.array(f0, dtype=float, copy=True)

    t = (E - E0) / (E1 - E0)
    code = int_code % 10 if int_code >= 10 else int_code
    code = 5 if code == 0 else code  # 10,20 → 0 → use 5

    # default lin-lin
    if code == 1:
        return np.array(f0, dtype=float, copy=True)  # histogram in E: hold left
    if code == 2:
        return (1.0 - t) * np.asarray(f0, dtype=float) + t * np.asarray(f1, dtype=float)

    # helpers
    f0 = np.asarray(f0, dtype=float)
    f1 = np.asarray(f1, dtype=float)

    # lin-log (y linear in ln E)
    if code == 3:
        if E0 <= 0 or E1 <= 0 or E <= 0:
            return (1.0 - t) * f0 + t * f1
        le0, le1, le = math.log(E0), math.log(E1), math.log(E)
        tt = (le - le0) / (le1 - le0)
        return (1.0 - tt) * f0 + tt * f1

    # log-lin (ln y linear in E)
    if code == 4:
        mask = (f0 > 0.0) & (f1 > 0.0)
        out = (1.0 - t) * f0 + t * f1
        if np.any(mask):
            ln_y = (1.0 - t) * np.log(f0[mask]) + t * np.log(f1[mask])
            out[mask] = np.exp(ln_y)
        return out

    # log-log (ln y linear in ln E)
    if code == 5:
        if E0 <= 0 or E1 <= 0 or E <= 0:
            return (1.0 - t) * f0 + t * f1
        le0, le1, le = math.log(E0), math.log(E1), math.log(E)
        tt = (le - le0) / (le1 - le0)
        mask = (f0 > 0.0) & (f1 > 0.0)
        out = (1.0 - t) * f0 + t * f1
        if np.any(mask):
            ln_y = (1.0 - tt) * np.log(f0[mask]) + tt * np.log(f1[mask])
            out[mask] = np.exp(ln_y)
        return out

    # fallback
    return (1.0 - t) * f0 + t * f1