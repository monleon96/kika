def build_xsdir_line(
    zaid_with_ext: str,
    awr: float,
    file_ref: str,
    xss_len: int,
    kT_MeV: float,
    *,
    dir_field: str | int = 0,
    file_type: int = 1,
    address: int = 1,
    record_len: int = 0,
    entries_per_record: int = 0,
    has_ptable: bool = False,
) -> str:
    """
    Build one xsdir entry line.

    Parameters
    ----------
    zaid_with_ext : str
        Full table name as it must appear in xsdir, e.g. "22048.01c".
        This is typically f"{hdr.zaid}{ext}".
    awr : float
        Atomic weight ratio (AWR). Example: hdr.atomic_weight_ratio.
    file_ref : str
        File reference in column 3. Can be a relative path (e.g. "../.../foo.01c")
        or just a filename, depending on how you want MCNP to resolve it.
    xss_len : int
        Length of the XSS array (number of double-precision words) for this table.
        In ACE terms this is usually NXS(2), e.g. hdr.nxs_array[1].
    kT_MeV : float
        Temperature in MeV (kT). Example: hdr.temperature.

    Keyword-only parameters (override defaults)
    -------------------------------------------
    dir_field : str | int, default 0
        Field 4. When nonzero/nonempty, MCNP uses this as a directory or access route.
        Leave as 0 if 'file_ref' already contains the full relative path you want.
    file_type : int, default 1
        Field 5. ACE file type: 1 = text (sequential), 2 = binary (direct access).
    address : int, default 1
        Field 6. For type 1, starting line number of the table; for type 2, record number.
        For one-table-per-file outputs, 1 is typical.
    record_len : int, default 0
        Field 8. Only used for file_type = 2 (binary). Bytes per record.
        Keep 0 for text ACE.
    entries_per_record : int, default 0
        Field 9. Only used for file_type = 2 (binary). Number of entries per record.
        Keep 0 for text ACE.
    has_ptable : bool, default False
        Whether to append the trailing "ptable" token (unresolved resonance probability tables).

    Returns
    -------
    str
        The complete xsdir line (without trailing newline).
    """
    fields = [
        zaid_with_ext,
        f"{awr:.6f}",
        file_ref,
        str(dir_field),
        str(file_type),
        str(address),
        str(xss_len),
        str(record_len),
        str(entries_per_record),
        f"{kT_MeV:.3E}",
    ]
    if has_ptable:
        fields.append("ptable")
    return " ".join(fields)


def write_xsdir_line(xsdir_path: str, line: str, mode: str = "w") -> None:
    """Write a single xsdir line to file. `mode='a'` appends, `mode='w'` overwrites."""
    with open(xsdir_path, mode) as fx:
        fx.write(line + "\n")


def create_xsdir_files_for_ace(
    ace_file_path: str,
    zaid: int,
    awr: float,
    xss_len: int,
    temperature_mev: float,
    sample_index: int,
    output_dir: str,
    master_xsdir_file: str = None,
    has_ptable: bool = False,
) -> None:
    """
    Create xsdir files for a perturbed ACE file.
    
    This function creates both per-sample xsdir files and optionally updates
    a master xsdir file. It's designed to be shared between ACE and ENDF 
    perturbation workflows.
    
    Parameters
    ----------
    ace_file_path : str
        Full path to the ACE file
    zaid : int
        ZAID of the isotope
    awr : float
        Atomic weight ratio
    xss_len : int
        Length of XSS array
    temperature_mev : float
        Temperature in MeV
    sample_index : int
        Sample index (0-based)
    output_dir : str
        Base output directory
    master_xsdir_file : str, optional
        Path to master xsdir file to copy and modify for each sample
    has_ptable : bool, default False
        Whether the ACE file has probability tables
    """
    import os
    import shutil
    
    # Extract file information
    base, ace_ext = os.path.splitext(os.path.basename(ace_file_path))
    sample_str = f"{sample_index+1:04d}"
    sample_dir = os.path.dirname(ace_file_path)
    
    # Remove sample string from base name if already present (avoid double sample numbers)
    if base.endswith(f"_{sample_str}"):
        base = base[:-len(f"_{sample_str}")]
    
    # — write per-sample .xsdir —
    rel = os.path.relpath(ace_file_path, output_dir).replace(os.sep, "/")
    rel_path = f"../{rel}"
    
    xsdir_line = build_xsdir_line(
        zaid_with_ext=f"{zaid}{ace_ext}",
        awr=awr,
        file_ref=rel_path,          
        xss_len=xss_len,   
        kT_MeV=temperature_mev,
        has_ptable=has_ptable,   
    )

    xsdir_path = os.path.join(sample_dir, f"{base}_{sample_str}.xsdir")
    write_xsdir_line(xsdir_path, xsdir_line, mode="w")

    # — write master xsdir files if requested —
    if master_xsdir_file:
        xsdir_dir = os.path.join(output_dir, "xsdir")
        os.makedirs(xsdir_dir, exist_ok=True)

        sample_tag = f"_{sample_index+1:04d}"
        orig_xs_base = os.path.splitext(os.path.basename(master_xsdir_file))[0]
        master_xs_name = orig_xs_base + sample_tag
        master_xs_path = os.path.join(xsdir_dir, master_xs_name)

        # Determine the source file: use existing modified file if available, otherwise original
        if os.path.exists(master_xs_path):
            # File already exists from previous isotope/run - use it as base
            source_xsdir = master_xs_path
        else:
            # First time creating this file - use original master as base
            source_xsdir = master_xsdir_file
            
        # Read from the appropriate source
        with open(source_xsdir, 'r') as f:
            xs_lines = f.readlines()

        # Build ZAID prefix for matching (e.g., "92235.01c")
        zaid_prefix = f"{zaid}{ace_ext}"
        line_found = False

        # Write master xsdir file: update existing entry or append new one
        with open(master_xs_path, 'w') as f:
            # First pass: write all lines, replacing any existing entry for this ZAID
            for line in xs_lines:
                stripped = line.strip()
                if stripped and stripped.startswith(zaid_prefix):
                    # Found existing entry - replace it with our perturbed version
                    f.write(xsdir_line + "\n")
                    line_found = True
                else:
                    # Write line as-is (already has \n from readlines)
                    f.write(line)
            
            # Second pass: if ZAID was not found, append it to the end
            if not line_found:
                # Ensure proper line separation before appending
                if xs_lines:
                    # Check if last line in the source file had a newline
                    if not xs_lines[-1].endswith('\n'):
                        # Last line didn't end with newline - add one for separation
                        f.write('\n')
                
                # Append our new xsdir entry
                f.write(xsdir_line + "\n")
