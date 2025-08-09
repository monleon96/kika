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
