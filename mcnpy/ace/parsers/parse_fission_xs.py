import logging
from typing import List, Optional
from mcnpy.ace.ace import Ace
from mcnpy.ace.classes.fission_xs import FissionCrossSection

# Setup logger
logger = logging.getLogger(__name__)

def read_fission_xs_block(ace: Ace, debug=False) -> None:
    """
    Read the FIS block containing total fission cross section data.
    
    Parameters
    ----------
    ace : Ace
        The Ace object to update with fission cross section data
    debug : bool, optional
        Whether to print debug information, defaults to False
    """
    if debug:
        logger.debug("\n===== FIS BLOCK PARSING =====")
        logger.debug(f"Header info: ZAID={ace.header.zaid}")
    
    # Initialize container if not already present
    if not hasattr(ace, "fission_xs"):
        ace.fission_xs = FissionCrossSection()
    
    # Check if we have the necessary data and if FIS block exists (JXS(21) ≠ 0)
    if not ace.header or not ace.header.jxs_array or not ace.xss_data:
        if debug:
            logger.debug("Skipping FIS block: required data missing")
        return
    
    # Get FIS block index (JXS(21))
    fis_idx = ace.header.jxs_array[20]  # JXS(21) - convert to 1-indexed array
    
    if debug:
        logger.debug(f"JXS(21) = {fis_idx} → Locator for FIS block (FORTRAN 1-indexed)")
    
    if fis_idx <= 0 or fis_idx > len(ace.xss_data):
        if debug:
            logger.debug(f"No FIS block present or invalid index: JXS(21)={fis_idx}")
        return  # FIS block does not exist or invalid index
    
    # Convert to 0-indexed
    fis_idx -= 1
    
    if debug:
        logger.debug(f"FIS block starts at index {fis_idx} (0-indexed)")
    
    # Read the energy grid index (IE)
    energy_grid_index = int(ace.xss_data[fis_idx].value)
    
    if debug:
        logger.debug(f"Energy grid index: {energy_grid_index}")
    
    # Read the number of consecutive entries (NE)
    if fis_idx + 1 >= len(ace.xss_data):
        if debug:
            logger.debug(f"ERROR: Not enough data to read number of entries")
        return
    
    num_entries = int(ace.xss_data[fis_idx + 1].value)
    
    if debug:
        logger.debug(f"Number of cross section entries: {num_entries}")
    
    if num_entries <= 0 or fis_idx + 2 + num_entries - 1 >= len(ace.xss_data):
        if debug:
            logger.debug(f"ERROR: Invalid number of entries or would read past end of XSS array")
        return
    
    # Read the cross section values - store the XssEntry objects
    cross_sections = [ace.xss_data[fis_idx + 2 + i] for i in range(num_entries)]
    
    # Store the data
    ace.fission_xs.energy_grid_index = energy_grid_index
    ace.fission_xs.num_entries = num_entries
    ace.fission_xs.cross_sections = cross_sections
    ace.fission_xs.has_data = True
    
    if debug:
        logger.debug(f"Successfully read {num_entries} fission cross section values")
