import logging
from typing import List, Optional
from mcnpy.ace.ace import Ace
from mcnpy.ace.classes.unresolved_resonance import UnresolvedResonanceTables, ProbabilityTable

# Setup logger
logger = logging.getLogger(__name__)

def read_unresolved_resonance_block(ace: Ace, debug=False) -> None:
    """
    Read the UNR block containing unresolved resonance probability tables.
    
    Parameters
    ----------
    ace : Ace
        The Ace object to update with unresolved resonance data
    debug : bool, optional
        Whether to print debug information, defaults to False
    """
    if debug:
        logger.debug("\n===== UNR BLOCK PARSING =====")
        logger.debug(f"Header info: ZAID={ace.header.zaid}")
    
    # Initialize container if not already present
    if not hasattr(ace, "unresolved_resonance"):
        ace.unresolved_resonance = UnresolvedResonanceTables()
    
    # Check if we have the necessary data and if UNR block exists (JXS(23) ≠ 0)
    if not ace.header or not ace.header.jxs_array or not ace.xss_data:
        if debug:
            logger.debug("Skipping UNR block: required data missing")
        return
    
    # Get UNR block index (JXS(23))
    unr_idx = ace.header.jxs_array[22]  # JXS(23) - convert to 0-indexed array
    
    if debug:
        logger.debug(f"JXS(23) = {unr_idx} → Locator for UNR block (FORTRAN 1-indexed)")
    
    if unr_idx <= 0 or unr_idx > len(ace.xss_data):
        if debug:
            logger.debug(f"No UNR block present or invalid index: JXS(23)={unr_idx}")
        return  # UNR block does not exist or invalid index
    
    # Convert to 0-indexed
    unr_idx -= 1
    
    if debug:
        logger.debug(f"UNR block starts at index {unr_idx} (0-indexed)")
    
    # Read the number of incident energies (N)
    if unr_idx >= len(ace.xss_data):
        if debug:
            logger.debug(f"ERROR: UNR index {unr_idx} is out of bounds ({len(ace.xss_data)})")
        return
    
    num_energies = int(ace.xss_data[unr_idx].value)
    
    if debug:
        logger.debug(f"Number of incident energies: {num_energies}")
    
    # Read the table length (M)
    if unr_idx + 1 >= len(ace.xss_data):
        if debug:
            logger.debug(f"ERROR: Not enough data to read table length")
        return
    
    table_length = int(ace.xss_data[unr_idx + 1].value)
    
    if debug:
        logger.debug(f"Table length: {table_length}")
    
    # Read the interpolation parameter (INT)
    if unr_idx + 2 >= len(ace.xss_data):
        if debug:
            logger.debug(f"ERROR: Not enough data to read interpolation parameter")
        return
    
    interpolation = int(ace.xss_data[unr_idx + 2].value)
    
    if debug:
        logger.debug(f"Interpolation parameter: {interpolation}")
    
    # Read the inelastic competition flag (ILF)
    if unr_idx + 3 >= len(ace.xss_data):
        if debug:
            logger.debug(f"ERROR: Not enough data to read inelastic flag")
        return
    
    inelastic_flag = int(ace.xss_data[unr_idx + 3].value)
    
    if debug:
        logger.debug(f"Inelastic competition flag: {inelastic_flag}")
    
    # Read the other absorption flag (IOA)
    if unr_idx + 4 >= len(ace.xss_data):
        if debug:
            logger.debug(f"ERROR: Not enough data to read absorption flag")
        return
    
    other_absorption_flag = int(ace.xss_data[unr_idx + 4].value)
    
    if debug:
        logger.debug(f"Other absorption flag: {other_absorption_flag}")
    
    # Read the factors flag (IFF)
    if unr_idx + 5 >= len(ace.xss_data):
        if debug:
            logger.debug(f"ERROR: Not enough data to read factors flag")
        return
    
    factors_flag = int(ace.xss_data[unr_idx + 5].value)
    
    if debug:
        logger.debug(f"Factors flag: {factors_flag}")
    
    # Check if we have enough data for the energy grid
    if unr_idx + 6 + num_energies - 1 >= len(ace.xss_data):
        if debug:
            logger.debug(f"ERROR: Not enough data for energy grid")
        return
    
    # Read the energy grid - store the XssEntry objects
    energies = [ace.xss_data[unr_idx + 6 + i] for i in range(num_energies)]
    
    if debug:
        logger.debug(f"Read {len(energies)} energy points for UNR grid")
    
    # Store the basic data
    ace.unresolved_resonance.num_energies = num_energies
    ace.unresolved_resonance.table_length = table_length
    ace.unresolved_resonance.interpolation = interpolation
    ace.unresolved_resonance.inelastic_flag = inelastic_flag
    ace.unresolved_resonance.other_absorption_flag = other_absorption_flag
    ace.unresolved_resonance.factors_flag = factors_flag
    ace.unresolved_resonance.energies = energies
    
    # Calculate the start of the probability tables
    ptable_idx = unr_idx + 6 + num_energies
    
    if debug:
        logger.debug(f"Probability tables start at index {ptable_idx}")
        logger.debug(f"Expected table size: 6 x {table_length} x {num_energies} = {6 * table_length * num_energies}")
    
    # Check if we have enough data for the tables
    if ptable_idx + 6 * table_length * num_energies - 1 >= len(ace.xss_data):
        if debug:
            logger.debug(f"WARNING: Not enough data for complete probability tables")
        # Still mark as having data even if incomplete
        ace.unresolved_resonance.has_data = True
        return
    
    # Read each probability table
    if debug:
        logger.debug(f"Reading {num_energies} probability tables")
    
    for i in range(num_energies):
        table = ProbabilityTable(energy=energies[i].value)
        
        if debug and i == 0:
            logger.debug(f"  Reading table for energy {energies[i].value}")
        
        # Read the 6 components of the probability table
        for j in range(6):
            start_idx = ptable_idx + (i * 6 + j) * table_length
            end_idx = start_idx + table_length
            
            if debug and i == 0:
                component_names = ["cumulative prob", "total XS", "elastic XS", "fission XS", "capture XS", "heating"]
                logger.debug(f"    Component {j+1}: {component_names[j]} - indices {start_idx}:{end_idx}")
            
            if end_idx > len(ace.xss_data):
                # If we run out of data, stop reading
                if debug:
                    logger.debug(f"    ERROR: Not enough data for component {j+1}")
                break
            
            # Store the XssEntry objects
            values = [ace.xss_data[start_idx + k] for k in range(table_length)]
            
            # Assign the values to the appropriate array based on j
            if j == 0:
                table.cumulative_probabilities = values
            elif j == 1:
                table.total_xs = values
            elif j == 2:
                table.elastic_xs = values
            elif j == 3:
                table.fission_xs = values
            elif j == 4:
                table.capture_xs = values
            elif j == 5:
                table.heating_numbers = values
        
        ace.unresolved_resonance.tables.append(table)
    
    ace.unresolved_resonance.has_data = True
    
    if debug:
        logger.debug(f"Successfully read {len(ace.unresolved_resonance.tables)} probability tables")
