import logging
from typing import List, Optional
from mcnpy.ace.classes.xs_data import CrossSectionData, ReactionCrossSection
from mcnpy.ace.parsers.xss import XssEntry

# Setup logger
logger = logging.getLogger(__name__)

def read_xs_data_block(ace, debug=False):
    """
    Read SIG block (reaction cross sections) from the XSS array if it exists.
    
    Parameters
    ----------
    ace : Ace
        The Ace object with XSS data and header
    debug : bool, optional
        Whether to print debug information, defaults to False
    """
    if (ace.header is None or ace.header.jxs_array is None or 
        ace.header.nxs_array is None or ace.xss_data is None or
        ace.xs_locators is None or not ace.xs_locators.has_neutron_data):
        if debug:
            logger.debug("Skipping XS data block: required data missing")
        return
    
    if debug:
        logger.debug("\n===== CROSS SECTION DATA BLOCK PARSING =====")
        logger.debug(f"Header info: ZAID={ace.header.zaid}")
    
    # Initialize xs_data if it doesn't exist
    if ace.xs_data is None:
        ace.xs_data = CrossSectionData()
    
    # Get the starting index for SIG block
    sig_idx = ace.header.jxs_array[7]  # JXS(7)
    
    if debug:
        logger.debug(f"JXS(7) = {sig_idx} → Starting index of SIG block")
    
    if sig_idx <= 0:
        if debug:
            logger.debug("No SIG block present (JXS(7) ≤ 0)")
        return
    
    if debug:
        logger.debug(f"SIG block starts at index {sig_idx} (0-indexed)")
    
    # Get MT numbers from reaction_mt_data
    if not ace.reaction_mt_data or not ace.reaction_mt_data.has_neutron_mt_data:
        if debug:
            logger.debug("No MT data available for neutron reactions")
        return
        
    mt_entries = ace.reaction_mt_data.incident_neutron
    locator_entries = ace.xs_locators.incident_neutron
    
    if debug:
        logger.debug(f"Found {len(mt_entries)} MT entries and {len(locator_entries)} locator entries")
    
    if len(mt_entries) != len(locator_entries):
        if debug:
            logger.debug(f"Number of MT entries ({len(mt_entries)}) doesn't match locator entries ({len(locator_entries)})")
        return
    
    # Process each reaction
    for i, (mt_entry, locator_entry) in enumerate(zip(mt_entries, locator_entries)):
        mt_value = int(mt_entry.value)
        locator_value = int(locator_entry.value)
        
        # Calculate absolute index
        abs_idx = sig_idx + locator_value 

        if debug:
            logger.debug(f"\nReaction {i+1}: MT={mt_value}")
            logger.debug(f"  Locator value: {locator_value}")
            logger.debug(f"  Absolute index: sig_idx + locator= {sig_idx} + {locator_value} = {abs_idx}")
        
        if abs_idx >= len(ace.xss_data):
            if debug:
                logger.debug(f"  ERROR: Absolute index {abs_idx} is out of bounds ({len(ace.xss_data)})")
            continue
            
        try:
            # Read energy grid index and number of energies
            energy_idx = int(ace.xss_data[abs_idx].value)
            num_energies = int(ace.xss_data[abs_idx + 1].value)
            
            if debug:
                logger.debug(f"  Energy grid index: {energy_idx}")
                logger.debug(f"  Number of energies: {num_energies}")
            
            # Read cross section values
            xs_start = abs_idx + 2
            xs_end = xs_start + num_energies
            
            if debug:
                logger.debug(f"  XS data range: XSS[{xs_start}:{xs_end}]")
            
            if xs_end <= len(ace.xss_data):
                # Store XssEntry objects directly
                xs_values = ace.xss_data[xs_start:xs_end]
                
                # Create and store ReactionCrossSection
                reaction_xs = ReactionCrossSection(
                    mt=mt_entry,
                    energy_idx=energy_idx,
                    num_energies=num_energies,
                    xs_values=xs_values
                )
                
                # Store using the integer MT value as the key for lookup
                ace.xs_data.reactions[mt_value] = reaction_xs
                
                if debug:
                    logger.debug(f"  Successfully read {len(xs_values)} XS values for MT={mt_value}")
            else:
                if debug:
                    logger.debug(f"  ERROR: XS data would extend beyond XSS array: {xs_end} > {len(ace.xss_data)}")
        except (IndexError, ValueError) as e:
            # Skip reaction if there's an error
            if debug:
                logger.debug(f"  ERROR processing reaction: {str(e)}")
            continue
    
    if debug:
        logger.debug(f"\nSuccessfully processed {len(ace.xs_data.reactions)} reaction cross sections")
