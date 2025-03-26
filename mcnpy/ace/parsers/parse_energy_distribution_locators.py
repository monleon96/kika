from typing import List, Dict, Optional
from mcnpy.ace.classes.ace import Ace
from mcnpy.ace.classes.energy_distribution_locators import EnergyDistributionLocators
from mcnpy.ace.parsers.xss import XssEntry
import logging

# Setup logger
logger = logging.getLogger(__name__)

def read_energy_locator_blocks(ace: Ace, debug: bool = False) -> None:
    """
    Read the LDLW, LDLWP, LDLWH, and DNEDL blocks from the ACE file.
    
    These blocks contain locators (indices) to the energy distribution data:
    - LDLW: For incident neutron reactions (JXS[10])
    - LDLWP: For photon production reactions (JXS[18])
    - LDLWH: For other particle production reactions (accessed through JXS[31])
    - DNEDL: For delayed neutron precursor groups (JXS[26])
    
    Parameters
    ----------
    ace : Ace
        The Ace object to update
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Raises
    ------
    ValueError
        If required data is missing or indices are invalid
    """
    if debug:
        logger.debug("Reading energy distribution locator blocks")
        
    if (ace.header is None or ace.header.jxs_array is None or 
        ace.header.nxs_array is None or ace.xss_data is None):
        raise ValueError("Cannot read energy locator blocks: header or XSS data missing")
    
    # Create a new container if not present
    if ace.energy_distribution_locators is None:
        ace.energy_distribution_locators = EnergyDistributionLocators()
    
    # Get the NXS values for number of MT values
    num_secondary_neutron_reactions = ace.header.num_secondary_neutron_reactions  # NXS(5)
    num_photon_production_reactions = ace.header.num_photon_production_reactions  # NXS(6)
    num_particle_types = ace.header.num_particle_types  # NXS(7)
    num_delayed_neutron_precursors = ace.header.num_delayed_neutron_precursors  # NXS(8)
    
    if debug:
        logger.debug(f"Secondary neutron reactions: {num_secondary_neutron_reactions}")
        logger.debug(f"Photon production reactions: {num_photon_production_reactions}")
        logger.debug(f"Particle types: {num_particle_types}")
        logger.debug(f"Delayed neutron precursors: {num_delayed_neutron_precursors}")
    
    # Store the number of MT values in the energy_distribution_locators container
    ace.energy_distribution_locators.num_secondary_neutron_reactions = num_secondary_neutron_reactions
    ace.energy_distribution_locators.num_photon_production_reactions = num_photon_production_reactions
    ace.energy_distribution_locators.num_particle_types = num_particle_types
    ace.energy_distribution_locators.num_delayed_neutron_precursors = num_delayed_neutron_precursors
    
    # Get the JXS pointers for each block
    ldlw_idx = ace.header.jxs_array[10] - 1  # JXS(10)
    ldlwp_idx = ace.header.jxs_array[18] - 1  # JXS(19)
    dnedl_idx = ace.header.jxs_array[26] - 1  # JXS(27)
    particle_pointer_idx = ace.header.jxs_array[31] - 1  # JXS(32)
    
    if debug:
        logger.debug(f"LDLW block index: {ldlw_idx}")
        logger.debug(f"LDLWP block index: {ldlwp_idx}")
        logger.debug(f"DNEDL block index: {dnedl_idx}")
        logger.debug(f"Particle pointer index: {particle_pointer_idx}")
    
    # Read LDLW block (incident neutron reactions with secondary neutrons)
    if num_secondary_neutron_reactions > 0 and ldlw_idx > 0:
        read_ldlw_block(ace, ldlw_idx, num_secondary_neutron_reactions, debug)
    
    # Read LDLWP block (photon production)
    if num_photon_production_reactions > 0 and ldlwp_idx > 0:
        read_ldlwp_block(ace, ldlwp_idx, num_photon_production_reactions, debug)
    
    # Read LDLWH block (other particle production)
    if num_particle_types > 0 and particle_pointer_idx > 0:
        read_ldlwh_block(ace, particle_pointer_idx, num_particle_types, debug)
    
    # Read DNEDL block (delayed neutron precursors)
    if num_delayed_neutron_precursors > 0 and dnedl_idx > 0:
        read_dnedl_block(ace, dnedl_idx, num_delayed_neutron_precursors, debug)


def read_ldlw_block(ace: Ace, ldlw_idx: int, num_reactions: int, debug: bool = False) -> None:
    """
    Read the LDLW block for incident neutron reaction energy distribution locators.
    
    Parameters
    ----------
    ace : Ace
        The Ace object to update
    ldlw_idx : int
        Starting index of the LDLW block in the XSS array
    num_reactions : int
        Number of reactions (NXS(5) for secondary neutron reactions)
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Raises
    ------
    ValueError
        If the LDLW block data is missing or invalid
    """
    if debug:
        logger.debug(f"Reading LDLW block at index {ldlw_idx} for {num_reactions} reactions")
        
    if ldlw_idx <= 0:
        raise ValueError(f"Invalid LDLW block index: {ldlw_idx}")
    
    if ldlw_idx >= len(ace.xss_data):
        raise ValueError(f"LDLW block index out of bounds: {ldlw_idx} >= {len(ace.xss_data)}")
    
    # Make sure we have enough data
    if ldlw_idx + num_reactions > len(ace.xss_data):
        raise ValueError(f"LDLW block truncated: need {num_reactions} entries, but only {len(ace.xss_data) - ldlw_idx} available")
    
    # Read the energy distribution locators
    # Store XssEntry objects directly
    ace.energy_distribution_locators.incident_neutron = ace.xss_data[ldlw_idx:ldlw_idx + num_reactions]
    
    if debug:
        logger.debug(f"Read {len(ace.energy_distribution_locators.incident_neutron)} incident neutron energy distribution locators")


def read_ldlwp_block(ace: Ace, ldlwp_idx: int, num_photon_reactions: int, debug: bool = False) -> None:
    """
    Read the LDLWP block for photon production energy distribution locators.
    
    Parameters
    ----------
    ace : Ace
        The Ace object to update
    ldlwp_idx : int
        Starting index of the LDLWP block in the XSS array
    num_photon_reactions : int
        Number of photon production reactions (NXS(6))
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Raises
    ------
    ValueError
        If the LDLWP block data is missing or invalid
    """
    if debug:
        logger.debug(f"Reading LDLWP block at index {ldlwp_idx} for {num_photon_reactions} reactions")
        
    if ldlwp_idx <= 0:
        raise ValueError(f"Invalid LDLWP block index: {ldlwp_idx}")
    
    if ldlwp_idx >= len(ace.xss_data):
        raise ValueError(f"LDLWP block index out of bounds: {ldlwp_idx} >= {len(ace.xss_data)}")
    
    # Make sure we have enough data
    if ldlwp_idx + num_photon_reactions > len(ace.xss_data):
        raise ValueError(f"LDLWP block truncated: need {num_photon_reactions} entries, but only {len(ace.xss_data) - ldlwp_idx} available")
    
    # Read the photon production energy distribution locators
    # Store XssEntry objects directly
    ace.energy_distribution_locators.photon_production = ace.xss_data[ldlwp_idx:ldlwp_idx + num_photon_reactions]
    
    if debug:
        logger.debug(f"Read {len(ace.energy_distribution_locators.photon_production)} photon production energy distribution locators")


def read_ldlwh_block(ace: Ace, particle_pointer_idx: int, num_particle_types: int, debug: bool = False) -> None:
    """
    Read the LDLWH block for other particle production energy distribution locators.
    
    Parameters
    ----------
    ace : Ace
        The Ace object to update
    particle_pointer_idx : int
        Starting index of the particle pointer block (JXS(31)) in the XSS array
    num_particle_types : int
        Number of particle types (NXS(7))
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Raises
    ------
    ValueError
        If the LDLWH block data is missing or invalid
    """
    if debug:
        logger.debug(f"Reading LDLWH block for {num_particle_types} particle types")
        
    if particle_pointer_idx <= 0:
        raise ValueError(f"Invalid particle pointer block index: {particle_pointer_idx}")
    
    if particle_pointer_idx >= len(ace.xss_data):
        raise ValueError(f"Particle pointer block index out of bounds: {particle_pointer_idx} >= {len(ace.xss_data)}")
    
    # Get the base index for JXS(32)
    if len(ace.header.jxs_array) <= 32:
        raise ValueError("JXS array too short: missing JXS(32) index")
    
    jxs32_idx = ace.header.jxs_array[32] - 1  # JXS(32)
    if jxs32_idx <= 0:
        raise ValueError(f"Invalid JXS(32) value: {jxs32_idx + 1}")
    
    # Initialize the particle production locators list
    ace.energy_distribution_locators.particle_production = []
    
    # For each particle type i, process its energy distribution locators
    for i in range(1, num_particle_types + 1):
        if debug:
            logger.debug(f"Processing energy distribution locators for particle type {i}")
            
        # 1. Get the number of MT values for this particle type
        mt_count_idx = particle_pointer_idx + (i - 1)
        
        if mt_count_idx >= len(ace.xss_data):
            raise ValueError(f"MT count index out of bounds for particle type {i}: {mt_count_idx} >= {len(ace.xss_data)}")
        
        num_mt_values = int(ace.xss_data[mt_count_idx].value)
        
        if debug:
            logger.debug(f"Particle type {i} has {num_mt_values} MT values")
            
        # 2. Get the LDLWH pointer from the particle's data block
        # LDLWH pointer is at XSS(JXS(32)+10*(i-1)+7)
        ldlwh_pointer_idx = jxs32_idx + 10 * (i - 1) + 7 
        
        if ldlwh_pointer_idx >= len(ace.xss_data):
            raise ValueError(f"LDLWH pointer index out of bounds for particle type {i}: {ldlwh_pointer_idx} >= {len(ace.xss_data)}")
        
        # Get the actual pointer value
        ldlwh_pointer = int(ace.xss_data[ldlwh_pointer_idx].value)
        
        if debug:
            logger.debug(f"LDLWH pointer for particle type {i}: {ldlwh_pointer}")
            
        # 3. Special case: If LDLWH pointer is 0, there's no energy distribution data for this particle
        if ldlwh_pointer <= 0:
            ace.energy_distribution_locators.particle_production.append([])
            if debug:
                logger.debug(f"No energy distribution data for particle type {i}")
            continue
        
        # 4. Convert to 0-indexed for accessing the XSS array
        ldlwh_pointer = ldlwh_pointer - 1
        
        # 5. Make sure we have enough data for all MT values
        if ldlwh_pointer + num_mt_values > len(ace.xss_data):
            raise ValueError(f"LDLWH block truncated for particle type {i}: need {num_mt_values} entries, but only {len(ace.xss_data) - ldlwh_pointer} available")
        
        # 6. Read the locators for this particle type
        # Each locator points to the energy distribution for the corresponding MT number
        # Store XssEntry objects directly
        locators = ace.xss_data[ldlwh_pointer:ldlwh_pointer + num_mt_values]
        ace.energy_distribution_locators.particle_production.append(locators)
        
        if debug:
            logger.debug(f"Read {len(locators)} energy distribution locators for particle type {i}")


def read_dnedl_block(ace: Ace, dnedl_idx: int, num_precursors: int, debug: bool = False) -> None:
    """
    Read the DNEDL block for delayed neutron energy distribution locators.
    
    Parameters
    ----------
    ace : Ace
        The Ace object to update
    dnedl_idx : int
        Starting index of the DNEDL block in the XSS array
    num_precursors : int
        Number of delayed neutron precursor groups (NXS(8))
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Raises
    ------
    ValueError
        If the DNEDL block data is missing or invalid
    """
    if debug:
        logger.debug(f"Reading DNEDL block at index {dnedl_idx} for {num_precursors} precursor groups")
        
    if dnedl_idx <= 0:
        raise ValueError(f"Invalid DNEDL block index: {dnedl_idx}")
    
    if dnedl_idx >= len(ace.xss_data):
        raise ValueError(f"DNEDL block index out of bounds: {dnedl_idx} >= {len(ace.xss_data)}")
    
    # Make sure we have enough data
    if dnedl_idx + num_precursors > len(ace.xss_data):
        raise ValueError(f"DNEDL block truncated: need {num_precursors} entries, but only {len(ace.xss_data) - dnedl_idx} available")
    
    # Read the delayed neutron energy distribution locators
    # Store XssEntry objects directly
    if ace.energy_distribution_locators is None:
        from mcnpy.ace.classes.energy_distribution_locators import EnergyDistributionLocators
        ace.energy_distribution_locators = EnergyDistributionLocators()
        
    ace.energy_distribution_locators.delayed_neutron = ace.xss_data[dnedl_idx:dnedl_idx + num_precursors]
    
    if debug:
        logger.debug(f"Read {len(ace.energy_distribution_locators.delayed_neutron)} delayed neutron energy distribution locators")
