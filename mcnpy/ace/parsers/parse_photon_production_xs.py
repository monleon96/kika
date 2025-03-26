import logging
from typing import List, Optional
from mcnpy.ace.classes.ace import Ace
from mcnpy.ace.classes.photon_production_xs import (
    PhotonProductionCrossSections, ParticleProductionCrossSections,
    YieldBasedCrossSection, DirectCrossSection
)

# Setup logger
logger = logging.getLogger(__name__)

def read_production_xs_blocks(ace: Ace, debug=False) -> None:
    """
    Read the SIGP and SIGH blocks containing production cross section data.
    
    Parameters
    ----------
    ace : Ace
        The Ace object to update with production cross section data
    debug : bool, optional
        Whether to print debug information, defaults to False
    """
    if debug:
        logger.debug("\n===== PRODUCTION CROSS SECTION BLOCKS PARSING =====")
        logger.debug(f"Header info: ZAID={ace.header.zaid}")
    
    # Initialize containers if not already present
    if not hasattr(ace, "photon_production_xs"):
        ace.photon_production_xs = PhotonProductionCrossSections()
    
    if not hasattr(ace, "particle_production_xs"):
        ace.particle_production_xs = ParticleProductionCrossSections()
    
    # Read photon production cross sections (SIGP block)
    read_photon_production_xs(ace, debug)
    
    # Read particle production cross sections (SIGH block)
    read_particle_production_xs(ace, debug)

def read_photon_production_xs(ace: Ace, debug=False) -> None:
    """
    Read photon production cross section data (SIGP block).
    
    Parameters
    ----------
    ace : Ace
        The Ace object to update
    debug : bool, optional
        Whether to print debug information, defaults to False
    """
    # Check if we have the necessary data
    if (not ace.header or not ace.header.jxs_array or not ace.header.nxs_array or 
        not ace.xss_data or not ace.reaction_mt_data or not ace.reaction_mt_data.photon_production):
        if debug:
            logger.debug("Skipping SIGP block: required data missing")
        return
    
    if debug:
        logger.debug("\n----- SIGP Block -----")
    
    # Get block parameters from Table 56
    sigp_idx = ace.header.jxs_array[15]  # JXS(15)
    nmt = ace.header.nxs_array[6]  # NXS(6)
    
    if debug:
        logger.debug(f"JXS(15) = {sigp_idx} → Starting index of SIGP block")
        logger.debug(f"NXS(6) = {nmt} → Number of photon production reactions")
    
    if sigp_idx <= 0 or nmt <= 0:
        if debug:
            logger.debug(f"No SIGP block present: JXS(15)={sigp_idx}, NXS(6)={nmt}")
        return
    
    # Convert to 0-indexed
    sigp_idx -= 1
    
    if debug:
        logger.debug(f"SIGP block starts at index {sigp_idx} (0-indexed)")
    
    if sigp_idx >= len(ace.xss_data):
        if debug:
            logger.debug(f"ERROR: SIGP index {sigp_idx} is out of bounds ({len(ace.xss_data)})")
        return
    
    # Get locators from LSIGP block
    if not ace.xs_locators or not ace.xs_locators.photon_production:
        if debug:
            logger.debug("No photon production locators available")
        return
    
    # Get MT numbers from MTRP block
    mts = ace.reaction_mt_data.photon_production
    
    if debug:
        logger.debug(f"Processing {len(mts)} photon production reactions")
    
    # Process each reaction
    for i, (mt, loc) in enumerate(zip(mts, ace.xs_locators.photon_production)):
        mt_value = int(mt.value)
        loc_value = int(loc.value)
        
        # Adjust locator to be relative to SIGP
        loc_value = loc_value - 1  # Adjust to 0-indexed
        
        if debug:
            logger.debug(f"\nReaction {i+1}: MT={mt_value}")
            logger.debug(f"  Locator value: {loc_value}")
            logger.debug(f"  Relative index: {loc_value}")
        
        # Check bounds
        if sigp_idx + loc_value >= len(ace.xss_data):
            if debug:
                logger.debug(f"  ERROR: Index {sigp_idx + loc_value} is out of bounds ({len(ace.xss_data)})")
            continue
        
        # Read the cross section data based on MFTYPE
        mftype_entry = ace.xss_data[sigp_idx + loc_value]
        mftype = int(mftype_entry.value)
        
        if debug:
            logger.debug(f"  MFTYPE = {mftype}")
        
        if mftype in (12, 16):
            if debug:
                logger.debug(f"  Processing yield-based XS (MFTYPE={mftype})")
            xs = read_yield_based_xs(ace.xss_data, sigp_idx + loc_value, mt_value, mftype, debug)
            if xs:
                ace.photon_production_xs.cross_sections[mt_value] = xs
                ace.photon_production_xs.has_data = True
                if debug:
                    logger.debug(f"  Successfully read yield-based XS for MT={mt_value}")
        elif mftype == 13:
            if debug:
                logger.debug("  Processing direct XS (MFTYPE=13)")
            xs = read_direct_xs(ace.xss_data, sigp_idx + loc_value, mt_value, debug)
            if xs:
                ace.photon_production_xs.cross_sections[mt_value] = xs
                ace.photon_production_xs.has_data = True
                if debug:
                    logger.debug(f"  Successfully read direct XS for MT={mt_value}")
        else:
            if debug:
                logger.debug(f"  Unsupported MFTYPE: {mftype}")

def read_particle_production_xs(ace: Ace, debug=False) -> None:
    """
    Read particle production cross section data (SIGH block).
    
    Parameters
    ----------
    ace : Ace
        The Ace object to update
    debug : bool, optional
        Whether to print debug information, defaults to False
    """
    # Check if we have the necessary data
    if (not ace.header or not ace.header.jxs_array or 
        not ace.xss_data or not ace.reaction_mt_data or not ace.reaction_mt_data.particle_production):
        if debug:
            logger.debug("Skipping SIGH block: required data missing")
        return
    
    if debug:
        logger.debug("\n----- SIGH Block -----")
    
    # Check if we have particle types
    if ace.header.nxs_array[13] <= 0:  # NXS(13) - Number of particle types
        if debug:
            logger.debug(f"No particle types: NXS(13)={ace.header.nxs_array[13]}")
        return
    
    n_types = ace.header.nxs_array[13]
    
    if debug:
        logger.debug(f"Number of particle types: {n_types}")
    
    # Iterate through each particle type
    for i in range(1, n_types + 1):
        if debug:
            logger.debug(f"\nProcessing particle type {i}:")
        
        # Get block parameters from Table 56
        jxs31_idx = ace.header.jxs_array[31]  # JXS(31)
        jxs32_idx = ace.header.jxs_array[32]  # JXS(32)
        
        if debug:
            logger.debug(f"  JXS(31) = {jxs31_idx}")
            logger.debug(f"  JXS(32) = {jxs32_idx}")
        
        if jxs31_idx < 0 or jxs32_idx < 0 or jxs31_idx >= len(ace.xss_data) or jxs32_idx >= len(ace.xss_data):
            if debug:
                logger.debug(f"  ERROR: Invalid indices: JXS(31)={jxs31_idx+1}, JXS(32)={jxs32_idx+1}")
            continue
        
        # Get NMT value for this particle type
        nmt_idx = jxs31_idx + i
        
        if debug:
            logger.debug(f"  NMT index: jxs31_idx + i - 1 = {jxs31_idx} + {i} - 1 = {nmt_idx}")
        
        if nmt_idx >= len(ace.xss_data):
            if debug:
                logger.debug(f"  ERROR: NMT index {nmt_idx} is out of bounds ({len(ace.xss_data)})")
            continue
        
        nmt_entry = ace.xss_data[nmt_idx]
        nmt = int(nmt_entry.value)
        
        if debug:
            logger.debug(f"  NMT = {nmt} → Number of reactions for this particle")
        
        # Get SIG value for this particle type
        sig_idx = jxs32_idx + 10 * (i - 1) + 4  # Position 4 in JXS array
        
        if debug:
            logger.debug(f"  SIG index: jxs32_idx + 10*(i-1) + 3 = {jxs32_idx} + 10*({i}-1) + 3 = {sig_idx}")
        
        if sig_idx >= len(ace.xss_data):
            if debug:
                logger.debug(f"  ERROR: SIG index {sig_idx} is out of bounds ({len(ace.xss_data)})")
            continue
        
        # Get locators and MTs for this particle type
        if not ace.xs_locators or not ace.xs_locators.particle_production or i > len(ace.xs_locators.particle_production):
            if debug:
                logger.debug("  No XS locators available for this particle type")
            continue
        
        locators = ace.xs_locators.particle_production[i-1]
        
        if not ace.reaction_mt_data or not ace.reaction_mt_data.particle_production or i > len(ace.reaction_mt_data.particle_production):
            if debug:
                logger.debug("  No MT numbers available for this particle type")
            continue
        
        mts = ace.reaction_mt_data.particle_production[i-1]
        
        if not mts or not locators or len(mts) != len(locators):
            if debug:
                logger.debug(f"  Invalid data: mts={len(mts) if mts else 0}, locators={len(locators) if locators else 0}")
            continue
        
        if debug:
            logger.debug(f"  Processing {len(mts)} reactions for particle type {i}")
        
        # Store the MT numbers for this particle type
        ace.particle_production_xs.particle_types[i] = mts.copy()
        
        # Process each reaction for this particle type
        for j, (mt, loc) in enumerate(zip(mts, locators)):
            mt_value = int(mt.value)
            loc_value = int(loc.value)
            
            if debug:
                logger.debug(f"\n  Reaction {j+1}: MT={mt_value}")
                logger.debug(f"    Locator value: {loc_value}")
                logger.debug(f"    Relative index: {loc_value}")
            
            # Check bounds
            if sig_idx + loc_value >= len(ace.xss_data):
                if debug:
                    logger.debug(f"    ERROR: Index {sig_idx + loc_value} is out of bounds ({len(ace.xss_data)})")
                continue
            
            # Read the cross section data based on MFTYPE
            mftype_entry = ace.xss_data[sig_idx + loc_value]
            mftype = int(mftype_entry.value)
            
            if debug:
                logger.debug(f"    MFTYPE = {mftype}")
            
            if mftype in (12, 16):
                if debug:
                    logger.debug(f"    Processing yield-based XS (MFTYPE={mftype})")
                xs = read_yield_based_xs(ace.xss_data, sig_idx + loc_value, mt_value, mftype, debug)
                if xs:
                    ace.particle_production_xs.cross_sections[mt_value] = xs
                    ace.particle_production_xs.has_data = True
                    if debug:
                        logger.debug(f"    Successfully read yield-based XS for MT={mt_value}")
            else:
                if debug:
                    logger.debug(f"    Unsupported MFTYPE: {mftype}")

def read_yield_based_xs(xss: List, start_idx: int, mt: int, mftype: int, debug=False) -> Optional[YieldBasedCrossSection]:
    """
    Read yield-based cross section data (MFTYPE = 12 or 16).
    
    Parameters
    ----------
    xss : List
        The XSS array containing XssEntry objects
    start_idx : int
        Starting index in the XSS array (where MFTYPE is located)
    mt : int
        MT number for this reaction
    mftype : int
        MFTYPE value (12 or 16)
    debug : bool, optional
        Whether to print debug information, defaults to False
    
    Returns
    -------
    Optional[YieldBasedCrossSection]
        The parsed cross section data or None if there was an error
    """
    if start_idx + 1 >= len(xss):
        if debug:
            logger.debug(f"    ERROR: Index {start_idx + 1} is out of bounds ({len(xss)})")
        return None
    
    xs = YieldBasedCrossSection(mt=mt, mftype=mftype)
    
    # Read MTMULT
    mtmult_entry = xss[start_idx + 1]
    xs.mtmult = int(mtmult_entry.value)
    
    if debug:
        logger.debug(f"    MTMULT = {xs.mtmult}")
    
    # Read number of interpolation regions
    num_regions_entry = xss[start_idx + 2]
    xs.num_regions = int(num_regions_entry.value)
    
    if debug:
        logger.debug(f"    Number of interpolation regions: {xs.num_regions}")
    
    current_idx = start_idx + 3
    
    # Read interpolation data if present
    if xs.num_regions > 0:
        if current_idx + 2 * xs.num_regions - 1 >= len(xss):
            if debug:
                logger.debug(f"    ERROR: Not enough data for interpolation regions")
            return None
        
        # Read NBT array - store XssEntry objects
        xs.interpolation_bounds = [xss[current_idx + i] for i in range(xs.num_regions)]
        current_idx += xs.num_regions
        
        # Read INT array - store XssEntry objects
        xs.interpolation_schemes = [xss[current_idx + i] for i in range(xs.num_regions)]
        current_idx += xs.num_regions
        
        if debug:
            bounds = [int(b.value) for b in xs.interpolation_bounds]
            schemes = [int(s.value) for s in xs.interpolation_schemes]
            logger.debug(f"    Interpolation bounds: {bounds}")
            logger.debug(f"    Interpolation schemes: {schemes}")
    
    # Read number of energies
    num_energies_entry = xss[current_idx]
    xs.num_energies = int(num_energies_entry.value)
    current_idx += 1
    
    if debug:
        logger.debug(f"    Number of energy points: {xs.num_energies}")
    
    # Check bounds
    if current_idx + 2 * xs.num_energies - 1 >= len(xss):
        if debug:
            logger.debug(f"    ERROR: Not enough data for energy/yield points")
        return None
    
    # Read energies - store XssEntry objects
    xs.energies = [xss[current_idx + i] for i in range(xs.num_energies)]
    current_idx += xs.num_energies
    
    # Read yields - store XssEntry objects
    xs.yields = [xss[current_idx + i] for i in range(xs.num_energies)]
    
    if debug:
        logger.debug(f"    Successfully read {xs.num_energies} energy/yield points")
    
    return xs

def read_direct_xs(xss: List, start_idx: int, mt: int, debug=False) -> Optional[DirectCrossSection]:
    """
    Read direct cross section data (MFTYPE = 13).
    
    Parameters
    ----------
    xss : List
        The XSS array containing XssEntry objects
    start_idx : int
        Starting index in the XSS array (where MFTYPE is located)
    mt : int
        MT number for this reaction
    debug : bool, optional
        Whether to print debug information, defaults to False
    
    Returns
    -------
    Optional[DirectCrossSection]
        The parsed cross section data or None if there was an error
    """
    if start_idx + 2 >= len(xss):
        if debug:
            logger.debug(f"    ERROR: Index {start_idx + 2} is out of bounds ({len(xss)})")
        return None
    
    xs = DirectCrossSection(mt=mt, mftype=13)
    
    # Read energy grid index
    energy_grid_idx_entry = xss[start_idx + 1]
    xs.energy_grid_index = int(energy_grid_idx_entry.value)
    
    if debug:
        logger.debug(f"    Energy grid index: {xs.energy_grid_index}")
    
    # Read number of entries
    num_entries_entry = xss[start_idx + 2]
    xs.num_entries = int(num_entries_entry.value)
    
    if debug:
        logger.debug(f"    Number of XS entries: {xs.num_entries}")
    
    # Check bounds
    if start_idx + 3 + xs.num_entries - 1 >= len(xss):
        if debug:
            logger.debug(f"    ERROR: Not enough data for XS values")
        return None
    
    # Read cross section values - store XssEntry objects
    xs.cross_sections = [xss[start_idx + 3 + i] for i in range(xs.num_entries)]
    
    if debug:
        logger.debug(f"    Successfully read {xs.num_entries} XS values")
    
    return xs
