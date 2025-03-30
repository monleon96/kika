from typing import List, Optional, Dict
from mcnpy.ace.classes.ace import Ace
from mcnpy.ace.classes.energy_distribution.energy_distribution import EnergyDistribution
from mcnpy.ace.classes.energy_distribution.energy_distribution_container import EnergyDistributionContainer
from mcnpy.ace.parsers.laws import (
    parse_tabular_energy_distribution,
    parse_discrete_energy_distribution,
    parse_level_scattering,
    parse_continuous_energy_angle_distribution,
    parse_general_evaporation_spectrum,
    parse_maxwell_fission_spectrum,
    parse_evaporation_spectrum,
    parse_energy_dependent_watt_spectrum,
    parse_tabular_linear_functions,
    parse_tabular_energy_multipliers,
    parse_kalbach_mann_distribution,
    parse_tabulated_angle_energy_distribution,
    parse_nbody_phase_space_distribution,
    parse_laboratory_angle_energy_distribution,
    parse_energy_dependent_yield
)
from mcnpy.ace.parsers.xss import XssEntry
import logging

# Setup logger
logger = logging.getLogger(__name__)

def read_energy_distribution_blocks(ace: Ace, debug: bool = False) -> EnergyDistributionContainer:
    """
    Read the DLW, DLWP, DLWH, and DNED blocks from the ACE file.
    
    These blocks contain energy distributions for secondary particles:
    - DLW: For incident neutron reactions (secondary neutrons)
    - DLWP: For photon production reactions
    - DLWH: For other particle production reactions
    - DNED: For delayed neutron precursor groups
    
    Parameters
    ----------
    ace : Ace
        The Ace object containing the XSS data and headers
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    EnergyDistributionContainer
        Container with the energy distribution data
        
    Raises
    ------
    ValueError
        If required data is missing or indices are invalid
    """
    if debug: 
        logger.debug("\n===== ENERGY DISTRIBUTION BLOCKS PARSING =====")
        logger.debug(f"Header info: ZAID={ace.header.zaid}")
        
    if (ace.header is None or ace.header.jxs_array is None or 
        ace.header.nxs_array is None or ace.xss_data is None or
        ace.energy_distribution_locators is None):
        raise ValueError("Cannot read energy distribution blocks: header, XSS data, or locators missing")
    
    # Create a new container for the return value
    result = EnergyDistributionContainer()
    
    # Get the JXS values for the energy distribution blocks
    jxs_dlw = ace.header.jxs_array[11]    # JXS(11)
    jxs_dlwp = ace.header.jxs_array[19]   # JXS(19)
    jxs_dned = ace.header.jxs_array[27]   # JXS(27)
    
    if debug:
        logger.debug(f"JXS(11) = {jxs_dlw} → DLW block for incident neutron reactions")
        logger.debug(f"JXS(19) = {jxs_dlwp} → DLWP block for photon production reactions")
        logger.debug(f"JXS(27) = {jxs_dned} → DNED block for delayed neutron precursors")
    
    # Store the container reference to be used by the helper functions
    _process_energy_distributions(ace, result, jxs_dlw, jxs_dlwp, jxs_dned, debug)
    
    return result


def _process_energy_distributions(ace: Ace, result: EnergyDistributionContainer, 
                                 jxs_dlw: int, jxs_dlwp: int, jxs_dned: int, debug: bool = False) -> None:
    """
    Process all energy distribution blocks and fill the result container.
    
    Parameters
    ----------
    ace : Ace
        The Ace object with XSS data and headers
    result : EnergyDistributionContainer
        Container to be filled with energy distribution data
    jxs_dlw : int
        JXS(11) - Starting index of the DLW block 
    jxs_dlwp : int
        JXS(19) - Starting index of the DLWP block
    jxs_dned : int
        JXS(27) - Starting index of the DNED block
    debug : bool, optional
        Whether to print debug information, defaults to False
    """
    # 1. Read DLW block (neutron energy distributions)
    if ace.energy_distribution_locators.has_neutron_data and jxs_dlw > 0:
        if debug: logger.debug(f"Reading DLW block starting at index {jxs_dlw}")
        read_dlw_block(ace, result, jxs_dlw, debug=debug)
    elif debug: logger.debug("Skipping DLW block (no neutron data or invalid JXS index)")
    
    # 2. Read DLWP block (photon production energy distributions)
    if ace.energy_distribution_locators.has_photon_production_data and jxs_dlwp > 0:
        if debug: logger.debug(f"Reading DLWP block starting at index {jxs_dlwp}")
        read_dlwp_block(ace, result, jxs_dlwp, debug=debug)
    elif debug: logger.debug("Skipping DLWP block (no photon production data or invalid JXS index)")
    
    # 3. Read DLWH block (other particle production energy distributions)
    if ace.energy_distribution_locators.has_particle_production_data:
        if debug: logger.debug("Reading DLWH block")
        read_dlwh_block(ace, result, debug=debug)
    elif debug: logger.debug("Skipping DLWH block (no particle production data)")
    
    # 4. Read DNED block (delayed neutron energy distributions)
    if ace.energy_distribution_locators.has_delayed_neutron_data and jxs_dned > 0:
        if debug: logger.debug(f"Reading DNED block starting at index {jxs_dned}")
        read_dned_block(ace, result, jxs_dned, debug=debug)
    elif debug: logger.debug("Skipping DNED block (no delayed neutron data or invalid JXS index)")
    
    if debug: logger.debug("Finished read_energy_distribution_blocks")


def read_dlw_block(ace: Ace, result: EnergyDistributionContainer, jxs_dlw: int, debug: bool = False) -> None:
    """
    Read the DLW block for neutron energy distributions.
    
    Parameters
    ----------
    ace : Ace
        The Ace object with XSS data and headers
    result : EnergyDistributionContainer
        Container to be filled with energy distribution data
    jxs_dlw : int
        JXS(11) - Starting index of the DLW block in the XSS array
    debug : bool, optional
        Whether to print debug information, defaults to False
    """
    if debug: logger.debug(f"Starting read_dlw_block at index {jxs_dlw}")
    # Get MT numbers and corresponding locators from the previously read LDLW block
    if not ace.reaction_mt_data or not ace.reaction_mt_data.has_neutron_mt_data:
        if debug: logger.debug("Exiting read_dlw_block: no neutron MT data available")
        return
    
    # Get the MT numbers for reactions with secondary neutrons
    neutron_mts = ace.reaction_mt_data.secondary_neutron_mt
    if debug: 
        logger.debug(f"Found {len(neutron_mts)} secondary neutron MT numbers")
        # Print the first few MT values
        sample_size = min(3, len(neutron_mts))
        mt_sample = [int(neutron_mts[i].value) if hasattr(neutron_mts[i], 'value') else neutron_mts[i] for i in range(sample_size)]
        logger.debug(f"First {sample_size} MT values: {mt_sample}")
    
    # Get the corresponding locators from the energy distribution locators
    locators = []
    
    # Extract integer values from XssEntry objects before creating the map
    mt_idx_map = {}
    for i, mt_entry in enumerate(ace.reaction_mt_data.incident_neutron):
        mt_value = int(mt_entry.value) if hasattr(mt_entry, 'value') else int(mt_entry)
        mt_idx_map[mt_value] = i
    
    if debug:
        logger.debug(f"Created MT index map with {len(mt_idx_map)} entries")
        logger.debug(f"Total incident_neutron MT entries: {len(ace.reaction_mt_data.incident_neutron)}")
        logger.debug(f"Total energy_distribution_locators: {len(ace.energy_distribution_locators.incident_neutron)}")
    
    # Debugging: print the full incident_neutron and LDLW arrays length
    if debug:
        logger.debug(f"Incident neutron MTs array length: {len(ace.reaction_mt_data.incident_neutron)}")
        logger.debug(f"LDLW locators array length: {len(ace.energy_distribution_locators.incident_neutron)}")
    
    for i, mt_item in enumerate(neutron_mts):
        # Make sure we're using integer MT values
        mt_value = int(mt_item.value) if hasattr(mt_item, 'value') else int(mt_item)
        
        if mt_value in mt_idx_map and mt_idx_map[mt_value] < len(ace.energy_distribution_locators.incident_neutron):
            locator_entry = ace.energy_distribution_locators.incident_neutron[mt_idx_map[mt_value]]
            locators.append(locator_entry)
            if debug:
                locator_value = int(locator_entry.value) if hasattr(locator_entry, 'value') else int(locator_entry)
                logger.debug(f"MT={mt_value}, locator={locator_value}, absolute index={jxs_dlw + locator_value - 1 if locator_value > 0 else 'N/A'}")
                # Print additional info about the mapping
                logger.debug(f"  MT index in incident_neutron: {mt_idx_map[mt_value]}")
                logger.debug(f"  Locator entry: {locator_entry.index}:{locator_entry.value}")
        elif debug:
            logger.debug(f"Could not find locator for MT={mt_value}")
            if mt_value in mt_idx_map:
                logger.debug(f"  MT index in map: {mt_idx_map[mt_value]}")
                logger.debug(f"  But this index is >= {len(ace.energy_distribution_locators.incident_neutron)}")
            else:
                logger.debug(f"  MT not found in incident_neutron array")
    
    # Make sure we have the same number of MT numbers and locators
    num_secondary_neutron_reactions = ace.header.nxs_array[4]  # NXS(5) in 1-indexed notation
    if debug:
        logger.debug(f"NXS(5)={num_secondary_neutron_reactions}, found MT numbers: {len(neutron_mts)}, locators: {len(locators)}")
        if len(neutron_mts) != len(locators):
            logger.debug(f"Warning: Mismatch between MT numbers and locators count")
    
    # Process each MT number with its corresponding locator
    for i, mt_item in enumerate(neutron_mts):
        if i >= len(locators):
            if debug: logger.debug(f"Skipping MT={mt_item}, index {i} out of range for locators array")
            continue
        
        mt_value = int(mt_item.value) if hasattr(mt_item, 'value') else int(mt_item)
            
        locator_entry = locators[i]
        locator_value = int(locator_entry.value) if hasattr(locator_entry, 'value') else int(locator_entry)
        
        if locator_value <= 0:
            if debug: logger.debug(f"Skipping MT={mt_value}, invalid locator: {locator_value}")
            continue  # Skip if the locator is invalid
        
        # According to Table 30, the energy distribution starts at JED + LOCC - 1
        offset = jxs_dlw + locator_value - 1  # Subtract 1 to match the documentation
        
        if debug: logger.debug(f"Processing MT={mt_value}, locator={locator_value}, offset={offset} (XSS length: {len(ace.xss_data)})")
        
        if offset >= len(ace.xss_data):
            if debug: logger.debug(f"Warning: Offset {offset} out of range for XSS data of length {len(ace.xss_data)}")
            continue
        
        try:
            if debug: logger.debug(f"Calling read_energy_distribution for MT={mt_value}")
            distributions = read_energy_distribution(ace, offset, debug=debug)
            if distributions:
                if debug: logger.debug(f"Successfully read {len(distributions)} distributions for MT={mt_value}")
                result.incident_neutron[mt_value] = distributions
                if debug: logger.debug(f"Stored distributions for MT={mt_value} (type: {type(mt_value)})")
            elif debug: logger.debug(f"No distributions found for MT={mt_value}")
        except ValueError as e:
            if debug: logger.debug(f"Error parsing energy distribution for MT={mt_value}: {e}")
        except IndexError as e:
            if debug:
                logger.debug(f"Index error parsing energy distribution for MT={mt_value}: {e}")
                logger.debug(f"Problematic offset: {offset}, locator: {locator_value}, jxs_dlw: {jxs_dlw}")
                
    # Process energy-dependent yields for neutron reactions
    if ace.particle_release and ace.particle_release.has_neutron_data:
        if debug: logger.debug("Processing energy-dependent yields for neutron reactions")
        for i, ty in enumerate(ace.particle_release.incident_neutron):
            ty_value = int(ty.value) if hasattr(ty, 'value') else int(ty)
            
            # According to documentation, yields are specified for TY values > 100 in absolute value
            if abs(ty_value) > 100 and i < len(neutron_mts):
                mt_item = neutron_mts[i]
                mt_value = int(mt_item.value) if hasattr(mt_item, 'value') else int(mt_item)
                
                # Formula from Table 52: KY = JED + |TY_i| - 101
                ky = jxs_dlw + abs(ty_value) - 101
                
                if debug: 
                    logger.debug(f"Energy-dependent yield for MT={mt_value}, TY={ty_value}")
                    logger.debug(f"KY calculation: JED + |TY_i| - 101 = {jxs_dlw} + {abs(ty_value)} - 101 = {ky}")
                
                if ky >= len(ace.xss_data):
                    if debug: logger.debug(f"Warning: KY={ky} out of range for XSS data of length {len(ace.xss_data)}")
                    continue
                
                try:
                    yield_data = parse_energy_dependent_yield(ace, ky)
                    result.neutron_yields[mt_value] = yield_data
                    if debug: 
                        logger.debug(f"Successfully parsed energy-dependent yield data for MT={mt_value}")
                        if yield_data and hasattr(yield_data, 'energies') and yield_data.energies:
                            num_points = len(yield_data.energies)
                            logger.debug(f"Yield has {num_points} energy points")
                            if num_points > 0:
                                first_e = yield_data.energies[0].value if hasattr(yield_data.energies[0], 'value') else yield_data.energies[0]
                                last_e = yield_data.energies[-1].value if hasattr(yield_data.energies[-1], 'value') else yield_data.energies[-1]
                                logger.debug(f"Energy range: {first_e} to {last_e} MeV")
                except ValueError as e:
                    if debug: logger.debug(f"Error parsing energy-dependent yield for MT={mt_value}: {e}")
                except IndexError as e:
                    if debug:
                        logger.debug(f"Index error parsing yield for MT={mt_value}: {e}")
                        logger.debug(f"Problematic KY: {ky}, TY: {ty_value}, jxs_dlw: {jxs_dlw}")
    
    if debug:
        logger.debug(f"Final incident_neutron keys: {list(result.incident_neutron.keys())}")
        logger.debug(f"Types of keys: {[type(k) for k in result.incident_neutron.keys()]}")
    
    if debug:
        logger.debug("Finished read_dlw_block")


def read_dlwp_block(ace: Ace, result: EnergyDistributionContainer, jxs_dlwp: int, debug: bool = False) -> None:
    """
    Read the DLWP block for photon energy distributions.
    
    According to Table 30, the energy distribution for reaction MT starts at:
    JED + LOCC - 1, where JED is JXS(19) and LOCC is provided by the LDLWP block.
    
    Parameters
    ----------
    ace : Ace
        The Ace object with XSS data and headers
    result : EnergyDistributionContainer
        Container to be filled with energy distribution data
    jxs_dlwp : int
        JXS(19) - Starting index of the DLWP block in the XSS array
    debug : bool, optional
        Whether to print debug information, defaults to False
    """
    if debug: logger.debug(f"Starting read_dlwp_block at index {jxs_dlwp}")
    if not ace.reaction_mt_data or not ace.reaction_mt_data.has_photon_production_mt_data:
        if debug: logger.debug("Exiting read_dlwp_block: no photon production MT data available")
        return
    
    photon_mts = ace.reaction_mt_data.photon_production
    locators = ace.energy_distribution_locators.photon_production
    
    if debug: logger.debug(f"Found {len(photon_mts)} photon MT numbers and {len(locators)} locators")
    
    if len(photon_mts) != len(locators) and debug:
        logger.debug(f"Warning: Mismatch between photon MT numbers ({len(photon_mts)}) and locators ({len(locators)})")
    
    for i, mt in enumerate(photon_mts):
        if i >= len(locators):
            if debug: logger.debug(f"Skipping MT={mt}, index {i} out of range for locators array")
            continue
            
        locator = locators[i]
        locator_value = int(locator.value) if hasattr(locator, 'value') else int(locator)
        
        # Get the actual MT value, not the XssEntry
        mt_value = int(mt.value) if hasattr(mt, 'value') else int(mt)
        
        if locator_value <= 0:
            if debug: logger.debug(f"Skipping MT={mt_value}, invalid locator: {locator_value}")
            continue
        
        # According to Table 30, the energy distribution starts at JED + LOCC - 1
        offset = jxs_dlwp + locator_value - 1  # Subtract 1 to match the documentation
        
        if debug: logger.debug(f"Processing MT={mt_value}, locator={locator_value}, offset={offset} (XSS length: {len(ace.xss_data)})")
        
        if offset >= len(ace.xss_data):
            if debug: logger.debug(f"Warning: Offset {offset} out of range for XSS data of length {len(ace.xss_data)}")
            continue
        
        try:
            if debug: logger.debug(f"Calling read_energy_distribution for MT={mt_value}")
            distributions = read_energy_distribution(ace, offset, debug=debug)
            if distributions:
                if debug: logger.debug(f"Successfully read {len(distributions)} distributions for MT={mt_value}")
                result.photon_production[mt_value] = distributions
            elif debug: logger.debug(f"No distributions found for MT={mt_value}")
        except ValueError as e:
            if debug: logger.debug(f"Error parsing photon energy distribution for MT={mt_value}: {e}")
        except IndexError as e:
            if debug:
                logger.debug(f"Index error parsing photon energy distribution for MT={mt_value}: {e}")
                logger.debug(f"Problematic offset: {offset}, locator: {locator}, jxs_dlwp: {jxs_dlwp}")
    
    # Process energy-dependent yields for photon production similarly to neutron reactions
    if ace.particle_release and ace.particle_release.has_neutron_data:
        if debug: logger.debug("Processing energy-dependent yields for photon production")
        if ace.reaction_mt_data and ace.reaction_mt_data.has_neutron_mt_data:
            neutron_mts = ace.reaction_mt_data.incident_neutron
            
            for i, ty in enumerate(ace.particle_release.incident_neutron):
                ty_value = int(ty.value) if hasattr(ty, 'value') else int(ty)
                
                # According to documentation, yields are specified for TY values > 100 in absolute value
                if abs(ty_value) > 100 and i < len(neutron_mts):
                    mt_item = neutron_mts[i]
                    mt_value = int(mt_item.value) if hasattr(mt_item, 'value') else int(mt_item)
                    
                    # Formula from Table 52: KY = JED + |TY_i| - 101 (where JED is DLWP block)
                    ky = int(jxs_dlwp + abs(ty_value) - 101)
                    
                    if debug: 
                        logger.debug(f"Energy-dependent yield for MT={mt_value}, TY={ty_value}")
                        logger.debug(f"KY calculation: JED + |TY_i| - 101 = {jxs_dlwp} + {abs(ty_value)} - 101 = {ky}")
                    
                    if ky >= len(ace.xss_data):
                        if debug: logger.debug(f"Warning: KY={ky} out of range for XSS data of length {len(ace.xss_data)}")
                        continue
                    
                    try:
                        yield_data = parse_energy_dependent_yield(ace, ky)
                        result.photon_yields[mt_value] = yield_data
                        if debug: 
                            logger.debug(f"Successfully parsed energy-dependent yield data for MT={mt_value}")
                            if yield_data and hasattr(yield_data, 'energies') and yield_data.energies:
                                num_points = len(yield_data.energies)
                                logger.debug(f"Yield has {num_points} energy points")
                                if num_points > 0:
                                    first_e = yield_data.energies[0].value if hasattr(yield_data.energies[0], 'value') else yield_data.energies[0]
                                    last_e = yield_data.energies[-1].value if hasattr(yield_data.energies[-1], 'value') else yield_data.energies[-1]
                                    logger.debug(f"Energy range: {first_e} to {last_e} MeV")
                    except ValueError as e:
                        if debug: logger.debug(f"Error parsing energy-dependent yield for MT={mt_value}: {e}")
                    except IndexError as e:
                        if debug:
                            logger.debug(f"Index error parsing yield for MT={mt_value}: {e}")
                            logger.debug(f"Problematic KY: {ky}, TY: {ty_value}, jxs_dlwp: {jxs_dlwp}")
    
    if debug:
        logger.debug("Finished read_dlwp_block")


def read_dlwh_block(ace: Ace, result: EnergyDistributionContainer, debug: bool = False) -> None:
    """
    Read the DLWH block for other particle energy distributions.
    
    According to Table 30, the energy distribution for reaction MT starts at:
    JED + LOCC - 1, where JED is XSS(JXS(32) + 10*(i-1) + 8) and LOCC is 
    provided by the LDLWH block.
    
    Note: According to the documentation, energy-dependent neutron yields are only 
    available in the DLW and DLWP blocks, not in the DLWH block.
    
    Parameters
    ----------
    ace : Ace
        The Ace object with XSS data and headers
    result : EnergyDistributionContainer
        Container to be filled with energy distribution data
    debug : bool, optional
        Whether to print debug information, defaults to False
    """
    if debug: logger.debug("Starting read_dlwh_block")
    if not ace.reaction_mt_data or not ace.reaction_mt_data.has_particle_production_mt_data:
        if debug: logger.debug("Exiting read_dlwh_block: no particle production MT data available")
        return
    
    jxs32_idx = ace.header.jxs_array[32]
    if jxs32_idx <= 0:
        if debug: logger.debug(f"Invalid JXS(32) index: {jxs32_idx}")
        return
    
    jxs31_idx = ace.header.jxs_array[31]
    if jxs31_idx <= 0:
        if debug: logger.debug(f"Invalid JXS(31) index: {jxs31_idx}")
        return
    
    if debug: logger.debug(f"JXS(31) index: {jxs31_idx}, JXS(32) index: {jxs32_idx}")
    
    num_particle_types = ace.header.num_particle_types
    if debug: logger.debug(f"Number of particle types: {num_particle_types}")
    result.particle_production = [{} for _ in range(num_particle_types)]
    
    for i in range(num_particle_types):
        if debug: logger.debug(f"Processing particle type {i+1}")
        if i >= len(ace.energy_distribution_locators.particle_production):
            if debug: logger.debug(f"Skipping particle type {i+1}: no locators available")
            continue
            
        locators = ace.energy_distribution_locators.particle_production[i]
        if not locators:
            if debug: logger.debug(f"Skipping particle type {i+1}: empty locators list")
            continue
            
        particle_mts = ace.reaction_mt_data.particle_production[i] if i < len(ace.reaction_mt_data.particle_production) else []
        if not particle_mts or len(particle_mts) != len(locators):
            if debug: logger.debug(f"Issue with particle type {i+1}: MT count={len(particle_mts)}, locator count={len(locators)}")
            continue
        
        jed_idx = jxs32_idx + 10 * i + 8 
        if jed_idx >= len(ace.xss_data):
            if debug: logger.debug(f"JED index {jed_idx} out of range for XSS data of length {len(ace.xss_data)}")
            continue
            
        jed = int(ace.xss_data[jed_idx].value)
        if debug: logger.debug(f"Particle type {i+1} JED={jed}, calculated at index {jed_idx}")
        
        for j, mt in enumerate(particle_mts):
            locator = locators[j]
            locator_value = int(locator.value) if hasattr(locator, 'value') else int(locator)
            
            # Get the actual MT value, not the XssEntry
            mt_value = int(mt.value) if hasattr(mt, 'value') else int(mt)
            
            if locator_value <= 0:
                if debug: logger.debug(f"Skipping MT={mt_value}, invalid locator: {locator_value}")
                continue
            
            # According to Table 30, the energy distribution starts at JED + LOCC - 1
            offset = jed + locator_value - 1  # Subtract 1 to match the documentation
            
            if debug: logger.debug(f"Processing MT={mt_value}, locator={locator_value}, JED={jed}, offset={offset} (XSS length: {len(ace.xss_data)})")
            
            if offset >= len(ace.xss_data):
                if debug: logger.debug(f"Warning: Offset {offset} out of range for XSS data of length {len(ace.xss_data)}")
                continue
            
            try:
                if debug: logger.debug(f"Calling read_energy_distribution for MT={mt_value}")
                distributions = read_energy_distribution(ace, offset, debug=debug)
                if distributions:
                    if debug: logger.debug(f"Successfully read {len(distributions)} distributions for MT={mt_value}")
                    result.particle_production[i][mt_value] = distributions
                elif debug: logger.debug(f"No distributions found for MT={mt_value}")
            except ValueError as e:
                if debug: logger.debug(f"Error parsing particle energy distribution for particle {i+1}, MT={mt_value}: {e}")
            except IndexError as e:
                if debug:
                    logger.debug(f"Index error parsing distribution for particle {i+1}, MT={mt_value}: {e}")
                    logger.debug(f"Problematic offset: {offset}, locator: {locator_value}, JED: {jed}")
    
    if debug:
        logger.debug("Finished read_dlwh_block")


def read_dned_block(ace: Ace, result: EnergyDistributionContainer, jxs_dned: int, debug: bool = False) -> None:
    """
    Read the DNED block for delayed neutron energy distributions.
    
    According to Table 30, the energy distribution for delayed neutron group starts at:
    JED + LOCC - 1, where JED is JXS(27) and LOCC is provided by the DNEDL block.
    
    Parameters
    ----------
    ace : Ace
        The Ace object with XSS data and headers
    result : EnergyDistributionContainer
        Container to be filled with energy distribution data
    jxs_dned : int
        JXS(27) - Starting index of the DNED block in the XSS array
    debug : bool, optional
        Whether to print debug information, defaults to False
    """
    if debug: logger.debug(f"Starting read_dned_block at index {jxs_dned}")
    if not ace.energy_distribution_locators.has_delayed_neutron_data:
        if debug: logger.debug("Exiting read_dned_block: no delayed neutron data available")
        return
    
    locators = ace.energy_distribution_locators.delayed_neutron
    num_groups = len(locators)
    if debug: logger.debug(f"Found {num_groups} delayed neutron groups")
    
    for i in range(num_groups):
        locator = locators[i]
        locator_value = int(locator.value) if hasattr(locator, 'value') else int(locator)
        
        if locator_value <= 0:
            if debug: logger.debug(f"Skipping group {i+1}, invalid locator: {locator_value}")
            continue
        
        # According to Table 30, the energy distribution starts at JED + LOCC - 1
        offset = jxs_dned + locator_value - 1  # Subtract 1 to match the documentation
        
        if debug: logger.debug(f"Processing group {i+1}, locator={locator_value}, offset={offset} (XSS length: {len(ace.xss_data)})")
        
        if offset >= len(ace.xss_data):
            if debug: logger.debug(f"Warning: Offset {offset} out of range for XSS data of length {len(ace.xss_data)}")
            continue
        
        try:
            if debug: logger.debug(f"Calling read_energy_distribution for group {i+1}")
            distributions = read_energy_distribution(ace, offset, debug=debug)
            if distributions and len(distributions) > 0:
                result.delayed_neutron.append(distributions[0])
                if debug: logger.debug(f"Successfully read distribution for group {i+1}")
            elif debug: logger.debug(f"No distributions found for group {i+1}")
        except ValueError as e:
            if debug: logger.debug(f"Error parsing delayed neutron energy distribution for group {i+1}: {e}")
        except IndexError as e:
            if debug:
                logger.debug(f"Index error parsing distribution for group {i+1}: {e}")
                logger.debug(f"Problematic offset: {offset}, locator: {locator_value}, jxs_dned: {jxs_dned}")
    
    if debug: logger.debug("Finished read_dned_block")


def read_energy_distribution(ace: Ace, offset: int, debug: bool = False) -> List[EnergyDistribution]:
    """
    Read energy distribution data starting at the given offset in the XSS array.
    
    According to Table 31, each energy distribution consists of:
    - LNW: Locator for the next law (0 if this is the last one)
    - LAW: Law number for the distribution
    - IDAT: Locator for the law data (relative to JED)
    - NR: Number of interpolation regions to define law applicability
    - NBT, INT arrays: ENDF interpolation parameters
    - NE: Number of energies for law applicability
    - E, P arrays: Energy points and probability for law validity
    
    For multiple laws:
    - If LNW=0, this is the last law (always used regardless of other conditions)
    - For a given incident energy E, LAW is used only if random_number < P(E)
    - If E < E(1), use P(1)
    - If E > E(NE), use P(NE)
    
    Parameters
    ----------
    ace : Ace
        The Ace object containing the XSS array
    offset : int
        Starting index in the XSS array (JED + LOCC - 1)
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    List[EnergyDistribution]
        List of energy distribution objects
    """
    distributions = []
    jed = offset  # Used as the reference for all offsets in this function
    
    current_offset = offset
    while current_offset < len(ace.xss_data):
        lnw_entry = ace.xss_data[current_offset]
        law_entry = ace.xss_data[current_offset + 1]
        idat_entry = ace.xss_data[current_offset + 2]
        
        lnw = int(lnw_entry.value)
        law = int(law_entry.value)
        idat = int(idat_entry.value)
        
        n_r_entry = ace.xss_data[current_offset + 3]
        n_r = int(n_r_entry.value)
        
        idx = current_offset + 4
        
        nbt = []
        interp = []
        if n_r > 0:
            nbt = [int(ace.xss_data[idx + i].value) for i in range(n_r)]
            idx += n_r
            interp = [int(ace.xss_data[idx + i].value) for i in range(n_r)]
            idx += n_r
        
        n_e_entry = ace.xss_data[idx]
        n_e = int(n_e_entry.value)
        idx += 1
        
        energies = []
        probabilities = []
        if n_e > 0:
            energies = ace.xss_data[idx:idx + n_e]
            idx += n_e
            probabilities = ace.xss_data[idx:idx + n_e]
            idx += n_e
        
        idat_absolute = jed + idat 
        
        if debug: 
            logger.debug(f"Law {law}, LNW={lnw}, IDAT={idat}, IDAT_abs={idat_absolute}, N_R={n_r}, N_E={n_e}")
            if n_r > 0:
                logger.debug(f"NBT={nbt}, INT={interp}")
            if n_e > 0:
                logger.debug(f"Energy points: {[entry.value for entry in energies]}")
                logger.debug(f"Probabilities: {[entry.value for entry in probabilities]}")
        
        distribution = create_energy_distribution(
            ace, law, idat, idat_absolute, 
            energies, probabilities, nbt, interp,
            debug=debug
        )
        
        if distribution:
            distributions.append(distribution)
            if debug: logger.debug(f"Successfully created distribution for law {law}")
        elif debug: logger.debug(f"Failed to create distribution for law {law}")
        
        if lnw == 0:
            if debug: logger.debug("Reached last law (LNW=0)")
            break
            
        current_offset = jed + lnw
        if debug:  logger.debug(f"Moving to next law at offset {current_offset}")
    
    if debug: logger.debug(f"Finished reading distributions, found {len(distributions)} laws")
    return distributions


def create_energy_distribution(
    ace: Ace, law: int, idat: int, idat_absolute: int, 
    applicability_energies: List[XssEntry], 
    applicability_probabilities: List[XssEntry],
    nbt: List[int], interp: List[int],
    debug: bool = False
) -> Optional[EnergyDistribution]:
    """
    Create an energy distribution object based on the law number.
    
    Parameters
    ----------
    ace : Ace
        The Ace object containing the XSS array
    law : int
        Law number for the distribution
    idat : int
        Locator for the distribution data (relative to JED)
    idat_absolute : int
        Absolute index in XSS array for the law data 
    applicability_energies : List[XssEntry]
        Energy points for law applicability
    applicability_probabilities : List[XssEntry]
        Probability of law validity at each energy point
    nbt : List[int]
        NBT interpolation parameters
    interp : List[int]
        INT interpolation scheme parameters
    debug: bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    Optional[EnergyDistribution]
        Energy distribution object, or None if the law is not supported
    """
    base_distribution = EnergyDistribution(
        law=law, 
        idat=idat
    )
    base_distribution.applicability_energies = applicability_energies
    base_distribution.applicability_probabilities = applicability_probabilities
    base_distribution.nbt = nbt
    base_distribution.interp = interp
    
    if debug: logger.debug(f"Creating energy distribution for law {law}")
    
    if law == 1:
        if debug: logger.debug("Parsing tabular energy distribution (Law 1)")
        return parse_tabular_energy_distribution(ace, base_distribution, idat_absolute)
    elif law == 2:
        if debug: logger.debug("Parsing discrete energy distribution (Law 2)")
        return parse_discrete_energy_distribution(ace, base_distribution, idat_absolute)
    elif law == 3:
        if debug: logger.debug("Parsing level scattering (Law 3)")
        return parse_level_scattering(ace, base_distribution, idat_absolute)
    elif law == 4:
        if debug: logger.debug("Parsing continuous energy-angle distribution (Law 4)")
        return parse_continuous_energy_angle_distribution(ace, base_distribution, idat_absolute)
    elif law == 5:
        if debug: logger.debug("Parsing general evaporation spectrum (Law 5)")
        return parse_general_evaporation_spectrum(ace, base_distribution, idat_absolute)
    elif law == 7:
        if debug: logger.debug("Parsing Maxwell fission spectrum (Law 7)")
        return parse_maxwell_fission_spectrum(ace, base_distribution, idat_absolute)
    elif law == 9:
        if debug:  logger.debug("Parsing evaporation spectrum (Law 9)")
        return parse_evaporation_spectrum(ace, base_distribution, idat_absolute)
    elif law == 11:
        if debug: logger.debug("Parsing energy-dependent Watt spectrum (Law 11)")
        return parse_energy_dependent_watt_spectrum(ace, base_distribution, idat_absolute)
    elif law == 22:
        if debug: logger.debug("Parsing tabular linear functions (Law 22)")
        return parse_tabular_linear_functions(ace, base_distribution, idat_absolute)
    elif law == 24:
        if debug: logger.debug("Parsing tabular energy multipliers (Law 24)")
        return parse_tabular_energy_multipliers(ace, base_distribution, idat_absolute)
    elif law == 44:
        if debug: logger.debug("Parsing Kalbach-Mann distribution (Law 44)")
        return parse_kalbach_mann_distribution(ace, base_distribution, idat_absolute)
    elif law == 61:
        if debug: logger.debug("Parsing tabulated angle-energy distribution (Law 61)")
        return parse_tabulated_angle_energy_distribution(ace, base_distribution, idat_absolute)
    elif law == 66:
        if debug: logger.debug("Parsing N-body phase space distribution (Law 66)")
        return parse_nbody_phase_space_distribution(ace, base_distribution, idat_absolute)
    elif law == 67:
        if debug: logger.debug("Parsing laboratory angle-energy distribution (Law 67)")
        return parse_laboratory_angle_energy_distribution(ace, base_distribution, idat_absolute)
    else:
        if debug: logger.debug(f"No specific parser for law {law}, returning base distribution")
        return base_distribution
