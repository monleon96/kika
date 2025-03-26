from typing import List, Optional
from mcnpy.ace.classes.ace import Ace
from mcnpy.ace.classes.angular_distribution.angular_distribution import (
    AngularDistributionContainer, AngularDistribution, 
    IsotropicAngularDistribution, EquiprobableAngularDistribution,
    TabulatedAngularDistribution, KalbachMannAngularDistribution,
)
from mcnpy.ace.parsers.xss import XssEntry
import logging

# Setup logger
logger = logging.getLogger(__name__)

def read_angular_distribution_blocks(ace: Ace, debug: bool = False) -> None:
    """
    Read the AND, ANDP, and ANDH blocks from the ACE file.
    
    These blocks contain angular distribution data:
    - AND: For incident neutron reactions (JXS[9])
    - ANDP: For photon production reactions (JXS[17])
    - ANDH: For other particle production reactions (accessed through particle pointers)
    
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
        logger.debug("Reading angular distribution blocks")
        
    if (ace.header is None or ace.header.jxs_array is None or 
        ace.header.nxs_array is None or ace.xss_data is None):
        raise ValueError("Cannot read angular distribution blocks: header or XSS data missing")
    
    # Make sure we have angular locators
    if ace.angular_locators is None:
        raise ValueError("Angular distribution locators missing. Call read_angular_locator_blocks first.")
    
    # Create a new container if not present
    if ace.angular_distributions is None:
        ace.angular_distributions = AngularDistributionContainer()
    
    # Get the JXS pointers for each block
    and_idx = ace.header.jxs_array[9]  # JXS(9)
    andp_idx = ace.header.jxs_array[17]  # JXS(17)
    
    if debug:
        logger.debug(f"AND block index: {and_idx}, ANDP block index: {andp_idx}")
    
    # Read AND block (incident neutron reactions)
    read_and_block(ace, and_idx, debug)
    
    # Read ANDP block (photon production)
    read_andp_block(ace, andp_idx, debug)
    
    # Read ANDH blocks (other particle production)
    read_andh_blocks(ace, debug)
    
def read_and_block(ace: Ace, and_idx: int, debug: bool = False) -> None:
    """
    Read the AND block containing angular distributions for incident neutron reactions.
    
    Parameters
    ----------
    ace : Ace
        The Ace object to update
    and_idx : int
        Starting index of the AND block in the XSS array
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Raises
    ------
    ValueError
        If the AND block data is missing or invalid
    """
    if debug:
        logger.debug(f"Reading AND block at index {and_idx}")
        
    # Check if AND block is present
    if and_idx <= 0:
        return  # No AND block
    
    try:
        # Make sure elastic scattering locator is available
        if ace.angular_locators.elastic_scattering is None:
            return
            
        elastic_locator_value = int(ace.angular_locators.elastic_scattering.value)
        
        if elastic_locator_value > 0:
            # Process elastic scattering angular distribution
            elastic_data_idx = and_idx + elastic_locator_value  
            
            if elastic_data_idx >= len(ace.xss_data):
                raise ValueError(f"Elastic scattering data index out of bounds: {elastic_data_idx} >= {len(ace.xss_data)}")
            
            try:
                # Get MT=2 for elastic from MT data if available
                mt_entry = None
                if ace.reaction_mt_data and len(ace.reaction_mt_data.incident_neutron) > 0:
                    for entry in ace.reaction_mt_data.incident_neutron:
                        if int(entry.value) == 2:  # MT=2 for elastic
                            mt_entry = entry
                            break
                
                # If not found in reaction_mt_data, create a dummy entry for MT=2
                if mt_entry is None:
                    mt_entry = XssEntry(0, 2)  # MT=2 for elastic
                
                elastic_dist = read_angular_distribution(ace, elastic_data_idx, mt_entry, debug)
                if elastic_dist:
                    ace.angular_distributions.elastic = elastic_dist
                    if debug:
                        logger.debug(f"Read elastic scattering distribution (MT=2)")
            except ValueError as e:
                raise ValueError(f"Error reading elastic scattering distribution: {e}")
        
        # Process other neutron reaction angular distributions
        for i, locator_entry in enumerate(ace.angular_locators.incident_neutron):
            locator_value = int(locator_entry.value)
            
            if locator_value == 0:
                # Isotropic distribution, no data needed
                continue
            elif locator_value == -1:
                # Angular distribution is in the DLW block using Law=44
                mt_entry = None
                if ace.reaction_mt_data and ace.reaction_mt_data.has_neutron_mt_data:
                    # Offset by 1 since this list doesn't include elastic scattering
                    if i < len(ace.reaction_mt_data.incident_neutron):
                        mt_entry = ace.reaction_mt_data.incident_neutron[i]
                
                if mt_entry is None:
                    continue  # Skip if MT number not available
                    
                # Create a Kalbach-Mann distribution object with the reaction index
                # This will be used to lookup the appropriate Law=44 distribution in the DLW block
                dist = KalbachMannAngularDistribution(
                    mt=mt_entry,
                    reaction_index=i,  # Store the reaction index for lookup in DLW
                    is_particle_production=False
                )
                
                # Store using the MT value as the key
                mt_value = int(mt_entry.value)
                ace.angular_distributions.incident_neutron[mt_value] = dist
                continue
            elif locator_value < -1:  # Invalid negative value
                continue
            
            # Get the corresponding MT number
            mt_entry = None
            if ace.reaction_mt_data and ace.reaction_mt_data.has_neutron_mt_data:
                # Offset by 1 since this list doesn't include elastic scattering
                if i < len(ace.reaction_mt_data.incident_neutron):
                    mt_entry = ace.reaction_mt_data.incident_neutron[i]
            
            if mt_entry is None:
                continue  # Skip if MT number not available
            
            # Calculate the data index
            data_idx = and_idx + locator_value  
            
            # Check if the index is valid before trying to read
            if data_idx < 0 or data_idx >= len(ace.xss_data):
                # Continue instead of failing if just one reaction has an issue
                continue
            
            try:
                # Read the angular distribution
                dist = read_angular_distribution(ace, data_idx, mt_entry, debug)
                if dist:
                    # Store using the MT value as the key
                    mt_value = int(mt_entry.value)
                    ace.angular_distributions.incident_neutron[mt_value] = dist
                    if debug:
                        logger.debug(f"Read neutron reaction distribution for MT={mt_value}")
            except ValueError:
                # Skip this reaction if there's an issue
                continue
    except Exception as e:
        raise ValueError(f"Error reading AND block: {e}")

def read_andp_block(ace: Ace, andp_idx: int, debug: bool = False) -> None:
    """
    Read the ANDP block containing angular distributions for photon production reactions.
    
    According to the ACE format specification, the angular distribution for the i-th 
    photon-producing reaction begins at JXS(17) + LOCBᵢ - 1, where LOCBᵢ comes from the 
    LANDP block. Each distribution contains:
    1. Number of energies (N_E)
    2. Energy grid (N_E values)
    3. Locators (L_C) for each energy (N_E values), relative to JXS(17)
    4. 32 equiprobable cosine bins (33 values) for each energy
    
    Parameters
    ----------
    ace : Ace
        The Ace object to update
    andp_idx : int
        Starting index of the ANDP block in the XSS array (JXS(17))
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Raises
    ------
    ValueError
        If the ANDP block data is missing or invalid
    """
    if debug:
        logger.debug(f"Reading ANDP block at index {andp_idx}")
        
    # Check if ANDP block is present
    if andp_idx <= 0:
        return  # No ANDP block
    
    try:
        # Process photon production angular distributions
        for i, locator_entry in enumerate(ace.angular_locators.photon_production):
            locator_value = int(locator_entry.value)
            
            if locator_value == 0:
                # Isotropic distribution, no data needed
                continue
            elif locator_value < 0:  # Invalid negative value
                logger.warning(f"Invalid negative locator value {locator_value} for photon production angular distribution at index {i}")
                continue
            
            # Get the corresponding MT number
            mt_entry = None
            if ace.reaction_mt_data and ace.reaction_mt_data.has_photon_production_mt_data:
                if i < len(ace.reaction_mt_data.photon_production):
                    mt_entry = ace.reaction_mt_data.photon_production[i]
            
            if mt_entry is None:
                continue  # Skip if MT number not available
            
            # Calculate the data index according to the documentation: JXS(17) + LOCBᵢ - 1
            data_idx = andp_idx + locator_value - 1
            
            if debug:
                logger.debug(f"Photon production distribution for MT={mt_entry.value}: LOCBᵢ={locator_value}, index={data_idx}")
            
            # Check if the index is valid before trying to read
            if data_idx < 0 or data_idx >= len(ace.xss_data):
                logger.warning(f"Invalid angular distribution index for photon production: {data_idx} out of bounds")
                continue
            
            try:
                # Read the angular distribution with the ANDP block base index
                dist = read_angular_distribution_photon(ace, data_idx, andp_idx, mt_entry, debug)
                if dist:
                    # Store using the MT value as the key
                    mt_value = int(mt_entry.value)
                    ace.angular_distributions.photon_production[mt_value] = dist
                    if debug:
                        logger.debug(f"Read photon production distribution for MT={mt_value}")
            except ValueError as e:
                logger.warning(f"Error reading photon production distribution: {e}")
                # Skip this reaction if there's an issue
                continue
    except Exception as e:
        raise ValueError(f"Error reading ANDP block: {e}")

def read_angular_distribution_photon(ace: Ace, data_idx: int, andp_idx: int, mt_entry: XssEntry, debug: bool = False) -> Optional[AngularDistribution]:
    """
    Read a photon production angular distribution from the XSS array.
    
    The distribution structure follows Table 54 from the ACE format specification:
    1. Number of energies (N_E)
    2. Energy grid (N_E values)
    3. Locators (L_C) for each energy (N_E values), relative to JXS(17)
    4. 32 equiprobable cosine bins (33 values) for each energy
    
    Parameters
    ----------
    ace : Ace
        The Ace object with XSS data
    data_idx : int
        Starting index of the angular distribution data in the XSS array
    andp_idx : int
        Base index of the ANDP block (JXS(17)) for relative locators
    mt_entry : XssEntry
        MT number entry for this reaction
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    AngularDistribution
        The parsed angular distribution
        
    Raises
    ------
    ValueError
        If data is invalid or inconsistent
    """
    if debug:
        logger.debug(f"Reading photon production angular distribution at index {data_idx} for MT={mt_entry.value}")
    
    if data_idx < 0 or data_idx >= len(ace.xss_data):
        raise ValueError(f"Angular distribution index out of bounds: {data_idx}")
    
    # First value is the number of energies (N_E)
    loc = data_idx
    num_energies = int(ace.xss_data[loc].value)
    
    if debug:
        logger.debug(f"Number of energy points: {num_energies}")
    
    if num_energies <= 0:
        # If N_E = 0, we have isotropic scattering for all energies
        return IsotropicAngularDistribution(mt=mt_entry)
    
    # Check if we have enough data
    if loc + 1 + 2*num_energies > len(ace.xss_data):
        raise ValueError(f"Photon angular distribution data truncated: need at least {1 + 2*num_energies} entries, but only {len(ace.xss_data) - loc} available")
    
    # Read the energy grid (N_E values)
    energies = ace.xss_data[loc + 1:loc + 1 + num_energies]
    
    # Read the locators (L_C) for each energy (N_E values) - relative to JXS(17)
    lc_start = loc + 1 + num_energies
    locators = ace.xss_data[lc_start:lc_start + num_energies]
    
    # Create an equiprobable angular distribution
    distribution = EquiprobableAngularDistribution(mt=mt_entry, energies=energies)
    
    # For each energy point with a non-zero locator
    for i, locator_entry in enumerate(locators):
        locator_value = int(locator_entry.value)
        
        if locator_value == 0:
            # Isotropic distribution at this energy
            # Add 33 values from -1 to 1 (uniformly spaced)
            # Create XssEntry objects for the uniformly spaced cosines
            cosines = [XssEntry(0, -1.0 + j * (2.0 / 32)) for j in range(33)]
            distribution.cosine_bins.append(cosines)
            if debug:
                logger.debug(f"Using isotropic distribution for energy point {i} ({energies[i].value})")
            continue
        
        # L_C is relative to JXS(17), so calculate the absolute index
        data_loc = andp_idx + locator_value - 1
        
        if debug:
            logger.debug(f"Reading cosine bins for energy point {i}: L_C={locator_value}, index={data_loc}")
        
        # Check if we have enough data
        if data_loc + 33 > len(ace.xss_data):
            raise ValueError(f"Equiprobable bin data truncated at energy {energies[i].value}: need 33 entries, but only {len(ace.xss_data) - data_loc} available")
        
        # Read the 33 cosine values for 32 equiprobable bins
        cosines = ace.xss_data[data_loc:data_loc + 33]
        distribution.cosine_bins.append(cosines)
        
        if debug and i == 0:  # Print details for first energy only to avoid verbose output
            logger.debug(f"Cosine bins for first energy: {[c.value for c in cosines[:5]]}... (showing first 5)")
    
    return distribution

def read_andh_blocks(ace: Ace, debug: bool = False) -> None:
    """
    Read the ANDH blocks containing angular distributions for other particle production reactions.
    
    Parameters
    ----------
    ace : Ace
        The Ace object to update
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Raises
    ------
    ValueError
        If the ANDH block data is missing or invalid
    """
    if debug:
        logger.debug("Reading ANDH blocks")
        logger.debug(f"Number of particle types: {ace.angular_locators.num_particle_types}")
    
    # Get the number of particle types
    num_particle_types = ace.angular_locators.num_particle_types
    if num_particle_types <= 0:
        return  # No particle production data
    
    # Initialize particle production dictionaries
    while len(ace.angular_distributions.particle_production) < num_particle_types:
        ace.angular_distributions.particle_production.append({})
    
    # Process each particle type
    for particle_idx in range(num_particle_types):
        if debug:
            logger.debug(f"Processing particle type {particle_idx}")
            
        # Get locators for this particle type
        locators = ace.angular_locators.get_particle_production_locators(particle_idx)
        if not locators:
            continue  # No locators for this particle type
        
        # Get MT numbers for this particle type
        mt_numbers = []
        if ace.reaction_mt_data and ace.reaction_mt_data.has_particle_production_mt_data:
            mt_numbers = ace.reaction_mt_data.get_particle_production_mt_numbers(particle_idx) or []
        
        if not mt_numbers or len(mt_numbers) != len(locators):
            continue  # MT numbers missing or count mismatch
        
        # Get the ANDH pointer for this particle
        andh_ptr = 0
        if len(ace.header.jxs_array) > 32:  # Need JXS(32)
            jxs32_idx = ace.header.jxs_array[32]  # JXS(32)
            if jxs32_idx > 0 and particle_idx < num_particle_types:
                # ANDH pointer is at XSS(JXS(32)+10*(i-1)+6)
                andh_pointer_idx = jxs32_idx + 10 * particle_idx + 6
                
                if andh_pointer_idx < len(ace.xss_data):
                    andh_ptr = int(ace.xss_data[andh_pointer_idx].value)
                    
                    # Validate the pointer value
                    if andh_ptr <= 0:
                        continue
                    
                    if andh_ptr > len(ace.xss_data):
                        continue
                    
                else:
                    continue
            else:
                continue
        else:
            continue
        
        if andh_ptr <= 0:
            continue  # No ANDH block for this particle type
            
        # Process angular distributions for this particle type
        for i, (mt, locator) in enumerate(zip(mt_numbers, locators)):
            locator_value = int(locator.value)
            
            if locator_value == 0:
                # Isotropic distribution, no data needed
                continue
            elif locator_value == -1:
                # Angular distribution is in the DLWH block using Law=44
                
                # Create a Kalbach-Mann distribution object with the reaction and particle indices
                # This will be used to lookup the appropriate Law=44 distribution in the DLWH block
                dist = KalbachMannAngularDistribution(
                    mt=mt,
                    reaction_index=i,
                    is_particle_production=True,
                    particle_idx=particle_idx
                )
                
                # Store using the MT value as the key
                mt_value = int(mt)
                ace.angular_distributions.particle_production[particle_idx][mt_value] = dist
                if debug:
                    logger.debug(f"Read particle production distribution for particle {particle_idx}, MT={mt_value}")
                continue
            elif locator_value < -1:  # Invalid negative value
                continue
            
            # Calculate the actual data index using the data entry at the ANDH address
            data_idx = andh_ptr
            
            # Validate the final index
            if data_idx < 0 or data_idx >= len(ace.xss_data):
                continue
            
            try:
                # Read the angular distribution from the ANDH address
                # The angular distribution at ANDH follows the same format as at AND
                dist = read_angular_distribution(ace, data_idx, mt, debug)
                if dist:
                    ace.angular_distributions.particle_production[particle_idx][mt] = dist
                    if debug:
                        logger.debug(f"Read particle production distribution for particle {particle_idx}, MT={mt}")
            except ValueError:
                # Skip this reaction if there's an issue
                continue

def read_angular_distribution(ace: Ace, data_idx: int, mt_entry: XssEntry, debug: bool = False) -> Optional[AngularDistribution]:
    """
    Read a single angular distribution from the XSS array.
    
    The distribution structure starts with:
    1. Number of energies (NE)
    2. Energy grid (NE values)
    3. Locators (LC) for each energy (NE values)
    4. Distribution data pointed to by each LC
    
    Parameters
    ----------
    ace : Ace
        The Ace object with XSS data
    data_idx : int
        Starting index of the angular distribution data in the XSS array
    mt_entry : XssEntry
        MT number entry for this reaction
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    AngularDistribution
        The parsed angular distribution
        
    Raises
    ------
    ValueError
        If data is invalid or inconsistent
    """
    if debug:
        logger.debug(f"Reading angular distribution at index {data_idx} for MT={mt_entry.value}")
    
    if data_idx < 0 or data_idx >= len(ace.xss_data):
        raise ValueError(f"Angular distribution index out of bounds: {data_idx}")
    
    # First value is the number of energies (NE)
    loc = data_idx
    num_energies = int(ace.xss_data[loc].value)
    
    if num_energies <= 0:
        # If NE = 0, we have isotropic scattering for all energies
        return IsotropicAngularDistribution(mt=mt_entry)
    
    # Check if we have enough data
    if loc + 1 + 2*num_energies > len(ace.xss_data):
        raise ValueError(f"Angular distribution data truncated: need at least {1 + 2*num_energies} entries, but only {len(ace.xss_data) - loc} available")
    
    # Read the energy grid (NE values)
    energies = ace.xss_data[loc + 1:loc + 1 + num_energies]
    
    # Read the locators (LC) for each energy (NE values)
    lc_start = loc + 1 + num_energies
    locators = ace.xss_data[lc_start:lc_start + num_energies]
    locator_values = [int(entry.value) for entry in locators]
    
    # For tabulated and equiprobable distributions, we need the base of the AND block
    # Find which block we're in based on the MT
    and_idx = ace.header.jxs_array[9]  # JXS(9)
    
    # Check distribution type based on the locators
    if all(lc_val == 0 for lc_val in locator_values):
        # All locators are 0, meaning isotropic for all energies
        return IsotropicAngularDistribution(mt=mt_entry, energies=energies)
    
    # Check if we have equiprobable bin or tabulated distributions
    if all(lc_val >= 0 for lc_val in locator_values if lc_val != 0):
        # All non-zero locators are positive - equiprobable bin distribution
        return read_equiprobable_distribution(ace, and_idx, mt_entry, num_energies, energies, locators, debug)
    
    elif all(lc_val <= 0 for lc_val in locator_values if lc_val != 0):
        # All non-zero locators are negative - tabulated distribution
        return read_tabulated_distribution(ace, and_idx, mt_entry, num_energies, energies, locators, debug)
    
    else:
        # Mixed locator signs - this shouldn't happen according to the format
        raise ValueError(f"Mixed angular distribution types in the same reaction: {locator_values}")

def read_equiprobable_distribution(ace: Ace, base_idx: int, mt_entry: XssEntry, 
                                   num_energies: int, energies: List[XssEntry], 
                                   locators: List[XssEntry], debug: bool = False) -> EquiprobableAngularDistribution:
    """
    Read a 32 equiprobable cosine bin angular distribution.
    
    Each distribution consists of 33 cosine values defining 32 equiprobable bins.
    
    Parameters
    ----------
    ace : Ace
        The Ace object with XSS data
    base_idx : int
        Base index of the AND block in the XSS array (not the distribution data!)
    mt_entry : XssEntry
        MT number entry for this reaction
    num_energies : int
        Number of energy points
    energies : List[XssEntry]
        Energy grid for the angular distribution
    locators : List[XssEntry]
        Locators (LC) for each energy
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    EquiprobableAngularDistribution
        The parsed equiprobable bin distribution
        
    Raises
    ------
    ValueError
        If data is invalid or inconsistent
    """
    if debug:
        logger.debug(f"Reading equiprobable distribution for MT={mt_entry.value} with {num_energies} energies")
    
    # Create an equiprobable angular distribution
    distribution = EquiprobableAngularDistribution(mt=mt_entry, energies=energies)
    
    # For each energy point with a non-zero locator
    for i, locator_entry in enumerate(locators):
        locator_value = int(locator_entry.value)
        
        if locator_value == 0:
            # Isotropic distribution at this energy
            # Add 33 values from -1 to 1 (uniformly spaced)
            # Create XssEntry objects for the uniformly spaced cosines
            cosines = [XssEntry(0, -1.0 + j * (2.0 / 32)) for j in range(33)]
            distribution.cosine_bins.append(cosines)
            continue
        
        # Positive locator points to 32 equiprobable bin boundaries
        data_loc = base_idx + locator_value  
        
        # Check if we have enough data
        if data_loc + 33 > len(ace.xss_data):
            raise ValueError(f"Equiprobable bin data truncated at energy {energies[i].value}: need 33 entries, but only {len(ace.xss_data) - data_loc} available")
        
        # Read the 33 cosine values
        cosines = ace.xss_data[data_loc:data_loc + 33]
        distribution.cosine_bins.append(cosines)
    
    return distribution

def read_tabulated_distribution(ace: Ace, base_idx: int, mt_entry: XssEntry, 
                                num_energies: int, energies: List[XssEntry], 
                                locators: List[XssEntry], debug: bool = False) -> TabulatedAngularDistribution:
    """
    Read a tabulated angular distribution.
    
    Each distribution consists of:
    1. Interpolation flag (1=histogram, 2=linear-linear)
    2. Number of points (Np)
    3. Cosine grid (Np values)
    4. PDF values (Np values)
    5. CDF values (Np values)
    
    Parameters
    ----------
    ace : Ace
        The Ace object with XSS data
    base_idx : int
        Base index of the AND block in the XSS array (not the distribution data!)
    mt_entry : XssEntry
        MT number entry for this reaction
    num_energies : int
        Number of energy points
    energies : List[XssEntry]
        Energy grid for the angular distribution
    locators : List[XssEntry]
        Locators (LC) for each energy
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    TabulatedAngularDistribution
        The parsed tabulated distribution
        
    Raises
    ------
    ValueError
        If data is invalid or inconsistent
    """
    if debug:
        logger.debug(f"Reading tabulated distribution for MT={mt_entry.value} with {num_energies} energies")
    
    # Create a tabulated angular distribution
    distribution = TabulatedAngularDistribution(mt=mt_entry, energies=energies)
    
    # For each energy point
    for i, locator_entry in enumerate(locators):
        locator_value = int(locator_entry.value)
        
        if locator_value == 0:
            # Isotropic distribution at this energy
            # Add a simple two-point distribution: μ=[-1,1], PDF=[0.5,0.5], CDF=[0,1]
            distribution.interpolation.append(2)  # linear-linear
            
            # Create XssEntry objects for the simple distribution
            cosines = [XssEntry(0, -1.0), XssEntry(0, 1.0)]
            pdfs = [XssEntry(0, 0.5), XssEntry(0, 0.5)]
            cdfs = [XssEntry(0, 0.0), XssEntry(0, 1.0)]
            
            distribution.cosine_grid.append(cosines)
            distribution.pdf.append(pdfs)
            distribution.cdf.append(cdfs)
            continue
        
        # Negative locator points to tabulated distribution
        lc_abs = abs(locator_value)
        # LC is relative to the AND block, not the specific distribution data
        data_loc = base_idx + lc_abs  
        
        # Check if we have enough data for the header (interp + num_points)
        if data_loc + 2 > len(ace.xss_data):
            raise ValueError(f"Tabulated distribution data truncated at energy {energies[i].value}: header missing")
        
        # Read interpolation flag
        interp_flag = int(ace.xss_data[data_loc].value)
        distribution.interpolation.append(interp_flag)
        
        # Read number of points (Np)
        num_points = int(ace.xss_data[data_loc + 1].value)
        
        if num_points <= 0:
            raise ValueError(f"Invalid number of points in tabulated distribution at energy {energies[i].value}: {num_points}")
        
        # Check if we have enough data for the full distribution
        # Need: interp(1) + num_points(1) + cosine(Np) + PDF(Np) + CDF(Np) = 2 + 3*Np
        if data_loc + 2 + 3*num_points > len(ace.xss_data):
            raise ValueError(f"Tabulated distribution data truncated at energy {energies[i].value}: need {2 + 3*num_points} entries, but only {len(ace.xss_data) - data_loc} available")
        
        # Read cosine grid (Np values)
        cosine_start = data_loc + 2
        cosines = ace.xss_data[cosine_start:cosine_start + num_points]
        distribution.cosine_grid.append(cosines)
        
        # Read PDF values (Np values)
        pdf_start = cosine_start + num_points
        pdfs = ace.xss_data[pdf_start:pdf_start + num_points]
        distribution.pdf.append(pdfs)
        
        # Read CDF values (Np values)
        cdf_start = pdf_start + num_points
        cdfs = ace.xss_data[cdf_start:cdf_start + num_points]
        distribution.cdf.append(cdfs)
    
    return distribution