import logging
from mcnpy.ace.classes.ace import Ace
from mcnpy.ace.parsers.xss import XssEntry
from mcnpy.ace.classes.header import Header
from mcnpy.ace.parsers.parse_esz import read_esz_block
from mcnpy.ace.parsers import read_header, read_nubar_data
from mcnpy.ace.parsers.parse_delayed import read_delayed_neutron_data
from mcnpy.ace.parsers.parse_mtr import read_mtr_blocks
from mcnpy.ace.parsers.parse_lqr import read_lqr_block
from mcnpy.ace.parsers.parse_tyr import read_tyr_blocks
from mcnpy.ace.parsers.parse_xs_locators import read_xs_locator_blocks
from mcnpy.ace.parsers.parse_xs_data import read_xs_data_block
from mcnpy.ace.parsers.parse_angular_locators import read_angular_locator_blocks
from mcnpy.ace.parsers.parse_angular_distribution import read_angular_distribution_blocks
from mcnpy.ace.parsers.parse_energy_distribution_locators import read_energy_locator_blocks
from mcnpy.ace.parsers.parse_energy_distributions import read_energy_distribution_blocks
from mcnpy.ace.classes.energy_distribution.energy_distribution_container import EnergyDistributionContainer
from mcnpy.ace.parsers.parse_gpd import read_gpd_block
from mcnpy.ace.parsers.parse_photon_production_xs import read_production_xs_blocks
from mcnpy.ace.parsers.parse_yield_multipliers import read_yield_multiplier_blocks
from mcnpy.ace.parsers.parse_fission_xs import read_fission_xs_block
from mcnpy.ace.parsers.parse_unresolved_resonance import read_unresolved_resonance_block
from mcnpy.ace.parsers.parse_secondary_particle_types import parse_ptype_block
from mcnpy.ace.parsers.parse_secondary_reaction_counts import parse_ntro_block
from mcnpy.ace.parsers.parse_secondary_data_locators import parse_ixs_block
from mcnpy.ace.parsers.parse_secondary_cross_sections import parse_hpd_block

# Setup logger
logger = logging.getLogger(__name__)

def read_ace(filename, debug=False):
    """
    Read and parse an ACE format file.
    
    This implementation eagerly loads all data except energy distribution data
    which is still loaded on-demand when accessed.
    
    Parameters
    ----------
    filename : str
        Path to the ACE file
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    ace : Ace
        An Ace object containing the parsed data
    """
    debug1 = True

    if debug:
        logger.debug(f"Reading ACE file: {filename}")
        
    ace = Ace()
    ace.filename = filename
    ace.header = Header()
    ace._debug = debug  # Store debug flag for later use by parsers
    
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Determine if it's a legacy or 2.0.1 header and read it
    if lines and "2.0" in lines[0][:10]:
        ace.header.format_version = "2.0.1"
    else:
        ace.header.format_version = "legacy"
    
    if debug:
        logger.debug(f"ACE format version: {ace.header.format_version}")
    
    # Read the entire header (opening and arrays)
    line_idx = read_header(ace.header, lines, debug=debug)
    
    # Read the XSS array - essential data needed for all parsers
    ace.xss_data = read_xss(lines[line_idx:])
    
    # Eagerly load all components except energy distribution data
    
    # ESZ Block
    ace.esz_block = read_esz_block(ace, debug)
    
    # Nubar data
    ace.nubar = read_nubar_data(ace, debug)
    
    # Delayed neutron data
    ace.delayed_neutron_data = read_delayed_neutron_data(ace, debug)
    
    # MT Reaction data
    ace.reaction_mt_data = read_mtr_blocks(ace, debug)
    
    # Q values
    ace.q_values = read_lqr_block(ace, debug)
    
    # Particle release data
    ace.particle_release = read_tyr_blocks(ace, debug)
    
    # Cross section locators
    ace.xs_locators = read_xs_locator_blocks(ace, debug)
    
    # Cross section data
    ace.xs_data = read_xs_data_block(ace, debug)
    
    # Angular distribution locators - Fix: Assign the return value instead of direct modification
    ace.angular_locators = read_angular_locator_blocks(ace, debug)
    
    # Angular distribution data - Fix: Assign the return value instead of direct modification
    ace.angular_distributions = read_angular_distribution_blocks(ace, debug)
    
    # Energy distribution locators
    ace.energy_distribution_locators = read_energy_locator_blocks(ace, debug)
    
    # Energy distribution data
    ace.energy_distributions = read_energy_distribution_blocks(ace, debug)

    # Photon production data
    ace.photon_production_data = read_gpd_block(ace, debug)
    
    # Secondary particle types (PTYPE block)
    # This must be read first as other secondary particle blocks depend on it
    ace.secondary_particle_types = parse_ptype_block(ace, debug)
    
    # Photon production cross sections
    photon_xs, particle_xs = read_production_xs_blocks(ace, debug)
    ace.photon_production_xs = photon_xs
    ace.particle_production_xs = particle_xs
    
    # Fission cross section
    ace.fission_xs = read_fission_xs_block(ace, debug)
    
    # Unresolved resonance tables
    ace.unresolved_resonance = read_unresolved_resonance_block(ace, debug)
    
    # Secondary particle reaction counts (NTRO block)
    ace.secondary_particle_reactions = parse_ntro_block(ace, debug)
    
    # Secondary particle data locations (IXS block)
    ace.secondary_particle_data_locations = parse_ixs_block(ace, debug)
    
    # Photon and secondary particle yield multipliers
    photon_yield_multipliers, particle_yield_multipliers = read_yield_multiplier_blocks(ace, debug)
    ace.photon_yield_multipliers = photon_yield_multipliers
    ace.particle_yield_multipliers = particle_yield_multipliers
    
    # Secondary particle cross sections (HPD block)
    ace.secondary_particle_cross_sections = parse_hpd_block(ace, debug)
    
    return ace

def read_xss(lines):
    """
    Read the XSS array from an ACE file and convert it to a list of XssEntry.
    
    The array uses 1-based indexing to match FORTRAN style indexing.
    Index 0 contains a placeholder value (0) to facilitate 1-based indexing.
    
    Parameters
    ----------
    lines : list
        List of lines from the file containing the XSS array
        
    Returns
    -------
    list
        The XSS array as a list of XssEntry objects starting at index 1
    """
    xss_data = [0]  # Placeholder at index 0 for FORTRAN-style 1-based indexing
    xss_index = 1  # Start at 1 for FORTRAN-style indexing
    
    for line in lines:
        # Each line contains 4 numbers in 4E20.0 format
        for i in range(4):
            start_idx = i * 20
            if start_idx + 20 <= len(line):
                value_str = line[start_idx:start_idx+20].strip()
                if value_str:
                    try:
                        value = float(value_str)
                        # Create XssEntry with the current index in the array
                        xss_data.append(XssEntry(xss_index, value))
                        xss_index += 1
                    except ValueError:
                        # Skip non-numeric entries
                        pass
    
    return xss_data