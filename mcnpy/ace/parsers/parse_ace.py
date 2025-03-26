import logging
from mcnpy.ace.classes.ace import Ace
from mcnpy.ace.parsers.xss import XssEntry
from mcnpy.ace.classes.header import Header
from mcnpy.ace.classes.esz import EszBlock
from mcnpy.ace.parsers.parse_esz import read_esz_block
from mcnpy.ace.classes.nubar.nubar import NuContainer
from mcnpy.ace.classes.delayed_neutron import DelayedNeutronData
from mcnpy.ace.classes.mtr import ReactionMTData
from mcnpy.ace.classes.q_values import QValues
from mcnpy.ace.classes.particle_release import ParticleRelease
from mcnpy.ace.classes.xs_locators import CrossSectionLocators
from mcnpy.ace.parsers import read_header, read_nubar_data
from mcnpy.ace.parsers.parse_delayed import read_delayed_neutron_data
from mcnpy.ace.parsers.parse_mtr import read_mtr_blocks
from mcnpy.ace.parsers.parse_lqr import read_lqr_block
from mcnpy.ace.parsers.parse_tyr import read_tyr_blocks
from mcnpy.ace.parsers.parse_xs_locators import read_xs_locator_blocks
from mcnpy.ace.classes.xs_data import CrossSectionData
from mcnpy.ace.parsers.parse_xs_data import read_xs_data_block
from mcnpy.ace.classes.angular_distribution.angular_locators import AngularDistributionLocators
from mcnpy.ace.parsers.parse_angular_locators import read_angular_locator_blocks
from mcnpy.ace.classes.angular_distribution.angular_distribution import AngularDistributionContainer
from mcnpy.ace.parsers.parse_angular_distribution import read_angular_distribution_blocks
from mcnpy.ace.classes.energy_distribution_locators import EnergyDistributionLocators
from mcnpy.ace.parsers.parse_energy_distribution_locators import read_energy_locator_blocks
from mcnpy.ace.classes.energy_distribution_container import EnergyDistributionContainer
from mcnpy.ace.parsers.parse_energy_distributions import read_energy_distribution_blocks
from mcnpy.ace.classes.gpd import PhotonProductionData
from mcnpy.ace.parsers.parse_gpd import read_gpd_block
from mcnpy.ace.classes.photon_production_xs import PhotonProductionCrossSections, ParticleProductionCrossSections
from mcnpy.ace.parsers.parse_photon_production_xs import read_production_xs_blocks
from mcnpy.ace.classes.yield_multipliers import PhotonYieldMultipliers, ParticleYieldMultipliers
from mcnpy.ace.parsers.parse_yield_multipliers import read_yield_multiplier_blocks
from mcnpy.ace.classes.fission_xs import FissionCrossSection
from mcnpy.ace.parsers.parse_fission_xs import read_fission_xs_block
from mcnpy.ace.classes.unresolved_resonance import UnresolvedResonanceTables
from mcnpy.ace.parsers.parse_unresolved_resonance import read_unresolved_resonance_block
from mcnpy.ace.classes.particle_production import ParticleProductionTypes
from mcnpy.ace.parsers.parse_particle_production import read_particle_types_block
from mcnpy.ace.classes.particle_reaction_counts import ParticleReactionCounts
from mcnpy.ace.parsers.parse_particle_reaction_counts import read_particle_reaction_counts_block
from mcnpy.ace.classes.particle_production_locators import ParticleProductionLocators
from mcnpy.ace.parsers.parse_particle_production_locators import read_particle_production_locators_block
from mcnpy.ace.classes.particle_production_xs_data import ParticleProductionXSContainer
from mcnpy.ace.parsers.parse_particle_production_xs_data import read_particle_production_xs_data_blocks

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
    
    # Read XSS array - essential data needed for all parsers
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
    
    # Angular distribution locators
    read_angular_locator_blocks(ace, debug)  # This function modifies ace directly
    
    # Angular distribution data
    read_angular_distribution_blocks(ace, debug)  # This function modifies ace directly
    
    # Energy distribution locators
    ace.energy_distribution_locators = read_energy_locator_blocks(ace, debug)
    
    # NOTE: Energy distribution data will be loaded on-demand
    # when accessed to maintain lazy loading for this component
    
    # Photon production data
    ace.photon_production_data = read_gpd_block(ace, debug)
    
    # Photon production cross sections
    result = read_production_xs_blocks(ace, debug)
    if result:
        ace.photon_production_xs, ace.particle_production_xs = result
    
    # Photon yield multipliers
    result = read_yield_multiplier_blocks(ace, debug)
    if result:
        ace.photon_yield_multipliers, ace.particle_yield_multipliers = result
    
    # Fission cross section
    ace.fission_xs = read_fission_xs_block(ace, debug)
    
    # Unresolved resonance tables
    ace.unresolved_resonance = read_unresolved_resonance_block(ace, debug)
    
    # Particle production types
    ace.secondary_particles = read_particle_types_block(ace, debug)
    ace.particle_types = ace.secondary_particles  # Set the alias
    
    # Particle reaction counts
    ace.particle_reaction_counts = read_particle_reaction_counts_block(ace, debug)
    
    # Particle production locators
    ace.particle_production_locators = read_particle_production_locators_block(ace, debug)
    
    # Particle production cross section data
    ace.particle_production_xs_data = read_particle_production_xs_data_blocks(ace, debug)
    
    return ace

def read_xss(lines):
    """
    Lee el array XSS de un archivo ACE y lo convierte en una lista de XssEntry.
    
    Parameters
    ----------
    lines : list
        Lista de líneas del archivo que contienen el array XSS
        
    Returns
    -------
    list
        El array XSS como una lista de objetos XssEntry
    """
    xss_data = [0]
    xss_index = 0  # Renamed for clarity
    
    for line in lines:
        # Cada línea contiene 4 números en formato 4E20.0
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
                        # Se omiten entradas no numéricas
                        pass
    
    return xss_data