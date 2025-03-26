from typing import List, Optional
from mcnpy.ace.classes.ace import Ace
from mcnpy.ace.classes.particle_production_xs_data import ParticleProductionXSData, ParticleProductionXSContainer
import logging

# Setup logger
logger = logging.getLogger(__name__)

def read_particle_production_xs_data_blocks(ace: Ace, debug: bool = False) -> None:
    """
    Read the HPD blocks containing particle production cross section and heating data.
    
    Parameters
    ----------
    ace : Ace
        The Ace object to update with particle production cross section data
    debug : bool, optional
        Whether to print debug information, defaults to False
    """
    if debug:
        logger.debug("Reading particle production cross section data blocks")
        
    # Initialize container if not already present
    if not hasattr(ace, "particle_production_xs_data"):
        ace.particle_production_xs_data = ParticleProductionXSContainer()
    
    # Check if we have the necessary data
    if (not ace.header or not ace.header.jxs_array or not ace.xss_data or 
        not ace.secondary_particles or not ace.secondary_particles.has_data or
        not ace.particle_production_locators or not ace.particle_production_locators.has_data):
        if debug:
            logger.debug("Missing required data for particle production XS, skipping")
        return
    
    # Get the number of particle types
    ntype = ace.secondary_particles.num_secondary_particles
    
    if ntype <= 0:
        if debug:
            logger.debug("No secondary particle types defined, skipping")
        return  # No particle types defined
    
    if debug:
        logger.debug(f"Processing {ntype} secondary particle types")
    
    # Process each particle type
    for j in range(1, ntype + 1):
        if debug:
            logger.debug(f"Processing particle type {j}")
            
        # Get the locator set for this particle
        locator_set = ace.particle_production_locators.get_locators(j)
        if not locator_set:
            if debug:
                logger.debug(f"No locator set for particle type {j}, skipping")
            continue
        
        # Get the HPD index for this particle
        hpd_idx = locator_set.hpd
        if hpd_idx <= 0:
            if debug:
                logger.debug(f"Invalid HPD index {hpd_idx} for particle type {j}, skipping")
            continue
        
        if debug:
            logger.debug(f"HPD index for particle type {j}: {hpd_idx}")
        
        # Check if we have enough data to read the basic parameters
        if hpd_idx + 2 >= len(ace.xss_data):
            if debug:
                logger.debug(f"Not enough data at HPD index {hpd_idx} for particle type {j}, skipping")
            continue
        
        # Read the energy grid index (IE)
        ie = int(ace.xss_data[hpd_idx].value)
        
        # Read the number of consecutive energies (NE)
        ne = int(ace.xss_data[hpd_idx + 1].value)
        
        if debug:
            logger.debug(f"Particle type {j}: energy grid index={ie}, num energies={ne}")
        
        # Check if we have enough data for the cross sections and heating numbers
        if hpd_idx + 2 + 2*ne - 1 >= len(ace.xss_data):
            if debug:
                logger.debug(f"Not enough data for XS and heating values for particle type {j}, skipping")
            continue
        
        # Create a data object for this particle
        particle_data = ParticleProductionXSData(
            energy_grid_index=ie,
            num_energies=ne
        )
        
        # Read the cross section values - store XssEntry objects
        particle_data.xs_values = [ace.xss_data[hpd_idx + 2 + i] for i in range(ne)]
        
        # Read the heating numbers - store XssEntry objects
        particle_data.heating_numbers = [ace.xss_data[hpd_idx + 2 + ne + i] for i in range(ne)]
        
        # Add the data to the container
        ace.particle_production_xs_data.particle_data[j] = particle_data
        
        if debug:
            logger.debug(f"Successfully read XS data for particle type {j}: {ne} energy points")
    
    if ace.particle_production_xs_data.particle_data:
        ace.particle_production_xs_data.has_data = True
        if debug:
            num_particles = len(ace.particle_production_xs_data.particle_data)
            logger.debug(f"Finished reading particle production XS data for {num_particles} particles")
    elif debug:
        logger.debug("No particle production XS data found")
