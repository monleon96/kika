import logging
from typing import List
from mcnpy.ace.classes.nubar.nubar import NuData, NuPolynomial, NuTabulated, NuContainer
from mcnpy.ace.xss import XssEntry

# Setup logger
logger = logging.getLogger(__name__)

def read_nubar_data(ace, debug=False):
    """
    Read both NU and DNU blocks from the XSS array if they exist.
    
    Parameters
    ----------
    ace : Ace
        The Ace object with XSS data and header
    debug : bool, optional
        Whether to print debug information, defaults to False
    """
    if (ace.header is None or ace.header.jxs_array is None or 
        ace.xss_data is None):
        raise ValueError("Cannot read nubar data: header or XSS data missing")
    
    # Create a new NuContainer if not present
    if ace.nubar is None:
        ace.nubar = NuContainer()
    
    if debug:
        logger.debug("\n===== NUBAR BLOCK PARSING =====")
        logger.debug(f"Header info: ZAID={ace.header.zaid}")
    
    # Read NU block (fission nubar data) if present
    jxs2 = ace.header.jxs_array[1]  # JXS(2)
    
    if debug:
        logger.debug(f"JXS(2) = {jxs2} → Locator for NU block (FORTRAN 1-indexed)")
    
    if jxs2 > 0:
        # Set flag indicating NU data is present
        ace.nubar.has_nubar = True
        
        if debug:
            logger.debug(f"NU block found at index {jxs2}")
        
        # Check if we have both prompt and total nubar data
        # Get the value at XSS(JXS(2))
        xss_jxs2 = ace.xss_data[jxs2 - 1]  # Convert to 0-indexed
        
        # Extract the value attribute before comparison
        xss_jxs2_value = xss_jxs2.value if hasattr(xss_jxs2, 'value') else xss_jxs2
        
        if xss_jxs2_value < 0:
            # Both prompt and total nubar data are present (Case 2)
            ace.nubar.has_both_nu_types = True
            
            if debug:
                logger.debug(f"Both prompt and total nubar data present (XSS[{jxs2-1}] = {xss_jxs2_value} < 0)")
            
            # Prompt nu array begins at XSS(KNU) where KNU = JXS(2) + 1
            prompt_nu_idx = jxs2  # Already converted to 0-indexed (+1 -1 = 0)
            
            # Total nu array begins at XSS(KNU) where KNU = JXS(2) + ABS(XSS(JXS(2))) + 1
            # Make sure to use the value attribute for abs()
            total_nu_idx = jxs2 + int(abs(xss_jxs2_value))  # Already converted to 0-indexed
            
            if debug:
                logger.debug(f"Prompt nubar data starts at index {prompt_nu_idx}")
                logger.debug(f"Total nubar data starts at index {total_nu_idx}")
            
            # Parse prompt nubar data
            ace.nubar.prompt = parse_nubar_array(ace.xss_data, prompt_nu_idx, debug)
            
            # Parse total nubar data
            ace.nubar.total = parse_nubar_array(ace.xss_data, total_nu_idx, debug)
        else:
            # Only one type of nubar data is present (Case 1)
            ace.nubar.has_both_nu_types = False
            
            if debug:
                logger.debug(f"Only one nubar data type present (XSS[{jxs2-1}] = {xss_jxs2_value} ≥ 0)")
            
            # A single nu array begins at location XSS(KNU) where KNU = JXS(2)
            single_nu_idx = jxs2 - 1  # Convert to 0-indexed
            
            if debug:
                logger.debug(f"Nubar data starts at index {single_nu_idx}")
            
            # Parse the single nubar array
            ace.nubar.total = parse_nubar_array(ace.xss_data, single_nu_idx, debug)
            # In this case, we assume it's total nubar (most common)
            # The user would need to check the evaluations to confirm this
    
    # Read DNU block (delayed fission nubar data) if present
    dnu_idx = ace.header.jxs_array[23]  # JXS(24)
    
    if debug:
        logger.debug("\n===== DELAYED NUBAR BLOCK PARSING =====")
        logger.debug(f"JXS(24) = {dnu_idx} → Locator for DNU block (FORTRAN 1-indexed)")
    
    if dnu_idx > 0:
        # Set flag indicating DNU data is present
        ace.nubar.has_delayed = True
        
        if debug:
            logger.debug(f"DNU block found at index {dnu_idx}")
        
        # Convert to 0-indexed
        dnu_idx -= 1
        
        if dnu_idx >= len(ace.xss_data):
            raise ValueError(f"Invalid DNU block index: {dnu_idx+1}")
        
        # Parse the delayed nubar data (always in tabulated form)
        ace.nubar.delayed = parse_nubar_array(ace.xss_data, dnu_idx, debug)
        
        # Validate that it's in tabulated form (LNU = 2)
        if ace.nubar.delayed.format != "tabulated":
            raise ValueError("Delayed nubar data must be in tabulated form (LNU = 2)")

def parse_nubar_array(xss_data: List[XssEntry], idx: int, debug=False) -> NuData:
    """
    Parse a single nubar array from the XSS data.
    
    Parameters
    ----------
    xss_data : List[float]
        The XSS data array
    idx : int
        Starting index in the XSS array for the nubar data
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    NuData
        A NuData object containing the parsed data
    """
    if idx < 0 or idx >= len(xss_data):
        raise ValueError(f"Invalid nubar array index: {idx}")
    
    # Create a NuData object to hold the result
    result = NuData()
    
    # Get the format flag (LNU)
    format_int = int(xss_data[idx].value)
    if format_int == 1:
        result.format = "polynomial"
    elif format_int == 2:
        result.format = "tabulated"
    else:
        raise ValueError(f"Unknown nubar format flag (LNU): {format_int}")
    
    if debug:
        logger.debug(f"Nubar format (LNU): {format_int} → {result.format}")
    
    if result.format == "polynomial":  # Polynomial form (LNU = 1)
        n_coeff = int(xss_data[idx + 1].value)  # Number of coefficients
        
        if debug:
            logger.debug(f"Polynomial form with {n_coeff} coefficients")
        
        # Create a NuPolynomial object and populate it
        result.polynomial = NuPolynomial()
        result.polynomial.coefficients = list(xss_data[idx + 2:idx + 2 + n_coeff])
        
    elif result.format == "tabulated":  # Tabulated form (LNU = 2)
        n_regions = int(xss_data[idx + 1].value)  # Number of interpolation regions
        
        if debug:
            logger.debug(f"Tabulated form with {n_regions} interpolation regions")
        
        # Create a NuTabulated object
        result.tabulated = NuTabulated()
        
        # Current position in the XSS array
        pos = idx + 2
        
        # Read interpolation regions if present
        if n_regions > 0:
            # Read NBT array
            nbt = [int(x.value) for x in xss_data[pos:pos + n_regions]]
            pos += n_regions
            
            # Read INT array
            interp = [int(x.value) for x in xss_data[pos:pos + n_regions]]
            pos += n_regions
            
            # Store interpolation regions
            result.tabulated.interpolation_regions = list(zip(nbt, interp))
            
            if debug:
                logger.debug(f"Interpolation regions: {result.tabulated.interpolation_regions}")
        
        # Read number of energy points
        n_energies = int(xss_data[pos].value)
        pos += 1
        
        if debug:
            logger.debug(f"Number of energy points: {n_energies}")
        
        # Read energy points
        result.tabulated.energies = xss_data[pos:pos + n_energies]
        pos += n_energies
        
        # Read nubar values
        result.tabulated.nubar_values = xss_data[pos:pos + n_energies]
    
    return result
