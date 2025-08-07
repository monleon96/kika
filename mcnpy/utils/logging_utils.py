import logging
import sys

def configure_ace_debug_logging(enable=True, level=logging.DEBUG, modules=None):
    """
    Configure logging for MCNPy ACE parsers.
    
    Parameters
    ----------
    enable : bool, optional
        Whether to enable debug logging, defaults to True
    level : int, optional
        Logging level to use, defaults to DEBUG
    modules : list, optional
        Specific modules to enable logging for. If None, enables for all ACE parsers
        
    Returns
    -------
    logger : logging.Logger
        The configured root logger
    """
    # Clear existing handlers to avoid duplication
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    if not enable:
        root_logger.setLevel(logging.WARNING)
        return root_logger
    
    # Create a console handler with a nice formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Configure the root logger
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # Default modules to enable if none specified
    if modules is None:
        modules = [
            'mcnpy.ace.parsers.parse_tyr',
            'mcnpy.ace.parsers',
            'mcnpy.ace.parse_ace'
        ]
    
    # Set specific module loggers
    for module in modules:
        logging.getLogger(module).setLevel(level)
    
    return root_logger


def configure_endf_debug_logging(enable=True, level=logging.DEBUG, modules=None):
    """
    Configure logging for MCNPy ENDF parsers.
    
    Parameters
    ----------
    enable : bool, optional
        Whether to enable debug logging, defaults to True
    level : int, optional
        Logging level to use, defaults to DEBUG
    modules : list, optional
        Specific modules to enable logging for. If None, enables for all ENDF parsers
        
    Returns
    -------
    logger : logging.Logger
        The configured root logger
    """
    # Clear existing handlers to avoid duplication
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    if not enable:
        root_logger.setLevel(logging.WARNING)
        return root_logger
    
    # Create a console handler with a nice formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
    console_handler.setFormatter(formatter)
    
    # Default modules to enable if none specified
    if modules is None:
        modules = [
            'mcnpy.endf.parsers.parse_endf',
            'mcnpy.endf.parsers.parse_mf1',
            'mcnpy.endf.parsers.parse_mf4',
            'mcnpy.endf.parsers.parse_mf34',
        ]
        # Configure the root logger for all modules
        root_logger.setLevel(level)
        root_logger.addHandler(console_handler)
        
        # Set specific module loggers
        for module in modules:
            logging.getLogger(module).setLevel(level)
    else:
        # When specific modules are requested, set root logger to higher level
        # to suppress other debug messages, and configure individual loggers
        root_logger.setLevel(logging.WARNING)
        root_logger.addHandler(console_handler)
        
        # First, disable debug for all ENDF parsers
        all_endf_modules = [
            'mcnpy.endf.parsers.parse_endf',
            'mcnpy.endf.parsers.parse_mf1',
            'mcnpy.endf.parsers.parse_mf4',
            'mcnpy.endf.parsers.parse_mf34',
        ]
        for module in all_endf_modules:
            module_logger = logging.getLogger(module)
            module_logger.setLevel(logging.WARNING)  # Disable debug for all
            # Clear any existing handlers
            for handler in module_logger.handlers[:]:
                module_logger.removeHandler(handler)
            module_logger.propagate = True  # Let them propagate to root
        
        # Now enable only the requested modules
        for module in modules:
            module_logger = logging.getLogger(module)
            module_logger.setLevel(level)  # Enable debug for this module
            module_logger.propagate = True  # Let it propagate to root
    
    return root_logger


def get_endf_logger(name):
    """
    Get a logger for ENDF parsing with consistent formatting.
    
    Parameters
    ----------
    name : str
        The name of the logger (typically __name__)
        
    Returns
    -------
    logger : logging.Logger
        Configured logger instance
    """
    return logging.getLogger(name)
