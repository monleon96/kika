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
