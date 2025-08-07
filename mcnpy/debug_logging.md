# ENDF Parser Debug Logging

This document describes the consistent debug logging system implemented for all ENDF parsers in MCNPy.

## Overview

The ENDF parsers now use a unified logging system that allows you to:
- Toggle debug output on/off
- Control the verbosity level
- Enable debugging for specific parsers only
- Get consistent, well-formatted debug information

## Quick Start

```python
from mcnpy.utils import configure_endf_debug_logging

# Enable debug logging for all ENDF parsers
configure_endf_debug_logging(enable=True)

# Now parse your ENDF file - you'll see detailed debug output
from mcnpy.endf.parsers.parse_endf import parse_endf_file
endf_data = parse_endf_file("your_file.endf")
```

## API Reference

### `configure_endf_debug_logging(enable=True, level=logging.DEBUG, modules=None)`

Configures debug logging for ENDF parsers.

**Parameters:**
- `enable` (bool): Whether to enable debug logging (default: True)
- `level` (int): Logging level to use (default: logging.DEBUG)
- `modules` (list): Specific modules to enable logging for (default: all ENDF parsers)

**Returns:**
- `logger`: The configured root logger

### `get_endf_logger(name)`

Gets a logger for ENDF parsing with consistent formatting.

**Parameters:**
- `name` (str): The name of the logger (typically `__name__`)

**Returns:**
- `logger`: Configured logger instance

## Logging Levels

The system supports standard Python logging levels:

- `logging.DEBUG` (10): Most verbose, shows all parsing steps
- `logging.INFO` (20): General information about parsing progress
- `logging.WARNING` (30): Only warnings and errors
- `logging.ERROR` (40): Only errors
- `logging.CRITICAL` (50): Only critical errors

## Examples

### Enable Debug for All Parsers

```python
import logging
from mcnpy.utils import configure_endf_debug_logging

# Most verbose - see everything
configure_endf_debug_logging(enable=True, level=logging.DEBUG)
```

### Enable Debug for Specific Parsers Only

```python
# Only debug MF4 parser
modules = ['mcnpy.endf.parsers.parse_mf4']
configure_endf_debug_logging(enable=True, modules=modules)

# Debug MF4 and MF34 parsers
modules = ['mcnpy.endf.parsers.parse_mf4', 'mcnpy.endf.parsers.parse_mf34']
configure_endf_debug_logging(enable=True, modules=modules)

# Debug all ENDF parsers
modules = [
    'mcnpy.endf.parsers.parse_endf',
    'mcnpy.endf.parsers.parse_mf1', 
    'mcnpy.endf.parsers.parse_mf4',
    'mcnpy.endf.parsers.parse_mf34'
]
configure_endf_debug_logging(enable=True, modules=modules)
```

### Different Verbosity Levels

```python
# Less verbose - only general info
configure_endf_debug_logging(enable=True, level=logging.INFO)

# Even less verbose - only warnings and errors
configure_endf_debug_logging(enable=True, level=logging.WARNING)
```

### Disable Debug Logging

```python
# Turn off debug output
configure_endf_debug_logging(enable=False)
```

### Complete Example with Selective Parsing

```python
from mcnpy.utils import configure_endf_debug_logging
from mcnpy.endf.read_endf import read_endf
import logging

# Configure debug for MF4 parser only
modules = ['mcnpy.endf.parsers.parse_mf4']
configure_endf_debug_logging(enable=True, level=logging.DEBUG, modules=modules)

# Parse only MF4 to avoid debug output from other parsers
endf_data = read_endf("your_file.endf", mf_numbers=4)

# Now you'll see debug output only from the MF4 parser
```

## Debug Output Format

The debug messages follow a consistent format:

```
[LEVEL] module_name: message
```

Examples:
```
[DEBUG] mcnpy.endf.parsers.parse_endf: Starting to parse ENDF file: example.endf
[DEBUG] mcnpy.endf.parsers.parse_mf1: Parsing MF1 with 45 lines
[DEBUG] mcnpy.endf.parsers.parse_mf34: LB=5 processing: NE=20
[WARNING] mcnpy.endf.parsers.parse_mf4: Error parsing MT2 in MF4: Invalid data
```

## Available Parsers

The following ENDF parsers support the unified debug logging:

- `mcnpy.endf.parsers.parse_endf`: Main ENDF file parser
- `mcnpy.endf.parsers.parse_mf1`: MF1 (General Information) parser
- `mcnpy.endf.parsers.parse_mf4`: MF4 (Angular Distributions) parser
- `mcnpy.endf.parsers.parse_mf34`: MF34 (Angular Distribution Covariances) parser

## Best Practices

1. **Enable debug logging during development**: Helps identify parsing issues early
2. **Use appropriate logging levels**: DEBUG for detailed analysis, INFO for general progress
3. **Target specific parsers**: If you're only interested in certain MF sections
4. **Combine with selective parsing**: Use `read_endf(file, mf_numbers=[4])` to avoid parsing unwanted sections
5. **Disable in production**: Turn off debug logging for better performance
6. **Use the example notebook**: See `mcnpy/endf/classes/mf4/test_plots.ipynb` for practical usage patterns

## Important Notes

### Selective Parsing vs. Selective Debugging

There are two ways to limit debug output:

1. **Selective debugging**: Configure debug for specific parsers only
   ```python
   modules = ['mcnpy.endf.parsers.parse_mf4']
   configure_endf_debug_logging(enable=True, modules=modules)
   ```

2. **Selective parsing**: Parse only specific MF sections
   ```python
   endf_data = read_endf("file.endf", mf_numbers=4)  # Only parse MF4
   ```

**Best practice**: Use both together for cleanest output:
```python
# Configure debug for MF4 only
configure_endf_debug_logging(enable=True, modules=['mcnpy.endf.parsers.parse_mf4'])
# Parse MF4 only  
endf_data = read_endf("file.endf", mf_numbers=4)
```

This prevents debug output from parsers you don't want AND avoids parsing sections you don't need.

## Migration from Print Statements

If you have existing code that used print statements for debugging, here's how to migrate:

**Old way (inconsistent):**
```python
print(f"DEBUG - Parsing MT{mt}")
print(f"WARNING - Error in parsing")
```

**New way (consistent):**
```python
from mcnpy.utils import get_endf_logger
logger = get_endf_logger(__name__)

logger.debug(f"Parsing MT{mt}")
logger.warning(f"Error in parsing")
```

## Implementation Details

- All parsers use the same logger hierarchy under `mcnpy.endf.parsers.*`
- Debug messages are sent to stdout with clear formatting
- The system preserves existing warning mechanisms while adding debug capabilities
- Logging configuration is reset when calling `configure_endf_debug_logging` to avoid conflicts

## Troubleshooting

**Issue**: Debug messages not appearing
**Solution**: Make sure you've called `configure_endf_debug_logging(enable=True)` before parsing

**Issue**: Too much output
**Solution**: Use a higher logging level like `logging.INFO` or `logging.WARNING`

**Issue**: Only want debug for one parser
**Solution**: Use the `modules` parameter and selective parsing:
```python
# Configure debug for MF4 only
configure_endf_debug_logging(enable=True, modules=['mcnpy.endf.parsers.parse_mf4'])
# Parse only MF4
endf_data = read_endf("file.endf", mf_numbers=4)
```

**Issue**: Still seeing debug from other parsers
**Solution**: Make sure you're using selective parsing (`mf_numbers` parameter) in addition to selective debug configuration

For more examples, see `mcnpy/endf/classes/mf4/test_plots.ipynb`.
