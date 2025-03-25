import re
import logging

# Setup logger
logger = logging.getLogger(__name__)

def read_header(header, lines, debug=False):
    """
    Read the complete ACE header (Opening, IZAW, NXS, JXS arrays).
    
    Parameters
    ----------
    header : Header
        The Header object to populate
    lines : list
        List of file lines containing the header
    debug : bool, optional
        Whether to print debug information, defaults to False
        
    Returns
    -------
    int
        The index of the next line to read after the header
    """
    if debug:
        logger.debug("Reading ACE header")
        
    line_idx = 0
    
    # Read header opening based on format
    if header.format_version == "legacy":
        # Legacy header format (Table 1)
        if debug:
            logger.debug("Reading legacy format header")
            
        if len(lines) < 2:
            raise ValueError("Not enough lines for legacy header")
            
        line1 = lines[line_idx]
        line_idx += 1
        line2 = lines[line_idx]
        line_idx += 1
        
        # Parse according to the format in Table 1
        try:
            # Split the line by spaces to extract components more reliably
            line1_parts = line1.split()
            
            # Must have at least 4 parts: ZAID, AWR, Temp, Date
            if len(line1_parts) < 4:
                raise ValueError(f"Invalid legacy header format, too few elements: {line1}")
            
            # Extract ZAID (first part) with extension
            zaid_str = line1[:20].strip()
            # Match numeric part and possible extension
            za_match = re.match(r'\s*(\d+)(\.\d+[a-z]*)?', zaid_str)
            if za_match:
                header.zaid = int(za_match.group(1))
                header.extension = za_match.group(2) if za_match.group(2) else None
            else:
                raise ValueError(f"Could not extract ZA from ZAID: {zaid_str}")
            
            # Extract AWR (second part)
            header.atomic_weight_ratio = float(line1_parts[1].strip())
            
            # Extract temperature (third part)
            header.temperature = float(line1_parts[2].strip())
            
            # Extract date (fourth part)
            header.date = line1_parts[3].strip()
            
            # Process the second line for comment and matid
            header.comment = line2[:70].strip()
            matid_str = line2[70:].strip()

            # Use a regular expression to match 'mat' followed by optional whitespace and then 3 or 4 digits
            match = re.fullmatch(r'mat\s*(\d{3,4})', matid_str)
            if match:
                header.matid = int(match.group(1))
            else:
                # If the format doesn't match, set matid to None or handle the error as needed
                header.matid = None
                
            if debug:
                logger.debug(f"Parsed legacy header: ZAID={header.zaid}, AWR={header.atomic_weight_ratio}, "
                             f"Temp={header.temperature}, Date={header.date}")
            
        except (ValueError, IndexError) as e:
            raise ValueError(f"Error parsing legacy header: {e}")
    else:
        # 2.0.1 header format (Table 2)
        if debug:
            logger.debug("Reading 2.0.1 format header")
            
        if len(lines) < 2:
            raise ValueError("Not enough lines for 2.0.1 header")
            
        line1 = lines[line_idx]
        line_idx += 1
        line2 = lines[line_idx]
        line_idx += 1
        
        # Parse according to the format in Table 2
        try:
            header.ace_version = line1[:10].strip()  # Version format string
            
            zaid_str = line1[10:22].strip()  # ZAID
            # Extract the numeric part of ZAID and extension using regex
            za_match = re.match(r'(\d+)(\.\d+[a-z]*)?', zaid_str)
            if za_match:
                header.zaid = int(za_match.group(1))
                header.extension = za_match.group(2) if za_match.group(2) else None
            else:
                raise ValueError(f"Could not extract ZA from ZAID: {zaid_str}")
                
            header.source = line1[22:46].strip()  # Evaluation source
            
            header.atomic_weight_ratio = float(line2[:12].strip())
            header.temperature = float(line2[12:24].strip())
            header.comment_line_count = int(line2[24:32].strip())
            
            src = line2[32:].strip()  # Evaluation source
            if src:  # Only set if not empty
                header.source = src
                
            if debug:
                logger.debug(f"Parsed 2.0.1 header: ZAID={header.zaid}, AWR={header.atomic_weight_ratio}, "
                             f"Temp={header.temperature}, Version={header.ace_version}")
                logger.debug(f"Comment lines: {header.comment_line_count}")
                
            # Read the comment lines if present
            comment_lines = []
            for _ in range(header.comment_line_count):
                if line_idx < len(lines):
                    comment_lines.append(lines[line_idx].strip())
                    line_idx += 1
                else:
                    break
            
            header.comment = '\n'.join(comment_lines)
        except ValueError as e:
            raise ValueError(f"Error parsing 2.0.1 header: {e}")
    
    # -------------------------
    # Read IZAW array (16 pairs over 4 lines)
    # Format for each line: 4(I7,F11.0)
    # -------------------------
    if debug:
        logger.debug("Reading IZAW array")
        
    izaw = []
    
    for i in range(4):  # 4 lines of IZAW data
        if line_idx < len(lines):
            line = lines[line_idx]
            line_idx += 1
            
            for j in range(4):  # 4 pairs per line
                start_idx = j * 18
                if start_idx + 18 <= len(line):
                    za_str = line[start_idx:start_idx+7].strip()
                    awr_str = line[start_idx+7:start_idx+18].strip()
                    
                    za = int(za_str) if za_str else 0
                    awr = float(awr_str) if awr_str else 0.0
                    izaw.append((za, awr))
    
    header.izaw_array = izaw
    
    if debug and izaw:
        logger.debug(f"Read {len(izaw)} IZAW entries")
    
    # -------------------------
    # Read NXS array (16 integers over 2 lines)
    # Format for each line: 8I9
    # -------------------------
    if debug:
        logger.debug("Reading NXS array")
        
    nxs = []
    
    for i in range(2):  # 2 lines of NXS data
        if line_idx < len(lines):
            line = lines[line_idx]
            line_idx += 1
            
            for j in range(8):  # 8 integers per line
                start_idx = j * 9
                if start_idx + 9 <= len(line):
                    value_str = line[start_idx:start_idx+9].strip()
                    value = int(value_str) if value_str else 0
                    nxs.append(value)
    
    header.nxs_array = nxs
    
    if debug and nxs:
        logger.debug(f"Read {len(nxs)} NXS entries")
        logger.debug(f"NXS values: {nxs}")
    
    # -------------------------
    # Read JXS array (32 integers over 4 lines)
    # Format for each line: 8I9
    # -------------------------
    if debug:
        logger.debug("Reading JXS array")
        
    jxs = []
    
    for i in range(4):  # 4 lines of JXS data
        if line_idx < len(lines):
            line = lines[line_idx]
            line_idx += 1
            
            for j in range(8):  # 8 integers per line
                start_idx = j * 9
                if start_idx + 9 <= len(line):
                    value_str = line[start_idx:start_idx+9].strip()
                    value = int(value_str) if value_str else 0
                    jxs.append(value)
    
    header.jxs_array = jxs
    
    if debug and jxs:
        logger.debug(f"Read {len(jxs)} JXS entries")
        logger.debug(f"JXS non-zero pointers: {[(i+1, val) for i, val in enumerate(jxs) if val > 0]}")
    
    if debug:
        logger.debug("Finished reading header")
        
    return line_idx
