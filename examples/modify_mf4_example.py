"""
Example script demonstrating how to modify MF4 sections in ENDF files.

This example shows how to:
1. Parse an ENDF file to extract MF4 data
2. Modify the angular distribution data
3. Write back the modified data to create a new ENDF file
"""

import os
from mcnpy.endf.parsers.parse_endf import parse_endf_file, parse_mf_from_file
from mcnpy.endf.writers.endf_writer import ENDFWriter, replace_mf_section, replace_mt_section


def example_modify_mf4_section():
    """
    Example: Modify an MF4 section and write it back to an ENDF file.
    """
    # Step 1: Define file paths
    original_file = "path/to/your/original.endf"
    modified_file = "path/to/your/modified.endf"
    
    # Step 2: Parse the original ENDF file to get MF4 data
    print("Parsing original ENDF file...")
    endf_data = parse_endf_file(original_file)
    
    # Get the MF4 section
    mf4 = endf_data.get_file(4)
    if mf4 is None:
        print("No MF4 section found in the file")
        return
    
    print(f"Found MF4 section with {len(mf4.sections)} MT sections")
    
    # Step 3: Modify the MF4 data
    # Example: Modify MT=2 (elastic scattering) if it exists
    if 2 in mf4.sections:
        mt2 = mf4.sections[2]
        print(f"Original MT2 type: {type(mt2).__name__}")
        
        # Example modification: Change a parameter (this depends on the specific MT type)
        if hasattr(mt2, '_awr'):
            original_awr = mt2._awr
            mt2._awr = original_awr * 1.001  # Small modification
            print(f"Modified AWR from {original_awr} to {mt2._awr}")
        
        # If it's a Legendre expansion, you could modify coefficients
        if hasattr(mt2, '_legendre_coeffs') and mt2._legendre_coeffs:
            print("Modifying Legendre coefficients...")
            for i, coeffs in enumerate(mt2._legendre_coeffs):
                # Example: Scale all coefficients by a small factor
                mt2._legendre_coeffs[i] = [c * 1.001 for c in coeffs]
    
    # Step 4: Write the modified MF4 section back to a new ENDF file
    print("Writing modified ENDF file...")
    
    # Method 1: Replace the entire MF4 section
    success = replace_mf_section(original_file, mf4, modified_file)
    
    if success:
        print(f"Successfully created modified file: {modified_file}")
    else:
        print("Failed to create modified file")
        return
    
    # Step 5: Verify the modification
    print("Verifying modifications...")
    modified_endf = parse_endf_file(modified_file)
    modified_mf4 = modified_endf.get_file(4)
    
    if modified_mf4 and 2 in modified_mf4.sections:
        modified_mt2 = modified_mf4.sections[2]
        if hasattr(modified_mt2, '_awr'):
            print(f"Verified: Modified AWR = {modified_mt2._awr}")


def example_modify_specific_mt_section():
    """
    Example: Modify a specific MT section within MF4.
    """
    original_file = "path/to/your/original.endf"
    modified_file = "path/to/your/modified_mt.endf"
    
    # Parse just the MF4 section
    print("Parsing MF4 section...")
    mf4 = parse_mf_from_file(original_file, 4)
    
    if mf4 is None or 2 not in mf4.sections:
        print("MF4/MT2 section not found")
        return
    
    # Get the specific MT section
    mt2 = mf4.sections[2]
    print(f"Found MT2 section: {type(mt2).__name__}")
    
    # Modify the MT section
    if hasattr(mt2, '_za'):
        print(f"Original ZA: {mt2._za}")
        # Don't actually change ZA as it identifies the isotope
        # mt2._za = mt2._za  # Keep same
    
    # Method 2: Replace just the specific MT section
    success = replace_mt_section(original_file, mt2, 4, modified_file)
    
    if success:
        print(f"Successfully modified MT2 in: {modified_file}")
    else:
        print("Failed to modify MT2 section")


def example_using_endf_writer_class():
    """
    Example: Using the ENDFWriter class for more control.
    """
    original_file = "path/to/your/original.endf"
    modified_file = "path/to/your/writer_modified.endf"
    
    # Create an ENDFWriter instance
    writer = ENDFWriter(original_file)
    
    # Find all MF4 sections in the file
    mf4_boundaries = writer.find_mf_boundaries(4)
    print(f"Found MF4 sections at line ranges: {mf4_boundaries}")
    
    # Find specific MT sections within MF4
    mt2_boundaries = writer.find_mt_boundaries_in_mf(4, 2)
    print(f"Found MF4/MT2 sections at line ranges: {mt2_boundaries}")
    
    # Parse and modify
    mf4 = parse_mf_from_file(original_file, 4)
    if mf4:
        # Make some modification
        for mt_num, mt_section in mf4.sections.items():
            if hasattr(mt_section, '_awr'):
                mt_section._awr = mt_section._awr * 1.0001
        
        # Replace using the writer instance
        success = writer.replace_mf_section(mf4, modified_file)
        print(f"Replacement {'succeeded' if success else 'failed'}")


def example_batch_modification():
    """
    Example: Batch modify multiple files.
    """
    input_dir = "path/to/input/files"
    output_dir = "path/to/output/files"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all ENDF files in input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.endf'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"modified_{filename}")
            
            print(f"Processing {filename}...")
            
            try:
                # Parse the file
                endf_data = parse_endf_file(input_path)
                mf4 = endf_data.get_file(4)
                
                if mf4:
                    # Apply consistent modifications to all files
                    for mt_num, mt_section in mf4.sections.items():
                        if hasattr(mt_section, '_awr'):
                            # Apply some systematic correction
                            mt_section._awr = mt_section._awr * 1.001
                    
                    # Write modified file
                    success = replace_mf_section(input_path, mf4, output_path)
                    print(f"  {'✓' if success else '✗'} {filename}")
                else:
                    print(f"  - No MF4 section in {filename}")
                    
            except Exception as e:
                print(f"  ✗ Error processing {filename}: {e}")


if __name__ == "__main__":
    print("ENDF MF4 Modification Examples")
    print("=" * 50)
    
    # Note: Update the file paths in the examples above before running
    print("To run these examples:")
    print("1. Update the file paths in each function")
    print("2. Uncomment the desired example function call below")
    print("3. Run this script")
    
    # Uncomment the example you want to run:
    # example_modify_mf4_section()
    # example_modify_specific_mt_section()
    # example_using_endf_writer_class()
    # example_batch_modification()
