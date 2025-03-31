import numpy as np
from mcnpy._constants import MT_TO_REACTION
from mcnpy.cov.covmat import CovMat

class EmptyParsingError(Exception):
    """Raised when no data was extracted during parsing."""
    pass

class InvalidDataFormatError(Exception):
    """Raised when the data format is invalid or corrupted."""
    pass

def read_scale_covmat(file_path: str, output_path: str = None):
    """
    Read a SCALE covariance matrix file and convert it to a CovMat object.
    
    Parameters
    ----------
    file_path : str
        Path to the SCALE covariance matrix text file
    output_path : str, optional
        Path to save the output as Excel, defaults to None
        
    Returns
    -------
    CovMat
        CovMat object containing the parsed covariance data
    
    Raises
    ------
    EmptyParsingError
        If no data was extracted from the file
    FileNotFoundError
        If the input file does not exist
    """
    # Read the file
    with open(file_path, "r") as f:
        file_lines = f.readlines()

    # Parse the group number from the second line
    num_groups = int(file_lines[1].split()[0])
    
    # Create CovMat object
    covmat = CovMat(num_groups)

    # Parse the file
    for i, line in enumerate(file_lines):
        if i > 2 and len(line.split()) == 5:
            try:
                # Parse isotope and reaction numbers
                reaction_row = int(line.split()[1])
                reaction_col = int(line.split()[3])
                
                if (reaction_row != 1 and reaction_col != 1 and 
                    reaction_row in MT_TO_REACTION and reaction_col in MT_TO_REACTION):
                    
                    isotope_row = int(line.split()[0])
                    isotope_col = int(line.split()[2])
                    
                    # Read matrix values
                    matrix_values = []
                    values_read = 0
                    j = 0
                    while values_read < num_groups * num_groups:
                        for val in file_lines[i + 1 + j].split():
                            matrix_values.append(float(val))
                        values_read += len(file_lines[i + 1 + j].split())
                        j += 1

                    # Convert to numpy array and reshape
                    matrix = np.array(matrix_values).reshape(num_groups, num_groups)
                    
                    # Add to CovMat object
                    covmat.add_matrix(isotope_row, reaction_row, isotope_col, reaction_col, matrix)
            except (ValueError, IndexError):
                # Skip lines with invalid data
                continue

    # Verify we found at least some valid data
    if covmat.num_matrices == 0:
        raise EmptyParsingError(f"No valid data was extracted from the covariance matrix file: {file_path}")

    # Save the output if requested
    if output_path is not None:
        covmat.save_excel(output_path)

    return covmat