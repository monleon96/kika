import numpy as np
from mcnpy._constants import MT_TO_REACTION
from mcnpy.cov.covmat import CovMat
from mcnpy.energy_grids.grids import SCALE44, SCALE56, SCALE238, SCALE252
import re
import pandas as pd

class EmptyParsingError(Exception):
    """Raised when no data was extracted during parsing."""
    pass

class InvalidDataFormatError(Exception):
    """Raised when the data format is invalid or corrupted."""
    pass

def read_scale_covmat(file_path: str):
    """
    Read a SCALE covariance matrix file and convert it to a CovMat object.
    
    Parameters
    ----------
    file_path : str
        Path to the SCALE covariance matrix text file
        
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
    covmat.cov_type = "SCALE"

    # Determine the energy grid based on the number of groups
    potential_grids = {
        len(SCALE44) - 1: SCALE44,
        len(SCALE56) - 1: SCALE56,
        len(SCALE238) - 1: SCALE238,
        len(SCALE252) - 1: SCALE252,
    }
    
    if num_groups in potential_grids:
        covmat.energy_grid = potential_grids[num_groups]
    # else: covmat.energy_grid remains None

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
                    
                    # Flip the matrix to get ascending order
                    matrix = np.flipud(np.fliplr(matrix))

                    # Add to CovMat object
                    covmat.add_matrix(isotope_row, reaction_row, isotope_col, reaction_col, matrix)
            except (ValueError, IndexError):
                # Skip lines with invalid data
                continue

    # Verify we found at least some valid data
    if covmat.num_matrices == 0:
        raise EmptyParsingError(f"No valid data was extracted from the covariance matrix file: {file_path}")

    return covmat

def read_njoy_covmat(file_path: str):
    """
    Read an NJOY covariance matrix file and convert it to a CovMat object.

    Parameters
    ----------
    file_path : str
        Path to the NJOY covariance matrix text file

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
    dikt_cov = {'ISO_H':[], 'REAC_H':[],'ISO_V':[], 'REAC_V':[],  'STD': []}

    with open(file_path, 'r') as f:
        lines = f.readlines()
    iMAT = lines[2][66:70]
    iMAT1 = iMAT

    group_nb = int(lines[2].split()[2])

    grep_data = False
    val_tot_nb, start_x_idx, start_y_idx = None, None, None
    vals = []
    i_line = 0
    energymesh = None

    while i_line < (len(lines)-4):
        i_line += 1
        line = lines[i_line]
        splited_part = [line[i*11:(i+1)*11].replace(' ','') for i in range(6)]
        splited_part = [x for x in splited_part[:] if x != '']

        infos_part = line[66:]
        iMAT, iMF, iMT, data_progress_id = infos_part[:4], str(int(infos_part[4:6])), str(int(infos_part[6:9])), infos_part[9:14]

        # If SEND, read next line
        if (iMAT, iMF) != ('0', '0') and iMT == '0': continue
        # If FEND, read next line
        if iMAT != '0' and (iMF, iMT) == ('0', '0'): continue

        # If reading info RECORD MF1 MT451
        if iMAT != '0' and iMF == '1' and iMT == '451':
            i_line+=1
            line = lines[i_line]
            LIST_MF1451 = [line[i*11:(i+1)*11].replace(' ','') for i in range(6)]
            LIST_MF1451 = [x for x in LIST_MF1451[:] if x != '']
            energymesh = []
            while len(energymesh) < int(LIST_MF1451[4]):
                i_line+=1
                line = lines[i_line]
                energylist = [line[i*11:(i+1)*11].replace(' ','') for i in range(6)]
                energylist = [x for x in energylist[:] if x != '']
                for energy in energylist:
                    if re.search('-', energy[-3:]):
                        valE = energy[:-3] + energy[-3:].split('-')[0]+'E-'+energy[-3:].split('-')[1]
                        valE = float(valE)
                    elif re.search('\+', energy):
                        valE = energy[:-3] + energy[-3:].split('+')[0]+'E+'+energy[-3:].split('+')[1]
                        valE = float(valE)
                    else:
                        valE = float(energy)
                    energymesh.append(valE)
            dikt_cov['ISO_H'].append('0')
            dikt_cov['REAC_H'].append('0')
            dikt_cov['ISO_V'].append('0')
            dikt_cov['REAC_V'].append('0')
            dikt_cov['STD'].append(energymesh)
            continue

        # If reading cross sections MF3 MTxxx
        crossSectionLine = []
        if iMAT != '0' and iMF == '3' and iMT != '0':
            while len(crossSectionLine) < int(splited_part[4]):
                i_line+=1
                line = lines[i_line]
                LIST_MF3 = [line[i*11:(i+1)*11].replace(' ','') for i in range(6)]
                LIST_MF3 = [x for x in LIST_MF3[:] if x != '']
                for CrossSec_str in LIST_MF3:
                    if re.search('-', CrossSec_str[-3:]):
                        valXS = CrossSec_str[:-3] + CrossSec_str[-3:].split('-')[0]+'E-'+CrossSec_str[-3:].split('-')[1]
                        valXS = float(valXS)
                    elif re.search('\+', CrossSec_str):
                        valXS = CrossSec_str[:-3] + CrossSec_str[-3:].split('+')[0]+'E+'+CrossSec_str[-3:].split('+')[1]
                        valXS = float(valXS)
                    else:
                        valXS = float(CrossSec_str)
                    crossSectionLine.append(valXS)
            dikt_cov['ISO_H'].append(iMAT)
            dikt_cov['REAC_H'].append(iMT)
            dikt_cov['ISO_V'].append('0')
            dikt_cov['REAC_V'].append('0')
            dikt_cov['STD'].append(crossSectionLine)
            continue

        if len(splited_part) > 4 and splited_part[2] == '0' and splited_part[4] == '0':
            reac_2_id = splited_part[3]
            grep_data = True
            sub_mat = np.zeros((group_nb, group_nb))
            continue

        elif grep_data:
            if (val_tot_nb, start_x_idx, start_y_idx) == (None, None, None):
                val_tot_nb = int(splited_part[2])
                start_x_idx = int(splited_part[3]) - 1
                start_y_idx = int(splited_part[5]) - 1
                continue

            for val_str in splited_part:
                if re.search('-', val_str[-3:]):
                    val = val_str[:-3] + val_str[-3:].split('-')[0]+'E-'+val_str[-3:].split('-')[1]
                    val = float(val)
                elif re.search('\+', val_str):
                    val = val_str[:-3] + val_str[-3:].split('+')[0]+'E+'+val_str[-3:].split('+')[1]
                    val = float(val)
                else:
                    val = float(val_str)
                vals.append(val)

            if len(vals) == val_tot_nb:
                sub_mat[start_y_idx][start_x_idx : start_x_idx+val_tot_nb] = vals
                val_tot_nb, start_x_idx, start_y_idx = None, None, None
                vals = []
                if (len(lines[i_line+1].split()) < 3 or lines[i_line+1].split()[2] == '0'):
                    if not sum(np.array(sub_mat[:]).flatten()) == 0.0:
                        dikt_cov['ISO_H'].append(iMAT)
                        dikt_cov['REAC_H'].append(iMT)
                        dikt_cov['ISO_V'].append(iMAT1)
                        dikt_cov['REAC_V'].append(reac_2_id)
                        dikt_cov['STD'].append(sub_mat.tolist())
                    grep_data = False
                continue

    # Now, build CovMat object
    covmat = CovMat(num_groups=group_nb)
    covmat.cov_type = "NJOY"
    # Set energy grid if found
    if len(dikt_cov['STD']) > 0 and isinstance(dikt_cov['STD'][0], list) and len(dikt_cov['STD'][0]) == group_nb+1:
        # If energy grid is present as first row
        covmat.energy_grid = dikt_cov['STD'][0]
        start_idx = 1
    elif energymesh is not None:
        covmat.energy_grid = energymesh
        start_idx = 0
    else:
        start_idx = 0

    # Add matrices to CovMat
    for idx in range(start_idx, len(dikt_cov['STD'])):
        try:
            iso_h = int(str(dikt_cov['ISO_H'][idx]).strip())
            reac_h = int(str(dikt_cov['REAC_H'][idx]).strip())
            iso_v = int(str(dikt_cov['ISO_V'][idx]).strip())
            reac_v = int(str(dikt_cov['REAC_V'][idx]).strip())
            matrix = np.array(dikt_cov['STD'][idx])
            if matrix.shape == (group_nb, group_nb):
                covmat.add_matrix(iso_h, reac_h, iso_v, reac_v, matrix)
        except Exception:
            continue

    if covmat.num_matrices == 0:
        raise EmptyParsingError(f"No valid data was extracted from the NJOY covariance matrix file: {file_path}")

    return covmat