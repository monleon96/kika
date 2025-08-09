#!/usr/bin/env python3
"""
Example script demonstrating ENDF perturbation with MF34 angular covariance data.

This script shows how to use the new endf_perturbation module to perturb 
ENDF angular distribution data using MF34 covariance matrices.
"""

import os
import sys
sys.path.insert(0, '.')

from mcnpy.sampling.endf_perturbation import perturb_ENDF_files

def example_endf_perturbation():
    """
    Example usage of ENDF perturbation functionality.
    
    This function demonstrates how to:
    1. Load ENDF files containing MF4 (angular distributions)
    2. Load ENDF files containing MF34 (angular covariance matrices)
    3. Generate perturbed ENDF files using the covariance data
    """
    
    # Example parameters - adjust these for your actual data
    endf_files = [
        "path/to/your/endf_file_with_mf4.endf"  # ENDF file containing MF4 angular distributions
    ]
    
    mf34_cov_files = [
        "path/to/your/endf_file_with_mf34.endf"  # ENDF file containing MF34 covariance data
    ]
    
    # Specify which MT reactions to perturb (empty list = all available)
    mt_list = [2]  # Example: elastic scattering (MT=2)
    
    # Specify which Legendre coefficients to perturb
    legendre_coeffs = [0, 1, 2]  # L=0 (isotropic), L=1, L=2
    
    # Number of perturbed samples to generate
    num_samples = 10
    
    # Output directory
    output_dir = "./endf_perturbation_output"
    
    # Run the perturbation
    try:
        perturb_ENDF_files(
            endf_files=endf_files,
            mf34_cov_files=mf34_cov_files,
            mt_list=mt_list,
            legendre_coeffs=legendre_coeffs,
            num_samples=num_samples,
            space="log",  # Use log space for perturbations
            decomposition_method="svd",  # Use SVD decomposition
            sampling_method="sobol",  # Use Sobol sampling
            output_dir=output_dir,
            seed=42,  # For reproducible results
            dry_run=False,  # Set to True to only generate factors without ENDF files
            verbose=True
        )
        print(f"ENDF perturbation completed successfully!")
        print(f"Results saved to: {output_dir}")
        
    except FileNotFoundError as e:
        print(f"Error: Input file not found - {e}")
        print("Please update the file paths in this example script to point to your actual ENDF files.")
        
    except Exception as e:
        print(f"Error during ENDF perturbation: {e}")

def example_dry_run():
    """
    Example of running in dry-run mode to only generate perturbation factors.
    """
    
    # Same parameters as above but with dry_run=True
    endf_files = ["path/to/your/endf_file_with_mf4.endf"]
    mf34_cov_files = ["path/to/your/endf_file_with_mf34.endf"]
    
    try:
        perturb_ENDF_files(
            endf_files=endf_files,
            mf34_cov_files=mf34_cov_files,
            mt_list=[2],
            legendre_coeffs=[0, 1, 2],
            num_samples=5,
            output_dir="./endf_dry_run_output",
            dry_run=True,  # Only generate factors, no ENDF files
            verbose=True
        )
        print("Dry run completed - only perturbation factors generated!")
        
    except Exception as e:
        print(f"Error during dry run: {e}")

if __name__ == "__main__":
    print("ENDF Perturbation Example")
    print("=" * 40)
    
    print("\\n1. Full perturbation example:")
    example_endf_perturbation()
    
    print("\\n2. Dry run example:")
    example_dry_run()
    
    print("\\nKey features of the ENDF perturbation module:")
    print("- Reuses existing infrastructure (MF34CovMat, generators, etc.)")
    print("- Uses ENDF parser to load MF34 covariance data automatically")
    print("- No autofix applied to covariance matrices (uses as-is)")
    print("- Generates perturbed ENDF files with modified MF4 angular coefficients")
    print("- Outputs: perturbed ENDF files, Parquet with all factors, per-sample summaries")
    print("- Supports dry-run mode for testing")
    print("- Consistent logging with ACE perturbation module")
