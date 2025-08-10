#!/usr/bin/env python3
"""
Example usage of the updated perturb_ACE_files function.

This example shows how to use the modified function that applies perturbations
to existing ACE files created by the ENDF perturbation process.
"""

from mcnpy.sampling.ace_perturbation_separate import perturb_ACE_files

def example_usage():
    """
    Example usage of the updated perturb_ACE_files function.
    """
    
    # Example parameters - adjust these to match your actual setup
    root_dir = "/path/to/your/output_directory"  # Root directory containing ace/ subdirectory
    temperatures = [300.0, 600.0, 900.0]  # Temperatures in Kelvin
    zaids = [92235, 92238, 94239]  # List of ZAID numbers to process
    cov_files = "/path/to/covariance_matrix.cov"  # Single covariance file for all ZAIDs
    # Or use multiple covariance files: cov_files = ["file1.cov", "file2.cov", "file3.cov"]
    
    mt_list = [1, 2, 18, 102]  # MT numbers to perturb, or [] for all available
    num_samples = 100  # Number of perturbation samples (must match existing ACE files)
    
    # Call the updated function
    perturb_ACE_files(
        root_dir=root_dir,
        temperatures=temperatures,
        zaids=zaids,
        cov_files=cov_files,
        mt_list=mt_list,
        num_samples=num_samples,
        space="log",  # "linear" or "log"
        decomposition_method="svd",  # "svd", "cholesky", "eigen", or "pca"
        sampling_method="sobol",  # "sobol", "lhs", or "random"
        seed=42,  # For reproducible results
        nprocs=4,  # Number of parallel processes
        dry_run=False,  # Set to True to only generate factors without modifying files
        autofix="soft",  # Covariance matrix autofix level: None, "soft", "medium", "hard"
        verbose=True
    )

def directory_structure_info():
    """
    Information about the expected directory structure.
    """
    print("""
    Expected directory structure for ACE files:
    
    root_dir/
    ├── ace/
    │   ├── 300.0K/
    │   │   ├── 92235/
    │   │   │   ├── 0001/
    │   │   │   │   └── filename.ace
    │   │   │   ├── 0002/
    │   │   │   │   └── filename.ace
    │   │   │   └── ...
    │   │   │   └── 92235_perturbation_summary_300.0K.txt  # Generated summary
    │   │   ├── 92238/
    │   │   │   └── ...
    │   │   ├── ace_perturbation_YYYYMMDD_HHMMSS.log        # Log file (copied)
    │   │   └── perturbation_matrix_YYYYMMDD_HHMMSS_master.parquet  # Matrix file (copied)
    │   ├── 600.0K/
    │   │   ├── (same structure as 300.0K)
    │   │   ├── ace_perturbation_YYYYMMDD_HHMMSS.log        # Log file (copied)
    │   │   └── perturbation_matrix_YYYYMMDD_HHMMSS_master.parquet  # Matrix file (copied)
    │   └── 900.0K/
    │       ├── (same structure as above)
    │       ├── ace_perturbation_YYYYMMDD_HHMMSS.log        # Log file (copied)
    │       └── perturbation_matrix_YYYYMMDD_HHMMSS_master.parquet  # Matrix file (copied)
    
    Notes:
    - The function will look for ACE files in sample directories (0001, 0002, etc.)
    - Each perturbation sample will be applied to the corresponding existing ACE file
    - Files will be modified in-place (original files are overwritten)
    - The SAME perturbation factors are applied to sample N across all temperatures
    - Summary files are created in each temperature directory for each ZAID
    - Log and parquet files are created in the first temperature directory and copied to all others
    - No new XSDIR files are created (assumes they already exist)
    - Samples without corresponding ACE files will be skipped
    """)

if __name__ == "__main__":
    print("Updated ACE Perturbation Usage Example")
    print("=" * 50)
    print("\nKey Updates:")
    print("- Log and matrix files are now created in ace/tempK/ directories")
    print("- Same perturbation factors applied to each sample across all temperatures")
    print("- Summary files created for each ZAID in each temperature directory")
    print("- Log and matrix files are copied to all temperature directories")
    
    directory_structure_info()
    
    print("\nTo use this script, modify the parameters in example_usage() and run:")
    print("python example_ace_perturbation_usage.py")
    
    # Uncomment the next line to run the example (after setting correct paths)
    # example_usage()
