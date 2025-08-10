# ACE Perturbation Script Modifications Summary

## Overview
Modified `mcnpy/sampling/ace_perturbation_separate.py` to work with existing ACE files from the ENDF perturbation process instead of creating new files.

## Latest Updates (Second Round of Changes)

### 1. File Location Changes
- **Log files**: Now created in `ace/tempK/` directory instead of root directory
- **Parquet files**: Now created in `ace/tempK/` directory and copied to all temperature directories
- **Summary files**: Created individually for each ZAID in each temperature directory

### 2. Temperature Handling Clarification
**Question**: Are the same perturbation factors applied across temperatures?
**Answer**: YES - Sample 0001 at temp1 and sample 0001 at temp2 receive identical perturbation factors.

**Implementation**: 
- Perturbation factors are generated once per ZAID
- The same `factors[j]` array is applied to all ACE files found for sample `j` across all temperatures
- Summary files are created for each temperature showing which samples were processed at that temperature

### 3. Cleaned Up Unused Functions
**Removed**:
- `_process_sample()` - Old function for creating new ACE files
- `_write_sample_summary()` - Replaced with temperature-aware summary function
- XSDIR-related imports: `write_xsdir_line`, `build_xsdir_line`, `create_xsdir_files_for_ace`

**Added**:
- `_copy_files_to_temperature_directories()` - Creates ZAID-specific summaries for each temperature
- `_copy_master_files_to_temperature_directories()` - Copies log and parquet files to all temperatures

## Key Changes Made

### 1. Function Signature Changes
**Before:**
```python
def perturb_ACE_files(
    ace_files: Union[str, List[str]],
    cov_files: Union[str, List[str]],
    mt_list: List[int],
    num_samples: int,
    # ... other parameters
    output_dir: str = '.',
    xsdir_file: Optional[str] = None,
    # ... 
):
```

**After:**
```python
def perturb_ACE_files(
    root_dir: str,
    temperatures: Union[float, List[float]],
    zaids: List[int],
    cov_files: Union[str, List[str]],
    mt_list: List[int],
    num_samples: int,
    # ... other parameters (xsdir_file and output_dir removed)
):
```

### 2. New Input Parameters
- `root_dir`: Root directory containing the ACE files in the expected structure
- `temperatures`: Temperature(s) for which ACE files exist
- `zaids`: List of ZAID numbers to process

### 3. Removed Parameters
- `ace_files`: No longer needed as files are found automatically
- `output_dir`: Now derived from `root_dir` and temperature structure 
- `xsdir_file`: Removed as XSDIR files are not generated

### 4. Directory Structure Handling
The script now expects and works with this structure:
```
root_dir/
├── ace/
│   ├── 300.0K/          # Temperature directories
│   │   ├── 92235/       # ZAID directories
│   │   │   ├── 0001/    # Sample directories
│   │   │   │   └── filename.ace
│   │   │   ├── 0002/
│   │   │   │   └── filename.ace
│   │   │   └── ...
│   │   │   └── 92235_perturbation_summary_300.0K.txt  # ZAID summary
│   │   ├── ace_perturbation_YYYYMMDD_HHMMSS.log       # Log file (copied)
│   │   └── perturbation_matrix_YYYYMMDD_HHMMSS_master.parquet  # Matrix (copied)
│   └── ...
```

### 5. New Function: `_process_sample_inplace`
Created a new function that:
- Takes an existing ACE file path
- Applies perturbation factors to it
- Overwrites the original file in-place
- Does not create XSDIR files
- Includes temperature parameter for organization

### 6. Main Processing Logic Changes
- **File Discovery**: Automatically finds ACE files based on directory structure
- **Sample Processing**: Each perturbation sample is applied to a different existing ACE file
- **Error Handling**: Skips samples where ACE files don't exist
- **In-place Modification**: Replaces existing files instead of creating new ones
- **Temperature Tracking**: Tracks which samples are processed at each temperature

### 7. Enhanced Logging and File Organization
- Log files created in first temperature directory and copied to all others
- Parquet matrix files created once and copied to all temperature directories  
- ZAID-specific summary files created in each temperature directory
- Better error messages for missing directory structures
- Temperature-aware progress reporting

### 8. Validation Changes
- Validates that temperatures and zaids are provided
- Checks for representative ACE files to determine structure
- Reports missing samples and continues with available files
- Tracks processed samples by temperature

## Usage Example

```python
from mcnpy.sampling.ace_perturbation_separate import perturb_ACE_files

perturb_ACE_files(
    root_dir="/path/to/endf_perturbation_output",
    temperatures=[300.0, 600.0, 900.0],
    zaids=[92235, 92238, 94239],
    cov_files="covariance_matrix.cov",
    mt_list=[1, 2, 18, 102],
    num_samples=100,
    space="log",
    decomposition_method="svd",
    sampling_method="sobol",
    seed=42,
    autofix="soft",
    verbose=True
)
```

## Files Created/Modified
1. **Modified**: `mcnpy/sampling/ace_perturbation_separate.py` - Main function with all changes
2. **Updated**: `example_ace_perturbation_usage.py` - Updated usage example and documentation  
3. **Updated**: `ACE_PERTURBATION_MODIFICATIONS.md` - This summary document

## Key Benefits
1. **Integration**: Works seamlessly with output from `perturb_ENDF_files`
2. **Efficiency**: No duplicate file creation or XSDIR generation
3. **Flexibility**: Handles missing samples gracefully
4. **Consistency**: Same perturbation factors applied across all temperatures for each sample
5. **Organization**: Files organized by temperature with summaries and copies in each directory
6. **Error Handling**: Robust handling of missing files and directory structure issues
7. **Traceability**: Clear documentation of which samples were processed at each temperature
