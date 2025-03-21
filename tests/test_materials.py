import pytest
import os
import filecmp
import tempfile
from mcnpy.input.parse_input import read_mcnp
from mcnpy.input.pert_generator import perturb_material
from mcnpy.input.parse_materials import read_material
from mcnpy.input.material import Mat


def test_material_parsing():

    matfile = "tests/data/mat/matfile_test_1.i"
    inputfile = "tests/data/input/inputfile_test_1.i"

    try:

        # Input file with only materials
        materials = read_mcnp(matfile).materials

        # --- Test #sym:Material 1 variables ---
        assert materials.mat[100000].nlib == '06c'
        assert materials.mat[300000].nlib == '06c'
        assert materials.mat[300000].plib == '02p'
        assert materials.mat[100000].plib == None
        assert materials.mat[300000].id == 300000
        assert materials.mat[300000].nuclide[6013].nlib == '03c'
        assert materials.mat[300000].nuclide[6013].plib == '04p'
        assert materials.mat[300000].nuclide[6013].zaid == 6013
        assert materials.mat[300000].nuclide[6013].element == 'C'
        assert materials.mat[300000].nuclide[6013].fraction == 0.0001022318
        assert materials.mat[300000].nuclide[26056].fraction == 0.8810613
        assert materials.mat[300000].nuclide[24054].fraction == 6.288632e-05

        materials.mat[200100].to_weight_fraction()
        materials.mat[300000].to_weight_fraction()

        assert materials.mat[300000].nuclide[6013].fraction == -2.397746932859688e-05
        assert materials.mat[300000].nuclide[26056].fraction == -0.8888964955001305
        assert materials.mat[300000].nuclide[24054].fraction == -6.118148592568498e-05

        assert materials.mat[200100].__str__() == 'm200100 nlib=06c\n\t1001 -1.119138e-01\n\t8016 -8.880762e-01\n\t5010 -1.843705e-06\n\t5011 -8.159614e-06'
        assert materials.mat[300000].__str__() == 'm300000 nlib=06c plib=02p\n\t6012 -1.975285e-03\n\t6013.03c -2.397747e-05\n\t6013.04p -2.397747e-05\n\t23050 -2.450280e-07\n\t23051 -9.971736e-05\n\t24050 -1.044235e-04\n\t24052 -2.091728e-03\n\t24053 -2.417237e-04\n\t24054 -6.118149e-05\n\t28058 -4.379269e-03\n\t28060 -1.731118e-03\n\t28061 -7.619993e-05\n\t28062 -2.460485e-04\n\t28064 -6.438272e-05\n\t42092 -7.086237e-04\n\t42094 -4.549656e-04\n\t42095 -7.861329e-04\n\t42096 -8.343767e-04\n\t42097 -4.846495e-04\n\t42098 -1.229221e-03\n\t42100 -4.996605e-04\n\t14028 -1.836737e-03\n\t14029 -9.632479e-05\n\t14030 -6.614224e-05\n\t25055 -1.349491e-02\n\t15031 -7.996991e-05\n\t16032 -7.577877e-05\n\t16033 -6.168259e-07\n\t16034 -3.567079e-06\n\t16036 -1.525164e-08\n\t29063 -5.480232e-04\n\t29065 -2.516629e-04\n\t27059 -2.998871e-04\n\t26054 -5.415784e-02\n\t26056 -8.888965e-01\n\t26057 -2.119071e-02\n\t26058 -2.908379e-03'

        materials.mat[200100].to_atomic_fraction()
        materials.mat[300000].to_atomic_fraction()

        assert materials.mat[300000].nuclide[6013].fraction == 0.00010223180430777937
        assert materials.mat[300000].nuclide[26056].fraction == 0.8810613371256076
        assert materials.mat[300000].nuclide[24054].fraction == 6.288632264986424e-05

        assert materials.mat[200100].__str__() == 'm200100 nlib=06c\n\t1001 6.666630e-01\n\t8016 3.333315e-01\n\t5010 1.105447e-06\n\t5011 4.449567e-06'
        assert materials.mat[300000].__str__() == 'm300000 nlib=06c plib=02p\n\t6012 9.126127e-03\n\t6013.03c 1.022318e-04\n\t6013.04p 1.022318e-04\n\t23050 2.719838e-07\n\t23051 1.085216e-04\n\t24050 1.159138e-04\n\t24052 2.232735e-03\n\t24053 2.531442e-04\n\t24054 6.288632e-05\n\t28058 4.190792e-03\n\t28060 1.601455e-03\n\t28061 6.933523e-05\n\t28062 2.202771e-04\n\t28064 5.583628e-05\n\t42092 4.274704e-04\n\t42094 2.686133e-04\n\t42095 4.592415e-04\n\t42096 4.823480e-04\n\t42097 2.772778e-04\n\t42098 6.960837e-04\n\t42100 2.772778e-04\n\t14028 3.639863e-03\n\t14029 1.843021e-04\n\t14030 1.223420e-04\n\t25055 1.361868e-02\n\t15031 1.431432e-04\n\t16032 1.314059e-04\n\t16033 1.037200e-06\n\t16034 5.822139e-06\n\t16036 2.350983e-08\n\t29063 4.828165e-04\n\t29065 2.148952e-04\n\t27059 2.821213e-04\n\t26054 5.566621e-02\n\t26056 8.810613e-01\n\t26057 2.063485e-02\n\t26058 2.783306e-03'

        # Complete input file
        materials = read_mcnp(inputfile).materials

        # --- Test #sym:Material 1 variables ---
        assert materials.mat[100000].nlib == '06c'
        assert materials.mat[300000].nlib == '06c'
        assert materials.mat[300000].plib == None
        assert materials.mat[100000].plib == None
        assert materials.mat[300000].id == 300000
        assert materials.mat[300000].nuclide[6013].nlib == None
        assert materials.mat[300000].nuclide[6013].plib == None
        assert materials.mat[300000].nuclide[6013].zaid == 6013
        assert materials.mat[300000].nuclide[6013].element == 'C'
        assert materials.mat[300000].nuclide[6013].fraction == 0.0001022318
        assert materials.mat[300000].nuclide[26056].fraction == 0.8810613
        assert materials.mat[300000].nuclide[24054].fraction == 6.288632e-05

        materials.mat[200100].to_weight_fraction()
        materials.mat[300000].to_weight_fraction()

        assert materials.mat[300000].nuclide[6013].fraction == -2.397746932859688e-05
        assert materials.mat[300000].nuclide[26056].fraction == -0.8888964955001305
        assert materials.mat[300000].nuclide[24054].fraction == -6.118148592568498e-05

        assert materials.mat[200100].__str__() == 'm200100 nlib=06c\n\t1001 -1.119138e-01\n\t8016 -8.880762e-01\n\t5010 -1.843705e-06\n\t5011 -8.159614e-06'
        assert materials.mat[300000].__str__() == 'm300000 nlib=06c\n\t6012 -1.975285e-03\n\t6013 -2.397747e-05\n\t23050 -2.450280e-07\n\t23051 -9.971736e-05\n\t24050 -1.044235e-04\n\t24052 -2.091728e-03\n\t24053 -2.417237e-04\n\t24054 -6.118149e-05\n\t28058 -4.379269e-03\n\t28060 -1.731118e-03\n\t28061 -7.619993e-05\n\t28062 -2.460485e-04\n\t28064 -6.438272e-05\n\t42092 -7.086237e-04\n\t42094 -4.549656e-04\n\t42095 -7.861329e-04\n\t42096 -8.343767e-04\n\t42097 -4.846495e-04\n\t42098 -1.229221e-03\n\t42100 -4.996605e-04\n\t14028 -1.836737e-03\n\t14029 -9.632479e-05\n\t14030 -6.614224e-05\n\t25055 -1.349491e-02\n\t15031 -7.996991e-05\n\t16032 -7.577877e-05\n\t16033 -6.168259e-07\n\t16034 -3.567079e-06\n\t16036 -1.525164e-08\n\t29063 -5.480232e-04\n\t29065 -2.516629e-04\n\t27059 -2.998871e-04\n\t26054 -5.415784e-02\n\t26056 -8.888965e-01\n\t26057 -2.119071e-02\n\t26058 -2.908379e-03'

        materials.mat[200100].to_atomic_fraction()
        materials.mat[300000].to_atomic_fraction()

        assert materials.mat[300000].nuclide[6013].fraction == 0.00010223180430777937
        assert materials.mat[300000].nuclide[26056].fraction == 0.8810613371256076
        assert materials.mat[300000].nuclide[24054].fraction == 6.288632264986424e-05

        assert materials.mat[200100].__str__() == 'm200100 nlib=06c\n\t1001 6.666630e-01\n\t8016 3.333315e-01\n\t5010 1.105447e-06\n\t5011 4.449567e-06'
        assert materials.mat[300000].__str__() == 'm300000 nlib=06c\n\t6012 9.126127e-03\n\t6013 1.022318e-04\n\t23050 2.719838e-07\n\t23051 1.085216e-04\n\t24050 1.159138e-04\n\t24052 2.232735e-03\n\t24053 2.531442e-04\n\t24054 6.288632e-05\n\t28058 4.190792e-03\n\t28060 1.601455e-03\n\t28061 6.933523e-05\n\t28062 2.202771e-04\n\t28064 5.583628e-05\n\t42092 4.274704e-04\n\t42094 2.686133e-04\n\t42095 4.592415e-04\n\t42096 4.823480e-04\n\t42097 2.772778e-04\n\t42098 6.960837e-04\n\t42100 2.772778e-04\n\t14028 3.639863e-03\n\t14029 1.843021e-04\n\t14030 1.223420e-04\n\t25055 1.361868e-02\n\t15031 1.431432e-04\n\t16032 1.314059e-04\n\t16033 1.037200e-06\n\t16034 5.822139e-06\n\t16036 2.350983e-08\n\t29063 4.828165e-04\n\t29065 2.148952e-04\n\t27059 2.821213e-04\n\t26054 5.566621e-02\n\t26056 8.810613e-01\n\t26057 2.063485e-02\n\t26058 2.783306e-03'

    except FileNotFoundError as e:
        pytest.fail(f"Test file not found: {str(e)}")
    except ValueError as e:
        pytest.fail(f"Invalid data in material file: {str(e)}")
    except Exception as e:
        pytest.fail(f"Unexpected error parsing material file: {str(e)}")


def test_mat_perturbation():
    # Define input and reference files
    matfile = "tests/data/mat/matfile_test_1.i"
    inputfile = "tests/data/input/inputfile_test_1.i"

    # Create temporary copies
    with tempfile.NamedTemporaryFile(suffix='.i', delete=False) as tmp_matfile:
        matfile_copy = tmp_matfile.name
    with tempfile.NamedTemporaryFile(suffix='.i', delete=False) as tmp_inputfile:
        inputfile_copy = tmp_inputfile.name
    
    # Copy the original files to the temporary files
    import shutil
    shutil.copy(matfile, matfile_copy)
    shutil.copy(inputfile, inputfile_copy)

    try:
        # Perturb material in both files - creating permanent files
        perturb_material(matfile_copy, 300000, 8.526729e-02, 26056, pert_mat_id=31, in_place=True, format='weight')
        perturb_material(inputfile_copy, 300000, 8.526729e-02, 26056, in_place=True)
        
        # Compare with reference files
        matfile_ref = "tests/data/mat/matfile_ref_1.i"
        inputfile_ref = "tests/data/mat/inputfile_ref_1.i"
        
        assert filecmp.cmp(matfile_copy, matfile_ref), "Perturbed material file doesn't match reference"
        assert filecmp.cmp(inputfile_copy, inputfile_ref), "Perturbed input file doesn't match reference"

    except FileNotFoundError as e:
        pytest.fail(f"Test file not found: {str(e)}")
    except ValueError as e:
        pytest.fail(f"Invalid data in material file: {str(e)}")
    except Exception as e:
        pytest.fail(f"Unexpected error in material perturbation test: {str(e)}")
    finally:
        # Clean up temporary files
        if os.path.exists(matfile_copy):
            os.unlink(matfile_copy)
        if os.path.exists(inputfile_copy):
            os.unlink(inputfile_copy)


def test_one_line_material_parsing():
    """Test parsing of materials formatted in a single line."""
    
    # Test case 1: Simple one-line material
    test_case_1 = ["m100 1001 0.66667 8016 0.33333"]
    mat_obj, idx = read_material(test_case_1, 0)
    
    assert mat_obj.id == 100
    assert len(mat_obj.nuclide) == 2
    assert mat_obj.nuclide[1001].zaid == 1001
    assert mat_obj.nuclide[1001].fraction == 0.66667
    assert mat_obj.nuclide[8016].zaid == 8016
    assert mat_obj.nuclide[8016].fraction == 0.33333
    assert idx == 1
    
    # Test case 2: One-line material with negative fractions (weight fractions)
    test_case_2 = ["m200 1001 -0.112 8016 -0.888"]
    mat_obj, idx = read_material(test_case_2, 0)
    
    assert mat_obj.id == 200
    assert len(mat_obj.nuclide) == 2
    assert mat_obj.nuclide[1001].fraction == -0.112
    assert mat_obj.nuclide[8016].fraction == -0.888
    
    # Test case 3: One-line material with material-level libraries
    test_case_3 = ["m300 nlib=70c plib=12p 1001 0.5 8016 0.5"]
    mat_obj, idx = read_material(test_case_3, 0)
    
    assert mat_obj.id == 300
    assert mat_obj.nlib == "70c"
    assert mat_obj.plib == "12p"
    assert len(mat_obj.nuclide) == 2
    assert mat_obj.nuclide[1001].fraction == 0.5
    assert mat_obj.nuclide[8016].fraction == 0.5
    
    # Test case 4: One-line material with nuclide-level libraries
    test_case_4 = ["m400 1001.80c 0.6 8016.70c 0.4"]
    mat_obj, idx = read_material(test_case_4, 0)
    
    assert mat_obj.id == 400
    assert len(mat_obj.nuclide) == 2
    assert mat_obj.nuclide[1001].nlib == "80c"
    assert mat_obj.nuclide[8016].nlib == "70c"
    assert mat_obj.nuclide[1001].fraction == 0.6
    assert mat_obj.nuclide[8016].fraction == 0.4
    
    # Test case 5: One-line material with both material and nuclide libraries
    test_case_5 = ["m500 nlib=70c 1001 0.4 8016.80c 0.6"]
    mat_obj, idx = read_material(test_case_5, 0)
    
    assert mat_obj.id == 500
    assert mat_obj.nlib == "70c"
    assert len(mat_obj.nuclide) == 2
    assert mat_obj.nuclide[1001].nlib is None  # Uses material-level lib
    assert mat_obj.nuclide[8016].nlib == "80c"  # Overrides material-level lib
    assert mat_obj.nuclide[1001].fraction == 0.4
    assert mat_obj.nuclide[8016].fraction == 0.6
    
    # Test case 6: One-line material with comments
    test_case_6 = ["m600 1001 0.7 8016 0.3 $ Water material"]
    mat_obj, idx = read_material(test_case_6, 0)
    
    assert mat_obj.id == 600
    assert len(mat_obj.nuclide) == 2
    assert mat_obj.nuclide[1001].fraction == 0.7
    assert mat_obj.nuclide[8016].fraction == 0.3
    
    # Test case 7: One-line material with line continuation
    test_case_7 = [
        "m700 1001 0.3 8016 0.4 &",
        "     24052 0.15 26056 0.15"
    ]
    mat_obj, idx = read_material(test_case_7, 0)
    
    assert mat_obj.id == 700
    assert len(mat_obj.nuclide) == 4
    assert mat_obj.nuclide[1001].fraction == 0.3
    assert mat_obj.nuclide[8016].fraction == 0.4
    assert mat_obj.nuclide[24052].fraction == 0.15
    assert mat_obj.nuclide[26056].fraction == 0.15
    assert idx == 2
    
    # Test case 8: Complex case with multiple libraries and photon libraries
    test_case_8 = [
        "m800 nlib=80c plib=12p 1001.81c 0.1 8016.80c 0.2 &",
        "     13027.70c 0.3 92235.80c 0.4"
    ]
    mat_obj, idx = read_material(test_case_8, 0)
    
    assert mat_obj.id == 800
    assert mat_obj.nlib == "80c"
    assert mat_obj.plib == "12p"
    assert len(mat_obj.nuclide) == 4
    assert mat_obj.nuclide[1001].nlib == "81c"
    assert mat_obj.nuclide[8016].nlib == "80c"
    assert mat_obj.nuclide[13027].nlib == "70c"
    assert mat_obj.nuclide[92235].nlib == "80c"
    assert mat_obj.nuclide[1001].fraction == 0.1
    assert mat_obj.nuclide[8016].fraction == 0.2
    assert mat_obj.nuclide[13027].fraction == 0.3
    assert mat_obj.nuclide[92235].fraction == 0.4


def test_natural_element_conversion():
    """Test conversion of natural elements to their isotopic compositions."""
    
    # Test case 1: Material with natural carbon (atomic fractions)
    mat = Mat(id=900)
    mat.add_nuclide(6000, 1.0)  # Natural carbon
    
    # Create copy to preserve the original
    original_mat = mat.copy(900)
    
    # Test convert_natural_elements method
    mat.convert_natural_elements()
    
    # Verify original material is preserved in our copy
    assert 6000 in original_mat.nuclide
    assert len(original_mat.nuclide) == 1
    
    # Verify expanded material has isotopes instead of natural element
    assert 6000 not in mat.nuclide
    assert 6012 in mat.nuclide
    assert 6013 in mat.nuclide
    
    # Verify fractions sum to approximately the original value
    total_fraction = sum(nuclide.fraction for nuclide in mat.nuclide.values())
    assert abs(total_fraction - 1.0) < 1e-10
    
    # Test case 2: Material with natural iron (weight fractions)
    mat = Mat(id=901)
    mat.add_nuclide(26000, -1.0)  # Natural iron (weight fraction)
    
    # Test in-place conversion
    mat.convert_natural_elements()
    
    # Verify natural element is replaced with isotopes
    assert 26000 not in mat.nuclide
    assert 26054 in mat.nuclide
    assert 26056 in mat.nuclide
    assert 26057 in mat.nuclide
    assert 26058 in mat.nuclide
    
    # Verify all fractions are negative (weight fractions)
    assert all(nuclide.fraction < 0 for nuclide in mat.nuclide.values())
    
    # Verify fractions sum to approximately the original value
    total_fraction = sum(abs(nuclide.fraction) for nuclide in mat.nuclide.values())
    assert abs(total_fraction - 1.0) < 1e-10
    
    # Test case 3: Specific ZAID conversion
    mat = Mat(id=902)
    mat.add_nuclide(6000, 0.5)    # Natural carbon
    mat.add_nuclide(8016, 0.5)    # Oxygen-16 (specific isotope)
    
    # Create copy to preserve the original state for assertions
    original_mat = mat.copy(902)
    
    # Convert only carbon
    mat.convert_natural_elements(zaid_to_expand=6000)
    
    # Verify specific conversion
    assert 6000 not in mat.nuclide
    assert 6012 in mat.nuclide
    assert 6013 in mat.nuclide
    assert 8016 in mat.nuclide  # Unchanged
    
    # Test error cases
    mat = Mat(id=903)
    mat.add_nuclide(8016, 1.0)    # Not a natural element
    
    # Attempt to convert non-natural element should raise ValueError
    try:
        mat.convert_natural_elements(zaid_to_expand=8016)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_fraction_conversion_with_natural_elements():
    """Test atomic/weight fraction conversion with natural elements."""
    
    # Create material with natural elements (atomic fractions)
    mat = Mat(id=904)
    mat.add_nuclide(6000, 0.3)    # Natural carbon
    mat.add_nuclide(26000, 0.7)   # Natural iron
    
    # Create a copy for later verification
    original_mat = mat.copy(904)
    
    # Convert to weight fractions (in-place)
    mat.to_weight_fraction()
    
    # Verify natural elements are preserved
    assert 6000 in mat.nuclide
    assert 26000 in mat.nuclide
    
    # Verify weight fractions are negative
    assert mat.nuclide[6000].fraction < 0
    assert mat.nuclide[26000].fraction < 0
    
    # Calculate expected weight fractions:
    # C: 0.3 * 12.011 = 3.6033
    # Fe: 0.7 * 55.845 = 39.0915
    # Total: 42.6948
    # C weight fraction: -(3.6033 / 42.6948) ≈ -0.0844
    # Fe weight fraction: -(39.0915 / 42.6948) ≈ -0.9156
    expected_c_weight = -0.0844
    expected_fe_weight = -0.9156
    
    # Check actual weight fractions against calculated values
    assert abs(mat.nuclide[6000].fraction - expected_c_weight) < 1e-3
    assert abs(mat.nuclide[26000].fraction - expected_fe_weight) < 1e-3
    
    # Convert back to atomic fractions (in-place)
    mat.to_atomic_fraction()
    
    # Verify original atomic fractions are restored
    expected_c_atomic = 0.3
    expected_fe_atomic = 0.7
    assert abs(mat.nuclide[6000].fraction - expected_c_atomic) < 1e-10
    assert abs(mat.nuclide[26000].fraction - expected_fe_atomic) < 1e-10
    
    # Test conversion with expanded isotopes
    # Start with a fresh copy
    expanded_mat = original_mat.copy(904)
    expanded_mat.convert_natural_elements()
    
    # Convert to weight fractions (in-place)
    expanded_mat.to_weight_fraction()
    
    # Verify all fractions are negative
    assert all(nuclide.fraction < 0 for nuclide in expanded_mat.nuclide.values())
    
    # Calculate total weight fraction by element after expansion
    c_weight_sum = sum(abs(nuclide.fraction) for zaid, nuclide in expanded_mat.nuclide.items() 
                     if zaid // 1000 == 6)
    fe_weight_sum = sum(abs(nuclide.fraction) for zaid, nuclide in expanded_mat.nuclide.items() 
                      if zaid // 1000 == 26)
    
    # Verify expanded isotope weight fractions sum to the expected element weight fractions
    assert abs(c_weight_sum - abs(expected_c_weight)) < 1e-3
    assert abs(fe_weight_sum - abs(expected_fe_weight)) < 1e-3
    
    # Convert back to atomic fractions (in-place)
    expanded_mat.to_atomic_fraction()
    
    # Sum up atomic fractions by element
    c_atomic_sum = sum(nuclide.fraction for zaid, nuclide in expanded_mat.nuclide.items() 
                     if zaid // 1000 == 6)
    fe_atomic_sum = sum(nuclide.fraction for zaid, nuclide in expanded_mat.nuclide.items() 
                      if zaid // 1000 == 26)
    
    # Verify expanded isotope atomic fractions sum to the original atomic fractions
    assert abs(c_atomic_sum - expected_c_atomic) < 1e-10
    assert abs(fe_atomic_sum - expected_fe_atomic) < 1e-10


