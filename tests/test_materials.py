import pytest
import os
import filecmp
import tempfile
from mcnpy.input.parse_input import read_mcnp
from mcnpy.input.pert_generator import perturb_material


def test_material_parsing():

    matfile = "tests/data/mat/PWRSphereMAT_test.i"
    inputfile = "tests/data/input/PWRSphere_test.i"

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
    matfile = "tests/data/mat/PWRSphereMAT_test.i"
    inputfile = "tests/data/input/PWRSphere_test.i"

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
        matfile_ref = "tests/data/mat/PWRSphereMAT_ref.i"
        inputfile_ref = "tests/data/mat/PWRSphere_ref.i"
        
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