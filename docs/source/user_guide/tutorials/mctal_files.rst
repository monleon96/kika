Working with MCTAL Files
========================

This tutorial covers how to use MCNPy to work with MCNP MCTAL output files, including parsing, accessing tally data, data analysis, and visualization.

Loading MCTAL Files
-------------------

Start by loading a MCTAL file:

.. code-block:: python

   import mcnpy
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   from pathlib import Path
   
   # Parse a mctal file
   mctal = mcnpy.read_mctal('path/to/mctalfile.m')

MCTAL File Structure
--------------------

The parsed MCTAL file is represented by a ``Mctal`` object containing header information and tally data.

.. code-block:: python

   # The Mctal object itself provides a nicely formatted summary
   mctal

   # Display specific header information
   print(f"Code: {mctal.code_name} {mctal.ver}")
   print(f"Problem ID: {mctal.probid}")
   print(f"Number of particle histories: {mctal.nps:.2e}")
   print(f"Problem Title: {mctal.problem_id}")
   
   # Display tally information
   print(f"Number of tallies: {mctal.ntal}")
   print(f"Tally numbers: {mctal.tally_numbers}")

Working with Tallies
--------------------

Tallies are the main output of MCNP simulations. The ``Mctal`` class provides access to all tallies in the MCTAL file:

.. code-block:: python

   # Get a list of all tally IDs
   tally_ids = list(mctal.tally.keys())
   print(f"Available tally IDs: {tally_ids}")
   
   # Access a specific tally
   tally_id = tally_ids[0]  # Select first tally as example
   tally = mctal.tally[tally_id]
   
   # Display the tally information
   display(tally)
   
   # Display tally dimensions and structure
   print(f"Tally name: {tally.name}")
   print(f"Tally dimensions: {tally.get_dimensions()}")
   
   # Display energy bin structure if available
   if tally.energies:
       print("Energy bin boundaries:")
       for i, energy in enumerate(tally.energies):
           print(f"  Bin {i}: {energy:.6e} MeV")

Converting to DataFrame
-----------------------

For more advanced data analysis, MCNPy allows you to convert tally data to pandas DataFrames:

.. code-block:: python

   # Convert tally to DataFrame
   tally_df = tally.to_dataframe()
   
   # Display the DataFrame
   display(tally_df)
   
   # For energy-integrated data (if available)
   integral_df = tally.get_integral_energy_dataframe()
   if not integral_df.empty:
       print("Energy-integrated data as DataFrame:")
       display(integral_df)

Working with Multidimensional Data
----------------------------------

MCNPy supports multidimensional data analysis using xarray, which provides labeled N-dimensional arrays:

.. code-block:: python

   # Convert tally data to xarray Dataset
   ds = tally.to_xarray()
   display(ds)
   
   # Extracting slices of multidimensional data
   dims = tally.get_dimensions()
   print(f"Tally dimensions: {dims}")
   
   # Get a slice for a specific energy value
   results, errors = tally.get_slice(energy=tally.energies[2])
   print(f"Results for selected energy bin: {results}")
   print(f"Errors for selected energy bin: {errors}")
   
   # Get a slice for a specific segment
   results, errors = tally.get_slice(segment=1)  # Segment numbering starts at 0
   print(f"Results for second segment: {results}")
   print(f"Errors for second segment: {errors}")

Working with Energy-Integrated Data
-----------------------------------

For tallies with energy bins, MCNPy provides methods to access energy-integrated data:

.. code-block:: python

   # Get the energy-integrated data
   integral_data = tally.get_integral_energy_data()
   
   # Display the energy-integrated data
   print(f"Result: {integral_data['Result']}")
   print(f"Error: {integral_data['Error']}")
   
   # Convert to DataFrame for tabular view
   integral_df = tally.get_integral_energy_dataframe()
   display(integral_df)

Analyzing Tally Convergence
---------------------------

Use TFC (Tally Fluctuation Chart) data to analyze tally convergence:

.. code-block:: python

   # Check if TFC data is available
   if hasattr(tally, 'tfc_nps') and tally.tfc_nps and len(tally.tfc_nps) > 0:
       print(f"Tally has {len(tally.tfc_nps)} TFC data points")
       
       # Show some TFC data values
       print("Sample of TFC data:")
       print(f"{'NPS':<12} {'Result':<15} {'Error':<10} {'FOM':<10}")
       for i in range(min(5, len(tally.tfc_nps))):  # Show up to first 5 points
           print(f"{tally.tfc_nps[i]:<12} {tally.tfc_results[i]:<15.6e} {tally.tfc_errors[i]:<10.6f} {tally.tfc_fom[i]:<10.2f}")
       
       # Plot TFC data using the built-in method
       tally.plot_tfc_data(figsize=(15, 4))
       
       # Plot without error bars for clearer visualization
       tally.plot_tfc_data(figsize=(15, 4), show_error_bars=False)
   else:
       print("No TFC data available for this tally.")

Working with Perturbations
--------------------------

MCNP perturbation data is stored in the MCTAL file for perturbed tallies:

.. code-block:: python

   # Find tallies with perturbation data
   tallies_with_pert = [tid for tid in tally_ids if hasattr(mctal.tally[tid], 'perturbation') and mctal.tally[tid].perturbation]
   
   if tallies_with_pert:
       # Get the first tally with perturbation data
       tally_id = tallies_with_pert[0]
       tally = mctal.tally[tally_id]
       
       # Display the perturbation collection
       print(f"Perturbation collection for Tally {tally_id}:")
       display(tally.perturbation)
       
       # Get list of perturbation IDs
       pert_ids = list(tally.perturbation.keys())
       
       if pert_ids:
           # Access a specific perturbation
           pert_id = pert_ids[0]
           pert = tally.perturbation[pert_id]
           
           # Display the perturbation details
           display(pert)

Converting Perturbation Data to DataFrames
------------------------------------------

For better analysis, perturbation data can be converted to pandas DataFrames:

.. code-block:: python

   if tallies_with_pert:
       tally_id = tallies_with_pert[0]
       tally = mctal.tally[tally_id]
       
       # Convert all perturbations to a DataFrame
       pert_df = tally.perturbation.to_dataframe()
       
       print("All perturbations as DataFrame:")
       display(pert_df)
       
       # Analyze a single perturbation
       if pert_ids:
           pert_id = pert_ids[0]
           single_pert = tally.perturbation[pert_id]
           single_df = single_pert.to_dataframe()
           
           print(f"Perturbation {pert_id} as DataFrame:")
           display(single_df)