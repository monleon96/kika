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

   # Display basic information
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
   print(f"Tally {tally_id} details:")
   print(tally)

Accessing Tally Data
--------------------

Each tally contains results, errors, and energy bin information:

.. code-block:: python

   # Display tally details
   print(f"Tally name: {tally.name}")
   print(f"Number of energy bins: {len(tally.energies)}")
   print(f"Number of results: {len(tally.results)}")
   
   # Display integral result if available
   if tally.integral_result is not None:
       print(f"Integral result: {tally.integral_result:.6e}")
       print(f"Integral error: {tally.integral_error:.6e}")
   
   # Display energy bin structure
   if tally.energies:
       print("Energy bin structure:")
       print(tally.energies)
   
   # Access energy bins and results
   for i, (energy, result, error) in enumerate(zip(tally.energies, tally.results, tally.errors)):
       print(f"Energy bin {i}: {energy:.3e} MeV, Result: {result:.6e} ± {error:.6e}")

Converting to DataFrame
-----------------------

For more advanced data analysis, MCNPy allows you to convert tally data to pandas DataFrames:

.. code-block:: python

   # Convert tally to DataFrame
   tally_df = tally.to_dataframe()
   
   # Examine the data
   print(tally_df.head())
   
   # Perform calculations and filtering
   # Example: Filter by energy range
   filtered_df = tally_df[tally_df['energy_bin'] > 1e-6]
   
   # Basic statistics
   print(f"Mean result: {tally_df['result'].mean():.6e}")
   print(f"Maximum result: {tally_df['result'].max():.6e}")

Visualizing Tally Data
----------------------

MCNPy provides built-in methods for visualizing tally results and convergence:

.. code-block:: python

   # Basic plotting of results vs energy
   plt.figure(figsize=(10, 6))
   plt.errorbar(tally_df['energy_bin'], tally_df['result'], 
               yerr=tally_df['result']*tally_df['rel_error'], 
               fmt='o', capsize=5)
   plt.xscale('log')
   plt.yscale('log')
   plt.xlabel('Energy (MeV)')
   plt.ylabel('Tally Result')
   plt.title(f'Tally {tally_id} Results vs Energy')
   plt.grid(True, which='both', linestyle='--', alpha=0.7)
   plt.show()

Analyzing Tally Convergence
---------------------------

Use TFC (Tally Fluctuation Chart) data to analyze tally convergence:

.. code-block:: python

   # Check if TFC data is available
   if tally.tfc_nps and len(tally.tfc_nps) > 0:
       # Plot TFC data using the built-in method
       print("Plotting TFC convergence data:")
       tally.plot_tfc_data()
       
       # The plot includes:
       # 1. Tally mean vs. histories
       # 2. Relative error vs. histories 
       # 3. Figure of merit vs. histories
   else:
       print("No TFC data available for this tally.")

Working with Perturbations
--------------------------

MCNP perturbation data is stored in the MCTAL file for perturbed tallies:

.. code-block:: python

   # Find tallies with perturbation data
   tallies_with_pert = [tid for tid in tally_ids if mctal.tally[tid].perturbation]
   
   if tallies_with_pert:
       # Get the first tally with perturbation data
       tally_id = tallies_with_pert[0]
       tally = mctal.tally[tally_id]
       
       # Display the perturbation collection
       print(tally.perturbation)
       
       # Get list of perturbation IDs
       pert_ids = list(tally.perturbation.keys())
       
       if pert_ids:
           # Access a specific perturbation
           pert_id = pert_ids[0]
           pert = tally.perturbation[pert_id]
           
           print(f"Details for perturbation {pert_id}:")
           print(pert)
           
           # Access specific perturbation data
           print(f"Result: {pert.integral_result:.6e} ± {pert.integral_error:.6e}")

Converting Perturbation Data to DataFrames
------------------------------------------

For better analysis, perturbation data can be converted to pandas DataFrames:

.. code-block:: python

   if tallies_with_pert:
       tally_id = tallies_with_pert[0]
       tally = mctal.tally[tally_id]
       
       # Convert all perturbations to a DataFrame
       pert_df = tally.perturbation.to_dataframe()
       
       if not pert_df.empty:
           print("All perturbations as DataFrame:")
           print(pert_df.head())
           
           # Analyze perturbation data
           if 'energy_bin' in pert_df.columns:
               # Group by energy bin
               energy_groups = pert_df.groupby('energy_bin')
               
               # Summary statistics by energy bin
               print(energy_groups['result'].mean())