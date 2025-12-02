#!/usr/bin/env python3
"""Test script for new __str__ and __repr__ methods"""

from mcnpy.input.material import Material, MaterialCollection

print("="*80)
print("Testing Material.__str__ and Material.__repr__")
print("="*80)

# Create a steel material to test display
steel = Material(
    id=2,
    name='Stainless Steel 316',
    libs={'nlib': '80c', 'plib': '12p'}
)

steel.add_nuclide(26056, 0.68)  # Fe-56
steel.add_nuclide(24052, 0.17)  # Cr-52
steel.add_nuclide(28058, 0.12)  # Ni-58
steel.add_nuclide(42095, 0.02)  # Mo-95
steel.add_nuclide(25055, 0.01)  # Mn-55
steel.add_nuclide(6012, 0.001, library='70c')  # C-12 with different library

print('\nTesting new __str__ display (via print):')
print(steel)

print('\n\nTesting new __repr__ display (via repr):')
print(repr(steel))

# Test MaterialCollection display
print("\n\n")
print("="*80)
print("Testing MaterialCollection.__str__ and MaterialCollection.__repr__")
print("="*80)

collection = MaterialCollection()
collection.add_material(steel)

# Add another material
water = Material(id=1, name="Water")
water.add_nuclide(1001, 2/3)
water.add_nuclide(8016, 1/3)
collection.add_material(water)

print('\nTesting new MaterialCollection __str__ display (via print):')
print(collection)

print('\n\nTesting new MaterialCollection __repr__ display (via repr):')
print(repr(collection))
