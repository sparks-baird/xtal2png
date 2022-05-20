# convert between a fixed crystal Structure to numeric representation (test), to PNG
# (test), and back to crystal Structure (test)

from os import path

from pymatgen.core.structure import Structure

EXAMPLE_CIF = "Zn2B2PbO6.cif"
fpath = path.join("data", "external", EXAMPLE_CIF)
S = Structure.from_file(fpath)

S.atomic_numbers  # atom
S.frac_coords  # xyz
S._lattice.abc  # abc
S._lattice.angles  # angles
S.get_space_group_info()  # space group
S.distance_matrix  # distance matrix

1 + 1
