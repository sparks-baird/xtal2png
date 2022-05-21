# convert between a fixed crystal Structure to numeric representation (test), to PNG
# (test), and back to crystal Structure (test)

from os import path

from pymatgen.core.structure import Structure

from xtal2png.skeleton import XtalConverter as xc

EXAMPLE_CIFS = ["Zn2B2PbO6.cif", "V2NiSe4.cif"]
S = []
for cif in EXAMPLE_CIFS:
    fpath = path.join("data", "external", cif)
    S.append(Structure.from_file(fpath))

xc.structures_to_arrays(S)

xc.structures_to_arrays(S[0])

1 + 1
