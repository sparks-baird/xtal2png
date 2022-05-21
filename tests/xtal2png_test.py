# convert between a fixed crystal Structure to numeric representation (test), to PNG
# (test), and back to crystal Structure (test)

from os import path

from pymatgen.core.structure import Structure

from xtal2png.skeleton import XtalConverter

EXAMPLE_CIFS = ["Zn2B2PbO6.cif", "V2NiSe4.cif"]
S = []
for cif in EXAMPLE_CIFS:
    fpath = path.join("..", "data", "external", cif)
    S.append(Structure.from_file(fpath))


def test_structures_to_arrays():
    xc = XtalConverter()
    data = xc.structures_to_arrays(S)
    return data


def test_structures_to_arrays_single():
    xc = XtalConverter()
    data = xc.structures_to_arrays([S[0]])
    return data


def test_xtal2png():
    xc = XtalConverter()
    imgs = xc.xtal2png(S, show=False, save=True)
    return imgs


def test_xtal2png_single():
    xc = XtalConverter()
    imgs = xc.xtal2png([S[0]], show=False, save=True)
    return imgs


1 + 1
