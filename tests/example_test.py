"""Make sure the example in the README works."""
from pymatgen.core import Structure

from xtal2png import XtalConverter, example_structures


def test_example():
    # https://github.com/sparks-baird/xtal2png/issues/184
    xc = XtalConverter(relax_on_decode=False)
    data = xc.xtal2png(example_structures, show=True, save=True)
    decoded_structures = xc.png2xtal(data, save=False)
    assert len(decoded_structures) == len(example_structures)
    for s in decoded_structures:
        assert isinstance(s, Structure)
