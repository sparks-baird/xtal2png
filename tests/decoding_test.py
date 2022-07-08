"""Test decoding functionality using default kwargs and assert matches."""


from os import path

from PIL import Image

from xtal2png.core import XtalConverter
from xtal2png.utils.data import assert_structures_approximate_match, example_structures


def test_arrays_to_structures():
    xc = XtalConverter()
    data, id_data, id_mapper = xc.structures_to_arrays(example_structures)
    structures = xc.arrays_to_structures(data, id_data, id_mapper)
    assert_structures_approximate_match(
        example_structures, structures, tol_multiplier=2.0
    )
    return structures


def test_arrays_to_structures_zero_one():
    xc = XtalConverter()
    data, id_data, id_mapper = xc.structures_to_arrays(
        example_structures, rgb_scaling=False
    )
    structures = xc.arrays_to_structures(data, id_data, id_mapper, rgb_scaling=False)
    assert_structures_approximate_match(
        example_structures, structures, tol_multiplier=2.0
    )
    return structures


def test_arrays_to_structures_single():
    xc = XtalConverter()
    data, id_data, id_mapper = xc.structures_to_arrays([example_structures[0]])
    structures = xc.arrays_to_structures(data, id_data, id_mapper)
    assert_structures_approximate_match(
        [example_structures[0]], structures, tol_multiplier=2.0
    )
    return structures


def test_png2xtal():
    xc = XtalConverter()
    imgs = xc.xtal2png(example_structures, show=True, save=True)
    decoded_structures = xc.png2xtal(imgs)
    assert_structures_approximate_match(
        example_structures, decoded_structures, tol_multiplier=2.0
    )


def test_png2xtal_single():
    xc = XtalConverter()
    imgs = xc.xtal2png([example_structures[0]], show=True, save=True)
    decoded_structures = xc.png2xtal(imgs, save=False)
    assert_structures_approximate_match(
        [example_structures[0]], decoded_structures, tol_multiplier=2.0
    )
    return decoded_structures


def test_png2xtal_rgb_image():
    xc = XtalConverter()
    imgs = xc.xtal2png(example_structures, show=False, save=False)
    imgs = [img.convert("RGB") for img in imgs]
    decoded_structures = xc.png2xtal(imgs)
    assert_structures_approximate_match(
        example_structures, decoded_structures, tol_multiplier=2.0
    )
    return decoded_structures


def test_png2xtal_from_saved_images():
    xc = XtalConverter()
    xc.xtal2png(example_structures, show=False, save=True)
    fpaths = [path.join(xc.save_dir, savename + ".png") for savename in xc.savenames]
    saved_imgs = [Image.open(fpath) for fpath in fpaths]

    decoded_structures = xc.png2xtal(saved_imgs)
    assert_structures_approximate_match(
        example_structures, decoded_structures, tol_multiplier=2.0
    )
