# convert between a fixed crystal Structure to numeric representation (test), to PNG
# (test), and back to crystal Structure (test)


from numpy.testing import assert_allclose

from xtal2png.core import XtalConverter
from xtal2png.utils.data import (
    element_wise_scaler,
    element_wise_unscaler,
    example_structures,
    rgb_scaler,
    rgb_unscaler,
)

rgb_tol = 1 / 255  # should this be 256?
rgb_loose_tol = 1.5 / 255


def assert_structures_approximate_match(example_structures, structures):
    for s, structure in zip(example_structures, structures):
        abc_check = s._lattice.abc
        angles_check = s._lattice.angles
        atomic_numbers_check = s.atomic_numbers
        frac_coords_check = s.frac_coords

        abc = structure._lattice.abc
        angles = structure._lattice.angles
        atomic_numbers = structure.atomic_numbers
        frac_coords = structure.frac_coords

        assert_allclose(
            abc_check,
            abc,
            rtol=rgb_loose_tol,
            err_msg="lattice parameter lengths not all close",
        )

        assert_allclose(
            angles_check,
            angles,
            rtol=rgb_loose_tol,
            err_msg="lattice parameter angles not all close",
        )

        assert_allclose(
            atomic_numbers_check,
            atomic_numbers,
            rtol=rgb_loose_tol,
            err_msg="atomic numbers not all close",
        )

        # use atol since frac_coords values are between 0 and 1
        assert_allclose(
            frac_coords_check,
            frac_coords,
            atol=rgb_tol,
            err_msg="atomic numbers not all close",
        )


def test_structures_to_arrays():
    xc = XtalConverter()
    data = xc.structures_to_arrays(example_structures)
    return data


def test_structures_to_arrays_single():
    xc = XtalConverter()
    data, _, _ = xc.structures_to_arrays([example_structures[0]])
    return data


def test_arrays_to_structures():
    xc = XtalConverter()
    data, _, _ = xc.structures_to_arrays(example_structures)
    structures = xc.arrays_to_structures(data)
    assert_structures_approximate_match(example_structures, structures)
    return structures


def test_xtal2png():
    xc = XtalConverter()
    imgs = xc.xtal2png(example_structures, show=False, save=True)
    return imgs


def test_xtal2png_single():
    xc = XtalConverter()
    imgs = xc.xtal2png([example_structures[0]], show=False, save=True)
    return imgs


def test_png2xtal():
    xc = XtalConverter()
    imgs = xc.xtal2png(example_structures, show=True, save=True)
    structures = xc.png2xtal(imgs)
    assert_structures_approximate_match(example_structures, structures)


def test_png2xtal_single():
    xc = XtalConverter()
    imgs = xc.xtal2png([example_structures[0]], show=True, save=True)
    decoded_structures = xc.png2xtal(imgs, save=False)
    return decoded_structures


def test_element_wise_scaler_unscaler():
    check_input = [[1, 2], [3, 4]]
    feature_range = [1, 4]
    data_range = [0, 8]
    check_output = [[1.375, 1.75], [2.125, 2.5]]
    scaled = element_wise_scaler(
        check_input, feature_range=feature_range, data_range=data_range
    )
    unscaled = element_wise_unscaler(
        check_output, feature_range=feature_range, data_range=data_range
    )
    assert_allclose(check_input, unscaled)
    assert_allclose(check_output, scaled)


def test_rgb_scaler_unscaler():
    check_input = [[1, 2], [3, 4]]
    check_unscaled = [[1.00392157, 2.00784314], [3.01176471, 4.01568627]]
    data_range = [0, 8]
    check_output = [[32, 64], [96, 128]]
    scaled = rgb_scaler(check_input, data_range=data_range)
    unscaled = rgb_unscaler(check_output, data_range=data_range)
    # NOTE: rtol = 1/255 seems to be an exact tolerance, maybe loosen slightly
    assert_allclose(
        check_input,
        unscaled,
        rtol=rgb_tol,
        err_msg=f"rgb_unscaler values not within {rgb_tol} of original",
    )
    assert_allclose(check_unscaled, unscaled)
    assert_allclose(check_output, scaled)


if __name__ == "__main__":
    test_element_wise_scaler_unscaler()
    test_rgb_scaler_unscaler()
    structures = test_arrays_to_structures()
    imgs = test_xtal2png()
    imgs = test_xtal2png_single()
    decoded_structures = test_png2xtal()
    decoded_structures = test_png2xtal_single()
    test_structures_to_arrays_single()
    test_xtal2png_single()

1 + 1

# %% Code Graveyard
#     xc = XtalConverter()
#     data = xc.xtal2png(example_structures, show=True, save=True)
#     decoded_structures = xc.png2xtal(data, save=False)
#     return decoded_structures
