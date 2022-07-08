"""Test fitting, scaling, and any other methods (if applicable)."""


from numpy.testing import assert_allclose, assert_array_equal, assert_equal

from xtal2png.core import XtalConverter
from xtal2png.utils.data import (
    RGB_TOL,
    dummy_structures,
    element_wise_scaler,
    element_wise_unscaler,
    example_structures,
    rgb_scaler,
    rgb_unscaler,
)


def test_fit():
    xc = XtalConverter(relax_on_decode=False)
    xc.fit(example_structures + dummy_structures)
    assert_array_equal((5, 82), xc._atom_range)
    assert_allclose((3.84, 12.718448099999998), xc.a_range)
    assert_allclose((3.395504, 11.292530369999998), xc.b_range)
    assert_allclose((3.84, 10.6047314973), xc.c_range)
    assert_array_equal((0.0, 180.0), xc.angles_range)
    assert_allclose((12, 227), xc.space_group_range)
    assert_allclose((2, 44), xc.num_sites_range)
    assert_allclose((1.383037596160554, 7.8291318247510695), xc.distance_range)
    assert_equal(44, xc.num_sites)


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
        rtol=RGB_TOL,
        err_msg=f"rgb_unscaler values not within {RGB_TOL} of original",
    )
    assert_allclose(check_unscaled, unscaled)
    assert_allclose(check_output, scaled)
