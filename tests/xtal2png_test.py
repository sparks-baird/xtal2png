# convert between a fixed crystal Structure to numeric representation (test), to PNG
# (test), and back to crystal Structure (test)


from warnings import warn

import plotly.express as px
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from xtal2png.core import XtalConverter
from xtal2png.utils.data import (
    dummy_structures,
    element_wise_scaler,
    element_wise_unscaler,
    example_structures,
    rgb_scaler,
    rgb_unscaler,
)
from xtal2png.utils.plotting import plot_and_save

rgb_tol = 1 / 255  # should this be 256?
rgb_loose_tol = 1.5 / 255


def assert_structures_approximate_match(
    example_structures, structures, tol_multiplier=1.0
):
    for i, (s, structure) in enumerate(zip(example_structures, structures)):
        dummy_matcher = StructureMatcher()
        ltol = dummy_matcher.ltol * tol_multiplier
        stol = dummy_matcher.stol * tol_multiplier
        angle_tol = dummy_matcher.angle_tol * tol_multiplier
        sm = StructureMatcher(
            ltol=ltol,
            stol=stol,
            angle_tol=angle_tol,
            comparator=ElementComparator(),
        )
        is_match = sm.fit(s, structure)
        if not is_match:
            warn(
                f"{i}-th original and decoded structures do not match according to StructureMatcher(comparator=ElementComparator()).fit(s, structure).\n\nOriginal (s): {s}\n\nDecoded (structure): {structure}"  # noqa: E501
            )

        sm = StructureMatcher(primitive_cell=False, comparator=ElementComparator())
        s2 = sm.get_s2_like_s1(s, structure)

        a_check = s._lattice.a
        b_check = s._lattice.b
        c_check = s._lattice.c
        angles_check = s._lattice.angles
        atomic_numbers_check = s.atomic_numbers
        frac_coords_check = s.frac_coords
        space_group_check = s.get_space_group_info()[1]

        latt_a = s2._lattice.a
        latt_b = s2._lattice.b
        latt_c = s2._lattice.c
        angles = s2._lattice.angles
        atomic_numbers = s2.atomic_numbers
        frac_coords = s2.frac_coords
        space_group = s.get_space_group_info()[1]

        assert_allclose(
            a_check,
            latt_a,
            rtol=rgb_loose_tol * tol_multiplier,
            err_msg="lattice parameter length `a` not all close",
        )

        assert_allclose(
            b_check,
            latt_b,
            rtol=rgb_loose_tol * tol_multiplier,
            err_msg="lattice parameter length `b` not all close",
        )

        assert_allclose(
            c_check,
            latt_c,
            rtol=rgb_loose_tol * 2 * tol_multiplier,
            err_msg="lattice parameter length `c` not all close",
        )

        assert_allclose(
            angles_check,
            angles,
            rtol=rgb_loose_tol * tol_multiplier,
            err_msg="lattice parameter angles not all close",
        )

        assert_allclose(
            atomic_numbers_check,
            atomic_numbers,
            rtol=rgb_loose_tol * tol_multiplier,
            err_msg="atomic numbers not all close",
        )

        # use atol since frac_coords values are between 0 and 1
        assert_allclose(
            frac_coords_check,
            frac_coords,
            atol=rgb_tol * tol_multiplier,
            err_msg="atomic numbers not all close",
        )

        assert_equal(
            space_group_check,
            space_group,
            err_msg=f"space groups do not match. Original: {space_group_check}. Decoded: {space_group}.",  # noqa: E501
        )


def test_structures_to_arrays():
    xc = XtalConverter(relax_on_decode=False)
    data, _, _ = xc.structures_to_arrays(example_structures)
    return data


def test_structures_to_arrays_single():
    xc = XtalConverter(relax_on_decode=False)
    data, _, _ = xc.structures_to_arrays([example_structures[0]])
    return data


def test_arrays_to_structures():
    xc = XtalConverter(relax_on_decode=False)
    data, id_data, id_mapper = xc.structures_to_arrays(example_structures)
    structures = xc.arrays_to_structures(data, id_data, id_mapper)
    assert_structures_approximate_match(example_structures, structures)
    return structures


def test_arrays_to_structures_single():
    xc = XtalConverter(relax_on_decode=False)
    data, id_data, id_mapper = xc.structures_to_arrays([example_structures[0]])
    structures = xc.arrays_to_structures(data, id_data, id_mapper)
    assert_structures_approximate_match([example_structures[0]], structures)
    return structures


def test_xtal2png():
    xc = XtalConverter(relax_on_decode=False)
    imgs = xc.xtal2png(example_structures, show=False, save=True)
    return imgs


def test_xtal2png_single():
    xc = XtalConverter(relax_on_decode=False)
    imgs = xc.xtal2png([example_structures[0]], show=False, save=True)
    return imgs


def test_png2xtal():
    xc = XtalConverter(relax_on_decode=False)
    imgs = xc.xtal2png(example_structures, show=True, save=True)
    decoded_structures = xc.png2xtal(imgs)
    assert_structures_approximate_match(example_structures, decoded_structures)


def test_png2xtal_single():
    xc = XtalConverter(relax_on_decode=False)
    imgs = xc.xtal2png([example_structures[0]], show=True, save=True)
    decoded_structures = xc.png2xtal(imgs, save=False)
    assert_structures_approximate_match([example_structures[0]], decoded_structures)
    return decoded_structures


def test_png2xtal_rgb_image():
    xc = XtalConverter(relax_on_decode=False)
    imgs = xc.xtal2png(example_structures, show=False, save=False)
    imgs = [img.convert("RGB") for img in imgs]
    decoded_structures = xc.png2xtal(imgs)
    assert_structures_approximate_match(example_structures, decoded_structures)
    return decoded_structures


def test_primitive_encoding():
    xc = XtalConverter(
        symprec=0.1,
        angle_tolerance=5.0,
        encode_as_primitive=True,
        decode_as_primitive=False,
        relax_on_decode=False,
    )
    input_structures = [
        SpacegroupAnalyzer(
            s, symprec=0.1, angle_tolerance=5.0
        ).get_conventional_standard_structure()
        for s in example_structures
    ]
    data, id_data, id_mapper = xc.structures_to_arrays(input_structures)
    decoded_structures = xc.arrays_to_structures(data, id_data, id_mapper)
    assert_structures_approximate_match(
        example_structures, decoded_structures, tol_multiplier=2.0
    )
    return decoded_structures


def test_primitive_decoding():
    xc = XtalConverter(
        symprec=0.1,
        angle_tolerance=5.0,
        encode_as_primitive=False,
        decode_as_primitive=True,
        relax_on_decode=False,
    )
    input_structures = [
        SpacegroupAnalyzer(
            s, symprec=0.1, angle_tolerance=5.0
        ).get_conventional_standard_structure()
        for s in example_structures
    ]
    data, id_data, id_mapper = xc.structures_to_arrays(input_structures)
    decoded_structures = xc.arrays_to_structures(data, id_data, id_mapper)
    # decoded has to be conventional too for compatibility with `get_s1_like_s2`
    decoded_structures = [
        SpacegroupAnalyzer(
            s, symprec=0.1, angle_tolerance=5.0
        ).get_conventional_standard_structure()
        for s in decoded_structures
    ]
    assert_structures_approximate_match(
        example_structures, decoded_structures, tol_multiplier=2.0
    )
    return decoded_structures


def test_fit():
    xc = XtalConverter(relax_on_decode=False)
    xc.fit(example_structures + dummy_structures)
    assert_array_equal((14, 82), xc.atom_range)
    assert_allclose((3.84, 12.718448099999998), xc.a_range)
    assert_allclose((3.395504, 11.292530369999998), xc.b_range)
    assert_allclose((3.84, 10.6047314973), xc.c_range)
    assert_array_equal((0.0, 180.0), xc.angles_range)
    assert_allclose((12, 227), xc.space_group_range)
    assert_allclose((40.03858081023111, 611.6423774462978), xc.volume_range)
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
        rtol=rgb_tol,
        err_msg=f"rgb_unscaler values not within {rgb_tol} of original",
    )
    assert_allclose(check_unscaled, unscaled)
    assert_allclose(check_output, scaled)


def test_relax_on_decode():
    xc = XtalConverter(relax_on_decode=True)
    imgs = xc.xtal2png(example_structures, show=False, save=False)
    decoded_structures = xc.png2xtal(imgs)
    assert_structures_approximate_match(
        example_structures, decoded_structures, tol_multiplier=4.0
    )
    return decoded_structures


def test_plot_and_save():
    df = px.data.tips()
    fig = px.histogram(df, x="day")
    plot_and_save("reports/figures/tmp", fig, mpl_kwargs={})


# TODO: test_matplotlibify with assertion


if __name__ == "__main__":
    test_relax_on_decode()
    test_primitive_decoding()
    test_primitive_encoding()
    test_fit()
    test_png2xtal_rgb_image()
    test_element_wise_scaler_unscaler()
    test_rgb_scaler_unscaler()
    structures = test_arrays_to_structures()
    imgs = test_xtal2png()
    imgs = test_xtal2png_single()
    decoded_structures = test_png2xtal()
    decoded_structures = test_png2xtal_single()
    test_structures_to_arrays_single()
    test_arrays_to_structures_single()
    test_xtal2png_single()
    test_png2xtal()
    test_png2xtal_single()

1 + 1

# %% Code Graveyard
# from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
# spa = SpacegroupAnalyzer(s, symprec=0.1, angle_tolerance=5.0)
# s = spa.get_refined_structure()
# spa = SpacegroupAnalyzer(structure, symprec=0.1, angle_tolerance=5.0)
# structure = spa.get_refined_structure()
